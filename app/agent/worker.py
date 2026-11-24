"""The chat worker: runs `answer_chat_message` jobs from the chat queue (record 0015, Q5).

Run it with `python -m app.agent.worker`. It reads only the chat queue
(`app.ingestion.jobs.CHAT_QUEUE`), separate from indexing, so a long indexing
run never holds up an answer. It reads `Settings`, `ModelSettings`,
`PipelineSettings`, `RetrievalSettings`, `AgentSettings`, `MessengerSettings`
and `ChatSettings` from the environment.

One job answers one message (`app.agent.chat.ChatResponder`). It first
settles whether a text sent right after a share names that post's collection,
with the decision model and `COLLECTION_HINT_MIN_PROBABILITY` (record 0018). A worker runs up
to `CHAT_CONCURRENCY` jobs at once. A failure before sending that may pass
later is tried again after `CHAT_RETRY_DELAY_SECONDS` times the try's number,
up to `CHAT_MAX_TRIES` runs; the last one sends the error message instead.
`CHAT_RUN_DEADLINE_SECONDS` bounds the preparation of each try, and arq's own
`CHAT_JOB_TIMEOUT_SECONDS` is only a backstop.

**"Delete my data"** goes through one `UserDataEraser`, built here with every
store outside Postgres: ingestion's Redis state and jobs, the pending deletion
request, short-term memory and the media store.

The job's arguments are ids, and its result is the outcome's name, so neither
holds a message or any other content.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Final

import httpx
from arq import Retry
from arq.connections import ArqRedis, RedisSettings
from arq.typing import StartupShutdown
from arq.worker import Worker, func
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.agent import ResponseAgent
from app.agent.chat import ChatPayloads, ChatResponder, RetryLaterError
from app.agent.deletion_requests import RedisDeletionRequests
from app.agent.prompts import AGENT_PROMPTS
from app.agent.replies.answers import AnswerSender
from app.agent.replies.messenger import InstagramMessenger
from app.agent.search.prompts import RETRIEVAL_PROMPTS
from app.agent.search.retrieval import HybridRetriever
from app.agent.tools import Toolbox
from app.config.agent import get_agent_settings
from app.config.base import Settings, get_settings
from app.config.chat import ChatSettings, get_chat_settings
from app.config.messenger import get_messenger_settings
from app.config.models import get_model_settings
from app.config.pipeline import PipelineSettings, get_pipeline_settings
from app.config.retrieval import get_retrieval_settings
from app.db.deletion import UserDataEraser
from app.db.session import create_engine, create_session_factory
from app.ingestion.erasure import IngestionDataEraser
from app.ingestion.jobs import ANSWER_CHAT_MESSAGE_JOB, CHAT_QUEUE, ArqJobQueue
from app.ingestion.prompts import ROUTING_PROMPTS
from app.ingestion.session_state import RedisIngestionState
from app.ingestion.text_routing import TextRouter
from app.logging_config import configure_logging
from app.memory.conversation import ConversationStore
from app.memory.session import SessionMemory
from app.models.base import Classifier, Embedder
from app.models.factory import build_model_clients, open_http_client
from app.pipeline.media.storage import LocalMediaStore
from app.pipeline.prompts import PIPELINE_PROMPTS
from app.pipeline.vocabulary.corrections import ItemCorrection
from app.pipeline.vocabulary.normalize import Normalizer, VocabularyRules
from app.types.registry import CONTENT_TYPES

# Where the worker keeps what its jobs share, in arq's context.
_RESOURCES: Final = "chat_resources"
# arq's context key for the number of the current run of a job, from 1.
_JOB_TRY: Final = "job_try"
# arq's context key for its own Redis client, which also holds ingestion's state.
_REDIS: Final = "redis"

# arq hands every job a context dictionary of its own making; its values are
# of any type, so it cannot be typed more narrowly.
JobContext = dict[str, Any]


@dataclass(frozen=True)
class ChatWorkerResources:
    """What every job of one chat worker shares, and what is closed when it stops.

    Attributes:
        responder: Answers one message.
        max_tries: `CHAT_MAX_TRIES`.
        retry_delay_seconds: `CHAT_RETRY_DELAY_SECONDS`.
        close: Closes the HTTP clients and the database engine.
    """

    responder: ChatResponder
    max_tries: int
    retry_delay_seconds: float
    close: Callable[[], Awaitable[None]]


async def answer_chat_message(ctx: JobContext, *, user_id: str, message_receipt_id: str) -> str:
    """Answer one chat message; the arq job queued by ingestion.

    Args:
        ctx: arq's job context.
        user_id: The sender, as text.
        message_receipt_id: The id of the text's receipt, as text.

    Returns:
        The outcome's name, e.g. `answered`.

    Raises:
        arq.Retry: If preparing the answer failed in a way that may pass
            later, and the job has tries left.
    """
    resources: ChatWorkerResources = ctx[_RESOURCES]
    job_try: int = ctx[_JOB_TRY]
    try:
        outcome = await resources.responder.respond(
            uuid.UUID(user_id),
            uuid.UUID(message_receipt_id),
            last_try=job_try >= resources.max_tries,
        )
    except RetryLaterError as error:
        raise Retry(defer=resources.retry_delay_seconds * job_try) from error
    return outcome.value


def build_toolbox(
    sessions: async_sessionmaker[AsyncSession],
    redis: ArqRedis,
    *,
    classifier: Classifier,
    embedder: Embedder,
    settings: Settings,
    pipeline_settings: PipelineSettings,
    chat_settings: ChatSettings,
) -> Toolbox:
    """Build the response agent's tools, with an eraser that reaches every store.

    Args:
        sessions: Opens a database session.
        redis: The Redis client of the job queue, ingestion's state and short-term memory.
        classifier: The decision model, for search and corrections.
        embedder: The embedding model.
        settings: The app's settings, for ingestion's state lifetime.
        pipeline_settings: The vocabulary's rules and the media folder.
        chat_settings: Short-term memory's lifetime and length.

    Returns:
        The toolbox.

    Raises:
        pydantic.ValidationError: If a required setting is missing or invalid.
    """
    agent_settings = get_agent_settings()
    rules = VocabularyRules(
        privacy_min_probability=pipeline_settings.vocab_privacy_min_probability,
        alias_min_probability=pipeline_settings.vocab_alias_min_probability,
        alias_min_users=pipeline_settings.vocab_alias_min_users,
        max_value_length=pipeline_settings.vocab_max_value_length,
        phone_min_digits=pipeline_settings.vocab_phone_min_digits,
        match_candidates=pipeline_settings.vocab_match_candidates,
    )
    deletion_requests = RedisDeletionRequests(redis)
    eraser = UserDataEraser(
        sessions,
        [
            IngestionDataEraser(
                RedisIngestionState(redis, timedelta(seconds=settings.ingestion_state_ttl_seconds)),
                ArqJobQueue(redis),
            ),
            deletion_requests,
            session_memory(redis, chat_settings),
            LocalMediaStore(Path(pipeline_settings.media_storage_dir)),
        ],
    )
    return Toolbox(
        sessions=sessions,
        retriever=HybridRetriever(
            classifier=classifier,
            embedder=embedder,
            prompts=RETRIEVAL_PROMPTS,
            settings=get_retrieval_settings(),
        ),
        corrections=ItemCorrection(
            sessions,
            Normalizer(sessions, classifier, embedder, PIPELINE_PROMPTS, rules),
            CONTENT_TYPES,
            alias_min_probability=rules.alias_min_probability,
            review_min_users=agent_settings.vocab_alias_review_min_users,
        ),
        registry=CONTENT_TYPES,
        deletion_requests=deletion_requests,
        eraser=eraser,
        deletion_window=timedelta(seconds=agent_settings.data_deletion_confirm_seconds),
    )


def session_memory(redis: ArqRedis, chat_settings: ChatSettings) -> SessionMemory:
    """Build short-term memory from the chat settings.

    Args:
        redis: The Redis client.
        chat_settings: `SESSION_TTL_SECONDS` and `SESSION_HISTORY_MESSAGES`.

    Returns:
        The store.
    """
    return SessionMemory(
        redis,
        lifetime=timedelta(seconds=chat_settings.session_ttl_seconds),
        max_messages=chat_settings.session_history_messages,
    )


async def start_worker(ctx: JobContext) -> None:
    """Build the responder and everything it needs, once per worker.

    Args:
        ctx: arq's worker context. Its Redis client also holds ingestion's
            payloads and short-term memory.

    Raises:
        pydantic.ValidationError: If a required setting is missing or invalid.
    """
    settings = get_settings()
    chat_settings = get_chat_settings()
    model_settings = get_model_settings()
    configure_logging(settings.log_level)
    engine = create_engine(str(settings.database_url))
    sessions = create_session_factory(engine)
    model_http = open_http_client(model_settings)
    messenger_http = httpx.AsyncClient()
    models = build_model_clients(model_settings, model_http)
    redis: ArqRedis = ctx[_REDIS]
    toolbox = build_toolbox(
        sessions,
        redis,
        classifier=models.classifier,
        embedder=models.embedder,
        settings=settings,
        pipeline_settings=get_pipeline_settings(),
        chat_settings=chat_settings,
    )
    # Instagram is the only platform with a messenger so far (ARCHITECTURE §3.1).
    sender = AnswerSender(InstagramMessenger(messenger_http, get_messenger_settings()))
    ingestion_state = RedisIngestionState(
        redis, timedelta(seconds=settings.ingestion_state_ttl_seconds)
    )
    ctx[_RESOURCES] = ChatWorkerResources(
        responder=ChatResponder(
            payloads=ChatPayloads(ingestion_state),
            routes=TextRouter(
                sessions,
                ingestion_state,
                models.classifier,
                ROUTING_PROMPTS,
                min_probability=chat_settings.collection_hint_min_probability,
                decision_timeout_seconds=chat_settings.collection_hint_decision_timeout_seconds,
            ),
            memory=session_memory(redis, chat_settings),
            conversations=ConversationStore(sessions),
            agent=ResponseAgent(
                generator=models.text_generator,
                toolbox=toolbox,
                prompts=AGENT_PROMPTS,
                settings=get_agent_settings(),
            ),
            toolbox=toolbox,
            sender=sender,
            run_deadline_seconds=chat_settings.chat_run_deadline_seconds,
            store_tries=chat_settings.chat_max_tries,
            store_retry_delay_seconds=chat_settings.chat_retry_delay_seconds,
        ),
        max_tries=chat_settings.chat_max_tries,
        retry_delay_seconds=chat_settings.chat_retry_delay_seconds,
        close=partial(_close, engine, (model_http, messenger_http)),
    )


async def _close(engine: AsyncEngine, http_clients: tuple[httpx.AsyncClient, ...]) -> None:
    await asyncio.gather(*(client.aclose() for client in http_clients))
    await engine.dispose()


async def stop_worker(ctx: JobContext) -> None:
    """Close the worker's HTTP clients and database engine.

    Args:
        ctx: arq's worker context.
    """
    resources: ChatWorkerResources | None = ctx.get(_RESOURCES)
    if resources is not None:
        await resources.close()


def build_worker(
    settings: Settings,
    chat_settings: ChatSettings,
    *,
    on_startup: StartupShutdown = start_worker,
    redis_pool: ArqRedis | None = None,
    burst: bool = False,
) -> Worker:
    """Create the arq worker for chat jobs.

    It reads only `CHAT_QUEUE`, so it never takes an indexing job.

    Args:
        settings: The app's settings, for the Redis URL.
        chat_settings: Tries, concurrency and the job timeout.
        on_startup: Builds the job resources; tests pass their own.
        redis_pool: A Redis client to use instead of one from the URL; tests pass one.
        burst: Stop once the queue is empty, instead of waiting for more jobs.

    Returns:
        The worker. `run()` starts it.
    """
    return Worker(
        functions=[
            func(
                answer_chat_message,
                name=ANSWER_CHAT_MESSAGE_JOB,
                max_tries=chat_settings.chat_max_tries,
            ),
        ],
        queue_name=CHAT_QUEUE,
        redis_settings=RedisSettings.from_dsn(str(settings.redis_url)),
        redis_pool=redis_pool,
        burst=burst,
        on_startup=on_startup,
        on_shutdown=stop_worker,
        max_jobs=chat_settings.chat_concurrency,
        # Each try stops preparing at CHAT_RUN_DEADLINE_SECONDS; the rest of this
        # limit is for sending, and it is only a backstop.
        job_timeout=chat_settings.chat_job_timeout_seconds,
        max_tries=chat_settings.chat_max_tries,
    )


def main() -> None:
    """Run the chat worker until it is stopped.

    Raises:
        pydantic.ValidationError: If a required setting is missing or invalid.
    """
    build_worker(get_settings(), get_chat_settings()).run()


if __name__ == "__main__":
    main()
