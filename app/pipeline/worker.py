"""The indexing worker: runs `index_saved_item` jobs from the arq queue.

Run it with `python -m app.pipeline.worker`. It reads only the indexing queue
(`app.ingestion.jobs.INDEXING_QUEUE`), so chat jobs wait for their own worker. It reads `Settings`,
`ModelSettings`, `PipelineSettings` and `MessengerSettings` from the
environment, and needs `ffmpeg` and `ffprobe` on the machine.

**Parallel fan-out.** One job indexes one saved item (decision record 0009).
A worker runs up to `INDEXING_CONCURRENCY` jobs at once, so the jobs of a bulk
import spread over every free slot of every running worker; more workers add
more slots. The providers' rate limits set the real ceiling.

**Retries.** A step that fails in a way that may pass later raises
`RetryLaterError`; the job is tried again after a growing wait, up to
`INDEXING_MAX_TRIES` runs of the item in all, counted in Postgres
(`saved_item.indexing_attempts`) across requeues. The last run marks the item
`failed` as well as it can within its recording deadline. Each run continues
after the steps that already finished (`app.pipeline.indexer`). Reading the
payload and the item at the start is retried the same way.

**The stuck-item sweeper.** An arq cron in this worker runs
`app.pipeline.recovery.sweeper` every `INDEXING_SWEEP_INTERVAL_SECONDS`. It repairs, from
Postgres, any unfinished item that no job has worked on for
`INDEXING_STALE_AFTER_SECONDS`: it queues it again, or marks it `failed`.

**One budget per run.** `INDEXING_RUN_DEADLINE_SECONDS` starts before the
payload is read and covers the start and every step; a run that passes it
stops itself, records where it was, and is tried again. Recording a failure
then has `INDEXING_FAILURE_RECORDING_SECONDS`, and `INDEXING_CLEANUP_RESERVE_SECONDS`
is kept free for stopping a media tool and rolling back. arq's own limit,
`INDEXING_JOB_TIMEOUT_SECONDS`, is longer than the three together and only a
backstop.

The job's arguments are ids, and its result is the outcome's name, so neither
holds a caption, a link or any other content.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final

import httpx
from arq import Retry
from arq.connections import ArqRedis, RedisSettings
from arq.cron import cron
from arq.typing import StartupShutdown
from arq.worker import Worker, func
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.replies.messenger import InstagramMessenger
from app.config.base import SECONDS_PER_MINUTE, Settings, get_settings
from app.config.messenger import get_messenger_settings
from app.config.models import get_model_settings
from app.config.pipeline import PipelineSettings, get_pipeline_settings
from app.db.session import create_engine, create_session_factory
from app.ingestion.jobs import INDEX_SAVED_ITEM_JOB, INDEXING_QUEUE, ArqJobQueue
from app.logging_config import configure_logging
from app.models.factory import build_model_clients, open_http_client
from app.pipeline.indexer import ItemIndexer, PipelineSteps, RetryLaterError
from app.pipeline.media.storage import LocalMediaStore
from app.pipeline.media.tools import FfmpegMediaTools
from app.pipeline.payloads import CapturePayloads
from app.pipeline.prompts import PIPELINE_PROMPTS
from app.pipeline.recovery.acknowledgment import CollectionAcknowledger
from app.pipeline.recovery.sweeper import StuckItemSweeper
from app.pipeline.steps.classify import PostClassifier
from app.pipeline.steps.confirm import SaveConfirmer
from app.pipeline.steps.embed import ItemEmbedder
from app.pipeline.steps.extract import Extractor
from app.pipeline.steps.fetch import MediaFetcher
from app.pipeline.steps.transcribe import SpeechToText
from app.pipeline.vocabulary.normalize import Normalizer, VocabularyRules
from app.types.registry import CONTENT_TYPES

# Where the worker keeps what its jobs share, in arq's context.
_RESOURCES: Final = "indexing_resources"
# arq's context key for the number of the current run of a job, from 1.
_JOB_TRY: Final = "job_try"
# arq's context key for its own Redis client, which also holds ingestion's state.
_REDIS: Final = "redis"
SWEEP_STUCK_ITEMS_JOB: Final = "sweep_stuck_items"

# arq hands every job a context dictionary of its own making; its values are
# of any type, so it cannot be typed more narrowly.
JobContext = dict[str, Any]


@dataclass(frozen=True)
class WorkerResources:
    """What every job of one worker shares, and what is closed when it stops.

    Attributes:
        indexer: Indexes one item.
        sweeper: Repairs stuck items.
        retry_delay_seconds: `INDEXING_RETRY_DELAY_SECONDS`.
        close: Closes the HTTP clients and the database engine.
    """

    indexer: ItemIndexer
    sweeper: StuckItemSweeper
    retry_delay_seconds: float
    close: Callable[[], Awaitable[None]]


async def sweep_stuck_items(ctx: JobContext) -> dict[str, int]:
    """Requeue or fail the items no job is working on any more; the sweeper's cron job.

    Args:
        ctx: arq's job context.

    Returns:
        How many items were requeued, failed or skipped; counts only.
    """
    resources: WorkerResources = ctx[_RESOURCES]
    report = await resources.sweeper.sweep()
    return report.as_log()


def sweep_schedule(interval_seconds: int) -> tuple[set[int] | None, set[int]]:
    """Turn the sweep interval into the minutes and seconds an arq cron runs at.

    Args:
        interval_seconds: `INDEXING_SWEEP_INTERVAL_SECONDS`; the settings check
            that it divides a minute, or is whole minutes that divide an hour.

    Returns:
        The minutes (`None` for every minute) and the seconds to run at.
    """
    if interval_seconds < SECONDS_PER_MINUTE:
        return None, set(range(0, SECONDS_PER_MINUTE, interval_seconds))
    return set(range(0, SECONDS_PER_MINUTE, interval_seconds // SECONDS_PER_MINUTE)), {0}


async def index_saved_item(ctx: JobContext, *, user_id: str, saved_item_id: str) -> str:
    """Index one saved item; the arq job queued by ingestion.

    Args:
        ctx: arq's job context.
        user_id: The item's owner, as text.
        saved_item_id: The item, as text.

    Returns:
        The outcome's name, e.g. `indexed`.

    Raises:
        arq.Retry: If a step failed in a way that may pass later, and the item
            has tries left.
    """
    resources: WorkerResources = ctx[_RESOURCES]
    job_try: int = ctx[_JOB_TRY]
    try:
        outcome = await resources.indexer.index(uuid.UUID(user_id), uuid.UUID(saved_item_id))
    except RetryLaterError as error:
        raise Retry(defer=resources.retry_delay_seconds * job_try) from error
    return outcome.value


async def start_worker(ctx: JobContext) -> None:
    """Build the indexer and everything it needs, once per worker.

    Args:
        ctx: arq's worker context. Its Redis client also reads ingestion's payloads.

    Raises:
        pydantic.ValidationError: If a required setting is missing or invalid.
    """
    settings = get_settings()
    pipeline_settings = get_pipeline_settings()
    model_settings = get_model_settings()
    configure_logging(settings.log_level)
    engine = create_engine(str(settings.database_url))
    sessions = create_session_factory(engine)
    model_http = open_http_client(model_settings)
    # Media links carry their own signature; nothing else is sent with them.
    media_http = httpx.AsyncClient()
    messenger_http = httpx.AsyncClient()
    models = build_model_clients(model_settings, model_http)
    store = LocalMediaStore(Path(pipeline_settings.media_storage_dir))
    tools = FfmpegMediaTools(
        ffmpeg_path=pipeline_settings.ffmpeg_path,
        ffprobe_path=pipeline_settings.ffprobe_path,
        timeout_seconds=pipeline_settings.media_tool_timeout_seconds,
        stop_timeout_seconds=pipeline_settings.media_tool_stop_timeout_seconds,
    )
    rules = VocabularyRules(
        privacy_min_probability=pipeline_settings.vocab_privacy_min_probability,
        alias_min_probability=pipeline_settings.vocab_alias_min_probability,
        alias_min_users=pipeline_settings.vocab_alias_min_users,
        max_value_length=pipeline_settings.vocab_max_value_length,
        phone_min_digits=pipeline_settings.vocab_phone_min_digits,
        match_candidates=pipeline_settings.vocab_match_candidates,
    )
    # Instagram is the only platform with a messenger so far (ARCHITECTURE §3.1).
    confirmer = SaveConfirmer(InstagramMessenger(messenger_http, get_messenger_settings()))
    steps = PipelineSteps(
        fetcher=MediaFetcher(
            media_http,
            store,
            tools,
            timeout_seconds=pipeline_settings.media_download_timeout_seconds,
            max_bytes=pipeline_settings.media_max_download_bytes,
            stored_height_pixels=pipeline_settings.media_stored_height_pixels,
        ),
        classifier=PostClassifier(
            models.classifier,
            CONTENT_TYPES,
            PIPELINE_PROMPTS,
            is_list_min_probability=pipeline_settings.classify_is_list_min_probability,
        ),
        speech=SpeechToText(
            models.transcriber,
            tools,
            store,
            min_speech_characters=pipeline_settings.transcript_min_speech_characters,
        ),
        extractor=Extractor(
            models.text_generator,
            models.escalation_generator,
            CONTENT_TYPES,
            PIPELINE_PROMPTS,
            store,
            temperature=pipeline_settings.extraction_temperature,
            max_output_tokens=pipeline_settings.extraction_max_output_tokens,
        ),
        normalizer=Normalizer(
            sessions, models.classifier, models.embedder, PIPELINE_PROMPTS, rules
        ),
        embedder=ItemEmbedder(models.embedder),
        confirmer=confirmer,
        store=store,
    )
    ctx[_RESOURCES] = WorkerResources(
        indexer=ItemIndexer(
            sessions,
            CapturePayloads(ctx[_REDIS]),
            steps,
            CONTENT_TYPES,
            PIPELINE_PROMPTS,
            alias_min_probability=rules.alias_min_probability,
            run_deadline_seconds=pipeline_settings.indexing_run_deadline_seconds,
            failure_recording_seconds=pipeline_settings.indexing_failure_recording_seconds,
            max_tries=pipeline_settings.indexing_max_tries,
        ),
        sweeper=StuckItemSweeper(
            sessions=sessions,
            payloads=CapturePayloads(ctx[_REDIS]),
            job_queue=ArqJobQueue(ctx[_REDIS]),
            stale_after_seconds=pipeline_settings.indexing_stale_after_seconds,
            max_tries=pipeline_settings.indexing_max_tries,
            batch_size=pipeline_settings.indexing_sweep_batch_size,
            acknowledger=CollectionAcknowledger(sessions, confirmer),
        ),
        retry_delay_seconds=pipeline_settings.indexing_retry_delay_seconds,
        close=partial(_close, engine, (model_http, media_http, messenger_http)),
    )


async def _close(engine: AsyncEngine, http_clients: tuple[httpx.AsyncClient, ...]) -> None:
    await asyncio.gather(*(client.aclose() for client in http_clients))
    await engine.dispose()


async def stop_worker(ctx: JobContext) -> None:
    """Close the worker's HTTP clients and database engine.

    Args:
        ctx: arq's worker context.
    """
    resources: WorkerResources | None = ctx.get(_RESOURCES)
    if resources is not None:
        await resources.close()


def build_worker(
    settings: Settings,
    pipeline_settings: PipelineSettings,
    *,
    on_startup: StartupShutdown = start_worker,
    redis_pool: ArqRedis | None = None,
    burst: bool = False,
) -> Worker:
    """Create the arq worker for indexing jobs.

    It reads only `INDEXING_QUEUE`, so it never takes a chat job, and runs
    the stuck-item sweeper every `INDEXING_SWEEP_INTERVAL_SECONDS`.

    Args:
        settings: The app's settings, for the Redis URL.
        pipeline_settings: The pipeline's settings: tries, concurrency, timeouts.
        on_startup: Builds the job resources; tests pass their own.
        redis_pool: A Redis client to use instead of one from the URL; tests pass one.
        burst: Stop once the queue is empty, instead of waiting for more jobs.

    Returns:
        The worker. `run()` starts it.
    """
    minutes, seconds = sweep_schedule(pipeline_settings.indexing_sweep_interval_seconds)
    return Worker(
        functions=[
            func(
                index_saved_item,
                name=INDEX_SAVED_ITEM_JOB,
                max_tries=pipeline_settings.indexing_max_tries,
            ),
        ],
        cron_jobs=[
            cron(
                sweep_stuck_items,
                name=SWEEP_STUCK_ITEMS_JOB,
                minute=minutes,
                second=seconds,
                run_at_startup=True,
            )
        ],
        queue_name=INDEXING_QUEUE,
        redis_settings=RedisSettings.from_dsn(str(settings.redis_url)),
        redis_pool=redis_pool,
        burst=burst,
        on_startup=on_startup,
        on_shutdown=stop_worker,
        max_jobs=pipeline_settings.indexing_concurrency,
        # The run stops itself at INDEXING_RUN_DEADLINE_SECONDS and records where
        # within INDEXING_FAILURE_RECORDING_SECONDS, with INDEXING_CLEANUP_RESERVE_SECONDS
        # to spare; this outer limit, longer than all three, is only a backstop.
        job_timeout=pipeline_settings.indexing_job_timeout_seconds,
        max_tries=pipeline_settings.indexing_max_tries,
    )


def main() -> None:
    """Run the indexing worker until it is stopped.

    Raises:
        pydantic.ValidationError: If a required setting is missing or invalid.
    """
    build_worker(get_settings(), get_pipeline_settings()).run()


if __name__ == "__main__":
    main()
