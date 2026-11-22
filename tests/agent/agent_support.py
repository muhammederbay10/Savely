"""Builds the response agent on fake models and a test database.

The text model is a `FakeGenerator`: each test scripts the tool calls and the
final answer, so the agent's loop and the tools run for real, offline.
"""

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.agent import ResponseAgent
from app.agent.deletion_requests import DeletionRequests, PendingDeletion
from app.agent.prompts import AGENT_PROMPTS
from app.agent.search.retrieval import HybridRetriever
from app.agent.tools import Toolbox
from app.config.agent import AgentSettings
from app.db.deletion import UserDataEraser, UserDataOutsidePostgres
from app.models.base import GenerationRequest, GenerationResult, Generator, ToolCall
from app.models.errors import ModelError
from app.models.fakes import FakeEmbedder, FakeGenerator
from app.pipeline.vocabulary.corrections import ItemCorrection
from app.pipeline.vocabulary.normalize import Normalizer
from app.types.registry import CONTENT_TYPES
from tests.agent.retrieval_support import build_retriever
from tests.database import TEST_EMBEDDING_DIMENSION
from tests.pipeline.harness import RULES
from tests.pipeline.support import TEST_PROMPTS as PIPELINE_PROMPTS
from tests.pipeline.support import ScriptedClassifier

TEXT_MODEL = "fake/text"
MAX_STEPS = 5
MAX_RESULTS = 5
CONFIRM_SECONDS = 600
REVIEW_MIN_USERS = 2
TURN_TIME = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)


@dataclass
class FakeClock:
    """A clock that only moves when a test moves it."""

    now: datetime = TURN_TIME

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class MemoryDeletionRequests(DeletionRequests):
    """Pending deletion requests in memory, with the same rules as the Redis store."""

    requests: dict[uuid.UUID, PendingDeletion] = field(default_factory=dict)
    used_up: dict[uuid.UUID, int] = field(default_factory=dict)

    async def save(self, user_id: uuid.UUID, pending: PendingDeletion, now: datetime) -> None:
        if pending.deadline <= now or self.used_up.get(user_id, 0) >= pending.awaited_message:
            return
        kept = self.requests.get(user_id)
        if kept is not None and (kept.awaited_message, kept.deadline) >= (
            pending.awaited_message,
            pending.deadline,
        ):
            return
        self.requests[user_id] = pending

    async def take(self, user_id: uuid.UUID, message_number: int) -> PendingDeletion | None:
        pending = self.requests.get(user_id)
        if pending is None or pending.awaited_message != message_number:
            return None
        self.used_up[user_id] = message_number
        return self.requests.pop(user_id)


def agent_settings(**overrides: int) -> AgentSettings:
    values = {
        "agent_max_steps": MAX_STEPS,
        "agent_max_results_per_answer": MAX_RESULTS,
        "agent_max_output_tokens": 400,
        "vocab_alias_review_min_users": REVIEW_MIN_USERS,
        "data_deletion_confirm_seconds": CONFIRM_SECONDS,
    }
    return AgentSettings(_env_file=None, **{**values, **overrides})  # type: ignore[arg-type]


def tool_calls(*calls: tuple[str, Mapping[str, object]]) -> GenerationResult:
    """A model reply that only calls tools."""
    return GenerationResult(
        text="",
        structured=None,
        tool_calls=tuple(
            ToolCall(id=f"call-{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(calls)
        ),
        finish_reason="tool_calls",
        model=TEXT_MODEL,
    )


def final_answer(
    language: str,
    reply: str,
    item_ids: Sequence[uuid.UUID] = (),
    notice: str | None = None,
) -> GenerationResult:
    """A model reply with the final answer."""
    structured = {
        "language": language,
        "reply": reply,
        "notice": notice,
        "item_ids": [str(item_id) for item_id in item_ids],
    }
    return structured_reply(structured)


def structured_reply(structured: Mapping[str, object]) -> GenerationResult:
    return GenerationResult(
        text=json.dumps(structured, ensure_ascii=False),
        structured=structured,
        tool_calls=(),
        finish_reason="stop",
        model=TEXT_MODEL,
    )


def translated_wording(language: str, separator: str = ", ") -> dict[str, str]:
    """What the model answers when asked for the bot's fixed words in `language`."""
    return {
        "saved_named": "«{item}»",
        "saved_as": "«{category}»",
        "streaming_on": "{platforms}",
        "which_one": f"{language}: which one?",
        "none_of_these": f"{language}: none",
        "import_progress": "{done}/{total}",
        "import_progress_with_counts": "{done}/{total} {counts}",
        "error": f"{language}: error",
        "open_now": f"{language}: open",
        "closed_now": f"{language}: closed",
        "list_separator": separator,
        "deletion_confirmation": f"{language}: delete everything? yes within {{minutes}} min",
        "delete_button": f"{language}: yes, delete",
        "deletion_done": f"{language}: everything deleted",
    }


def tool_results(request: GenerationRequest) -> list[dict[str, object]]:
    """The JSON of every tool result the model was shown in a request."""
    return [
        json.loads(str(message.content)) for message in request.messages if message.role == "tool"
    ]


@dataclass
class AgentUnderTest:
    agent: ResponseAgent
    generator: FakeGenerator
    classifier: ScriptedClassifier
    toolbox: Toolbox


def build_agent(
    sessions: async_sessionmaker[AsyncSession],
    script: Sequence[GenerationResult | ModelError],
    *,
    retriever: HybridRetriever | None = None,
    classifier: ScriptedClassifier | None = None,
    deletion_requests: DeletionRequests | None = None,
    outside_postgres: Sequence[UserDataOutsidePostgres] = (),
    clock: Callable[[], datetime] | None = None,
    wrap_generator: Callable[[FakeGenerator], Generator] | None = None,
    **settings: int,
) -> AgentUnderTest:
    """The agent with real tools on the test database and a scripted text model."""
    generator = FakeGenerator(script)
    vocabulary_classifier = classifier or ScriptedClassifier()
    normalizer = Normalizer(
        sessions,
        vocabulary_classifier,
        FakeEmbedder(dimensions=TEST_EMBEDDING_DIMENSION),
        PIPELINE_PROMPTS,
        RULES,
    )
    toolbox = Toolbox(
        sessions=sessions,
        retriever=retriever or build_retriever(),
        corrections=ItemCorrection(
            sessions,
            normalizer,
            CONTENT_TYPES,
            alias_min_probability=RULES.alias_min_probability,
            review_min_users=REVIEW_MIN_USERS,
        ),
        registry=CONTENT_TYPES,
        deletion_requests=deletion_requests or MemoryDeletionRequests(),
        eraser=UserDataEraser(sessions, outside_postgres),
        deletion_window=timedelta(seconds=CONFIRM_SECONDS),
        now=clock or FakeClock(),
    )
    agent = ResponseAgent(
        generator=wrap_generator(generator) if wrap_generator else generator,
        toolbox=toolbox,
        prompts=AGENT_PROMPTS,
        settings=agent_settings(**settings),
    )
    return AgentUnderTest(
        agent=agent, generator=generator, classifier=vocabulary_classifier, toolbox=toolbox
    )
