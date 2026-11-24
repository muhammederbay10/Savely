"""Builds a whole indexer on fakes and a test database, for the end-to-end pipeline tests."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.base import CaptureEvent
from app.db.models import PlatformAccount, SavedItem, User
from app.models.base import GenerationResult, Transcript
from app.models.clients.transcriber import FallbackTranscriber
from app.models.errors import ModelError
from app.models.fakes import FakeEmbedder, FakeGenerator, FakeTranscriber
from app.pipeline.indexer import ItemIndexer, PipelineSteps
from app.pipeline.media.storage import LocalMediaStore
from app.pipeline.recovery.acknowledgment import CollectionAcknowledger
from app.pipeline.recovery.sweeper import Requeuer, StuckItemSweeper, SweepReport
from app.pipeline.steps.classify import PostClassifier
from app.pipeline.steps.confirm import SaveConfirmer
from app.pipeline.steps.embed import ItemEmbedder
from app.pipeline.steps.extract import Extractor
from app.pipeline.steps.fetch import MediaFetcher
from app.pipeline.steps.transcribe import SpeechToText
from app.pipeline.vocabulary.normalize import Normalizer, VocabularyRules
from app.types.registry import CONTENT_TYPES
from tests.database import TEST_EMBEDDING_DIMENSION
from tests.pipeline.support import (
    TEST_PROMPTS,
    FakeMediaTools,
    MediaServer,
    RecordingMessenger,
    ScriptedClassifier,
)

SENDER_ID = "1000000000000141"
RUN_DEADLINE_SECONDS = 30.0
FAILURE_RECORDING_SECONDS = 10.0
MAX_TRIES = 3
STALE_AFTER_SECONDS = 300.0
SWEEP_BATCH_SIZE = 10
RULES = VocabularyRules(
    privacy_min_probability=0.9,
    alias_min_probability=0.9,
    alias_min_users=2,
    max_value_length=40,
    phone_min_digits=7,
    match_candidates=3,
)


class IdleSweeper:
    """A sweeper for test workers that must not sweep: it finds nothing."""

    async def sweep(self) -> SweepReport:
        return SweepReport()


class MemoryPayloads:
    """Capture events by item, standing in for ingestion's Redis hash."""

    def __init__(self) -> None:
        self.events: dict[uuid.UUID, CaptureEvent] = {}

    async def capture(self, user_id: uuid.UUID, saved_item_id: uuid.UUID) -> CaptureEvent | None:
        event = self.events.get(saved_item_id)
        return event if event is not None and event.user_id == user_id else None


@dataclass
class Pipeline:
    """An indexer and every fake behind it."""

    sessions: async_sessionmaker[AsyncSession]
    media_root: Path
    classifier: ScriptedClassifier = field(default_factory=ScriptedClassifier)
    media_server: MediaServer = field(default_factory=lambda: MediaServer(b"video", "video/mp4"))
    tools: FakeMediaTools = field(default_factory=FakeMediaTools)
    primary_stt: FakeTranscriber = field(default_factory=lambda: FakeTranscriber([]))
    fallback_stt: FakeTranscriber = field(default_factory=lambda: FakeTranscriber([]))
    text_model: FakeGenerator = field(default_factory=lambda: FakeGenerator([]))
    escalation_model: FakeGenerator = field(default_factory=lambda: FakeGenerator([]))
    embedder: FakeEmbedder = field(
        default_factory=lambda: FakeEmbedder(dimensions=TEST_EMBEDDING_DIMENSION)
    )
    messenger: RecordingMessenger = field(default_factory=RecordingMessenger)
    payloads: MemoryPayloads = field(default_factory=MemoryPayloads)

    def script(
        self,
        *,
        transcripts: Sequence[Transcript | ModelError] = (),
        fallback_transcripts: Sequence[Transcript | ModelError] = (),
        extractions: Sequence[GenerationResult | ModelError] = (),
        escalations: Sequence[GenerationResult | ModelError] = (),
    ) -> None:
        self.primary_stt = FakeTranscriber(transcripts)
        self.fallback_stt = FakeTranscriber(fallback_transcripts)
        self.text_model = FakeGenerator(extractions)
        self.escalation_model = FakeGenerator(escalations)

    @property
    def store(self) -> LocalMediaStore:
        return LocalMediaStore(self.media_root)

    def normalizer(self) -> Normalizer:
        return Normalizer(self.sessions, self.classifier, self.embedder, TEST_PROMPTS, RULES)

    def indexer(
        self,
        *,
        sessions: object | None = None,
        payloads: object | None = None,
        run_deadline_seconds: float = RUN_DEADLINE_SECONDS,
        failure_recording_seconds: float = FAILURE_RECORDING_SECONDS,
        max_tries: int = MAX_TRIES,
    ) -> ItemIndexer:
        """Build the indexer; tests may swap in failing sessions or payloads."""
        store = self.store
        steps = PipelineSteps(
            fetcher=MediaFetcher(
                self.media_server.client(),
                store,
                self.tools,
                timeout_seconds=5,
                max_bytes=10_000,
                stored_height_pixels=360,
            ),
            classifier=PostClassifier(
                self.classifier, CONTENT_TYPES, TEST_PROMPTS, is_list_min_probability=0.5
            ),
            speech=SpeechToText(
                FallbackTranscriber(primary=self.primary_stt, fallback=self.fallback_stt),
                self.tools,
                store,
                min_speech_characters=4,
            ),
            extractor=Extractor(
                self.text_model,
                self.escalation_model,
                CONTENT_TYPES,
                TEST_PROMPTS,
                store,
                temperature=0.0,
                max_output_tokens=800,
            ),
            normalizer=self.normalizer(),
            embedder=ItemEmbedder(self.embedder),
            confirmer=SaveConfirmer(self.messenger),
            store=store,
        )
        return ItemIndexer(
            sessions or self.sessions,  # type: ignore[arg-type]
            payloads or self.payloads,  # type: ignore[arg-type]
            steps,
            CONTENT_TYPES,
            TEST_PROMPTS,
            alias_min_probability=RULES.alias_min_probability,
            run_deadline_seconds=run_deadline_seconds,
            failure_recording_seconds=failure_recording_seconds,
            max_tries=max_tries,
        )

    def sweeper(
        self,
        job_queue: Requeuer,
        *,
        stale_after_seconds: float = STALE_AFTER_SECONDS,
        max_tries: int = MAX_TRIES,
        batch_size: int = SWEEP_BATCH_SIZE,
        sessions: object | None = None,
    ) -> StuckItemSweeper:
        """Build the stuck-item sweeper on the same database and payloads."""
        return StuckItemSweeper(
            sessions=sessions or self.sessions,  # type: ignore[arg-type]
            payloads=self.payloads,
            job_queue=job_queue,
            stale_after_seconds=stale_after_seconds,
            max_tries=max_tries,
            batch_size=batch_size,
            acknowledger=CollectionAcknowledger(
                sessions or self.sessions,  # type: ignore[arg-type]
                SaveConfirmer(self.messenger),
            ),
        )


async def add_user(
    sessions: async_sessionmaker[AsyncSession], *, language: str | None = "en"
) -> uuid.UUID:
    async with sessions() as session, session.begin():
        user = User(display_name="Synthetic User", language=language)
        session.add(user)
        await session.flush()
        session.add(
            PlatformAccount(
                user_id=user.id, platform="instagram", platform_user_id=f"{SENDER_ID}-{user.id}"
            )
        )
        return user.id


async def add_item(
    sessions: async_sessionmaker[AsyncSession],
    pipeline: Pipeline,
    event: CaptureEvent,
    *,
    collection: str | None = None,
) -> uuid.UUID:
    async with sessions() as session, session.begin():
        item = SavedItem(
            user_id=event.user_id,
            platform=event.platform,
            media_type=event.media_type,
            caption=event.caption,
            collection=collection,
            source_message_id=event.source_message_id,
        )
        session.add(item)
        await session.flush()
        item_id = item.id
    pipeline.payloads.events[item_id] = event
    return item_id


async def load(sessions: async_sessionmaker[AsyncSession], item_id: uuid.UUID) -> SavedItem:
    async with sessions() as session:
        item = await session.get(SavedItem, item_id)
        assert item is not None
        return item
