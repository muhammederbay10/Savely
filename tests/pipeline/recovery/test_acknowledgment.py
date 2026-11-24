"""The collection confirmation of a failed item never outlives "delete my data" (SAV-39).

These tests need Postgres with PostGIS and pgvector (`TEST_DATABASE_ADMIN_URL`)
and Redis (`TEST_REDIS_URL`). Their writes are committed, so the eraser and the
sender lock the user's row on separate connections. The Send API is a recording
fake. Every id and text is synthetic.
"""

import asyncio
import uuid
from datetime import datetime, timedelta

import pytest
from arq.connections import ArqRedis
from sqlalchemy import func, select, update

from app.agent.replies.messenger import OutgoingMessage, SendResult
from app.db.deletion import UserDataEraser
from app.db.models import EnrichmentRun, ItemStatus, SavedItem, User
from app.ingestion.erasure import IngestionDataEraser
from app.ingestion.jobs import ArqJobQueue
from app.ingestion.repository import lock_user
from app.ingestion.session_state import RedisIngestionState
from app.pipeline.recovery.acknowledgment import (
    CollectionAcknowledger,
    CollectionAcknowledgment,
    claim_collection_acknowledgment,
)
from app.pipeline.recovery.runs import load_item
from app.pipeline.steps.confirm import SaveConfirmer
from tests.pipeline.harness import Pipeline, add_item, add_user
from tests.pipeline.support import RecordingMessenger, capture_event

COLLECTION = "Filmler"
STATE_LIFETIME = timedelta(hours=1)
# Long enough for an erasure that doesn't wait to finish.
ERASURE_WAITS_SECONDS = 0.5


class PausedMessenger(RecordingMessenger):
    """Records each DM, but holds the first send until the test lets it go."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def send(
        self, recipient_id: str, message: OutgoingMessage, last_user_message_at: datetime
    ) -> SendResult:
        self.entered.set()
        await self.release.wait()
        return await super().send(recipient_id, message, last_user_message_at)


@pytest.fixture
def pipeline(committed_pipeline: Pipeline) -> Pipeline:
    return committed_pipeline


def eraser(pipeline: Pipeline, redis: ArqRedis) -> UserDataEraser:
    """The eraser with Postgres, ingestion's Redis state and jobs, and the media store."""
    state = RedisIngestionState(redis, STATE_LIFETIME)
    return UserDataEraser(
        pipeline.sessions, [IngestionDataEraser(state, ArqJobQueue(redis)), pipeline.store]
    )


async def failed_named_item(pipeline: Pipeline) -> CollectionAcknowledgment:
    """A named item whose indexing failed for good, with its confirmation claimed."""
    user_id = await add_user(pipeline.sessions, language="tr")
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    async with pipeline.sessions() as session, session.begin():
        assert await lock_user(session, user_id)
        await session.execute(
            update(SavedItem)
            .where(SavedItem.id == item_id)
            .values(collection=COLLECTION, status=ItemStatus.FAILED)
        )
        item = await load_item(session, item_id)
        assert item is not None
        claimed = await claim_collection_acknowledgment(session, item)
    assert claimed is not None
    return claimed


async def left_of(pipeline: Pipeline, claimed: CollectionAcknowledgment) -> tuple[int, int, int]:
    """How many of the user's rows, items and step rows are left."""
    async with pipeline.sessions() as session:
        users = await session.scalar(
            select(func.count()).select_from(User).where(User.id == claimed.user_id)
        )
        items = await session.scalar(
            select(func.count()).select_from(SavedItem).where(SavedItem.user_id == claimed.user_id)
        )
        runs = await session.scalar(
            select(func.count())
            .select_from(EnrichmentRun)
            .where(EnrichmentRun.saved_item_id == claimed.saved_item_id)
        )
    return users or 0, items or 0, runs or 0


async def redis_keys_of(redis: ArqRedis, user_id: uuid.UUID) -> list[bytes]:
    return list(await redis.keys(f"savely:*{user_id}*"))


async def test_a_user_deleted_after_the_claim_gets_no_dm_and_nothing_is_recreated(
    pipeline: Pipeline, ingestion_redis: ArqRedis
) -> None:
    claimed = await failed_named_item(pipeline)
    acknowledger = CollectionAcknowledger(pipeline.sessions, SaveConfirmer(pipeline.messenger))
    assert await eraser(pipeline, ingestion_redis).erase(claimed.user_id) == 1
    assert await left_of(pipeline, claimed) == (0, 0, 0)

    await acknowledger.send(claimed)

    assert pipeline.messenger.sent == []
    assert await left_of(pipeline, claimed) == (0, 0, 0)
    assert await redis_keys_of(ingestion_redis, claimed.user_id) == []


async def test_deletion_waits_for_an_acknowledgment_being_sent_and_then_erases_everything(
    pipeline: Pipeline, ingestion_redis: ArqRedis
) -> None:
    claimed = await failed_named_item(pipeline)
    messenger = PausedMessenger()
    acknowledger = CollectionAcknowledger(pipeline.sessions, SaveConfirmer(messenger))

    sending = asyncio.create_task(acknowledger.send(claimed))
    await messenger.entered.wait()
    erasing = asyncio.create_task(eraser(pipeline, ingestion_redis).erase(claimed.user_id))
    await asyncio.sleep(ERASURE_WAITS_SECONDS)

    # The send holds the user's row, so the erasure waits for it.
    assert not erasing.done()
    assert await left_of(pipeline, claimed) == (1, 1, 1)
    messenger.release.set()
    await sending
    assert await erasing == 1

    assert [text for _, text in messenger.sent] == ["Filmler koleksiyonuna kaydedildi"]
    assert await left_of(pipeline, claimed) == (0, 0, 0)
    assert await redis_keys_of(ingestion_redis, claimed.user_id) == []
    await acknowledger.send(claimed)
    assert len(messenger.sent) == 1
    assert await left_of(pipeline, claimed) == (0, 0, 0)
