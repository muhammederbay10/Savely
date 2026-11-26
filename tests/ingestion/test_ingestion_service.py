"""Ingestion: first contact, saved items and their dedupe keys, texts bound to shares, jobs.

These tests need Postgres with PostGIS and pgvector. See `tests/database.py`.
All ids, links and texts are synthetic.
"""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from arq.connections import ArqRedis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.base import CaptureEvent, ChatMessage
from app.db.models import ItemStatus, PlatformAccount, SavedItem, User
from app.ingestion.jobs import ANSWER_CHAT_MESSAGE_JOB, INDEX_SAVED_ITEM_JOB
from app.ingestion.service import IngestionOutcome, IngestionService, UnknownUserError
from app.ingestion.session_state import RedisIngestionState, TextReceipt, TextRoute
from tests.db_factories import (
    INSTAGRAM,
    MESSAGE_ID,
    OTHER_MESSAGE_ID,
    OTHER_REEL_ID,
    REEL_ID,
    REEL_URL,
    SENDER_ID,
    count_rows,
)
from tests.webhooks import RecordingJobQueue, capture_payload, chat_payload

OTHER_SENDER_ID = "1000000000000002"
THIRD_MESSAGE_ID = "aWdfZAGFAKEMID0009"
HINT_MESSAGE_ID = "aWdfZAGFAKEMID0007"
QUESTION_MESSAGE_ID = "aWdfZAGFAKEMID0005"
SIGNED_VIDEO_URL = (
    "https://lookaside.fbsbx.com/ig_messaging_cdn/"
    "?asset_id=18000000000000801&signature=FAKE_SIGNATURE_008"
)
CAPTION = "Top 5 sci-fi movies to see"
HINT = "For my movies collection"
QUESTION = "Restaurants in beşiktaş"
SHARED_AT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
HINT_WINDOW = timedelta(seconds=30)
# The gap between a share and its hint in the recorded webhooks (record 0002).
RECORDED_HINT_DELAY = timedelta(seconds=9.1)


class QueueThatFailsOnce(RecordingJobQueue):
    """A queue whose Redis is down for the first job and back for the next."""

    has_failed = False

    async def enqueue(self, job_name: str, job_id: str, arguments: Mapping[str, Any]) -> bool:
        if not self.has_failed:
            self.has_failed = True
            raise RedisConnectionError("synthetic outage")
        return await super().enqueue(job_name, job_id, arguments)


@pytest.fixture
def job_queue() -> RecordingJobQueue:
    return RecordingJobQueue()


@pytest.fixture
def ingestion(
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
) -> IngestionService:
    return IngestionService(session_factory, job_queue, ingestion_state, HINT_WINDOW)


@pytest.fixture
async def user_id(ingestion: IngestionService) -> uuid.UUID:
    return await ingestion.resolve_user(INSTAGRAM, SENDER_ID)


def reel_share(user_id: uuid.UUID, **overrides: object) -> CaptureEvent:
    """A reel shared by DM: a post id, a public link, a caption and a message id."""
    values: dict[str, object] = {
        "user_id": user_id,
        "platform": INSTAGRAM,
        "source": "dm",
        "post_url": REEL_URL,
        "platform_post_id": REEL_ID,
        "caption": CAPTION,
        "media_url": None,
        "media_type": "reel",
        "collection_hint": None,
        "source_message_id": MESSAGE_ID,
        "received_at": SHARED_AT,
    }
    return CaptureEvent.model_validate({**values, **overrides})


def collection_save(user_id: uuid.UUID, message_id: str, **overrides: object) -> CaptureEvent:
    """A reel saved into a collaborative collection: a message id and a video link only."""
    return reel_share(
        user_id,
        post_url=None,
        platform_post_id=None,
        caption=None,
        media_url=SIGNED_VIDEO_URL,
        source_message_id=message_id,
        **overrides,
    )


def text_message(
    user_id: uuid.UUID,
    text: str = HINT,
    message_id: str = HINT_MESSAGE_ID,
    sent_after: timedelta = RECORDED_HINT_DELAY,
) -> ChatMessage:
    return ChatMessage(
        user_id=user_id,
        platform=INSTAGRAM,
        text=text,
        message_id=message_id,
        received_at=SHARED_AT + sent_after,
    )


async def saved_items(session_factory: async_sessionmaker[AsyncSession]) -> list[SavedItem]:
    async with session_factory() as session:
        return list(
            (await session.execute(select(SavedItem).order_by(SavedItem.saved_at))).scalars()
        )


async def only_saved_item(session_factory: async_sessionmaker[AsyncSession]) -> SavedItem:
    (item,) = await saved_items(session_factory)
    return item


async def receipt_of(
    state: RedisIngestionState, user_id: uuid.UUID, message_id: str = HINT_MESSAGE_ID
) -> TextReceipt:
    receipt = await state.text_receipt(user_id, INSTAGRAM, message_id)
    assert receipt is not None
    return receipt


async def count(session_factory: async_sessionmaker[AsyncSession], model: type) -> int:
    async with session_factory() as session:
        return await count_rows(session, model)


# --- First contact -----------------------------------------------------------


async def test_first_message_creates_the_user_and_the_platform_account(
    ingestion: IngestionService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    user_id = await ingestion.resolve_user(INSTAGRAM, SENDER_ID)

    async with session_factory() as session:
        account = (await session.execute(select(PlatformAccount))).scalar_one()
        user = (await session.execute(select(User))).scalar_one()
    assert user.id == user_id
    assert account.user_id == user_id
    assert account.platform == INSTAGRAM
    assert account.platform_user_id == SENDER_ID
    assert account.username is None


async def test_later_messages_from_the_same_account_find_the_same_user(
    ingestion: IngestionService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    first = await ingestion.resolve_user(INSTAGRAM, SENDER_ID)
    second = await ingestion.resolve_user(INSTAGRAM, SENDER_ID)

    assert first == second
    assert await count(session_factory, User) == 1
    assert await count(session_factory, PlatformAccount) == 1


async def test_another_account_gets_its_own_user(
    ingestion: IngestionService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    first = await ingestion.resolve_user(INSTAGRAM, SENDER_ID)
    second = await ingestion.resolve_user(INSTAGRAM, OTHER_SENDER_ID)

    assert first != second
    assert await count(session_factory, User) == 2


async def test_same_account_id_on_another_platform_is_another_user(
    ingestion: IngestionService,
) -> None:
    on_instagram = await ingestion.resolve_user(INSTAGRAM, SENDER_ID)
    on_tiktok = await ingestion.resolve_user("tiktok", SENDER_ID)

    assert on_instagram != on_tiktok


# --- Saved items -------------------------------------------------------------


async def test_reel_share_is_stored_as_a_pending_item_with_its_message_id(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
) -> None:
    outcome = await ingestion.ingest_capture(reel_share(user_id))

    item = await only_saved_item(session_factory)
    assert outcome is IngestionOutcome.ITEM_SAVED
    assert item.user_id == user_id
    assert item.platform == INSTAGRAM
    assert item.platform_post_id == REEL_ID
    assert item.post_url == REEL_URL
    assert item.media_type == "reel"
    assert item.caption == CAPTION
    assert item.source_message_id == MESSAGE_ID
    assert item.saved_at == SHARED_AT
    assert item.status == ItemStatus.PENDING
    assert item.collection is None


async def test_collection_save_is_stored_without_a_post_id_or_post_link(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(collection_save(user_id, MESSAGE_ID))

    item = await only_saved_item(session_factory)
    assert item.platform_post_id is None
    assert item.post_url is None
    assert item.caption is None
    assert item.source_message_id == MESSAGE_ID
    assert item.media_type == "reel"


async def test_new_item_queues_one_indexing_job_whose_payload_is_the_capture(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_redis: ArqRedis,
    user_id: uuid.UUID,
) -> None:
    event = collection_save(user_id, MESSAGE_ID)

    await ingestion.ingest_capture(event)

    item = await only_saved_item(session_factory)
    (job,) = job_queue.jobs
    payload = await capture_payload(ingestion_redis, user_id, item.id)
    assert job.name == INDEX_SAVED_ITEM_JOB
    assert job.job_id == f"index:{item.id}"
    # Ids only: captions, texts and links stay in the payload "delete my data" erases.
    assert job.arguments == {"user_id": str(user_id), "saved_item_id": str(item.id)}
    # The signed media link is not stored on the item, so the payload must carry it.
    assert payload["media_url"] == SIGNED_VIDEO_URL
    assert CaptureEvent.model_validate(payload) == event


# --- Duplicates --------------------------------------------------------------


async def test_duplicate_share_does_not_create_second_item(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    user_id: uuid.UUID,
) -> None:
    first = await ingestion.ingest_capture(reel_share(user_id))
    second = await ingestion.ingest_capture(
        reel_share(
            user_id,
            source_message_id=OTHER_MESSAGE_ID,
            received_at=SHARED_AT + timedelta(days=3),
        )
    )

    item = await only_saved_item(session_factory)
    assert (first, second) == (IngestionOutcome.ITEM_SAVED, IngestionOutcome.ITEM_ALREADY_SAVED)
    assert item.source_message_id == MESSAGE_ID
    assert item.saved_at == SHARED_AT
    assert len(job_queue.named(INDEX_SAVED_ITEM_JOB)) == 1


async def test_redelivered_share_does_not_create_second_item(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    user_id: uuid.UUID,
) -> None:
    first = await ingestion.ingest_capture(reel_share(user_id))
    second = await ingestion.ingest_capture(reel_share(user_id))

    assert (first, second) == (IngestionOutcome.ITEM_SAVED, IngestionOutcome.ITEM_ALREADY_SAVED)
    assert await count(session_factory, SavedItem) == 1
    assert len(job_queue.named(INDEX_SAVED_ITEM_JOB)) == 1


async def test_redelivered_collection_save_does_not_create_second_item(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(collection_save(user_id, MESSAGE_ID))
    outcome = await ingestion.ingest_capture(collection_save(user_id, MESSAGE_ID))

    assert outcome is IngestionOutcome.ITEM_ALREADY_SAVED
    assert await count(session_factory, SavedItem) == 1
    assert len(job_queue.named(INDEX_SAVED_ITEM_JOB)) == 1


async def test_two_collection_saves_without_a_post_id_stay_two_items(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    user_id: uuid.UUID,
) -> None:
    first = await ingestion.ingest_capture(collection_save(user_id, MESSAGE_ID))
    second = await ingestion.ingest_capture(
        collection_save(user_id, OTHER_MESSAGE_ID, received_at=SHARED_AT + timedelta(seconds=20))
    )

    items = await saved_items(session_factory)
    assert (first, second) == (IngestionOutcome.ITEM_SAVED, IngestionOutcome.ITEM_SAVED)
    assert [item.source_message_id for item in items] == [MESSAGE_ID, OTHER_MESSAGE_ID]
    assert len(job_queue.named(INDEX_SAVED_ITEM_JOB)) == 2


async def test_same_reel_shared_by_dm_and_saved_into_the_collection_is_two_items(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    await ingestion.ingest_capture(collection_save(user_id, OTHER_MESSAGE_ID))

    assert await count(session_factory, SavedItem) == 2


async def test_two_different_posts_are_two_items(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    await ingestion.ingest_capture(
        reel_share(user_id, platform_post_id=OTHER_REEL_ID, source_message_id=OTHER_MESSAGE_ID)
    )

    assert await count(session_factory, SavedItem) == 2


async def test_same_post_shared_by_two_users_is_one_item_each(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    user_id: uuid.UUID,
) -> None:
    other_user_id = await ingestion.resolve_user(INSTAGRAM, OTHER_SENDER_ID)

    await ingestion.ingest_capture(reel_share(user_id))
    outcome = await ingestion.ingest_capture(
        reel_share(other_user_id, source_message_id=OTHER_MESSAGE_ID)
    )

    items = await saved_items(session_factory)
    assert outcome is IngestionOutcome.ITEM_SAVED
    assert {item.user_id for item in items} == {user_id, other_user_id}


async def test_share_of_an_item_that_is_already_indexed_queues_no_new_job(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    async with session_factory.begin() as session:
        await session.execute(update(SavedItem).values(status=ItemStatus.DONE))
    job_queue.jobs.clear()

    outcome = await ingestion.ingest_capture(
        reel_share(user_id, source_message_id=OTHER_MESSAGE_ID)
    )

    assert outcome is IngestionOutcome.ITEM_ALREADY_SAVED
    assert job_queue.jobs == []


async def test_item_whose_job_could_not_be_queued_is_queued_when_the_share_is_redelivered(
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    job_queue = QueueThatFailsOnce()
    ingestion = IngestionService(session_factory, job_queue, ingestion_state, HINT_WINDOW)

    with pytest.raises(RedisConnectionError):
        await ingestion.ingest_capture(reel_share(user_id))
    outcome = await ingestion.ingest_capture(reel_share(user_id))

    item = await only_saved_item(session_factory)
    assert outcome is IngestionOutcome.ITEM_ALREADY_SAVED
    assert [job.job_id for job in job_queue.jobs] == [f"index:{item.id}"]


# --- Texts: bound to a share by send time, decided by the chat worker ------------


async def test_text_right_after_a_share_is_bound_to_that_item_and_queued_for_the_chat_worker(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))

    outcome = await ingestion.ingest_chat_message(text_message(user_id))

    # No model call and no label here: the chat worker decides (decision record 0018).
    item = await only_saved_item(session_factory)
    receipt = await receipt_of(ingestion_state, user_id)
    assert outcome is IngestionOutcome.CHAT_QUEUED
    assert (receipt.route, receipt.saved_item_id) == (TextRoute.UNDECIDED, item.id)
    assert item.collection is None
    (job,) = job_queue.named(ANSWER_CHAT_MESSAGE_JOB)
    assert job.arguments["message_receipt_id"] == str(receipt.id)


async def test_text_after_a_collection_save_is_bound_to_it(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(collection_save(user_id, MESSAGE_ID))

    await ingestion.ingest_chat_message(text_message(user_id))

    receipt = await receipt_of(ingestion_state, user_id)
    assert receipt.saved_item_id == (await only_saved_item(session_factory)).id


async def test_text_at_the_end_of_the_window_is_still_bound(
    ingestion: IngestionService,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))

    await ingestion.ingest_chat_message(text_message(user_id, sent_after=HINT_WINDOW))

    assert (await receipt_of(ingestion_state, user_id)).route is TextRoute.UNDECIDED


async def test_text_after_the_window_is_chat_with_no_share_to_decide_about(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    just_too_late = HINT_WINDOW + timedelta(milliseconds=1)

    outcome = await ingestion.ingest_chat_message(
        text_message(user_id, text=QUESTION, sent_after=just_too_late)
    )

    # The window is the upper bound: the chat worker makes no model call for it.
    receipt = await receipt_of(ingestion_state, user_id)
    assert outcome is IngestionOutcome.CHAT_QUEUED
    assert (receipt.route, receipt.saved_item_id) == (TextRoute.CHAT, None)
    assert (await only_saved_item(session_factory)).collection is None
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1


async def test_window_length_comes_from_the_service_setting(
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    short_window = timedelta(seconds=5)
    ingestion = IngestionService(session_factory, job_queue, ingestion_state, short_window)
    await ingestion.ingest_capture(reel_share(user_id))

    await ingestion.ingest_chat_message(text_message(user_id))

    assert short_window < RECORDED_HINT_DELAY
    assert (await receipt_of(ingestion_state, user_id)).route is TextRoute.CHAT


async def test_text_from_a_user_without_a_capture_is_a_chat_message(
    ingestion: IngestionService,
    job_queue: RecordingJobQueue,
    ingestion_redis: ArqRedis,
    user_id: uuid.UUID,
) -> None:
    message = text_message(user_id, text=QUESTION, message_id=QUESTION_MESSAGE_ID)

    outcome = await ingestion.ingest_chat_message(message)

    (job,) = job_queue.jobs
    receipt_id = uuid.UUID(job.arguments["message_receipt_id"])
    payload = await chat_payload(ingestion_redis, user_id, receipt_id)
    assert outcome is IngestionOutcome.CHAT_QUEUED
    assert job.name == ANSWER_CHAT_MESSAGE_JOB
    assert job.job_id == f"chat:{receipt_id}"
    assert job.arguments == {"user_id": str(user_id), "message_receipt_id": str(receipt_id)}
    assert ChatMessage.model_validate(payload) == message


async def test_text_sent_before_the_share_is_a_chat_message(
    ingestion: IngestionService,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))

    outcome = await ingestion.ingest_chat_message(
        text_message(user_id, sent_after=timedelta(seconds=-1))
    )

    assert outcome is IngestionOutcome.CHAT_QUEUED
    assert (await receipt_of(ingestion_state, user_id)).route is TextRoute.CHAT


async def test_two_texts_in_the_window_are_both_bound_and_both_queued(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    # Which one names the collection is the chat worker's decision; the first wins.
    await ingestion.ingest_capture(reel_share(user_id))
    await ingestion.ingest_chat_message(text_message(user_id))

    await ingestion.ingest_chat_message(
        text_message(
            user_id,
            text=QUESTION,
            message_id=QUESTION_MESSAGE_ID,
            sent_after=timedelta(seconds=15),
        )
    )

    item = await only_saved_item(session_factory)
    first = await receipt_of(ingestion_state, user_id)
    second = await receipt_of(ingestion_state, user_id, QUESTION_MESSAGE_ID)
    assert first.saved_item_id == second.saved_item_id == item.id
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 2


async def test_redelivered_bound_text_keeps_its_receipt_and_is_queued_once(
    ingestion: IngestionService,
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    first = await ingestion.ingest_chat_message(text_message(user_id))
    receipt = await receipt_of(ingestion_state, user_id)

    replay = await ingestion.ingest_chat_message(text_message(user_id))

    assert (first, replay) == (IngestionOutcome.CHAT_QUEUED, IngestionOutcome.CHAT_ALREADY_QUEUED)
    assert await receipt_of(ingestion_state, user_id) == receipt
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1


async def test_text_is_bound_to_the_latest_capture_only(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(collection_save(user_id, MESSAGE_ID))
    await ingestion.ingest_capture(
        collection_save(user_id, OTHER_MESSAGE_ID, received_at=SHARED_AT + timedelta(seconds=4))
    )

    await ingestion.ingest_chat_message(text_message(user_id))

    _, latest = await saved_items(session_factory)
    assert (await receipt_of(ingestion_state, user_id)).saved_item_id == latest.id


async def test_text_is_never_bound_to_another_users_capture(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    other_user_id = await ingestion.resolve_user(INSTAGRAM, OTHER_SENDER_ID)
    await ingestion.ingest_capture(reel_share(user_id))

    outcome = await ingestion.ingest_chat_message(text_message(other_user_id))

    receipt = await receipt_of(ingestion_state, other_user_id)
    assert outcome is IngestionOutcome.CHAT_QUEUED
    assert (receipt.route, receipt.saved_item_id) == (TextRoute.CHAT, None)
    assert (await only_saved_item(session_factory)).collection is None


async def test_redelivered_chat_message_is_queued_once(
    ingestion: IngestionService, job_queue: RecordingJobQueue, user_id: uuid.UUID
) -> None:
    message = text_message(user_id, text=QUESTION, message_id=QUESTION_MESSAGE_ID)

    first = await ingestion.ingest_chat_message(message)
    second = await ingestion.ingest_chat_message(message)

    assert (first, second) == (
        IngestionOutcome.CHAT_QUEUED,
        IngestionOutcome.CHAT_ALREADY_QUEUED,
    )
    assert len(job_queue.jobs) == 1


async def test_two_different_texts_are_two_chat_jobs(
    ingestion: IngestionService, job_queue: RecordingJobQueue, user_id: uuid.UUID
) -> None:
    await ingestion.ingest_chat_message(
        text_message(user_id, text=QUESTION, message_id=QUESTION_MESSAGE_ID)
    )
    await ingestion.ingest_chat_message(
        text_message(user_id, text=QUESTION, message_id=THIRD_MESSAGE_ID)
    )

    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 2


async def test_every_new_message_gets_the_next_number_and_a_redelivery_keeps_its_own(
    ingestion: IngestionService, ingestion_state: RedisIngestionState, user_id: uuid.UUID
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    question = text_message(user_id, text=QUESTION, message_id=QUESTION_MESSAGE_ID)
    await ingestion.ingest_chat_message(question)
    await ingestion.ingest_chat_message(question)
    await ingestion.ingest_chat_message(
        text_message(user_id, text=QUESTION, message_id=THIRD_MESSAGE_ID)
    )

    numbers = [
        receipt.number
        for message_id in (QUESTION_MESSAGE_ID, THIRD_MESSAGE_ID)
        if (receipt := await ingestion_state.text_receipt(user_id, INSTAGRAM, message_id))
    ]
    # The share is message 1; the redelivered question keeps number 2.
    assert numbers == [2, 3]
    assert await ingestion_state.last_message_number(user_id) == 3


# --- Regressions from the first review (fix round 1) ---------------------------

POST_A = "18000000000000211"
POST_B = "18000000000000212"
SHARE_A_MESSAGE_ID = "aWdfZAGFAKEMID0021"
SHARE_B_MESSAGE_ID = "aWdfZAGFAKEMID0022"
SHARE_A_AGAIN_MESSAGE_ID = "aWdfZAGFAKEMID0023"
LATE_HINT_MESSAGE_ID = "aWdfZAGFAKEMID0024"


async def item_with_post_id(
    session_factory: async_sessionmaker[AsyncSession], post_id: str
) -> SavedItem:
    async with session_factory() as session:
        statement = select(SavedItem).where(SavedItem.platform_post_id == post_id)
        return (await session.execute(statement)).scalar_one()


async def test_text_after_a_repeated_share_is_bound_to_the_repeated_post_not_a_later_saved_one(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(
        reel_share(user_id, platform_post_id=POST_A, source_message_id=SHARE_A_MESSAGE_ID)
    )
    await ingestion.ingest_capture(
        reel_share(
            user_id,
            platform_post_id=POST_B,
            source_message_id=SHARE_B_MESSAGE_ID,
            received_at=SHARED_AT + timedelta(seconds=60),
        )
    )
    repeat = await ingestion.ingest_capture(
        reel_share(
            user_id,
            platform_post_id=POST_A,
            source_message_id=SHARE_A_AGAIN_MESSAGE_ID,
            received_at=SHARED_AT + timedelta(seconds=65),
        )
    )

    await ingestion.ingest_chat_message(
        text_message(user_id, message_id=LATE_HINT_MESSAGE_ID, sent_after=timedelta(seconds=70))
    )

    post_a = await item_with_post_id(session_factory, POST_A)
    receipt = await receipt_of(ingestion_state, user_id, LATE_HINT_MESSAGE_ID)
    assert repeat is IngestionOutcome.ITEM_ALREADY_SAVED
    assert receipt.saved_item_id == post_a.id
    # The item keeps the message it came in with, so answers still reply to it.
    assert post_a.source_message_id == SHARE_A_MESSAGE_ID
    assert post_a.saved_at == SHARED_AT
    assert await count(session_factory, SavedItem) == 2


async def test_redelivered_text_keeps_its_chat_route_after_a_delayed_earlier_share(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    user_id: uuid.UUID,
) -> None:
    question = text_message(user_id, text=QUESTION, message_id=QUESTION_MESSAGE_ID)
    first = await ingestion.ingest_chat_message(question)
    # The share was sent 9 seconds before the question but delivered after it.
    await ingestion.ingest_capture(reel_share(user_id))

    replay = await ingestion.ingest_chat_message(question)

    assert first is IngestionOutcome.CHAT_QUEUED
    assert replay is IngestionOutcome.CHAT_ALREADY_QUEUED
    assert (await only_saved_item(session_factory)).collection is None
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1


async def test_text_whose_chat_job_failed_to_queue_keeps_its_route_on_redelivery(
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    job_queue = QueueThatFailsOnce()
    ingestion = IngestionService(session_factory, job_queue, ingestion_state, HINT_WINDOW)
    question = text_message(user_id, text=QUESTION, message_id=QUESTION_MESSAGE_ID)

    with pytest.raises(RedisConnectionError):
        await ingestion.ingest_chat_message(question)
    await ingestion.ingest_capture(reel_share(user_id))
    replay = await ingestion.ingest_chat_message(question)
    second_replay = await ingestion.ingest_chat_message(question)

    assert replay is IngestionOutcome.CHAT_QUEUED
    assert second_replay is IngestionOutcome.CHAT_ALREADY_QUEUED
    assert (await only_saved_item(session_factory)).collection is None
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1


async def test_redelivered_text_stays_bound_to_its_item_after_a_delayed_share(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    hint = text_message(user_id)
    await ingestion.ingest_chat_message(hint)
    # Sent between the first share and the hint, but delivered after the hint.
    await ingestion.ingest_capture(
        reel_share(
            user_id,
            platform_post_id=OTHER_REEL_ID,
            source_message_id=OTHER_MESSAGE_ID,
            received_at=SHARED_AT + timedelta(seconds=5),
        )
    )

    replay = await ingestion.ingest_chat_message(hint)

    first_item = await item_with_post_id(session_factory, REEL_ID)
    assert replay is IngestionOutcome.CHAT_ALREADY_QUEUED
    assert (await receipt_of(ingestion_state, user_id)).saved_item_id == first_item.id
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1


@pytest.mark.parametrize("message_kind", ["capture", "text"])
async def test_message_of_a_user_deleted_meanwhile_is_refused_and_leaves_nothing_in_redis(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
    message_kind: str,
) -> None:
    async with session_factory.begin() as session:
        await session.execute(delete(User).where(User.id == user_id))

    with pytest.raises(UnknownUserError):
        if message_kind == "capture":
            await ingestion.ingest_capture(reel_share(user_id))
        else:
            await ingestion.ingest_chat_message(text_message(user_id))

    assert (
        await ingestion_state.latest_share_between(user_id, INSTAGRAM, SHARED_AT, SHARED_AT) is None
    )
    assert await ingestion_state.text_receipt(user_id, INSTAGRAM, HINT_MESSAGE_ID) is None
    assert job_queue.jobs == []
    assert await count(session_factory, SavedItem) == 0


# --- Regressions from the second review (fix round 2) ---------------------------

FRESH_SIGNED_IMAGE_URL = (
    "https://lookaside.fbsbx.com/ig_messaging_cdn/"
    "?asset_id=18000000000000301&signature=FAKE_SIGNATURE_301"
)
OLD_SIGNED_IMAGE_URL = (
    "https://lookaside.fbsbx.com/ig_messaging_cdn/"
    "?asset_id=18000000000000301&signature=FAKE_SIGNATURE_300"
)


def photo_post_share(user_id: uuid.UUID, message_id: str, **overrides: object) -> CaptureEvent:
    """A photo post shared by DM: a post id and a signed image link, but no public link."""
    values: dict[str, object] = {
        "media_type": "post",
        "post_url": None,
        "media_url": OLD_SIGNED_IMAGE_URL,
        "source_message_id": message_id,
    }
    return reel_share(user_id, **{**values, **overrides})


async def test_delayed_text_is_bound_to_the_share_sent_before_it_not_a_later_one(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    # Sent: A at 0 s, the hint at 10 s, B at 20 s. Delivered: A, B, then the hint.
    await ingestion.ingest_capture(
        reel_share(user_id, platform_post_id=POST_A, source_message_id=SHARE_A_MESSAGE_ID)
    )
    await ingestion.ingest_capture(
        reel_share(
            user_id,
            platform_post_id=POST_B,
            source_message_id=SHARE_B_MESSAGE_ID,
            received_at=SHARED_AT + timedelta(seconds=20),
        )
    )
    hint = text_message(user_id, message_id=LATE_HINT_MESSAGE_ID, sent_after=timedelta(seconds=10))

    await ingestion.ingest_chat_message(hint)
    await ingestion.ingest_chat_message(hint)

    post_a = await item_with_post_id(session_factory, POST_A)
    receipt = await receipt_of(ingestion_state, user_id, LATE_HINT_MESSAGE_ID)
    assert receipt.saved_item_id == post_a.id


async def test_text_whose_share_already_has_a_collection_is_chat_and_skips_older_shares(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    # One item per text: an earlier share in the window doesn't take over.
    await ingestion.ingest_capture(
        reel_share(user_id, platform_post_id=POST_A, source_message_id=SHARE_A_MESSAGE_ID)
    )
    await ingestion.ingest_capture(
        reel_share(
            user_id,
            platform_post_id=POST_B,
            source_message_id=SHARE_B_MESSAGE_ID,
            received_at=SHARED_AT + timedelta(seconds=5),
        )
    )
    async with session_factory.begin() as session:
        await session.execute(
            update(SavedItem).where(SavedItem.platform_post_id == POST_B).values(collection=HINT)
        )

    await ingestion.ingest_chat_message(
        text_message(
            user_id,
            text=QUESTION,
            message_id=LATE_HINT_MESSAGE_ID,
            sent_after=timedelta(seconds=9),
        )
    )

    receipt = await receipt_of(ingestion_state, user_id, LATE_HINT_MESSAGE_ID)
    assert (receipt.route, receipt.saved_item_id) == (TextRoute.CHAT, None)


async def test_replayed_old_share_keeps_the_fresh_media_link_of_a_later_share(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_redis: ArqRedis,
    user_id: uuid.UUID,
) -> None:
    old_share = photo_post_share(user_id, MESSAGE_ID)
    fresh_share = photo_post_share(
        user_id,
        OTHER_MESSAGE_ID,
        media_url=FRESH_SIGNED_IMAGE_URL,
        received_at=SHARED_AT + timedelta(minutes=5),
    )

    await ingestion.ingest_capture(old_share)
    await ingestion.ingest_capture(fresh_share)
    await ingestion.ingest_capture(old_share)

    item = await only_saved_item(session_factory)
    payload = await capture_payload(ingestion_redis, user_id, item.id)
    assert payload["media_url"] == FRESH_SIGNED_IMAGE_URL
    assert item.source_message_id == MESSAGE_ID
    assert item.saved_at == SHARED_AT


async def test_delayed_old_share_keeps_the_fresh_media_link_delivered_before_it(
    ingestion: IngestionService,
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_redis: ArqRedis,
    user_id: uuid.UUID,
) -> None:
    fresh_share = photo_post_share(
        user_id,
        OTHER_MESSAGE_ID,
        media_url=FRESH_SIGNED_IMAGE_URL,
        received_at=SHARED_AT + timedelta(minutes=5),
    )
    delayed_old_share = photo_post_share(user_id, MESSAGE_ID)

    await ingestion.ingest_capture(fresh_share)
    outcome = await ingestion.ingest_capture(delayed_old_share)

    item = await only_saved_item(session_factory)
    payload = await capture_payload(ingestion_redis, user_id, item.id)
    assert outcome is IngestionOutcome.ITEM_ALREADY_SAVED
    assert payload["media_url"] == FRESH_SIGNED_IMAGE_URL
    assert item.source_message_id == OTHER_MESSAGE_ID


class QueueWhoseJobDecidesAtOnce(RecordingJobQueue):
    """A chat worker that settles the route between queuing and the queued mark."""

    def __init__(self, state: RedisIngestionState, message: ChatMessage) -> None:
        super().__init__()
        self._state = state
        self._message = message

    async def enqueue(self, job_name: str, job_id: str, arguments: Mapping[str, Any]) -> bool:
        queued = await super().enqueue(job_name, job_id, arguments)
        if job_name != ANSWER_CHAT_MESSAGE_JOB:
            return queued
        message = self._message
        receipt = await self._state.text_receipt(
            message.user_id, message.platform, message.message_id
        )
        assert receipt is not None
        decided = receipt.model_copy(update={"route": TextRoute.COLLECTION_HINT})
        await self._state.save_text_receipt(
            message.user_id, message.platform, message.message_id, decided
        )
        return queued


async def test_marking_the_job_queued_keeps_a_route_the_chat_worker_decided_meanwhile(
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    hint = text_message(user_id)
    ingestion = IngestionService(
        session_factory,
        QueueWhoseJobDecidesAtOnce(ingestion_state, hint),
        ingestion_state,
        HINT_WINDOW,
    )
    await ingestion.ingest_capture(reel_share(user_id))

    await ingestion.ingest_chat_message(hint)

    receipt = await receipt_of(ingestion_state, user_id)
    assert (receipt.route, receipt.chat_queued) == (TextRoute.COLLECTION_HINT, True)


async def test_a_button_tap_right_after_a_share_is_chat_with_no_share_to_decide_about(
    ingestion: IngestionService,
    ingestion_state: RedisIngestionState,
    user_id: uuid.UUID,
) -> None:
    await ingestion.ingest_capture(reel_share(user_id))
    tap = text_message(user_id).model_copy(update={"quick_reply_payload": "synthetic-payload"})

    await ingestion.ingest_chat_message(tap)

    receipt = await receipt_of(ingestion_state, user_id)
    assert (receipt.route, receipt.saved_item_id) == (TextRoute.CHAT, None)
