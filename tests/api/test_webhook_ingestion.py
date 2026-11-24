"""From a signed webhook to stored items and queued jobs, with the recorded payloads.

The request goes through the real endpoint, adapter, ingestion service and
database. Only the job queue is an in-memory stand-in for Redis.

These tests need Postgres with PostGIS and pgvector. See `tests/database.py`.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from arq.connections import ArqRedis
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.webhooks import create_instagram_webhook_router
from app.config.base import Settings
from app.db.models import ItemStatus, PlatformAccount, SavedItem, User
from app.ingestion.jobs import ANSWER_CHAT_MESSAGE_JOB, INDEX_SAVED_ITEM_JOB
from app.ingestion.service import IngestionService
from app.ingestion.session_state import RedisIngestionState, TextReceipt, TextRoute
from tests.db_factories import count_rows
from tests.webhooks import (
    COLLECTION_PHOTO,
    COLLECTION_REEL,
    HINT_AFTER_REEL,
    LEGACY_SHARE,
    OTHER_COLLECTION_REEL,
    PHOTO_POST_SHARE,
    REEL_BEFORE_HINT,
    REEL_SHARE,
    SHARE_FIXTURES,
    TEXT_QUESTION,
    RecordingJobQueue,
    body_of,
    capture_payload,
    chat_payload,
    fixture_body,
    fixture_json,
    fixture_message,
    signed_headers,
)

WEBHOOK_PATH = "/webhooks/instagram"
SENDER_ID = "1000000000000001"
OTHER_MESSAGE_ID = "aWdfZAGFAKEMID0099"


@pytest.fixture
def job_queue() -> RecordingJobQueue:
    return RecordingJobQueue()


@pytest.fixture
async def webhooks(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
) -> AsyncIterator[AsyncClient]:
    ingestion = IngestionService(
        session_factory,
        job_queue,
        ingestion_state,
        timedelta(seconds=settings.collection_hint_window_seconds),
    )
    app = FastAPI()
    app.include_router(create_instagram_webhook_router(settings, ingestion))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as http_client:
        yield http_client


async def deliver(webhooks: AsyncClient, body: bytes) -> None:
    response = await webhooks.post(WEBHOOK_PATH, content=body, headers=signed_headers(body))
    assert response.status_code == 200


def shared_again(name: str) -> bytes:
    """The same share sent a second time: the same post in a new message."""
    payload = fixture_json(name)
    event = payload["entry"][0]["messaging"][0]
    event["message"]["mid"] = OTHER_MESSAGE_ID
    event["timestamp"] += 60_000
    return body_of(payload)


async def saved_items(session_factory: async_sessionmaker[AsyncSession]) -> list[SavedItem]:
    async with session_factory() as session:
        statement = select(SavedItem).order_by(SavedItem.saved_at)
        return list((await session.execute(statement)).scalars())


# --- Acceptance: one saved item per post and per message ------------------------


@pytest.mark.parametrize("name", [REEL_SHARE, PHOTO_POST_SHARE, COLLECTION_PHOTO])
async def test_sharing_the_same_post_twice_creates_only_one_saved_item(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    name: str,
) -> None:
    await deliver(webhooks, fixture_body(name))
    await deliver(webhooks, shared_again(name))

    (item,) = await saved_items(session_factory)
    assert item.source_message_id == fixture_message(name)["mid"]
    assert len(job_queue.named(INDEX_SAVED_ITEM_JOB)) == 1


@pytest.mark.parametrize("name", SHARE_FIXTURES)
async def test_webhook_delivered_twice_creates_only_one_saved_item(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    name: str,
) -> None:
    await deliver(webhooks, fixture_body(name))
    await deliver(webhooks, fixture_body(name))

    (item,) = await saved_items(session_factory)
    assert item.source_message_id == fixture_message(name)["mid"]
    assert item.status == ItemStatus.PENDING
    assert len(job_queue.named(INDEX_SAVED_ITEM_JOB)) == 1


async def test_two_reels_saved_into_the_collection_stay_two_saved_items(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
) -> None:
    await deliver(webhooks, fixture_body(COLLECTION_REEL))
    await deliver(webhooks, fixture_body(OTHER_COLLECTION_REEL))

    items = await saved_items(session_factory)
    assert [item.source_message_id for item in items] == [
        fixture_message(COLLECTION_REEL)["mid"],
        fixture_message(OTHER_COLLECTION_REEL)["mid"],
    ]
    assert all(item.platform_post_id is None and item.post_url is None for item in items)
    assert len(job_queue.named(INDEX_SAVED_ITEM_JOB)) == 2


# --- What is stored and queued ---------------------------------------------------


async def test_reel_share_is_stored_with_its_caption_link_and_message_id(
    webhooks: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await deliver(webhooks, fixture_body(REEL_SHARE))

    (item,) = await saved_items(session_factory)
    payload = fixture_message(REEL_SHARE)["attachments"][0]["payload"]
    assert item.platform == "instagram"
    assert item.media_type == "reel"
    assert item.platform_post_id == payload["reel_video_id"]
    assert item.post_url == payload["url"]
    assert item.caption == payload["title"]
    assert item.source_message_id == fixture_message(REEL_SHARE)["mid"]


@pytest.mark.parametrize("name", [PHOTO_POST_SHARE, COLLECTION_REEL, LEGACY_SHARE])
async def test_signed_media_link_reaches_the_indexing_job_but_not_the_database(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_redis: ArqRedis,
    name: str,
) -> None:
    signed_link = fixture_message(name)["attachments"][0]["payload"]["url"]

    await deliver(webhooks, fixture_body(name))

    (item,) = await saved_items(session_factory)
    (job,) = job_queue.named(INDEX_SAVED_ITEM_JOB)
    payload = await capture_payload(ingestion_redis, item.user_id, item.id)
    assert item.post_url is None
    assert job.arguments["saved_item_id"] == str(item.id)
    assert signed_link not in str(job.arguments)
    assert payload["media_url"] == signed_link
    assert payload["post_url"] is None
    assert payload["source_message_id"] == item.source_message_id


async def test_first_message_from_an_account_creates_its_user(
    webhooks: AsyncClient, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await deliver(webhooks, fixture_body(REEL_SHARE))
    await deliver(webhooks, fixture_body(TEXT_QUESTION))

    async with session_factory() as session:
        account = (await session.execute(select(PlatformAccount))).scalar_one()
        user_count = await count_rows(session, User)
    (item,) = await saved_items(session_factory)
    assert user_count == 1
    assert (account.platform, account.platform_user_id) == ("instagram", SENDER_ID)
    assert item.user_id == account.user_id


# --- Texts -----------------------------------------------------------------------


async def bound_receipt(
    ingestion_state: RedisIngestionState, item: SavedItem, fixture_name: str
) -> TextReceipt:
    receipt = await ingestion_state.text_receipt(
        item.user_id, "instagram", fixture_message(fixture_name)["mid"]
    )
    assert receipt is not None
    return receipt


async def test_text_sent_nine_seconds_after_a_reel_is_bound_to_it_and_left_to_the_chat_worker(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
) -> None:
    await deliver(webhooks, fixture_body(REEL_BEFORE_HINT))
    await deliver(webhooks, fixture_body(HINT_AFTER_REEL))

    # The webhook calls no model: the chat worker decides (decision record 0018).
    (item,) = await saved_items(session_factory)
    receipt = await bound_receipt(ingestion_state, item, HINT_AFTER_REEL)
    assert item.collection is None
    assert (receipt.route, receipt.saved_item_id) == (TextRoute.UNDECIDED, item.id)
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1


async def test_question_with_no_share_before_it_is_queued_for_the_response_agent(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_redis: ArqRedis,
) -> None:
    await deliver(webhooks, fixture_body(TEXT_QUESTION))
    await deliver(webhooks, fixture_body(TEXT_QUESTION))

    (job,) = job_queue.jobs
    payload = await chat_payload(
        ingestion_redis,
        uuid.UUID(job.arguments["user_id"]),
        uuid.UUID(job.arguments["message_receipt_id"]),
    )
    assert job.name == ANSWER_CHAT_MESSAGE_JOB
    assert payload["text"] == "Restaurants in beşiktaş"
    assert payload["message_id"] == fixture_message(TEXT_QUESTION)["mid"]
    assert await saved_items(session_factory) == []


async def test_question_long_after_a_share_is_not_taken_as_its_collection_hint(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
) -> None:
    # In the recordings the question was sent 50 seconds after the reel.
    await deliver(webhooks, fixture_body(REEL_SHARE))
    await deliver(webhooks, fixture_body(TEXT_QUESTION))

    (item,) = await saved_items(session_factory)
    assert item.collection is None
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1


async def test_hint_delivered_after_a_later_share_is_still_bound_to_the_reel_sent_before_it(
    webhooks: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    job_queue: RecordingJobQueue,
    ingestion_state: RedisIngestionState,
) -> None:
    hint_event = fixture_json(HINT_AFTER_REEL)["entry"][0]["messaging"][0]
    later_share = fixture_json(REEL_SHARE)
    later_event = later_share["entry"][0]["messaging"][0]
    later_event["message"]["mid"] = OTHER_MESSAGE_ID
    later_event["timestamp"] = hint_event["timestamp"] + 10_000

    await deliver(webhooks, fixture_body(REEL_BEFORE_HINT))
    await deliver(webhooks, body_of(later_share))
    await deliver(webhooks, fixture_body(HINT_AFTER_REEL))

    hinted_reel, later_reel = await saved_items(session_factory)
    receipt = await bound_receipt(ingestion_state, hinted_reel, HINT_AFTER_REEL)
    assert hinted_reel.source_message_id == fixture_message(REEL_BEFORE_HINT)["mid"]
    assert receipt.saved_item_id == hinted_reel.id
    assert later_reel.collection is None
    assert len(job_queue.named(ANSWER_CHAT_MESSAGE_JOB)) == 1
