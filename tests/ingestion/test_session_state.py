"""Ingestion's per-user state in Redis: the share history and the capture payloads.

These tests need a Redis server (`TEST_REDIS_URL`). Every test uses a new user id.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from arq.connections import ArqRedis

from app.adapters.base import CaptureEvent
from app.ingestion.session_state import RedisIngestionState
from tests.webhooks import capture_payload

LIFETIME = timedelta(hours=1)
SENT_AT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
INSTAGRAM = "instagram"
WINDOW = timedelta(seconds=30)
ITEM_A = uuid.UUID("00000000-0000-4000-8000-00000000000a")
ITEM_B = uuid.UUID("00000000-0000-4000-8000-00000000000b")
SIGNED_LINK = "https://lookaside.fbsbx.com/ig_messaging_cdn/?asset_id=18000000000000301&signature="


@pytest.fixture
def state(ingestion_redis: ArqRedis) -> RedisIngestionState:
    return RedisIngestionState(ingestion_redis, LIFETIME)


@pytest.fixture
def user_id() -> uuid.UUID:
    return uuid.uuid4()


def shares_key(user_id: uuid.UUID) -> str:
    return f"savely:ingestion:{user_id}:shares"


def photo_post(user_id: uuid.UUID, sent_at: datetime, signature: str) -> CaptureEvent:
    return CaptureEvent(
        user_id=user_id,
        platform=INSTAGRAM,
        source="dm",
        post_url=None,
        platform_post_id="18000000000000301",
        caption=None,
        media_url=SIGNED_LINK + signature,
        media_type="post",
        collection_hint=None,
        source_message_id=f"aWdfZAGFAKEMID{signature}",
        received_at=sent_at,
    )


async def test_last_share_sent_before_a_time_is_found_although_a_later_one_arrived_first(
    state: RedisIngestionState, user_id: uuid.UUID
) -> None:
    await state.record_share(user_id, INSTAGRAM, ITEM_B, SENT_AT + timedelta(seconds=20))
    await state.record_share(user_id, INSTAGRAM, ITEM_A, SENT_AT)

    text_sent_at = SENT_AT + timedelta(seconds=10)
    share = await state.latest_share_between(
        user_id, INSTAGRAM, text_sent_at - WINDOW, text_sent_at
    )

    assert share is not None
    assert (share.saved_item_id, share.sent_at) == (ITEM_A, SENT_AT)


async def test_share_outside_the_span_is_not_found(
    state: RedisIngestionState, user_id: uuid.UUID
) -> None:
    await state.record_share(user_id, INSTAGRAM, ITEM_A, SENT_AT)

    text_sent_at = SENT_AT + WINDOW + timedelta(milliseconds=1)
    share = await state.latest_share_between(
        user_id, INSTAGRAM, text_sent_at - WINDOW, text_sent_at
    )

    assert share is None


async def test_share_on_another_platform_is_not_found(
    state: RedisIngestionState, user_id: uuid.UUID
) -> None:
    await state.record_share(user_id, "tiktok", ITEM_B, SENT_AT + timedelta(seconds=2))
    await state.record_share(user_id, INSTAGRAM, ITEM_A, SENT_AT)

    share = await state.latest_share_between(
        user_id, INSTAGRAM, SENT_AT, SENT_AT + timedelta(seconds=5)
    )

    assert share is not None
    assert share.saved_item_id == ITEM_A


async def test_repeated_share_of_one_item_is_kept_at_each_send_time(
    state: RedisIngestionState, user_id: uuid.UUID
) -> None:
    await state.record_share(user_id, INSTAGRAM, ITEM_A, SENT_AT)
    await state.record_share(user_id, INSTAGRAM, ITEM_B, SENT_AT + timedelta(seconds=60))
    await state.record_share(user_id, INSTAGRAM, ITEM_A, SENT_AT + timedelta(seconds=65))

    at_seventy = SENT_AT + timedelta(seconds=70)
    share = await state.latest_share_between(user_id, INSTAGRAM, at_seventy - WINDOW, at_seventy)

    assert share is not None
    assert share.saved_item_id == ITEM_A


async def test_shares_older_than_one_lifetime_before_the_newest_are_dropped(
    state: RedisIngestionState, ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    await state.record_share(user_id, INSTAGRAM, ITEM_A, SENT_AT)
    await state.record_share(user_id, INSTAGRAM, ITEM_B, SENT_AT + LIFETIME)
    await state.record_share(
        user_id, INSTAGRAM, ITEM_B, SENT_AT + LIFETIME + timedelta(milliseconds=1)
    )

    assert await ingestion_redis.zcard(shares_key(user_id)) == 2
    assert await state.latest_share_between(user_id, INSTAGRAM, SENT_AT, SENT_AT) is None
    assert 0 < await ingestion_redis.ttl(shares_key(user_id)) <= LIFETIME.total_seconds()


async def test_erasing_a_user_removes_their_share_history(
    state: RedisIngestionState, ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    await state.record_share(user_id, INSTAGRAM, ITEM_A, SENT_AT)

    await state.erase_user(user_id)

    assert await ingestion_redis.exists(shares_key(user_id)) == 0


async def test_messages_are_numbered_in_order_until_the_user_is_erased(
    state: RedisIngestionState, ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    assert await state.last_message_number(user_id) == 0
    numbers = [await state.number_message(user_id, INSTAGRAM, f"mid-{n}") for n in range(3)]
    assert numbers == [1, 2, 3]
    # A repeated delivery keeps its number; a message without an id always counts.
    assert await state.number_message(user_id, INSTAGRAM, "mid-1") == 2
    assert await state.number_message(user_id, INSTAGRAM, None) == 4
    assert await state.last_message_number(user_id) == 4
    assert await ingestion_redis.ttl(f"savely:ingestion:{user_id}:messages") > 0

    await state.erase_user(user_id)

    assert await state.last_message_number(user_id) == 0


async def test_capture_payload_sent_later_replaces_an_earlier_one(
    state: RedisIngestionState, ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    await state.save_capture_payload(user_id, ITEM_A, photo_post(user_id, SENT_AT, "OLD"))
    fresh = photo_post(user_id, SENT_AT + timedelta(minutes=5), "FRESH")

    await state.save_capture_payload(user_id, ITEM_A, fresh)

    assert (await capture_payload(ingestion_redis, user_id, ITEM_A))["media_url"] == fresh.media_url


async def test_capture_payload_sent_earlier_never_replaces_a_later_one(
    state: RedisIngestionState, ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    fresh = photo_post(user_id, SENT_AT + timedelta(minutes=5), "FRESH")
    await state.save_capture_payload(user_id, ITEM_A, fresh)

    await state.save_capture_payload(user_id, ITEM_A, photo_post(user_id, SENT_AT, "OLD"))

    assert (await capture_payload(ingestion_redis, user_id, ITEM_A))["media_url"] == fresh.media_url


async def test_same_capture_delivered_again_keeps_its_payload(
    state: RedisIngestionState, ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    share = photo_post(user_id, SENT_AT, "SAME")

    await state.save_capture_payload(user_id, ITEM_A, share)
    await state.save_capture_payload(user_id, ITEM_A, share)

    assert (
        CaptureEvent.model_validate(await capture_payload(ingestion_redis, user_id, ITEM_A))
        == share
    )
