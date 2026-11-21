"""Short-term memory in Redis: the last messages, the working set, expiry and erasure.

These tests need Redis (`TEST_REDIS_URL`). Every id and text is synthetic.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from arq.connections import ArqRedis

from app.agent.tools import SEARCH_ITEMS_TOOL, SearchUsed
from app.memory.session import Session, SessionMemory, SessionMessage, WorkingSet

DAY = timedelta(hours=24)
AT = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
SEARCH = SearchUsed(SEARCH_ITEMS_TOOL, {"type": "place", "filters": {"area": "Beşiktaş"}})
SHORT_LIFETIME = timedelta(milliseconds=200)


@pytest.fixture
async def user_id(ingestion_redis: ArqRedis) -> AsyncIterator[uuid.UUID]:
    user_id = uuid.uuid4()
    yield user_id
    await SessionMemory(ingestion_redis, lifetime=DAY, max_messages=1).erase_user(user_id, ())


def message(number: int) -> SessionMessage:
    role = "user" if number % 2 else "assistant"
    return SessionMessage(role=role, content=f"message {number}", at=AT + timedelta(minutes=number))


def working_set(*item_ids: uuid.UUID, shown: int = 1) -> WorkingSet:
    return WorkingSet(item_ids=item_ids, shown=shown, searches=(SEARCH,))


async def test_an_exchange_and_its_working_set_are_read_back(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    results = working_set(uuid.uuid4(), uuid.uuid4(), shown=1)

    await memory.remember(user_id, [message(1), message(2)], results)

    assert await memory.load(user_id) == Session(
        history=(message(1), message(2)), working_set=results
    )


async def test_only_the_last_messages_are_kept_oldest_first(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=3)

    await memory.remember(user_id, [message(1), message(2)], None)
    await memory.remember(user_id, [message(3), message(4)], None)

    session = await memory.load(user_id)
    assert session.history == (message(2), message(3), message(4))


async def test_an_answer_without_results_keeps_the_last_working_set(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    first, second = working_set(uuid.uuid4()), working_set(uuid.uuid4())

    await memory.remember(user_id, [message(1)], first)
    await memory.remember(user_id, [message(2)], None)
    assert (await memory.load(user_id)).working_set == first

    await memory.remember(user_id, [message(3)], second)
    assert (await memory.load(user_id)).working_set == second


async def test_the_session_expires_after_its_lifetime(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=SHORT_LIFETIME, max_messages=20)
    await memory.remember(user_id, [message(1), message(2)], working_set(uuid.uuid4()))
    await memory.start_reply(user_id, uuid.uuid4())

    await asyncio.sleep(SHORT_LIFETIME.total_seconds() * 2)

    assert await memory.load(user_id) == Session(history=(), working_set=None)
    assert await ingestion_redis.keys(f"*{user_id}*") == []


async def test_every_write_restarts_the_lifetime_of_the_whole_session(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    await memory.remember(user_id, [message(1)], working_set(uuid.uuid4()))
    keys = await ingestion_redis.keys(f"*{user_id}*")
    for key in keys:
        await ingestion_redis.pexpire(key, 1000)

    await memory.remember(user_id, [message(2)], None)

    assert len(keys) == 2
    for key in keys:
        remaining_ms = await ingestion_redis.pttl(key)
        assert DAY.total_seconds() * 1000 - 5000 < remaining_ms <= DAY.total_seconds() * 1000


async def test_only_the_first_try_may_start_a_reply(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    receipt_id, other_receipt_id = uuid.uuid4(), uuid.uuid4()

    assert not await memory.reply_started(user_id, receipt_id)
    assert await memory.start_reply(user_id, receipt_id)
    assert not await memory.start_reply(user_id, receipt_id)
    assert await memory.reply_started(user_id, receipt_id)
    assert not await memory.reply_started(user_id, other_receipt_id)


async def test_erasing_a_user_removes_their_session_and_no_one_elses(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    other_user_id = uuid.uuid4()
    for someone in (user_id, other_user_id):
        await memory.remember(someone, [message(1)], working_set(uuid.uuid4()))
        await memory.start_reply(someone, uuid.uuid4())

    try:
        await memory.erase_user(user_id, ())

        assert await ingestion_redis.keys(f"*{user_id}*") == []
        assert (await memory.load(other_user_id)).history == (message(1),)
    finally:
        await memory.erase_user(other_user_id, ())


@pytest.mark.parametrize(
    ("lifetime", "max_messages"), [(timedelta(0), 20), (DAY, 0)], ids=["no-lifetime", "no-messages"]
)
def test_a_session_that_would_keep_nothing_is_refused(
    ingestion_redis: ArqRedis, lifetime: timedelta, max_messages: int
) -> None:
    with pytest.raises(ValueError):
        SessionMemory(ingestion_redis, lifetime=lifetime, max_messages=max_messages)
