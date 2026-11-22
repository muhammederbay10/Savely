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
from app.memory.session import (
    Exchange,
    Reply,
    ReplyState,
    Session,
    SessionMemory,
    SessionMessage,
    WorkingSet,
)

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


def exchange(*numbers: int, results: WorkingSet | None = None) -> Exchange:
    return Exchange(
        messages=tuple(message(number) for number in numbers),
        working_set=results,
        language="en",
    )


async def remember(
    memory: SessionMemory, user_id: uuid.UUID, numbers: list[int], results: WorkingSet | None
) -> None:
    """Send and store one answered message's exchange, in order."""
    receipt_id = uuid.uuid4()
    answered = exchange(*numbers, results=results)
    order = await memory.start_reply(user_id, receipt_id, answered)
    assert order is not None
    await memory.remember(user_id, receipt_id, answered, order)


async def test_an_exchange_and_its_working_set_are_read_back(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    results = working_set(uuid.uuid4(), uuid.uuid4(), shown=1)

    await remember(memory, user_id, [1, 2], results)

    assert await memory.load(user_id) == Session(
        history=(message(1), message(2)), working_set=results
    )


async def test_only_the_last_messages_are_kept_oldest_first(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=3)

    await remember(memory, user_id, [1, 2], None)
    await remember(memory, user_id, [3, 4], None)

    session = await memory.load(user_id)
    assert session.history == (message(2), message(3), message(4))


async def test_an_answer_without_results_keeps_the_last_working_set(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    first, second = working_set(uuid.uuid4()), working_set(uuid.uuid4())

    await remember(memory, user_id, [1], first)
    await remember(memory, user_id, [2], None)
    assert (await memory.load(user_id)).working_set == first

    await remember(memory, user_id, [3], second)
    assert (await memory.load(user_id)).working_set == second


async def test_the_session_expires_after_its_lifetime(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=SHORT_LIFETIME, max_messages=20)
    await remember(memory, user_id, [1, 2], working_set(uuid.uuid4()))
    await memory.start_reply(user_id, uuid.uuid4(), exchange(3))

    await asyncio.sleep(SHORT_LIFETIME.total_seconds() * 2)

    assert await memory.load(user_id) == Session(history=(), working_set=None)
    assert await ingestion_redis.keys(f"*{user_id}*") == []


async def test_every_write_restarts_the_lifetime_of_the_whole_session(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    await remember(memory, user_id, [1], working_set(uuid.uuid4()))
    keys = await ingestion_redis.keys(f"*{user_id}*")
    for key in keys:
        await ingestion_redis.pexpire(key, 1000)

    await remember(memory, user_id, [2], None)

    assert len(keys) == 3
    for key in keys:
        remaining_ms = await ingestion_redis.pttl(key)
        assert DAY.total_seconds() * 1000 - 5000 < remaining_ms <= DAY.total_seconds() * 1000


async def test_only_the_first_try_may_start_a_reply(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    receipt_id, other_receipt_id = uuid.uuid4(), uuid.uuid4()

    first, second = exchange(1, 2), exchange(3, 4)

    assert await memory.reply(user_id, receipt_id) is None
    assert await memory.start_reply(user_id, receipt_id, first)
    assert not await memory.start_reply(user_id, receipt_id, second)
    assert await memory.reply(user_id, receipt_id) == Reply(ReplyState.SENDING, first, order=1)
    assert await memory.start_reply(user_id, other_receipt_id, None) == 2
    assert await memory.reply(user_id, other_receipt_id) == Reply(ReplyState.SENDING, order=2)


async def test_a_reply_goes_from_sending_to_delivered_to_stored_with_its_exchange(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    receipt_id = uuid.uuid4()
    sent = exchange(1, 2, results=working_set(uuid.uuid4(), uuid.uuid4()))
    order = await memory.start_reply(user_id, receipt_id, sent)
    assert order == 1

    await memory.reply_delivered(user_id, receipt_id, sent_whole=True)
    delivered = Reply(ReplyState.DELIVERED, sent, sent_whole=True, order=order)
    assert await memory.reply(user_id, receipt_id) == delivered
    # Only a reply that is still being sent moves on.
    await memory.reply_delivered(user_id, receipt_id, sent_whole=False)
    assert await memory.reply(user_id, receipt_id) == delivered
    assert (await memory.load(user_id)).history == ()

    await memory.remember(user_id, receipt_id, sent, order)
    assert await memory.reply(user_id, receipt_id) == Reply(ReplyState.STORED)
    assert await memory.load(user_id) == Session(
        history=(message(1), message(2)), working_set=sent.working_set
    )


async def test_an_exchange_stored_late_lands_in_its_place_once_and_keeps_newer_results(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=3)
    older_id, newer_id = uuid.uuid4(), uuid.uuid4()
    older = exchange(1, 2, results=working_set(uuid.uuid4()))
    newer = exchange(3, 4, results=working_set(uuid.uuid4()))
    older_order = await memory.start_reply(user_id, older_id, older)
    newer_order = await memory.start_reply(user_id, newer_id, newer)
    assert older_order is not None and newer_order is not None

    await memory.remember(user_id, newer_id, newer, newer_order)
    for _ in range(2):  # the older answer's store is tried again
        await memory.remember(user_id, older_id, older, older_order)

    session = await memory.load(user_id)
    # Merged by time, the oldest dropped beyond three messages, nothing twice.
    assert session.history == (message(2), message(3), message(4))
    assert session.working_set == newer.working_set
    assert await memory.reply(user_id, older_id) == Reply(ReplyState.STORED)


async def test_a_delivery_of_an_erased_reply_writes_nothing(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    receipt_id = uuid.uuid4()
    await memory.start_reply(user_id, receipt_id, exchange(1, 2))
    await memory.erase_user(user_id, ())

    await memory.reply_delivered(user_id, receipt_id, sent_whole=True)

    assert await ingestion_redis.keys(f"*{user_id}*") == []


async def test_erasing_a_user_removes_their_session_and_no_one_elses(
    ingestion_redis: ArqRedis, user_id: uuid.UUID
) -> None:
    memory = SessionMemory(ingestion_redis, lifetime=DAY, max_messages=20)
    other_user_id = uuid.uuid4()
    for someone in (user_id, other_user_id):
        await remember(memory, someone, [1], working_set(uuid.uuid4()))
        await memory.start_reply(someone, uuid.uuid4(), None)

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
