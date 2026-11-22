"""An ordinary answer that overlaps "delete my data" never outlives the deletion.

These tests use a database of their own with real commits, so the answer and
the deletion run on separate connections and really wait for each other's
locks. The text model is scripted and Meta's Send API is a recording mock.
They need Postgres (`TEST_DATABASE_ADMIN_URL`) and Redis (`TEST_REDIS_URL`).
Every id and text is synthetic.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.agent import AgentAnswer, AgentTurn
from app.agent.chat import ChatOutcome
from app.agent.replies.answers import AnswerMessages, DeliveryReport
from app.agent.replies.templates import (
    DELETION_CONFIRM_PAYLOAD,
    deletion_confirmation,
    deletion_done,
    reply_wording,
)
from app.agent.tools import (
    CONFIRM_DATA_DELETION_TOOL,
    REQUEST_DATA_DELETION_TOOL,
    SEARCH_ITEMS_TOOL,
)
from app.db.session import create_engine, create_session_factory
from tests.agent.agent_support import CONFIRM_SECONDS, final_answer, tool_calls
from tests.agent.chat_support import (
    ChatStores,
    ChatUnderTest,
    build_chat,
    count_messages,
    seed_place,
    stored_user,
)
from tests.database import run_alembic
from tests.redis_inspection import all_stored_bytes

SENDER_ID = "1000000000000193"
REQUEST = tool_calls((REQUEST_DATA_DELETION_TOOL, {}))
CONFIRM = tool_calls((CONFIRM_DATA_DELETION_TOOL, {}))
QUESTION = deletion_confirmation(CONFIRM_SECONDS // 60, reply_wording("en")).text
DONE = deletion_done(reply_wording("en")).text
# How long the deletion is given to finish while an answer is being sent. It
# never finishes then, so this only bounds how long the test waits.
BLOCKED_SECONDS = 1.0


@pytest.fixture
async def session_factory(
    empty_database_url: URL, tmp_path: Path
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """The worker's session factory on a migrated database of this test's own."""
    run_alembic(empty_database_url, tmp_path, "upgrade", "head")
    engine = create_engine(empty_database_url.render_as_string(hide_password=False))
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@dataclass
class Pause:
    """Holds a step until the test lets it go on."""

    reached: asyncio.Event = field(default_factory=asyncio.Event)
    resume: asyncio.Event = field(default_factory=asyncio.Event)

    async def hold(self) -> None:
        self.reached.set()
        await self.resume.wait()


async def ask_and_tap(stores: ChatStores, user_id: uuid.UUID, chat: ChatUnderTest) -> uuid.UUID:
    """The user asks to delete their data, gets the question, and taps "Yes, delete"."""
    asked = await stores.send_text(user_id, "delete my data")
    assert await chat.responder.respond(user_id, asked, last_try=False) is ChatOutcome.ANSWERED
    return await stores.send_text(
        user_id, "Yes, delete", quick_reply_payload=DELETION_CONFIRM_PAYLOAD
    )


async def assert_nothing_left(stores: ChatStores, user_id: uuid.UUID, redis_url: str) -> None:
    assert await stored_user(stores, user_id) is None
    assert await count_messages(stores, user_id) == 0
    stored = await all_stored_bytes(redis_url)
    assert str(user_id).encode() not in stored
    assert b"synthetic stale question" not in stored


async def test_an_answer_ready_before_a_deletion_is_neither_sent_nor_stored_after_it(
    chat_stores: ChatStores, redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    ordinary = build_chat(chat_stores, [final_answer("en", "A stale answer.")])
    deleting = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    question = await chat_stores.send_text(user_id, "synthetic stale question")
    pause = Pause()
    answer = ordinary.agent.agent.answer

    async def answer_then_pause(turn: AgentTurn) -> AgentAnswer:
        result = await answer(turn)
        await pause.hold()
        return result

    monkeypatch.setattr(ordinary.agent.agent, "answer", answer_then_pause)
    answering = asyncio.create_task(ordinary.responder.respond(user_id, question, last_try=False))
    await pause.reached.wait()

    tapped = await ask_and_tap(chat_stores, user_id, deleting)
    assert await deleting.responder.respond(user_id, tapped, last_try=False) is ChatOutcome.ANSWERED
    pause.resume.set()

    assert await answering is ChatOutcome.NO_USER
    assert ordinary.sent_bodies() == []
    assert deleting.sent_texts() == [QUESTION, DONE]
    await assert_nothing_left(chat_stores, user_id, redis_url)


async def test_a_deletion_waits_for_an_answer_being_sent_and_then_erases_it(
    chat_stores: ChatStores, redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    ordinary = build_chat(chat_stores, [final_answer("en", "Sent before the deletion.")])
    deleting = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    question = await chat_stores.send_text(user_id, "synthetic stale question")
    pause = Pause()
    send = ordinary.sender.send

    async def pause_then_send(
        recipient_id: str, messages: AnswerMessages, last_user_message_at: datetime
    ) -> DeliveryReport:
        await pause.hold()
        return await send(recipient_id, messages, last_user_message_at)

    monkeypatch.setattr(ordinary.sender, "send", pause_then_send)
    # The question is delivered first: answers to one sender go out one at a time.
    tapped = await ask_and_tap(chat_stores, user_id, deleting)
    answering = asyncio.create_task(ordinary.responder.respond(user_id, question, last_try=False))
    await pause.reached.wait()

    deleting_task = asyncio.create_task(deleting.responder.respond(user_id, tapped, last_try=False))
    finished, _ = await asyncio.wait({deleting_task}, timeout=BLOCKED_SECONDS)
    assert not finished, "the deletion did not wait for the answer being sent"
    pause.resume.set()

    assert await answering is ChatOutcome.ANSWERED
    assert await deleting_task is ChatOutcome.ANSWERED
    assert ordinary.sent_texts() == ["Sent before the deletion."]
    assert deleting.sent_texts() == [QUESTION, DONE]
    await assert_nothing_left(chat_stores, user_id, redis_url)


async def test_answers_to_one_sender_are_sent_and_stored_one_at_a_time(
    chat_stores: ChatStores, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    first_place = await seed_place(
        chat_stores, user_id, "Synthetic Meze", district="Bebek", price="cheap", saved_days_ago=2
    )
    second_place = await seed_place(
        chat_stores,
        user_id,
        "Synthetic Köfte",
        district="Beşiktaş",
        price="cheap",
        saved_days_ago=1,
    )
    search = tool_calls((SEARCH_ITEMS_TOOL, {"category": "restaurant"}))
    first = build_chat(chat_stores, [search, final_answer("en", "List A:", [first_place.id])])
    second = build_chat(
        chat_stores,
        [
            search,
            final_answer("en", "List B:", [second_place.id]),
            final_answer("en", "That one."),
        ],
    )
    delivered: list[tuple[uuid.UUID, ...]] = []
    pause = Pause()

    def recording(chat: ChatUnderTest, *, paused: bool) -> None:
        send = chat.sender.send

        async def send_and_record(
            recipient_id: str, messages: AnswerMessages, last_user_message_at: datetime
        ) -> DeliveryReport:
            if paused:
                await pause.hold()
            report = await send(recipient_id, messages, last_user_message_at)
            delivered.append(tuple(card.item_id for card in messages.results))
            return report

        monkeypatch.setattr(chat.sender, "send", send_and_record)

    recording(first, paused=True)
    recording(second, paused=False)
    turns: list[AgentTurn] = []
    answer = second.agent.agent.answer

    async def answer_and_record(turn: AgentTurn) -> AgentAnswer:
        turns.append(turn)
        return await answer(turn)

    monkeypatch.setattr(second.agent.agent, "answer", answer_and_record)
    text_a = await chat_stores.send_text(user_id, "restaurants in Bebek")
    text_b = await chat_stores.send_text(user_id, "restaurants in Beşiktaş")

    # A has its delivery order and is inside its sender when B is ready to send.
    answering_a = asyncio.create_task(first.responder.respond(user_id, text_a, last_try=False))
    await pause.reached.wait()
    answering_b = asyncio.create_task(second.responder.respond(user_id, text_b, last_try=False))
    finished, _ = await asyncio.wait({answering_b}, timeout=BLOCKED_SECONDS)
    assert not finished, "B was sent while A, sent before it, was still being sent"
    pause.resume.set()

    assert await answering_a is ChatOutcome.ANSWERED
    assert await answering_b is ChatOutcome.ANSWERED
    assert delivered == [(first_place.id,), (second_place.id,)]
    last_list = delivered[-1]
    session = await chat_stores.memory.load(user_id)
    assert session.working_set is not None
    assert session.working_set.item_ids == last_list
    follow_up = await chat_stores.send_text(user_id, "the first one")
    await second.responder.respond(user_id, follow_up, last_try=False)
    assert tuple(turns[-1].working_set) == last_list
