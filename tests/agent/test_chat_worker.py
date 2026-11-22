"""The chat worker: a DM in, the agent's answer out, the exchange remembered.

Also the "Yes, delete" routing (record 0015, Q9), the erasure of everything
the worker stores, retries and the rule that an answer is never sent twice.
The text model is scripted and Meta's Send API is a recording mock. These tests
need Postgres (`TEST_DATABASE_ADMIN_URL`) and Redis (`TEST_REDIS_URL`). Every
id and text is synthetic.
"""

import itertools
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pytest
from arq import Retry
from arq.connections import ArqRedis
from arq.jobs import Job
from redis.exceptions import RedisError
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from app.agent.chat import ChatOutcome, RetryLaterError
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
    SET_PROFILE_TOOL,
)
from app.agent.worker import ChatWorkerResources, answer_chat_message, build_worker
from app.config.base import Settings
from app.config.chat import ChatSettings
from app.db.deletion import UserDataEraser
from app.ingestion.jobs import ANSWER_CHAT_MESSAGE_JOB, CHAT_QUEUE, chat_job_id
from app.memory.session import ReplyState, Session
from app.models.errors import ModelRequestError, ModelUnavailableError
from tests.agent.agent_support import CONFIRM_SECONDS, MAX_STEPS, final_answer, tool_calls
from tests.agent.chat_support import (
    ChatStores,
    ChatUnderTest,
    build_chat,
    count_messages,
    seed_place,
    stored_messages,
    stored_user,
)
from tests.agent.support import json_response
from tests.redis_inspection import all_stored_bytes

SENDER_ID = "1000000000000191"
OTHER_SENDER_ID = "1000000000000192"
REPEATED_POST_ID = "18000000000000291"
CONFIRM_MINUTES = CONFIRM_SECONDS // 60
RETRY_DELAY_SECONDS = 3.0
MAX_TRIES = 3

REQUEST = tool_calls((REQUEST_DATA_DELETION_TOOL, {}))
CONFIRM = tool_calls((CONFIRM_DATA_DELETION_TOOL, {}))
MODEL_DOWN = ModelUnavailableError("synthetic outage", operation="generate", model="fake/text")
REJECTED = ModelRequestError("synthetic rejection", operation="generate", model="fake/text")
QUESTION = deletion_confirmation(CONFIRM_MINUTES, reply_wording("en")).text
DONE = deletion_done(reply_wording("en")).text


def chat_settings(**overrides: object) -> ChatSettings:
    values: dict[str, object] = {
        "session_ttl_seconds": 86400,
        "session_history_messages": 20,
        "chat_concurrency": 4,
        "chat_max_tries": MAX_TRIES,
        "chat_retry_delay_seconds": RETRY_DELAY_SECONDS,
        "chat_run_deadline_seconds": 60.0,
        "chat_job_timeout_seconds": 120.0,
    }
    return ChatSettings(_env_file=None, **{**values, **overrides})  # type: ignore[arg-type]


async def no_close() -> None:
    return None


async def test_a_text_dm_goes_through_the_worker_to_an_answer_sent_by_the_agent(
    chat_stores: ChatStores, ingestion_redis: ArqRedis, settings: Settings
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    item = await seed_place(
        chat_stores, user_id, "Synthetic Meze", district="Beşiktaş", price="cheap", saved_days_ago=1
    )
    chat = build_chat(
        chat_stores,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "restaurant"})),
            final_answer("en", "Your restaurant:", [item.id]),
        ],
    )
    receipt = await chat_stores.send_text(user_id, "my restaurants")

    async def inject(ctx: dict[str, Any]) -> None:
        ctx["chat_resources"] = ChatWorkerResources(
            responder=chat.responder,
            max_tries=MAX_TRIES,
            retry_delay_seconds=RETRY_DELAY_SECONDS,
            close=no_close,
        )

    worker = build_worker(
        settings, chat_settings(), on_startup=inject, redis_pool=ingestion_redis, burst=True
    )
    try:
        # The worker borrows the fixture's Redis client, which the fixture closes.
        await worker.main()

        job = Job(chat_job_id(receipt), ingestion_redis, _queue_name=CHAT_QUEUE)
        assert await job.result(timeout=5) == ChatOutcome.ANSWERED.value
    finally:
        await chat_stores.jobs.discard([chat_job_id(receipt)])

    intro, result = chat.sent_bodies()
    assert intro == {"recipient": {"id": SENDER_ID}, "message": {"text": "Your restaurant:"}}
    assert result["recipient"] == {"id": SENDER_ID}
    assert result["reply_to"] == {"mid": item.source_message_id}
    assert await stored_messages(chat_stores, user_id) == [
        ("user", "my restaurants"),
        ("assistant", "Your restaurant:"),
    ]
    session = await chat_stores.memory.load(user_id)
    assert [(message.role, message.content) for message in session.history] == [
        ("user", "my restaurants"),
        ("assistant", "Your restaurant:"),
    ]
    assert session.working_set is not None
    assert session.working_set.item_ids == (item.id,)
    user = await stored_user(chat_stores, user_id)
    assert user is not None
    assert user.language == "en"


async def test_a_yes_delete_tap_right_after_the_delivered_question_deletes(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    asked = await chat_stores.send_text(user_id, "delete my data")
    assert await chat.responder.respond(user_id, asked, last_try=False) is ChatOutcome.ANSWERED

    tapped = await chat_stores.send_text(
        user_id, "Yes, delete", quick_reply_payload=DELETION_CONFIRM_PAYLOAD
    )
    outcome = await chat.responder.respond(user_id, tapped, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert chat.sent_texts() == [QUESTION, DONE]
    assert chat.sent_bodies()[-1]["recipient"] == {"id": SENDER_ID}
    assert await stored_user(chat_stores, user_id) is None


@pytest.mark.parametrize("when", ["typed-yes", "later-tap", "expired-tap"])
async def test_a_typed_yes_a_later_tap_or_an_expired_tap_deletes_nothing(
    chat_stores: ChatStores, when: str
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    # The model calls confirm_data_deletion every time; only code decides.
    script = [REQUEST, final_answer("en", "…")]
    if when == "later-tap":
        script.append(final_answer("en", "Hi!"))
    script.extend([CONFIRM, final_answer("en", "Nothing was deleted.")])
    chat = build_chat(chat_stores, script)
    asked = await chat_stores.send_text(user_id, "delete my data")
    await chat.responder.respond(user_id, asked, last_try=False)
    if when == "typed-yes":
        last = await chat_stores.send_text(user_id, "yes")
    else:
        if when == "later-tap":
            other = await chat_stores.send_text(user_id, "hello")
            await chat.responder.respond(user_id, other, last_try=False)
        else:
            chat_stores.clock.advance(CONFIRM_SECONDS)
        last = await chat_stores.send_text(
            user_id, "Yes, delete", quick_reply_payload=DELETION_CONFIRM_PAYLOAD
        )

    outcome = await chat.responder.respond(user_id, last, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert await stored_user(chat_stores, user_id) is not None
    if when == "typed-yes":
        # The question is asked again, with its button.
        assert chat.sent_texts()[-1] == QUESTION
        assert chat.sent_bodies()[-1]["message"]["quick_replies"]  # type: ignore[index]
    else:
        assert chat.sent_texts()[-1] == "Nothing was deleted."


async def test_another_users_tap_deletes_neither_of_them(chat_stores: ChatStores) -> None:
    asking = await chat_stores.new_user(SENDER_ID)
    tapping = await chat_stores.new_user(OTHER_SENDER_ID)
    chat = build_chat(
        chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM, final_answer("en", "No.")]
    )
    asked = await chat_stores.send_text(asking, "delete my data")
    await chat.responder.respond(asking, asked, last_try=False)

    tapped = await chat_stores.send_text(
        tapping, "Yes, delete", quick_reply_payload=DELETION_CONFIRM_PAYLOAD
    )
    await chat.responder.respond(tapping, tapped, last_try=False)

    assert await stored_user(chat_stores, asking) is not None
    assert await stored_user(chat_stores, tapping) is not None
    assert chat.sent_bodies()[-1]["recipient"] == {"id": OTHER_SENDER_ID}


async def ask_to_delete(chat_stores: ChatStores, chat: ChatUnderTest, user_id: uuid.UUID) -> None:
    """The user asks to delete their data, and the question is delivered."""
    asked = await chat_stores.send_text(user_id, "delete my data")
    assert await chat.responder.respond(user_id, asked, last_try=False) is ChatOutcome.ANSWERED


async def tap_yes_delete(chat_stores: ChatStores, user_id: uuid.UUID) -> uuid.UUID:
    return await chat_stores.send_text(
        user_id, "Yes, delete", quick_reply_payload=DELETION_CONFIRM_PAYLOAD
    )


async def test_a_tap_after_another_message_deletes_nothing_even_if_its_job_runs_first(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    # The model calls confirm_data_deletion on every turn; only code decides.
    chat = build_chat(
        chat_stores,
        [
            REQUEST,
            final_answer("en", "…"),
            CONFIRM,
            final_answer("en", "Nothing was deleted."),
            final_answer("en", "Kept."),
        ],
    )
    await ask_to_delete(chat_stores, chat, user_id)
    declined = await chat_stores.send_text(user_id, "No, keep my data")
    tapped = await tap_yes_delete(chat_stores, user_id)
    # A repeated delivery of the earlier "no" changes nothing either.
    assert await chat_stores.deliver_again(declined) == declined

    # The tap's job runs before the job of the message the user sent before it.
    assert await chat.responder.respond(user_id, tapped, last_try=False) is ChatOutcome.ANSWERED
    assert await chat.responder.respond(user_id, declined, last_try=False) is ChatOutcome.ANSWERED

    assert await stored_user(chat_stores, user_id) is not None
    assert chat.sent_texts() == [QUESTION, "Nothing was deleted.", "Kept."]


async def test_a_tap_after_a_shared_post_deletes_nothing(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(
        chat_stores,
        [REQUEST, final_answer("en", "…"), CONFIRM, final_answer("en", "Nothing was deleted.")],
    )
    await ask_to_delete(chat_stores, chat, user_id)
    await chat_stores.share(user_id)
    tapped = await tap_yes_delete(chat_stores, user_id)

    await chat.responder.respond(user_id, tapped, last_try=False)

    assert await stored_user(chat_stores, user_id) is not None
    assert chat.sent_texts()[-1] == "Nothing was deleted."


async def test_a_repeated_delivery_of_an_earlier_share_does_not_stand_in_the_way(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    shared = await chat_stores.share(user_id)
    await ask_to_delete(chat_stores, chat, user_id)
    # Meta delivers the share sent before the question again.
    await chat_stores.ingestion.ingest_capture(shared)
    tapped = await tap_yes_delete(chat_stores, user_id)

    await chat.responder.respond(user_id, tapped, last_try=False)

    assert await stored_user(chat_stores, user_id) is None
    assert chat.sent_texts() == [QUESTION, DONE]


async def test_a_new_share_of_an_already_shared_post_still_stands_in_the_way(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(
        chat_stores,
        [REQUEST, final_answer("en", "…"), CONFIRM, final_answer("en", "Nothing was deleted.")],
    )
    await chat_stores.share(user_id, post_id=REPEATED_POST_ID)
    await ask_to_delete(chat_stores, chat, user_id)
    await chat_stores.share(user_id, post_id=REPEATED_POST_ID)
    tapped = await tap_yes_delete(chat_stores, user_id)

    await chat.responder.respond(user_id, tapped, last_try=False)

    assert await stored_user(chat_stores, user_id) is not None
    assert chat.sent_texts()[-1] == "Nothing was deleted."


def failing(
    monkeypatch: pytest.MonkeyPatch, target: object, name: str, times: int | None
) -> list[bool]:
    """Make a Redis write fail, `times` times or until the returned switch is emptied."""
    original = getattr(target, name)
    switch = [True]
    failures = itertools.count() if times is None else iter(range(times))

    async def fail_then_work(*args: object, **kwargs: object) -> object:
        if switch and next(failures, None) is not None:
            raise RedisError("synthetic outage")
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, name, fail_then_work)
    return switch


async def test_a_question_whose_request_failed_to_save_once_still_becomes_confirmable(
    chat_stores: ChatStores, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    failing(monkeypatch, chat_stores.requests, "save", times=1)

    await ask_to_delete(chat_stores, chat, user_id)
    assert await stored_messages(chat_stores, user_id) == [
        ("user", "delete my data"),
        ("assistant", QUESTION),
    ]
    tapped = await tap_yes_delete(chat_stores, user_id)
    await chat.responder.respond(user_id, tapped, last_try=False)

    assert await stored_user(chat_stores, user_id) is None
    assert chat.sent_texts() == [QUESTION, DONE]


@pytest.mark.parametrize("meanwhile", ["nothing", "another-message", "expiry"])
async def test_a_question_whose_request_is_saved_by_a_later_try_keeps_its_message_and_deadline(
    chat_stores: ChatStores, monkeypatch: pytest.MonkeyPatch, meanwhile: str
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    script = [REQUEST, final_answer("en", "…")]
    if meanwhile == "another-message":
        script.append(final_answer("en", "Hello."))
    script.extend([CONFIRM, final_answer("en", "Nothing was deleted.")])
    chat = build_chat(chat_stores, script)
    outage = failing(monkeypatch, chat_stores.requests, "save", times=None)
    asked = await chat_stores.send_text(user_id, "delete my data")

    with pytest.raises(RetryLaterError):
        await chat.responder.respond(user_id, asked, last_try=False)
    if meanwhile == "another-message":
        hello = await chat_stores.send_text(user_id, "hello")
        await chat.responder.respond(user_id, hello, last_try=False)
    elif meanwhile == "expiry":
        chat_stores.clock.advance(CONFIRM_SECONDS)
    outage.clear()
    assert await chat.responder.respond(user_id, asked, last_try=False) is ChatOutcome.ANSWERED
    assert ("assistant", QUESTION) in await stored_messages(chat_stores, user_id)
    tapped = await tap_yes_delete(chat_stores, user_id)
    await chat.responder.respond(user_id, tapped, last_try=False)

    assert chat.sent_texts().count(QUESTION) == 1
    if meanwhile == "nothing":
        assert await stored_user(chat_stores, user_id) is None
        assert chat.sent_texts()[-1] == DONE
    else:
        assert await stored_user(chat_stores, user_id) is not None
        assert chat.sent_texts()[-1] == "Nothing was deleted."


async def test_a_tap_whose_job_is_retried_later_still_deletes(
    chat_stores: ChatStores, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    await ask_to_delete(chat_stores, chat, user_id)
    tapped = await tap_yes_delete(chat_stores, user_id)
    load = chat_stores.memory.load
    failures = [RedisError("synthetic outage")]

    async def load_after_an_outage(user: uuid.UUID) -> Session:
        if failures:
            raise failures.pop()
        return await load(user)

    monkeypatch.setattr(chat_stores.memory, "load", load_after_an_outage)

    with pytest.raises(RetryLaterError):
        await chat.responder.respond(user_id, tapped, last_try=False)
    outcome = await chat.responder.respond(user_id, tapped, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert await stored_user(chat_stores, user_id) is None
    assert chat.sent_texts() == [QUESTION, DONE]


async def test_a_message_retried_while_a_later_tap_runs_lets_the_tap_delete_nothing(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(
        chat_stores,
        [
            REQUEST,
            final_answer("en", "…"),
            MODEL_DOWN,
            CONFIRM,
            final_answer("en", "Nothing was deleted."),
            final_answer("en", "Okay."),
        ],
    )
    await ask_to_delete(chat_stores, chat, user_id)
    other = await chat_stores.send_text(user_id, "wait, first show my cafés")
    tapped = await tap_yes_delete(chat_stores, user_id)

    with pytest.raises(RetryLaterError):
        await chat.responder.respond(user_id, other, last_try=False)
    await chat.responder.respond(user_id, tapped, last_try=False)
    await chat.responder.respond(user_id, other, last_try=True)

    assert await stored_user(chat_stores, user_id) is not None
    assert chat.sent_texts() == [QUESTION, "Nothing was deleted.", "Okay."]


async def test_a_repeated_delivery_of_an_earlier_message_does_not_stand_in_the_way(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    asked = await chat_stores.send_text(user_id, "delete my data")
    await chat.responder.respond(user_id, asked, last_try=False)
    # Meta delivers the request again after the question was sent.
    await chat_stores.deliver_again(asked)
    tapped = await tap_yes_delete(chat_stores, user_id)

    await chat.responder.respond(user_id, tapped, last_try=False)

    assert await stored_user(chat_stores, user_id) is None
    assert chat.sent_texts() == [QUESTION, DONE]


async def test_a_tap_delivered_twice_is_one_job_that_deletes_once(
    chat_stores: ChatStores, ingestion_redis: ArqRedis
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [REQUEST, final_answer("en", "…"), CONFIRM])
    await ask_to_delete(chat_stores, chat, user_id)
    tapped = await tap_yes_delete(chat_stores, user_id)

    assert await chat_stores.deliver_again(tapped) == tapped
    queued = await ingestion_redis.queued_jobs(queue_name=CHAT_QUEUE)
    assert [job.job_id for job in queued].count(chat_job_id(tapped)) == 1
    await chat.responder.respond(user_id, tapped, last_try=False)
    assert await chat.responder.respond(user_id, tapped, last_try=False) is ChatOutcome.NO_MESSAGE

    assert await stored_user(chat_stores, user_id) is None
    assert chat.sent_texts() == [QUESTION, DONE]


async def test_after_delete_my_data_the_session_working_set_messages_and_profile_are_gone(
    chat_stores: ChatStores, redis_url: str
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    staying_id = await chat_stores.new_user(OTHER_SENDER_ID)
    item = await seed_place(
        chat_stores,
        user_id,
        "Synthetic Köfte",
        district="Beşiktaş",
        price="cheap",
        saved_days_ago=1,
    )
    marker = f"synthetic-{uuid.uuid4().hex}"
    chat = build_chat(
        chat_stores,
        [
            tool_calls((SET_PROFILE_TOOL, {"home_area": marker, "country": "TR"})),
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "restaurant"})),
            final_answer("en", f"Noted {marker}. Your restaurant:", [item.id]),
            final_answer("en", f"Hello {marker}"),
            REQUEST,
            final_answer("en", "…"),
            CONFIRM,
        ],
    )
    for sender, text in ((user_id, f"I live in {marker}"), (staying_id, f"hi {marker}")):
        receipt = await chat_stores.send_text(sender, text)
        await chat.responder.respond(sender, receipt, last_try=False)
    session = await chat_stores.memory.load(user_id)
    assert session.history and session.working_set is not None
    assert await count_messages(chat_stores, user_id) == 2
    user = await stored_user(chat_stores, user_id)
    assert user is not None
    assert (user.home_area, user.country, user.language) == (marker, "TR", "en")

    asked = await chat_stores.send_text(user_id, "delete my data")
    await chat.responder.respond(user_id, asked, last_try=False)
    tapped = await chat_stores.send_text(
        user_id, "Yes, delete", quick_reply_payload=DELETION_CONFIRM_PAYLOAD
    )
    await chat.responder.respond(user_id, tapped, last_try=False)

    assert await stored_user(chat_stores, user_id) is None
    assert await count_messages(chat_stores, user_id) == 0
    erased = await chat_stores.memory.load(user_id)
    assert (erased.history, erased.working_set) == ((), None)
    stored = await all_stored_bytes(redis_url)
    assert str(user_id).encode() not in stored
    assert f"I live in {marker}".encode() not in stored
    # The other user keeps everything, their own words included.
    assert f"hi {marker}".encode() in stored
    assert await count_messages(chat_stores, staying_id) == 2


async def test_a_passing_failure_is_retried_and_sends_nothing_yet(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [MODEL_DOWN, final_answer("en", "Here you go.")])
    receipt = await chat_stores.send_text(user_id, "anything saved?")

    with pytest.raises(RetryLaterError):
        await chat.responder.respond(user_id, receipt, last_try=False)

    assert chat.sent_bodies() == []
    assert (await chat_stores.memory.load(user_id)).history == ()
    assert await chat.responder.respond(user_id, receipt, last_try=True) is ChatOutcome.ANSWERED
    assert chat.sent_texts() == ["Here you go."]


async def test_the_last_try_sends_the_error_message_in_the_users_language(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("tr", "Merhaba!"), MODEL_DOWN])
    first = await chat_stores.send_text(user_id, "merhaba")
    await chat.responder.respond(user_id, first, last_try=False)
    receipt = await chat_stores.send_text(user_id, "kaydettiklerim?")

    outcome = await chat.responder.respond(user_id, receipt, last_try=True)

    assert outcome is ChatOutcome.FAILED
    assert chat.sent_texts()[-1] == reply_wording("tr").error
    assert await count_messages(chat_stores, user_id) == 2


@pytest.mark.parametrize(
    "script",
    [[REJECTED], [tool_calls((SEARCH_ITEMS_TOOL, {"category": "x"}))] * MAX_STEPS],
    ids=["rejected-request", "step-limit"],
)
async def test_a_failure_that_would_repeat_sends_the_error_message_at_once(
    chat_stores: ChatStores, script: list[Any]
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, script)
    receipt = await chat_stores.send_text(user_id, "anything saved?")

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.FAILED
    assert chat.sent_texts() == [reply_wording("en").error]


async def test_an_answer_whose_sending_started_is_never_sent_again(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Once.")])
    receipt = await chat_stores.send_text(user_id, "hi")
    # An earlier try started sending, then its worker stopped.
    await chat_stores.memory.start_reply(user_id, receipt, None)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ALREADY_SENT
    assert chat.sent_bodies() == []
    assert chat.agent.generator.requests == []


async def test_a_failed_send_is_not_retried_or_remembered(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    refused = json_response("synthetic_error_invalid_parameter.json", 400)
    chat = build_chat(chat_stores, [final_answer("en", "Hi.")], send_answers=[refused])
    receipt = await chat_stores.send_text(user_id, "hi")

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.SEND_FAILED
    reply = await chat_stores.memory.reply(user_id, receipt)
    assert reply is not None
    assert reply.state is ReplyState.SENDING
    assert (await chat_stores.memory.load(user_id)).history == ()
    assert await count_messages(chat_stores, user_id) == 0


@contextmanager
def conversation_inserts_fail(
    sessions: async_sessionmaker[AsyncSession], times: int | None
) -> Iterator[None]:
    """Make the database refuse `conversation_msg` rows, `times` times or until the block ends."""
    bind = sessions.kw["bind"]
    target = bind.sync_connection if isinstance(bind, AsyncConnection) else bind.sync_engine
    failures = itertools.count() if times is None else iter(range(times))

    def refuse(
        conn: object, cursor: object, statement: str, *args: object, **kwargs: object
    ) -> None:
        if (
            statement.startswith("INSERT INTO conversation_msg")
            and next(failures, None) is not None
        ):
            raise OperationalError(statement, None, ConnectionError("synthetic outage"))

    event.listen(target, "before_cursor_execute", refuse)
    try:
        yield
    finally:
        event.remove(target, "before_cursor_execute", refuse)


async def assert_stored_once(
    chat_stores: ChatStores, chat: ChatUnderTest, user_id: uuid.UUID, answer_text: str
) -> None:
    """One delivered answer, and the whole exchange in Postgres and in short-term memory."""
    assert chat.sent_texts() == [answer_text]
    exchange = [("user", "hi"), ("assistant", answer_text)]
    assert await stored_messages(chat_stores, user_id) == exchange
    session = await chat_stores.memory.load(user_id)
    assert [(message.role, message.content) for message in session.history] == exchange
    user = await stored_user(chat_stores, user_id)
    assert user is not None
    assert user.language == "en"


async def test_a_database_failure_after_sending_is_stored_again_without_sending_again(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Stored after all.")])
    receipt = await chat_stores.send_text(user_id, "hi")

    with conversation_inserts_fail(chat_stores.sessions, times=1):
        outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    await assert_stored_once(chat_stores, chat, user_id, "Stored after all.")


async def test_a_redis_failure_after_sending_is_stored_again_without_sending_again(
    chat_stores: ChatStores, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Stored after all.")])
    receipt = await chat_stores.send_text(user_id, "hi")
    remember = chat_stores.memory.remember
    failures = [RedisError("synthetic outage")]

    async def remember_after_an_outage(*args: object, **kwargs: object) -> None:
        if failures:
            raise failures.pop()
        await remember(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(chat_stores.memory, "remember", remember_after_an_outage)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    await assert_stored_once(chat_stores, chat, user_id, "Stored after all.")


async def test_a_later_try_stores_a_delivered_answer_instead_of_stopping_as_already_sent(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Stored on the next try.")])
    receipt = await chat_stores.send_text(user_id, "hi")

    with (
        conversation_inserts_fail(chat_stores.sessions, times=None),
        pytest.raises(RetryLaterError),
    ):
        await chat.responder.respond(user_id, receipt, last_try=False)
    outcome = await chat.responder.respond(user_id, receipt, last_try=True)

    assert outcome is ChatOutcome.ANSWERED
    await assert_stored_once(chat_stores, chat, user_id, "Stored on the next try.")
    assert chat.agent.generator.requests[1:] == []


async def test_a_redis_outage_right_after_sending_is_stored_by_the_next_try(
    chat_stores: ChatStores, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Stored on the next try.")])
    receipt = await chat_stores.send_text(user_id, "hi")
    # Postgres works throughout; Redis fails from the moment the answer was sent.
    outage = failing(monkeypatch, chat_stores.memory, "reply_delivered", times=None)
    remember_outage = failing(monkeypatch, chat_stores.memory, "remember", times=None)

    with pytest.raises(RetryLaterError):
        await chat.responder.respond(user_id, receipt, last_try=False)
    outage.clear()
    remember_outage.clear()
    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    await assert_stored_once(chat_stores, chat, user_id, "Stored on the next try.")
    assert len(chat.agent.generator.requests) == 1


async def test_an_answer_recorded_only_in_postgres_is_not_brought_back_after_a_deletion(
    chat_stores: ChatStores, monkeypatch: pytest.MonkeyPatch, redis_url: str
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Delivered.")])
    marker = f"synthetic-{uuid.uuid4().hex}"
    receipt = await chat_stores.send_text(user_id, marker)
    outage = failing(monkeypatch, chat_stores.memory, "reply_delivered", times=None)

    with pytest.raises(RetryLaterError):
        await chat.responder.respond(user_id, receipt, last_try=False)
    assert await count_messages(chat_stores, user_id) == 2
    eraser = UserDataEraser(chat_stores.sessions, chat_stores.outside_postgres())
    assert await eraser.erase(user_id) == 0
    outage.clear()
    outcome = await chat.responder.respond(user_id, receipt, last_try=True)

    assert outcome is ChatOutcome.NO_MESSAGE
    assert await count_messages(chat_stores, user_id) == 0
    stored = await all_stored_bytes(redis_url)
    assert str(user_id).encode() not in stored
    assert marker.encode() not in stored


async def test_a_last_try_that_cannot_store_gives_up_without_an_error_message(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Delivered.")])
    receipt = await chat_stores.send_text(user_id, "hi")

    with conversation_inserts_fail(chat_stores.sessions, times=None):
        outcome = await chat.responder.respond(user_id, receipt, last_try=True)

    assert outcome is ChatOutcome.NOT_STORED
    assert chat.sent_texts() == ["Delivered."]
    assert await count_messages(chat_stores, user_id) == 0


async def test_an_answer_stored_late_is_not_brought_back_after_a_deletion(
    chat_stores: ChatStores, redis_url: str
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Delivered.")])
    marker = f"synthetic-{uuid.uuid4().hex}"
    receipt = await chat_stores.send_text(user_id, marker)

    with (
        conversation_inserts_fail(chat_stores.sessions, times=None),
        pytest.raises(RetryLaterError),
    ):
        await chat.responder.respond(user_id, receipt, last_try=False)
    eraser = UserDataEraser(chat_stores.sessions, chat_stores.outside_postgres())
    assert await eraser.erase(user_id) == 0
    outcome = await chat.responder.respond(user_id, receipt, last_try=True)

    assert outcome is ChatOutcome.NO_MESSAGE
    assert chat.sent_texts() == ["Delivered."]
    assert await count_messages(chat_stores, user_id) == 0
    stored = await all_stored_bytes(redis_url)
    assert str(user_id).encode() not in stored
    assert marker.encode() not in stored


async def test_a_message_whose_payload_is_gone_is_not_answered(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Hi.")])
    receipt = await chat_stores.send_text(user_id, "hi")
    await chat_stores.state.erase_user(user_id)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.NO_MESSAGE
    assert chat.sent_bodies() == []


async def test_an_answer_cut_off_by_the_closed_window_keeps_only_the_users_text(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = build_chat(chat_stores, [final_answer("en", "Too late.")])
    # Sent long before the mock Send API's clock: Instagram's window has closed.
    chat_stores.texts_sent = -int(timedelta(days=2).total_seconds() // 60)
    receipt = await chat_stores.send_text(user_id, "hi")

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.OUTSIDE_REPLY_WINDOW
    assert chat.sent_bodies() == []
    assert await stored_messages(chat_stores, user_id) == [("user", "hi")]


@dataclass
class StubResponder:
    """Stands in for `ChatResponder` and records how it was called."""

    retry: bool = False
    calls: list[tuple[uuid.UUID, uuid.UUID, bool]] = field(default_factory=list)

    async def respond(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, *, last_try: bool
    ) -> ChatOutcome:
        self.calls.append((user_id, receipt_id, last_try))
        if self.retry:
            raise RetryLaterError("TimeoutError")
        return ChatOutcome.ANSWERED


def context(responder: StubResponder, job_try: int) -> dict[str, Any]:
    resources = ChatWorkerResources(
        responder=responder,  # type: ignore[arg-type]
        max_tries=MAX_TRIES,
        retry_delay_seconds=RETRY_DELAY_SECONDS,
        close=no_close,
    )
    return {"chat_resources": resources, "job_try": job_try}


@pytest.mark.parametrize(("job_try", "last_try"), [(1, False), (MAX_TRIES, True)])
async def test_the_job_takes_ids_as_text_and_knows_its_last_try(
    job_try: int, last_try: bool
) -> None:
    responder = StubResponder()
    user_id, receipt_id = uuid.uuid4(), uuid.uuid4()

    result = await answer_chat_message(
        context(responder, job_try), user_id=str(user_id), message_receipt_id=str(receipt_id)
    )

    assert result == "answered"
    assert responder.calls == [(user_id, receipt_id, last_try)]


@pytest.mark.parametrize("job_try", [1, 2])
async def test_a_retry_waits_longer_with_each_try(job_try: int) -> None:
    with pytest.raises(Retry) as raised:
        await answer_chat_message(
            context(StubResponder(retry=True), job_try),
            user_id=str(uuid.uuid4()),
            message_receipt_id=str(uuid.uuid4()),
        )

    assert raised.value.defer_score == RETRY_DELAY_SECONDS * job_try * 1000


def test_the_worker_reads_only_the_chat_queue(settings: Settings) -> None:
    worker = build_worker(settings, chat_settings(chat_concurrency=7))

    assert worker.queue_name == CHAT_QUEUE
    assert list(worker.functions) == [ANSWER_CHAT_MESSAGE_JOB]
    assert worker.max_jobs == 7
