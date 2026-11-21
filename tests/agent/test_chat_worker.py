"""The chat worker: a DM in, the agent's answer out, the exchange remembered.

Also the "Yes, delete" routing (record 0015, Q9), the erasure of everything
the worker stores, retries and the rule that an answer is never sent twice.
The text model is scripted and Meta's Send API is a recording mock. These tests
need Postgres (`TEST_DATABASE_ADMIN_URL`) and Redis (`TEST_REDIS_URL`). Every
id and text is synthetic.
"""

import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pytest
from arq import Retry
from arq.connections import ArqRedis
from arq.jobs import Job

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
from app.ingestion.jobs import ANSWER_CHAT_MESSAGE_JOB, CHAT_QUEUE, chat_job_id
from app.models.errors import ModelRequestError, ModelUnavailableError
from tests.agent.agent_support import CONFIRM_SECONDS, MAX_STEPS, final_answer, tool_calls
from tests.agent.chat_support import (
    ChatStores,
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


@pytest.mark.parametrize("when", ["typed-yes", "later-tap"])
async def test_a_typed_yes_or_a_later_tap_deletes_nothing(
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
    if when == "later-tap":
        other = await chat_stores.send_text(user_id, "hello")
        await chat.responder.respond(user_id, other, last_try=False)
        last = await chat_stores.send_text(
            user_id, "Yes, delete", quick_reply_payload=DELETION_CONFIRM_PAYLOAD
        )
    else:
        last = await chat_stores.send_text(user_id, "yes")

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
    await chat_stores.memory.start_reply(user_id, receipt)

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
    assert await chat_stores.memory.reply_started(user_id, receipt)
    assert (await chat_stores.memory.load(user_id)).history == ()
    assert await count_messages(chat_stores, user_id) == 0


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
