""" "Delete my data" from the DM: two steps enforced in code, and one eraser for every store.

The text model is scripted (fake), and in the adversarial tests it calls
`confirm_data_deletion` when it should not. Only a tap on the "Yes, delete"
button (`deletion_confirmed=True`, set by SAV-18's worker in production) on the
user's next message, after the question was delivered and before its deadline,
erases anything. These tests need Postgres with PostGIS and pgvector
(`TEST_DATABASE_ADMIN_URL`) and Redis (`TEST_REDIS_URL`). Every id, text and
file is synthetic.
"""

import logging
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
from arq.connections import ArqRedis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.base import CaptureEvent, ChatMessage
from app.agent.agent import AgentAnswer, AgentTurn, AgentWordingError
from app.agent.deletion_requests import PendingDeletion, RedisDeletionRequests
from app.agent.replies.answers import AnswerSender, DeliveryReport
from app.agent.replies.messenger import MessengerRequestError
from app.agent.replies.templates import (
    DELETION_CONFIRM_PAYLOAD,
    deletion_confirmation,
    deletion_done,
    reply_wording,
)
from app.agent.send import send_answer
from app.agent.tools import (
    CONFIRM_DATA_DELETION_TOOL,
    REQUEST_DATA_DELETION_TOOL,
    SEARCH_ITEMS_TOOL,
    PlatformRecipient,
)
from app.db.deletion import UserDataEraser, UserDataOutsidePostgres
from app.db.models import PlatformAccount, SavedItem, User
from app.ingestion.erasure import IngestionDataEraser
from app.ingestion.jobs import CHAT_QUEUE, INDEXING_QUEUE, ArqJobQueue
from app.ingestion.service import IngestionService
from app.ingestion.session_state import RedisIngestionState
from app.models.base import GenerationRequest, GenerationResult, Generator
from app.models.errors import ModelError, ModelUnavailableError
from app.models.fakes import FakeGenerator
from app.pipeline.media.storage import LocalMediaStore
from tests.agent.agent_support import (
    CONFIRM_SECONDS,
    AgentUnderTest,
    FakeClock,
    build_agent,
    final_answer,
    structured_reply,
    tool_calls,
    tool_results,
    translated_wording,
)
from tests.agent.support import NOW as MESSENGER_NOW
from tests.agent.support import build_messenger, json_response, sent
from tests.db_factories import INSTAGRAM
from tests.redis_inspection import all_stored_bytes

HINT_WINDOW = timedelta(seconds=30)
LEAVING_SENDER_ID = "1000000000000171"
STAYING_SENDER_ID = "1000000000000172"
CONFIRM_MINUTES = CONFIRM_SECONDS // 60
VIDEO = b"synthetic video bytes"

REQUEST = tool_calls((REQUEST_DATA_DELETION_TOOL, {}))
CONFIRM = tool_calls((CONFIRM_DATA_DELETION_TOOL, {}))
MODEL_DOWN = ModelUnavailableError("synthetic outage", operation="generate", model="fake/text")


class FailingMediaStore:
    """A media store whose erase fails, as a full disk or a lost mount would."""

    async def erase_user(self, user_id: uuid.UUID, saved_item_ids: Sequence[uuid.UUID]) -> None:
        raise OSError("synthetic media store failure")


@dataclass
class Stores:
    sessions: async_sessionmaker[AsyncSession]
    redis: ArqRedis
    state: RedisIngestionState
    jobs: ArqJobQueue
    media: LocalMediaStore
    media_root: Path
    requests: RedisDeletionRequests
    clock: FakeClock

    def outside_postgres(
        self, media: UserDataOutsidePostgres | None = None
    ) -> list[UserDataOutsidePostgres]:
        return [
            IngestionDataEraser(self.state, self.jobs),
            self.requests,
            media or self.media,
        ]

    def agent(
        self,
        script: Sequence[GenerationResult | ModelError],
        wrap_generator: Callable[[FakeGenerator], Generator] | None = None,
    ) -> AgentUnderTest:
        return build_agent(
            self.sessions,
            script,
            deletion_requests=self.requests,
            outside_postgres=self.outside_postgres(),
            clock=self.clock,
            wrap_generator=wrap_generator,
        )


@dataclass
class SavedUser:
    user_id: uuid.UUID
    sender_id: str
    item_id: uuid.UUID
    marker: str
    # Each turn answers the user's next message, numbered as ingestion would.
    messages_sent: int = 0

    def next_message(self) -> int:
        self.messages_sent += 1
        return self.messages_sent


@pytest.fixture
async def stores(
    session_factory: async_sessionmaker[AsyncSession],
    ingestion_redis: ArqRedis,
    ingestion_state: RedisIngestionState,
    tmp_path: Path,
) -> AsyncIterator[Stores]:
    stores = Stores(
        sessions=session_factory,
        redis=ingestion_redis,
        state=ingestion_state,
        jobs=ArqJobQueue(ingestion_redis),
        media=LocalMediaStore(tmp_path / "media"),
        media_root=tmp_path / "media",
        requests=RedisDeletionRequests(ingestion_redis),
        clock=FakeClock(),
    )
    yield stores


async def save_something(stores: Stores, sender_id: str) -> SavedUser:
    """A user with a shared reel, a queued question and a stored media file."""
    marker = f"synthetic-{uuid.uuid4().hex}"
    ingestion = IngestionService(stores.sessions, stores.jobs, stores.state, HINT_WINDOW)
    user_id = await ingestion.resolve_user(INSTAGRAM, sender_id)
    await ingestion.ingest_capture(
        CaptureEvent(
            user_id=user_id,
            platform=INSTAGRAM,
            source="dm",
            post_url=None,
            platform_post_id=None,
            caption=f"caption {marker}",
            media_url=f"https://lookaside.fbsbx.com/ig_messaging_cdn/?asset_id=1&sig={marker}",
            media_type="post",
            collection_hint=None,
            source_message_id=f"mid-{marker}-share",
            received_at=stores.clock.now,
        )
    )
    await ingestion.ingest_chat_message(
        ChatMessage(
            user_id=user_id,
            platform=INSTAGRAM,
            text=f"question {marker}",
            message_id=f"mid-{marker}-text",
            received_at=stores.clock.now + HINT_WINDOW * 2,
        )
    )
    async with stores.sessions() as session:
        item_id = await session.scalar(select(SavedItem.id).where(SavedItem.user_id == user_id))
    assert item_id is not None
    await stores.media.put(user_id, item_id, "video.mp4", VIDEO)
    return SavedUser(user_id=user_id, sender_id=sender_id, item_id=item_id, marker=marker)


async def answer(
    agent: AgentUnderTest, user: SavedUser, text: str, *, confirmed: bool = False
) -> AgentAnswer:
    return await agent.agent.answer(
        AgentTurn(
            user_id=user.user_id,
            country=None,
            text=text,
            deletion_confirmed=confirmed,
            message_number=user.next_message(),
        )
    )


async def deliver(
    agent: AgentUnderTest, user: SavedUser, result: AgentAnswer, *, refuse: bool = False
) -> DeliveryReport:
    """Send an answer as the chat worker will: through `send_answer`, to a mock Send API."""
    reply = json_response("synthetic_error_invalid_parameter.json", 400) if refuse else sent()
    under_test = build_messenger([reply])
    return await send_answer(
        AnswerSender(under_test.messenger),
        agent.toolbox,
        user.user_id,
        user.sender_id,
        result,
        MESSENGER_NOW,
        user.messages_sent,
    )


async def request_deletion(agent: AgentUnderTest, user: SavedUser, text: str) -> AgentAnswer:
    """Turn 1: the user asks, and the confirmation question is delivered."""
    result = await answer(agent, user, text)
    await deliver(agent, user, result)
    return result


async def user_exists(stores: Stores, user_id: uuid.UUID) -> bool:
    async with stores.sessions() as session:
        return await session.get(User, user_id) is not None


async def queued_job_users(redis: ArqRedis) -> set[str]:
    jobs = [
        *await redis.queued_jobs(queue_name=INDEXING_QUEUE),
        *await redis.queued_jobs(queue_name=CHAT_QUEUE),
    ]
    return {str(job.kwargs["user_id"]) for job in jobs}


async def clean_up(stores: Stores, *users: SavedUser) -> None:
    for user in users:
        await stores.state.erase_user(user.user_id)
        await stores.requests.erase_user(user.user_id, ())
    await stores.jobs.discard(
        [
            job.job_id
            for queue in (INDEXING_QUEUE, CHAT_QUEUE)
            for job in await stores.redis.queued_jobs(queue_name=queue)
            if str(job.kwargs["user_id"]) in {str(user.user_id) for user in users}
        ]
    )


@pytest.mark.parametrize(
    ("language", "text"),
    [("en", "delete my data"), ("tr", "verilerimi sil"), ("es", "borra mis datos")],
)
async def test_a_request_in_any_language_gets_the_confirmation_question(
    stores: Stores, language: str, text: str
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    script: list[GenerationResult | ModelError] = [REQUEST, final_answer(language, "…")]
    if language == "es":
        script.append(structured_reply(translated_wording("es")))
    agent = stores.agent(script)

    try:
        result = await answer(agent, user, text)

        if language == "es":
            wording = translated_wording("es")
            expected = wording["deletion_confirmation"].format(minutes=CONFIRM_MINUTES)
            done = wording["deletion_done"]
        else:
            expected = deletion_confirmation(CONFIRM_MINUTES, reply_wording(language)).text
            done = deletion_done(reply_wording(language)).text
        assert result.messages.intro == expected
        assert result.messages.results == ()
        assert result.deletion_confirmation is not None
        assert result.deletion_confirmation.done_text == done
        assert result.deletion_confirmation.language == language
        assert result.deleted_recipients == ()
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_an_explicit_yes_erases_every_store_and_ends_the_turn(
    stores: Stores, redis_url: str
) -> None:
    leaving = await save_something(stores, LEAVING_SENDER_ID)
    staying = await save_something(stores, STAYING_SENDER_ID)
    # After the erasure nothing else may be asked: the model is down by then.
    agent = stores.agent([REQUEST, final_answer("en", "…"), CONFIRM, MODEL_DOWN])

    try:
        await request_deletion(agent, leaving, "delete my data")
        stores.clock.advance(CONFIRM_SECONDS - 1)
        result = await answer(agent, leaving, "yes", confirmed=True)

        assert len(agent.generator.requests) == 3
        assert result.messages.intro == deletion_done(reply_wording("en")).text
        assert result.deleted_recipients == (PlatformRecipient(INSTAGRAM, LEAVING_SENDER_ID),)
        assert result.deletion_confirmation is None
        assert not await user_exists(stores, leaving.user_id)
        async with stores.sessions() as session:
            leftover_accounts = await session.scalar(
                select(func.count())
                .select_from(PlatformAccount)
                .where(PlatformAccount.platform_user_id == LEAVING_SENDER_ID)
            )
        assert leftover_accounts == 0
        stored = await all_stored_bytes(redis_url)
        assert leaving.marker.encode() not in stored
        assert str(leaving.user_id).encode() not in stored
        assert str(leaving.user_id) not in await queued_job_users(stores.redis)
        assert not (stores.media_root / str(leaving.user_id)).exists()

        assert await user_exists(stores, staying.user_id)
        assert staying.marker.encode() in stored
        assert str(staying.user_id) in await queued_job_users(stores.redis)
        assert (stores.media_root / str(staying.user_id) / str(staying.item_id)).exists()
    finally:
        await clean_up(stores, leaving, staying)


async def test_completion_needs_no_model_or_translation_after_the_erasure(
    stores: Stores, caplog: pytest.LogCaptureFixture
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent(
        [
            REQUEST,
            final_answer("es", "…"),
            structured_reply(translated_wording("es")),
            # The model also asks for another tool in the same reply; it never runs.
            tool_calls((CONFIRM_DATA_DELETION_TOOL, {}), (SEARCH_ITEMS_TOOL, {"category": "x"})),
            MODEL_DOWN,
        ]
    )

    try:
        await request_deletion(agent, user, "borra mis datos")
        caplog.clear()
        with caplog.at_level(logging.INFO):
            result = await answer(agent, user, "sí", confirmed=True)

        assert result.messages.intro == translated_wording("es")["deletion_done"]
        assert result.language == "es"
        assert result.deleted_recipients == (PlatformRecipient(INSTAGRAM, LEAVING_SENDER_ID),)
        assert len(agent.generator.requests) == 4
        tools_run = [r.tool for r in caplog.records if r.getMessage() == "agent_tool_called"]
        assert tools_run == [CONFIRM_DATA_DELETION_TOOL]
        assert not await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


@pytest.mark.parametrize("reply", ["no, keep my data", "cafés in Brooklyn?"])
async def test_a_wrong_confirm_call_after_no_or_another_message_erases_nothing(
    stores: Stores, reply: str
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    # The model calls the confirm tool although the user did not tap the button.
    agent = stores.agent(
        [
            REQUEST,
            final_answer("en", "…"),
            CONFIRM,
            final_answer("en", "…"),
            CONFIRM,
            final_answer("en", "Nothing deleted."),
        ]
    )

    try:
        await request_deletion(agent, user, "delete my data")
        asked_again = await answer(agent, user, reply)
        # That message used the request up, and the repeated question was never
        # delivered, so even a tap afterwards finds nothing to confirm.
        await answer(agent, user, "Yes, delete", confirmed=True)

        assert asked_again.deletion_confirmation is not None
        assert asked_again.deleted_recipients == ()
        (refused,) = tool_results(agent.generator.requests[5])
        assert "Nothing was deleted" in str(refused["error"])
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_a_typed_yes_shows_the_question_and_its_button_again(stores: Stores) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent(
        [REQUEST, final_answer("en", "…"), CONFIRM, final_answer("en", "…"), CONFIRM]
    )

    try:
        await request_deletion(agent, user, "delete my data")
        typed_yes = await answer(agent, user, "yes")

        expected = deletion_confirmation(CONFIRM_MINUTES, reply_wording("en"))
        assert typed_yes.messages.intro == expected.text
        assert typed_yes.messages.intro_quick_replies == expected.quick_replies
        assert typed_yes.deleted_recipients == ()
        assert await user_exists(stores, user.user_id)

        # The repeated question is delivered, and the tap on its button deletes.
        await deliver(agent, user, typed_yes)
        tapped = await answer(agent, user, "Yes, delete", confirmed=True)

        assert tapped.deleted_recipients == (PlatformRecipient(INSTAGRAM, LEAVING_SENDER_ID),)
        assert not await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_the_delivered_question_carries_the_delete_button(stores: Stores) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent([REQUEST, final_answer("tr", "…")])

    try:
        result = await answer(agent, user, "verilerimi sil")
        under_test = build_messenger([sent()])
        await send_answer(
            AnswerSender(under_test.messenger),
            agent.toolbox,
            user.user_id,
            user.sender_id,
            result,
            MESSENGER_NOW,
            user.messages_sent,
        )

        (body,) = under_test.api.bodies()
        assert body["message"]["quick_replies"] == [  # type: ignore[index]
            {
                "content_type": "text",
                "title": reply_wording("tr").delete_button,
                "payload": DELETION_CONFIRM_PAYLOAD,
            }
        ]
        assert await stores.requests.take(user.user_id, user.messages_sent + 1) is not None
    finally:
        await clean_up(stores, user)


async def test_a_question_the_model_never_finished_is_not_executable(stores: Stores) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent([REQUEST, MODEL_DOWN, CONFIRM, final_answer("en", "…")])

    try:
        with pytest.raises(ModelUnavailableError):
            await answer(agent, user, "delete my data")
        await answer(agent, user, "yes", confirmed=True)

        assert "Nothing was deleted" in str(tool_results(agent.generator.requests[3])[0])
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_a_question_that_could_not_be_translated_is_not_executable(
    stores: Stores,
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent(
        [
            REQUEST,
            final_answer("ko", "…"),
            structured_reply({"deletion_done": "incomplete wording"}),
            CONFIRM,
            final_answer("ko", "…"),
        ]
    )

    try:
        with pytest.raises(AgentWordingError):
            await answer(agent, user, "내 데이터를 삭제해 줘")
        await answer(agent, user, "네", confirmed=True)

        assert "Nothing was deleted" in str(tool_results(agent.generator.requests[4])[0])
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_a_question_that_was_not_sent_is_not_executable(stores: Stores) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent([REQUEST, final_answer("en", "…"), CONFIRM, final_answer("en", "…")])

    try:
        result = await answer(agent, user, "delete my data")
        with pytest.raises(MessengerRequestError):
            await deliver(agent, user, result, refuse=True)
        await answer(agent, user, "yes", confirmed=True)

        assert "Nothing was deleted" in str(tool_results(agent.generator.requests[3])[0])
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_an_expired_request_deletes_nothing(stores: Stores) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent(
        [REQUEST, final_answer("en", "…"), CONFIRM, final_answer("en", "It expired.")]
    )

    try:
        await request_deletion(agent, user, "delete my data")
        stores.clock.advance(CONFIRM_SECONDS)
        await answer(agent, user, "yes", confirmed=True)

        (refused,) = tool_results(agent.generator.requests[3])
        assert "Nothing was deleted" in str(refused["error"])
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


class SlowGenerator(Generator):
    """Lets the clock pass a given number of seconds while one scripted call is answered."""

    def __init__(self, inner: Generator, clock: FakeClock, delays: dict[int, float]) -> None:
        self._inner = inner
        self._clock = clock
        self._delays = delays
        self._calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self._calls += 1
        self._clock.advance(self._delays.get(self._calls, 0.0))
        return await self._inner.generate(request)


async def test_a_turn_that_starts_before_the_deadline_cannot_delete_after_it(
    stores: Stores,
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    # The confirming turn starts one second before the deadline; the model takes two.
    agent = stores.agent(
        [REQUEST, final_answer("en", "…"), CONFIRM, final_answer("en", "It expired.")],
        wrap_generator=lambda inner: SlowGenerator(inner, stores.clock, {3: 2.0}),
    )

    try:
        await request_deletion(agent, user, "delete my data")
        stores.clock.advance(CONFIRM_SECONDS - 1)
        await answer(agent, user, "yes", confirmed=True)

        (refused,) = tool_results(agent.generator.requests[3])
        assert "Nothing was deleted" in str(refused["error"])
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_confirming_without_a_request_or_in_the_same_turn_deletes_nothing(
    stores: Stores,
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    agent = stores.agent(
        [
            CONFIRM,
            final_answer("en", "Nothing to confirm."),
            tool_calls((REQUEST_DATA_DELETION_TOOL, {}), (CONFIRM_DATA_DELETION_TOOL, {})),
            final_answer("en", "…"),
        ]
    )

    try:
        await answer(agent, user, "yes", confirmed=True)
        await answer(agent, user, "delete my data, yes I'm sure", confirmed=True)

        (no_request,) = tool_results(agent.generator.requests[1])
        _, same_turn = tool_results(agent.generator.requests[3])
        assert "Nothing was deleted" in str(no_request["error"])
        assert "Nothing was deleted" in str(same_turn["error"])
        assert await user_exists(stores, user.user_id)
    finally:
        await clean_up(stores, user)


async def test_another_users_request_never_lets_this_user_delete(stores: Stores) -> None:
    asking = await save_something(stores, LEAVING_SENDER_ID)
    other = await save_something(stores, STAYING_SENDER_ID)
    agent = stores.agent(
        [
            REQUEST,
            final_answer("en", "…"),
            CONFIRM,
            final_answer("en", "Nothing to confirm."),
            tool_calls((CONFIRM_DATA_DELETION_TOOL, {"user_id": str(asking.user_id)})),
            final_answer("en", "Nothing to confirm."),
        ]
    )

    try:
        await request_deletion(agent, asking, "delete my data")
        await answer(agent, other, "yes", confirmed=True)
        await answer(agent, other, "yes, delete them", confirmed=True)

        (refused,) = tool_results(agent.generator.requests[3])
        (with_user_id,) = tool_results(agent.generator.requests[5])
        assert "Nothing was deleted" in str(refused["error"])
        assert "user_id" in str(with_user_id["error"])
        assert await user_exists(stores, asking.user_id)
        assert await user_exists(stores, other.user_id)
        # The asking user's own request is still waiting for their own next message.
        assert await stores.requests.take(asking.user_id, asking.messages_sent + 1) is not None
    finally:
        await clean_up(stores, asking, other)


async def test_a_request_is_taken_only_by_the_message_it_waits_for(stores: Stores) -> None:
    user_id = uuid.uuid4()
    pending = PendingDeletion(
        deadline=stores.clock.now + timedelta(seconds=CONFIRM_SECONDS),
        awaited_message=5,
        language="en",
        done_text="done",
    )
    await stores.requests.save(user_id, pending, stores.clock.now)

    try:
        assert await stores.requests.take(user_id, 4) is None
        assert await stores.requests.take(user_id, 6) is None
        assert await stores.requests.take(user_id, 5) == pending
        assert await stores.requests.take(user_id, 5) is None
    finally:
        await stores.requests.erase_user(user_id, ())


def pending_deletion(stores: Stores, awaited: int, *, minutes: int = 10) -> PendingDeletion:
    return PendingDeletion(
        deadline=stores.clock.now + timedelta(minutes=minutes),
        awaited_message=awaited,
        language="en",
        done_text=f"done {awaited} {minutes}",
    )


async def test_a_request_stored_again_never_comes_back_or_replaces_a_newer_one(
    stores: Stores,
) -> None:
    user_id = uuid.uuid4()
    first, newer = pending_deletion(stores, 5), pending_deletion(stores, 7)
    now = stores.clock.now

    try:
        await stores.requests.save(user_id, first, now)
        assert await stores.requests.take(user_id, 5) == first
        # Used up: storing it again, as a late retry would, changes nothing.
        await stores.requests.save(user_id, first, now)
        assert await stores.requests.take(user_id, 5) is None

        await stores.requests.save(user_id, newer, now)
        await stores.requests.save(user_id, first, now)
        await stores.requests.save(user_id, pending_deletion(stores, 7, minutes=5), now)
        assert await stores.requests.take(user_id, 7) == newer

        # A request past its deadline is not stored.
        await stores.requests.save(user_id, pending_deletion(stores, 9, minutes=-1), now)
        assert await stores.requests.take(user_id, 9) is None
    finally:
        await stores.requests.erase_user(user_id, ())


async def test_a_failing_store_leaves_postgres_unchanged_and_a_repeat_succeeds(
    stores: Stores,
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)

    try:
        failing = UserDataEraser(stores.sessions, stores.outside_postgres(FailingMediaStore()))
        with pytest.raises(OSError, match="synthetic media store failure"):
            await failing.erase(user.user_id)

        assert await user_exists(stores, user.user_id)
        async with stores.sessions() as session:
            assert await session.get(SavedItem, user.item_id) is not None

        deleted = await UserDataEraser(stores.sessions, stores.outside_postgres()).erase(
            user.user_id
        )

        assert deleted == 1
        assert not await user_exists(stores, user.user_id)
        assert not (stores.media_root / str(user.user_id)).exists()
        assert str(user.user_id) not in await queued_job_users(stores.redis)
    finally:
        await clean_up(stores, user)


async def test_erasing_an_unknown_user_changes_nothing(stores: Stores) -> None:
    eraser = UserDataEraser(stores.sessions, stores.outside_postgres())

    assert await eraser.erase(uuid.uuid4()) is None


def logged_text(records: Sequence[logging.LogRecord]) -> str:
    """Every message and every field of the records, as one text."""
    return "\n".join(
        f"{record.getMessage()} {' '.join(str(value) for value in vars(record).values())}"
        for record in records
    )


async def test_deletion_logs_hold_only_events_and_counts(
    stores: Stores, caplog: pytest.LogCaptureFixture
) -> None:
    user = await save_something(stores, LEAVING_SENDER_ID)
    request_text = f"please erase everything {user.marker}"
    confirm_text = f"yes, I am sure {user.marker}"
    agent = stores.agent([REQUEST, final_answer("en", "…"), CONFIRM])

    try:
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            await request_deletion(agent, user, request_text)
            await answer(agent, user, confirm_text, confirmed=True)

        events = [record.getMessage() for record in caplog.records]
        assert "data_deletion_requested" in events
        assert "user_data_deleted" in events
        logged = logged_text(caplog.records)
        forbidden = {
            "user id": str(user.user_id),
            "user id (hex)": user.user_id.hex,
            "item id": str(user.item_id),
            "sender id": LEAVING_SENDER_ID,
            "message marker": user.marker,
            "request text": request_text,
            "confirmation text": confirm_text,
        }
        assert [name for name, value in forbidden.items() if value in logged] == []
    finally:
        await clean_up(stores, user)
