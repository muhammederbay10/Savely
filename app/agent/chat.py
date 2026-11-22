"""Answers one chat message: the work of one `answer_chat_message` job (record 0015, Q5).

For each text the user wrote to the bot:

1. Read the message and its number from ingestion's state in Redis; no
   message means the user was deleted, and the job stops.
2. Load the sender's profile (their platform id, country and home area) and
   their short-term memory (the last messages and the working set).
3. Run the response agent for that user only. The user comes from the DM's
   sender, never from the model. `AgentTurn.deletion_confirmed` is set only
   when the message is a tap on the "Yes, delete" button
   (`confirms_deletion`), and `AgentTurn.message_number` lets only the
   sender's next message after a delivered question confirm it. A typed
   "yes", a tap after another message (whatever order the jobs run in) or
   another user's tap deletes nothing.
4. Send the answer within the 24-hour window, counted from when the user
   sent this message; a deletion question's request is planned before
   (`app.agent.send.deletion_request`) and stored with the exchange.
5. Store the exchange: the messages and the new working set in short-term
   memory (`app/memory/session.py`), the messages in `conversation_msg` and
   the user's language (`app/memory/conversation.py`).

**One answer at a time per sender** (owner decision of 22 Nov 2026). Steps 4
and 5 run inside `ConversationStore.one_answer_at_a_time`, so the sender's
next answer is sent only once this one was sent and stored, and the working
set always holds the list the user saw last. The agent runs (step 3) still
overlap, and other users never wait.

**"Delete my data" never races an answer.** Step 4 and every write of step 5
run inside `ConversationStore.hold`, which holds the user's row the erasure
locks too, and only if the user still exists. An answer whose user was
deleted before it got there is not sent and stores nothing; an erasure that
starts while an answer is being sent or stored waits for it, then erases what
it stored. Only the "done" message after a deletion is sent without the user,
to the recipients read before the erasure.

**Retries before sending.** Steps 1 to 3 have `CHAT_RUN_DEADLINE_SECONDS`. A
failure there that may pass later (a model outage, the deadline, Redis or the
database) raises `RetryLaterError`, and the worker tries the job again; on the
last try the bot sends its error message instead. A failure that would happen
again (a rejected request, an unusable answer) sends the error message at
once. A deletion request is used up by the first try that ran the agent, so a
"Yes, delete" tap that hits a failure deletes nothing: the user asks again.

**Never sent twice, always stored.** Right before sending, the job marks the
reply as being sent (`SessionMemory.start_reply`), together with the exchange
the answer adds once sent: the messages, the working set, the language and a
deletion question's request, planned with its awaited message and deadline
(`app.agent.send.deletion_request`). A later try that finds the mark sends
nothing, because some of the answer may already have arrived; a send that
fails is logged and not tried again.

Once the answer was sent, storing the exchange is tried up to
`CHAT_MAX_TRIES` times, `CHAT_RETRY_DELAY_SECONDS` times the attempt apart.
Postgres comes first: the exchange's `conversation_msg` rows, which also prove
the answer was sent. Then Redis: the reply is marked delivered, and the
deletion request, history and working set are stored with the `stored` mark.
Each write can be repeated without doubling anything. If a store stays down,
the job's next try picks the exchange up from whichever store recorded the
delivery (a `delivered` mark, or the rows next to a `sending` mark) and only
stores it, without running the agent or sending again. With no such record it
can't tell whether the answer arrived and stores nothing; the last try gives
up and logs it.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.adapters.base import ChatMessage
from app.agent.agent import AgentAnswer, AgentError, AgentTurn, ResponseAgent
from app.agent.deletion_requests import PendingDeletion
from app.agent.replies.answers import AnswerMessages, AnswerSender
from app.agent.replies.messenger import MessengerError
from app.agent.replies.templates import (
    ReplyWording,
    ReplyWordingUnavailableError,
    confirms_deletion,
    reply_wording,
)
from app.agent.send import deletion_request
from app.agent.tools import Toolbox
from app.ingestion.session_state import RedisIngestionState
from app.memory.conversation import ChatProfile, ConversationStore
from app.memory.session import (
    Exchange,
    Reply,
    ReplyState,
    Session,
    SessionMemory,
    SessionMessage,
    WorkingSet,
)
from app.models.base import Message
from app.models.errors import ModelError, ModelRateLimitError, ModelUnavailableError
from app.pipeline.steps.confirm import FALLBACK_LANGUAGE

logger = logging.getLogger(__name__)

# Failures before sending that may pass if the job runs again later. `TimeoutError`
# is the run deadline.
RETRYABLE_ERRORS: Final = (
    TimeoutError,
    ModelRateLimitError,
    ModelUnavailableError,
    RedisError,
    SQLAlchemyError,
)
# Failures that would happen again: the bot apologizes at once.
FINAL_ERRORS: Final = (AgentError, ModelError)
# Failures of the stores after sending: storing the exchange is tried again.
STORE_ERRORS: Final = (RedisError, SQLAlchemyError)


class ChatOutcome(StrEnum):
    """What one try of a chat job did."""

    ANSWERED = "answered"
    OUTSIDE_REPLY_WINDOW = "outside_reply_window"
    NO_MESSAGE = "no_message"
    NO_USER = "no_user"
    ALREADY_SENT = "already_sent"
    SEND_FAILED = "send_failed"
    FAILED = "failed"
    NOT_STORED = "not_stored"


class RetryLaterError(Exception):
    """The job failed in a way that may pass; the worker tries it again."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ReceivedMessage:
    """A chat message with its number among the sender's messages.

    Attributes:
        message: The message.
        number: Its number, given by ingestion in the order messages arrived.
    """

    message: ChatMessage
    number: int


class ChatPayloads:
    """Reads the chat message a job answers, from ingestion's state in Redis."""

    def __init__(self, state: RedisIngestionState) -> None:
        """Create the reader.

        Args:
            state: Ingestion's per-user state.
        """
        self._state = state

    async def message(self, user_id: uuid.UUID, receipt_id: uuid.UUID) -> ReceivedMessage | None:
        """Return the message of a chat job and its number, if they are still kept.

        Args:
            user_id: The sender.
            receipt_id: The id of the text's receipt, from the job.

        Returns:
            The message; `None` when the user was deleted or it outlived its lifetime.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        message = await self._state.chat_payload(user_id, receipt_id)
        if message is None:
            return None
        receipt = await self._state.text_receipt(user_id, message.platform, message.message_id)
        return None if receipt is None else ReceivedMessage(message, receipt.number)

    async def last_message_number(self, user_id: uuid.UUID) -> int:
        """Return the number of the sender's last message, 0 if none is known.

        Args:
            user_id: The sender.

        Returns:
            The number.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        return await self._state.last_message_number(user_id)


@dataclass(frozen=True)
class _Prepared:
    """A message ready for the agent, with everything needed to answer it."""

    user_id: uuid.UUID
    receipt_id: uuid.UUID
    received: ReceivedMessage
    profile: ChatProfile
    session: Session


@dataclass(frozen=True)
class _Sent:
    """An answer that was sent, with the exchange it adds.

    Attributes:
        planned: The exchange planned before sending.
        sent_whole: Whether all of the answer was sent; if not, only the
            user's text is kept (`Exchange.cut_off`).
        order: When it started to be sent among the user's answers, so a
            late store doesn't replace a newer answer's working set.
    """

    planned: Exchange
    sent_whole: bool
    order: int

    def exchange(self) -> Exchange:
        return self.planned if self.sent_whole else self.planned.cut_off()


class ChatResponder:
    """Answers one chat message with the response agent, and remembers the exchange."""

    def __init__(
        self,
        *,
        payloads: ChatPayloads,
        memory: SessionMemory,
        conversations: ConversationStore,
        agent: ResponseAgent,
        toolbox: Toolbox,
        sender: AnswerSender,
        run_deadline_seconds: float,
        store_tries: int,
        store_retry_delay_seconds: float,
        now: Callable[[], datetime] = _utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create the responder.

        Args:
            payloads: Reads the message of a job.
            memory: Short-term memory: history, working set, reply marks.
            conversations: The profile and `conversation_msg`, and the hold on the user.
            agent: The response agent.
            toolbox: The agent's tools, to record a delivered deletion question.
            sender: Sends an answer.
            run_deadline_seconds: `CHAT_RUN_DEADLINE_SECONDS`.
            store_tries: How often one try stores a sent exchange before it
                gives up (`CHAT_MAX_TRIES`).
            store_retry_delay_seconds: The wait before storing again, times the
                attempt (`CHAT_RETRY_DELAY_SECONDS`).
            now: The clock, for the time of the bot's reply.
            sleep: Waits between attempts to store.

        Raises:
            ValueError: If `store_tries` is below 1.
        """
        if store_tries < 1:
            raise ValueError("store_tries must be at least 1")
        self._payloads = payloads
        self._memory = memory
        self._conversations = conversations
        self._agent = agent
        self._toolbox = toolbox
        self._sender = sender
        self._run_deadline_seconds = run_deadline_seconds
        self._store_tries = store_tries
        self._store_retry_delay_seconds = store_retry_delay_seconds
        self._now = now
        self._sleep = sleep

    async def respond(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, *, last_try: bool
    ) -> ChatOutcome:
        """Answer one message, unless it was answered or its user is gone.

        Args:
            user_id: The sender, from the job.
            receipt_id: The id of the text's receipt, from the job.
            last_try: Whether the worker will not try this job again.

        Returns:
            What happened.

        Raises:
            RetryLaterError: If preparing the answer, or storing it once sent,
                failed in a way that may pass, and this is not the last try.
        """
        deadline = asyncio.get_running_loop().time() + self._run_deadline_seconds
        try:
            async with asyncio.timeout_at(deadline):
                step = await self._prepare(user_id, receipt_id)
        except RETRYABLE_ERRORS as error:
            return await self._retry_or_fail(None, error, last_try=last_try)
        if isinstance(step, ChatOutcome):
            return step
        if isinstance(step, _Sent):
            # An earlier try sent the answer; only storing it is left.
            return await self._store_sent(user_id, receipt_id, step, last_try=last_try)
        try:
            async with asyncio.timeout_at(deadline):
                answer = await self._agent.answer(_turn(step))
        except RETRYABLE_ERRORS as error:
            return await self._retry_or_fail(step, error, last_try=last_try)
        except FINAL_ERRORS as error:
            return await self._fail(step, error)
        return await self._deliver(step, answer, last_try=last_try)

    async def _prepare(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID
    ) -> _Prepared | _Sent | ChatOutcome:
        received = await self._payloads.message(user_id, receipt_id)
        if received is None:
            return ChatOutcome.NO_MESSAGE
        reply = await self._memory.reply(user_id, receipt_id)
        if reply is not None:
            return await self._earlier_reply(user_id, receipt_id, reply)
        profile = await self._conversations.profile(user_id, received.message.platform)
        if profile is None:
            return ChatOutcome.NO_USER
        session = await self._memory.load(user_id)
        return _Prepared(user_id, receipt_id, received, profile, session)

    async def _earlier_reply(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, reply: Reply
    ) -> _Sent | ChatOutcome:
        """Find what is left of an answer an earlier try started to send."""
        if reply.state is ReplyState.STORED or reply.planned is None or reply.order is None:
            return ChatOutcome.ALREADY_SENT
        if reply.state is ReplyState.DELIVERED:
            return _Sent(reply.planned, sent_whole=reply.sent_whole is True, order=reply.order)
        # Still `sending`: whether it arrived is unknown, unless its rows prove it,
        # because they are written only after the answer was sent.
        planned_messages = len(reply.planned.messages)
        stored = await self._conversations.exchange_messages_stored(
            user_id, receipt_id, planned_messages
        )
        if stored == 0:
            return ChatOutcome.ALREADY_SENT
        return _Sent(reply.planned, sent_whole=stored == planned_messages, order=reply.order)

    async def _retry_or_fail(
        self, prepared: _Prepared | None, error: Exception, *, last_try: bool
    ) -> ChatOutcome:
        if not last_try:
            logger.warning("chat_answer_retry", extra={"error": type(error).__name__})
            raise RetryLaterError(type(error).__name__) from error
        return await self._fail(prepared, error)

    async def _deliver(
        self, prepared: _Prepared, answer: AgentAnswer, *, last_try: bool
    ) -> ChatOutcome:
        if answer.deleted_recipients:
            return await self._send_after_deletion(prepared, answer)
        try:
            # Sent and stored before the sender's next answer goes out.
            async with self._conversations.one_answer_at_a_time(prepared.user_id) as conversations:
                sent = await self._send(conversations, prepared, answer)
                if isinstance(sent, ChatOutcome):
                    return sent
                return await self._store(
                    conversations, prepared.user_id, prepared.receipt_id, sent, last_try=last_try
                )
        except RETRYABLE_ERRORS as error:
            # A later try answers, or finds the reply mark and sends nothing.
            return await self._retry_or_fail(prepared, error, last_try=last_try)

    async def _send(
        self, conversations: ConversationStore, prepared: _Prepared, answer: AgentAnswer
    ) -> _Sent | ChatOutcome:
        """Send the answer while holding the user, unless they were deleted or it was sent."""
        async with conversations.hold(prepared.user_id) as held:
            if held is None:
                return ChatOutcome.NO_USER
            # Read before sending: a message that arrives while the question is
            # on its way is the next one, and only a tap there can confirm it.
            last_message_number = await self._payloads.last_message_number(prepared.user_id)
            planned = self._exchange(
                prepared, answer, deletion_request(self._toolbox, answer, last_message_number)
            )
            order = await self._memory.start_reply(prepared.user_id, prepared.receipt_id, planned)
            if order is None:
                return ChatOutcome.ALREADY_SENT
            try:
                report = await self._sender.send(
                    prepared.profile.recipient_id,
                    answer.messages,
                    prepared.received.message.received_at,
                )
            except MessengerError as error:
                logger.warning("chat_answer_send_failed", extra={"error": type(error).__name__})
                return ChatOutcome.SEND_FAILED
        return _Sent(planned, sent_whole=report.complete, order=order)

    async def _send_after_deletion(self, prepared: _Prepared, answer: AgentAnswer) -> ChatOutcome:
        """Send the "done" message; the user is gone, so nothing is held, marked or stored."""
        try:
            await self._sender.send(
                prepared.profile.recipient_id,
                answer.messages,
                prepared.received.message.received_at,
            )
        except MessengerError as error:
            logger.warning("chat_answer_send_failed", extra={"error": type(error).__name__})
            return ChatOutcome.SEND_FAILED
        return ChatOutcome.ANSWERED

    def _exchange(
        self, prepared: _Prepared, answer: AgentAnswer, request: PendingDeletion | None
    ) -> Exchange:
        """The exchange the answer adds once all of it was sent."""
        message = prepared.received.message
        working_set = None
        if answer.working_set:
            working_set = WorkingSet(
                item_ids=answer.working_set,
                shown=len(answer.messages.results),
                searches=answer.searches,
            )
        return Exchange(
            messages=(
                SessionMessage(role="user", content=message.text, at=message.received_at),
                SessionMessage(role="assistant", content=answer.messages.intro, at=self._now()),
            ),
            working_set=working_set,
            language=answer.language,
            deletion_request=request,
        )

    async def _store_sent(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, sent: _Sent, *, last_try: bool
    ) -> ChatOutcome:
        """Store an exchange an earlier try sent, one answer of the sender at a time."""
        try:
            async with self._conversations.one_answer_at_a_time(user_id) as conversations:
                return await self._store(
                    conversations, user_id, receipt_id, sent, last_try=last_try
                )
        except STORE_ERRORS as error:
            logger.warning("chat_exchange_store_failed", extra={"error": type(error).__name__})
            return _give_up_storing(last_try=last_try)

    async def _store(
        self,
        conversations: ConversationStore,
        user_id: uuid.UUID,
        receipt_id: uuid.UUID,
        sent: _Sent,
        *,
        last_try: bool,
    ) -> ChatOutcome:
        """Store a sent exchange, trying again after a failure; never send anything."""
        for attempt in range(1, self._store_tries + 1):
            try:
                await self._store_once(conversations, user_id, receipt_id, sent)
            except STORE_ERRORS as error:
                logger.warning(
                    "chat_exchange_store_failed",
                    extra={"error": type(error).__name__, "attempt": attempt},
                )
                if attempt < self._store_tries:
                    await self._sleep(self._store_retry_delay_seconds * attempt)
            else:
                return ChatOutcome.ANSWERED if sent.sent_whole else ChatOutcome.OUTSIDE_REPLY_WINDOW
        return _give_up_storing(last_try=last_try)

    async def _store_once(
        self,
        conversations: ConversationStore,
        user_id: uuid.UUID,
        receipt_id: uuid.UUID,
        sent: _Sent,
    ) -> None:
        """Record the delivery in both stores, then store the rest in short-term memory.

        The rows and the `delivered` mark are each tried, so either store alone
        lets a later try find the sent exchange. Redis is written last, with
        the `stored` mark, and each step for a deleted user writes nothing.
        """
        exchange = sent.exchange()
        rows_error: SQLAlchemyError | None = None
        try:
            async with conversations.hold(user_id) as held:
                if held is None:
                    return
                await held.record(receipt_id, exchange.messages, exchange.language)
        except SQLAlchemyError as error:
            rows_error = error
        await self._memory.reply_delivered(user_id, receipt_id, sent_whole=sent.sent_whole)
        if rows_error is not None:
            raise rows_error
        async with conversations.hold(user_id) as held:
            if held is None:
                return
            reply = await self._memory.reply(user_id, receipt_id)
            if reply is None or reply.state is ReplyState.STORED:
                return
            if exchange.deletion_request is not None:
                await self._toolbox.keep_deletion_request(user_id, exchange.deletion_request)
            await self._memory.remember(user_id, receipt_id, exchange, sent.order)

    async def _fail(self, prepared: _Prepared | None, error: Exception) -> ChatOutcome:
        """Send the bot's error message, if the message and its sender are known."""
        logger.error("chat_answer_failed", extra={"error": type(error).__name__})
        if prepared is None:
            return ChatOutcome.FAILED
        async with (
            self._conversations.one_answer_at_a_time(prepared.user_id) as conversations,
            conversations.hold(prepared.user_id) as held,
        ):
            if held is None:
                return ChatOutcome.NO_USER
            if await self._memory.start_reply(prepared.user_id, prepared.receipt_id, None) is None:
                return ChatOutcome.ALREADY_SENT
            wording = _error_wording(prepared.profile.language)
            try:
                await self._sender.send(
                    prepared.profile.recipient_id,
                    AnswerMessages(intro=wording.error, results=(), wording=None),
                    prepared.received.message.received_at,
                )
            except MessengerError as send_error:
                logger.warning("chat_error_send_failed", extra={"error": type(send_error).__name__})
        return ChatOutcome.FAILED


def _give_up_storing(*, last_try: bool) -> ChatOutcome:
    """Leave a sent exchange to the job's next try, or, on the last one, unstored."""
    if not last_try:
        raise RetryLaterError("the sent exchange is not stored yet")
    logger.error("chat_exchange_not_stored")
    return ChatOutcome.NOT_STORED


def _turn(prepared: _Prepared) -> AgentTurn:
    working_set = prepared.session.working_set
    message = prepared.received.message
    return AgentTurn(
        user_id=prepared.user_id,
        country=prepared.profile.country,
        text=message.text,
        history=tuple(
            Message(role=message.role, content=message.content)
            for message in prepared.session.history
        ),
        working_set=working_set.item_ids if working_set is not None else (),
        deletion_confirmed=confirms_deletion(message.quick_reply_payload),
        home_area=prepared.profile.home_area,
        working_set_shown=working_set.shown if working_set is not None else None,
        working_set_searches=working_set.searches if working_set is not None else (),
        message_number=prepared.received.number,
    )


def _error_wording(language: str | None) -> ReplyWording:
    """Return the error wording in the user's language, or the fallback language."""
    if language is not None:
        try:
            return reply_wording(language)
        except ReplyWordingUnavailableError:
            pass
    return reply_wording(FALLBACK_LANGUAGE)
