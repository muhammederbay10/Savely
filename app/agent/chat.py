"""Answers one chat message: the work of one `answer_chat_message` job (record 0015, Q5).

For each text the user wrote to the bot:

1. Read the message from ingestion's payloads in Redis; no payload means the
   user was deleted, and the job stops.
2. Load the sender's profile (their platform id, country and home area) and
   their short-term memory (the last messages and the working set).
3. Run the response agent for that user only. The user comes from the DM's
   sender, never from the model. `AgentTurn.deletion_confirmed` is set only
   when the message is a tap on the "Yes, delete" button
   (`confirms_deletion`); the tools still check that it is the sender's next
   message after a delivered question, so a typed "yes", a later tap or
   another user's tap deletes nothing.
4. Send the answer (`app.agent.send.send_answer`) within the 24-hour window,
   counted from when the user sent this message.
5. Store the exchange and the new working set (`app/memory/session.py`), the
   messages in `conversation_msg` and the user's language
   (`app/memory/conversation.py`). Nothing is stored after a deletion.

**Retries.** Steps 1 to 3 have `CHAT_RUN_DEADLINE_SECONDS`. A failure there
that may pass later (a model outage, the deadline, Redis or the database)
raises `RetryLaterError`, and the worker tries the job again; on the last
try the bot sends its error message instead. A failure that would happen
again (a rejected request, an unusable answer) sends the error message at
once. A deletion request is used up by the first try that ran the agent, so
a "Yes, delete" tap that hits a failure deletes nothing: the user asks again.

**Never sent twice.** Right before sending, the job marks the message as
answered in short-term memory (`SessionMemory.start_reply`). A later try of
the same job that finds the mark sends nothing, because some of the answer may
already have arrived. A send that fails is logged and not tried again. The
"done" message after a deletion sets no mark, so nothing is written for the
deleted user; the erasure removed the message, so a later try stops anyway.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.adapters.base import ChatMessage
from app.agent.agent import AgentAnswer, AgentError, AgentTurn, ResponseAgent
from app.agent.replies.answers import AnswerMessages, AnswerSender
from app.agent.replies.messenger import MessengerError
from app.agent.replies.templates import (
    ReplyWording,
    ReplyWordingUnavailableError,
    confirms_deletion,
    reply_wording,
)
from app.agent.send import send_answer
from app.agent.tools import Toolbox
from app.ingestion.session_state import chat_payload_field, payloads_key
from app.memory.conversation import ChatProfile, ConversationStore
from app.memory.session import Session, SessionMemory, SessionMessage, WorkingSet
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


class ChatOutcome(StrEnum):
    """What one try of a chat job did."""

    ANSWERED = "answered"
    OUTSIDE_REPLY_WINDOW = "outside_reply_window"
    NO_MESSAGE = "no_message"
    NO_USER = "no_user"
    ALREADY_SENT = "already_sent"
    SEND_FAILED = "send_failed"
    FAILED = "failed"


class RetryLaterError(Exception):
    """Preparing the answer failed in a way that may pass; the worker tries again."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ChatPayloads:
    """Reads the chat message a job answers, from ingestion's per-user payloads hash."""

    def __init__(self, redis: Redis) -> None:
        """Create the reader.

        Args:
            redis: The Redis client ingestion writes to.
        """
        self._redis = redis

    async def message(self, user_id: uuid.UUID, receipt_id: uuid.UUID) -> ChatMessage | None:
        """Return the message of a chat job, if it is still kept.

        Args:
            user_id: The sender.
            receipt_id: The id of the text's receipt, from the job.

        Returns:
            The message; `None` when the user was deleted or it outlived its lifetime.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        value = await cast(
            Awaitable[bytes | None],
            self._redis.hget(payloads_key(user_id), chat_payload_field(receipt_id)),
        )
        return None if value is None else ChatMessage.model_validate_json(value)


@dataclass(frozen=True)
class _Prepared:
    """A message ready for the agent, with everything needed to answer it."""

    user_id: uuid.UUID
    receipt_id: uuid.UUID
    message: ChatMessage
    profile: ChatProfile
    session: Session


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
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Create the responder.

        Args:
            payloads: Reads the message of a job.
            memory: Short-term memory: history, working set, reply marks.
            conversations: The profile and `conversation_msg`.
            agent: The response agent.
            toolbox: The agent's tools, to record a delivered deletion question.
            sender: Sends an answer.
            run_deadline_seconds: `CHAT_RUN_DEADLINE_SECONDS`.
            now: The clock, for the time of the bot's reply.
        """
        self._payloads = payloads
        self._memory = memory
        self._conversations = conversations
        self._agent = agent
        self._toolbox = toolbox
        self._sender = sender
        self._run_deadline_seconds = run_deadline_seconds
        self._now = now

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
            RetryLaterError: If preparing the answer failed in a way that may
                pass, and this is not the last try.
            redis.exceptions.RedisError: If Redis failed after the answer was sent.
            sqlalchemy.exc.SQLAlchemyError: If the database failed after the
                answer was sent.
        """
        prepared: _Prepared | ChatOutcome | None = None
        try:
            async with asyncio.timeout(self._run_deadline_seconds):
                prepared = await self._prepare(user_id, receipt_id)
                if isinstance(prepared, ChatOutcome):
                    return prepared
                answer = await self._agent.answer(_turn(prepared))
        except RETRYABLE_ERRORS as error:
            if not last_try:
                logger.warning("chat_answer_retry", extra={"error": type(error).__name__})
                raise RetryLaterError(type(error).__name__) from error
            return await self._fail(prepared, error)
        except FINAL_ERRORS as error:
            return await self._fail(prepared, error)
        return await self._deliver(prepared, answer)

    async def _prepare(self, user_id: uuid.UUID, receipt_id: uuid.UUID) -> _Prepared | ChatOutcome:
        message = await self._payloads.message(user_id, receipt_id)
        if message is None:
            return ChatOutcome.NO_MESSAGE
        if await self._memory.reply_started(user_id, receipt_id):
            return ChatOutcome.ALREADY_SENT
        profile = await self._conversations.profile(user_id, message.platform)
        if profile is None:
            return ChatOutcome.NO_USER
        session = await self._memory.load(user_id)
        return _Prepared(user_id, receipt_id, message, profile, session)

    async def _deliver(self, prepared: _Prepared, answer: AgentAnswer) -> ChatOutcome:
        # After a deletion nothing may be written for the user again; a later try
        # finds no message, because the erasure removed it.
        if not answer.deleted_recipients and not await self._memory.start_reply(
            prepared.user_id, prepared.receipt_id
        ):
            return ChatOutcome.ALREADY_SENT
        try:
            report = await send_answer(
                self._sender,
                self._toolbox,
                prepared.user_id,
                prepared.profile.recipient_id,
                answer,
                prepared.message.received_at,
            )
        except MessengerError as error:
            logger.warning("chat_answer_send_failed", extra={"error": type(error).__name__})
            return ChatOutcome.SEND_FAILED
        if answer.deleted_recipients:
            # The user is gone, with their short-term memory and messages.
            return ChatOutcome.ANSWERED
        await self._remember(prepared, answer, sent_whole=report.complete)
        return ChatOutcome.ANSWERED if report.complete else ChatOutcome.OUTSIDE_REPLY_WINDOW

    async def _remember(
        self, prepared: _Prepared, answer: AgentAnswer, *, sent_whole: bool
    ) -> None:
        """Keep the exchange; an answer cut off by the closed window keeps only the user's text."""
        user_message = SessionMessage(
            role="user", content=prepared.message.text, at=prepared.message.received_at
        )
        messages = [user_message]
        working_set: WorkingSet | None = None
        if sent_whole:
            messages.append(
                SessionMessage(role="assistant", content=answer.messages.intro, at=self._now())
            )
            if answer.working_set:
                working_set = WorkingSet(
                    item_ids=answer.working_set,
                    shown=len(answer.messages.results),
                    searches=answer.searches,
                )
        await self._memory.remember(prepared.user_id, messages, working_set)
        await self._conversations.record(prepared.user_id, messages, answer.language)

    async def _fail(
        self, prepared: _Prepared | ChatOutcome | None, error: Exception
    ) -> ChatOutcome:
        """Send the bot's error message, if the message and its sender are known."""
        logger.error("chat_answer_failed", extra={"error": type(error).__name__})
        if not isinstance(prepared, _Prepared):
            return ChatOutcome.FAILED
        if not await self._memory.start_reply(prepared.user_id, prepared.receipt_id):
            return ChatOutcome.ALREADY_SENT
        wording = _error_wording(prepared.profile.language)
        try:
            await self._sender.send(
                prepared.profile.recipient_id,
                AnswerMessages(intro=wording.error, results=(), wording=None),
                prepared.message.received_at,
            )
        except MessengerError as send_error:
            logger.warning("chat_error_send_failed", extra={"error": type(send_error).__name__})
        return ChatOutcome.FAILED


def _turn(prepared: _Prepared) -> AgentTurn:
    working_set = prepared.session.working_set
    return AgentTurn(
        user_id=prepared.user_id,
        country=prepared.profile.country,
        text=prepared.message.text,
        history=tuple(
            Message(role=message.role, content=message.content)
            for message in prepared.session.history
        ),
        working_set=working_set.item_ids if working_set is not None else (),
        deletion_confirmed=confirms_deletion(prepared.message.quick_reply_payload),
        home_area=prepared.profile.home_area,
        working_set_shown=working_set.shown if working_set is not None else None,
        working_set_searches=working_set.searches if working_set is not None else (),
    )


def _error_wording(language: str | None) -> ReplyWording:
    """Return the error wording in the user's language, or the fallback language."""
    if language is not None:
        try:
            return reply_wording(language)
        except ReplyWordingUnavailableError:
            pass
    return reply_wording(FALLBACK_LANGUAGE)
