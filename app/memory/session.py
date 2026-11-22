"""Short-term memory in Redis: the last messages and the working set (ARCHITECTURE §3.7).

Follow-ups such as "the second one", "show more" or "only the cheap ones" need
what was said and shown last. The chat worker loads this before each answer and
stores the new exchange after it was sent.

Three keys per user, named `savely:memory:<user id>:<name>`:

| Name | Type | Holds |
|---|---|---|
| `history` | list | the last `SESSION_HISTORY_MESSAGES` messages, oldest first (`SessionMessage`) |
| `working_set` | string | the last result list, how many were shown, and the searches behind it |
| `replies` | hash | per receipt, how far its answer got (`Reply`), so a retry never sends twice |

Every write sets all three to expire `SESSION_TTL_SECONDS` after it, about the
24 hours of Instagram's reply window, so a session ends a day after the last
exchange. The history is appended and trimmed in one transaction, so two
answers that end at the same time both keep their messages; the working set is
replaced whole, and the later answer wins.

**A reply goes through three states** (`ReplyState`): `sending` right before
the answer is sent, `delivered` once it was sent, with the exchange still to be
stored, and `stored` once the exchange is in. `remember` appends the exchange
and marks the reply stored in one transaction, so storing it again after a
failure never doubles it. The chat worker writes here only while it holds the
user's row (`app.memory.conversation.ConversationStore.hold`), so nothing is
written for a user who was deleted.

"Delete my data" erases the three keys (`erase_user`, one of
`UserDataEraser`'s stores).
"""

import json
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Final, Literal, cast

from redis.asyncio import Redis

from app.agent.tools import SearchUsed

_KEY_PREFIX: Final = "savely:memory"
_HISTORY: Final = "history"
_WORKING_SET: Final = "working_set"
_REPLIES: Final = "replies"
_ALL_KEY_NAMES: Final = (_HISTORY, _WORKING_SET, _REPLIES)
_STATE: Final = "state"
_EXCHANGE: Final = "exchange"

SessionRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class SessionMessage:
    """One message of the conversation, as the response agent reads it back.

    Attributes:
        role: `user` for what the user wrote, `assistant` for the bot's reply.
        content: The text.
        at: When it was written.
    """

    role: SessionRole
    content: str = field(repr=False)
    at: datetime


@dataclass(frozen=True)
class WorkingSet:
    """The results of the user's last answer that listed items.

    Attributes:
        item_ids: Every result, in the order they were numbered; shown first,
            then the ones kept for "show more".
        shown: How many of them were sent to the user.
        searches: The searches the results came from, so a follow-up can
            narrow or change them.
    """

    item_ids: tuple[uuid.UUID, ...]
    shown: int
    searches: tuple[SearchUsed, ...] = field(repr=False)


@dataclass(frozen=True)
class Exchange:
    """What answering one message adds to the conversation.

    Attributes:
        messages: The user's text and, when the whole answer was sent, the
            bot's reply, oldest first.
        working_set: The answer's results; `None` keeps the last ones.
        language: The BCP 47 tag of the language the user wrote in; `None`
            leaves the stored one.
        sent_whole: Whether the whole answer was sent; the closed reply
            window can cut it off.
    """

    messages: tuple[SessionMessage, ...]
    working_set: WorkingSet | None
    language: str | None
    sent_whole: bool


class ReplyState(StrEnum):
    """How far the answer to one message got."""

    SENDING = "sending"
    DELIVERED = "delivered"
    STORED = "stored"


@dataclass(frozen=True)
class Reply:
    """The answer to one message, as far as it got.

    Attributes:
        state: Being sent (it may have arrived), sent, or sent and stored.
        exchange: While `DELIVERED`, the exchange that is still to be stored.
    """

    state: ReplyState
    exchange: Exchange | None = None


@dataclass(frozen=True)
class Session:
    """What short-term memory holds for one user.

    Attributes:
        history: The last messages, oldest first; empty when the session expired.
        working_set: The last results; `None` when there are none.
    """

    history: tuple[SessionMessage, ...]
    working_set: WorkingSet | None


class SessionMemory:
    """Keeps each user's short-term memory in Redis, for `SESSION_TTL_SECONDS`."""

    def __init__(self, redis: Redis, *, lifetime: timedelta, max_messages: int) -> None:
        """Create the store.

        Args:
            redis: The Redis client.
            lifetime: How long a session lives after its last write.
            max_messages: How many of the last messages are kept.

        Raises:
            ValueError: If the lifetime is not positive or no message would be kept.
        """
        if lifetime <= timedelta(0):
            raise ValueError("lifetime must be positive")
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self._redis = redis
        self._lifetime = lifetime
        self._max_messages = max_messages

    async def load(self, user_id: uuid.UUID) -> Session:
        """Return the user's history and working set.

        Args:
            user_id: The user.

        Returns:
            The session; empty when it expired or was never written.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        stored_history = await cast(
            Awaitable[list[bytes]], self._redis.lrange(_key(user_id, _HISTORY), 0, -1)
        )
        stored_working_set = await cast(
            Awaitable[bytes | None], self._redis.get(_key(user_id, _WORKING_SET))
        )
        return Session(
            history=tuple(_message_from_json(value) for value in stored_history),
            working_set=None
            if stored_working_set is None
            else _working_set_from_json(stored_working_set),
        )

    async def remember(self, user_id: uuid.UUID, receipt_id: uuid.UUID, exchange: Exchange) -> None:
        """Append an exchange, replace the working set if given, and mark its reply stored.

        It is one transaction that also restarts the lifetime: either all of it
        is written or nothing is. The caller checks first that the reply is
        not stored yet, so a repeat never doubles the exchange.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message the exchange answers.
            exchange: The messages, the working set and how much was sent.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        history_key = _key(user_id, _HISTORY)
        async with self._redis.pipeline(transaction=True) as pipeline:
            if exchange.messages:
                pipeline.rpush(
                    history_key, *(_message_to_json(message) for message in exchange.messages)
                )
                pipeline.ltrim(history_key, -self._max_messages, -1)
            if exchange.working_set is not None:
                pipeline.set(
                    _key(user_id, _WORKING_SET), _working_set_to_json(exchange.working_set)
                )
            pipeline.hset(
                _key(user_id, _REPLIES), str(receipt_id), _reply_to_json(Reply(ReplyState.STORED))
            )
            for name in _ALL_KEY_NAMES:
                pipeline.pexpire(_key(user_id, name), self._lifetime)
            await pipeline.execute()

    async def start_reply(self, user_id: uuid.UUID, receipt_id: uuid.UUID) -> bool:
        """Mark that the answer to a message starts to be sent; only the first call wins.

        A chat job calls it right before sending. If a worker stopped after
        that, a later try of the same job finds the mark and sends nothing,
        because some of the answer may already have reached the user.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message being answered.

        Returns:
            Whether this call set the mark; `False` if an earlier try did.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        replies_key = _key(user_id, _REPLIES)
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hsetnx(replies_key, str(receipt_id), _reply_to_json(Reply(ReplyState.SENDING)))
            pipeline.pexpire(replies_key, self._lifetime)
            newly_set, _ = await pipeline.execute()
        return bool(newly_set)

    async def reply_delivered(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, exchange: Exchange
    ) -> None:
        """Record that the answer to a message was sent, with the exchange still to store.

        A later try of the job finds it (`reply`) and stores the exchange
        without sending anything.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message.
            exchange: What the answer adds to the conversation.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        replies_key = _key(user_id, _REPLIES)
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(
                replies_key,
                str(receipt_id),
                _reply_to_json(Reply(ReplyState.DELIVERED, exchange)),
            )
            pipeline.pexpire(replies_key, self._lifetime)
            await pipeline.execute()

    async def reply(self, user_id: uuid.UUID, receipt_id: uuid.UUID) -> Reply | None:
        """Return how far the answer to a message got.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message.

        Returns:
            The reply; `None` if no try started to send it.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        stored = await cast(
            Awaitable[bytes | None], self._redis.hget(_key(user_id, _REPLIES), str(receipt_id))
        )
        return None if stored is None else _reply_from_json(stored)

    async def erase_user(self, user_id: uuid.UUID, saved_item_ids: Sequence[uuid.UUID]) -> None:
        """Erase the user's short-term memory; part of "delete my data".

        Args:
            user_id: The user.
            saved_item_ids: Not needed; the argument matches `UserDataOutsidePostgres`.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        await self._redis.delete(*(_key(user_id, name) for name in _ALL_KEY_NAMES))


def _key(user_id: uuid.UUID, name: str) -> str:
    return f"{_KEY_PREFIX}:{user_id}:{name}"


def _message_to_json(message: SessionMessage) -> str:
    return json.dumps(_message_to_value(message), ensure_ascii=False)


def _message_from_json(stored: bytes) -> SessionMessage:
    return _message_from_value(json.loads(stored))


def _message_to_value(message: SessionMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content, "at": message.at.isoformat()}


def _message_from_value(value: dict[str, Any]) -> SessionMessage:
    return SessionMessage(
        role=value["role"], content=value["content"], at=datetime.fromisoformat(value["at"])
    )


def _working_set_to_json(working_set: WorkingSet) -> str:
    return json.dumps(_working_set_to_value(working_set), ensure_ascii=False)


def _working_set_from_json(stored: bytes) -> WorkingSet:
    return _working_set_from_value(json.loads(stored))


def _working_set_to_value(working_set: WorkingSet) -> dict[str, object]:
    return {
        "item_ids": [str(item_id) for item_id in working_set.item_ids],
        "shown": working_set.shown,
        "searches": [
            {"tool": search.tool, "arguments": search.arguments} for search in working_set.searches
        ],
    }


def _reply_to_json(reply: Reply) -> str:
    exchange = reply.exchange
    return json.dumps(
        {
            _STATE: reply.state.value,
            _EXCHANGE: None
            if exchange is None
            else {
                "messages": [_message_to_value(message) for message in exchange.messages],
                "working_set": None
                if exchange.working_set is None
                else _working_set_to_value(exchange.working_set),
                "language": exchange.language,
                "sent_whole": exchange.sent_whole,
            },
        },
        ensure_ascii=False,
    )


def _reply_from_json(stored: bytes) -> Reply:
    value = json.loads(stored)
    exchange = value[_EXCHANGE]
    return Reply(
        state=ReplyState(value[_STATE]),
        exchange=None
        if exchange is None
        else Exchange(
            messages=tuple(_message_from_value(message) for message in exchange["messages"]),
            working_set=None
            if exchange["working_set"] is None
            else _working_set_from_value(exchange["working_set"]),
            language=exchange["language"],
            sent_whole=exchange["sent_whole"],
        ),
    )


def _working_set_from_value(value: dict[str, Any]) -> WorkingSet:
    return WorkingSet(
        item_ids=tuple(uuid.UUID(item_id) for item_id in value["item_ids"]),
        shown=value["shown"],
        searches=tuple(
            SearchUsed(tool=search["tool"], arguments=search["arguments"])
            for search in value["searches"]
        ),
    )
