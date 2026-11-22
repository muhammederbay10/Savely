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

**A reply goes through three states** (`ReplyState`), kept in `replies` under
the receipt's id:

- `sending`, set right before the answer is sent, with the exchange the answer
  will add once sent (`<receipt>:planned`) and its delivery order
  (`<receipt>:order`, from the hash's own `delivery_order` counter). A try that
  finds it never sends again, because some of the answer may have arrived.
- `delivered`, once it was sent, with whether all of it was
  (`<receipt>:sent_whole`); the exchange is still to be stored.
- `stored`, once the exchange is in; the planned exchange is dropped.

`remember` stores the exchange and marks the reply stored in one Redis script,
which does nothing for a reply already stored, so storing it again after a
failure never doubles it. An exchange stored late, after newer ones, still
lands in its place: its messages are merged into the history by time (then
trimmed to the last `SESSION_HISTORY_MESSAGES`), and its working set replaces
the stored one only if its answer was sent after the answer that set it
(`working_set_order` in `replies`), so the user's last visible list stays the
one "the second one" refers to. The chat
worker writes here only while it holds the user's row
(`app.memory.conversation.ConversationStore.hold`), so nothing is written for
a user who was deleted; `reply_delivered` instead changes only a reply that is
still `sending`, in one step, so it never brings an erased reply back.

"Delete my data" erases the three keys (`erase_user`, one of
`UserDataEraser`'s stores).
"""

import json
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final, Literal, cast

from redis.asyncio import Redis

from app.agent.deletion_requests import PendingDeletion, pending_from_value, pending_to_value
from app.agent.tools import SearchUsed

_KEY_PREFIX: Final = "savely:memory"
_HISTORY: Final = "history"
_WORKING_SET: Final = "working_set"
_REPLIES: Final = "replies"
_ALL_KEY_NAMES: Final = (_HISTORY, _WORKING_SET, _REPLIES)
_PLANNED_SUFFIX: Final = ":planned"
_SENT_WHOLE_SUFFIX: Final = ":sent_whole"
_ORDER_SUFFIX: Final = ":order"
_DELIVERY_ORDER: Final = "delivery_order"
_WORKING_SET_ORDER: Final = "working_set_order"
_TRUE: Final = "1"
_FALSE: Final = "0"
_MICROSECOND: Final = timedelta(microseconds=1)
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
# The fixed arguments of `_REMEMBER` before the messages; Lua counts from 1.
_REMEMBER_FIXED_ARGUMENTS: Final = 8
_MILLISECONDS_PER_SECOND: Final = 1000

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
    """What answering one message adds to the conversation, once it is sent.

    Attributes:
        messages: The user's text, then the bot's reply, oldest first.
        working_set: The answer's results; `None` keeps the last ones.
        language: The BCP 47 tag of the language the user wrote in; `None`
            leaves the stored one.
        deletion_request: When the answer is a deletion question, the request
            it makes confirmable, planned before sending
            (`app.agent.send.deletion_request`).
    """

    messages: tuple[SessionMessage, ...]
    working_set: WorkingSet | None
    language: str | None
    deletion_request: PendingDeletion | None = field(default=None, repr=False)

    def cut_off(self) -> "Exchange":
        """Return what is kept when the closed reply window cut the answer off: the user's text.

        Returns:
            The exchange without the bot's reply, its results or its question.
        """
        return Exchange(messages=self.messages[:1], working_set=None, language=self.language)


class ReplyState(StrEnum):
    """How far the answer to one message got."""

    SENDING = "sending"
    DELIVERED = "delivered"
    STORED = "stored"


# Sets the `sending` mark and keeps the planned exchange with it, only for the first try.
_START_REPLY: Final = f"""
if redis.call("HSETNX", KEYS[1], ARGV[1], "{ReplyState.SENDING.value}") == 0 then
    return 0
end
local order = redis.call("HINCRBY", KEYS[1], "{_DELIVERY_ORDER}", 1)
redis.call("HSET", KEYS[1], ARGV[1] .. "{_ORDER_SUFFIX}", order)
if ARGV[3] ~= "" then
    redis.call("HSET", KEYS[1], ARGV[2], ARGV[3])
end
redis.call("PEXPIRE", KEYS[1], ARGV[4])
return order
"""
# Stores an exchange once: its messages merged into the history by time, its
# working set only if no later answer set one, and the `stored` mark.
# KEYS: history, working set, replies. ARGV: the reply's field, its planned,
# sent-whole and order fields, the working set (or ""), the delivery order, the
# history length, the lifetime in ms, then the messages.
_REMEMBER: Final = f"""
if redis.call("HGET", KEYS[3], ARGV[1]) == "{ReplyState.STORED.value}" then
    return 0
end
local entries = redis.call("LRANGE", KEYS[1], 0, -1)
for i = {_REMEMBER_FIXED_ARGUMENTS} + 1, #ARGV do
    table.insert(entries, ARGV[i])
end
local keyed = {{}}
for position, entry in ipairs(entries) do
    keyed[position] = {{at = cjson.decode(entry).at_us or 0, position = position, entry = entry}}
end
table.sort(keyed, function(a, b)
    if a.at ~= b.at then
        return a.at < b.at
    end
    return a.position < b.position
end)
redis.call("DEL", KEYS[1])
for i = math.max(1, #keyed - tonumber(ARGV[7]) + 1), #keyed do
    redis.call("RPUSH", KEYS[1], keyed[i].entry)
end
local order = tonumber(ARGV[6])
if ARGV[5] ~= "" then
    local kept = tonumber(redis.call("HGET", KEYS[3], "{_WORKING_SET_ORDER}") or "0")
    if order > kept then
        redis.call("SET", KEYS[2], ARGV[5])
        redis.call("HSET", KEYS[3], "{_WORKING_SET_ORDER}", order)
    end
end
redis.call("HSET", KEYS[3], ARGV[1], "{ReplyState.STORED.value}")
redis.call("HDEL", KEYS[3], ARGV[2], ARGV[3], ARGV[4])
for _, key in ipairs(KEYS) do
    redis.call("PEXPIRE", key, ARGV[8])
end
return 1
"""
# Moves a reply that is still `sending` to `delivered`; anything else stays as it is.
_MARK_DELIVERED: Final = f"""
if redis.call("HGET", KEYS[1], ARGV[1]) == "{ReplyState.SENDING.value}" then
    redis.call("HSET", KEYS[1], ARGV[1], "{ReplyState.DELIVERED.value}", ARGV[2], ARGV[3])
end
return 0
"""


@dataclass(frozen=True)
class Reply:
    """The answer to one message, as far as it got.

    Attributes:
        state: Being sent (it may have arrived), sent, or sent and stored.
        planned: Until it is stored, the exchange it adds once sent; `None`
            for the bot's error message, which adds nothing.
        sent_whole: Once delivered, whether all of it was sent; `None` before.
        order: Until it is stored, when it started to be sent among the user's
            answers (`start_reply`); a later one has a higher number.
    """

    state: ReplyState
    planned: Exchange | None = None
    sent_whole: bool | None = None
    order: int | None = None


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

    async def remember(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, exchange: Exchange, order: int
    ) -> None:
        """Store an exchange once, in its place, and mark its reply stored.

        One Redis script that also restarts the lifetime: either all of it is
        written or nothing is, and a reply already stored is left alone. The
        messages are merged into the history by time and the oldest dropped
        beyond `max_messages`; the working set replaces the stored one only if
        this answer started to be sent after the one that set it.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message the exchange answers.
            exchange: The messages, the working set and the language.
            order: When the answer started to be sent (`start_reply`).

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        field_name = str(receipt_id)
        working_set = (
            "" if exchange.working_set is None else _working_set_to_json(exchange.working_set)
        )
        await cast(
            Awaitable[int],
            self._redis.eval(
                _REMEMBER,
                len(_ALL_KEY_NAMES),
                *(_key(user_id, name) for name in _ALL_KEY_NAMES),
                field_name,
                field_name + _PLANNED_SUFFIX,
                field_name + _SENT_WHOLE_SUFFIX,
                field_name + _ORDER_SUFFIX,
                working_set,
                str(order),
                str(self._max_messages),
                str(_milliseconds(self._lifetime)),
                *(_message_to_json(message) for message in exchange.messages),
            ),
        )

    async def start_reply(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, planned: Exchange | None
    ) -> int | None:
        """Mark that the answer to a message starts to be sent; only the first call wins.

        A chat job calls it right before sending. If a worker stopped after
        that, a later try of the same job finds the mark and sends nothing,
        because some of the answer may already have reached the user.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message being answered.
            planned: The exchange the answer adds once sent, kept with the mark
                so a later try can store it; `None` for the error message.

        Returns:
            The answer's delivery order among the user's answers, if this call
            set the mark; `None` if an earlier try did.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        field_name = str(receipt_id)
        planned_json = "" if planned is None else _exchange_to_json(planned)
        started = await cast(
            Awaitable[int],
            self._redis.eval(
                _START_REPLY,
                1,
                _key(user_id, _REPLIES),
                field_name,
                field_name + _PLANNED_SUFFIX,
                planned_json,
                str(_milliseconds(self._lifetime)),
            ),
        )
        return started or None

    async def reply_delivered(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, *, sent_whole: bool
    ) -> None:
        """Record that the answer to a message was sent, if its reply is still `sending`.

        A reply that is gone (erased with its user, or expired) or further on
        is left as it is, in the same step, so this needs no hold on the user.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message.
            sent_whole: Whether all of the answer was sent.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        field_name = str(receipt_id)
        await cast(
            Awaitable[int],
            self._redis.eval(
                _MARK_DELIVERED,
                1,
                _key(user_id, _REPLIES),
                field_name,
                field_name + _SENT_WHOLE_SUFFIX,
                _TRUE if sent_whole else _FALSE,
            ),
        )

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
        field_name = str(receipt_id)
        state, planned, sent_whole, order = await cast(
            Awaitable[list[bytes | None]],
            self._redis.hmget(
                _key(user_id, _REPLIES),
                [
                    field_name,
                    field_name + _PLANNED_SUFFIX,
                    field_name + _SENT_WHOLE_SUFFIX,
                    field_name + _ORDER_SUFFIX,
                ],
            ),
        )
        if state is None:
            return None
        return Reply(
            state=ReplyState(state.decode()),
            planned=None if planned is None else _exchange_from_json(planned),
            sent_whole=None if sent_whole is None else sent_whole.decode() == _TRUE,
            order=None if order is None else int(order),
        )

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


def _message_to_value(message: SessionMessage) -> dict[str, object]:
    # `at_us` lets the history be ordered by time inside Redis.
    return {
        "role": message.role,
        "content": message.content,
        "at": message.at.isoformat(),
        "at_us": (message.at - _EPOCH) // _MICROSECOND,
    }


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


def _exchange_to_json(exchange: Exchange) -> str:
    return json.dumps(
        {
            "messages": [_message_to_value(message) for message in exchange.messages],
            "working_set": None
            if exchange.working_set is None
            else _working_set_to_value(exchange.working_set),
            "language": exchange.language,
            "deletion_request": None
            if exchange.deletion_request is None
            else pending_to_value(exchange.deletion_request),
        },
        ensure_ascii=False,
    )


def _exchange_from_json(stored: bytes) -> Exchange:
    value = json.loads(stored)
    return Exchange(
        messages=tuple(_message_from_value(message) for message in value["messages"]),
        working_set=None
        if value["working_set"] is None
        else _working_set_from_value(value["working_set"]),
        language=value["language"],
        deletion_request=None
        if value["deletion_request"] is None
        else pending_from_value(value["deletion_request"]),
    )


def _milliseconds(duration: timedelta) -> int:
    return int(duration.total_seconds() * _MILLISECONDS_PER_SECOND)


def _working_set_from_value(value: dict[str, Any]) -> WorkingSet:
    return WorkingSet(
        item_ids=tuple(uuid.UUID(item_id) for item_id in value["item_ids"]),
        shown=value["shown"],
        searches=tuple(
            SearchUsed(tool=search["tool"], arguments=search["arguments"])
            for search in value["searches"]
        ),
    )
