"""Short-term memory in Redis: the last messages and the working set (ARCHITECTURE §3.7).

Follow-ups such as "the second one", "show more" or "only the cheap ones" need
what was said and shown last. The chat worker loads this before each answer and
stores the new exchange after it was sent.

Three keys per user, named `savely:memory:<user id>:<name>`:

| Name | Type | Holds |
|---|---|---|
| `history` | list | the last `SESSION_HISTORY_MESSAGES` messages, oldest first (`SessionMessage`) |
| `working_set` | string | the last result list, how many were shown, and the searches behind it |
| `replies` | hash | receipts whose answer started to be sent, so a retry never sends twice |

Every write sets all three to expire `SESSION_TTL_SECONDS` after it, about the
24 hours of Instagram's reply window, so a session ends a day after the last
exchange. The history is appended and trimmed in one transaction, so two
answers that end at the same time both keep their messages; the working set is
replaced whole, and the later answer wins.

"Delete my data" erases the three keys (`erase_user`, one of
`UserDataEraser`'s stores).
"""

import json
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final, Literal, cast

from redis.asyncio import Redis

from app.agent.tools import SearchUsed

_KEY_PREFIX: Final = "savely:memory"
_HISTORY: Final = "history"
_WORKING_SET: Final = "working_set"
_REPLIES: Final = "replies"
_ALL_KEY_NAMES: Final = (_HISTORY, _WORKING_SET, _REPLIES)
# Marks a receipt in `replies`; the field is what counts.
_STARTED: Final = "1"

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
        self,
        user_id: uuid.UUID,
        messages: Sequence[SessionMessage],
        working_set: WorkingSet | None,
    ) -> None:
        """Append an exchange, replace the working set if given, and restart the lifetime.

        Args:
            user_id: The user.
            messages: The new messages, oldest first.
            working_set: The new working set; `None` keeps the stored one.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        history_key = _key(user_id, _HISTORY)
        async with self._redis.pipeline(transaction=True) as pipeline:
            if messages:
                pipeline.rpush(history_key, *(_message_to_json(message) for message in messages))
                pipeline.ltrim(history_key, -self._max_messages, -1)
            if working_set is not None:
                pipeline.set(_key(user_id, _WORKING_SET), _working_set_to_json(working_set))
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
            pipeline.hsetnx(replies_key, str(receipt_id), _STARTED)
            pipeline.pexpire(replies_key, self._lifetime)
            newly_set, _ = await pipeline.execute()
        return bool(newly_set)

    async def reply_started(self, user_id: uuid.UUID, receipt_id: uuid.UUID) -> bool:
        """Tell whether an earlier try already started to send the answer to a message.

        Args:
            user_id: The user.
            receipt_id: The receipt of the message.

        Returns:
            Whether the mark of `start_reply` is there.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        return bool(
            await cast(
                Awaitable[bool], self._redis.hexists(_key(user_id, _REPLIES), str(receipt_id))
            )
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
    return json.dumps(
        {"role": message.role, "content": message.content, "at": message.at.isoformat()},
        ensure_ascii=False,
    )


def _message_from_json(stored: bytes) -> SessionMessage:
    value = json.loads(stored)
    return SessionMessage(
        role=value["role"], content=value["content"], at=datetime.fromisoformat(value["at"])
    )


def _working_set_to_json(working_set: WorkingSet) -> str:
    return json.dumps(
        {
            "item_ids": [str(item_id) for item_id in working_set.item_ids],
            "shown": working_set.shown,
            "searches": [
                {"tool": search.tool, "arguments": search.arguments}
                for search in working_set.searches
            ],
        },
        ensure_ascii=False,
    )


def _working_set_from_json(stored: bytes) -> WorkingSet:
    value = json.loads(stored)
    return WorkingSet(
        item_ids=tuple(uuid.UUID(item_id) for item_id in value["item_ids"]),
        shown=value["shown"],
        searches=tuple(
            SearchUsed(tool=search["tool"], arguments=search["arguments"])
            for search in value["searches"]
        ),
    )
