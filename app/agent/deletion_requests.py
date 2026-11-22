"""Pending "delete my data" requests, kept in Redis until the user confirms or they expire.

Deleting a user can't be undone, so it takes two steps (SAV-17, owner update of
20 Nov 2026), and code, not the prompt, decides when the second may run:

1. `request_data_deletion` only answers with the confirmation question. The
   request is stored here **once that question was delivered**
   (`app.agent.send.send_answer`), with its real deadline and the "done"
   message already written in the user's language. A question that was never
   generated, translated or sent leaves nothing that can be confirmed.
2. The request waits for one message: the user's next one after the question
   was sent. Ingestion numbers every message a user sends, in the order it
   takes them in (`app.ingestion.session_state`), and the request keeps the
   number of the one after the last message before the question went out.
   Only the turn of the message with that number takes it out (`take`),
   whenever its job runs, so "no", a share or any other message in between
   uses the chance up, also when a later tap's job runs first. `confirm_data_deletion` then also
   needs the caller's explicit confirmation and checks the deadline right
   before erasing.

One key per user, `savely:agent:<user id>:deletion_request`, expires at the
deadline and is erased with the user's data. It holds the deadline, the
message number, the language and the bot's own "done" wording; nothing the
user wrote.
"""

import json
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Protocol, cast

from redis.asyncio import Redis

_KEY_PREFIX: Final = "savely:agent"
_KEY_NAME: Final = "deletion_request"
_MILLISECONDS_PER_SECOND: Final = 1000
_MINIMUM_LIFETIME_MS: Final = 1
_DEADLINE: Final = "deadline"
_AWAITED_MESSAGE: Final = "awaited_message"
# Removes and returns the request only if it waits for the given message number,
# in one step, so a question delivered again meanwhile is never taken by mistake.
_TAKE_IF_AWAITED: Final = f"""
local stored = redis.call("GET", KEYS[1])
if not stored or cjson.decode(stored).{_AWAITED_MESSAGE} ~= tonumber(ARGV[1]) then
    return false
end
redis.call("DEL", KEYS[1])
return stored
"""
_LANGUAGE: Final = "language"
_DONE_TEXT: Final = "done_text"


@dataclass(frozen=True)
class PendingDeletion:
    """A delivered deletion question that the user's next message may confirm.

    Attributes:
        deadline: After this moment the request can't be confirmed any more.
        awaited_message: The number of the only message that can confirm it:
            the user's next one after the question was sent.
        language: The BCP 47 tag of the language the question was asked in.
        done_text: The final "done" message, already in that language, so
            nothing has to be generated or translated after the erasure.
    """

    deadline: datetime
    awaited_message: int
    language: str
    done_text: str = field(repr=False)


class DeletionRequests(Protocol):
    """Pending deletion requests, at most one per user; `RedisDeletionRequests` in production."""

    async def save(self, user_id: uuid.UUID, pending: PendingDeletion, now: datetime) -> None:
        """Store a delivered request, replacing an earlier one of the same user."""
        ...

    async def take(self, user_id: uuid.UUID, message_number: int) -> PendingDeletion | None:
        """Remove and return the user's request if it waits for this message, expired or not."""
        ...


class RedisDeletionRequests(DeletionRequests):
    """Stores at most one pending deletion request per user in Redis."""

    def __init__(self, redis: Redis) -> None:
        """Create the store.

        Args:
            redis: The Redis client.
        """
        self._redis = redis

    async def save(self, user_id: uuid.UUID, pending: PendingDeletion, now: datetime) -> None:
        """Store a delivered request until its deadline, replacing an earlier one.

        Args:
            user_id: The DM's sender.
            pending: The request, with its deadline.
            now: The current time, with a time zone, for the key's expiry.

        Raises:
            ValueError: If the deadline or `now` has no time zone.
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        if pending.deadline.tzinfo is None or now.tzinfo is None:
            raise ValueError("the deadline and now must carry a time zone")
        remaining_ms = int((pending.deadline - now).total_seconds() * _MILLISECONDS_PER_SECOND)
        value = json.dumps(
            {
                _DEADLINE: pending.deadline.isoformat(),
                _AWAITED_MESSAGE: pending.awaited_message,
                _LANGUAGE: pending.language,
                _DONE_TEXT: pending.done_text,
            },
            ensure_ascii=False,
        )
        await self._redis.set(_key(user_id), value, px=max(_MINIMUM_LIFETIME_MS, remaining_ms))

    async def take(self, user_id: uuid.UUID, message_number: int) -> PendingDeletion | None:
        """Remove the user's request and return it, if it waits for this message.

        A request that waits for another message stays; the caller checks the
        deadline. Checking and removing are one step in Redis.

        Args:
            user_id: The DM's sender.
            message_number: The number of the message being answered.

        Returns:
            The request, or `None` if there is none for this message.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        stored = await cast(
            Awaitable[bytes | None],
            self._redis.eval(_TAKE_IF_AWAITED, 1, _key(user_id), str(message_number)),
        )
        return None if stored is None else _pending_from_json(stored)

    async def erase_user(self, user_id: uuid.UUID, saved_item_ids: Sequence[uuid.UUID]) -> None:
        """Erase the user's pending request; part of "delete my data".

        Args:
            user_id: The user.
            saved_item_ids: Not needed; the argument matches `UserDataOutsidePostgres`.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        await self._redis.delete(_key(user_id))


def _key(user_id: uuid.UUID) -> str:
    return f"{_KEY_PREFIX}:{user_id}:{_KEY_NAME}"


def _pending_from_json(stored: bytes) -> PendingDeletion:
    value = json.loads(stored)
    return PendingDeletion(
        deadline=datetime.fromisoformat(value[_DEADLINE]),
        awaited_message=value[_AWAITED_MESSAGE],
        language=value[_LANGUAGE],
        done_text=value[_DONE_TEXT],
    )
