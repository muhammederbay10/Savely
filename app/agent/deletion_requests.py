"""Pending "delete my data" requests, kept in Redis until the user confirms or they expire.

Deleting a user can't be undone, so it takes two steps (SAV-17, owner update of
20 Nov 2026), and code, not the prompt, decides when the second may run:

1. `request_data_deletion` only answers with the confirmation question. The
   request is planned right before the question is sent, with its deadline
   and the "done" message already written in the user's language, and stored
   here **once that question was delivered** (`app.agent.send`). A question that was never
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

**Storing late never wins over newer state.** The chat worker may store a
request again after a failure, also after the user moved on, so a store only
counts when nothing newer is there: a request waiting for a later message, or
one with the same awaited message and a later or equal deadline (the same
request, or a question asked again before the user wrote). Taking a request
leaves a used-up mark with its awaited message until its deadline, so storing
the same request again can't bring it back. A request past its deadline is not
stored at all. Each of these steps is one Redis script.

One key per user, `savely:agent:<user id>:deletion_request`, expires at the
deadline and is erased with the user's data. It holds the deadline, the
message number, the language and the bot's own "done" wording, or only the
used-up mark; nothing the user wrote.
"""

import json
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol, cast

from redis.asyncio import Redis

_KEY_PREFIX: Final = "savely:agent"
_KEY_NAME: Final = "deletion_request"
_MINIMUM_LIFETIME_MS: Final = 1
_DEADLINE_MS: Final = "deadline_ms"
_AWAITED_MESSAGE: Final = "awaited_message"
_USED_UP: Final = "used_up"
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_MILLISECOND: Final = timedelta(milliseconds=1)
# Returns the request and leaves the used-up mark until its deadline, only if
# the request waits for the given message number; nothing otherwise.
_TAKE_IF_AWAITED: Final = f"""
local stored = redis.call("GET", KEYS[1])
if not stored then
    return false
end
local request = cjson.decode(stored)
if request.{_USED_UP} or request.{_AWAITED_MESSAGE} ~= tonumber(ARGV[1]) then
    return false
end
redis.call("SET", KEYS[1], ARGV[2], "KEEPTTL")
return stored
"""
# Stores a request unless a newer one, or this one used up, is there already.
_SAVE_UNLESS_NEWER: Final = f"""
local stored = redis.call("GET", KEYS[1])
if stored then
    local kept = cjson.decode(stored)
    local awaited = tonumber(ARGV[2])
    if kept.{_AWAITED_MESSAGE} > awaited then
        return 0
    end
    if kept.{_AWAITED_MESSAGE} == awaited
        and (kept.{_USED_UP} or kept.{_DEADLINE_MS} >= tonumber(ARGV[3])) then
        return 0
    end
end
redis.call("SET", KEYS[1], ARGV[1], "PX", ARGV[4])
return 1
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
        """Store a delivered request, unless a newer one or this one used up is there."""
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
        """Store a delivered request until its deadline, replacing an older one.

        Storing it again, after it was used up, or after a newer question was
        stored changes nothing, and a request past its deadline is not stored.

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
        remaining_ms = (pending.deadline - now) // _MILLISECOND
        if remaining_ms < _MINIMUM_LIFETIME_MS:
            return
        await cast(
            Awaitable[int],
            self._redis.eval(
                _SAVE_UNLESS_NEWER,
                1,
                _key(user_id),
                json.dumps(pending_to_value(pending), ensure_ascii=False),
                str(pending.awaited_message),
                str(_milliseconds_since_epoch(pending.deadline)),
                str(remaining_ms),
            ),
        )

    async def take(self, user_id: uuid.UUID, message_number: int) -> PendingDeletion | None:
        """Remove the user's request and return it, if it waits for this message.

        A request that waits for another message stays; the caller checks the
        deadline. A taken request leaves its used-up mark until its deadline.
        Checking and taking are one step in Redis.

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
            self._redis.eval(
                _TAKE_IF_AWAITED,
                1,
                _key(user_id),
                str(message_number),
                json.dumps({_AWAITED_MESSAGE: message_number, _USED_UP: True}),
            ),
        )
        return None if stored is None else pending_from_value(json.loads(stored))

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


def pending_to_value(pending: PendingDeletion) -> dict[str, object]:
    """Return a request as a JSON object.

    Args:
        pending: The request.

    Returns:
        Its deadline, awaited message, language and "done" wording.
    """
    return {
        _DEADLINE_MS: _milliseconds_since_epoch(pending.deadline),
        _AWAITED_MESSAGE: pending.awaited_message,
        _LANGUAGE: pending.language,
        _DONE_TEXT: pending.done_text,
    }


def pending_from_value(value: dict[str, Any]) -> PendingDeletion:
    """Read a request back from `pending_to_value`'s JSON object.

    Args:
        value: The JSON object.

    Returns:
        The request.
    """
    return PendingDeletion(
        deadline=_EPOCH + value[_DEADLINE_MS] * _MILLISECOND,
        awaited_message=value[_AWAITED_MESSAGE],
        language=value[_LANGUAGE],
        done_text=value[_DONE_TEXT],
    )


def _milliseconds_since_epoch(moment: datetime) -> int:
    return (moment - _EPOCH) // _MILLISECOND
