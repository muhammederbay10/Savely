"""What ingestion keeps in Redis about a user's recent messages.

Four keys per user, named `savely:ingestion:<user id>:<name>`, hold
everything ingestion writes to Redis about a user, so "delete my data" can
erase it by deleting four keys:

| Name | Type | Entries |
|---|---|---|
| `payloads` | hash | `capture:<saved item id>`: the `CaptureEvent` an indexing job works on |
| | | `chat:<receipt id>`: the `ChatMessage` a chat job answers |
| `receipts` | hash | `<platform>:<message id>`: how a text is routed (`TextReceipt`) |
| `shares` | sorted set | the user's recent shares (`RecordedShare`), scored by send time |
| `messages` | hash | `last`: the last message number; `<platform>:<message id>`: each number |

**Message numbers.** Every new message, a share or a text, gets the next
number (`number_message`), once per platform message id: a repeated delivery
of the same message keeps its first number, while a new message sharing a post
that was shared before gets a new one. A share without a message id can't be
told from its repeats, so each of its deliveries counts. A deletion question waits for one
message number: the next one after the question was sent
(`app.agent.deletion_requests`), so a message in between, in whatever order the
jobs run, leaves the question unconfirmable.

The jobs on the queue carry only ids. A worker reads the payload from here, and
finds nothing once the user is deleted.

Every write sets the key to expire after the configured lifetime. It must be
longer than the time Meta keeps delivering a failed webhook again, so that a
late delivery still finds its receipt and the shares sent before it. Shares
sent more than one lifetime before the newest recorded share are dropped.

Callers hold the user's row lock (`app.ingestion.repository.lock_user`) while
they read and write a user's keys. That makes each read-then-write safe and
keeps writes from racing "delete my data".
"""

import uuid
from collections.abc import Awaitable
from datetime import datetime, timedelta
from enum import StrEnum
from typing import cast

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis

from app.adapters.base import CaptureEvent, ChatMessage

_KEY_PREFIX = "savely:ingestion"
_PAYLOADS = "payloads"
_RECEIPTS = "receipts"
_SHARES = "shares"
_MESSAGES = "messages"
_LAST_MESSAGE = "last"
# Returns the number of a message, giving it the next one if it has none yet.
_NUMBER_MESSAGE = f"""
local number = redis.call("HGET", KEYS[1], ARGV[1])
if not number then
    number = redis.call("HINCRBY", KEYS[1], "{_LAST_MESSAGE}", 1)
    redis.call("HSET", KEYS[1], ARGV[1], number)
end
redis.call("EXPIRE", KEYS[1], ARGV[2])
return tonumber(number)
"""
_MILLISECONDS_PER_SECOND = 1000
_CAPTURE_PAYLOAD_PREFIX = "capture"
_CHAT_PAYLOAD_PREFIX = "chat"


class TextRoute(StrEnum):
    """Where a text goes: onto a just-shared item as its collection, or to the response agent.

    `UNDECIDED` is a text sent shortly after a share to an item with no
    collection yet: the chat worker asks the decision model, then keeps
    `COLLECTION_HINT` or `CHAT` in the receipt (`app.ingestion.text_routing`).
    """

    UNDECIDED = "undecided"
    COLLECTION_HINT = "collection_hint"
    CHAT = "chat"


class TextReceipt(BaseModel):
    """How one text was routed, kept so a repeated delivery is routed the same way.

    Ingestion writes it first, with the route `CHAT` or `UNDECIDED`; the chat
    worker writes the decision once. Every text gets a chat job.

    Attributes:
        id: The receipt's own id. The chat job and its payload are named after it.
        number: The text's number among the user's messages (`number_message`).
        route: Where the text goes.
        saved_item_id: The share the text may name a collection for (`UNDECIDED`),
            or the item it labels (`COLLECTION_HINT`); bound by send time at
            ingestion, never changed afterwards.
        chat_queued: Whether the chat job was queued.
        confirm_collection: For a collection hint, whether the item's save
            confirmation had started without it, so the chat worker sends a
            collection confirmation of its own.
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    number: int
    route: TextRoute
    saved_item_id: uuid.UUID | None
    chat_queued: bool
    confirm_collection: bool = False


class RecordedShare(BaseModel):
    """One share from a user: where and when it was sent, and the item it is stored as.

    A share of a post that is already saved points to the existing item, with
    the time of the new share.

    Attributes:
        platform: The platform the share was sent on.
        saved_item_id: The item the share is stored as.
        sent_at: When the user sent the share.
    """

    model_config = ConfigDict(frozen=True)

    platform: str
    saved_item_id: uuid.UUID
    sent_at: datetime


class RedisIngestionState:
    """Ingestion's per-user state in Redis: payloads, text receipts and recent shares."""

    def __init__(self, redis: Redis, lifetime: timedelta) -> None:
        """Create the state store.

        Args:
            redis: The Redis client to use.
            lifetime: How long a user's keys live after their last write.
        """
        self._redis = redis
        self._lifetime = lifetime

    async def save_capture_payload(
        self, user_id: uuid.UUID, saved_item_id: uuid.UUID, event: CaptureEvent
    ) -> None:
        """Keep the capture event an indexing job works on, unless a newer one is kept.

        A share of the same item that was sent later replaces it, so the job
        gets the freshest media link. A share sent earlier, such as an old
        message delivered again or late, leaves the newer one in place.

        Args:
            user_id: The user who shared the post.
            saved_item_id: The item the job indexes.
            event: The capture event.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        field = capture_payload_field(saved_item_id)
        kept = await self._hget(_key(user_id, _PAYLOADS), field)
        if (
            kept is not None
            and CaptureEvent.model_validate_json(kept).received_at > event.received_at
        ):
            return
        await self._write(user_id, _PAYLOADS, field, event.model_dump_json())

    async def save_chat_payload(
        self, user_id: uuid.UUID, receipt_id: uuid.UUID, message: ChatMessage
    ) -> None:
        """Keep the chat message a chat job answers.

        Args:
            user_id: The user who wrote the text.
            receipt_id: The id of the text's receipt.
            message: The chat message.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        field = chat_payload_field(receipt_id)
        await self._write(user_id, _PAYLOADS, field, message.model_dump_json())

    async def chat_payload(self, user_id: uuid.UUID, receipt_id: uuid.UUID) -> ChatMessage | None:
        """Return the chat message a chat job answers, if it is still kept.

        Args:
            user_id: The user who wrote the text.
            receipt_id: The id of the text's receipt.

        Returns:
            The message; `None` when the user was deleted or it outlived its lifetime.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        value = await self._hget(_key(user_id, _PAYLOADS), chat_payload_field(receipt_id))
        return None if value is None else ChatMessage.model_validate_json(value)

    async def number_message(
        self, user_id: uuid.UUID, platform: str, message_id: str | None
    ) -> int:
        """Return a message's number among the user's messages, giving it the next one if new.

        Args:
            user_id: The user who sent the message.
            platform: The platform it was sent on.
            message_id: The platform's id of the message; `None` when it has
                none, and then every call gives a new number.

        Returns:
            The message's number, from 1.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        key = _key(user_id, _MESSAGES)
        if message_id is None:
            async with self._redis.pipeline(transaction=True) as pipeline:
                pipeline.hincrby(key, _LAST_MESSAGE, 1)
                pipeline.expire(key, self._lifetime)
                number, _ = await pipeline.execute()
            return int(number)
        lifetime_seconds = str(int(self._lifetime.total_seconds()))
        field = _receipt_field(platform, message_id)
        return int(
            await cast(
                Awaitable[int],
                self._redis.eval(_NUMBER_MESSAGE, 1, key, field, lifetime_seconds),
            )
        )

    async def last_message_number(self, user_id: uuid.UUID) -> int:
        """Return the number of the user's last message, 0 if none is known.

        Args:
            user_id: The user.

        Returns:
            The number `number_message` gave last.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        value = await self._hget(_key(user_id, _MESSAGES), _LAST_MESSAGE)
        return 0 if value is None else int(value)

    async def record_share(
        self, user_id: uuid.UUID, platform: str, saved_item_id: uuid.UUID, sent_at: datetime
    ) -> None:
        """Remember a share, so a text sent after it can find it.

        Webhooks can arrive out of order, so shares are kept by the time they
        were sent, not by the order they arrived in. Shares sent more than one
        lifetime before this one are dropped.

        Args:
            user_id: The user who shared the post.
            platform: The platform it was shared on.
            saved_item_id: The item the share is stored as.
            sent_at: When the user sent the share.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        share = RecordedShare(platform=platform, saved_item_id=saved_item_id, sent_at=sent_at)
        key = _key(user_id, _SHARES)
        score = _milliseconds(sent_at)
        oldest_kept = score - _milliseconds_of(self._lifetime)
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.zadd(key, {share.model_dump_json(): score})
            pipeline.zremrangebyscore(key, "-inf", f"({oldest_kept}")
            pipeline.expire(key, self._lifetime)
            await pipeline.execute()

    async def latest_share_between(
        self, user_id: uuid.UUID, platform: str, earliest: datetime, latest: datetime
    ) -> RecordedShare | None:
        """Return the user's last share on a platform that was sent within a time span.

        Args:
            user_id: The user.
            platform: The platform.
            earliest: The earliest send time to consider, included.
            latest: The latest send time to consider, included.

        Returns:
            The share sent last in the span, or `None` if there is none.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        members = await cast(
            Awaitable[list[bytes]],
            self._redis.zrevrangebyscore(
                _key(user_id, _SHARES), _milliseconds(latest), _milliseconds(earliest)
            ),
        )
        shares = (RecordedShare.model_validate_json(member) for member in members)
        return next((share for share in shares if share.platform == platform), None)

    async def text_receipt(
        self, user_id: uuid.UUID, platform: str, message_id: str
    ) -> TextReceipt | None:
        """Return how a text was routed, if it was seen before.

        Args:
            user_id: The user who wrote the text.
            platform: The platform it was sent on.
            message_id: The platform's id of the message.

        Returns:
            The text's receipt, or `None` for a text seen for the first time.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        field = _receipt_field(platform, message_id)
        value = await self._hget(_key(user_id, _RECEIPTS), field)
        return None if value is None else TextReceipt.model_validate_json(value)

    async def save_text_receipt(
        self, user_id: uuid.UUID, platform: str, message_id: str, receipt: TextReceipt
    ) -> None:
        """Keep how a text was routed.

        Args:
            user_id: The user who wrote the text.
            platform: The platform it was sent on.
            message_id: The platform's id of the message.
            receipt: The receipt to keep.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        field = _receipt_field(platform, message_id)
        await self._write(user_id, _RECEIPTS, field, receipt.model_dump_json())

    async def chat_receipt_ids(self, user_id: uuid.UUID) -> list[uuid.UUID]:
        """Return the receipt ids of the user's texts, whatever their route.

        Every text gets a chat job, which decides a collection hint too.

        Args:
            user_id: The user.

        Returns:
            The ids, which name the user's chat jobs.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        values = await self._hvals(_key(user_id, _RECEIPTS))
        return [TextReceipt.model_validate_json(value).id for value in values]

    async def erase_user(self, user_id: uuid.UUID) -> None:
        """Delete everything ingestion keeps in Redis for a user.

        Args:
            user_id: The user.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        await self._redis.delete(*(_key(user_id, name) for name in _ALL_KEY_NAMES))

    # redis-py declares its commands once for its blocking and its async client,
    # so their return types are unions. The async client always returns an awaitable.
    async def _hget(self, key: str, field: str) -> bytes | None:
        return await cast(Awaitable[bytes | None], self._redis.hget(key, field))

    async def _hvals(self, key: str) -> list[bytes]:
        return await cast(Awaitable[list[bytes]], self._redis.hvals(key))

    async def _write(self, user_id: uuid.UUID, key_name: str, field: str, value: str) -> None:
        key = _key(user_id, key_name)
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(key, field, value)
            pipeline.expire(key, self._lifetime)
            await pipeline.execute()


_ALL_KEY_NAMES = (_PAYLOADS, _RECEIPTS, _SHARES, _MESSAGES)


def payloads_key(user_id: uuid.UUID) -> str:
    """Return the name of the hash that holds a user's job payloads.

    Args:
        user_id: The user.

    Returns:
        The Redis key.
    """
    return _key(user_id, _PAYLOADS)


def capture_payload_field(saved_item_id: uuid.UUID) -> str:
    """Return the field that holds the capture event of an indexing job.

    Args:
        saved_item_id: The item the job indexes.

    Returns:
        The field in the user's payloads hash.
    """
    return f"{_CAPTURE_PAYLOAD_PREFIX}:{saved_item_id}"


def chat_payload_field(receipt_id: uuid.UUID) -> str:
    """Return the field that holds the chat message of a chat job.

    Args:
        receipt_id: The id of the text's receipt.

    Returns:
        The field in the user's payloads hash.
    """
    return f"{_CHAT_PAYLOAD_PREFIX}:{receipt_id}"


def _key(user_id: uuid.UUID, name: str) -> str:
    return f"{_KEY_PREFIX}:{user_id}:{name}"


def _receipt_field(platform: str, message_id: str) -> str:
    return f"{platform}:{message_id}"


def _milliseconds(moment: datetime) -> int:
    return round(moment.timestamp() * _MILLISECONDS_PER_SECOND)


def _milliseconds_of(duration: timedelta) -> int:
    return round(duration.total_seconds() * _MILLISECONDS_PER_SECOND)
