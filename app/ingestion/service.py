"""Ingestion service: stores captures, routes texts and queues the jobs that follow.

Each message is handled in two steps:

1. In one database transaction that holds the user's row lock: number the
   message, store the item or decide the text's route, and write what the job
   needs to Redis (`app/ingestion/session_state.py`). "Delete my data" takes the same lock,
   so nothing is written for a user while they are being deleted.
2. After the commit: queue the job. Jobs carry only ids, so a worker never sees
   an item before it is committed, and a job queued for a user who was deleted
   meanwhile holds nothing personal.

If queuing fails, the caller reports an error and the platform delivers the
message again. Every step can be repeated safely:

- a capture finds the item that is already stored, and is queued again only
  while the item is still waiting to be indexed;
- a text is routed once. Its receipt is kept before anything is changed, and a
  repeated delivery follows the receipt, also when shares arrived in between;
- a chat job is queued again only if its receipt says it was not queued;
- a job is queued under an id made from the item or the receipt, and the queue
  refuses an id it already knows.
"""

import logging
import uuid
from datetime import timedelta
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.base import CaptureEvent, ChatMessage, Platform
from app.db.models import ItemStatus
from app.ingestion.jobs import (
    ANSWER_CHAT_MESSAGE_JOB,
    INDEX_SAVED_ITEM_JOB,
    JobQueue,
    chat_job_id,
    index_job_id,
)
from app.ingestion.repository import (
    StoredCapture,
    find_or_create_user,
    is_unlabelled_item,
    lock_user,
    set_collection_if_missing,
    store_capture,
)
from app.ingestion.session_state import RedisIngestionState, TextReceipt, TextRoute

logger = logging.getLogger(__name__)


class IngestionOutcome(StrEnum):
    """What ingestion did with one capture or one text."""

    ITEM_SAVED = "item_saved"
    ITEM_ALREADY_SAVED = "item_already_saved"
    COLLECTION_HINT = "collection_hint"
    CHAT_QUEUED = "chat_queued"
    CHAT_ALREADY_QUEUED = "chat_already_queued"


class UnknownUserError(LookupError):
    """The user a message belongs to was deleted while the message was being handled.

    The platform delivers the message again, and it then starts a new user.
    """


class IngestionService:
    """Turns capture events and chat messages into stored items and queued jobs."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        job_queue: JobQueue,
        state: RedisIngestionState,
        collection_hint_window: timedelta,
    ) -> None:
        """Create the service.

        Args:
            session_factory: Opens the database sessions.
            job_queue: Where indexing and chat jobs are queued.
            state: Ingestion's per-user state in Redis.
            collection_hint_window: How long after a share a text from the
                same user still counts as that item's collection hint.
        """
        self._session_factory = session_factory
        self._job_queue = job_queue
        self._state = state
        self._collection_hint_window = collection_hint_window

    async def resolve_user(self, platform: Platform, platform_user_id: str) -> uuid.UUID:
        """Return our user for a platform account, creating both on first contact.

        Args:
            platform: The platform the account is on.
            platform_user_id: The platform's id of the account.

        Returns:
            The user's id.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database cannot be reached.
        """
        async with self._session_factory.begin() as session:
            return await find_or_create_user(session, platform, platform_user_id)

    async def ingest_capture(self, event: CaptureEvent) -> IngestionOutcome:
        """Store a capture as a saved item, record the share and queue the item's indexing.

        A post or a message that is already stored creates no second item, but
        is still recorded as a share, so a hint after it labels that item. Its
        indexing is queued again only while the item is still pending, which
        covers a first delivery that stored the item but failed to queue it.

        Args:
            event: The capture event.

        Returns:
            Whether the capture created an item or found one.

        Raises:
            UnknownUserError: If the user was deleted meanwhile.
            sqlalchemy.exc.SQLAlchemyError: If the database cannot be reached.
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        async with self._session_factory.begin() as session:
            await _require_user(session, event.user_id)
            # A share is a message too: one in between makes a later "Yes, delete"
            # tap no longer the next message after the deletion question.
            await self._state.number_message(event.user_id)
            stored = await store_capture(session, event)
            await self._state.record_share(
                event.user_id, event.platform, stored.saved_item_id, event.received_at
            )
            needs_indexing = stored.created or stored.status == ItemStatus.PENDING
            if needs_indexing:
                await self._state.save_capture_payload(event.user_id, stored.saved_item_id, event)
        if needs_indexing:
            await self._job_queue.enqueue(
                INDEX_SAVED_ITEM_JOB,
                index_job_id(stored.saved_item_id),
                {"user_id": str(event.user_id), "saved_item_id": str(stored.saved_item_id)},
            )
        return _log_capture(event, stored)

    async def ingest_chat_message(self, message: ChatMessage) -> IngestionOutcome:
        """Attach a text to the share just before it as a collection hint, or queue it as chat.

        A text seen before follows its receipt and is not routed again.

        Args:
            message: The chat message.

        Returns:
            Whether the text became a collection hint or a chat job, and
            whether that chat job was already queued.

        Raises:
            UnknownUserError: If the user was deleted meanwhile.
            sqlalchemy.exc.SQLAlchemyError: If the database cannot be reached.
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        async with self._session_factory.begin() as session:
            await _require_user(session, message.user_id)
            receipt = await self._state.text_receipt(
                message.user_id, message.platform, message.message_id
            )
            if receipt is None:
                receipt = await self._route_new_text(session, message)
            if receipt.route is TextRoute.CHAT:
                if not receipt.chat_queued:
                    await self._state.save_chat_payload(message.user_id, receipt.id, message)
            elif receipt.saved_item_id is not None:
                await set_collection_if_missing(
                    session, message.user_id, receipt.saved_item_id, message.text
                )
        outcome = await self._finish_route(message, receipt)
        logger.info(
            "chat_message_ingested",
            extra={
                "outcome": outcome,
                "user_id": str(message.user_id),
                "platform": message.platform,
            },
        )
        return outcome

    async def _route_new_text(self, session: AsyncSession, message: ChatMessage) -> TextReceipt:
        hinted_item_id = await self._item_awaiting_hint(session, message)
        receipt = TextReceipt(
            id=uuid.uuid4(),
            number=await self._state.number_message(message.user_id),
            route=TextRoute.CHAT if hinted_item_id is None else TextRoute.COLLECTION_HINT,
            saved_item_id=hinted_item_id,
            chat_queued=False,
        )
        # Kept before the item is changed, so a repeated delivery is routed the same way.
        await self._state.save_text_receipt(
            message.user_id, message.platform, message.message_id, receipt
        )
        return receipt

    async def _item_awaiting_hint(
        self, session: AsyncSession, message: ChatMessage
    ) -> uuid.UUID | None:
        # The last share sent before the text, even if later shares arrived first.
        share = await self._state.latest_share_between(
            message.user_id,
            message.platform,
            earliest=message.received_at - self._collection_hint_window,
            latest=message.received_at,
        )
        if share is None:
            return None
        if not await is_unlabelled_item(session, message.user_id, share.saved_item_id):
            return None
        return share.saved_item_id

    async def _finish_route(self, message: ChatMessage, receipt: TextReceipt) -> IngestionOutcome:
        if receipt.route is TextRoute.COLLECTION_HINT:
            return IngestionOutcome.COLLECTION_HINT
        if receipt.chat_queued:
            return IngestionOutcome.CHAT_ALREADY_QUEUED
        await self._job_queue.enqueue(
            ANSWER_CHAT_MESSAGE_JOB,
            chat_job_id(receipt.id),
            {"user_id": str(message.user_id), "message_receipt_id": str(receipt.id)},
        )
        await self._mark_chat_queued(message, receipt)
        return IngestionOutcome.CHAT_QUEUED

    async def _mark_chat_queued(self, message: ChatMessage, receipt: TextReceipt) -> None:
        async with self._session_factory.begin() as session:
            # A user deleted since the job was queued keeps no receipt.
            if await lock_user(session, message.user_id):
                await self._state.save_text_receipt(
                    message.user_id,
                    message.platform,
                    message.message_id,
                    receipt.model_copy(update={"chat_queued": True}),
                )


async def _require_user(session: AsyncSession, user_id: uuid.UUID) -> None:
    if not await lock_user(session, user_id):
        raise UnknownUserError("the user was deleted while a message was being handled")


def _log_capture(event: CaptureEvent, stored: StoredCapture) -> IngestionOutcome:
    outcome = IngestionOutcome.ITEM_SAVED if stored.created else IngestionOutcome.ITEM_ALREADY_SAVED
    logger.info(
        "capture_ingested",
        extra={
            "outcome": outcome,
            "saved_item_id": str(stored.saved_item_id),
            "user_id": str(event.user_id),
            "platform": event.platform,
            "media_type": event.media_type,
        },
    )
    return outcome
