"""Confirms a named collection when the item's indexing fails for good (decision record 0018).

A collection the user named before the save confirmation started is left to
that confirmation ("Saved as café · in Food"). An item whose indexing fails
for good never gets one, so whatever marks it `failed` (the indexer, or the
stuck-item sweeper) confirms the collection itself, once, in the user's
language: "Saved to Food".

Two steps, like the save confirmation:

1. In the transaction that marks the item `failed`, under the owner's row lock:
   if the item has a collection and its confirm step never started, the
   confirm step is marked `started` (`claim_collection_acknowledgment`). The
   chat worker reads the same mark and the `failed` status under the same lock
   (`save_confirmation_misses_collection`), so for any order of "label set" and
   "indexing failed", exactly one of them confirms the collection.
2. After the commit: send it, and record how it went (`CollectionAcknowledger`).
   The mark is written before sending, so it is never sent twice; a failed
   send is recorded and not tried again.

**"Delete my data" never races the send** (the chat worker's protocol). Step 2
takes the user's row `FOR KEY SHARE`, which only the erasure's `FOR UPDATE`
waits for, and keeps it while it reads what to send, sends and records; the
webhook, which locks the user's row too, never waits for a send. The claim
carries only ids: the recipient, the language and the collection are read
again under that hold. A user or item deleted after the claim gets nothing,
and nothing is written for them; an erasure that starts while the message is
being sent waits for it, then erases everything.

**Once.** Step 2 also locks the item's confirm-step row `FOR UPDATE`, and sends
only while it is still `started`. An overlapping send of the same claim waits
for that row, then finds it recorded and sends nothing.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Final

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import SavedItem, User
from app.pipeline.recovery.runs import (
    RunStatus,
    Step,
    confirmation_started,
    language_of,
    load_item,
    lock_step,
    recipient_of,
    record_step,
)
from app.pipeline.steps.confirm import Confirmation, ConfirmationStatus, SaveConfirmer

logger = logging.getLogger(__name__)

# How the confirm step's outcome is recorded in `enrichment_run`.
CONFIRMATION_RUN_STATUS: Final = {
    ConfirmationStatus.SENT: RunStatus.DONE,
    ConfirmationStatus.OUTSIDE_REPLY_WINDOW: RunStatus.SKIPPED,
    ConfirmationStatus.SKIPPED: RunStatus.SKIPPED,
    ConfirmationStatus.FAILED: RunStatus.FAILED,
}
# Marks a confirm step that confirmed only the collection, after indexing failed.
AFTER_FAILURE: Final = "after_failure"
_OUTPUT: Final = {"collection_included": True, AFTER_FAILURE: True}


@dataclass(frozen=True)
class CollectionAcknowledgment:
    """A collection confirmation claimed for a failed item. Ids only, no content.

    Attributes:
        user_id: The item's owner.
        saved_item_id: The item.
    """

    user_id: uuid.UUID
    saved_item_id: uuid.UUID


async def claim_collection_acknowledgment(
    session: AsyncSession, item: SavedItem
) -> CollectionAcknowledgment | None:
    """Claim the collection confirmation of an item that is being marked `failed`.

    Call it under the owner's row lock, in the transaction that marks the item
    `failed`; send the claim only after that commits.

    Args:
        session: The session to work in.
        item: The item, loaded in that session.

    Returns:
        The claim; `None` when the item has no collection, or its save or
        collection confirmation already started.
    """
    if item.collection is None or await confirmation_started(session, item.id):
        return None
    await record_step(session, item.id, Step.CONFIRM, RunStatus.STARTED, output=dict(_OUTPUT))
    return CollectionAcknowledgment(user_id=item.user_id, saved_item_id=item.id)


class CollectionAcknowledger:
    """Sends a claimed collection confirmation and records how it went."""

    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], confirmer: SaveConfirmer
    ) -> None:
        """Create the acknowledger.

        Args:
            sessions: Opens a database session.
            confirmer: Sends the DM in the user's language.
        """
        self._sessions = sessions
        self._confirmer = confirmer

    async def send(self, claimed: CollectionAcknowledgment) -> None:
        """Send the confirmation once while holding the user, then record its outcome.

        Nothing is sent or written when the user or the item is gone, or the
        claim was already used. A database failure is logged: the `started`
        mark already keeps the message from being sent again.

        Args:
            claimed: What `claim_collection_acknowledgment` returned.
        """
        try:
            async with self._sessions.begin() as session:
                confirmation = await self._send_held(session, claimed)
        except SQLAlchemyError as error:
            logger.error(
                "collection_acknowledgment_not_recorded",
                extra={"saved_item_id": str(claimed.saved_item_id), "error": type(error).__name__},
            )
            return
        if confirmation is None:
            logger.info(
                "collection_acknowledgment_dropped",
                extra={"saved_item_id": str(claimed.saved_item_id)},
            )
            return
        logger.info(
            "collection_acknowledged_after_failure",
            extra={
                "saved_item_id": str(claimed.saved_item_id),
                "status": confirmation.status,
                "reason": confirmation.reason,
            },
        )

    async def _send_held(
        self, session: AsyncSession, claimed: CollectionAcknowledgment
    ) -> Confirmation | None:
        """Read, send and record, holding the user and the claim; `None` if nothing may be sent."""
        # Both are held until the transaction ends. The erasure's FOR UPDATE on the
        # user waits for the send and the record; ingestion's lock doesn't.
        if not await _hold_user(session, claimed.user_id):
            return None
        # An overlapping send of the same claim waits here, then finds it used.
        if await lock_step(session, claimed.saved_item_id, Step.CONFIRM) != RunStatus.STARTED:
            return None
        item = await load_item(session, claimed.saved_item_id)
        if item is None or item.user_id != claimed.user_id or item.collection is None:
            return None
        confirmation = await self._confirmer.confirm_collection(
            recipient_id=await recipient_of(session, item.user_id, item.platform),
            language=await language_of(session, item.user_id),
            collection=item.collection,
            last_user_message_at=item.saved_at,
        )
        await record_step(
            session,
            item.id,
            Step.CONFIRM,
            CONFIRMATION_RUN_STATUS[confirmation.status],
            output={**confirmation.to_output(), **_OUTPUT},
        )
        return confirmation


async def _hold_user(session: AsyncSession, user_id: uuid.UUID) -> bool:
    """Take the user's row `FOR KEY SHARE` until the transaction ends, like an answer's hold.

    It conflicts only with the erasure's `FOR UPDATE`, not with ingestion, the
    indexer or another hold, so the webhook never waits for a send.

    Returns:
        Whether the user exists. `False` means they were deleted.
    """
    statement = select(User.id).where(User.id == user_id).with_for_update(read=True, key_share=True)
    return (await session.execute(statement)).scalar_one_or_none() is not None
