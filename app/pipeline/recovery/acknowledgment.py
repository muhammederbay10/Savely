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
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import SavedItem
from app.ingestion.repository import lock_user
from app.pipeline.recovery.runs import (
    RunStatus,
    Step,
    confirmation_started,
    language_of,
    load_item,
    recipient_of,
    record_step,
)
from app.pipeline.steps.confirm import ConfirmationStatus, SaveConfirmer

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
    """A collection confirmation claimed for a failed item, ready to send.

    Attributes:
        user_id: The item's owner.
        saved_item_id: The item.
        recipient_id: The owner's id on the item's platform; `None` if they have none.
        language: The owner's language from their profile; `None` while unknown.
        collection: The collection, as the user wrote it.
        shared_at: When the user shared the item, for the platform's reply window.
    """

    user_id: uuid.UUID
    saved_item_id: uuid.UUID
    recipient_id: str | None = field(repr=False)
    language: str | None
    collection: str = field(repr=False)
    shared_at: datetime


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
        What to send; `None` when the item has no collection, or its save or
        collection confirmation already started.
    """
    if item.collection is None or await confirmation_started(session, item.id):
        return None
    await record_step(session, item.id, Step.CONFIRM, RunStatus.STARTED, output=dict(_OUTPUT))
    return CollectionAcknowledgment(
        user_id=item.user_id,
        saved_item_id=item.id,
        recipient_id=await recipient_of(session, item.user_id, item.platform),
        language=await language_of(session, item.user_id),
        collection=item.collection,
        shared_at=item.saved_at,
    )


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
        """Send the confirmation once, then record its outcome on the confirm step.

        A failure to record is logged: the `started` mark already keeps it
        from being sent again.

        Args:
            claimed: What `claim_collection_acknowledgment` returned.
        """
        confirmation = await self._confirmer.confirm_collection(
            recipient_id=claimed.recipient_id,
            language=claimed.language,
            collection=claimed.collection,
            last_user_message_at=claimed.shared_at,
        )
        try:
            async with self._sessions.begin() as session:
                if not await lock_user(session, claimed.user_id):
                    return
                if await load_item(session, claimed.saved_item_id) is None:
                    return
                await record_step(
                    session,
                    claimed.saved_item_id,
                    Step.CONFIRM,
                    CONFIRMATION_RUN_STATUS[confirmation.status],
                    output={**confirmation.to_output(), **_OUTPUT},
                )
        except SQLAlchemyError as error:
            logger.error(
                "collection_acknowledgment_not_recorded",
                extra={"saved_item_id": str(claimed.saved_item_id), "error": type(error).__name__},
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
