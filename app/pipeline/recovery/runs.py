"""Database work for indexing one item: its row, its status and its steps in `enrichment_run`.

Every function runs inside the caller's transaction and never commits. Each
step of an item has at most one `enrichment_run` row, written when the step
finishes or fails. A job that is tried again reads the finished steps back and
continues after them, so no model is asked twice for the same step.

The rows hold the step's output, which can include the post's content (the
extraction, timestamps of what was said). They belong to the item and are
deleted with it, and with "delete my data".
"""

import uuid
from enum import StrEnum
from typing import Final

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EnrichmentRun, PlatformAccount, SavedItem, User


class Step(StrEnum):
    """The steps of indexing, in order."""

    FETCH = "fetch"
    CLASSIFY = "classify"
    TRANSCRIBE = "transcribe"
    EXTRACT = "extract"
    NORMALIZE = "normalize"
    EMBED = "embed"
    CONFIRM = "confirm"


class RunStatus(StrEnum):
    """How a step ended, or, for the confirmation, that it started to be sent."""

    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    STARTED = "started"


JsonObject = dict[str, object]
_FINISHED: Final = (RunStatus.DONE, RunStatus.SKIPPED)


async def load_item(session: AsyncSession, saved_item_id: uuid.UUID) -> SavedItem | None:
    """Return a saved item.

    Args:
        session: The session to work in.
        saved_item_id: The item.

    Returns:
        The item, or `None` if it is gone.
    """
    return await session.get(SavedItem, saved_item_id)


async def mark_worked_on(session: AsyncSession, saved_item_id: uuid.UUID) -> None:
    """Set the item's `indexing_updated_at` to the database's time.

    The stuck-item sweeper compares it with the database's clock, so it never
    depends on a worker's clock.

    Args:
        session: The session to work in.
        saved_item_id: The item.
    """
    await session.execute(
        update(SavedItem)
        .where(SavedItem.id == saved_item_id)
        .values(indexing_updated_at=func.now())
        .execution_options(synchronize_session=False)
    )


async def finished_steps(session: AsyncSession, saved_item_id: uuid.UUID) -> dict[Step, JsonObject]:
    """Return the outputs of the item's steps that finished.

    Args:
        session: The session to work in.
        saved_item_id: The item.

    Returns:
        Each finished step's output, by step.
    """
    rows = await session.execute(
        select(EnrichmentRun.step, EnrichmentRun.output).where(
            EnrichmentRun.saved_item_id == saved_item_id,
            EnrichmentRun.status.in_(_FINISHED),
        )
    )
    return {Step(row.step): dict(row.output or {}) for row in rows}


async def confirmation_started(session: AsyncSession, saved_item_id: uuid.UUID) -> bool:
    """Tell whether the item's save confirmation started to be sent, whatever came of it.

    The indexer records the start under the owner's row lock, together with
    reading the item's collection (`ItemIndexer`). A caller that holds the same
    lock learns whether a collection it sets now still makes it into that
    confirmation (`False`) or comes too late for it (`True`).

    Args:
        session: The session to work in.
        saved_item_id: The item.

    Returns:
        Whether the item has a row for the confirm step.
    """
    rows = await session.execute(
        select(EnrichmentRun.id).where(
            EnrichmentRun.saved_item_id == saved_item_id, EnrichmentRun.step == Step.CONFIRM
        )
    )
    return rows.first() is not None


async def record_step(
    session: AsyncSession,
    saved_item_id: uuid.UUID,
    step: Step,
    status: RunStatus,
    *,
    output: JsonObject | None = None,
    error: str | None = None,
) -> None:
    """Write how a step ended, replacing an earlier row of the same step.

    Args:
        session: The session to work in.
        saved_item_id: The item.
        step: The step.
        status: How it ended.
        output: What it produced.
        error: What went wrong, for a failed step. It never holds content.
    """
    existing = await session.execute(
        select(EnrichmentRun).where(
            EnrichmentRun.saved_item_id == saved_item_id, EnrichmentRun.step == step
        )
    )
    run = existing.scalars().first()
    if run is None:
        session.add(
            EnrichmentRun(
                saved_item_id=saved_item_id, step=step, status=status, output=output, error=error
            )
        )
        return
    run.status = status
    run.output = output
    run.error = error


async def recipient_of(session: AsyncSession, user_id: uuid.UUID, platform: str) -> str | None:
    """Return the user's id on a platform, to send them a DM.

    Args:
        session: The session to work in.
        user_id: The user.
        platform: The platform, e.g. `instagram`.

    Returns:
        The platform's id of the user's account, or `None` if they have none there.
    """
    result = await session.execute(
        select(PlatformAccount.platform_user_id).where(
            PlatformAccount.user_id == user_id, PlatformAccount.platform == platform
        )
    )
    return result.scalars().first()


async def language_of(session: AsyncSession, user_id: uuid.UUID) -> str | None:
    """Return the user's language from their profile.

    Args:
        session: The session to work in.
        user_id: The user.

    Returns:
        A language tag, or `None` when the profile has none.
    """
    result = await session.execute(select(User.language).where(User.id == user_id))
    return result.scalar_one_or_none()
