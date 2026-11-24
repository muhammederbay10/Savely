"""The stuck-item sweeper: Postgres is the source of truth for unfinished items.

A job can't always finish its own bookkeeping. Redis or Postgres may be down,
the owner's row may be locked by "delete my data", or the worker may die. The
sweeper repairs what a job couldn't, from what Postgres holds, so nothing
depends on a Redis job surviving or on its expiry (record 0012).

It runs as an arq cron in the indexing worker. Each sweep:

1. Reads unfinished items (`pending`, `enriching`) that no job or sweep has
   worked on for `INDEXING_STALE_AFTER_SECONDS` (`saved_item.indexing_updated_at`,
   on the database's clock). That threshold is validated to be longer than a
   live job can stay silent, so a job that is still running or waiting for its
   retry is never touched.
2. Handles each item in its own transaction, under the owner's row lock taken
   with `SKIP LOCKED`. A row that is locked (a job writing, "delete my data")
   is skipped at once and comes back on the next sweep; the owner's other items
   are skipped too in this sweep, without asking again. Staleness is checked
   again under the lock, so two workers sweeping together handle an item once.
   Candidates are read page by page, so skipped items don't count against
   `INDEXING_SWEEP_BATCH_SIZE`: a sweep repairs up to that many items and
   moves past locked owners to reach them.
3. An item with tries left (`indexing_attempts` below `INDEXING_MAX_TRIES`) and
   its payload still in Redis is queued again as `index_saved_item`, with its
   usual job id: arq refuses a duplicate of a job that is still queued or
   running, a finished job's kept result is dropped first, and the run resumes
   after the finished steps. Any other stuck item is marked `failed`; if it
   has a collection the user named, it gets one collection confirmation after
   the commit (`app.pipeline.recovery.acknowledgment`, decision record 0018).

Deleted users and items are gone from Postgres, so they are never swept.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol

from sqlalchemy import ColumnElement, func, literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ItemStatus, SavedItem, User
from app.ingestion.jobs import INDEX_SAVED_ITEM_JOB, JobArguments, index_job_id
from app.pipeline.payloads import CaptureSource
from app.pipeline.recovery.acknowledgment import (
    CollectionAcknowledger,
    CollectionAcknowledgment,
    claim_collection_acknowledgment,
)
from app.pipeline.recovery.runs import mark_worked_on

logger = logging.getLogger(__name__)

_UNFINISHED = (ItemStatus.PENDING, ItemStatus.ENRICHING)


class Requeuer(Protocol):
    """Queues a job again whose earlier run is over (`ArqJobQueue.requeue`)."""

    async def requeue(self, job_name: str, job_id: str, arguments: JobArguments) -> bool:
        """Queue the job, unless it is still queued or running.

        Args:
            job_name: The worker function to run.
            job_id: The job's id.
            arguments: The function's keyword arguments.

        Returns:
            Whether the job was queued.
        """
        ...


@dataclass
class SweepReport:
    """What one sweep did. It holds counts only, no ids or content.

    Attributes:
        requeued: Items queued again (or found still queued).
        failed: Items marked `failed`.
        skipped: Items whose owner's row was locked, or that were no longer
            stuck when the sweep got to them. They don't count against the
            batch size.
    """

    requeued: int = 0
    failed: int = 0
    skipped: int = 0

    def as_log(self) -> dict[str, int]:
        """Return the counts for a log line or a job result."""
        return {"requeued": self.requeued, "failed": self.failed, "skipped": self.skipped}


@dataclass(frozen=True)
class _Candidate:
    item_id: uuid.UUID
    user_id: uuid.UUID
    worked_on_at: datetime


class _Handled(Enum):
    REPAIRED = "repaired"
    OWNER_LOCKED = "owner_locked"
    NOT_STUCK = "not_stuck"


@dataclass
class StuckItemSweeper:
    """Requeues or fails unfinished items that no job is working on any more.

    Attributes:
        sessions: Opens a database session.
        payloads: Tells whether an item's capture payload is still in Redis.
        job_queue: Where `index_saved_item` jobs are queued again.
        stale_after_seconds: `INDEXING_STALE_AFTER_SECONDS`.
        max_tries: `INDEXING_MAX_TRIES`, the runs an item gets in all.
        batch_size: `INDEXING_SWEEP_BATCH_SIZE`, the most items one sweep handles.
        acknowledger: Confirms the collection of an item it marks `failed`.
    """

    sessions: async_sessionmaker[AsyncSession]
    payloads: CaptureSource
    job_queue: Requeuer
    stale_after_seconds: float
    max_tries: int
    batch_size: int
    acknowledger: CollectionAcknowledger

    async def sweep(self) -> SweepReport:
        """Handle the stuck items found now, oldest first.

        Returns:
            How many were requeued, failed or skipped.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database can't be reached.
            redis.exceptions.RedisError: If Redis can't be reached; the item
                being handled is left as it was for the next sweep.
        """
        report = SweepReport()
        locked_owners: set[uuid.UUID] = set()
        after: _Candidate | None = None
        while report.requeued + report.failed < self.batch_size:
            page = await self._candidates(after)
            if not page:
                break
            for candidate in page:
                after = candidate
                if candidate.user_id in locked_owners:
                    report.skipped += 1
                    continue
                handled = await self._handle(candidate, report)
                if handled is _Handled.OWNER_LOCKED:
                    locked_owners.add(candidate.user_id)
                if report.requeued + report.failed >= self.batch_size:
                    break
        if report.requeued or report.failed or report.skipped:
            logger.info("stuck_items_swept", extra=report.as_log())
        return report

    async def _candidates(self, after: _Candidate | None) -> list[_Candidate]:
        """Read the next page of stuck items, oldest first, after the last one seen.

        Paging by `(indexing_updated_at, id)` lets a sweep move past items whose
        owner is locked, so they can't fill every batch.
        """
        statement = (
            select(SavedItem.id, SavedItem.user_id, SavedItem.indexing_updated_at)
            .where(self._is_stuck())
            .order_by(SavedItem.indexing_updated_at, SavedItem.id)
            .limit(self.batch_size)
        )
        if after is not None:
            statement = statement.where(
                tuple_(SavedItem.indexing_updated_at, SavedItem.id)
                > tuple_(literal(after.worked_on_at), literal(after.item_id))
            )
        async with self.sessions() as session:
            rows = await session.execute(statement)
            return [
                _Candidate(
                    item_id=row.id, user_id=row.user_id, worked_on_at=row.indexing_updated_at
                )
                for row in rows
            ]

    async def _handle(self, candidate: _Candidate, report: SweepReport) -> _Handled:
        handled, acknowledgment = await self._repair(candidate, report)
        if acknowledgment is not None:
            await self.acknowledger.send(acknowledgment)
        return handled

    async def _repair(
        self, candidate: _Candidate, report: SweepReport
    ) -> tuple[_Handled, CollectionAcknowledgment | None]:
        async with self.sessions() as session, session.begin():
            owner = await session.execute(
                select(User.id)
                .where(User.id == candidate.user_id)
                .with_for_update(key_share=True, skip_locked=True)
            )
            if owner.scalar_one_or_none() is None:
                report.skipped += 1
                return _Handled.OWNER_LOCKED, None
            item = (
                await session.execute(
                    select(SavedItem).where(SavedItem.id == candidate.item_id, self._is_stuck())
                )
            ).scalar_one_or_none()
            if item is None:
                report.skipped += 1
                return _Handled.NOT_STUCK, None
            if item.indexing_attempts < self.max_tries and await self._has_payload(candidate):
                await self.job_queue.requeue(
                    INDEX_SAVED_ITEM_JOB,
                    index_job_id(candidate.item_id),
                    {"user_id": str(candidate.user_id), "saved_item_id": str(candidate.item_id)},
                )
                await mark_worked_on(session, candidate.item_id)
                report.requeued += 1
                logger.info("stuck_item_requeued", extra={"saved_item_id": str(item.id)})
                return _Handled.REPAIRED, None
            item.status = ItemStatus.FAILED
            acknowledgment = await claim_collection_acknowledgment(session, item)
            report.failed += 1
            logger.warning(
                "stuck_item_failed",
                extra={"saved_item_id": str(item.id), "attempts": item.indexing_attempts},
            )
            return _Handled.REPAIRED, acknowledgment

    async def _has_payload(self, candidate: _Candidate) -> bool:
        event = await self.payloads.capture(candidate.user_id, candidate.item_id)
        return event is not None

    def _is_stuck(self) -> ColumnElement[bool]:
        stale_since = func.now() - timedelta(seconds=self.stale_after_seconds)
        return SavedItem.status.in_(_UNFINISHED) & (SavedItem.indexing_updated_at < stale_since)
