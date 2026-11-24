"""Indexes one saved item: runs the steps in order, keeps each result, and resumes after a failure.

The steps (`docs/ARCHITECTURE.md` §3.3): fetch → classify → speech-to-text →
extract (with the escalation) → normalize → embed, then the item is `done` and
the save confirmation is sent. Resolving entities (SAV-15) will go between
normalize and embed.

**Status:** `pending → enriching → done | failed`. `needs_user` is set by the
resolver's "which one?" question (SAV-15).

**Resuming.** Each step writes its result to `enrichment_run` when it finishes,
in the same transaction as its changes to the item. A job that is tried again
reads the finished steps and continues after them. A failure that may pass
later (a model or network error, a busy database) is recorded on the step and
raised as `RetryLaterError`; the worker tries the job again until its last try,
which marks the item `failed`.

**"Delete my data" while a job runs.** Every write first locks the user's row,
as ingestion does, and stops if the user or the item is gone. Media the fetch
step stored for a user who was deleted meanwhile is erased again.

**Logs** hold ids, the type, outcomes and costs, never content. Each indexed
item logs whether it was escalated and what its model calls cost, which gives
the escalation rate and the cost per item.
"""

import asyncio
import copy
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.base import CaptureEvent
from app.db.models import ItemStatus, SavedItem
from app.ingestion.repository import lock_user
from app.models.base import ContentPart, Image, Video
from app.models.errors import ModelError
from app.pipeline.costs import CallCosts
from app.pipeline.media.storage import MediaNotFoundError, MediaStore
from app.pipeline.media.tools import MediaToolError
from app.pipeline.payloads import CaptureSource
from app.pipeline.prompts import PipelinePrompts
from app.pipeline.recovery.acknowledgment import (
    CONFIRMATION_RUN_STATUS,
    CollectionAcknowledger,
    CollectionAcknowledgment,
    claim_collection_acknowledgment,
)
from app.pipeline.recovery.runs import (
    JsonObject,
    RunStatus,
    Step,
    finished_steps,
    language_of,
    load_item,
    mark_worked_on,
    recipient_of,
    record_step,
)
from app.pipeline.steps.classify import Classification, PostClassifier
from app.pipeline.steps.confirm import SaveConfirmer
from app.pipeline.steps.embed import ItemEmbedder
from app.pipeline.steps.extract import ExtractionResult, Extractor, PostText
from app.pipeline.steps.fetch import MediaDownloadError, MediaFetcher, MediaKind, StoredMedia
from app.pipeline.steps.transcribe import SpeechOutcome, SpeechToText
from app.pipeline.vocabulary.normalize import PROVENANCE_VALUES, Normalizer, raw_values_of
from app.pipeline.vocabulary.open_values import find_open_values, item_values, set_value
from app.pipeline.vocabulary.store import apply_item_values
from app.types.registry import ContentTypeRegistry

logger = logging.getLogger(__name__)


# Failures that may pass when the job runs again: model and network errors,
# ffmpeg under load, and a database or Redis that is briefly unreachable.
class IndexingDeadlineError(Exception):
    """A run of the job passed `INDEXING_RUN_DEADLINE_SECONDS` and was stopped.

    The step it was in is recorded as failed, and the job runs again from there.
    """


RETRYABLE_ERRORS: Final = (
    IndexingDeadlineError,
    ModelError,
    MediaDownloadError,
    MediaToolError,
    MediaNotFoundError,
    OSError,
    SQLAlchemyError,
    RedisError,
)
# Errors whose message is written to carry no content, so it can be stored.
_DESCRIBED_ERRORS: Final = (
    IndexingDeadlineError,
    ModelError,
    MediaDownloadError,
    MediaToolError,
)
# The name of the work before the first step, in logs: reading the payload and the item.
_START: Final = "start"
_RESUMABLE_STATUSES: Final = frozenset({ItemStatus.PENDING, ItemStatus.ENRICHING})


class IndexOutcome(StrEnum):
    """How one run of an indexing job ended."""

    INDEXED = "indexed"
    GONE = "gone"
    NOT_PENDING = "not_pending"
    FAILED = "failed"


class RetryLaterError(Exception):
    """A step failed in a way that may pass when the job runs again.

    Attributes:
        step: The step that failed; `None` when the run failed before its first
            step, while reading the payload or the item.
    """

    def __init__(self, step: Step | None, cause: BaseException) -> None:
        """Create the error.

        Args:
            step: The step that failed, or `None` before the first step.
            cause: What went wrong.
        """
        super().__init__(
            f"indexing step {step or _START} failed and can be tried again: {describe(cause)}"
        )
        self.step = step


class NothingToIndexError(Exception):
    """The item has no media and no caption, so there is nothing to understand."""


class _ItemGoneError(Exception):
    """The user or the item was deleted while the job ran."""


@dataclass(frozen=True)
class PipelineSteps:
    """The steps the indexer runs, each with its models and settings.

    Attributes:
        fetcher: Step 1.
        classifier: Step 2.
        speech: Step 3.
        extractor: Step 4, with the escalation.
        normalizer: Step 5.
        embedder: Step 6.
        confirmer: Step 7.
        store: Where the media is kept.
    """

    fetcher: MediaFetcher
    classifier: PostClassifier
    speech: SpeechToText
    extractor: Extractor
    normalizer: Normalizer
    embedder: ItemEmbedder
    confirmer: SaveConfirmer
    store: MediaStore


@dataclass
class _Run:
    """The state of one run of one job."""

    user_id: uuid.UUID
    item_id: uuid.UUID
    finished: dict[Step, JsonObject]
    current: Step = Step.FETCH
    costs: CallCosts = field(default_factory=CallCosts)

    def resume(self, step: Step) -> JsonObject | None:
        """Make `step` the current one, and return its output if it already finished."""
        self.current = step
        output = self.finished.get(step)
        if output is not None:
            self.costs = self.costs + CallCosts.from_output(output.get("costs"))
        return output

    def count(self, costs: CallCosts) -> None:
        """Add the charges of a step that just ran."""
        self.costs = self.costs + costs


@dataclass
class _Attempt:
    """One run of one job: whose item it is, and the steps once they started."""

    user_id: uuid.UUID
    item_id: uuid.UUID
    run: _Run | None = None
    last_try: bool = False
    acknowledgment: CollectionAcknowledgment | None = None

    def log_details(self, error: BaseException) -> dict[str, object]:
        """Return the log fields of a failure: ids, where it happened, what it was."""
        return {
            "saved_item_id": str(self.item_id),
            "step": self.run.current if self.run is not None else _START,
            "error": describe(error),
        }


def describe(error: BaseException) -> str:
    """Describe an error for `enrichment_run.error` and logs, without any content.

    Args:
        error: The error.

    Returns:
        The class name, plus the message for errors whose messages are written
        to carry no content.
    """
    if isinstance(error, _DESCRIBED_ERRORS):
        return f"{type(error).__name__}: {error}"
    return type(error).__name__


class ItemIndexer:
    """Indexes one saved item at a time."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        payloads: CaptureSource,
        steps: PipelineSteps,
        registry: ContentTypeRegistry,
        prompts: PipelinePrompts,
        *,
        alias_min_probability: float,
        run_deadline_seconds: float,
        failure_recording_seconds: float,
        max_tries: int,
    ) -> None:
        """Create the indexer.

        Args:
            sessions: Opens a database session.
            payloads: Reads the capture event of an item.
            steps: The steps.
            registry: The built-in types.
            prompts: The wording; its version is recorded with each step.
            alias_min_probability: `VOCAB_ALIAS_MIN_PROBABILITY`.
            run_deadline_seconds: `INDEXING_RUN_DEADLINE_SECONDS`: how long a whole
                run may take, from before the payload is read, before it is stopped
                and retried.
            failure_recording_seconds: `INDEXING_FAILURE_RECORDING_SECONDS`: how
                long recording a failure may take after the run stopped.
            max_tries: `INDEXING_MAX_TRIES`: how many runs an item gets in all,
                counted in `saved_item.indexing_attempts` across every requeue.
        """
        self._sessions = sessions
        self._payloads = payloads
        self._steps = steps
        self._registry = registry
        self._prompts = prompts
        self._alias_min_probability = alias_min_probability
        self._run_deadline_seconds = run_deadline_seconds
        self._failure_recording_seconds = failure_recording_seconds
        self._max_tries = max_tries
        self._acknowledger = CollectionAcknowledger(sessions, steps.confirmer)

    async def index(self, user_id: uuid.UUID, saved_item_id: uuid.UUID) -> IndexOutcome:
        """Index one item, continuing after the steps an earlier try finished.

        One deadline covers the whole run, from before the payload is read:
        reading the payload and the item, and every step. Recording a failure
        afterwards has its own allowance, from one deadline that starts when
        the run stops. Both, with the clean-up reserve, end before arq's own
        job timeout, so the job always ends itself.

        Each run that starts counts against the item's tries
        (`saved_item.indexing_attempts`), also across requeues by the sweeper.
        A failure on the item's last try marks it `failed`, as well as the job
        can; what it can't write, the stuck-item sweeper repairs later.

        Args:
            user_id: The item's owner, from the job.
            saved_item_id: The item, from the job.

        Returns:
            How the run ended.

        Raises:
            RetryLaterError: If the run failed in a way that may pass later,
                and this is not the item's last try.
            Exception: Any other error, after the item is marked `failed`.
        """
        attempt = _Attempt(user_id=user_id, item_id=saved_item_id)
        deadline = asyncio.timeout(self._run_deadline_seconds)
        try:
            try:
                async with deadline:
                    return await self._index_within_deadline(attempt)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                raise IndexingDeadlineError(
                    f"indexing ran past its deadline of {self._run_deadline_seconds} s"
                ) from error
        except _ItemGoneError:
            logger.info("indexing_item_gone", extra={"saved_item_id": str(saved_item_id)})
            return IndexOutcome.GONE
        except NothingToIndexError as error:
            return await self._fail(attempt, error)
        except RETRYABLE_ERRORS as error:
            if attempt.last_try:
                return await self._fail(attempt, error)
            await self._record_retry(attempt, error)
            step = attempt.run.current if attempt.run is not None else None
            raise RetryLaterError(step, error) from error
        except Exception as error:
            # A bug, not a passing failure: trying again would fail the same way.
            await self._fail(attempt, error)
            raise

    async def _index_within_deadline(self, attempt: _Attempt) -> IndexOutcome:
        event = await self._payloads.capture(attempt.user_id, attempt.item_id)
        if event is None:
            logger.info("indexing_payload_missing", extra={"saved_item_id": str(attempt.item_id)})
            return IndexOutcome.GONE
        started = await self._start(attempt)
        if isinstance(started, IndexOutcome):
            if attempt.acknowledgment is not None:
                await self._acknowledger.send(attempt.acknowledgment)
            return started
        attempt.run = _Run(user_id=attempt.user_id, item_id=attempt.item_id, finished=started)
        await self._run_steps(attempt.run, event)
        return IndexOutcome.INDEXED

    async def _start(self, attempt: _Attempt) -> dict[Step, JsonObject] | IndexOutcome:
        """Reserve one of the item's tries, and read the steps that already finished.

        The stored count is checked under the owner's row lock before a try is
        reserved, so an item never starts more than `INDEXING_MAX_TRIES` runs,
        whatever arq's own try count says after a requeue. An item whose tries
        are all spent is marked `failed` here, with no step run.
        """
        async with self._sessions() as session, session.begin():
            if not await lock_user(session, attempt.user_id):
                return IndexOutcome.GONE
            item = await load_item(session, attempt.item_id)
            if item is None or item.user_id != attempt.user_id:
                return IndexOutcome.GONE
            if item.status not in _RESUMABLE_STATUSES:
                logger.info(
                    "indexing_skipped",
                    extra={"saved_item_id": str(attempt.item_id), "status": item.status},
                )
                return IndexOutcome.NOT_PENDING
            if item.indexing_attempts >= self._max_tries:
                # Every try is spent (a run reserved the last one and was then
                # cancelled, and arq ran the job again). No step runs.
                item.status = ItemStatus.FAILED
                attempt.acknowledgment = await claim_collection_acknowledgment(session, item)
                logger.warning(
                    "saved_item_tries_spent",
                    extra={
                        "saved_item_id": str(attempt.item_id),
                        "attempts": item.indexing_attempts,
                    },
                )
                return IndexOutcome.FAILED
            item.status = ItemStatus.ENRICHING
            item.indexing_attempts += 1
            await mark_worked_on(session, attempt.item_id)
            attempt.last_try = item.indexing_attempts >= self._max_tries
            return await finished_steps(session, attempt.item_id)

    async def _run_steps(self, run: _Run, event: CaptureEvent) -> None:
        media = await self._fetch(run, event)
        caption, collection = await self._post_text(run)
        if media.kind is None and caption is None:
            raise NothingToIndexError("the item has no media and no caption")
        classification = await self._classify(run, media, caption, collection)
        transcript = await self._transcribe(run, media)
        post = PostText(caption, collection, transcript)
        extraction = await self._extract(run, classification, post, media)
        summary, category = await self._normalize(run, classification, extraction)
        await self._embed(run, summary, transcript)
        await self._finish(
            run, media, classification, extraction, has_speech=transcript is not None
        )
        await self._confirm(run, event, category)

    async def _fetch(self, run: _Run, event: CaptureEvent) -> StoredMedia:
        output = run.resume(Step.FETCH)
        if output is not None:
            return StoredMedia.from_output(output)
        media = await self._steps.fetcher.fetch(run.user_id, run.item_id, event.media_url)
        try:
            async with self._locked(run) as (session, _):
                await record_step(
                    session, run.item_id, Step.FETCH, RunStatus.DONE, output=media.to_output()
                )
        except _ItemGoneError:
            await self._steps.store.erase_item(run.user_id, run.item_id)
            raise
        return media

    async def _post_text(self, run: _Run) -> tuple[str | None, str | None]:
        async with self._sessions() as session:
            item = await load_item(session, run.item_id)
        if item is None:
            raise _ItemGoneError
        caption = item.caption if item.caption and item.caption.strip() else None
        return caption, item.collection

    async def _classify(
        self, run: _Run, media: StoredMedia, caption: str | None, collection: str | None
    ) -> Classification:
        output = run.resume(Step.CLASSIFY)
        if output is not None:
            return Classification.from_output(output)
        parts = await self._media_parts(media)
        classification = await self._steps.classifier.classify(parts, caption, collection)
        run.count(classification.costs)
        async with self._locked(run) as (session, item):
            item.type = classification.type_name
            await record_step(
                session,
                run.item_id,
                Step.CLASSIFY,
                RunStatus.DONE,
                output=classification.to_output(self._prompts.version),
            )
        return classification

    async def _transcribe(self, run: _Run, media: StoredMedia) -> str | None:
        output = run.resume(Step.TRANSCRIBE)
        if output is not None:
            async with self._sessions() as session:
                item = await load_item(session, run.item_id)
            return item.transcript if item is not None and output.get("has_speech") else None
        speech = await self._steps.speech.transcribe(media)
        run.count(speech.costs)
        async with self._locked(run) as (session, item):
            item.transcript = speech.speech_text
            await record_step(
                session, run.item_id, Step.TRANSCRIBE, RunStatus.DONE, output=speech.to_output()
            )
        if speech.outcome is SpeechOutcome.NO_SPEECH:
            logger.info("transcript_has_no_speech", extra={"saved_item_id": str(run.item_id)})
        return speech.speech_text

    async def _extract(
        self, run: _Run, classification: Classification, post: PostText, media: StoredMedia
    ) -> ExtractionResult:
        output = run.resume(Step.EXTRACT)
        if output is not None:
            return ExtractionResult.from_output(output)
        extraction = await self._steps.extractor.extract(classification.type_name, post, media)
        run.count(extraction.costs)
        async with self._locked(run) as (session, _):
            await record_step(
                session,
                run.item_id,
                Step.EXTRACT,
                RunStatus.DONE,
                output=extraction.to_output(self._prompts.version),
            )
        return extraction

    async def _normalize(
        self, run: _Run, classification: Classification, extraction: ExtractionResult
    ) -> tuple[str, str]:
        output = run.resume(Step.NORMALIZE)
        if output is not None:
            async with self._sessions() as session:
                item = await load_item(session, run.item_id)
            if item is None or item.summary is None or item.category is None:
                raise _ItemGoneError
            return item.summary, item.category
        schema = self._registry.extraction_schema(classification.type_name)
        document = copy.deepcopy(extraction.document)
        open_values = find_open_values(schema, document)
        normalization = await self._steps.normalizer.normalize(open_values)
        run.count(normalization.costs)
        for path, normalized in normalization.values.items():
            set_value(document, path, normalized.value)
        values = item_values(schema, document)
        provenance = {
            open_value.path: normalization.values[open_value.path].provenance(open_value.field)
            for open_value in open_values
        }
        async with self._locked(run) as (session, item):
            await apply_item_values(session, item, values, raw_values=raw_values_of(open_values))
            await record_step(
                session,
                run.item_id,
                Step.NORMALIZE,
                RunStatus.DONE,
                output={
                    PROVENANCE_VALUES: provenance,
                    "costs": normalization.costs.to_output(),
                    "prompts_version": self._prompts.version,
                },
            )
        await self._steps.normalizer.verify_aliases(
            normalization.confident_alias_ids(self._alias_min_probability)
        )
        return values.summary, values.category

    async def _embed(self, run: _Run, summary: str, transcript: str | None) -> None:
        if run.resume(Step.EMBED) is not None:
            return
        embedding = await self._steps.embedder.embed(summary, transcript)
        run.count(embedding.costs)
        async with self._locked(run) as (session, item):
            item.embedding = embedding.vector
            await record_step(
                session,
                run.item_id,
                Step.EMBED,
                RunStatus.DONE,
                output={"model": embedding.model, "costs": embedding.costs.to_output()},
            )

    async def _finish(
        self,
        run: _Run,
        media: StoredMedia,
        classification: Classification,
        extraction: ExtractionResult,
        *,
        has_speech: bool,
    ) -> None:
        async with self._locked(run) as (_, item):
            item.status = ItemStatus.DONE
        logger.info(
            "saved_item_indexed",
            extra={
                "saved_item_id": str(run.item_id),
                "type": classification.type_name,
                "media": media.kind,
                "has_speech": has_speech,
                "escalation": extraction.escalation,
                "escalated": extraction.escalation.was_called,
                "cost_usd": run.costs.cost_usd,
                "model_calls": run.costs.calls,
                "unpriced_model_calls": run.costs.unpriced_calls,
                "prompts_version": self._prompts.version,
            },
        )

    async def _confirm(self, run: _Run, event: CaptureEvent, category: str) -> None:
        if run.resume(Step.CONFIRM) is not None:
            return
        async with self._locked(run) as (session, item):
            recipient = await recipient_of(session, run.user_id, event.platform)
            language = await language_of(session, run.user_id)
            collection = item.collection
            # Read and marked under the owner's row lock: a collection the chat
            # worker sets after this is confirmed by its own message (record 0018).
            await record_step(
                session,
                run.item_id,
                Step.CONFIRM,
                RunStatus.STARTED,
                output={"collection_included": collection is not None},
            )
        confirmation = await self._steps.confirmer.confirm(
            recipient_id=recipient,
            language=language,
            category=category,
            collection=collection,
            last_user_message_at=event.received_at,
        )
        try:
            async with self._locked(run) as (session, _):
                await record_step(
                    session,
                    run.item_id,
                    Step.CONFIRM,
                    CONFIRMATION_RUN_STATUS[confirmation.status],
                    output={
                        **confirmation.to_output(),
                        "collection_included": collection is not None,
                    },
                )
        except _ItemGoneError:
            return
        logger.info(
            "save_confirmation_finished",
            extra={
                "saved_item_id": str(run.item_id),
                "status": confirmation.status,
                "reason": confirmation.reason,
            },
        )

    async def _media_parts(self, media: StoredMedia) -> Sequence[ContentPart]:
        if media.media_key is None or media.mime_type is None:
            return []
        data = await self._steps.store.get(media.media_key)
        if media.kind is MediaKind.VIDEO:
            return [Video(data=data, mime_type=media.mime_type)]
        return [Image(data=data, mime_type=media.mime_type)]

    @asynccontextmanager
    async def _locked(self, run: _Run) -> AsyncIterator[tuple[AsyncSession, SavedItem]]:
        """Open a transaction that holds the user's row lock, with the item loaded.

        Raises:
            _ItemGoneError: If the user or the item was deleted.
        """
        async with self._sessions() as session, session.begin():
            if not await lock_user(session, run.user_id):
                raise _ItemGoneError
            item = await load_item(session, run.item_id)
            if item is None:
                raise _ItemGoneError
            # Every write shows the sweeper that a job is working on the item.
            await mark_worked_on(session, run.item_id)
            yield session, item

    async def _fail(self, attempt: _Attempt, error: BaseException) -> IndexOutcome:
        """Mark the item `failed`, as well as this job can within its recording deadline.

        If the write can't happen in time (the owner's row is locked, the
        database is down), the item stays unfinished in Postgres, and the
        stuck-item sweeper marks it `failed` once it is stale.
        """
        logger.warning("saved_item_indexing_failed", extra=attempt.log_details(error))
        await self._record_within_allowance(attempt, error, ItemStatus.FAILED)
        return IndexOutcome.FAILED

    async def _record_retry(self, attempt: _Attempt, error: BaseException) -> None:
        logger.info("saved_item_indexing_retry", extra=attempt.log_details(error))
        await self._record_within_allowance(attempt, error, ItemStatus.ENRICHING)

    async def _record_within_allowance(
        self, attempt: _Attempt, error: BaseException, item_status: ItemStatus
    ) -> None:
        """Record a failure, but give up after `INDEXING_FAILURE_RECORDING_SECONDS`.

        The allowance is one absolute deadline that starts now, when the run
        has stopped and cleaned up. arq cancels the job without a retry at its
        own timeout, so a database that is locked or unreachable must not hold
        the job past it; the sweeper repairs what isn't written.
        """
        record_by = asyncio.get_running_loop().time() + self._failure_recording_seconds
        try:
            async with asyncio.timeout_at(record_by):
                await self._record_failure(attempt, error, item_status)
        except TimeoutError:
            logger.error(
                "indexing_failure_not_recorded",
                extra={
                    "saved_item_id": str(attempt.item_id),
                    "error": "recording took longer than INDEXING_FAILURE_RECORDING_SECONDS",
                },
            )

    async def _record_failure(
        self, attempt: _Attempt, error: BaseException, item_status: ItemStatus
    ) -> None:
        """Write the item's status and, once a step had started, that step's failure.

        A failure before the first step records no step, so every step that
        already finished stays finished. An item failed for good that has a
        collection gets the collection confirmation, after the commit.
        """
        acknowledgment: CollectionAcknowledgment | None = None
        try:
            async with self._sessions() as session, session.begin():
                if not await lock_user(session, attempt.user_id):
                    return
                item = await load_item(session, attempt.item_id)
                if item is None:
                    return
                # A run stopped after the item was done (while confirming) leaves it done.
                if item.status in _RESUMABLE_STATUSES:
                    item.status = item_status
                    if item_status is ItemStatus.FAILED:
                        acknowledgment = await claim_collection_acknowledgment(session, item)
                if attempt.run is not None:
                    await record_step(
                        session,
                        attempt.item_id,
                        attempt.run.current,
                        RunStatus.FAILED,
                        error=describe(error),
                    )
            if acknowledgment is not None:
                await self._acknowledger.send(acknowledgment)
        except (SQLAlchemyError, OSError) as record_error:
            # The database is what failed; the job's own outcome still stands.
            logger.error(
                "indexing_failure_not_recorded",
                extra={"saved_item_id": str(attempt.item_id), "error": describe(record_error)},
            )
