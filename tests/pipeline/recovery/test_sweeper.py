"""The stuck-item sweeper repairs unfinished items from Postgres, whatever Redis has kept.

These tests need Postgres with PostGIS and pgvector (`TEST_DATABASE_ADMIN_URL`)
and Redis (`TEST_REDIS_URL`). Their writes are committed, so another connection
can hold an owner's row lock; they are deleted afterwards.
"""

import asyncio
import time
import uuid
from datetime import timedelta

import pytest
from arq.connections import ArqRedis
from arq.constants import result_key_prefix
from arq.jobs import Job, JobStatus
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config.base import Settings
from app.db.deletion import delete_user_data
from app.db.models import ItemStatus, SavedItem
from app.ingestion.jobs import (
    INDEX_SAVED_ITEM_JOB,
    INDEXING_QUEUE,
    ArqJobQueue,
    create_arq_redis,
    index_job_id,
)
from app.models.errors import ModelUnavailableError
from app.pipeline.indexer import RetryLaterError
from app.pipeline.media.storage import LocalMediaStore
from app.pipeline.recovery.sweeper import StuckItemSweeper
from app.pipeline.worker import WorkerResources, build_worker
from tests.pipeline.conftest import hold_owner_lock
from tests.pipeline.harness import Pipeline, add_item, add_user, load
from tests.pipeline.support import (
    capture_event,
    open_extraction,
    pipeline_settings,
    structured_reply,
    transcript,
)
from tests.pipeline.test_worker_deadline import age, no_close, run_burst_worker

STALE_AFTER = timedelta(minutes=30)
MAX_TRIES = 3


@pytest.fixture
def pipeline(committed_pipeline: Pipeline) -> Pipeline:
    return committed_pipeline


async def new_item(pipeline: Pipeline, *, attempts: int = 0) -> tuple[uuid.UUID, uuid.UUID]:
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    if attempts:
        async with pipeline.sessions() as session, session.begin():
            await session.execute(
                update(SavedItem)
                .where(SavedItem.id == item_id)
                .values(indexing_attempts=attempts, status=ItemStatus.ENRICHING)
            )
    return user_id, item_id


def sweeper(pipeline: Pipeline, job_queue: ArqJobQueue) -> StuckItemSweeper:
    return pipeline.sweeper(
        job_queue, stale_after_seconds=STALE_AFTER.total_seconds(), max_tries=MAX_TRIES
    )


async def queued_index_jobs(redis: ArqRedis, item_id: uuid.UUID) -> int:
    queued = await redis.queued_jobs(queue_name=INDEXING_QUEUE)
    return sum(1 for job in queued if job.job_id == index_job_id(item_id))


async def worked_on_ago(pipeline: Pipeline, item_id: uuid.UUID) -> timedelta:
    async with pipeline.sessions() as session:
        age_now = await session.scalar(
            select(func.now() - SavedItem.indexing_updated_at).where(SavedItem.id == item_id)
        )
    assert age_now is not None
    return age_now


async def test_fresh_and_running_items_are_left_alone(
    pipeline: Pipeline, job_queue: ArqJobQueue, ingestion_redis: ArqRedis
) -> None:
    _, fresh = await new_item(pipeline)
    _, running = await new_item(pipeline, attempts=1)
    # A live job can be silent for a while (a long step, a retry wait), but
    # never as long as the stale threshold, which the settings check.
    await age(pipeline, running, STALE_AFTER - timedelta(minutes=1))

    report = await sweeper(pipeline, job_queue).sweep()

    assert (report.requeued, report.failed) == (0, 0)
    assert (await load(pipeline.sessions, fresh)).status == ItemStatus.PENDING
    assert (await load(pipeline.sessions, running)).status == ItemStatus.ENRICHING
    assert await queued_index_jobs(ingestion_redis, fresh) == 0
    assert await queued_index_jobs(ingestion_redis, running) == 0


async def test_a_stuck_item_with_tries_left_is_requeued_once(
    pipeline: Pipeline, job_queue: ArqJobQueue, ingestion_redis: ArqRedis
) -> None:
    _, item_id = await new_item(pipeline, attempts=1)
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))

    first = await sweeper(pipeline, job_queue).sweep()
    second = await sweeper(pipeline, job_queue).sweep()

    assert first.requeued == 1
    assert (second.requeued, second.failed) == (0, 0)
    assert await queued_index_jobs(ingestion_redis, item_id) == 1
    assert await worked_on_ago(pipeline, item_id) < timedelta(minutes=1)
    item = await load(pipeline.sessions, item_id)
    assert (item.status, item.indexing_attempts) == (ItemStatus.ENRICHING, 1)


async def test_a_stuck_item_without_tries_left_is_failed(
    pipeline: Pipeline, job_queue: ArqJobQueue, ingestion_redis: ArqRedis
) -> None:
    _, item_id = await new_item(pipeline, attempts=MAX_TRIES)
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))

    report = await sweeper(pipeline, job_queue).sweep()

    assert report.failed == 1
    assert (await load(pipeline.sessions, item_id)).status == ItemStatus.FAILED
    assert await queued_index_jobs(ingestion_redis, item_id) == 0


async def test_a_stuck_named_item_that_is_failed_gets_one_collection_confirmation(
    pipeline: Pipeline, job_queue: ArqJobQueue
) -> None:
    _, item_id = await new_item(pipeline, attempts=MAX_TRIES)
    async with pipeline.sessions() as session, session.begin():
        await session.execute(
            update(SavedItem).where(SavedItem.id == item_id).values(collection="Food")
        )
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))

    first = await sweeper(pipeline, job_queue).sweep()
    second = await sweeper(pipeline, job_queue).sweep()

    assert (first.failed, second.failed) == (1, 0)
    assert [text for _, text in pipeline.messenger.sent] == ["Saved to Food"]


async def test_a_stuck_item_whose_payload_is_gone_is_failed(
    pipeline: Pipeline, job_queue: ArqJobQueue
) -> None:
    _, item_id = await new_item(pipeline, attempts=1)
    del pipeline.payloads.events[item_id]
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))

    report = await sweeper(pipeline, job_queue).sweep()

    assert report.failed == 1
    assert (await load(pipeline.sessions, item_id)).status == ItemStatus.FAILED


async def test_a_locked_owner_is_skipped_at_once_and_handled_on_the_next_sweep(
    pipeline: Pipeline,
    committed_engine: AsyncEngine,
    job_queue: ArqJobQueue,
    ingestion_redis: ArqRedis,
) -> None:
    user_id, item_id = await new_item(pipeline, attempts=1)
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))
    locked = asyncio.Event()
    lock = asyncio.create_task(hold_owner_lock(committed_engine, user_id, 2.0, locked))
    await locked.wait()

    started = time.monotonic()
    blocked = await sweeper(pipeline, job_queue).sweep()
    elapsed = time.monotonic() - started
    await lock
    released = await sweeper(pipeline, job_queue).sweep()

    assert (blocked.requeued, blocked.skipped) == (0, 1)
    assert elapsed < 1.0
    assert released.requeued == 1
    assert await queued_index_jobs(ingestion_redis, item_id) == 1


async def test_two_workers_sweeping_at_once_requeue_an_item_once(
    pipeline: Pipeline, job_queue: ArqJobQueue, ingestion_redis: ArqRedis, redis_url: str
) -> None:
    item_ids = [(await new_item(pipeline, attempts=1))[1] for _ in range(5)]
    for item_id in item_ids:
        await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))
    # Two worker processes: each has its own Redis connections.
    other_worker_redis = create_arq_redis(redis_url)

    try:
        reports = await asyncio.gather(
            sweeper(pipeline, job_queue).sweep(),
            sweeper(pipeline, ArqJobQueue(other_worker_redis)).sweep(),
        )
    finally:
        await other_worker_redis.aclose(close_connection_pool=True)

    assert sum(report.requeued for report in reports) == len(item_ids)
    for item_id in item_ids:
        assert await queued_index_jobs(ingestion_redis, item_id) == 1


async def test_deleting_a_user_leaves_nothing_to_sweep(
    pipeline: Pipeline, job_queue: ArqJobQueue, ingestion_redis: ArqRedis
) -> None:
    user_id, item_id = await new_item(pipeline, attempts=1)
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))
    async with pipeline.sessions() as session, session.begin():
        await delete_user_data(session, user_id, LocalMediaStore(pipeline.media_root))

    report = await sweeper(pipeline, job_queue).sweep()

    assert (report.requeued, report.failed, report.skipped) == (0, 0, 0)
    assert await queued_index_jobs(ingestion_redis, item_id) == 0


async def test_requeueing_drops_a_finished_jobs_kept_result_but_never_a_live_job(
    job_queue: ArqJobQueue, ingestion_redis: ArqRedis
) -> None:
    finished, live = uuid.uuid4(), uuid.uuid4()
    arguments = {"user_id": str(uuid.uuid4()), "saved_item_id": str(finished)}
    # What arq keeps after a job has run: its result, under the job id.
    await ingestion_redis.set(result_key_prefix + index_job_id(finished), b"synthetic result")
    assert await job_queue.enqueue(INDEX_SAVED_ITEM_JOB, index_job_id(finished), arguments) is False
    await job_queue.enqueue(INDEX_SAVED_ITEM_JOB, index_job_id(live), arguments)

    assert await job_queue.requeue(INDEX_SAVED_ITEM_JOB, index_job_id(finished), arguments)
    assert not await job_queue.requeue(INDEX_SAVED_ITEM_JOB, index_job_id(live), arguments)
    for item_id in (finished, live):
        status = await Job(
            index_job_id(item_id), ingestion_redis, _queue_name=INDEXING_QUEUE
        ).status()
        assert status is JobStatus.queued


async def test_an_outage_longer_than_a_day_is_recovered_without_repeating_finished_steps(
    pipeline: Pipeline,
    settings: Settings,
    job_queue: ArqJobQueue,
    ingestion_redis: ArqRedis,
) -> None:
    # Try 1 finishes fetch, classify and transcribe, then fails at extract.
    outage = ModelUnavailableError("synthetic outage", operation="generate", model="fake/text")
    pipeline.script(
        transcripts=[transcript("synthetic speech")],
        extractions=[outage, structured_reply(open_extraction())],
    )
    user_id, item_id = await new_item(pipeline)
    with pytest.raises(RetryLaterError):
        await pipeline.indexer(max_tries=MAX_TRIES).index(user_id, item_id)
    # Then nothing runs for 25 hours: the retry job is gone (arq's own job
    # expiry is a day), and only Postgres still knows about the item.
    await job_queue.discard([index_job_id(item_id)])
    await age(pipeline, item_id, timedelta(hours=25))

    report = await sweeper(pipeline, job_queue).sweep()
    await run_burst_worker(
        settings,
        ingestion_redis,
        pipeline.indexer(max_tries=MAX_TRIES),
        max_tries=MAX_TRIES,
        run_deadline_seconds=30,
        recording_seconds=5,
        cleanup_reserve_seconds=2,
        job_timeout_seconds=60,
    )

    assert report.requeued == 1
    item = await load(pipeline.sessions, item_id)
    assert (item.status, item.indexing_attempts) == (ItemStatus.DONE, 2)
    assert pipeline.media_server.requests == 1
    assert pipeline.classifier.asked_keys().count("content_type") == 1
    assert len(pipeline.primary_stt.calls) == 1


async def test_the_workers_cron_sweeps_at_startup_and_the_same_worker_finishes_the_item(
    pipeline: Pipeline,
    settings: Settings,
    job_queue: ArqJobQueue,
    ingestion_redis: ArqRedis,
) -> None:
    _, item_id = await new_item(pipeline, attempts=1)
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))
    pipeline.script(
        transcripts=[transcript("synthetic speech")],
        extractions=[structured_reply(open_extraction())],
    )
    indexer = pipeline.indexer(max_tries=MAX_TRIES)
    real_sweeper = sweeper(pipeline, job_queue)

    async def startup(ctx: dict[str, object]) -> None:
        ctx["indexing_resources"] = WorkerResources(
            indexer=indexer, sweeper=real_sweeper, retry_delay_seconds=0.01, close=no_close
        )

    worker = build_worker(
        settings,
        pipeline_settings(indexing_max_tries=MAX_TRIES),
        on_startup=startup,
        redis_pool=ingestion_redis,
        burst=True,
    )
    # The worker borrows the fixture's Redis client, which the fixture closes.
    await worker.main()

    item = await load(pipeline.sessions, item_id)
    assert (item.status, item.indexing_attempts) == (ItemStatus.DONE, 2)


async def test_a_run_cancelled_after_a_requeue_never_starts_a_try_past_the_cap(
    pipeline: Pipeline,
    settings: Settings,
    job_queue: ArqJobQueue,
    ingestion_redis: ArqRedis,
) -> None:
    # Max 2 tries, one already spent. The sweeper requeues the item, so arq's own
    # try count starts at 1 again. Try 2 reserves its try, then is cancelled in
    # classify (a worker shutting down); arq runs the job again. That run must
    # find the budget spent and do no model work.
    max_tries = 2
    _, item_id = await new_item(pipeline, attempts=1)
    await age(pipeline, item_id, STALE_AFTER + timedelta(minutes=1))
    pipeline.classifier.cancelled_calls = 1
    pipeline.script(
        transcripts=[transcript("synthetic speech")],
        extractions=[structured_reply(open_extraction())],
    )
    requeued = await pipeline.sweeper(
        job_queue, stale_after_seconds=STALE_AFTER.total_seconds(), max_tries=max_tries
    ).sweep()
    assert requeued.requeued == 1

    await run_burst_worker(
        settings,
        ingestion_redis,
        pipeline.indexer(max_tries=max_tries),
        max_tries=max_tries,
        run_deadline_seconds=30,
        recording_seconds=5,
        cleanup_reserve_seconds=2,
        job_timeout_seconds=60,
    )

    item = await load(pipeline.sessions, item_id)
    assert (item.status, item.indexing_attempts) == (ItemStatus.FAILED, max_tries)
    assert pipeline.primary_stt.calls == []
    assert pipeline.text_model.requests == []


async def test_a_locked_owner_filling_a_whole_batch_does_not_starve_other_users(
    pipeline: Pipeline,
    committed_engine: AsyncEngine,
    job_queue: ArqJobQueue,
    ingestion_redis: ArqRedis,
) -> None:
    batch_size = 3
    locked_user = await add_user(pipeline.sessions)
    oldest = [
        await add_item(pipeline.sessions, pipeline, capture_event(locked_user))
        for _ in range(batch_size)
    ]
    for item_id in oldest:
        await age(pipeline, item_id, STALE_AFTER + timedelta(hours=2))
    _, other_item = await new_item(pipeline, attempts=1)
    await age(pipeline, other_item, STALE_AFTER + timedelta(hours=1))
    locked = asyncio.Event()
    lock = asyncio.create_task(hold_owner_lock(committed_engine, locked_user, 2.0, locked))
    await locked.wait()

    try:
        report = await pipeline.sweeper(
            job_queue,
            stale_after_seconds=STALE_AFTER.total_seconds(),
            max_tries=MAX_TRIES,
            batch_size=batch_size,
        ).sweep()
    finally:
        await lock

    assert report.requeued == 1
    assert report.skipped == batch_size
    assert await queued_index_jobs(ingestion_redis, other_item) == 1
    for item_id in oldest:
        assert await queued_index_jobs(ingestion_redis, item_id) == 0
