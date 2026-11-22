"""The background jobs ingestion queues, and the queue it puts them on.

Ingestion only queues jobs. The workers that run them belong to the indexing
pipeline and the response agent, and must register functions with these names
and keyword arguments:

| Job | Keyword arguments | Field in the user's payloads hash |
|---|---|---|
| `index_saved_item` | `user_id`, `saved_item_id` | `capture:<saved_item_id>` |
| `answer_chat_message` | `user_id`, `message_receipt_id` | `chat:<message_receipt_id>` |

The indexing worker's stuck-item sweeper queues `index_saved_item` again, with
the same job id, for an item no job is working on any more (record 0012).

The arguments are ids only, as text. Captions, texts and media links stay in
the payloads hash (`app/ingestion/session_state.py`), which "delete my data"
erases. A worker that finds no payload stops: the user was deleted.

**Each job name has its own queue** (`INDEXING_QUEUE`, `CHAT_QUEUE`). A worker
reads only the queue of the jobs it runs, so the indexing worker never takes a
chat job it has no function for: that job would fail for good and be lost.
Chat jobs are run by the chat worker (`app/agent/worker.py`).

Every job has an id made from what it works on. arq refuses a second job with
an id it still knows, which is how a webhook that Meta delivers twice ends up
as one job. arq knows an id while the job is queued or running, and afterwards
for as long as the worker keeps the job's result.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

from arq.connections import ArqRedis
from arq.constants import default_queue_name, job_key_prefix, result_key_prefix, retry_key_prefix
from arq.jobs import Job, JobStatus
from redis.asyncio import ConnectionPool

# A job's keyword arguments are JSON-compatible values of any shape. arq takes
# them next to its own typed options, so they cannot be typed more narrowly than `Any`.
JobArguments = Mapping[str, Any]

INDEX_SAVED_ITEM_JOB = "index_saved_item"
ANSWER_CHAT_MESSAGE_JOB = "answer_chat_message"

# arq's queues are sorted sets named like its own default queue (`arq:queue`).
INDEXING_QUEUE = f"{default_queue_name}:indexing"
CHAT_QUEUE = f"{default_queue_name}:chat"
_QUEUE_OF_JOB = {INDEX_SAVED_ITEM_JOB: INDEXING_QUEUE, ANSWER_CHAT_MESSAGE_JOB: CHAT_QUEUE}
# Every queue a job of ours can be in. arq's default queue is included because
# jobs queued before the queues were split are still there, and "delete my data"
# must remove them too.
_ALL_QUEUES = (INDEXING_QUEUE, CHAT_QUEUE, default_queue_name)


def queue_of(job_name: str) -> str:
    """Return the queue that jobs of one name go to.

    Args:
        job_name: `INDEX_SAVED_ITEM_JOB` or `ANSWER_CHAT_MESSAGE_JOB`.

    Returns:
        The queue's Redis key.

    Raises:
        ValueError: If no queue is set for the job name.
    """
    try:
        return _QUEUE_OF_JOB[job_name]
    except KeyError as error:
        raise ValueError(f"no queue is set for job {job_name!r}") from error


# What arq keeps under a job id once the job is queued, retried or finished.
_JOB_KEY_PREFIXES = (job_key_prefix, retry_key_prefix, result_key_prefix)


def index_job_id(saved_item_id: UUID) -> str:
    """Return the id of the job that indexes one saved item.

    Args:
        saved_item_id: The saved item's id.

    Returns:
        The job id. It is the same for every share of the same item.
    """
    return f"index:{saved_item_id}"


def chat_job_id(message_receipt_id: UUID) -> str:
    """Return the id of the job that answers one chat message.

    Args:
        message_receipt_id: The id of the text's receipt.

    Returns:
        The job id. It is the same for every delivery of the same message.
    """
    return f"chat:{message_receipt_id}"


class JobQueue(Protocol):
    """Where ingestion puts background jobs."""

    async def enqueue(self, job_name: str, job_id: str, arguments: JobArguments) -> bool:
        """Queue a job, unless a job with the same id is already known.

        Args:
            job_name: The name of the worker function to run.
            job_id: The job's id. A second job with a known id is not queued.
            arguments: The function's keyword arguments, as JSON-compatible values.

        Returns:
            Whether the job was queued. `False` means the id was already known.
        """
        ...

    async def discard(self, job_ids: Sequence[str]) -> None:
        """Remove jobs from the queue, with their retry state and kept results.

        Args:
            job_ids: The ids of the jobs. Unknown ids are skipped.
        """
        ...


class ArqJobQueue:
    """A `JobQueue` on Redis, read by arq workers."""

    def __init__(self, redis: ArqRedis) -> None:
        """Wrap an arq Redis client.

        Args:
            redis: The client to queue jobs through.
        """
        self._redis = redis

    async def enqueue(self, job_name: str, job_id: str, arguments: JobArguments) -> bool:
        """Queue a job on Redis, unless arq already knows the job id.

        Args:
            job_name: The name of the worker function to run.
            job_id: The job's id.
            arguments: The function's keyword arguments.

        Returns:
            Whether the job was queued.

        Raises:
            ValueError: If no queue is set for the job name.
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        job = await self._redis.enqueue_job(
            job_name, _job_id=job_id, _queue_name=queue_of(job_name), **arguments
        )
        return job is not None

    async def requeue(self, job_name: str, job_id: str, arguments: JobArguments) -> bool:
        """Queue a job again whose earlier run is over, even while arq keeps its result.

        arq refuses a job id while it still keeps the result of a finished run
        (an hour by default). A finished job is over, so its kept result is
        dropped first. A job that is queued, deferred or running is left alone,
        and the new one is refused as a duplicate.

        Args:
            job_name: The name of the worker function to run.
            job_id: The job's id.
            arguments: The function's keyword arguments.

        Returns:
            Whether the job was queued. `False` means it is queued or running.

        Raises:
            ValueError: If no queue is set for the job name.
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        earlier = Job(job_id, self._redis, _queue_name=queue_of(job_name))
        if await earlier.status() is JobStatus.complete:
            await self._redis.delete(result_key_prefix + job_id)
        return await self.enqueue(job_name, job_id, arguments)

    async def discard(self, job_ids: Sequence[str]) -> None:
        """Remove jobs from every queue, with their retry state and kept results.

        A job a worker is running at this moment finishes. arq then keeps its
        result, which holds only the job's ids.

        Args:
            job_ids: The ids of the jobs.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        if not job_ids:
            return
        async with self._redis.pipeline(transaction=True) as pipeline:
            for queue_name in _ALL_QUEUES:
                pipeline.zrem(queue_name, *job_ids)
            pipeline.delete(
                *(prefix + job_id for job_id in job_ids for prefix in _JOB_KEY_PREFIXES)
            )
            await pipeline.execute()


def create_arq_redis(redis_url: str) -> ArqRedis:
    """Create the Redis client for the job queue and ingestion's state.

    No connection is opened here: the first command opens one.

    Args:
        redis_url: The server's `redis://` URL.

    Returns:
        The client. Close it with `aclose(close_connection_pool=True)`.
    """
    return ArqRedis(ConnectionPool.from_url(redis_url))
