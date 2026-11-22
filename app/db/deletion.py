"""Delete my data: remove one user's rows and lower the vocabulary's item counts.

The foreign keys delete a user's rows when the `user` row goes. The vocabulary
has no reference to a user, so its `item_count` has to be lowered here, before
the user's items are gone.

The user row is locked first. A second request for the same user then waits for
the first one to finish and finds no user, so the counts are lowered only once.
Ingestion locks the same row while it handles a message, so no message of the
user is half-stored while the user is deleted.

What the user left outside Postgres (ingestion's job payloads and receipts in
Redis, a pending deletion request, short-term memory, stored media files) is
erased through `UserDataOutsidePostgres`, while the lock is held.

`UserDataEraser` is the one place that erases a user everywhere: every store
outside Postgres, in order, then the rows. The chat worker builds it with
every such store (`app.agent.worker.build_toolbox`); a new store of user data
joins that list.
"""

import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    CATEGORY_VOCABULARY_FIELD,
    SUBTYPE_VOCABULARY_FIELD,
    TAG_VOCABULARY_FIELD,
    SavedItem,
    User,
)

logger = logging.getLogger(__name__)

# One row per (item, field, value) the user's items use: the category, the
# subtype, each tag, and each open value in `fields` (a string, or the strings
# in a list). Wording that never entered the vocabulary matches no row there,
# so it changes nothing. A count never goes below zero.
_LOWER_ITEM_COUNTS = text(
    """
    WITH item_values AS (
        SELECT id AS item_id, CAST(:category_field AS text) AS field, category AS value
        FROM saved_item
        WHERE user_id = :user_id AND category IS NOT NULL
        UNION
        SELECT id, CAST(:subtype_field AS text), subtype
        FROM saved_item
        WHERE user_id = :user_id AND subtype IS NOT NULL
        UNION
        SELECT id, CAST(:tag_field AS text), unnest(tags)
        FROM saved_item
        WHERE user_id = :user_id
        UNION
        SELECT saved_item.id, open_field.key, open_value.value
        FROM saved_item
        CROSS JOIN LATERAL jsonb_each(saved_item.fields) AS open_field
        CROSS JOIN LATERAL (
            SELECT open_field.value #>> '{}' AS value
            WHERE jsonb_typeof(open_field.value) = 'string'
            UNION ALL
            SELECT jsonb_array_elements_text(open_field.value)
            WHERE jsonb_typeof(open_field.value) = 'array'
        ) AS open_value
        WHERE saved_item.user_id = :user_id
    ),
    uses AS (
        SELECT field, value, count(*) AS item_count
        FROM item_values
        GROUP BY field, value
    )
    UPDATE vocabulary
    SET item_count = greatest(vocabulary.item_count - uses.item_count, 0)
    FROM uses
    WHERE vocabulary.field = uses.field AND vocabulary.value = uses.value
    """
)


class UserDataOutsidePostgres(Protocol):
    """Erases what a user left in stores other than Postgres."""

    async def erase_user(self, user_id: uuid.UUID, saved_item_ids: Sequence[uuid.UUID]) -> None:
        """Erase everything one user left outside Postgres.

        Args:
            user_id: The user.
            saved_item_ids: The ids of the user's saved items.
        """
        ...


async def delete_user_data(
    session: AsyncSession, user_id: uuid.UUID, outside_postgres: UserDataOutsidePostgres
) -> int | None:
    """Delete a user and every database row that belongs to them.

    The user's saved items go with their raw wording, entity links and
    enrichment runs, as do platform accounts, entity statuses and conversation
    messages. Each vocabulary value loses one from its `item_count` for every
    deleted item that used it. The shared tables keep their rows: entities,
    streaming availability, vocabulary values and aliases.

    What the user left outside Postgres is erased first, through
    `outside_postgres`. If that fails, nothing is deleted from Postgres either,
    and the request can be repeated.

    The caller commits the session. The user row stays locked until then, so a
    second request for the same user waits and then changes nothing. Nothing
    happens for an unknown user id either. Stored media files are not touched.

    Args:
        session: The database session to run the statements on.
        user_id: The id of the user to delete.
        outside_postgres: Erases the user's data in the other stores, such as
            the job payloads and receipts ingestion keeps in Redis.

    Returns:
        How many saved items the user had; `None` if there is no such user.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If the database rejects a statement.
    """
    # Without this lock, a second request that starts before the first one
    # commits still sees the user's items and lowers the counts a second time.
    if not await _lock_user_row(session, user_id):
        return None
    saved_item_ids = list(
        (await session.execute(select(SavedItem.id).where(SavedItem.user_id == user_id))).scalars()
    )
    await outside_postgres.erase_user(user_id, saved_item_ids)
    await session.execute(
        _LOWER_ITEM_COUNTS,
        {
            "user_id": user_id,
            "category_field": CATEGORY_VOCABULARY_FIELD,
            "subtype_field": SUBTYPE_VOCABULARY_FIELD,
            "tag_field": TAG_VOCABULARY_FIELD,
        },
    )
    await session.execute(delete(User).where(User.id == user_id))
    return len(saved_item_ids)


class ErasureDeadlinePassedError(Exception):
    """The deadline passed while the erasure waited for the user's row lock; nothing was erased."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _lock_user_row(session: AsyncSession, user_id: uuid.UUID) -> bool:
    """Lock the user's row for deletion; return whether the user exists.

    Ingestion and indexing hold the same row while they work, so this can wait.
    """
    locked_user = await session.execute(select(User.id).where(User.id == user_id).with_for_update())
    return locked_user.scalar_one_or_none() is not None


class UserDataEraser:
    """Erases one user everywhere: every store outside Postgres, then their rows.

    The stores outside Postgres are erased first, in the given order, while
    the user's row is locked. If one fails, the transaction is rolled back, so
    Postgres is unchanged and the whole erasure can be repeated: every step
    removes what is still there and nothing else.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        outside_postgres: Sequence[UserDataOutsidePostgres],
    ) -> None:
        """Create the eraser.

        Args:
            sessions: Opens a database session.
            outside_postgres: Every store outside Postgres that keeps user
                data, e.g. ingestion's Redis state and jobs, and the media store.
        """
        self._sessions = sessions
        self._outside_postgres = tuple(outside_postgres)

    async def erase(
        self,
        user_id: uuid.UUID,
        *,
        deadline: datetime | None = None,
        now: Callable[[], datetime] = _utc_now,
    ) -> int | None:
        """Erase one user and everything they left anywhere.

        Args:
            user_id: The user.
            deadline: When set, the erasure runs only if it is still before this
                moment once the user's row is locked, checked before any store
                is erased. Taking the lock can wait for ingestion or indexing.
            now: The clock for the deadline.

        Returns:
            How many saved items were deleted; `None` if there is no such user.

        Raises:
            ErasureDeadlinePassedError: If the deadline passed while waiting for
                the lock; nothing was erased anywhere.
            redis.exceptions.RedisError: If Redis failed; nothing in Postgres changed.
            OSError: If the media store failed; nothing in Postgres changed.
            sqlalchemy.exc.SQLAlchemyError: If the database failed.
        """
        async with self._sessions() as session, session.begin():
            if deadline is not None:
                # `delete_user_data` takes the same lock again in this transaction,
                # so nothing can change between this check and the erasure.
                if not await _lock_user_row(session, user_id):
                    return None
                if now() >= deadline:
                    raise ErasureDeadlinePassedError("the deletion deadline passed")
            deleted_items = await delete_user_data(session, user_id, self)
        if deleted_items is not None:
            # Counts only: the user is gone, and nothing about them is logged.
            logger.info("user_data_deleted", extra={"saved_items": deleted_items})
        return deleted_items

    async def erase_user(self, user_id: uuid.UUID, saved_item_ids: Sequence[uuid.UUID]) -> None:
        """Erase the user from every store outside Postgres, in order.

        Args:
            user_id: The user.
            saved_item_ids: The user's saved items.
        """
        for store in self._outside_postgres:
            await store.erase_user(user_id, saved_item_ids)
