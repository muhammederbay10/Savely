"""Database work for ingestion: users on first contact, saved items, collection hints.

Collection hints are written by the chat worker once it has decided a text's
route (`app.ingestion.text_routing`), under the same user row lock.

Every function runs inside the caller's transaction and never commits.

Ingestion locks the user's row for the whole of its work on one message
(`lock_user`). "Delete my data" locks the same row, so the two never overlap:
a deletion waits for the message being ingested, and a message waits for the
deletion and then finds no user.

A saved item is stored once per post id and once per message id (decision
record 0007). Both rules are unique keys in the database, so a second share of
a post, or a webhook that Meta delivers twice, finds the item that is already
there instead of creating another.
"""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import ColumnElement, and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import CaptureEvent
from app.db.models import PlatformAccount, SavedItem, User

logger = logging.getLogger(__name__)

# Seed of the 64-bit hash that turns an account into an advisory-lock key.
_LOCK_HASH_SEED = 0


@dataclass(frozen=True)
class StoredCapture:
    """The saved item a capture event belongs to.

    Attributes:
        saved_item_id: The item's id.
        status: The item's place in the indexing pipeline.
        created: Whether this capture created the item. `False` means the
            post, or the message, was already stored.
    """

    saved_item_id: uuid.UUID
    status: str
    created: bool


async def find_or_create_user(
    session: AsyncSession, platform: str, platform_user_id: str
) -> uuid.UUID:
    """Return our user for a platform account, creating both on first contact.

    Two first messages from one account can arrive at the same moment. The
    second one waits for the first and then finds its user, so an account never
    gets two users.

    Args:
        session: The session to work in.
        platform: The platform the account is on.
        platform_user_id: The platform's id of the account.

    Returns:
        The user's id.
    """
    user_id = await _find_user_id(session, platform, platform_user_id)
    if user_id is not None:
        return user_id
    await _lock_first_contact(session, platform, platform_user_id)
    user_id = await _find_user_id(session, platform, platform_user_id)
    if user_id is not None:
        return user_id
    user = User()
    session.add(user)
    await session.flush()
    session.add(
        PlatformAccount(user_id=user.id, platform=platform, platform_user_id=platform_user_id)
    )
    await session.flush()
    logger.info("user_created", extra={"user_id": str(user.id), "platform": platform})
    return user.id


async def store_capture(session: AsyncSession, event: CaptureEvent) -> StoredCapture:
    """Store a capture event as a saved item, once per post and once per message.

    The media link is not stored. It is passed on to the indexing job, which
    downloads the file before the link expires.

    Args:
        session: The session to work in.
        event: The capture event.

    Returns:
        The new item, or the item that already holds this post or message.

    Raises:
        LookupError: If the database refuses the item as a duplicate but no
            item with its post id or message id exists.
    """
    insert_item = (
        insert(SavedItem)
        .values(
            user_id=event.user_id,
            platform=event.platform,
            platform_post_id=event.platform_post_id,
            post_url=event.post_url,
            media_type=event.media_type,
            caption=event.caption,
            collection=event.collection_hint,
            saved_at=event.received_at,
            source_message_id=event.source_message_id,
        )
        .on_conflict_do_nothing()
        .returning(SavedItem.id, SavedItem.status)
    )
    created = (await session.execute(insert_item)).one_or_none()
    if created is not None:
        return StoredCapture(saved_item_id=created.id, status=created.status, created=True)
    existing = await _find_stored_capture(session, event)
    if existing is None:
        raise LookupError("a saved item was refused as a duplicate, but no matching item exists")
    return StoredCapture(saved_item_id=existing.id, status=existing.status, created=False)


async def lock_user(session: AsyncSession, user_id: uuid.UUID) -> bool:
    """Lock a user's row until the transaction ends.

    The lock lets new rows refer to the user, but makes "delete my data" for
    this user wait, and waits for it.

    Args:
        session: The session to work in.
        user_id: The user to lock.

    Returns:
        Whether the user exists. `False` means the user was deleted.
    """
    statement = select(User.id).where(User.id == user_id).with_for_update(key_share=True)
    return (await session.execute(statement)).scalar_one_or_none() is not None


async def is_unlabelled_item(
    session: AsyncSession, user_id: uuid.UUID, saved_item_id: uuid.UUID
) -> bool:
    """Check that a user's saved item exists and has no collection yet.

    Args:
        session: The session to work in.
        user_id: The user the item must belong to.
        saved_item_id: The item's id.

    Returns:
        Whether the item can take a collection hint.
    """
    statement = select(SavedItem.collection).where(
        SavedItem.id == saved_item_id, SavedItem.user_id == user_id
    )
    row = (await session.execute(statement)).one_or_none()
    return row is not None and row.collection is None


async def set_collection_if_missing(
    session: AsyncSession, user_id: uuid.UUID, saved_item_id: uuid.UUID, collection: str
) -> bool:
    """Give a user's saved item its collection, unless it already has one.

    Setting the same hint again changes nothing, so a repeated try is safe.

    Args:
        session: The session to work in.
        user_id: The user the item must belong to.
        saved_item_id: The item's id.
        collection: The collection, as the user wrote it.

    Returns:
        Whether this call set it. `False` means the item has a collection
        already, or is not the user's or is gone.
    """
    result = await session.execute(
        update(SavedItem)
        .where(
            SavedItem.id == saved_item_id,
            SavedItem.user_id == user_id,
            SavedItem.collection.is_(None),
        )
        .values(collection=collection)
        .returning(SavedItem.id)
    )
    return result.first() is not None


async def _find_user_id(
    session: AsyncSession, platform: str, platform_user_id: str
) -> uuid.UUID | None:
    statement = select(PlatformAccount.user_id).where(
        PlatformAccount.platform == platform,
        PlatformAccount.platform_user_id == platform_user_id,
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def _lock_first_contact(session: AsyncSession, platform: str, platform_user_id: str) -> None:
    # Held until the transaction ends. Two accounts whose keys hash to the same
    # number only wait for each other; nothing else depends on the number.
    lock_key = func.hashtextextended(
        f"platform_account:{platform}:{platform_user_id}", _LOCK_HASH_SEED
    )
    await session.execute(select(func.pg_advisory_xact_lock(lock_key)))


async def _find_stored_capture(session: AsyncSession, event: CaptureEvent) -> SavedItem | None:
    # The same two keys as the unique indexes on `saved_item`.
    dedupe_keys: list[ColumnElement[bool]] = []
    if event.platform_post_id is not None:
        dedupe_keys.append(
            and_(
                SavedItem.platform == event.platform,
                SavedItem.platform_post_id == event.platform_post_id,
            )
        )
    if event.source_message_id is not None:
        dedupe_keys.append(SavedItem.source_message_id == event.source_message_id)
    if not dedupe_keys:
        return None
    statement = select(SavedItem).where(SavedItem.user_id == event.user_id, or_(*dedupe_keys))
    return (await session.execute(statement.limit(1))).scalar_one_or_none()
