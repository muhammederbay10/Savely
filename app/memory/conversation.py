"""The conversation and the basic profile in Postgres, for the chat worker.

The chat worker reads the sender's profile before an answer: where to send it
(their platform id), their country and home area. After the answer was sent it
stores the exchange in `conversation_msg` and the language the user wrote in
as `user.language`, which the save confirmations use (ARCHITECTURE §3.7).

The home area and the country are set by the response agent's `set_profile`
tool when the user tells them. Every row and column here belongs to the user
and is deleted with the `user` row by "delete my data".

**Holding the user** (`hold`). The chat worker sends an answer and writes the
exchange anywhere only inside `hold`: a transaction that holds the user's row
with `FOR KEY SHARE`. "Delete my data" locks the same row `FOR UPDATE`, so it
waits until the block ends and then erases what the block wrote; a block that
starts after the erasure finds no user and writes nothing. `FOR KEY SHARE`
doesn't make ingestion wait, nor another answer of the same user.

An exchange's rows get ids made from the message's receipt and their place in
the exchange, so storing the same exchange again adds nothing.
"""

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ConversationMessage, PlatformAccount, User
from app.memory.session import SessionMessage


@dataclass(frozen=True)
class ChatProfile:
    """What the chat worker needs to know about a sender.

    Attributes:
        recipient_id: The user's id on the platform they wrote on, e.g. the IGSID.
        country: ISO 3166-1 alpha-2; `None` if not known yet.
        home_area: Where they live, in their own words; `None` if not known.
        language: The BCP 47 tag of the language they last wrote in; `None`
            before their first answer.
    """

    recipient_id: str = field(repr=False)
    country: str | None
    home_area: str | None = field(repr=False)
    language: str | None


class ConversationStore:
    """Reads the sender's profile and stores the conversation, in Postgres."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        """Create the store.

        Args:
            sessions: Opens a database session.
        """
        self._sessions = sessions

    async def profile(self, user_id: uuid.UUID, platform: str) -> ChatProfile | None:
        """Return the sender's profile and their id on a platform.

        Args:
            user_id: The DM's sender.
            platform: The platform they wrote on.

        Returns:
            The profile; `None` if the user, or their account on that
            platform, no longer exists.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database failed.
        """
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(
                        PlatformAccount.platform_user_id,
                        User.country,
                        User.home_area,
                        User.language,
                    )
                    .join(User, User.id == PlatformAccount.user_id)
                    .where(PlatformAccount.user_id == user_id, PlatformAccount.platform == platform)
                )
            ).first()
        if row is None:
            return None
        recipient_id, country, home_area, language = row
        return ChatProfile(
            recipient_id=recipient_id, country=country, home_area=home_area, language=language
        )

    @asynccontextmanager
    async def hold(self, user_id: uuid.UUID) -> AsyncIterator["HeldUser | None"]:
        """Hold the user's row until the block ends, so "delete my data" waits for it.

        Args:
            user_id: The DM's sender.

        Yields:
            The held user, to store the exchange with; `None` if the user was
            deleted, and then nothing may be sent or written for them.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database failed; what the
                block wrote to Postgres is rolled back.
        """
        async with self._sessions() as session, session.begin():
            held = await session.execute(
                select(User.id).where(User.id == user_id).with_for_update(read=True, key_share=True)
            )
            yield None if held.scalar_one_or_none() is None else HeldUser(session, user_id)


class HeldUser:
    """A user whose row `ConversationStore.hold` holds; writes commit when the block ends."""

    def __init__(self, session: AsyncSession, user_id: uuid.UUID) -> None:
        """Wrap the holding transaction; use `ConversationStore.hold`.

        Args:
            session: The session whose transaction holds the user's row.
            user_id: The user.
        """
        self._session = session
        self._user_id = user_id

    async def record(
        self, receipt_id: uuid.UUID, messages: Sequence[SessionMessage], language: str | None
    ) -> None:
        """Store an exchange in `conversation_msg`, and the language the user wrote in.

        Storing the same exchange again adds no row.

        Args:
            receipt_id: The receipt of the message the exchange answers.
            messages: The messages, oldest first.
            language: The BCP 47 tag of the user's language in this exchange;
                `None` leaves `user.language` as it is.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database failed.
        """
        if messages:
            await self._session.execute(
                insert(ConversationMessage)
                .values(
                    [
                        {
                            "id": uuid.uuid5(receipt_id, str(position)),
                            "user_id": self._user_id,
                            "role": message.role,
                            "content": message.content,
                            "at": message.at,
                        }
                        for position, message in enumerate(messages)
                    ]
                )
                .on_conflict_do_nothing(index_elements=[ConversationMessage.id])
            )
        if language is not None:
            await self._session.execute(
                update(User).where(User.id == self._user_id).values(language=language)
            )
