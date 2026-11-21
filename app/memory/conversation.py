"""The conversation and the basic profile in Postgres, for the chat worker.

The chat worker reads the sender's profile before an answer: where to send it
(their platform id), their country and home area. After the answer was sent it
stores the exchange in `conversation_msg` and the language the user wrote in
as `user.language`, which the save confirmations use (ARCHITECTURE §3.7).

The home area and the country are set by the response agent's `set_profile`
tool when the user tells them. Every row and column here belongs to the user
and is deleted with the `user` row by "delete my data".
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ConversationMessage, PlatformAccount, User
from app.ingestion.repository import lock_user
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

    async def record(
        self, user_id: uuid.UUID, messages: Sequence[SessionMessage], language: str | None
    ) -> bool:
        """Store an exchange in `conversation_msg`, and the language the user wrote in.

        Args:
            user_id: The DM's sender.
            messages: The messages, oldest first.
            language: The BCP 47 tag of the user's language in this exchange;
                `None` leaves `user.language` as it is.

        Returns:
            Whether it was stored; `False` if the user was deleted meanwhile.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: If the database failed.
        """
        async with self._sessions() as session, session.begin():
            # Holds "delete my data" off until the rows are in, or finds the user gone.
            if not await lock_user(session, user_id):
                return False
            session.add_all(
                ConversationMessage(
                    user_id=user_id, role=message.role, content=message.content, at=message.at
                )
                for message in messages
            )
            if language is not None:
                await session.execute(
                    update(User).where(User.id == user_id).values(language=language)
                )
        return True
