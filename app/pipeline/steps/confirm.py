"""Step 7, confirm: tell the user how the item was saved, in their language.

The confirmation says the item's open category ("Saved as car review"), also
for an item with no built-in type, so the user can correct it
(`docs/ARCHITECTURE.md` §3.3 and §3.4). Names found by a resolver come with SAV-15.

It is written in the language of the user's profile, which the chat worker
sets to the language the user last wrote in. Until a language is known, or
when that language has no wording yet, it is written in English, the owner's
choice on SAV-14 (18 Nov 2026); the language is never guessed from the post.
The output records which happened.
A failed send is recorded and not tried again, because a message whose
delivery is unknown would be sent twice.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, Protocol

from app.agent.replies.messenger import MessengerError, OutgoingMessage, SendResult, SendStatus
from app.agent.replies.templates import (
    ReplyWording,
    ReplyWordingUnavailableError,
    SavedItemSummary,
    reply_wording,
    save_confirmation,
)

logger = logging.getLogger(__name__)

# The language of the confirmation while the user's own is unknown or has no wording.
FALLBACK_LANGUAGE: Final = "en"


class Messenger(Protocol):
    """Sends a DM to a user on the platform an item came from."""

    async def send(
        self, recipient_id: str, message: OutgoingMessage, last_user_message_at: datetime
    ) -> SendResult:
        """Send one message, unless the platform's reply window has closed.

        Args:
            recipient_id: The user's id on the platform.
            message: The message.
            last_user_message_at: When the user last wrote to the bot.

        Returns:
            Whether it was sent or skipped for the window.

        Raises:
            MessengerError: If it could not be sent.
        """
        ...


class ConfirmationStatus(StrEnum):
    """What happened to the confirmation."""

    SENT = "sent"
    OUTSIDE_REPLY_WINDOW = "outside_reply_window"
    SKIPPED = "skipped"
    FAILED = "failed"


class SkipReason(StrEnum):
    """Why the confirmation was not sent."""

    NO_RECIPIENT = "no_recipient"


class LanguageChoice(StrEnum):
    """Which language the confirmation was written in."""

    USER = "user"
    FALLBACK_NO_LANGUAGE = "fallback_no_language"
    FALLBACK_NO_WORDING = "fallback_no_wording"


@dataclass(frozen=True)
class Confirmation:
    """The result of the confirm step.

    Attributes:
        status: What happened.
        reason: Why it was skipped, or the error's class when it failed.
        language: The language it was written in, when one was chosen.
        language_choice: Whether that is the user's language or the fallback.
    """

    status: ConfirmationStatus
    reason: str | None = None
    language: str | None = None
    language_choice: LanguageChoice | None = None

    def to_output(self) -> dict[str, object]:
        """Return the step's output for `enrichment_run.output`.

        Returns:
            A JSON object with every attribute.
        """
        return {
            "status": self.status,
            "reason": self.reason,
            "language": self.language,
            "language_choice": self.language_choice,
        }


class SaveConfirmer:
    """Sends the save confirmation."""

    def __init__(self, messenger: Messenger) -> None:
        """Create the step.

        Args:
            messenger: Sends the DM.
        """
        self._messenger = messenger

    async def confirm(
        self,
        *,
        recipient_id: str | None,
        language: str | None,
        category: str,
        last_user_message_at: datetime,
    ) -> Confirmation:
        """Tell the user how their item was saved.

        Args:
            recipient_id: The user's id on the item's platform.
            language: The user's language from their profile; `None` while unknown.
            category: The item's open category.
            last_user_message_at: When the user sent the share.

        Returns:
            Whether the confirmation was sent, skipped or failed.
        """
        if recipient_id is None:
            return Confirmation(ConfirmationStatus.SKIPPED, SkipReason.NO_RECIPIENT)
        used_language, choice, wording = _wording_for(language)
        message = save_confirmation(SavedItemSummary(category=category), wording)
        try:
            result = await self._messenger.send(recipient_id, message, last_user_message_at)
        except MessengerError as error:
            logger.warning("save_confirmation_failed", extra={"error": type(error).__name__})
            return Confirmation(
                ConfirmationStatus.FAILED, type(error).__name__, used_language, choice
            )
        status = (
            ConfirmationStatus.OUTSIDE_REPLY_WINDOW
            if result.status is SendStatus.OUTSIDE_REPLY_WINDOW
            else ConfirmationStatus.SENT
        )
        return Confirmation(status, None, used_language, choice)


def _wording_for(language: str | None) -> tuple[str, LanguageChoice, ReplyWording]:
    if language is None:
        return FALLBACK_LANGUAGE, LanguageChoice.FALLBACK_NO_LANGUAGE, _fallback_wording()
    try:
        return language, LanguageChoice.USER, reply_wording(language)
    except ReplyWordingUnavailableError:
        return FALLBACK_LANGUAGE, LanguageChoice.FALLBACK_NO_WORDING, _fallback_wording()


def _fallback_wording() -> ReplyWording:
    return reply_wording(FALLBACK_LANGUAGE)
