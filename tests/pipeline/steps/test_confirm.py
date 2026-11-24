"""The save confirmation: the user's language when known, English until then."""

from datetime import datetime

import pytest

from app.agent.replies.messenger import (
    MessengerDeliveryUnknownError,
    OutgoingMessage,
    SendResult,
    SendStatus,
)
from app.pipeline.steps.confirm import (
    FALLBACK_LANGUAGE,
    Confirmation,
    ConfirmationStatus,
    LanguageChoice,
    SaveConfirmer,
    SkipReason,
)
from tests.pipeline.support import SHARED_AT, RecordingMessenger

RECIPIENT = "1000000000000141"


async def confirm(
    messenger: RecordingMessenger,
    *,
    language: str | None,
    recipient: str | None = RECIPIENT,
    collection: str | None = None,
) -> Confirmation:
    return await SaveConfirmer(messenger).confirm(
        recipient_id=recipient,
        language=language,
        category="car review",
        collection=collection,
        last_user_message_at=SHARED_AT,
    )


@pytest.mark.parametrize(
    ("language", "text", "used", "choice"),
    [
        ("tr", "car review olarak kaydedildi", "tr", LanguageChoice.USER),
        ("en-GB", "Saved as car review", "en-GB", LanguageChoice.USER),
        (None, "Saved as car review", FALLBACK_LANGUAGE, LanguageChoice.FALLBACK_NO_LANGUAGE),
        ("ja", "Saved as car review", FALLBACK_LANGUAGE, LanguageChoice.FALLBACK_NO_WORDING),
    ],
)
async def test_the_confirmation_uses_the_users_language_or_english_until_it_is_known(
    language: str | None, text: str, used: str, choice: LanguageChoice
) -> None:
    messenger = RecordingMessenger()

    confirmation = await confirm(messenger, language=language)

    assert messenger.sent == [(RECIPIENT, text)]
    assert confirmation.to_output() == {
        "status": ConfirmationStatus.SENT,
        "reason": None,
        "language": used,
        "language_choice": choice,
    }


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "Saved as car review · in Arabalar"),
        ("tr", "car review olarak kaydedildi · Arabalar koleksiyonunda"),
    ],
)
async def test_a_collection_the_user_named_is_in_the_one_confirmation(
    language: str, text: str
) -> None:
    messenger = RecordingMessenger()

    await confirm(messenger, language=language, collection="Arabalar")

    assert messenger.sent == [(RECIPIENT, text)]


async def test_without_an_account_on_the_platform_nothing_is_sent() -> None:
    messenger = RecordingMessenger()

    confirmation = await confirm(messenger, language="en", recipient=None)

    assert messenger.sent == []
    assert confirmation.status is ConfirmationStatus.SKIPPED
    assert confirmation.reason == SkipReason.NO_RECIPIENT


class FailingMessenger(RecordingMessenger):
    async def send(
        self, recipient_id: str, message: OutgoingMessage, last_user_message_at: datetime
    ) -> SendResult:
        raise MessengerDeliveryUnknownError(
            "synthetic timeout", status=None, error_code=None, error_subcode=None, attempts=1
        )


class ClosedWindowMessenger(RecordingMessenger):
    async def send(
        self, recipient_id: str, message: OutgoingMessage, last_user_message_at: datetime
    ) -> SendResult:
        return SendResult(status=SendStatus.OUTSIDE_REPLY_WINDOW, message_id=None, attempts=0)


async def test_a_failed_send_is_recorded_and_not_tried_again() -> None:
    confirmation = await confirm(FailingMessenger(), language="en")

    assert confirmation.status is ConfirmationStatus.FAILED
    assert confirmation.reason == "MessengerDeliveryUnknownError"


async def test_a_closed_reply_window_is_recorded() -> None:
    confirmation = await confirm(ClosedWindowMessenger(), language="en")

    assert confirmation.status is ConfirmationStatus.OUTSIDE_REPLY_WINDOW
