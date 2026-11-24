"""A text right after a share: the decision model decides if it names a collection (record 0018).

The chat worker settles the route before the response agent runs. The decision
model is a fake that answers a scripted probability; the text model is
scripted and Meta's Send API is a recording mock. These tests need Postgres
(`TEST_DATABASE_ADMIN_URL`) and Redis (`TEST_REDIS_URL`). Every id and text is
synthetic.
"""

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select, update

from app.adapters.base import CaptureEvent
from app.agent.chat import ChatOutcome, RetryLaterError
from app.agent.replies.templates import reply_wording
from app.db.deletion import UserDataEraser
from app.db.models import EnrichmentRun, SavedItem, User
from app.ingestion.service import IngestionOutcome
from app.ingestion.session_state import TextReceipt, TextRoute
from app.ingestion.text_routing import COLLECTION_NAME_QUESTION_KEY
from app.models.base import Answer, Unanswered, YesNoAnswer
from app.models.errors import ModelUnavailableError
from app.models.fakes import FakeClassifier
from app.pipeline.recovery.runs import RunStatus, Step
from tests.agent.agent_support import final_answer
from tests.agent.chat_support import (
    HINT_MIN_PROBABILITY,
    HINT_WINDOW,
    ChatStores,
    ChatUnderTest,
    build_chat,
    stored_messages,
)
from tests.agent.support import json_response
from tests.db_factories import INSTAGRAM

SENDER_ID = "1000000000000391"
# The gap between a share and its hint in the recorded webhooks (record 0002).
RIGHT_AFTER = timedelta(seconds=9)
QUESTION = "What did i save?"
COLLECTION = "Filmler"
NAMES_IT = YesNoAnswer(probability_yes=0.97, confidence=None)
DOES_NOT = YesNoAnswer(probability_yes=0.03, confidence=None)
UNSURE = YesNoAnswer(probability_yes=HINT_MIN_PROBABILITY - 0.01, confidence=None)
MODEL_DOWN = ModelUnavailableError("synthetic outage", operation="decide", model="fake/decision")


def routing_model(answer: Answer) -> FakeClassifier:
    return FakeClassifier({COLLECTION_NAME_QUESTION_KEY: answer})


def asked_texts(model: FakeClassifier) -> list[str]:
    """What the decision model was shown, one entry per call."""
    return [str(state[0]) for state, _ in model.calls]


async def item_of(stores: ChatStores, share: CaptureEvent) -> SavedItem:
    async with stores.sessions() as session:
        statement = select(SavedItem).where(SavedItem.source_message_id == share.source_message_id)
        return (await session.execute(statement)).scalar_one()


async def receipt_for(stores: ChatStores, user_id: uuid.UUID, receipt_id: uuid.UUID) -> TextReceipt:
    message = stores.delivered[receipt_id]
    receipt = await stores.state.text_receipt(user_id, INSTAGRAM, message.message_id)
    assert receipt is not None
    return receipt


async def start_save_confirmation(stores: ChatStores, item: SavedItem) -> None:
    """What the indexer writes, under the owner's lock, right before sending the confirmation."""
    async with stores.sessions.begin() as session:
        session.add(
            EnrichmentRun(
                saved_item_id=item.id,
                step=Step.CONFIRM,
                status=RunStatus.STARTED,
                output={"collection_included": False},
            )
        )


async def set_language(stores: ChatStores, user_id: uuid.UUID, language: str) -> None:
    async with stores.sessions.begin() as session:
        await session.execute(update(User).where(User.id == user_id).values(language=language))


async def text_right_after(
    stores: ChatStores, user_id: uuid.UUID, text: str, *, after: timedelta = RIGHT_AFTER
) -> tuple[CaptureEvent, uuid.UUID]:
    share = await stores.share(user_id)
    receipt = await stores.send_text(user_id, text, sent_at=share.received_at + after)
    return share, receipt


def answering(
    stores: ChatStores, model: FakeClassifier, reply: str = "Here you go."
) -> ChatUnderTest:
    return build_chat(stores, [final_answer("en", reply)], routing_model=model)


# --- Routing --------------------------------------------------------------------------------


async def test_a_question_right_after_a_share_is_answered(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(DOES_NOT)
    chat = answering(chat_stores, model)
    share, receipt = await text_right_after(chat_stores, user_id, QUESTION)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert chat.sent_texts() == ["Here you go."]
    assert (await item_of(chat_stores, share)).collection is None
    assert asked_texts(model) == [f"Message sent right after sharing a post: {QUESTION}"]
    assert (await receipt_for(chat_stores, user_id, receipt)).route is TextRoute.CHAT


async def test_a_collection_name_right_after_a_share_is_stored_for_the_save_confirmation(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = answering(chat_stores, routing_model(NAMES_IT))
    share, receipt = await text_right_after(chat_stores, user_id, COLLECTION)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    # The save confirmation hasn't started: it will say the collection, so nothing is sent now.
    stored = await receipt_for(chat_stores, user_id, receipt)
    assert outcome is ChatOutcome.COLLECTION_HINT
    assert chat.sent_texts() == []
    assert (await item_of(chat_stores, share)).collection == COLLECTION
    assert (stored.route, stored.confirm_collection) == (TextRoute.COLLECTION_HINT, False)
    assert chat.agent.generator.requests == []
    assert await stored_messages(chat_stores, user_id) == []


async def test_a_name_after_the_save_was_confirmed_is_confirmed_once_in_the_users_language(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    await set_language(chat_stores, user_id, "tr")
    chat = answering(chat_stores, routing_model(NAMES_IT))
    share, receipt = await text_right_after(chat_stores, user_id, COLLECTION)
    await start_save_confirmation(chat_stores, await item_of(chat_stores, share))

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.COLLECTION_CONFIRMED
    assert chat.sent_texts() == [reply_wording("tr").saved_to.format(collection=COLLECTION)]
    assert (await item_of(chat_stores, share)).collection == COLLECTION


async def test_a_user_without_a_known_language_is_confirmed_in_english(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = answering(chat_stores, routing_model(NAMES_IT))
    share, receipt = await text_right_after(chat_stores, user_id, "Food")
    await start_save_confirmation(chat_stores, await item_of(chat_stores, share))

    await chat.responder.respond(user_id, receipt, last_try=False)

    assert chat.sent_texts() == ["Saved to Food"]


async def test_a_text_after_the_time_limit_makes_no_model_call(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(NAMES_IT)
    chat = answering(chat_stores, model)
    share, receipt = await text_right_after(
        chat_stores, user_id, COLLECTION, after=HINT_WINDOW + timedelta(seconds=1)
    )

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert model.calls == []
    assert (await item_of(chat_stores, share)).collection is None


async def test_a_text_with_no_share_before_it_makes_no_model_call(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(NAMES_IT)
    chat = answering(chat_stores, model)
    receipt = await chat_stores.send_text(user_id, COLLECTION)

    assert await chat.responder.respond(user_id, receipt, last_try=False) is ChatOutcome.ANSWERED
    assert model.calls == []


# --- Unsure or failed decisions go to chat ----------------------------------------------


async def test_an_unsure_decision_goes_to_chat(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = answering(chat_stores, routing_model(UNSURE))
    share, receipt = await text_right_after(chat_stores, user_id, "Food?")

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert chat.sent_texts() == ["Here you go."]
    assert (await item_of(chat_stores, share)).collection is None


async def test_an_unanswered_question_goes_to_chat(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    chat = answering(chat_stores, routing_model(Unanswered("refused")))
    share, receipt = await text_right_after(chat_stores, user_id, COLLECTION)

    assert await chat.responder.respond(user_id, receipt, last_try=False) is ChatOutcome.ANSWERED
    assert (await item_of(chat_stores, share)).collection is None


async def test_a_failed_decision_goes_to_chat_and_is_kept(chat_stores: ChatStores) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = FakeClassifier(error=MODEL_DOWN)
    chat = answering(chat_stores, model)
    share, receipt = await text_right_after(chat_stores, user_id, QUESTION)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert chat.sent_texts() == ["Here you go."]
    assert (await item_of(chat_stores, share)).collection is None
    assert (await receipt_for(chat_stores, user_id, receipt)).route is TextRoute.CHAT
    assert len(model.calls) == 1


# --- Repeated deliveries and retries follow the stored route ----------------------------


async def test_a_redelivered_collection_name_is_routed_the_same_way_and_confirmed_once(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(NAMES_IT)
    chat = answering(chat_stores, model)
    share, receipt = await text_right_after(chat_stores, user_id, COLLECTION)
    await start_save_confirmation(chat_stores, await item_of(chat_stores, share))
    first = await chat.responder.respond(user_id, receipt, last_try=False)

    redelivered = await chat_stores.ingestion.ingest_chat_message(chat_stores.delivered[receipt])
    again = await chat.responder.respond(user_id, receipt, last_try=True)

    assert (first, again) == (ChatOutcome.COLLECTION_CONFIRMED, ChatOutcome.ALREADY_SENT)
    assert redelivered is IngestionOutcome.CHAT_ALREADY_QUEUED
    assert chat.sent_texts() == ["Saved to Filmler"]
    assert len(model.calls) == 1


async def test_a_retried_question_keeps_its_chat_route_without_asking_again(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(DOES_NOT)
    agent_down = ModelUnavailableError("synthetic", operation="generate", model="fake/text")
    chat = build_chat(
        chat_stores, [agent_down, final_answer("en", "Here you go.")], routing_model=model
    )
    _, receipt = await text_right_after(chat_stores, user_id, QUESTION)
    with pytest.raises(RetryLaterError):
        await chat.responder.respond(user_id, receipt, last_try=False)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert chat.sent_texts() == ["Here you go."]
    assert len(model.calls) == 1


async def test_a_decided_hint_whose_commit_was_lost_is_labelled_on_the_next_try(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(NAMES_IT)
    chat = answering(chat_stores, model)
    share, receipt = await text_right_after(chat_stores, user_id, COLLECTION)
    item = await item_of(chat_stores, share)
    message = chat_stores.delivered[receipt]
    bound = await receipt_for(chat_stores, user_id, receipt)
    # The receipt reached Redis, but the database rolled back the collection.
    await chat_stores.state.save_text_receipt(
        user_id,
        INSTAGRAM,
        message.message_id,
        bound.model_copy(update={"route": TextRoute.COLLECTION_HINT}),
    )
    await start_save_confirmation(chat_stores, item)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.COLLECTION_CONFIRMED
    assert (await item_of(chat_stores, share)).collection == COLLECTION
    assert model.calls == []


async def test_a_second_name_for_an_already_labelled_share_is_chat_without_a_model_call(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(NAMES_IT)
    chat = build_chat(chat_stores, [final_answer("en", "Noted.")], routing_model=model)
    share, first = await text_right_after(chat_stores, user_id, COLLECTION)
    second = await chat_stores.send_text(
        user_id, "Movies", sent_at=share.received_at + RIGHT_AFTER + timedelta(seconds=2)
    )
    await chat.responder.respond(user_id, first, last_try=False)

    outcome = await chat.responder.respond(user_id, second, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    assert (await item_of(chat_stores, share)).collection == COLLECTION
    assert len(model.calls) == 1


async def test_a_failed_collection_confirmation_is_not_sent_again(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    refused = json_response("synthetic_error_invalid_parameter.json", 400)
    chat = build_chat(
        chat_stores, [], send_answers=[refused], routing_model=routing_model(NAMES_IT)
    )
    share, receipt = await text_right_after(chat_stores, user_id, COLLECTION)
    await start_save_confirmation(chat_stores, await item_of(chat_stores, share))

    first = await chat.responder.respond(user_id, receipt, last_try=False)
    again = await chat.responder.respond(user_id, receipt, last_try=True)

    assert (first, again) == (ChatOutcome.SEND_FAILED, ChatOutcome.ALREADY_SENT)
    assert len(chat.send_api.requests) == 1
    assert (await item_of(chat_stores, share)).collection == COLLECTION


# --- Deletion -----------------------------------------------------------------------------


async def test_a_text_of_a_user_deleted_before_its_job_asks_no_model_and_labels_nothing(
    chat_stores: ChatStores,
) -> None:
    user_id = await chat_stores.new_user(SENDER_ID)
    model = routing_model(NAMES_IT)
    chat = answering(chat_stores, model)
    _, receipt = await text_right_after(chat_stores, user_id, COLLECTION)
    await UserDataEraser(chat_stores.sessions, chat_stores.outside_postgres()).erase(user_id)

    outcome = await chat.responder.respond(user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.NO_MESSAGE
    assert model.calls == []
    assert chat.sent_texts() == []
