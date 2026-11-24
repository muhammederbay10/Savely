"""The whole pipeline on fake models: one test per kind of post, plus failures, retries and logs.

These tests need Postgres with PostGIS and pgvector (see `tests/database.py`).
Every caption, transcript and name is synthetic.
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.replies.messenger import OutgoingMessage, SendResult
from app.db.models import EnrichmentRun, ItemStatus, SavedItem, Vocabulary
from app.logging_config import configure_logging
from app.models.base import TranscriptionAttempt, YesNoAnswer
from app.models.errors import ModelUnavailableError
from app.pipeline.indexer import IndexOutcome, RetryLaterError
from app.pipeline.media.tools import MediaProbe
from app.pipeline.recovery.runs import Step, confirmation_started
from app.pipeline.steps.classify import NONE_OF_THESE
from app.types.registry import UnknownContentTypeError
from tests.pipeline.harness import Pipeline, add_item, add_user, load
from tests.pipeline.support import (
    ESCALATION_MODEL,
    FALLBACK_STT_MODEL,
    IMAGE_BYTES,
    MediaServer,
    RecordingMessenger,
    capture_event,
    movie_extraction,
    open_extraction,
    place_extraction,
    structured_reply,
    study_extraction,
    transcript,
)

SILENT = MediaProbe(has_video=True, has_audio=False, duration_seconds=9)
IMAGE = MediaProbe(has_video=True, has_audio=False, duration_seconds=None)


@pytest.fixture
def pipeline(session_factory: async_sessionmaker[AsyncSession], tmp_path: Path) -> Pipeline:
    return Pipeline(sessions=session_factory, media_root=tmp_path / "media")


async def runs_of(pipeline: Pipeline, item_id: object) -> dict[str, EnrichmentRun]:
    async with pipeline.sessions() as session:
        rows = await session.execute(
            select(EnrichmentRun).where(EnrichmentRun.saved_item_id == item_id)
        )
        return {run.step: run for run in rows.scalars()}


async def vocabulary_values(pipeline: Pipeline, field: str) -> dict[str, int]:
    async with pipeline.sessions() as session:
        rows = await session.execute(
            select(Vocabulary.value, Vocabulary.item_count).where(Vocabulary.field == field)
        )
        return {row.value: row.item_count for row in rows}


async def test_a_place_reel_is_fetched_transcribed_extracted_and_normalized(
    pipeline: Pipeline,
) -> None:
    pipeline.classifier.content_type = "place"
    pipeline.script(
        transcripts=[transcript("Bebek'teki Kronotrop'a gittik")],
        extractions=[
            structured_reply(
                place_extraction(("Kronotrop", "Specialty Coffee"), category="Café", tags=["View"])
            )
        ],
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(
        pipeline.sessions,
        pipeline,
        capture_event(user_id, caption="Kahve 📍Bebek"),
        collection="Food",
    )

    outcome = await pipeline.indexer().index(user_id, item_id)

    assert outcome is IndexOutcome.INDEXED
    item = await load(pipeline.sessions, item_id)
    assert item.status == ItemStatus.DONE
    assert item.type == "place"
    assert item.transcript == "Bebek'teki Kronotrop'a gittik"
    assert item.category == "café"
    assert item.tags == ["view"]
    assert item.fields["cuisine"] == ["specialty coffee"]
    assert item.fields["entities"][0]["name"] == "Kronotrop"
    assert item.raw_values["category"] == {
        "field": "category",
        "wording": "Café",
        "user_set": False,
    }
    assert item.embedding is not None
    assert set(await runs_of(pipeline, item_id)) == {step.value for step in Step}
    assert await vocabulary_values(pipeline, "cuisine") == {"specialty coffee": 1}
    # The collection the user named for the post is part of its one confirmation.
    assert pipeline.messenger.sent == [(f"1000000000000141-{user_id}", "Saved as café · in Food")]


async def test_a_movie_list_post_keeps_every_movie_and_skips_speech_to_text(
    pipeline: Pipeline,
) -> None:
    pipeline.classifier.content_type = "movie"
    pipeline.classifier.is_list_probability = 0.97
    pipeline.media_server = MediaServer(IMAGE_BYTES, "image/jpeg")
    pipeline.tools.probe_result = IMAGE
    pipeline.script(
        extractions=[structured_reply(movie_extraction("Inception", "Dune", "Arrival"))]
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(
        pipeline.sessions, pipeline, capture_event(user_id, caption="3 films", media_type="post")
    )

    await pipeline.indexer().index(user_id, item_id)

    item = await load(pipeline.sessions, item_id)
    assert [entity["title"] for entity in item.fields["entities"]] == [
        "Inception",
        "Dune",
        "Arrival",
    ]
    assert pipeline.primary_stt.calls == []
    assert pipeline.escalation_model.requests == []
    runs = await runs_of(pipeline, item_id)
    assert runs["classify"].output["is_list"] is True


async def test_a_silent_reel_skips_speech_to_text_and_escalates_to_the_video_model(
    pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline.classifier.content_type = "place"
    # The decision model would claim speech in a file with no audio track; it is never asked.
    pipeline.classifier.extra_answers = {
        "has_speech": YesNoAnswer(probability_yes=0.86, confidence=None)
    }
    pipeline.tools.probe_result = SILENT
    pipeline.script(
        extractions=[structured_reply(place_extraction(category="cafe"))],
        escalations=[
            structured_reply(
                place_extraction(("Sign Café", None), category="cafe"),
                model=ESCALATION_MODEL,
                cost_usd=0.004,
            )
        ],
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    caplog.set_level(logging.INFO, logger="app.pipeline")

    await pipeline.indexer().index(user_id, item_id)

    item = await load(pipeline.sessions, item_id)
    assert item.transcript is None
    assert item.fields["entities"][0]["name"] == "Sign Café"
    assert pipeline.primary_stt.calls == [] and pipeline.fallback_stt.calls == []
    assert "has_speech" not in pipeline.classifier.asked_keys()
    runs = await runs_of(pipeline, item_id)
    assert runs["transcribe"].output["outcome"] == "no_audio_track"
    assert runs["extract"].output["escalation"] == "found"
    indexed = next(record for record in caplog.records if record.message == "saved_item_indexed")
    assert indexed.escalated is True
    # Text model, escalation, and three decisions: the type, and the privacy gate
    # of the new category and tag. The fake embedder reports no charge.
    assert indexed.cost_usd == pytest.approx(0.0002 + 0.004 + 3 * 0.0001)
    assert indexed.unpriced_model_calls == 3
    assert indexed.type == "place"


async def test_every_audio_track_is_transcribed_even_if_the_caption_names_the_place(
    pipeline: Pipeline,
) -> None:
    pipeline.classifier.content_type = "place"
    pipeline.classifier.extra_answers = {
        "has_speech": YesNoAnswer(probability_yes=0.02, confidence=None)
    }
    pipeline.script(
        transcripts=[transcript("")],
        fallback_transcripts=[transcript("♪", model=FALLBACK_STT_MODEL)],
        extractions=[structured_reply(place_extraction(("Kronotrop", None)))],
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(
        pipeline.sessions, pipeline, capture_event(user_id, caption="Kronotrop Bebek ☕️")
    )

    await pipeline.indexer().index(user_id, item_id)

    assert len(pipeline.primary_stt.calls) == 1
    assert len(pipeline.fallback_stt.calls) == 1
    runs = await runs_of(pipeline, item_id)
    assert runs["transcribe"].output["outcome"] == "no_speech"
    assert (await load(pipeline.sessions, item_id)).transcript is None


async def test_a_study_video_keeps_its_open_subject_and_is_not_escalated(
    pipeline: Pipeline,
) -> None:
    pipeline.classifier.content_type = "study"
    pipeline.script(
        transcripts=[transcript("Review your notes just before you forget them")],
        extractions=[structured_reply(study_extraction())],
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))

    await pipeline.indexer().index(user_id, item_id)

    item = await load(pipeline.sessions, item_id)
    assert item.type == "study"
    assert item.fields["subject"] == "study skills"
    assert item.fields["format"] == "tip"
    assert item.fields["topic"] == "spaced repetition"
    assert pipeline.escalation_model.requests == []


async def test_a_post_with_no_built_in_type_keeps_its_open_category_and_is_confirmed(
    pipeline: Pipeline,
) -> None:
    pipeline.classifier.content_type = NONE_OF_THESE
    pipeline.script(
        transcripts=[transcript("This sports car does zero to sixty fast")],
        extractions=[structured_reply(open_extraction("Car Review", ["Sports Car"]))],
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))

    await pipeline.indexer().index(user_id, item_id)

    item = await load(pipeline.sessions, item_id)
    assert item.type is None
    assert item.category == "car review"
    assert item.tags == ["sports car"]
    assert item.fields == {}
    assert pipeline.messenger.sent[0][1] == "Saved as car review"
    assert await vocabulary_values(pipeline, "category") == {"car review": 1}


async def test_a_failed_step_is_tried_again_without_repeating_the_finished_steps(
    pipeline: Pipeline,
) -> None:
    pipeline.classifier.content_type = "place"
    outage = ModelUnavailableError("synthetic outage", operation="generate", model="fake/text")
    pipeline.script(
        transcripts=[transcript("Kronotrop'a gittik")],
        extractions=[outage, structured_reply(place_extraction(("Kronotrop", None)))],
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    indexer = pipeline.indexer()

    with pytest.raises(RetryLaterError) as raised:
        await indexer.index(user_id, item_id)

    assert raised.value.step is Step.EXTRACT
    first_try = await runs_of(pipeline, item_id)
    assert first_try["extract"].status == "failed"
    assert "ModelUnavailableError" in str(first_try["extract"].error)
    assert (await load(pipeline.sessions, item_id)).status == ItemStatus.ENRICHING

    assert await indexer.index(user_id, item_id) is IndexOutcome.INDEXED

    assert pipeline.media_server.requests == 1
    assert pipeline.classifier.asked_keys().count("content_type") == 1
    assert len(pipeline.primary_stt.calls) == 1
    assert len(pipeline.text_model.requests) == 2
    assert (await load(pipeline.sessions, item_id)).status == ItemStatus.DONE


async def test_a_failure_on_the_last_try_marks_the_item_failed(pipeline: Pipeline) -> None:
    failure = ModelUnavailableError("synthetic outage", operation="transcribe", model="fake/mai")
    pipeline.script(transcripts=[failure], fallback_transcripts=[failure])
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))

    outcome = await pipeline.indexer(max_tries=1).index(user_id, item_id)

    assert outcome is IndexOutcome.FAILED
    assert (await load(pipeline.sessions, item_id)).status == ItemStatus.FAILED
    runs = await runs_of(pipeline, item_id)
    assert runs["transcribe"].status == "failed"
    assert "TranscriptionFailedError" in str(runs["transcribe"].error)
    assert pipeline.text_model.requests == []


async def test_no_words_then_a_fallback_error_is_tried_again_not_saved_as_silence(
    pipeline: Pipeline,
) -> None:
    failure = ModelUnavailableError("synthetic outage", operation="transcribe", model="fake/gpt")
    pipeline.script(transcripts=[transcript("")], fallback_transcripts=[failure])
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))

    with pytest.raises(RetryLaterError):
        await pipeline.indexer().index(user_id, item_id)

    runs = await runs_of(pipeline, item_id)
    assert runs["transcribe"].status == "failed"
    assert "IncompleteTranscriptionError" in str(runs["transcribe"].error)


async def test_a_dm_reel_without_a_video_is_indexed_from_its_caption(pipeline: Pipeline) -> None:
    pipeline.classifier.content_type = "movie"
    pipeline.script(extractions=[structured_reply(movie_extraction())])
    user_id = await add_user(pipeline.sessions)
    event = capture_event(user_id, media_url=None, caption="Bu filmi izleyin")
    item_id = await add_item(pipeline.sessions, pipeline, event)

    await pipeline.indexer().index(user_id, item_id)

    assert pipeline.media_server.requests == 0
    assert pipeline.escalation_model.requests == []
    runs = await runs_of(pipeline, item_id)
    assert runs["fetch"].output["unavailable_reason"] == "no_media_link"
    assert runs["extract"].output["escalation"] == "no_media"
    assert (await load(pipeline.sessions, item_id)).status == ItemStatus.DONE


async def test_an_item_with_no_media_and_no_caption_fails(pipeline: Pipeline) -> None:
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id, media_url=None))

    outcome = await pipeline.indexer().index(user_id, item_id)

    assert outcome is IndexOutcome.FAILED
    assert pipeline.classifier.calls == []


class MessengerThatSeesTheStartMark(RecordingMessenger):
    """Records, at the moment of sending, whether the confirmation was marked started."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        super().__init__()
        self.sessions = sessions
        self.item_id: uuid.UUID | None = None
        self.marked_when_sent: list[bool] = []

    async def send(
        self, recipient_id: str, message: OutgoingMessage, last_user_message_at: datetime
    ) -> SendResult:
        assert self.item_id is not None
        async with self.sessions() as session:
            self.marked_when_sent.append(await confirmation_started(session, self.item_id))
        return await super().send(recipient_id, message, last_user_message_at)


async def test_the_confirmation_is_marked_started_before_it_is_sent(pipeline: Pipeline) -> None:
    # A collection the chat worker sets after the mark is confirmed by its own message.
    messenger = MessengerThatSeesTheStartMark(pipeline.sessions)
    pipeline.messenger = messenger
    pipeline.script(
        transcripts=[transcript("")],
        fallback_transcripts=[transcript("")],
        extractions=[structured_reply(open_extraction())],
    )
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    messenger.item_id = item_id

    await pipeline.indexer().index(user_id, item_id)

    assert messenger.marked_when_sent == [True]
    assert [text for _, text in messenger.sent] == ["Saved as car review"]
    assert (await runs_of(pipeline, item_id))["confirm"].output["collection_included"] is False


async def test_a_user_without_a_known_language_is_confirmed_in_english(
    pipeline: Pipeline,
) -> None:
    pipeline.script(
        transcripts=[transcript("")],
        fallback_transcripts=[transcript("")],
        extractions=[structured_reply(open_extraction())],
    )
    user_id = await add_user(pipeline.sessions, language=None)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))

    await pipeline.indexer().index(user_id, item_id)

    assert [text for _, text in pipeline.messenger.sent] == ["Saved as car review"]
    runs = await runs_of(pipeline, item_id)
    assert runs["confirm"].status == "done"
    assert runs["confirm"].output == {
        "status": "sent",
        "reason": None,
        "language": "en",
        "language_choice": "fallback_no_language",
        "collection_included": False,
    }


async def test_a_job_without_a_payload_stops_because_the_user_was_deleted(
    pipeline: Pipeline,
) -> None:
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    del pipeline.payloads.events[item_id]

    assert await pipeline.indexer().index(user_id, item_id) is IndexOutcome.GONE
    assert pipeline.media_server.requests == 0


async def test_an_item_that_is_already_done_is_not_indexed_again(pipeline: Pipeline) -> None:
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    async with pipeline.sessions() as session, session.begin():
        item = await session.get(SavedItem, item_id)
        assert item is not None
        item.status = ItemStatus.DONE

    outcome = await pipeline.indexer().index(user_id, item_id)

    assert outcome is IndexOutcome.NOT_PENDING
    assert pipeline.media_server.requests == 0


async def test_logs_and_step_outputs_never_hold_the_media_link(
    pipeline: Pipeline, caplog: pytest.LogCaptureFixture
) -> None:
    pipeline.script(
        transcripts=[transcript("synthetic private words")],
        extractions=[structured_reply(open_extraction())],
    )
    user_id = await add_user(pipeline.sessions)
    event = capture_event(user_id, caption="synthetic private caption")
    item_id = await add_item(pipeline.sessions, pipeline, event)
    configure_logging("DEBUG")
    caplog.set_level(logging.DEBUG)

    await pipeline.indexer().index(user_id, item_id)

    logged = " ".join(f"{record.getMessage()} {record.__dict__}" for record in caplog.records)
    assert "synthetic private" not in logged
    assert str(event.media_url) not in logged
    outputs = {step: run.output for step, run in (await runs_of(pipeline, item_id)).items()}
    assert str(event.media_url) not in str(outputs)


def test_attempts_type_is_shared_with_the_models_package() -> None:
    assert TranscriptionAttempt(model="m", outcome="text").error is None


async def test_an_unexpected_error_marks_the_item_failed_and_is_raised(pipeline: Pipeline) -> None:
    pipeline.classifier.content_type = "unregistered"
    pipeline.script(transcripts=[transcript("synthetic speech")])
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id, caption="x"))

    with pytest.raises(UnknownContentTypeError):
        await pipeline.indexer().index(user_id, item_id)

    assert (await load(pipeline.sessions, item_id)).status == ItemStatus.FAILED


async def test_every_run_counts_against_the_items_tries_and_the_last_marks_it_failed(
    pipeline: Pipeline,
) -> None:
    failure = ModelUnavailableError("synthetic outage", operation="transcribe", model="fake/mai")
    pipeline.script(transcripts=[failure] * 3, fallback_transcripts=[failure] * 3)
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))

    # A new indexer each run, as after a requeue: the count lives on the item.
    for _ in range(2):
        with pytest.raises(RetryLaterError):
            await pipeline.indexer(max_tries=3).index(user_id, item_id)
    outcome = await pipeline.indexer(max_tries=3).index(user_id, item_id)

    item = await load(pipeline.sessions, item_id)
    assert outcome is IndexOutcome.FAILED
    assert item.indexing_attempts == 3
    assert item.status == ItemStatus.FAILED


async def test_a_run_marks_the_item_as_worked_on(pipeline: Pipeline) -> None:
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    async with pipeline.sessions() as session, session.begin():
        await session.execute(
            update(SavedItem)
            .where(SavedItem.id == item_id)
            .values(indexing_updated_at=func.now() - timedelta(days=2))
        )
    pipeline.script(
        transcripts=[transcript("synthetic speech")],
        extractions=[structured_reply(open_extraction())],
    )

    await pipeline.indexer().index(user_id, item_id)

    async with pipeline.sessions() as session:
        age = await session.scalar(
            select(func.now() - SavedItem.indexing_updated_at).where(SavedItem.id == item_id)
        )
    assert age is not None and age < timedelta(minutes=1)


async def test_an_item_whose_tries_are_spent_is_failed_without_any_step(
    pipeline: Pipeline,
) -> None:
    user_id = await add_user(pipeline.sessions)
    item_id = await add_item(pipeline.sessions, pipeline, capture_event(user_id))
    async with pipeline.sessions() as session, session.begin():
        await session.execute(
            update(SavedItem)
            .where(SavedItem.id == item_id)
            .values(indexing_attempts=2, status=ItemStatus.ENRICHING)
        )

    outcome = await pipeline.indexer(max_tries=2).index(user_id, item_id)

    item = await load(pipeline.sessions, item_id)
    assert outcome is IndexOutcome.FAILED
    assert (item.status, item.indexing_attempts) == (ItemStatus.FAILED, 2)
    assert pipeline.media_server.requests == 0
    assert pipeline.classifier.calls == []
    assert pipeline.primary_stt.calls == []
    assert pipeline.text_model.requests == []
