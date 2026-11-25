"""Opt-in check: on the real text model, the response agent searches and answers with items.

SAV-38 found that the configured text model answered "Let me pull up your
saves." without calling a tool. The unit tests use a scripted model, so they
can't see that. This check asks the real `ResponseAgent` a few questions in
two languages, each `RUNS_PER_QUESTION` times, and every run must call a
search tool and send at least one saved item. Each question has matching
saves, so "nothing found" is a failure here.

What is real: the text model (the response agent) and the decision model
(search's filter normalization), from the same settings as production. What
is not: the saves are the synthetic test saves in the test database, and
embeddings are offline (the test database stores 8-dimensional vectors), so
ranking by meaning is not tested.

It is skipped unless `LIVE_MODEL_TESTS=1`, `OPENROUTER_API_KEY` and
`TEST_DATABASE_ADMIN_URL` are set, and always skipped when `CI` is set: the
module's skip mark is decided at import, before any fixture builds a model
client (`tests/live/test_response_agent_live_gate.py` proves it). It never
runs in CI or in an agent's worktree; the owner runs it locally before a
push (SAV-22, 24 Nov 2026 update). `docs/development.md` ("Live response agent check") shows how and
how to read the result. A run costs a few cents.
"""

import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.agent import AgentAnswer, AgentTurn, ResponseAgent
from app.agent.prompts import AGENT_PROMPTS
from app.agent.search.prompts import RETRIEVAL_PROMPTS
from app.agent.search.retrieval import HybridRetriever
from app.agent.tools import NEAR_TOOL, SEARCH_ITEMS_TOOL
from app.config.agent import AgentSettings
from app.config.models import ModelSettings
from app.db.models import Entity, ItemEntity, ItemStatus, SavedItem
from app.db.types import GeoPoint
from app.models.base import GenerationRequest, GenerationResult, Generator
from app.models.errors import SchemaOutputError
from app.models.factory import ModelClients, build_model_clients, open_http_client
from tests.agent.agent_support import (
    FakeClock,
    MemoryDeletionRequests,
    agent_settings,
    build_toolbox,
)
from tests.agent.retrieval_support import (
    MappedEmbedder,
    Saves,
    place_fields,
    retrieval_settings,
    seed_saves,
)
from tests.pipeline.support import ScriptedClassifier

OPT_IN_VARIABLE = "LIVE_MODEL_TESTS"
API_KEY_VARIABLE = "OPENROUTER_API_KEY"
# Set by GitHub Actions and most CI services. Any value but these counts as CI.
CI_VARIABLE = "CI"
NOT_CI_VALUES = frozenset({"", "0", "false"})


def live_run_blocker(environ: Mapping[str, str]) -> str | None:
    """Return why the live check may not run in this environment, or `None` if it may."""
    if environ.get(CI_VARIABLE, "").strip().casefold() not in NOT_CI_VALUES:
        return f"the live response agent check never runs in CI ({CI_VARIABLE} is set)"
    if environ.get(OPT_IN_VARIABLE) != "1" or not environ.get(API_KEY_VARIABLE):
        return f"live model tests need {OPT_IN_VARIABLE}=1 and {API_KEY_VARIABLE}"
    return None


_BLOCKER = live_run_blocker(os.environ)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_BLOCKER is not None, reason=_BLOCKER or ""),
]

RUNS_PER_QUESTION = 3
SEARCH_TOOLS = frozenset({SEARCH_ITEMS_TOOL, NEAR_TOOL})
# The agent runs with these settings from `.env.example`, so it has production's
# step limit, result cap and output limit, not the unit tests' smaller ones.
ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
AGENT_SETTINGS_FROM_ENV_EXAMPLE = (
    "AGENT_MAX_STEPS",
    "AGENT_MAX_RESULTS_PER_ANSWER",
    "AGENT_MAX_OUTPUT_TOKENS",
)
# Matches all of the user's saves, more than one answer shows, so the answer
# must show at most the cap and keep the rest for "show more".
ALL_SAVES_QUESTION = "What did i save?"
# (language, question). The seeded user's country is US, where a saved horror
# movie streams on Netflix.
QUESTIONS = [
    ("en", ALL_SAVES_QUESTION),
    ("en", "restaurants in Kadıköy"),
    ("tr", "kaydettiğim tatlıcılar"),
    ("tr", "netflixte ne izleyebilirim"),
]
KADIKOY = GeoPoint(latitude=40.9903, longitude=29.0290)
KARAKOY = GeoPoint(latitude=41.0225, longitude=28.9770)


@dataclass
class RecordingGenerator(Generator):
    """Passes each call to the real model and keeps its replies, to see which tools ran."""

    model: Generator
    replies: list[GenerationResult] = field(default_factory=list)
    cut_offs: int = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            reply = await self.model.generate(request)
        except SchemaOutputError:
            self.cut_offs += 1
            raise
        self.replies.append(reply)
        return reply

    @property
    def tools_called(self) -> list[str]:
        return [call.name for reply in self.replies for call in reply.tool_calls]


@pytest.fixture
async def clients() -> AsyncIterator[ModelClients]:
    settings = ModelSettings(_env_file=None)
    async with open_http_client(settings) as http_client:
        yield build_model_clients(settings, http_client)


@pytest.fixture
async def saves(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[Saves]:
    async with session_factory() as session, session.begin():
        saves = await seed_saves(session)
        await _add_question_saves(session, saves)
    yield saves


async def _add_question_saves(session: AsyncSession, saves: Saves) -> None:
    """Add a restaurant in Kadıköy and a dessert shop, so every question has an answer."""
    kebab = _place("Synthetic Kebap Evi", KADIKOY, district="Kadıköy")
    baklava = _place("Synthetic Baklavacı", KARAKOY, district="Beyoğlu")
    session.add_all([kebab, baklava])
    await session.flush()
    items = {
        "kebab": _place_item(
            saves, "kebab", "restaurant", ("Synthetic Kebap Evi", "Kadıköy", "kebab")
        ),
        "baklava": _place_item(
            saves, "baklava", "dessert shop", ("Synthetic Baklavacı", "Karaköy", "baklava")
        ),
    }
    session.add_all(items.values())
    await session.flush()
    session.add_all(
        [
            ItemEntity(saved_item_id=items["kebab"].id, entity_id=kebab.id),
            ItemEntity(saved_item_id=items["baklava"].id, entity_id=baklava.id),
        ]
    )
    saves.items.update(items)


def _place(name: str, location: GeoPoint, *, district: str) -> Entity:
    return Entity(
        type="place",
        source="google_places",
        external_id=f"synthetic-live-{name}",
        name=name,
        location=location,
        country="Türkiye",
        city="İstanbul",
        district=district,
    )


def _place_item(
    saves: Saves, name: str, category: str, venue: tuple[str, str | None, str | None]
) -> SavedItem:
    return SavedItem(
        user_id=saves.user.id,
        platform="instagram",
        platform_post_id=f"synthetic-live-{name}",
        media_type="reel",
        status=ItemStatus.DONE,
        type="place",
        category=category,
        collection="Food",
        fields=place_fields(venue),
    )


def env_example_agent_settings() -> AgentSettings:
    """Return agent settings with the step limit, cap and output limit of `.env.example`.

    Raises:
        KeyError: If `.env.example` lacks one of them.
    """
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() in AGENT_SETTINGS_FROM_ENV_EXAMPLE:
            values[name.strip()] = value.strip()
    return agent_settings(
        **{name.lower(): int(values[name]) for name in AGENT_SETTINGS_FROM_ENV_EXAMPLE}
    )


def build_live_agent(
    sessions: async_sessionmaker[AsyncSession], clients: ModelClients, model: Generator
) -> ResponseAgent:
    retriever = HybridRetriever(
        classifier=clients.classifier,
        embedder=MappedEmbedder(),
        prompts=RETRIEVAL_PROMPTS,
        settings=retrieval_settings(),
    )
    toolbox = build_toolbox(
        sessions,
        retriever=retriever,
        classifier=ScriptedClassifier(),
        deletion_requests=MemoryDeletionRequests(),
        clock=FakeClock(),
    )
    return ResponseAgent(
        generator=model,
        toolbox=toolbox,
        prompts=AGENT_PROMPTS,
        settings=env_example_agent_settings(),
    )


def _describe(run: int, recorder: RecordingGenerator, answer: AgentAnswer) -> str:
    return (
        f"run={run} prompts={AGENT_PROMPTS.version} answered_in={answer.language} "
        f"model_calls={len(recorder.replies)} cut_offs={recorder.cut_offs} "
        f"tools={recorder.tools_called} items={len(answer.messages.results)} "
        f"kept_for_more={len(answer.more_item_ids)}"
    )


@pytest.mark.parametrize("run", range(1, RUNS_PER_QUESTION + 1))
@pytest.mark.parametrize(("language", "question"), QUESTIONS)
async def test_the_agent_searches_and_answers_with_saved_items(
    session_factory: async_sessionmaker[AsyncSession],
    saves: Saves,
    clients: ModelClients,
    language: str,
    question: str,
    run: int,
) -> None:
    recorder = RecordingGenerator(clients.text_generator)
    agent = build_live_agent(session_factory, clients, recorder)

    answer = await agent.answer(
        AgentTurn(user_id=saves.user.id, country=saves.user.country, text=question)
    )

    outcome = _describe(run, recorder, answer)
    assert SEARCH_TOOLS & set(recorder.tools_called), f"no search tool was called: {outcome}"
    assert answer.messages.results, f"no saved item was sent: {outcome}"
    cap = env_example_agent_settings().agent_max_results_per_answer
    assert len(answer.messages.results) <= cap, f"more items than one answer sends: {outcome}"
    if question == ALL_SAVES_QUESTION:
        assert _owned_saves(saves) > cap, "the seed must hold more saves than one answer shows"
        assert answer.more_item_ids, f"the saves not shown were not kept: {outcome}"


def _owned_saves(saves: Saves) -> int:
    return sum(1 for item in saves.items.values() if item.user_id == saves.user.id)
