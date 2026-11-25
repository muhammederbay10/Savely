"""An answer about many saves never fails because of its length (SAV-38 live check, 25 Nov 2026).

In the live run, the model listed all 11 ids of "What did i save?" in the
answer step, its JSON was cut off at the output limit, and the turn ended in
`SchemaOutputError`. These tests script that with a fake model: the answer
lists at most `AGENT_MAX_RESULTS_PER_ANSWER` items, the other results are kept
in order for "show more", and a cut-off answer is asked for again.

These tests need Postgres with PostGIS and pgvector (see `tests/database.py`).
Every name and id is synthetic.
"""

import uuid
from collections.abc import AsyncIterator, Sequence

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.agent import AgentStepLimitError, AgentTurn
from app.agent.prompts import AGENT_PROMPTS
from app.agent.replies.answers import ResultCard
from app.agent.tools import SEARCH_ITEMS_TOOL, SearchUsed
from app.models.base import GenerationRequest
from tests.agent.agent_support import (
    MAX_RESULTS,
    AgentUnderTest,
    build_agent,
    cut_off_answer,
    final_answer,
    text_reply,
    tool_calls,
    tool_results,
)
from tests.agent.retrieval_support import Saves, seed_saves

SEARCH_EVERYTHING = tool_calls((SEARCH_ITEMS_TOOL, {}))
LONG_DRAFT = "You have 9 saved posts; here are your 5 most recent."
CUT_OFF_PROMPT = AGENT_PROMPTS.answer_cut_off.format(max_results=MAX_RESULTS)
# The seeded user's fully indexed saves, newest first, as `search_items` without
# filters returns them: more than one answer shows.
NEWEST_FIRST = [
    "car",
    "integrals",
    "izakaya",
    "diner",
    "kimchi",
    "kronotrop",
    "hereditary",
    "conjuring",
    "inception",
]


@pytest.fixture
async def saves(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[Saves]:
    async with session_factory() as session, session.begin():
        saves = await seed_saves(session)
    yield saves


def turn(saves: Saves, text: str = "What did i save?", **values: object) -> AgentTurn:
    defaults: dict[str, object] = {"user_id": saves.user.id, "country": "US", "text": text}
    return AgentTurn(**{**defaults, **values})  # type: ignore[arg-type]


def found_ids(request: GenerationRequest) -> list[uuid.UUID]:
    (search,) = tool_results(request)
    results = search["results"]
    assert isinstance(results, list)
    return [uuid.UUID(str(result["item_id"])) for result in results]


def shown_ids(cards: Sequence[ResultCard]) -> list[uuid.UUID]:
    return [card.item_id for card in cards]


def is_answer_step(request: GenerationRequest) -> bool:
    return request.output_schema is not None and request.tool_choice == "none"


def answer_step_ids(under_test: AgentUnderTest) -> list[bool]:
    return [is_answer_step(request) for request in under_test.generator.requests]


async def test_a_long_id_list_cut_off_in_the_answer_step_is_asked_for_again(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    """The live failure: the answer step's JSON was cut off, and the turn must still answer."""
    newest = saves.ids(*NEWEST_FIRST)
    under_test = build_agent(
        session_factory,
        [
            SEARCH_EVERYTHING,
            text_reply(LONG_DRAFT),
            cut_off_answer(),
            final_answer("en", LONG_DRAFT, newest[:MAX_RESULTS]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves))

    assert answer_step_ids(under_test) == [False, False, True, True]
    retry = under_test.generator.requests[3]
    assert (retry.messages[-1].role, retry.messages[-1].content) == ("system", CUT_OFF_PROMPT)
    assert shown_ids(answer.messages.results) == newest[:MAX_RESULTS]
    found = found_ids(under_test.generator.requests[1])
    assert list(answer.more_item_ids) == found[MAX_RESULTS:]


async def test_the_answer_format_caps_the_items_at_what_one_answer_sends(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    newest = saves.ids(*NEWEST_FIRST)
    under_test = build_agent(
        session_factory,
        [SEARCH_EVERYTHING, text_reply(LONG_DRAFT), final_answer("en", LONG_DRAFT, newest[:2])],
    )

    await under_test.agent.answer(turn(saves))

    schema = under_test.generator.requests[2].output_schema
    assert schema is not None
    item_ids = schema.schema["properties"]["item_ids"]  # type: ignore[index]
    assert item_ids["maxItems"] == MAX_RESULTS
    system = under_test.generator.requests[0].messages[0].content
    assert f"List at most {MAX_RESULTS} items" in str(system)


async def test_a_cut_off_answer_at_the_last_step_gets_one_more_call(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    newest = saves.ids(*NEWEST_FIRST)
    under_test = build_agent(
        session_factory,
        [SEARCH_EVERYTHING, cut_off_answer(), final_answer("en", "Latest:", newest[:2])],
        agent_max_steps=2,
    )

    answer = await under_test.agent.answer(turn(saves))

    assert answer_step_ids(under_test) == [False, True, True]
    assert shown_ids(answer.messages.results) == newest[:2]


async def test_a_second_cut_off_beyond_the_step_limit_is_a_step_limit_error(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    under_test = build_agent(
        session_factory,
        [SEARCH_EVERYTHING, cut_off_answer(), cut_off_answer()],
        agent_max_steps=2,
    )

    with pytest.raises(AgentStepLimitError):
        await under_test.agent.answer(turn(saves))

    assert len(under_test.generator.requests) == 3


async def test_a_prose_reply_cut_off_in_a_tool_step_is_not_kept_as_the_draft(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    newest = saves.ids(*NEWEST_FIRST)
    under_test = build_agent(
        session_factory,
        [
            SEARCH_EVERYTHING,
            text_reply(f"{LONG_DRAFT} {newest}", finish_reason="length"),
            final_answer("en", LONG_DRAFT, newest[:MAX_RESULTS]),
        ],
    )

    await under_test.agent.answer(turn(saves))

    *_, before, ask = under_test.generator.requests[2].messages
    assert before.role == "tool"
    assert ask.content == AGENT_PROMPTS.answer_now


async def test_an_answer_listing_every_result_shows_the_cap_and_keeps_the_rest_in_order(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    newest = saves.ids(*NEWEST_FIRST)
    under_test = build_agent(
        session_factory, [SEARCH_EVERYTHING, final_answer("en", LONG_DRAFT, newest)]
    )

    answer = await under_test.agent.answer(turn(saves))

    found = found_ids(under_test.generator.requests[1])
    assert shown_ids(answer.messages.results) == newest[:MAX_RESULTS]
    not_listed = [item_id for item_id in found if item_id not in newest]
    assert list(answer.more_item_ids) == newest[MAX_RESULTS:] + not_listed


async def test_search_results_the_answer_does_not_list_are_kept_for_show_more(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    newest = saves.ids(*NEWEST_FIRST)
    under_test = build_agent(
        session_factory, [SEARCH_EVERYTHING, final_answer("en", "Your two latest:", newest[:2])]
    )

    answer = await under_test.agent.answer(turn(saves))

    found = found_ids(under_test.generator.requests[1])
    assert found[:2] == newest[:2]
    assert shown_ids(answer.messages.results) == newest[:2]
    assert list(answer.more_item_ids) == found[2:]
    assert list(answer.working_set) == found


async def test_show_more_continues_with_the_results_not_shown_yet(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    last_results = saves.ids(*NEWEST_FIRST)
    last_search = SearchUsed(SEARCH_ITEMS_TOOL, {})
    under_test = build_agent(
        session_factory, [final_answer("en", "The next ones:", last_results[3:8])]
    )

    answer = await under_test.agent.answer(
        turn(
            saves,
            "show more",
            working_set=last_results,
            working_set_shown=3,
            working_set_searches=(last_search,),
        )
    )

    assert shown_ids(answer.messages.results) == last_results[3:8]
    assert list(answer.more_item_ids) == last_results[8:]
    assert answer.searches == (last_search,)


async def test_an_answer_about_one_of_the_last_results_adds_no_others(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    last_results = saves.ids(*NEWEST_FIRST)
    under_test = build_agent(
        session_factory, [final_answer("en", "That one is a car review.", last_results[:1])]
    )

    answer = await under_test.agent.answer(
        turn(saves, "what was the first one?", working_set=last_results, working_set_shown=3)
    )

    assert shown_ids(answer.messages.results) == last_results[:1]
    assert answer.more_item_ids == ()


async def test_a_narrower_search_replaces_a_broad_one_for_show_more(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    """Reviewer's case: the broad search's hits must not page into the restaurant results."""
    kimchi, diner = saves.ids("kimchi", "diner")
    unrelated = saves.ids("car", "integrals", "izakaya")
    under_test = build_agent(
        session_factory,
        [
            SEARCH_EVERYTHING,
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "restaurant"})),
            final_answer("en", "One restaurant; ask for more.", [kimchi]),
            # "show more": the model tries an unrelated item, then lists what is allowed.
            final_answer("en", "The next ones:", [diner, *unrelated]),
            final_answer("en", "The next one:", [diner]),
        ],
    )

    first = await under_test.agent.answer(turn(saves, "restaurants"))

    restaurant_search = tool_results(under_test.generator.requests[2])[-1]
    restaurants = [uuid.UUID(str(result["item_id"])) for result in restaurant_search["results"]]  # type: ignore[union-attr]
    assert set(restaurants) == {kimchi, diner}
    assert first.more_item_ids == (diner,)
    assert not set(unrelated) & set(first.working_set)

    second = await under_test.agent.answer(
        turn(
            saves,
            "show more",
            working_set=first.working_set,
            working_set_shown=len(first.messages.results),
            working_set_searches=first.searches,
        )
    )

    assert shown_ids(second.messages.results) == [diner]
    assert second.more_item_ids == ()


async def test_an_answer_from_two_searches_keeps_both_searches_unshown_results(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    kronotrop, hereditary = saves.ids("kronotrop", "hereditary")
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (SEARCH_ITEMS_TOOL, {"type": "place"}),
                (SEARCH_ITEMS_TOOL, {"type": "movie"}),
            ),
            final_answer("en", "A place and a movie:", [kronotrop, hereditary]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "places and movies I saved"))

    places, movies = (
        [uuid.UUID(str(result["item_id"])) for result in search["results"]]  # type: ignore[union-attr]
        for search in tool_results(under_test.generator.requests[1])
    )
    shown = {kronotrop, hereditary}
    expected = [item_id for item_id in [*places, *movies] if item_id not in shown]
    assert expected
    assert list(answer.more_item_ids) == expected
