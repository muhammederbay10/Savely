"""Which request sends the tools and which sends the answer format (SAV-38).

The configured text model skipped the tools when both were sent together, so
tool steps send only the tools and the answer step only the format. The text
model is a scripted fake: these tests check the requests the agent builds and
the checks on its answers, not the real model's behaviour (that is the opt-in
check in `tests/live/test_response_agent_live.py`).

These tests need Postgres with PostGIS and pgvector (see `tests/database.py`).
Every name and id is synthetic.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.agent import AgentTurn, AnswerOutcome
from app.agent.prompts import AGENT_PROMPTS
from app.agent.tools import REQUEST_DATA_DELETION_TOOL, SEARCH_ITEMS_TOOL
from app.models.base import GenerationRequest
from tests.agent.agent_support import (
    build_agent,
    final_answer,
    structured_reply,
    text_reply,
    tool_calls,
)
from tests.agent.retrieval_support import Saves, seed_saves

ANSWER_SCHEMA_NAME = "answer"
SEARCH_EVERYTHING = tool_calls((SEARCH_ITEMS_TOOL, {}))


@pytest.fixture
async def saves(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[Saves]:
    async with session_factory() as session, session.begin():
        saves = await seed_saves(session)
    yield saves


def turn(saves: Saves, text: str) -> AgentTurn:
    return AgentTurn(user_id=saves.user.id, country="US", text=text)


def is_tool_step(request: GenerationRequest) -> bool:
    return request.output_schema is None and request.tool_choice == "auto" and bool(request.tools)


def is_answer_step(request: GenerationRequest) -> bool:
    return (
        request.output_schema is not None
        and request.output_schema.name == ANSWER_SCHEMA_NAME
        and request.tool_choice == "none"
    )


async def test_tool_steps_offer_the_tools_without_the_answer_format(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    under_test = build_agent(
        session_factory, [SEARCH_EVERYTHING, final_answer("en", "Your latest save:", [car])]
    )

    answer = await under_test.agent.answer(turn(saves, "what did I save?"))

    assert [is_tool_step(request) for request in under_test.generator.requests] == [True, True]
    assert [card.item_id for card in answer.messages.results] == [car]


async def test_a_prose_reply_after_the_tools_is_asked_for_once_more_with_the_format(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    draft = "Here is your latest save."
    under_test = build_agent(
        session_factory,
        [SEARCH_EVERYTHING, text_reply(draft), final_answer("en", draft, [car])],
    )

    answer = await under_test.agent.answer(turn(saves, "what did I save?"))

    first, second, third = under_test.generator.requests
    assert is_tool_step(first) and is_tool_step(second)
    assert is_answer_step(third)
    assert third.tools == second.tools
    *_, kept_draft, ask = third.messages
    assert (kept_draft.role, kept_draft.content) == ("assistant", draft)
    assert (ask.role, ask.content) == ("system", AGENT_PROMPTS.answer_now)
    assert [card.item_id for card in answer.messages.results] == [car]


async def test_the_answer_format_asks_for_the_outcome(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    under_test = build_agent(
        session_factory, [text_reply("Hi!"), final_answer("en", "Hi!", outcome=AnswerOutcome.OTHER)]
    )

    await under_test.agent.answer(turn(saves, "hi"))

    schema = under_test.generator.requests[1].output_schema
    assert schema is not None
    assert "outcome" in schema.schema["required"]  # type: ignore[operator]
    properties = schema.schema["properties"]
    assert properties["outcome"]["enum"] == ["results", "no_results", "other"]  # type: ignore[index]


async def test_the_last_step_is_an_answer_step_even_while_the_model_calls_tools(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    under_test = build_agent(
        session_factory,
        [SEARCH_EVERYTHING, SEARCH_EVERYTHING, final_answer("en", "Your latest save:", [car])],
        agent_max_steps=3,
    )

    await under_test.agent.answer(turn(saves, "what did I save?"))

    steps = [is_answer_step(request) for request in under_test.generator.requests]
    assert steps == [False, False, True]
    assert under_test.generator.requests[2].messages[-1].content == AGENT_PROMPTS.answer_now


async def test_an_answer_claiming_results_without_items_goes_back_to_the_tools(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    under_test = build_agent(
        session_factory,
        [
            text_reply("Let me pull up your saves."),
            final_answer(
                "en", "Here's what you saved most recently.", [], outcome=AnswerOutcome.RESULTS
            ),
            SEARCH_EVERYTHING,
            final_answer("en", "Here's what you saved most recently.", [car]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "What did i save?"))

    requests = under_test.generator.requests
    assert [is_answer_step(request) for request in requests] == [False, True, False, False]
    correction = requests[2].messages[-1]
    assert (correction.role, correction.content) == ("system", AGENT_PROMPTS.outcome_mismatch)
    assert [card.item_id for card in answer.messages.results] == [car]


async def test_an_answer_saying_nothing_was_found_may_not_list_items(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    under_test = build_agent(
        session_factory,
        [
            SEARCH_EVERYTHING,
            final_answer("en", "Nothing matches.", [car], outcome=AnswerOutcome.NO_RESULTS),
            final_answer("en", "Your latest save:", [car]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "what did I save?"))

    assert under_test.generator.requests[2].messages[-1].content == AGENT_PROMPTS.outcome_mismatch
    assert [card.item_id for card in answer.messages.results] == [car]


async def test_an_answer_without_items_that_claims_none_is_sent(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "bowling alley"})),
            final_answer(
                "en", "You haven't saved any bowling alleys.", [], outcome=AnswerOutcome.NO_RESULTS
            ),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "bowling alleys I saved"))

    assert len(under_test.generator.requests) == 2
    assert answer.messages.intro == "You haven't saved any bowling alleys."
    assert answer.messages.results == ()


async def test_a_malformed_answer_in_the_answer_step_is_asked_for_again_with_the_format(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    under_test = build_agent(
        session_factory,
        [
            SEARCH_EVERYTHING,
            text_reply("Here is your latest save."),
            structured_reply({"language": "en", "reply": "Here is your latest save."}),
            final_answer("en", "Here is your latest save.", [car]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "what did I save?"))

    requests = under_test.generator.requests
    assert [is_answer_step(request) for request in requests] == [False, False, True, True]
    assert requests[3].messages[-1].content == AGENT_PROMPTS.invalid_answer
    assert [card.item_id for card in answer.messages.results] == [car]


async def test_the_deletion_question_replaces_the_reply_whatever_its_outcome(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    under_test = build_agent(
        session_factory,
        [
            tool_calls((REQUEST_DATA_DELETION_TOOL, {})),
            final_answer("en", "Deleting.", [], outcome=AnswerOutcome.RESULTS),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "delete my data"))

    assert len(under_test.generator.requests) == 2
    assert answer.deletion_confirmation is not None
