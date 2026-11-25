"""The response agent answers by calling tools, driven by a scripted (fake) text model.

These tests need Postgres with PostGIS and pgvector (see `tests/database.py`).
Every name, place and id is synthetic.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.agent import AgentStepLimitError, AgentTurn, AgentWordingError
from app.agent.tools import (
    GET_ITEM_TOOL,
    NEAR_TOOL,
    SEARCH_ITEMS_TOOL,
    SET_STATUS_TOOL,
    UPDATE_ITEM_TOOL,
    Notice,
)
from app.db.deletion import UserDataEraser
from app.db.models import (
    Entity,
    EntityStatus,
    ItemEntity,
    ItemStatus,
    SavedItem,
    UserEntity,
    Vocabulary,
    VocabularyAlias,
)
from app.models.errors import ModelUnavailableError
from tests.agent.agent_support import (
    MAX_RESULTS,
    build_agent,
    final_answer,
    structured_reply,
    tool_calls,
    tool_results,
    translated_wording,
)
from tests.agent.retrieval_support import (
    BEBEK,
    NOW,
    MappedEmbedder,
    Saves,
    build_retriever,
    seed_saves,
)

MEZE_MESSAGE_ID = "aWdfZAGFAKEMID0170"
MEZE_LINK = "https://maps.example.test/place/synthetic-meze"


@pytest.fixture
async def saves(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[Saves]:
    async with session_factory() as session, session.begin():
        saves = await seed_saves(session)
        meze = Entity(
            type="place",
            source="google_places",
            external_id="synthetic-meze",
            name="Meze Evi",
            location=BEBEK,
            country="Türkiye",
            city="İstanbul",
            district="Beşiktaş",
            data={"link": MEZE_LINK},
        )
        session.add(meze)
        await session.flush()
        item = SavedItem(
            user_id=saves.user.id,
            platform="instagram",
            media_type="reel",
            source_message_id=MEZE_MESSAGE_ID,
            type="place",
            category="restaurant",
            collection="Food",
            status=ItemStatus.DONE,
            fields={"entities": [{"name": "Meze Evi", "area_hint": "Beşiktaş", "cuisine": None}]},
            saved_at=NOW,
        )
        session.add(item)
        await session.flush()
        session.add(ItemEntity(saved_item_id=item.id, entity_id=meze.id))
        saves.items["meze"] = item
        saves.entities["meze"] = meze
    yield saves


def turn(saves: Saves, text: str, **values: object) -> AgentTurn:
    defaults: dict[str, object] = {"user_id": saves.user.id, "country": "US", "text": text}
    return AgentTurn(**{**defaults, **values})  # type: ignore[arg-type]


async def test_restaurants_in_a_district(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    meze = saves.items["meze"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (
                    SEARCH_ITEMS_TOOL,
                    {"type": "place", "category": "restaurant", "filters": {"area": "Beşiktaş"}},
                )
            ),
            final_answer("tr", "Beşiktaş'ta kaydettiğin 1 restoran:", [meze]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "Beşiktaş'taki restoranlar"))

    (search,) = tool_results(under_test.generator.requests[1])
    assert [result["item_id"] for result in search["results"]] == [str(meze)]  # type: ignore[index]
    (card,) = answer.messages.results
    assert (card.names, card.area, card.link) == (("Meze Evi",), "Beşiktaş", MEZE_LINK)
    assert card.source_message_id == MEZE_MESSAGE_ID
    assert answer.language == "tr"
    assert answer.messages.intro == "Beşiktaş'ta kaydettiğin 1 restoran:"
    assert answer.messages.wording is not None
    assert answer.notices == frozenset()


async def test_horror_movies_on_netflix_in_the_movies_collection(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    hereditary = saves.items["hereditary"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (
                    SEARCH_ITEMS_TOOL,
                    {
                        "collection": "Movies",
                        "type": "movie",
                        "filters": {"genre": "horror", "streaming": "Netflix"},
                    },
                )
            ),
            final_answer("en", "One horror movie from Movies is on Netflix:", [hereditary]),
        ],
    )

    answer = await under_test.agent.answer(
        turn(saves, "horror movies on Netflix in my Movies collection")
    )

    (search,) = tool_results(under_test.generator.requests[1])
    assert [result["item_id"] for result in search["results"]] == [str(hereditary)]  # type: ignore[index]
    assert [card.names for card in answer.messages.results] == [("Hereditary",)]


async def test_maths_study_videos(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    integrals = saves.items["integrals"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"type": "study", "filters": {"subject": "maths"}})),
            final_answer("en", "Your maths study video:", [integrals]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "my maths study videos"))

    (search,) = tool_results(under_test.generator.requests[1])
    assert [result["item_id"] for result in search["results"]] == [str(integrals)]  # type: ignore[index]
    (card,) = answer.messages.results
    assert card.names == ()
    assert card.label == "maths tutorial"


async def test_mark_the_second_one_as_watched(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    hereditary, conjuring = saves.items["hereditary"].id, saves.items["conjuring"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SET_STATUS_TOOL, {"item_id": str(conjuring), "status": "done"})),
            final_answer("en", "Marked The Conjuring as watched."),
        ],
    )

    answer = await under_test.agent.answer(
        turn(saves, "mark the second one as watched", working_set=(hereditary, conjuring))
    )

    first_request = under_test.generator.requests[0]
    working_set = [
        message.content for message in first_request.messages if message.role == "system"
    ]
    assert f"1. {hereditary}\n2. {conjuring}" in str(working_set[1])
    async with session_factory() as session:
        status = await session.scalar(
            select(UserEntity.status).where(
                UserEntity.user_id == saves.user.id,
                UserEntity.entity_id == saves.entities["conjuring"].id,
            )
        )
    assert status == EntityStatus.DONE
    assert answer.messages.results == ()


async def test_a_streaming_search_without_the_users_country_asks_for_it(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"type": "movie", "filters": {"streaming": "Netflix"}})),
            final_answer("en", "Which country are you in? Streaming differs by country."),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "movies on Netflix", country=None))

    (search,) = tool_results(under_test.generator.requests[1])
    assert search["ask_user_for"] == "country"
    assert answer.messages.results == ()


async def test_nothing_found_is_a_clear_answer_with_no_results(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "bowling alley"})),
            final_answer("en", "You haven't saved any bowling alleys yet."),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "bowling alleys I saved"))

    (search,) = tool_results(under_test.generator.requests[1])
    assert search["found"] == 0
    assert answer.messages.results == ()
    assert answer.messages.wording is None
    assert answer.notices == frozenset()


async def test_backup_mode_must_be_told_and_a_missing_notice_is_sent_back(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    broken_embeddings = MappedEmbedder(
        error=ModelUnavailableError("embed timed out", operation="embed", model="fake/embedding")
    )
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"text": "sedan test drive"})),
            final_answer("en", "One car video:", [car]),
            final_answer("en", "One car video:", [car], notice="Search is in backup mode."),
        ],
        retriever=build_retriever(embedder=broken_embeddings),
    )

    answer = await under_test.agent.answer(turn(saves, "that sedan test drive video"))

    (search,) = tool_results(under_test.generator.requests[1])
    assert search["notices"] == [Notice.BACKUP_MODE]
    correction = under_test.generator.requests[2].messages[-1]
    assert correction.role == "system"
    assert "notice" in str(correction.content)
    assert Notice.BACKUP_MODE in answer.notices
    assert answer.messages.intro.endswith("Search is in backup mode.")


async def test_relaxed_filters_are_reported_as_broader_results(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    kimchi, meze = saves.items["kimchi"].id, saves.items["meze"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (
                    SEARCH_ITEMS_TOOL,
                    {"type": "place", "category": "restaurant", "filters": {"area": "Bebek"}},
                )
            ),
            final_answer(
                "en", "Nothing in Bebek itself; nearby:", [meze, kimchi], notice="Broader matches."
            ),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "restaurants in Bebek"))

    (search,) = tool_results(under_test.generator.requests[1])
    assert search["relaxed"] == ["filters.area"]
    assert {result["item_id"] for result in search["results"]} == {str(kimchi), str(meze)}  # type: ignore[union-attr]
    assert search["notices"] == [Notice.BROADER_RESULTS]
    assert answer.notices == frozenset({Notice.BROADER_RESULTS})


async def test_an_answer_may_only_list_items_the_tools_returned(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    meze = saves.items["meze"].id
    other_users_item = saves.items["other_place"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "restaurant", "collection": "Food"})),
            final_answer("en", "Restaurants:", [meze, other_users_item]),
            final_answer("en", "Restaurants:", [meze]),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "restaurants in Food"))

    assert "no tool returned" in str(under_test.generator.requests[2].messages[-1].content)
    assert [card.item_id for card in answer.messages.results] == [meze]


async def test_tools_never_reach_another_users_items(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    other_users_item = str(saves.items["other_place"].id)
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (GET_ITEM_TOOL, {"item_id": other_users_item}),
                (SET_STATUS_TOOL, {"item_id": other_users_item, "status": "done"}),
                (
                    UPDATE_ITEM_TOOL,
                    {"item_id": other_users_item, "path": "category", "value": "bar"},
                ),
                (SEARCH_ITEMS_TOOL, {"user_id": str(saves.other_user.id)}),
            ),
            final_answer("en", "I couldn't find that."),
        ],
    )

    await under_test.agent.answer(turn(saves, "show me that one"))

    results = tool_results(under_test.generator.requests[1])
    assert [result["error"] for result in results[:3]] == [
        "The user has no saved item with that id."
    ] * 3
    assert "user_id" in str(results[3]["error"])
    async with session_factory() as session:
        statuses = await session.scalar(
            select(func.count())
            .select_from(UserEntity)
            .where(UserEntity.user_id == saves.other_user.id)
        )
        category = await session.scalar(
            select(SavedItem.category).where(SavedItem.id == saves.items["other_place"].id)
        )
    assert statuses == 0
    assert category == "café"


async def test_a_category_correction_changes_one_item_and_creates_no_synonym(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    car = saves.items["car"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (
                    UPDATE_ITEM_TOOL,
                    {"item_id": str(car), "path": "category", "value": "travel video"},
                )
            ),
            final_answer("en", "Saved it as a travel video."),
        ],
    )
    async with session_factory() as session:
        aliases_before = await session.scalar(select(func.count()).select_from(VocabularyAlias))
        before = {
            item.id: (item.category, item.tags, item.fields)
            for item in (await session.execute(select(SavedItem))).scalars()
        }

    await under_test.agent.answer(turn(saves, "no, the car one is a travel video"))

    (update,) = tool_results(under_test.generator.requests[1])
    assert (update["previous"], update["value"]) == ("car review", "travel video")
    async with session_factory() as session:
        after = {
            item.id: (item.category, item.tags, item.fields)
            for item in (await session.execute(select(SavedItem))).scalars()
        }
        corrected = await session.get(SavedItem, car)
        aliases_after = await session.scalar(select(func.count()).select_from(VocabularyAlias))
        car_review_aliases = await session.scalar(
            select(func.count())
            .select_from(VocabularyAlias)
            .join(Vocabulary, Vocabulary.id == VocabularyAlias.vocabulary_id)
            .where(Vocabulary.value.in_(["car review", "travel video"]))
        )
    changed = {item_id for item_id in before if before[item_id] != after[item_id]}
    assert changed == {car}
    assert corrected is not None
    assert corrected.category == "travel video"
    assert corrected.raw_values["category"] == {
        "field": "category",
        "wording": "travel video",
        "user_set": True,
    }
    assert aliases_after == aliases_before
    assert car_review_aliases == 0


async def test_near_searches_the_type_with_a_location_filter(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (
                    NEAR_TOOL,
                    {"lat": BEBEK.latitude, "lng": BEBEK.longitude, "radius_m": 500.0},
                ),
                (
                    NEAR_TOOL,
                    {
                        "lat": BEBEK.latitude,
                        "lng": BEBEK.longitude,
                        "radius_m": 500.0,
                        "open_now": True,
                    },
                ),
            ),
            final_answer("en", "Two places near you.", [saves.items["kronotrop"].id]),
        ],
    )

    await under_test.agent.answer(turn(saves, "what did I save around here?"))

    near, open_now = tool_results(under_test.generator.requests[1])
    found = {result["item_id"] for result in near["results"]}  # type: ignore[union-attr]
    assert found == {str(saves.items["kronotrop"].id), str(saves.items["meze"].id)}
    assert "opening hours" in str(open_now["error"])


async def test_results_are_capped_and_the_rest_kept_for_show_more(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    names = ["meze", "car", "integrals", "izakaya", "diner", "kimchi", "kronotrop"]
    ids = [saves.items[name].id for name in names]
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {})),
            final_answer("en", "Your 7 latest saves; ask for more to see the rest.", ids),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, "what did I save lately?"))

    (search,) = tool_results(under_test.generator.requests[1])
    found = [uuid.UUID(str(result["item_id"])) for result in search["results"]]  # type: ignore[union-attr]
    not_listed = [item_id for item_id in found if item_id not in ids]
    assert [card.item_id for card in answer.messages.results] == ids[:MAX_RESULTS]
    # What the answer listed beyond the cap, then the search results it did not list.
    assert list(answer.more_item_ids) == ids[MAX_RESULTS:] + not_listed
    assert list(answer.working_set) == ids + not_listed


async def test_the_last_step_forbids_tools_and_the_step_limit_is_an_error(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    search = (SEARCH_ITEMS_TOOL, {"category": "restaurant"})
    under_test = build_agent(
        session_factory, [tool_calls(search), tool_calls(search)], agent_max_steps=2
    )

    with pytest.raises(AgentStepLimitError):
        await under_test.agent.answer(turn(saves, "restaurants"))

    assert [request.tool_choice for request in under_test.generator.requests] == ["auto", "none"]


@pytest.mark.parametrize(
    ("language", "separator", "question"),
    [("ja", "、", "保存した場所"), ("es", " y ", "¿qué restaurantes guardé en Beşiktaş?")],
)
async def test_a_language_without_built_in_wording_gets_translated_wording(
    session_factory: async_sessionmaker[AsyncSession],
    saves: Saves,
    language: str,
    separator: str,
    question: str,
) -> None:
    meze = saves.items["meze"].id
    translated = translated_wording(language, separator)
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "restaurant", "filters": {}})),
            final_answer(language, f"{language}: 1", [meze]),
            structured_reply(translated),
        ],
    )

    answer = await under_test.agent.answer(turn(saves, question))

    assert answer.language == language
    assert answer.messages.wording is not None
    assert answer.messages.wording.list_separator == separator
    wording_request = under_test.generator.requests[-1]
    assert wording_request.tools == ()
    assert f'"language": "{language}"' in str(wording_request.messages[-1].content)


async def test_unusable_translated_wording_is_an_error_not_another_language(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    meze = saves.items["meze"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SEARCH_ITEMS_TOOL, {"category": "restaurant"})),
            final_answer("ko", "1", [meze]),
            structured_reply({"saved_named": "no placeholder"}),
        ],
    )

    with pytest.raises(AgentWordingError):
        await under_test.agent.answer(turn(saves, "베식타시 식당"))


async def test_an_item_without_an_entity_keeps_its_status_on_the_item(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    integrals, pending = saves.items["integrals"].id, saves.items["pending"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls(
                (SET_STATUS_TOOL, {"item_id": str(integrals), "status": "done"}),
                (SET_STATUS_TOOL, {"item_id": str(pending), "status": "want"}),
                (
                    SET_STATUS_TOOL,
                    {"item_id": str(integrals), "status": "want", "entity_position": 0},
                ),
                (GET_ITEM_TOOL, {"item_id": str(integrals)}),
            ),
            final_answer("en", "Marked."),
        ],
    )

    await under_test.agent.answer(turn(saves, "I studied the integrals one; want the other"))

    done, wanted, with_position, details = tool_results(under_test.generator.requests[1])
    assert (done["status"], wanted["status"]) == ("done", "want")
    assert "entity_position" in str(with_position["error"])
    assert details["status"] == "done"
    async with session_factory() as session:
        rows = {
            row.id: (row.user_status, row.status)
            for row in (
                await session.execute(
                    select(SavedItem).where(SavedItem.id.in_([integrals, pending]))
                )
            ).scalars()
        }
    # The user's status and the indexing lifecycle stay apart.
    assert rows == {
        integrals: (EntityStatus.DONE, ItemStatus.DONE),
        pending: (EntityStatus.WANT, ItemStatus.PENDING),
    }


async def test_get_item_shows_the_users_status_of_each_entity(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    inception = saves.items["inception"].id
    under_test = build_agent(
        session_factory,
        [tool_calls((GET_ITEM_TOOL, {"item_id": str(inception)})), final_answer("en", "…")],
    )

    await under_test.agent.answer(turn(saves, "did I watch Inception?"))

    (details,) = tool_results(under_test.generator.requests[1])
    assert details["status"] is None
    assert [entity["status"] for entity in details["entities"]] == [EntityStatus.DONE]  # type: ignore[union-attr]


async def test_an_items_status_is_deleted_with_the_item(
    session_factory: async_sessionmaker[AsyncSession], saves: Saves
) -> None:
    integrals = saves.items["integrals"].id
    under_test = build_agent(
        session_factory,
        [
            tool_calls((SET_STATUS_TOOL, {"item_id": str(integrals), "status": "done"})),
            final_answer("en", "Marked."),
        ],
    )
    await under_test.agent.answer(turn(saves, "I studied the integrals video"))

    assert await UserDataEraser(session_factory, ()).erase(saves.user.id) is not None

    async with session_factory() as session:
        left = await session.scalar(
            select(func.count()).select_from(SavedItem).where(SavedItem.user_status.is_not(None))
        )
    assert left == 0
