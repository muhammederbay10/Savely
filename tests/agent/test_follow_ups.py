"""Follow-ups across messages: "the second one", "show more", "only the cheap ones", the profile.

Each test is a conversation of several DMs through the chat worker's responder,
with short-term memory in Redis between them (ARCHITECTURE §3.7). The text
model is scripted, so the tests check what the model is shown and what the
tools and stores do with its calls, not the model's judgement. They need
Postgres (`TEST_DATABASE_ADMIN_URL`) and Redis (`TEST_REDIS_URL`).
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from sqlalchemy import select

from app.agent.chat import ChatOutcome
from app.agent.tools import (
    SEARCH_ITEMS_TOOL,
    SET_PROFILE_TOOL,
    SET_STATUS_TOOL,
    SearchUsed,
)
from app.db.models import EntityStatus, ItemEntity, SavedItem, UserEntity
from app.models.base import GenerationRequest, GenerationResult
from tests.agent.agent_support import final_answer, tool_calls, tool_results
from tests.agent.chat_support import (
    ChatStores,
    ChatUnderTest,
    build_chat,
    seed_place,
    stored_user,
)

SENDER_ID = "1000000000000181"
BESIKTAS_SEARCH = {"type": "place", "category": "restaurant", "filters": {"area": "Beşiktaş"}}
CHEAP_BESIKTAS_SEARCH = {
    "type": "place",
    "category": "restaurant",
    "filters": {"area": "Beşiktaş", "price": "cheap"},
}


@dataclass
class Restaurants:
    user_id: uuid.UUID
    items: dict[str, SavedItem]

    def id(self, name: str) -> uuid.UUID:
        return self.items[name].id


@pytest.fixture
async def restaurants(chat_stores: ChatStores) -> Restaurants:
    """Three saved restaurants in one district, two of them cheap, and one elsewhere."""
    user_id = await chat_stores.new_user(SENDER_ID)
    places = {
        "lahmacun": ("Beşiktaş", "cheap", 1),
        "fish": ("Beşiktaş", "expensive", 2),
        "kofte": ("Beşiktaş", "cheap", 3),
        "noodles": ("Kadıköy", "cheap", 4),
    }
    items = {
        name: await seed_place(
            chat_stores, user_id, name, district=district, price=price, saved_days_ago=days
        )
        for name, (district, price, days) in places.items()
    }
    return Restaurants(user_id=user_id, items=items)


def system_texts(request: GenerationRequest) -> list[str]:
    return [str(message.content) for message in request.messages if message.role == "system"]


def found_ids(request: GenerationRequest) -> list[uuid.UUID]:
    """The items of the last search result the model was shown, in order."""
    results = tool_results(request)[-1]["results"]
    return [uuid.UUID(result["item_id"]) for result in results]  # type: ignore[attr-defined]


def answer_order(restaurants: Restaurants) -> list[uuid.UUID]:
    """The order the first answer lists the three Beşiktaş restaurants in, as the model chose."""
    return [restaurants.id("fish"), restaurants.id("kofte"), restaurants.id("lahmacun")]


async def first_search(
    chat_stores: ChatStores,
    restaurants: Restaurants,
    later_turns: Sequence[GenerationResult],
    **agent_settings: int,
) -> ChatUnderTest:
    """Turn 1: "restaurants in Beşiktaş"; `later_turns` scripts the model for the next ones."""
    chat = build_chat(
        chat_stores,
        [
            tool_calls((SEARCH_ITEMS_TOOL, BESIKTAS_SEARCH)),
            final_answer("en", "Restaurants in Beşiktaş:", answer_order(restaurants)),
            *later_turns,
        ],
        **agent_settings,
    )
    receipt = await chat_stores.send_text(restaurants.user_id, "restaurants in Beşiktaş")
    outcome = await chat.responder.respond(restaurants.user_id, receipt, last_try=False)
    assert outcome is ChatOutcome.ANSWERED
    return chat


def quoted_messages(chat: ChatUnderTest) -> list[str]:
    """The saved messages each result reply quoted, in the order they were sent."""
    return [
        str(body["reply_to"]["mid"])  # type: ignore[index]
        for body in chat.sent_bodies()
        if "reply_to" in body
    ]


async def test_the_second_one_resolves_to_the_second_result_the_user_was_sent(
    chat_stores: ChatStores, restaurants: Restaurants
) -> None:
    order = answer_order(restaurants)
    second = order[1]
    chat = await first_search(
        chat_stores,
        restaurants,
        [
            tool_calls((SET_STATUS_TOOL, {"item_id": str(second), "status": "done"})),
            final_answer("en", "Marked it as visited."),
        ],
    )
    assert quoted_messages(chat)[1] == restaurants_by_id(restaurants)[second].source_message_id

    receipt = await chat_stores.send_text(restaurants.user_id, "I went to the second one")
    outcome = await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    turn_two = chat.agent.generator.requests[2]
    working_set = next(text for text in system_texts(turn_two) if "saw last" in text)
    assert f"1. {order[0]}\n2. {second}\n3. {order[2]}" in working_set
    conversation = [message.content for message in turn_two.messages if message.role != "system"]
    # The history from short-term memory, then the new message; the turn's tool calls follow.
    assert conversation[:3] == [
        "restaurants in Beşiktaş",
        "Restaurants in Beşiktaş:",
        "I went to the second one",
    ]
    async with chat_stores.sessions() as session:
        entity_id = await session.scalar(
            select(ItemEntity.entity_id).where(ItemEntity.saved_item_id == second)
        )
        status = await session.scalar(
            select(UserEntity.status).where(
                UserEntity.user_id == restaurants.user_id, UserEntity.entity_id == entity_id
            )
        )
    assert status == EntityStatus.DONE


async def test_only_the_cheap_ones_repeats_the_last_search_and_narrows_its_results(
    chat_stores: ChatStores, restaurants: Restaurants
) -> None:
    before = answer_order(restaurants)
    cheap = [restaurants.id("kofte"), restaurants.id("lahmacun")]
    chat = await first_search(
        chat_stores,
        restaurants,
        [
            tool_calls((SEARCH_ITEMS_TOOL, CHEAP_BESIKTAS_SEARCH)),
            final_answer("en", "The cheap ones:", cheap),
        ],
    )
    stored = (await chat_stores.memory.load(restaurants.user_id)).working_set
    assert stored is not None
    assert stored.item_ids == tuple(before)
    assert stored.searches == (SearchUsed(SEARCH_ITEMS_TOOL, BESIKTAS_SEARCH),)

    receipt = await chat_stores.send_text(restaurants.user_id, "only the cheap ones")
    outcome = await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    turn_two = chat.agent.generator.requests[2]
    searches = next(text for text in system_texts(turn_two) if "came from these searches" in text)
    assert '"filters": {"area": "Beşiktaş"}' in searches
    after = found_ids(chat.agent.generator.requests[3])
    assert set(after) == set(cheap)
    assert set(after) < set(before)
    narrowed = (await chat_stores.memory.load(restaurants.user_id)).working_set
    assert narrowed is not None
    assert narrowed.item_ids == tuple(cheap)
    assert narrowed.searches == (SearchUsed(SEARCH_ITEMS_TOOL, CHEAP_BESIKTAS_SEARCH),)


async def test_show_more_lists_the_results_that_were_not_shown_yet(
    chat_stores: ChatStores, restaurants: Restaurants
) -> None:
    order = answer_order(restaurants)
    chat = await first_search(
        chat_stores,
        restaurants,
        [final_answer("en", "One more:", [order[2]])],
        agent_max_results_per_answer=2,
    )
    assert len(quoted_messages(chat)) == 2

    receipt = await chat_stores.send_text(restaurants.user_id, "show more")
    outcome = await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    assert outcome is ChatOutcome.ANSWERED
    turn_two = chat.agent.generator.requests[2]
    working_set = next(text for text in system_texts(turn_two) if "saw last" in text)
    assert "The first 2 were shown" in working_set
    assert f"3. {order[2]}" in working_set
    assert quoted_messages(chat)[-1] == restaurants_by_id(restaurants)[order[2]].source_message_id
    shown_now = (await chat_stores.memory.load(restaurants.user_id)).working_set
    assert shown_now is not None
    assert shown_now.item_ids == (order[2],)
    # The result still comes from the first search, so a narrowing follow-up can repeat it.
    assert shown_now.searches == (SearchUsed(SEARCH_ITEMS_TOOL, BESIKTAS_SEARCH),)


async def test_a_reply_without_results_keeps_the_last_results_for_a_later_follow_up(
    chat_stores: ChatStores, restaurants: Restaurants
) -> None:
    chat = await first_search(chat_stores, restaurants, [final_answer("en", "You're welcome!")])

    receipt = await chat_stores.send_text(restaurants.user_id, "thanks")
    await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    kept = (await chat_stores.memory.load(restaurants.user_id)).working_set
    assert kept is not None
    assert kept.item_ids == tuple(answer_order(restaurants))


async def test_where_the_user_lives_is_kept_from_chat_and_shown_on_later_turns(
    chat_stores: ChatStores, restaurants: Restaurants
) -> None:
    chat = build_chat(
        chat_stores,
        [
            tool_calls((SET_PROFILE_TOOL, {"home_area": "Beşiktaş", "country": "tr"})),
            final_answer("tr", "Tamam, Beşiktaş'ta yaşadığını not ettim."),
            final_answer("tr", "Merhaba!"),
        ],
    )
    receipt = await chat_stores.send_text(restaurants.user_id, "Beşiktaş'ta yaşıyorum")
    await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    (saved,) = tool_results(chat.agent.generator.requests[1])
    assert saved == {"saved": {"home_area": "Beşiktaş", "country": "TR"}}
    user = await stored_user(chat_stores, restaurants.user_id)
    assert user is not None
    assert (user.home_area, user.country, user.language) == ("Beşiktaş", "TR", "tr")

    receipt = await chat_stores.send_text(restaurants.user_id, "evime yakın restoranlar")
    await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    turn_two = chat.agent.generator.requests[2]
    assert "The user lives in: Beşiktaş." in system_texts(turn_two)[1]


async def test_a_country_given_in_chat_is_used_by_a_search_in_the_same_turn(
    chat_stores: ChatStores, restaurants: Restaurants
) -> None:
    streaming = {"type": "movie", "filters": {"streaming": "Netflix"}}
    chat = build_chat(
        chat_stores,
        [
            tool_calls((SEARCH_ITEMS_TOOL, streaming)),
            tool_calls((SET_PROFILE_TOOL, {"country": "TR"}), (SEARCH_ITEMS_TOOL, streaming)),
            final_answer("en", "Nothing on Netflix in your saves."),
        ],
    )
    receipt = await chat_stores.send_text(restaurants.user_id, "movies on Netflix? I'm in Turkey")

    await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    asked, saved, searched = tool_results(chat.agent.generator.requests[2])
    assert asked["ask_user_for"] == "country"
    assert saved == {"saved": {"country": "TR"}}
    assert "error" not in searched


@pytest.mark.parametrize(
    "arguments",
    [{}, {"country": "Turkey"}, {"home_area": " "}, {"user_id": str(uuid.uuid4())}],
    ids=["nothing", "not-a-code", "blank", "another-user"],
)
async def test_a_wrong_profile_change_is_an_error_the_model_can_correct(
    chat_stores: ChatStores, restaurants: Restaurants, arguments: dict[str, object]
) -> None:
    chat = build_chat(
        chat_stores,
        [tool_calls((SET_PROFILE_TOOL, arguments)), final_answer("en", "Where do you live?")],
    )
    receipt = await chat_stores.send_text(restaurants.user_id, "I live somewhere")

    await chat.responder.respond(restaurants.user_id, receipt, last_try=False)

    (result,) = tool_results(chat.agent.generator.requests[1])
    assert "Invalid arguments" in str(result["error"])
    user = await stored_user(chat_stores, restaurants.user_id)
    assert user is not None
    assert (user.home_area, user.country) == (None, None)


def restaurants_by_id(restaurants: Restaurants) -> dict[uuid.UUID, SavedItem]:
    return {item.id: item for item in restaurants.items.values()}
