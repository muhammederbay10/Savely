"""Place filters from what the reel said, for mentions with no linked entity (SAV-35).

Without a resolver (SAV-15, Phase 2) no place mention has an entity. The area
filter then reads the mention's `area_hint`, split into its levels, and value
filters such as `cuisine` read the mention itself. A linked entity's verified
data still wins for its own mention. Decision record 0017.

These tests need Postgres with PostGIS and pgvector. See `tests/database.py`.
No model is called: the decision model and the embedder are fakes.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.search.conditions import area_hint_separator_pattern
from app.agent.search.fallback import FallbackLevel
from app.agent.search.normalization import NormalizationMethod
from app.agent.search.query import SearchQuery
from app.agent.search.retrieval import SearchResult
from app.db.models import ItemEntity, ItemStatus, SavedItem, User
from tests.agent.retrieval_support import (
    NOW,
    PickingClassifier,
    Saves,
    build_retriever,
    place_fields,
    seed_saves,
)

Venue = tuple[str, str | None, str | None]


@pytest.fixture
async def saves(db_session: AsyncSession) -> Saves:
    return await seed_saves(db_session)


async def add_unresolved_post(
    session: AsyncSession, user: User, name: str, *venues: Venue
) -> SavedItem:
    """A place post whose mentions no resolver has linked to an entity."""
    item = SavedItem(
        user_id=user.id,
        platform="instagram",
        platform_post_id=f"synthetic-unresolved-{user.id.hex[:8]}-{name}",
        type="place",
        category="restaurant",
        collection="Food",
        fields=place_fields(*venues),
        status=ItemStatus.DONE,
        saved_at=NOW,
    )
    session.add(item)
    await session.flush()
    return item


def ids(result: SearchResult) -> list[object]:
    return [hit.item_id for hit in result.hits]


def strict_ids(result: SearchResult) -> list[object]:
    """The items that met every filter; relaxed L2 results are left out."""
    return [hit.item_id for hit in result.hits if hit.level is not FallbackLevel.RELAXED]


def in_area(saves: Saves, area: str, **filters: str) -> SearchQuery:
    return SearchQuery(user_id=saves.user.id, type="place", filters={"area": area, **filters})


def normalized(result: SearchResult, slot: str) -> tuple[str, NormalizationMethod]:
    found = next(item for item in result.normalized if item.slot == slot)
    return found.value, found.method


async def test_an_unresolved_place_is_found_by_its_area_hint_without_a_fallback(
    db_session: AsyncSession, saves: Saves
) -> None:
    reel = await add_unresolved_post(
        db_session, saves.user, "kadikoy", ("Çiya Sofrası", "Kadıköy", "anatolian")
    )

    result = await build_retriever().search(db_session, in_area(saves, "Kadıköy"))

    assert ids(result) == [reel.id]
    assert result.level is FallbackLevel.NORMAL
    assert result.fallbacks == ()
    assert result.relaxed == ()


@pytest.mark.parametrize(
    ("hint", "areas"),
    [
        pytest.param("Kadıköy, İstanbul", ("Kadıköy", "İstanbul"), id="turkish-comma"),
        pytest.param("渋谷、東京", ("渋谷", "東京"), id="japanese-ideographic-comma"),
        pytest.param("الزمالك، القاهرة", ("الزمالك", "القاهرة"), id="arabic-comma"),
        pytest.param("Le Marais; Paris", ("Le Marais", "Paris"), id="semicolon"),
        pytest.param(
            "Saint-Germain-des-Prés,Paris", ("Saint-Germain-des-Prés", "Paris"), id="no-space"
        ),
    ],
)
async def test_every_level_of_a_hint_matches_on_its_own(
    db_session: AsyncSession, saves: Saves, hint: str, areas: tuple[str, ...]
) -> None:
    reel = await add_unresolved_post(db_session, saves.user, "levels", ("Somewhere", hint, None))
    retriever = build_retriever()

    for area in areas:
        result = await retriever.search(db_session, in_area(saves, area))

        assert reel.id in strict_ids(result), area
        assert result.level is FallbackLevel.NORMAL


@pytest.mark.parametrize(
    ("hint", "area"),
    [
        pytest.param("Kadıköy, İstanbul", "Kadıköy, İstanbul", id="whole-hint-is-not-a-level"),
        pytest.param("Saint-Germain-des-Prés", "Saint", id="hyphen-is-not-a-separator"),
        pytest.param("東京都渋谷区", "渋谷", id="no-separator-stays-one-part"),
        pytest.param("Moda", "Mod", id="only-whole-parts-match"),
    ],
)
async def test_an_area_must_equal_a_whole_part_of_the_hint(
    db_session: AsyncSession, saves: Saves, hint: str, area: str
) -> None:
    reel = await add_unresolved_post(db_session, saves.user, "parts", ("Somewhere", hint, None))

    result = await build_retriever().search(db_session, in_area(saves, area))

    assert reel.id not in strict_ids(result)


async def test_normalization_offers_the_parts_of_the_users_hints(
    db_session: AsyncSession, saves: Saves
) -> None:
    reel = await add_unresolved_post(
        db_session, saves.user, "kadikoy", ("Çiya Sofrası", "Kadıköy, İstanbul", None)
    )
    classifier = PickingClassifier(picks={("area", "kadikoy"): "Kadıköy"})

    result = await build_retriever(classifier=classifier).search(
        db_session, in_area(saves, "kadikoy")
    )

    assert normalized(result, "filters.area") == ("Kadıköy", NormalizationMethod.DECISION)
    assert ids(result) == [reel.id]
    assert result.level is FallbackLevel.NORMAL
    assert {"Kadıköy", "İstanbul"} <= classifier.offered_values
    assert "Kadıköy, İstanbul" not in classifier.offered_values


async def test_a_hint_in_another_case_is_an_exact_hit_without_a_decision(
    db_session: AsyncSession, saves: Saves
) -> None:
    reel = await add_unresolved_post(db_session, saves.user, "shibuya", ("Bar", "渋谷、東京", None))
    other = await add_unresolved_post(db_session, saves.user, "moda", ("Kafe", "Moda", None))
    classifier = PickingClassifier()
    retriever = build_retriever(classifier=classifier)

    tokyo = await retriever.search(db_session, in_area(saves, " 東京 "))
    moda = await retriever.search(db_session, in_area(saves, "MODA"))

    assert normalized(tokyo, "filters.area") == ("東京", NormalizationMethod.SAVED_VALUE)
    assert ids(tokyo) == [reel.id]
    assert normalized(moda, "filters.area") == ("Moda", NormalizationMethod.SAVED_VALUE)
    assert ids(moda) == [other.id]
    assert classifier.calls == []


async def test_cuisine_matches_an_unresolved_mention(
    db_session: AsyncSession, saves: Saves
) -> None:
    reel = await add_unresolved_post(
        db_session, saves.user, "ramen", ("Menya", "中目黒、東京", "ramen")
    )
    retriever = build_retriever()

    by_cuisine = await retriever.search(
        db_session, SearchQuery(user_id=saves.user.id, type="place", filters={"cuisine": "ramen"})
    )
    by_both = await retriever.search(db_session, in_area(saves, "中目黒", cuisine="ramen"))

    assert ids(by_cuisine) == [reel.id]
    assert by_cuisine.level is FallbackLevel.NORMAL
    assert ids(by_both) == [reel.id]
    assert by_both.level is FallbackLevel.NORMAL


async def test_a_list_post_never_combines_two_unresolved_mentions(
    db_session: AsyncSession, saves: Saves
) -> None:
    listing = await add_unresolved_post(
        db_session,
        saves.user,
        "two-venues",
        ("Seoul Table", "Moda, Kadıköy", "korean barbecue"),
        ("Sea Café", "Bebek", "specialty coffee"),
    )
    retriever = build_retriever()

    split = await retriever.search(db_session, in_area(saves, "Bebek", cuisine="korean barbecue"))
    same = await retriever.search(db_session, in_area(saves, "Moda", cuisine="korean barbecue"))

    assert listing.id not in strict_ids(split)
    assert ids(same) == [listing.id]
    assert same.level is FallbackLevel.NORMAL


async def test_a_resolved_and_an_unresolved_mention_cannot_split_the_filters(
    db_session: AsyncSession, saves: Saves
) -> None:
    # Mention 0 is linked to Kronotrop in Bebek; mention 1, a korean place in Moda, is not.
    listing = await add_unresolved_post(
        db_session,
        saves.user,
        "mixed",
        ("Kronotrop", "Bebek", "specialty coffee"),
        ("Seoul Table", "Moda", "korean barbecue"),
    )
    db_session.add(
        ItemEntity(saved_item_id=listing.id, entity_id=saves.entities["kronotrop"].id, position=0)
    )
    await db_session.flush()
    retriever = build_retriever()

    split = await retriever.search(db_session, in_area(saves, "Bebek", cuisine="korean barbecue"))
    unresolved = await retriever.search(
        db_session, in_area(saves, "Moda", cuisine="korean barbecue")
    )
    resolved = await retriever.search(
        db_session, in_area(saves, "Beşiktaş", cuisine="specialty coffee")
    )

    assert listing.id not in strict_ids(split)
    assert ids(unresolved) == [listing.id]
    assert listing.id in strict_ids(resolved)


async def test_a_linked_entitys_address_wins_over_its_mentions_hint(
    db_session: AsyncSession, saves: Saves
) -> None:
    # The reel said Kadıköy, but the resolver linked the mention to Kronotrop in Bebek.
    reel = await add_unresolved_post(
        db_session, saves.user, "resolved", ("Kronotrop", "Kadıköy", None)
    )
    db_session.add(
        ItemEntity(saved_item_id=reel.id, entity_id=saves.entities["kronotrop"].id, position=0)
    )
    await db_session.flush()
    classifier = PickingClassifier(picks={("area", "kadikoy"): "Kadıköy"})
    retriever = build_retriever(classifier=classifier)

    by_hint = await retriever.search(db_session, in_area(saves, "Kadıköy"))
    by_address = await retriever.search(db_session, in_area(saves, "Bebek"))
    misspelled = await retriever.search(db_session, in_area(saves, "kadikoy"))

    assert reel.id not in strict_ids(by_hint)
    assert reel.id in strict_ids(by_address)
    assert "Kadıköy" not in classifier.offered_values
    assert normalized(misspelled, "filters.area") == ("kadikoy", NormalizationMethod.UNCHANGED)


async def test_another_users_hints_are_never_used(db_session: AsyncSession, saves: Saves) -> None:
    await add_unresolved_post(
        db_session, saves.other_user, "theirs", ("Hidden Place", "Cihangir, İstanbul", "meyhane")
    )
    await add_unresolved_post(db_session, saves.user, "mine", ("Kafe", "Moda", None))
    classifier = PickingClassifier(picks={("area", "cihangr"): "Cihangir"})
    retriever = build_retriever(classifier=classifier)

    exact = await retriever.search(db_session, in_area(saves, "Cihangir"))
    misspelled = await retriever.search(db_session, in_area(saves, "cihangr"))
    by_cuisine = await retriever.search(
        db_session,
        SearchQuery(user_id=saves.user.id, type="place", filters={"cuisine": "meyhane"}),
    )

    assert strict_ids(exact) == []
    assert normalized(misspelled, "filters.area") == ("cihangr", NormalizationMethod.UNCHANGED)
    assert strict_ids(by_cuisine) == []
    assert "Cihangir" not in classifier.offered_values
    assert "Moda" in classifier.offered_values


async def test_relaxing_keeps_a_hint_area_that_is_no_known_district(
    db_session: AsyncSession, saves: Saves
) -> None:
    reel = await add_unresolved_post(
        db_session, saves.user, "kadikoy", ("Çiya Sofrası", "Kadıköy", "anatolian")
    )

    result = await build_retriever().search(db_session, in_area(saves, "Kadıköy", cuisine="pizza"))

    assert result.level is FallbackLevel.RELAXED
    assert ids(result) == [reel.id]
    assert result.relaxed == ("filters.cuisine",)


async def test_widening_a_linked_district_keeps_hints_of_that_district_only(
    db_session: AsyncSession, saves: Saves
) -> None:
    # Kronotrop is linked in district Beşiktaş, city İstanbul, so L2 widens Beşiktaş to
    # İstanbul for entities. A hint of Beşiktaş must still count; hints of other parts
    # of İstanbul, or of the city itself, must not.
    reel = await add_unresolved_post(
        db_session, saves.user, "besiktas", ("Lokanta", "Beşiktaş", "anatolian")
    )
    elsewhere = await add_unresolved_post(
        db_session, saves.user, "beyoglu", ("Meyhane", "Beyoğlu", "anatolian")
    )
    city = await add_unresolved_post(
        db_session, saves.user, "istanbul", ("Kebapçı", "İstanbul", "anatolian")
    )
    theirs = await add_unresolved_post(
        db_session, saves.other_user, "besiktas", ("Gizli", "Beşiktaş", "anatolian")
    )

    result = await build_retriever().search(db_session, in_area(saves, "Beşiktaş", cuisine="pizza"))

    assert result.level is FallbackLevel.RELAXED
    assert set(result.relaxed) == {"filters.area", "filters.cuisine"}
    found = ids(result)
    assert reel.id in found
    assert saves.items["kronotrop"].id in found
    assert saves.items["kimchi"].id in found
    assert elsewhere.id not in found
    assert city.id not in found
    assert theirs.id not in found


async def test_a_mention_without_a_hint_never_matches_an_area(
    db_session: AsyncSession, saves: Saves
) -> None:
    reel = await add_unresolved_post(db_session, saves.user, "no-hint", ("Kafe", None, "tea"))

    by_area = await build_retriever().search(db_session, in_area(saves, "Kadıköy"))
    by_cuisine = await build_retriever().search(
        db_session, SearchQuery(user_id=saves.user.id, type="place", filters={"cuisine": "tea"})
    )

    assert reel.id not in strict_ids(by_area)
    assert ids(by_cuisine) == [reel.id]


@pytest.mark.parametrize(
    "separator",
    # Written as escapes: fullwidth marks look like ASCII ones in an editor.
    [",", ";", "\u060c", "\u3001", "\uff0c", "\uff1b", "\uff64"],
)
def test_commas_and_semicolons_of_every_script_separate_levels(separator: str) -> None:
    assert f"\\U{ord(separator):08x}" in area_hint_separator_pattern()


@pytest.mark.parametrize("mark", ["-", "/", "'", ".", "·", "&"])
def test_marks_inside_names_do_not_separate_levels(mark: str) -> None:
    assert f"\\U{ord(mark):08x}" not in area_hint_separator_pattern()
