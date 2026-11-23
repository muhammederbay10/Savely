"""The registry lists the built-in types, and a type's declaration must be coherent."""

import ast
from pathlib import Path

import pytest

from app.types.base import (
    ContentType,
    ContentTypeDefinitionError,
    EntityExtraction,
    EntityMention,
    Extraction,
    FilterKind,
    FilterSource,
    SearchFilter,
)
from app.types.registry import CONTENT_TYPES, ContentTypeRegistry, UnknownContentTypeError

BUILT_IN_TYPE_NAMES = ["place", "movie", "study"]

APP_DIR = Path(__file__).parents[2] / "app"
# The only folders that may hold logic for one specific content type (ARCHITECTURE §6).
CONTENT_TYPE_LOGIC_DIRS = (APP_DIR / "types", APP_DIR / "resolvers")


class RecipeExtraction(Extraction):
    """A schema for a made-up type without entities."""

    dish: str


class BookMention(EntityMention):
    """A mention for a made-up type with entities."""

    title: str


class BookExtraction(EntityExtraction):
    """A schema for a made-up type with entities."""

    entities: tuple[BookMention, ...]


def make_content_type(
    *,
    name: str = "recipe",
    extraction_schema: type[Extraction] = RecipeExtraction,
    resolver: str | None = None,
    search_filters: tuple[SearchFilter, ...] = (),
) -> ContentType:
    return ContentType(
        name=name,
        description="A made-up type for tests.",
        extraction_schema=extraction_schema,
        resolver=resolver,
        search_filters=search_filters,
    )


def search_filter(type_name: str, filter_name: str) -> SearchFilter:
    matches = [
        candidate
        for candidate in CONTENT_TYPES.get(type_name).search_filters
        if candidate.name == filter_name
    ]
    assert len(matches) == 1
    return matches[0]


def test_registry_lists_every_built_in_type() -> None:
    assert [content_type.name for content_type in CONTENT_TYPES.all()] == BUILT_IN_TYPE_NAMES


@pytest.mark.parametrize("type_name", BUILT_IN_TYPE_NAMES)
def test_registry_returns_a_type_by_its_name(type_name: str) -> None:
    assert CONTENT_TYPES.get(type_name).name == type_name


def test_unknown_type_name_fails_and_names_the_registered_types() -> None:
    with pytest.raises(UnknownContentTypeError, match=r"'podcast'.*place, movie, study"):
        CONTENT_TYPES.get("podcast")


def test_there_is_no_catch_all_type() -> None:
    with pytest.raises(UnknownContentTypeError):
        CONTENT_TYPES.get("other")


def test_a_post_without_a_built_in_type_uses_the_shared_schema() -> None:
    assert CONTENT_TYPES.extraction_schema(None) is Extraction


@pytest.mark.parametrize("type_name", BUILT_IN_TYPE_NAMES)
def test_a_built_in_type_extends_the_shared_schema(type_name: str) -> None:
    schema = CONTENT_TYPES.extraction_schema(type_name)

    assert schema is CONTENT_TYPES.get(type_name).extraction_schema
    assert issubclass(schema, Extraction)
    assert schema is not Extraction


def test_schema_for_an_unknown_type_name_fails() -> None:
    with pytest.raises(UnknownContentTypeError, match="'podcast'"):
        CONTENT_TYPES.extraction_schema("podcast")


def test_a_new_type_is_found_once_it_is_registered() -> None:
    recipe = make_content_type(name="recipe")

    registry = ContentTypeRegistry((*CONTENT_TYPES.all(), recipe))

    assert registry.get("recipe") is recipe
    assert registry.extraction_schema("recipe") is RecipeExtraction
    assert [content_type.name for content_type in registry.all()] == [
        *BUILT_IN_TYPE_NAMES,
        "recipe",
    ]


def test_registering_the_same_name_twice_is_rejected() -> None:
    with pytest.raises(ContentTypeDefinitionError, match="'recipe' is registered twice"):
        ContentTypeRegistry((make_content_type(name="recipe"), make_content_type(name="recipe")))


@pytest.mark.parametrize("type_name", BUILT_IN_TYPE_NAMES)
def test_every_type_describes_itself_for_the_type_decision(type_name: str) -> None:
    assert CONTENT_TYPES.get(type_name).description.strip()


@pytest.mark.parametrize(
    ("type_name", "resolver"),
    [("place", "google_places"), ("movie", "tmdb"), ("study", None)],
)
def test_type_declares_its_resolver(type_name: str, resolver: str | None) -> None:
    assert CONTENT_TYPES.get(type_name).resolver == resolver


@pytest.mark.parametrize(
    ("type_name", "expected_filters"),
    [
        ("place", ["area", "near", "open_now", "cuisine", "price"]),
        ("movie", ["genre", "year", "streaming", "watched"]),
        ("study", ["subject", "topic", "format", "exam"]),
    ],
)
def test_type_offers_the_filters_from_the_design(
    type_name: str, expected_filters: list[str]
) -> None:
    names = [candidate.name for candidate in CONTENT_TYPES.get(type_name).search_filters]

    assert names == expected_filters


@pytest.mark.parametrize("type_name", BUILT_IN_TYPE_NAMES)
def test_every_filter_is_described_for_the_response_agent(type_name: str) -> None:
    for candidate in CONTENT_TYPES.get(type_name).search_filters:
        assert candidate.description.strip()


def test_place_is_filtered_by_an_area_at_any_level_not_by_a_district_list() -> None:
    place_filters = {candidate.name for candidate in CONTENT_TYPES.get("place").search_filters}

    assert search_filter("place", "area").kind is FilterKind.AREA
    assert "district" not in place_filters


@pytest.mark.parametrize(
    ("type_name", "filter_name", "vocabulary"),
    [
        ("place", "cuisine", "cuisine"),
        ("movie", "genre", "genre"),
        ("study", "subject", "subject"),
        ("study", "format", "format"),
        ("study", "exam", "exam"),
    ],
)
def test_open_value_filter_declares_its_vocabulary_field(
    type_name: str, filter_name: str, vocabulary: str
) -> None:
    declared = search_filter(type_name, filter_name)

    assert declared.kind is FilterKind.VALUE
    assert declared.vocabulary == vocabulary


@pytest.mark.parametrize(("type_name", "filter_name"), [("place", "price"), ("movie", "streaming")])
def test_filter_with_resolver_values_is_open_but_outside_the_vocabulary(
    type_name: str, filter_name: str
) -> None:
    declared = search_filter(type_name, filter_name)

    assert declared.kind is FilterKind.VALUE
    assert declared.vocabulary is None


@pytest.mark.parametrize(
    ("type_name", "filter_name", "kind"),
    [
        ("place", "near", FilterKind.LOCATION),
        ("place", "open_now", FilterKind.BOOLEAN),
        ("movie", "year", FilterKind.INTEGER),
        ("movie", "watched", FilterKind.BOOLEAN),
        ("study", "topic", FilterKind.TEXT),
    ],
)
def test_other_filter_declares_its_kind(type_name: str, filter_name: str, kind: FilterKind) -> None:
    declared = search_filter(type_name, filter_name)

    assert declared.kind is kind
    assert declared.vocabulary is None


@pytest.mark.parametrize(
    "kind",
    [
        FilterKind.AREA,
        FilterKind.LOCATION,
        FilterKind.BOOLEAN,
        FilterKind.INTEGER,
        FilterKind.TEXT,
    ],
)
def test_only_an_open_value_filter_can_use_the_vocabulary(kind: FilterKind) -> None:
    with pytest.raises(ContentTypeDefinitionError, match="'diet' is not an open value"):
        SearchFilter(name="diet", kind=kind, description="The diet.", vocabulary="diet")


def test_the_place_area_filter_reads_the_mentions_area_hint() -> None:
    assert search_filter("place", "area").mention_field == "area_hint"


@pytest.mark.parametrize(
    "kind",
    [
        FilterKind.VALUE,
        FilterKind.LOCATION,
        FilterKind.BOOLEAN,
        FilterKind.INTEGER,
        FilterKind.TEXT,
    ],
)
def test_only_an_area_filter_can_name_a_mention_field(kind: FilterKind) -> None:
    with pytest.raises(ContentTypeDefinitionError, match="'shelf' is not an area"):
        SearchFilter(name="shelf", kind=kind, description="The shelf.", mention_field="shelf")


def test_a_mention_field_must_be_a_field_of_the_types_mentions() -> None:
    area = SearchFilter(
        name="area", kind=FilterKind.AREA, description="Where.", mention_field="area_hint"
    )

    with pytest.raises(ContentTypeDefinitionError, match="reads 'area_hint'"):
        make_content_type(
            name="book",
            extraction_schema=BookExtraction,
            resolver="open_library",
            search_filters=(area,),
        )
    with pytest.raises(ContentTypeDefinitionError, match="reads 'area_hint'"):
        make_content_type(search_filters=(area,))


def test_a_mention_field_of_the_types_mentions_is_accepted() -> None:
    title = SearchFilter(
        name="area", kind=FilterKind.AREA, description="Where.", mention_field="title"
    )

    book = make_content_type(
        name="book",
        extraction_schema=BookExtraction,
        resolver="open_library",
        search_filters=(title,),
    )

    assert book.search_filters == (title,)


def test_type_rejects_two_filters_with_the_same_name() -> None:
    filters = (
        SearchFilter(name="time", kind=FilterKind.INTEGER, description="Minutes to cook."),
        SearchFilter(name="time", kind=FilterKind.TEXT, description="When to eat it."),
    )

    with pytest.raises(
        ContentTypeDefinitionError, match="Duplicate filters in content type 'recipe': time"
    ):
        make_content_type(search_filters=filters)


def test_type_with_a_resolver_must_extract_entities() -> None:
    with pytest.raises(ContentTypeDefinitionError, match="must extend EntityExtraction"):
        make_content_type(extraction_schema=RecipeExtraction, resolver="recipe_site")


def test_type_that_extracts_entities_must_have_a_resolver() -> None:
    with pytest.raises(ContentTypeDefinitionError, match="needs a resolver"):
        make_content_type(name="book", extraction_schema=BookExtraction, resolver=None)


def test_type_with_entities_and_a_resolver_is_accepted() -> None:
    book = make_content_type(name="book", extraction_schema=BookExtraction, resolver="open_library")

    assert book.resolver == "open_library"


def string_constants(node: ast.expr) -> set[str]:
    """Collect the string literals in an expression, looking inside tuples, lists and sets."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return set().union(*(string_constants(element) for element in node.elts))
    return set()


def lines_comparing_against(source: str, names: set[str]) -> list[int]:
    """Find the lines that compare something with one of `names`, or `match` on one."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            compared = string_constants(ast.Tuple(elts=[node.left, *node.comparators]))
            if names & compared:
                lines.append(node.lineno)
        elif isinstance(node, ast.MatchValue) and names & string_constants(node.value):
            lines.append(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize(
    "source",
    [
        'if item_type == "movie":\n    pass\n',
        'if "place" != item_type:\n    pass\n',
        'needs_entity = item_type in ("place", "movie")\n',
        'match item_type:\n    case "study":\n        pass\n',
    ],
)
def test_guard_spots_a_comparison_with_a_type_name(source: str) -> None:
    assert lines_comparing_against(source, set(BUILT_IN_TYPE_NAMES))


@pytest.mark.parametrize(
    "source",
    [
        'content_type = CONTENT_TYPES.get("movie")\n',
        'if media_type == "reel":\n    pass\n',
        "if content_type.resolver is not None:\n    pass\n",
    ],
)
def test_guard_allows_code_that_reads_the_registry(source: str) -> None:
    assert lines_comparing_against(source, set(BUILT_IN_TYPE_NAMES)) == []


def test_no_code_outside_the_type_modules_branches_on_a_type_name() -> None:
    type_names = {content_type.name for content_type in CONTENT_TYPES.all()}
    violations = [
        f"{path.relative_to(APP_DIR.parent)}:{line}"
        for path in sorted(APP_DIR.rglob("*.py"))
        if not any(path.is_relative_to(allowed) for allowed in CONTENT_TYPE_LOGIC_DIRS)
        for line in lines_comparing_against(path.read_text(encoding="utf-8"), type_names)
    ]

    assert violations == [], (
        "Branching on a content type's name belongs in app/types/ or app/resolvers/. "
        "Read the type's declaration from the registry instead."
    )


@pytest.mark.parametrize(
    ("type_name", "filter_name", "source"),
    [
        ("movie", "streaming", FilterSource.COUNTRY_AVAILABILITY),
        ("movie", "watched", FilterSource.USER_STATUS),
        ("place", "open_now", FilterSource.OPENING_HOURS),
        ("place", "area", FilterSource.SAVED_VALUES),
        ("movie", "genre", FilterSource.SAVED_VALUES),
        ("study", "subject", FilterSource.SAVED_VALUES),
    ],
)
def test_filter_declares_where_search_reads_its_values(
    type_name: str, filter_name: str, source: FilterSource
) -> None:
    filters = {item.name: item for item in CONTENT_TYPES.get(type_name).search_filters}

    assert filters[filter_name].source is source


@pytest.mark.parametrize(
    ("kind", "source"),
    [
        (FilterKind.BOOLEAN, FilterSource.COUNTRY_AVAILABILITY),
        (FilterKind.AREA, FilterSource.COUNTRY_AVAILABILITY),
        (FilterKind.VALUE, FilterSource.USER_STATUS),
        (FilterKind.INTEGER, FilterSource.OPENING_HOURS),
    ],
)
def test_a_source_only_serves_the_kinds_it_can_check(
    kind: FilterKind, source: FilterSource
) -> None:
    with pytest.raises(ContentTypeDefinitionError, match="'served' is a"):
        SearchFilter(name="served", kind=kind, description="Anything.", source=source)
