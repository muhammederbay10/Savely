"""SQL conditions on `saved_item` for each part of a search, and the orders results come in.

Each condition is checked on one user's items. Type filters are checked per
mention (`fields.entities`), so a list post matches when any one of its places
or movies meets them all. A mention with a linked entity is judged by the
entity's verified data; a mention without one by what the post said, such as
its area hint (decision record 0017). Where a filter reads its values is
declared by the content type (`FilterSource`); nothing here names a type or a
filter.

Values come normalized (`filter_normalization`), so they are compared exactly,
the way they are stored, and the JSON indexes can be used.
"""

import sys
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import Any, Final

from sqlalchemy import (
    Boolean,
    ColumnElement,
    Float,
    Text,
    and_,
    any_,
    bindparam,
    case,
    cast,
    column,
    false,
    func,
    literal,
    literal_column,
    not_,
    or_,
    select,
    true,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSQUERY
from sqlalchemy.orm import QueryableAttribute
from sqlalchemy.sql.selectable import TableValuedAlias

from app.agent.search.query import FilterValue, Near
from app.db.models import (
    FULL_TEXT_SEARCH_CONFIG,
    Entity,
    EntityAvailability,
    EntityStatus,
    ItemEntity,
    ItemStatus,
    SavedItem,
    UserEntity,
    saved_item_search_vector,
)
from app.db.types import GeographyPoint, Vector
from app.types.base import MENTIONS_FIELD, FilterKind, FilterSource, SearchFilter

# The address levels of a place, from the widest to the narrowest. An area filter
# matches any of them (record 0005).
ADDRESS_LEVELS: Final = (
    Entity.country,
    Entity.region,
    Entity.city,
    Entity.district,
    Entity.neighbourhood,
)
# The levels L2 widens an area from, to the city they are in.
WIDENED_LEVELS: Final = (Entity.district, Entity.neighbourhood)

# The key of a post's mentions in `saved_item.fields`: `EntityExtraction.entities`.
MENTIONS_KEY: Final = MENTIONS_FIELD

# The offer kinds of a providers object (`{"<offer kind>": ["<provider name>", …]}`, TMDB's
# watch-provider kinds) under which a title counts as "on" a platform: watchable without
# paying extra. Rent and buy offers stay stored for replies but never satisfy a platform
# filter. Owner decision on SAV-16, 18 Nov 2026 (comment 01a1efd3-a638-7119-b6ed-616d0f2c9576).
WATCHABLE_OFFER_KINDS: Final = ("flatrate", "free", "ads")
_LIKE_ESCAPE: Final = "\\"

# The words in the Unicode name of a punctuation mark that separates the levels of an
# area hint: every comma and semicolon of every script, e.g. "," in "Kadıköy, İstanbul",
# "、" in "渋谷、東京" or "،" in Arabic. Hyphens, slashes, apostrophes and full stops
# are not separators, because they appear inside names ("Saint-Germain-des-Prés",
# "St. John's"). Decision record 0017.
_AREA_HINT_SEPARATOR_WORDS: Final = frozenset({"COMMA", "SEMICOLON"})
_OTHER_PUNCTUATION: Final = "Po"


def owned_by(user_id: uuid.UUID) -> ColumnElement[bool]:
    """Keep only the items of one user."""
    return SavedItem.user_id == user_id


def fully_indexed() -> ColumnElement[bool]:
    """Keep only items the pipeline has finished."""
    return SavedItem.status == ItemStatus.DONE


def not_fully_indexed() -> ColumnElement[bool]:
    """Keep only items the pipeline has not finished: pending, enriching, waiting or failed."""
    return SavedItem.status != ItemStatus.DONE


def in_collection(collection: str) -> ColumnElement[bool]:
    """Keep the items in one of the user's collections."""
    return SavedItem.collection == collection


def of_type(type_name: str) -> ColumnElement[bool]:
    """Keep the items of one built-in content type."""
    return SavedItem.type == type_name


def in_category(category: str) -> ColumnElement[bool]:
    """Keep the items with one open category."""
    return SavedItem.category == category


def tagged(tag: str) -> ColumnElement[bool]:
    """Keep the items with one tag."""
    return SavedItem.tags.contains([tag])


def from_platform(platform: str) -> ColumnElement[bool]:
    """Keep the items from one platform."""
    return SavedItem.platform == platform


def saved_after(moment: datetime) -> ColumnElement[bool]:
    """Keep the items saved after a moment."""
    return SavedItem.saved_at > moment


@dataclass(frozen=True)
class FilterContext:
    """What a type filter's condition needs besides its value.

    Attributes:
        user_id: The asking user, for their own statuses.
        country: The asking user's country, upper case, for availability.
    """

    user_id: uuid.UUID
    country: str | None


@dataclass(frozen=True)
class FilterPredicates:
    """When one of a type's filters holds: on the item itself, or on one of its mentions.

    A list post has several mentions. Every filter of a search must hold on
    the **same** mention, so two titles cannot each satisfy half of the
    filters. A mention with a linked entity is judged by `on_entity`, one
    without by `on_mention`; so once a resolver links an entity, its verified
    data wins over what the post said. A value stored on the item itself
    (`saved_item.fields`) holds for all of its mentions.

    Attributes:
        on_item: The filter holds by the item's own fields; `None` if it never can.
        on_entity: The filter holds by one linked entity, written on `Entity`
            and, for its own mention, on `ItemEntity.position`; `None` if
            entities never decide it.
        on_mention: The filter holds by one mention with no linked entity,
            written on `unresolved_mention()`; `None` if such a mention never
            decides it, e.g. for streaming or distance, which need an entity.
    """

    on_item: ColumnElement[bool] | None
    on_entity: ColumnElement[bool] | None
    on_mention: ColumnElement[bool] | None = None


def filter_predicates(
    search_filter: SearchFilter,
    value: FilterValue,
    context: FilterContext,
    *,
    item_wide: bool,
) -> FilterPredicates:
    """Build where one of a type's filters holds.

    Args:
        search_filter: The filter as the type declares it.
        value: The normalized value.
        context: The asking user and their country.
        item_wide: Whether `saved_item.fields` holds the value for the whole
            item. Otherwise only the linked entity's data and its own mention
            in `fields.entities` decide; the list of every mention's values at
            the top of `fields` never does.

    Returns:
        The filter's predicates on the item and on one linked entity.
    """
    if search_filter.source is FilterSource.COUNTRY_AVAILABILITY:
        return FilterPredicates(None, _available_in_country(str(value), context.country))
    if search_filter.source is FilterSource.USER_STATUS:
        done = _done_by_user(context.user_id)
        return FilterPredicates(None, done if value else not_(done))
    if isinstance(value, Near):
        return near_predicates(value)
    if search_filter.kind is FilterKind.AREA:
        return area_predicates(str(value), search_filter.mention_field)
    if search_filter.kind is FilterKind.TEXT:
        return _text_predicates(search_filter.name, str(value), item_wide=item_wide)
    return _value_predicates(search_filter.name, value, item_wide=item_wide)


def area_predicates(area: str, mention_field: str | None) -> FilterPredicates:
    """Hold for an entity whose address has the area at any level, or a mention whose hint has it.

    Args:
        area: The normalized area.
        mention_field: The key of a mention that holds where the post said it
            is. A mention with no linked entity holds when one part of that
            hint is the area (`area_hint_parts`). `None` if mentions never decide.

    Returns:
        The predicates on one linked entity and on one mention without an entity.
    """
    on_entity = or_(*(level == area for level in ADDRESS_LEVELS))
    if mention_field is None:
        return FilterPredicates(None, on_entity)
    on_mention = literal(area, Text) == any_(area_hint_parts(mention_field))
    return FilterPredicates(None, on_entity, on_mention)


def city_predicates(cities: Sequence[tuple[str | None, str]]) -> FilterPredicates:
    """Hold for an entity in one of the given cities.

    Args:
        cities: `(country, city)` pairs. The country keeps two cities with the
            same name in different countries apart.

    Returns:
        The predicates; they never hold when `cities` is empty.
    """
    if not cities:
        return FilterPredicates(None, false())
    return FilterPredicates(
        None,
        or_(
            *(
                and_(Entity.city == city, Entity.country.is_not_distinct_from(country))
                for country, city in cities
            )
        ),
    )


def near_predicates(near: Near) -> FilterPredicates:
    """Hold for an entity within a distance of a point."""
    return FilterPredicates(None, func.ST_DWithin(Entity.location, _point(near), near.radius_m))


def all_filters_hold(predicates: Sequence[FilterPredicates]) -> ColumnElement[bool]:
    """Keep the items on which every filter holds, all on the same mention.

    An item matches when its own fields satisfy every filter, when one of its
    linked entities satisfies every filter its fields do not, or when one of
    its mentions with no linked entity does.

    Args:
        predicates: One per filter of the search.

    Returns:
        The condition on `saved_item`.
    """
    if not predicates:
        return true()
    alternatives = [_on_one_entity(predicates)]
    on_one_unresolved_mention = _on_one_unresolved_mention(predicates)
    if on_one_unresolved_mention is not None:
        alternatives.append(on_one_unresolved_mention)
    item_conditions = [predicate.on_item for predicate in predicates]
    if all(condition is not None for condition in item_conditions):
        alternatives.insert(
            0, and_(*(condition for condition in item_conditions if condition is not None))
        )
    return or_(*alternatives)


def _on_one_entity(predicates: Sequence[FilterPredicates]) -> ColumnElement[bool]:
    per_entity = and_(
        *(_either(predicate.on_item, predicate.on_entity) for predicate in predicates)
    )
    return (
        select(ItemEntity.saved_item_id)
        .join(Entity, Entity.id == ItemEntity.entity_id)
        .where(ItemEntity.saved_item_id == SavedItem.id, per_entity)
        .exists()
    )


def _on_one_unresolved_mention(
    predicates: Sequence[FilterPredicates],
) -> ColumnElement[bool] | None:
    """One mention with no linked entity meets every filter; `None` if no such mention can.

    No such mention can meet a filter that needs an entity, such as streaming
    or distance. And when no filter reads mentions, a mention could only match
    by the item's own fields, which the item-wide alternative already covers.
    """
    if any(predicate.on_item is None and predicate.on_mention is None for predicate in predicates):
        return None
    if all(predicate.on_mention is None for predicate in predicates):
        return None
    per_mention = and_(
        *(_either(predicate.on_item, predicate.on_mention) for predicate in predicates)
    )
    mention = unresolved_mention()
    return (
        select(mention.c.value)
        .where(mention_is_unresolved(), per_mention)
        .correlate(SavedItem)
        .exists()
    )


def _either(*conditions: ColumnElement[bool] | None) -> ColumnElement[bool]:
    parts = [condition for condition in conditions if condition is not None]
    return or_(*parts) if parts else false()


@cache
def unresolved_mention() -> TableValuedAlias:
    """Each mention of the item in `fields.entities`, with its 1-based `ordinality`.

    Used in a subquery on `saved_item`, together with `mention_is_unresolved()`.
    The rows are mentions of any kind; the caller keeps those without a linked
    entity. It is built once, so every condition refers to the same rows.

    Returns:
        A table-valued function with the columns `value` (the mention's JSON)
        and `ordinality`.
    """
    mentions = _mention_list()
    # A field that is not a list (or is missing) yields no mentions instead of an error.
    as_list = case((func.jsonb_typeof(mentions) == "array", mentions))
    return (
        func.jsonb_array_elements(as_list)
        .table_valued(column("value", JSONB), with_ordinality="ordinality")
        .render_derived(name="mention")
    )


def mention_is_unresolved() -> ColumnElement[bool]:
    """No entity is linked to the current mention of `unresolved_mention()`.

    The resolver links the entity of the mention at index `i` of
    `fields.entities` with `item_entity.position = i`.
    """
    mention = unresolved_mention()
    linked = (
        select(ItemEntity.saved_item_id)
        .where(
            ItemEntity.saved_item_id == SavedItem.id,
            ItemEntity.position == mention.c.ordinality - 1,
        )
        # `saved_item` is the outer query's row, two levels up, so only `item_entity` is
        # this subquery's own table.
        .correlate_except(ItemEntity)
        .exists()
    )
    return not_(linked)


def area_hint_parts(mention_field: str) -> ColumnElement[list[str]]:
    """The parts of the current mention's area hint, split on its separators.

    "Kadıköy, İstanbul" → ["Kadıköy", "İstanbul"]; "渋谷、東京" → ["渋谷", "東京"].
    A hint with no separator is one part. Parts are trimmed and compared as
    they are stored, so normalization offers exactly these values.

    Args:
        mention_field: The key of the mention that holds the hint.

    Returns:
        A text array; null when the mention has no hint.
    """
    hint = unresolved_mention().c.value.op("->>", return_type=Text)(mention_field)
    separators = literal(area_hint_separator_pattern(), Text)
    return func.regexp_split_to_array(func.btrim(hint), separators, type_=ARRAY(Text))


@cache
def area_hint_separator_pattern() -> str:
    """The Postgres regular expression that splits an area hint into its levels.

    It matches one comma or semicolon of any script, with the white space
    around it. The marks come from the Unicode names, not from a list.
    """
    separators = "".join(f"\\U{ord(mark):08x}" for mark in _area_hint_separators())
    return rf"\s*[{separators}]\s*"


def _area_hint_separators() -> list[str]:
    return [
        mark
        for mark in map(chr, range(sys.maxunicode + 1))
        if unicodedata.category(mark) == _OTHER_PUNCTUATION
        and not _AREA_HINT_SEPARATOR_WORDS.isdisjoint(unicodedata.name(mark, "").split())
    ]


def distance_from(near: Near) -> ColumnElement[float]:
    """The distance in metres from a point to the item's nearest entity; null if none has one."""
    return (
        select(func.min(func.ST_Distance(Entity.location, _point(near))))
        .join(ItemEntity, ItemEntity.entity_id == Entity.id)
        .where(ItemEntity.saved_item_id == SavedItem.id)
        .scalar_subquery()
    )


def vector_distance(embedding: Sequence[float]) -> ColumnElement[float]:
    """The cosine distance from a query embedding to the item's embedding."""
    query_vector = bindparam("query_embedding", list(embedding), type_=Vector())
    return SavedItem.embedding.op("<=>", return_type=Float)(query_vector)


def keyword_query(words: Sequence[str]) -> ColumnElement[str]:
    """A full-text query that matches any of the words, in any language.

    Args:
        words: The words. At least one.

    Returns:
        A `tsquery` that matches text with any of the words.
    """
    config: ColumnElement[str] = literal_column(f"'{FULL_TEXT_SEARCH_CONFIG}'::regconfig")
    queries: list[ColumnElement[str]] = [
        func.plainto_tsquery(config, word, type_=TSQUERY) for word in words
    ]
    combined = queries[0]
    for query in queries[1:]:
        combined = combined.op("||", return_type=TSQUERY)(query)
    return combined


def matches_keywords(query: ColumnElement[str]) -> ColumnElement[bool]:
    """Keep the items whose caption, summary or transcript has a word of the query."""
    return saved_item_search_vector().op("@@", return_type=Boolean)(query)


def keyword_rank(query: ColumnElement[str]) -> ColumnElement[float]:
    """How well the item's caption, summary and transcript match the query."""
    return func.ts_rank(saved_item_search_vector(), query, type_=Float)


def _value_predicates(key: str, value: FilterValue, *, item_wide: bool) -> FilterPredicates:
    """The value under `key`, alone or in a list: item-wide, or on one mention.

    A mention with a linked entity holds by the entity's data or by its own
    mention, one without by the mention alone.
    """
    on_entity = or_(_holds_value(Entity.data, key, value), _holds_value(_own_mention(), key, value))
    on_mention = _holds_value(unresolved_mention().c.value, key, value)
    on_item = _holds_value(SavedItem.fields, key, value) if item_wide else None
    return FilterPredicates(on_item, on_entity, on_mention)


def _text_predicates(key: str, value: str, *, item_wide: bool) -> FilterPredicates:
    pattern = f"%{_escape_like(value)}%"
    on_entity = or_(
        Entity.data[key].astext.ilike(pattern, escape=_LIKE_ESCAPE),
        _own_mention().op("->>", return_type=Text)(key).ilike(pattern, escape=_LIKE_ESCAPE),
    )
    on_mention = (
        unresolved_mention()
        .c.value.op("->>", return_type=Text)(key)
        .ilike(pattern, escape=_LIKE_ESCAPE)
    )
    on_item = (
        SavedItem.fields[key].astext.ilike(pattern, escape=_LIKE_ESCAPE) if item_wide else None
    )
    return FilterPredicates(on_item, on_entity, on_mention)


def _holds_value(
    # A JSON column or expression; its Python type is the whole JSON document.
    document: ColumnElement[Any] | QueryableAttribute[Any],
    key: str,
    value: FilterValue,
) -> ColumnElement[bool]:
    return or_(document.contains({key: value}), document.contains({key: [value]}))


def _own_mention() -> ColumnElement[Any]:
    """The mention in `saved_item.fields` that the linked entity was resolved from.

    The resolver links the entity of the mention at index `i` of
    `fields.entities` with `item_entity.position = i`.
    """
    return _mention_list().op("->", return_type=JSONB)(ItemEntity.position)


def _mention_list() -> ColumnElement[Any]:
    """The item's mentions, `fields.entities`."""
    return SavedItem.fields.op("->", return_type=JSONB)(literal(MENTIONS_KEY))


def _available_in_country(provider: str, country: str | None) -> ColumnElement[bool]:
    """The entity is watchable on the provider in the country without paying extra.

    Only the country's own row counts, and only its `WATCHABLE_OFFER_KINDS`.
    """
    watchable = or_(
        *(
            EntityAvailability.providers.contains({offer_kind: [provider]})
            for offer_kind in WATCHABLE_OFFER_KINDS
        )
    )
    return (
        select(EntityAvailability.entity_id)
        .where(
            EntityAvailability.entity_id == Entity.id,
            EntityAvailability.country == country,
            watchable,
        )
        .exists()
    )


def _done_by_user(user_id: uuid.UUID) -> ColumnElement[bool]:
    """The user marked the entity as done (visited, watched, read)."""
    return (
        select(UserEntity.entity_id)
        .where(
            UserEntity.entity_id == Entity.id,
            UserEntity.user_id == user_id,
            UserEntity.status == EntityStatus.DONE,
        )
        .exists()
    )


def _point(near: Near) -> ColumnElement[object]:
    return cast(bindparam(None, near.point, type_=GeographyPoint()), GeographyPoint())


def _escape_like(value: str) -> str:
    for special in (_LIKE_ESCAPE, "%", "_"):
        value = value.replace(special, _LIKE_ESCAPE + special)
    return value
