"""The values a search filter can take: those in one user's own saves, and the vocabulary's.

Filter normalization (ARCHITECTURE §3.4 and §3.5) maps what the user wrote to a
value that exists in **their** saves. These queries list those values. Every
query is limited to one user's items; the shared tables (`entity`,
`entity_availability`, `vocabulary`) are only read through that user's items,
or hold only gated, generic wording.

An area's values are the address levels of the user's linked entities and the
parts of the area hints of their mentions with no linked entity: exactly the
values the area filter matches (decision record 0017).

Nothing here lists real-world values in code: every value comes from the data.
"""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

from sqlalchemy import Executable, Float, Select, bindparam, func, select, text, true, union_all
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.search.conditions import (
    ADDRESS_LEVELS,
    WATCHABLE_OFFER_KINDS,
    area_hint_parts,
    mention_is_unresolved,
    owned_by,
    unresolved_mention,
)
from app.agent.search.query import CATEGORY_SLOT, COLLECTION_SLOT, Wording
from app.db.models import Entity, ItemEntity, SavedItem, Vocabulary
from app.db.types import Vector
from app.types.base import FilterKind, FilterSource

# A JSON value, or each element of a JSON array, that is text. Lax mode treats a
# single value as an array of one, so one path covers both shapes.
_TEXT_ELEMENTS_PATH: Final = 'lax $[*] ? (@.type() == "string")'

_USER_COLLECTIONS: Final = text(
    """
    SELECT DISTINCT collection AS value FROM saved_item
    WHERE user_id = :user_id AND collection IS NOT NULL
    """
)

_USER_CATEGORIES: Final = text(
    """
    SELECT DISTINCT category AS value FROM saved_item
    WHERE user_id = :user_id AND category IS NOT NULL
    """
)

_USER_TAGS: Final = text(
    """
    SELECT DISTINCT tag.value FROM saved_item
    CROSS JOIN LATERAL unnest(saved_item.tags) AS tag(value)
    WHERE saved_item.user_id = :user_id
    """
)

_USER_FIELD_VALUES: Final = text(
    f"""
    SELECT DISTINCT found.value FROM (
        SELECT jsonb_path_query(saved_item.fields -> :key, '{_TEXT_ELEMENTS_PATH}') #>> '{{}}'
            AS value
        FROM saved_item
        WHERE saved_item.user_id = :user_id AND saved_item.type = :type
        UNION ALL
        SELECT jsonb_path_query(entity.data -> :key, '{_TEXT_ELEMENTS_PATH}') #>> '{{}}'
        FROM saved_item
        JOIN item_entity ON item_entity.saved_item_id = saved_item.id
        JOIN entity ON entity.id = item_entity.entity_id
        WHERE saved_item.user_id = :user_id AND saved_item.type = :type
    ) AS found
    WHERE found.value IS NOT NULL
    """
)

_USER_PROVIDERS_IN_COUNTRY: Final = text(
    f"""
    SELECT DISTINCT
        jsonb_path_query(entity_availability.providers -> offer.kind, '{_TEXT_ELEMENTS_PATH}')
            #>> '{{}}' AS value
    FROM saved_item
    JOIN item_entity ON item_entity.saved_item_id = saved_item.id
    JOIN entity_availability
        ON entity_availability.entity_id = item_entity.entity_id
        AND entity_availability.country = :country
    CROSS JOIN unnest(CAST(:offer_kinds AS text[])) AS offer(kind)
    WHERE saved_item.user_id = :user_id AND saved_item.type = :type
    """
)


@dataclass(frozen=True)
class ValueScope:
    """Whose saves, and which part of them, a wording's values come from.

    Attributes:
        user_id: The asking user.
        type: The content type the query is limited to, if any.
        country: The asking user's country, upper case, if known.
    """

    user_id: uuid.UUID
    type: str | None
    country: str | None


async def saved_values(session: AsyncSession, scope: ValueScope, wording: Wording) -> list[str]:
    """List the values in the user's own saves that a wording could stand for.

    Args:
        session: The session to read in.
        scope: Whose saves to read.
        wording: The wording to find values for.

    Returns:
        The distinct values, as they are stored.
    """
    statement, parameters = _values_statement(scope, wording)
    rows = await session.execute(statement, parameters)
    return [value for value in rows.scalars() if value]


def _values_statement(scope: ValueScope, wording: Wording) -> tuple[Executable, dict[str, object]]:
    parameters: dict[str, object] = {"user_id": scope.user_id}
    if wording.slot == COLLECTION_SLOT:
        return _USER_COLLECTIONS, parameters
    if wording.slot == CATEGORY_SLOT:
        return _USER_CATEGORIES, parameters
    search_filter = wording.search_filter
    if search_filter is None:
        return _USER_TAGS, parameters
    parameters["type"] = scope.type
    if search_filter.source is FilterSource.COUNTRY_AVAILABILITY:
        return _USER_PROVIDERS_IN_COUNTRY, {
            **parameters,
            "country": scope.country,
            "offer_kinds": list(WATCHABLE_OFFER_KINDS),
        }
    if search_filter.kind is FilterKind.AREA:
        return _user_areas(scope, search_filter.mention_field), {}
    return _USER_FIELD_VALUES, {**parameters, "key": search_filter.name}


def _user_areas(scope: ValueScope, mention_field: str | None) -> Select[tuple[str]]:
    """List the areas of the user's items of the scope's type.

    Args:
        scope: Whose saves to read, and which type.
        mention_field: The key of a mention that holds its area hint; `None`
            if the type's mentions have none.

    Returns:
        The distinct, non-empty areas.
    """
    in_scope = (owned_by(scope.user_id), SavedItem.type == scope.type)
    levels = func.unnest(array(ADDRESS_LEVELS)).table_valued("value").render_derived(name="level")
    sources = [
        select(levels.c.value)
        .select_from(SavedItem)
        .join(ItemEntity, ItemEntity.saved_item_id == SavedItem.id)
        .join(Entity, Entity.id == ItemEntity.entity_id)
        .join(levels, true())
        .where(*in_scope)
    ]
    if mention_field is not None:
        mention = unresolved_mention()
        parts = (
            func.unnest(area_hint_parts(mention_field))
            .table_valued("value")
            .render_derived(name="part")
        )
        sources.append(
            select(parts.c.value)
            .select_from(SavedItem)
            .join(mention, true())
            .join(parts, true())
            .where(*in_scope, mention_is_unresolved())
        )
    found = union_all(*sources).subquery("found")
    return select(found.c.value).where(found.c.value.is_not(None), found.c.value != "").distinct()


async def vocabulary_values(session: AsyncSession, ids: Collection[uuid.UUID]) -> set[str]:
    """Read the canonical values with the given ids.

    Args:
        session: The session to read in.
        ids: Vocabulary ids, e.g. the values wording was rejected as an alias of.

    Returns:
        The canonical values.
    """
    if not ids:
        return set()
    rows = await session.execute(select(Vocabulary.value).where(Vocabulary.id.in_(list(ids))))
    return set(rows.scalars())


async def nearest_vocabulary_values(
    session: AsyncSession,
    field: str,
    embedding: Sequence[float],
    *,
    among: Collection[str],
    limit: int,
) -> list[str]:
    """List the vocabulary values nearest in meaning to some wording, among the given ones.

    A near value is only a candidate for the decision model, never a match.
    Unlike the pipeline's `nearest_values`, which searches the whole field,
    only the values in `among` (those in the asking user's saves) can come
    back, so search never offers a value the user doesn't have.

    Args:
        session: The session to read in.
        field: The vocabulary field.
        embedding: The wording's embedding.
        among: The only values that may be returned, e.g. those in the user's saves.
        limit: How many values at most.

    Returns:
        The nearest values first, by cosine distance.
    """
    if not among:
        return []
    distance = Vocabulary.embedding.op("<=>", return_type=Float)(
        bindparam("wording_embedding", list(embedding), type_=Vector())
    )
    rows = await session.execute(
        select(Vocabulary.value)
        .where(
            Vocabulary.field == field,
            Vocabulary.embedding.is_not(None),
            Vocabulary.value.in_(list(among)),
        )
        .order_by(distance)
        .limit(limit)
    )
    return list(rows.scalars())
