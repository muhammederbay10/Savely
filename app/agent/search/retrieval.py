"""Hybrid retrieval over one user's saves: filter first, then rank (ARCHITECTURE §3.5 and §3.6).

1. The query is checked against the content types (`search_query`).
2. Its wordings are normalized against the user's own values
   (`filter_normalization`), while the query text is embedded.
3. **L0:** SQL/PostGIS filters keep the matching items, and pgvector ranks them
   by meaning. Without a query text they are ranked by distance from a "near"
   point, or newest first.
4. **L1:** if the embedding or the vector search fails, the same filters are
   combined with full-text search, ranked by keyword.
5. **L2:** if fewer than `SEARCH_MIN_RESULTS` items were found, the filters are
   relaxed and items that are not fully indexed yet are searched by keyword.

Everything runs inside `SEARCH_TIME_BUDGET_SECONDS`. When the budget runs
out, search answers with what it found and marks the answer incomplete.

    retriever = HybridRetriever(classifier=..., embedder=..., prompts=..., settings=...)
    result = await retriever.search(session, SearchQuery(user_id=user.id, type="place",
                                                         filters={"area": "beşilktaş"}))
"""

import asyncio
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from sqlalchemy import ColumnElement, and_, or_, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.search.conditions import (
    WIDENED_LEVELS,
    FilterContext,
    FilterPredicates,
    all_filters_hold,
    area_predicates,
    city_predicates,
    distance_from,
    filter_predicates,
    from_platform,
    fully_indexed,
    in_category,
    in_collection,
    keyword_query,
    keyword_rank,
    matches_keywords,
    near_predicates,
    not_fully_indexed,
    of_type,
    owned_by,
    saved_after,
    tagged,
    vector_distance,
)
from app.agent.search.fallback import (
    FallbackLevel,
    FallbackReason,
    FallbackStep,
    log_fallback,
)
from app.agent.search.normalization import FilterNormalizer, NormalizedWording
from app.agent.search.prompts import RetrievalPrompts
from app.agent.search.query import (
    CATEGORY_SLOT,
    COLLECTION_SLOT,
    FILTER_SLOT_PREFIX,
    TAG_SLOT_PREFIX,
    Near,
    SearchPlan,
    SearchQuery,
    TypedFilter,
    plan_search,
)
from app.agent.search.saved_values import ValueScope
from app.config.retrieval import RetrievalSettings
from app.db.models import Entity, EntityAvailability, ItemEntity, ItemStatus, SavedItem
from app.models.base import Classifier, Embedder
from app.models.errors import ModelError
from app.types.base import FilterKind, FilterSource
from app.types.registry import CONTENT_TYPES, ContentTypeRegistry


class AvailabilityGapReason(StrEnum):
    """Why an entity's streaming availability must be fetched for a country."""

    MISSING = "missing"
    STALE = "stale"


@dataclass(frozen=True)
class AvailabilityGap:
    """Streaming availability that is missing or stale for the asking user's country.

    The resolver (SAV-15) acts on it: it fetches or refreshes exactly this
    `(entity_id, country)` row. Another country's row never stands in for it.

    Attributes:
        entity_id: The movie or series.
        country: The asking user's country.
        reason: Whether the row is missing or older than `AVAILABILITY_MAX_AGE_DAYS`.
    """

    entity_id: uuid.UUID
    country: str
    reason: AvailabilityGapReason


@dataclass(frozen=True)
class SearchHit:
    """One saved item search found.

    Attributes:
        item_id: The saved item.
        level: The fallback level that found it.
        fully_indexed: Whether indexing has finished. Items that have not are
            only found at L2, by the words of their caption.
    """

    item_id: uuid.UUID
    level: FallbackLevel
    fully_indexed: bool


@dataclass(frozen=True)
class SearchResult:
    """What one search found, and how.

    Attributes:
        hits: The items, best first.
        level: The highest fallback level search reached.
        fallbacks: Every step off the normal path, in order.
        relaxed: The query slots L2 dropped or widened, e.g. `filters.area`.
            The reply tells the user these results match less exactly.
        incomplete: The time budget ran out, so items may be missing.
        normalized: How each wording of the query was normalized.
        availability_gaps: Streaming availability to fetch or refresh for the
            asking user's country.
    """

    hits: tuple[SearchHit, ...]
    level: FallbackLevel
    fallbacks: tuple[FallbackStep, ...]
    relaxed: tuple[str, ...]
    incomplete: bool
    normalized: tuple[NormalizedWording, ...]
    availability_gaps: tuple[AvailabilityGap, ...]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class _Run:
    """What one search has found so far. It survives the time budget running out."""

    user_id: uuid.UUID
    hits: list[SearchHit] = field(default_factory=list)
    level: FallbackLevel = FallbackLevel.NORMAL
    fallbacks: list[FallbackStep] = field(default_factory=list)
    relaxed: tuple[str, ...] = ()
    incomplete: bool = False
    normalized: dict[str, NormalizedWording] = field(default_factory=dict)
    availability_gaps: tuple[AvailabilityGap, ...] = ()

    def fall_back(self, level: FallbackLevel, reason: FallbackReason, **details: object) -> None:
        step = FallbackStep(level=level, reason=reason)
        self.fallbacks.append(step)
        self.level = max(self.level, level)
        log_fallback(self.user_id, step, **details)

    def result(self) -> SearchResult:
        return SearchResult(
            hits=tuple(self.hits),
            level=self.level,
            fallbacks=tuple(self.fallbacks),
            relaxed=self.relaxed,
            incomplete=self.incomplete,
            normalized=tuple(self.normalized.values()),
            availability_gaps=self.availability_gaps,
        )


@dataclass(frozen=True)
class _Ranking:
    """The order results come in, best first.

    Attributes:
        order: The ORDER BY expressions.
        required: A condition the ranking needs, e.g. a keyword match at L1.
        uses_vector: Whether the ranking is a vector search.
    """

    # Orders compare floats, dates and ids, so one element type cannot name them all.
    order: tuple[ColumnElement[Any], ...]
    required: ColumnElement[bool] | None = None
    uses_vector: bool = False


@dataclass(frozen=True)
class _Conditions:
    """The SQL conditions of one normalized query.

    Attributes:
        scope: The user, collection, type, category, platform and date.
        tags: One condition per tag.
        filters: Where each type filter holds, by slot. They must all hold on
            the same linked entity.
        availability_slots: The slots of filters on streaming availability.
    """

    scope: tuple[ColumnElement[bool], ...]
    tags: dict[str, ColumnElement[bool]]
    filters: dict[str, FilterPredicates]
    availability_slots: frozenset[str]

    def strict(self) -> list[ColumnElement[bool]]:
        return [*self.scope, *self.tags.values(), all_filters_hold(list(self.filters.values()))]

    def without_availability(self) -> list[ColumnElement[bool]]:
        kept = [
            predicates
            for slot, predicates in self.filters.items()
            if slot not in self.availability_slots
        ]
        return [*self.scope, *self.tags.values(), all_filters_hold(kept)]


class HybridRetriever:
    """Searches one user's saves: normalized filters first, then ranking, with fallbacks."""

    def __init__(
        self,
        *,
        classifier: Classifier,
        embedder: Embedder,
        prompts: RetrievalPrompts,
        settings: RetrievalSettings,
        registry: ContentTypeRegistry = CONTENT_TYPES,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Create the retriever.

        Args:
            classifier: The decision model, for filter normalization.
            embedder: The embedding model, for the query text and normalization.
            prompts: The wording sent to the decision model.
            settings: Limits, thresholds and the time budget.
            registry: The built-in content types.
            now: The clock, for the age of streaming availability.
        """
        self._embedder = embedder
        self._settings = settings
        self._registry = registry
        self._now = now
        self._normalizer = FilterNormalizer(
            classifier=classifier, embedder=embedder, prompts=prompts, settings=settings
        )

    async def search(self, session: AsyncSession, query: SearchQuery) -> SearchResult:
        """Search one user's saves.

        Args:
            session: The session to read in. A search only reads. If the time
                budget runs out during a query, the session's transaction
                should be rolled back before it is used again.
            query: What to search for.

        Returns:
            The items found, best first, and how they were found.

        Raises:
            SearchQueryError: If the query asks for something that cannot be
                searched for. Its subclasses `CountryRequiredError` and
                `UnsupportedFilterError` name the cases the agent handles.
        """
        plan = plan_search(query, self._registry)
        run = _Run(user_id=query.user_id)
        budget = asyncio.timeout(self._settings.search_time_budget_seconds)
        try:
            async with budget:
                await self._climb(session, plan, run)
        except TimeoutError:
            if not budget.expired():
                raise
            run.incomplete = True
            run.fall_back(run.level, FallbackReason.TIME_BUDGET_EXCEEDED, found=len(run.hits))
        return run.result()

    async def _climb(self, session: AsyncSession, plan: SearchPlan, run: _Run) -> None:
        country = _country_of(plan.query)
        scope = ValueScope(user_id=plan.query.user_id, type=plan.query.type, country=country)
        normalized, embedding = await asyncio.gather(
            self._normalizer.normalize(session, scope, plan.wordings),
            self._embed_text(plan.query.text, run),
        )
        run.normalized = normalized
        conditions = _conditions(plan, normalized, country)
        limit = min(
            plan.query.limit or self._settings.search_max_results, self._settings.search_max_results
        )
        ranking = _ranking(plan, embedding)
        found = await self._find_strict(session, conditions, ranking, limit, run)
        if ranking.uses_vector and found is None:
            ranking = _ranking(plan, None)
            found = await self._find(session, conditions.strict(), ranking, limit)
        run.hits = [_hit(row, run.level) for row in found or []]
        if len(run.hits) < self._settings.search_min_results:
            run.fall_back(FallbackLevel.RELAXED, FallbackReason.FEW_RESULTS, found=len(run.hits))
            await self._relax(session, plan, normalized, ranking, limit, run)
        run.availability_gaps = await self._availability_gaps(
            session, plan, conditions, country, run
        )

    async def _embed_text(self, text: str | None, run: _Run) -> tuple[float, ...] | None:
        if text is None or not text.strip():
            return None
        try:
            result = await self._embedder.embed([text])
        except ModelError as error:
            run.fall_back(
                FallbackLevel.DEGRADED,
                FallbackReason.EMBEDDING_FAILED,
                error=type(error).__name__,
                model=error.model,
            )
            return None
        return result.vectors[0]

    async def _find_strict(
        self,
        session: AsyncSession,
        conditions: _Conditions,
        ranking: _Ranking,
        limit: int,
        run: _Run,
    ) -> list[tuple[uuid.UUID, str]] | None:
        """Run L0, or L1 when the embedding already failed; `None` if the vector search fails."""
        if not ranking.uses_vector:
            return await self._find(session, conditions.strict(), ranking, limit)
        try:
            async with session.begin_nested():
                return await self._find(session, conditions.strict(), ranking, limit)
        except DBAPIError as error:
            run.fall_back(
                FallbackLevel.DEGRADED,
                FallbackReason.VECTOR_SEARCH_FAILED,
                error=type(error.orig).__name__,
            )
            return None

    async def _find(
        self,
        session: AsyncSession,
        conditions: Sequence[ColumnElement[bool]],
        ranking: _Ranking,
        limit: int,
        excluded: Sequence[uuid.UUID] = (),
    ) -> list[tuple[uuid.UUID, str]]:
        statement = select(SavedItem.id, SavedItem.status).where(*conditions)
        if ranking.required is not None:
            statement = statement.where(ranking.required)
        if excluded:
            statement = statement.where(SavedItem.id.not_in(list(excluded)))
        statement = statement.order_by(*ranking.order).limit(limit)
        rows = await session.execute(statement)
        return [(row.id, row.status) for row in rows]

    async def _relax(
        self,
        session: AsyncSession,
        plan: SearchPlan,
        normalized: dict[str, NormalizedWording],
        ranking: _Ranking,
        limit: int,
        run: _Run,
    ) -> None:
        relaxed_conditions, relaxed_slots = await _relaxed_conditions(session, plan, normalized)
        run.relaxed = relaxed_slots
        room = limit - len(run.hits)
        found = [hit.item_id for hit in run.hits]
        if room > 0:
            # A keyword match that stands in for vector ranking (L1) is kept: relaxing
            # filters must not turn a text search into "everything".
            rows = await self._find(session, relaxed_conditions, ranking, room, found)
            run.hits.extend(_hit(row, FallbackLevel.RELAXED) for row in rows)
            found.extend(item_id for item_id, _ in rows)
        room = limit - len(run.hits)
        if room > 0:
            conditions, catch_up = _catch_up(plan, normalized)
            rows = await self._find(session, conditions, catch_up, room, found)
            run.hits.extend(_hit(row, FallbackLevel.RELAXED) for row in rows)

    async def _availability_gaps(
        self,
        session: AsyncSession,
        plan: SearchPlan,
        conditions: _Conditions,
        country: str | None,
        run: _Run,
    ) -> tuple[AvailabilityGap, ...]:
        entity_types = [
            content_type.name
            for content_type in self._registry.all()
            if any(
                search_filter.source is FilterSource.COUNTRY_AVAILABILITY
                for search_filter in content_type.search_filters
            )
        ]
        if country is None or not entity_types:
            return ()
        returned = SavedItem.id.in_([hit.item_id for hit in run.hits])
        # With a streaming filter, the titles it rejected for lack of a row count
        # too, and so do the titles L2 returned after relaxing it.
        items = (
            [or_(and_(*conditions.without_availability()), returned)]
            if plan.uses_country
            else [returned]
        )
        cutoff = self._now() - timedelta(days=self._settings.availability_max_age_days)
        rows = await session.execute(
            select(Entity.id, EntityAvailability.refreshed_at)
            .select_from(SavedItem)
            .join(ItemEntity, ItemEntity.saved_item_id == SavedItem.id)
            .join(Entity, Entity.id == ItemEntity.entity_id)
            .outerjoin(
                EntityAvailability,
                and_(
                    EntityAvailability.entity_id == Entity.id,
                    EntityAvailability.country == country,
                ),
            )
            .where(
                *items,
                Entity.type.in_(entity_types),
                or_(
                    EntityAvailability.entity_id.is_(None),
                    EntityAvailability.refreshed_at < cutoff,
                ),
            )
            .distinct()
            .order_by(Entity.id)
        )
        return tuple(
            AvailabilityGap(
                entity_id=row.id,
                country=country,
                reason=AvailabilityGapReason.MISSING
                if row.refreshed_at is None
                else AvailabilityGapReason.STALE,
            )
            for row in rows
        )


def _country_of(query: SearchQuery) -> str | None:
    if query.country is None or not query.country.strip():
        return None
    return query.country.strip().upper()


def _hit(row: tuple[uuid.UUID, str], level: FallbackLevel) -> SearchHit:
    item_id, status = row
    return SearchHit(item_id=item_id, level=level, fully_indexed=status == ItemStatus.DONE)


def _value_of(slot: str, normalized: dict[str, NormalizedWording], wording: str) -> str:
    found = normalized.get(slot)
    return found.value if found is not None else wording


def _scope_conditions(
    plan: SearchPlan, normalized: dict[str, NormalizedWording], *, indexed: bool
) -> list[ColumnElement[bool]]:
    """List the conditions of the scope: user, status, collection, platform, date, type, category.

    Items that are not fully indexed have no type or category yet, so those
    two only apply to fully indexed items.
    """
    query = plan.query
    conditions = [owned_by(query.user_id)]
    conditions.append(fully_indexed() if indexed else not_fully_indexed())
    if query.collection is not None:
        conditions.append(in_collection(_value_of(COLLECTION_SLOT, normalized, query.collection)))
    if query.platform is not None:
        conditions.append(from_platform(query.platform))
    if query.saved_after is not None:
        conditions.append(saved_after(query.saved_after))
    if indexed and query.type is not None:
        conditions.append(of_type(query.type))
    if indexed and query.category is not None:
        conditions.append(in_category(_value_of(CATEGORY_SLOT, normalized, query.category)))
    return conditions


def _normalized_filter(typed: TypedFilter, normalized: dict[str, NormalizedWording]) -> TypedFilter:
    slot = f"{FILTER_SLOT_PREFIX}{typed.search_filter.name}"
    found = normalized.get(slot)
    return typed if found is None else replace(typed, value=found.value)


def _conditions(
    plan: SearchPlan, normalized: dict[str, NormalizedWording], country: str | None
) -> _Conditions:
    context = FilterContext(user_id=plan.query.user_id, country=country)
    tags = {
        f"{TAG_SLOT_PREFIX}{index}": tagged(_value_of(f"{TAG_SLOT_PREFIX}{index}", normalized, tag))
        for index, tag in enumerate(plan.query.tags)
    }
    filters: dict[str, FilterPredicates] = {}
    availability: set[str] = set()
    for typed in plan.filters:
        slot = f"{FILTER_SLOT_PREFIX}{typed.search_filter.name}"
        value = _normalized_filter(typed, normalized).value
        filters[slot] = filter_predicates(
            typed.search_filter, value, context, item_wide=typed.item_wide
        )
        if typed.search_filter.source is FilterSource.COUNTRY_AVAILABILITY:
            availability.add(slot)
    return _Conditions(
        scope=tuple(_scope_conditions(plan, normalized, indexed=True)),
        tags=tags,
        filters=filters,
        availability_slots=frozenset(availability),
    )


def _near_of(plan: SearchPlan) -> Near | None:
    for typed in plan.filters:
        if isinstance(typed.value, Near):
            return typed.value
    return None


def _ranking(plan: SearchPlan, embedding: tuple[float, ...] | None) -> _Ranking:
    newest_first = (SavedItem.saved_at.desc(), SavedItem.id.asc())
    text = plan.query.text
    if text is not None and text.strip():
        if embedding is not None:
            distance = vector_distance(embedding)
            return _Ranking(order=(distance.asc().nulls_last(), *newest_first), uses_vector=True)
        keywords = keyword_query(_words(text))
        return _Ranking(
            order=(keyword_rank(keywords).desc(), *newest_first),
            required=matches_keywords(keywords),
        )
    near = _near_of(plan)
    if near is not None:
        return _Ranking(order=(distance_from(near).asc().nulls_last(), *newest_first))
    return _Ranking(order=newest_first)


async def _relaxed_conditions(
    session: AsyncSession, plan: SearchPlan, normalized: dict[str, NormalizedWording]
) -> tuple[list[ColumnElement[bool]], tuple[str, ...]]:
    """Build L2's conditions on fully indexed items: widen the area, drop secondary filters.

    Kept: the user, collection, type, category, platform, date and a "near"
    point. Widened: an area that is a district or neighbourhood of a linked
    place becomes its city for linked places; mentions without an entity keep
    matching the area as written. Dropped: tags and every other type filter.
    """
    conditions = _scope_conditions(plan, normalized, indexed=True)
    relaxed = [f"{TAG_SLOT_PREFIX}{index}" for index in range(len(plan.query.tags))]
    kept: list[FilterPredicates] = []
    for typed in plan.filters:
        slot = f"{FILTER_SLOT_PREFIX}{typed.search_filter.name}"
        value = _normalized_filter(typed, normalized).value
        if isinstance(value, Near):
            kept.append(near_predicates(value))
        elif typed.search_filter.kind is FilterKind.AREA:
            area = area_predicates(str(value), typed.search_filter.mention_field)
            cities = await _cities_around(session, plan.query.user_id, str(value))
            if cities:
                # Only linked places are widened to their city; a hint keeps the area as
                # written, since a hint doesn't say which of its parts is the city.
                widened = city_predicates(cities)
                kept.append(replace(widened, on_mention=area.on_mention))
                relaxed.append(slot)
            else:
                kept.append(area)
        else:
            relaxed.append(slot)
    conditions.append(all_filters_hold(kept))
    return conditions, tuple(relaxed)


async def _cities_around(
    session: AsyncSession, user_id: uuid.UUID, area: str
) -> list[tuple[str | None, str]]:
    """Find the cities of the user's places whose district or neighbourhood is the area.

    Returns:
        `(country, city)` pairs; empty when the area is not a district or
        neighbourhood of any of the user's places, e.g. when it is a city already.
    """
    rows = await session.execute(
        select(Entity.country, Entity.city)
        .select_from(SavedItem)
        .join(ItemEntity, ItemEntity.saved_item_id == SavedItem.id)
        .join(Entity, Entity.id == ItemEntity.entity_id)
        .where(
            owned_by(user_id),
            Entity.city.is_not(None),
            or_(*(level == area for level in WIDENED_LEVELS)),
        )
        .distinct()
    )
    return [(country, city) for country, city in rows if city is not None]


def _catch_up(
    plan: SearchPlan, normalized: dict[str, NormalizedWording]
) -> tuple[list[ColumnElement[bool]], _Ranking]:
    """Build L2's search of items that are not fully indexed, by the words of the query.

    Such items have no category, fields or entities yet, only their caption
    and whatever indexing stored before it stopped. They are matched by any
    word of the query text, of what the user wrote, or of its normalized value.
    """
    conditions = _scope_conditions(plan, normalized, indexed=False)
    newest_first = (SavedItem.saved_at.desc(), SavedItem.id.asc())
    words = _catch_up_words(plan, normalized)
    if not words:
        return conditions, _Ranking(order=newest_first)
    keywords = keyword_query(words)
    ranking = _Ranking(
        order=(keyword_rank(keywords).desc(), *newest_first), required=matches_keywords(keywords)
    )
    return conditions, ranking


def _catch_up_words(plan: SearchPlan, normalized: dict[str, NormalizedWording]) -> list[str]:
    texts = [plan.query.text or ""]
    for wording in plan.wordings:
        if wording.slot == COLLECTION_SLOT:
            continue
        texts.append(wording.wording)
        found = normalized.get(wording.slot)
        if found is not None:
            texts.append(found.value)
    texts.extend(
        typed.value
        for typed in plan.filters
        if isinstance(typed.value, str) and typed.search_filter.kind is FilterKind.TEXT
    )
    return _words(" ".join(texts))


def _words(text: str) -> list[str]:
    """Split text into distinct words on white space, which works for any language with spaces."""
    seen: dict[str, str] = {}
    for word in text.split():
        seen.setdefault(word.casefold(), word)
    return list(seen.values())
