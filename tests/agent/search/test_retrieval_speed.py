"""A typical search answers in under 200 ms locally, model calls excluded (SAV-16).

The decision model and the embedder are fakes that answer at once, so the time
measured is normalization's reads, the filtered vector search and the
availability check. Needs Postgres with PostGIS and pgvector.
"""

import statistics
import time
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.search.fallback import FallbackLevel
from app.agent.search.query import SearchQuery
from app.db.models import Entity, ItemEntity, ItemStatus, SavedItem, User
from app.db.types import GeoPoint
from tests.agent.retrieval_support import MappedEmbedder, build_retriever, place_fields
from tests.database import TEST_EMBEDDING_DIMENSION

TYPICAL_QUERY_LIMIT_SECONDS = 0.2
SEEDED_ITEMS = 600
DISTRICTS = ("Kadıköy", "Brooklyn", "Shibuya", "Le Marais", "Palermo", "Kreuzberg")
MEASURED_RUNS = 5


def _vector(seed: int) -> list[float]:
    raw = [((seed * (index + 3)) % 11) - 5.0 for index in range(TEST_EMBEDDING_DIMENSION)]
    length = sum(value * value for value in raw) ** 0.5 or 1.0
    return [value / length for value in raw]


async def _seed(session: AsyncSession, *, resolved: bool) -> uuid.UUID:
    """Seed one user's places, linked to entities or, as in Phase 1, known only from hints."""
    user = User(display_name="Busy User", country="US")
    session.add(user)
    await session.flush()
    for index in range(SEEDED_ITEMS):
        district = DISTRICTS[index % len(DISTRICTS)]
        entity = Entity(
            type="place",
            source="google_places",
            external_id=f"speed-{index}",
            name=f"Place {index}",
            city=f"City of {district}",
            district=district,
            location=GeoPoint(latitude=40.0 + index / 1000, longitude=29.0),
        )
        item = SavedItem(
            user_id=user.id,
            platform="instagram",
            platform_post_id=f"speed-{index}",
            type="place",
            category="restaurant" if index % 2 else "café",
            collection="Food",
            fields=place_fields(
                (f"Place {index}", f"{district}, City of {district}", f"cuisine {index % 9}")
            ),
            summary=f"Saved place number {index}",
            embedding=_vector(index),
            status=ItemStatus.DONE,
        )
        session.add(item)
        if resolved:
            session.add(entity)
            await session.flush()
            session.add(ItemEntity(saved_item_id=item.id, entity_id=entity.id))
    await session.flush()
    return user.id


@pytest.mark.parametrize("resolved", [True, False], ids=["entities", "area-hints"])
async def test_a_typical_filtered_vector_search_takes_under_200_ms(
    db_session: AsyncSession, resolved: bool
) -> None:
    user_id = await _seed(db_session, resolved=resolved)
    query = SearchQuery(
        user_id=user_id,
        collection="Food",
        type="place",
        category="restaurant",
        # Brooklyn is every 6th place from the 2nd and "cuisine 4" every 9th from the 5th,
        # so about 30 restaurants meet both and L0 answers on its own.
        filters={"area": "Brooklyn", "cuisine": "cuisine 4"},
        text="cozy place for dinner",
    )
    retriever = build_retriever(embedder=MappedEmbedder(axes={"cozy place for dinner": 0}))
    await retriever.search(db_session, query)

    durations = []
    for _ in range(MEASURED_RUNS):
        started = time.perf_counter()
        result = await retriever.search(db_session, query)
        durations.append(time.perf_counter() - started)

    assert result.hits
    assert result.level is FallbackLevel.NORMAL
    assert statistics.median(durations) < TYPICAL_QUERY_LIMIT_SECONDS
