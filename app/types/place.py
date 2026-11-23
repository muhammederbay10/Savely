"""The `place` content type: somewhere to go, anywhere in the world."""

from typing import Annotated

from pydantic import Field

from app.types.base import (
    ContentType,
    EntityExtraction,
    EntityMention,
    FilterKind,
    FilterSource,
    NonEmptyText,
    SearchFilter,
    Vocabulary,
)

GOOGLE_PLACES_RESOLVER = "google_places"
CUISINE_VOCABULARY = "cuisine"
AREA_HINT_FIELD = "area_hint"


class PlaceMention(EntityMention):
    """One venue a post mentions."""

    name: NonEmptyText = Field(
        description="The venue's name exactly as the post writes or says it."
    )
    area_hint: NonEmptyText | None = Field(
        description=(
            "Where the post says the venue is, at whatever level it gives: a neighbourhood, "
            "district, city, region or country, e.g. 'Bebek', 'Brooklyn, New York' or "
            "'Shibuya'. Null if the post does not say."
        )
    )
    cuisine: Annotated[NonEmptyText | None, Vocabulary(CUISINE_VOCABULARY)] = Field(
        description=(
            "The kind of food or drink the venue serves, in a few plain words of your own, "
            "e.g. 'korean barbecue' or 'specialty coffee'. There is no list to pick from. "
            "Null if it serves none or the post is unclear."
        )
    )


class PlaceExtraction(EntityExtraction):
    """What the text model extracts from a post about one or more places."""

    entities: tuple[PlaceMention, ...] = Field(
        description=(
            "Every venue the post mentions, in order. A list post has several. "
            "Empty if no venue is named."
        )
    )


PLACE = ContentType(
    name="place",
    description=(
        "A real-world venue to go to, such as a restaurant, café, bar, shop, hotel or attraction."
    ),
    extraction_schema=PlaceExtraction,
    resolver=GOOGLE_PLACES_RESOLVER,
    search_filters=(
        SearchFilter(
            name="area",
            kind=FilterKind.AREA,
            description=(
                "Where the place is, at any level: a country, region, city, district or "
                "neighbourhood, e.g. 'Japan', 'Brooklyn' or 'Beşiktaş'."
            ),
            mention_field=AREA_HINT_FIELD,
        ),
        SearchFilter(
            name="near",
            kind=FilterKind.LOCATION,
            description="Places within a distance of a point, such as where the user is now.",
        ),
        SearchFilter(
            name="open_now",
            kind=FilterKind.BOOLEAN,
            description="True to keep only places that are open right now.",
            source=FilterSource.OPENING_HOURS,
        ),
        SearchFilter(
            name="cuisine",
            kind=FilterKind.VALUE,
            description="The kind of food or drink the place serves.",
            vocabulary=CUISINE_VOCABULARY,
        ),
        SearchFilter(
            name="price",
            kind=FilterKind.VALUE,
            description="How expensive the place is.",
        ),
    ),
)
