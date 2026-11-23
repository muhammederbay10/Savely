"""Building blocks that every content type is declared with.

A content type is data, not code paths: the pipeline and the search read these
declarations, so adding a type never needs an `if type == ...` anywhere else.

No declaration lists real-world values. Categories, cuisines, genres, subjects
and the like are open text in any language. The shared vocabulary keeps them
consistent, and a field or filter only says which part of the vocabulary its
values belong to.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CATEGORY_VOCABULARY = "category"
TAG_VOCABULARY = "tag"
# The field of an `EntityExtraction` that lists a post's mentions.
MENTIONS_FIELD = "entities"

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ContentTypeDefinitionError(ValueError):
    """A content type, or one of its parts, is declared in a way that cannot work."""


@dataclass(frozen=True)
class Vocabulary:
    """Marks an open-text schema field whose values are normalized against the vocabulary.

    The text model writes such a value in its own words, in any language. The
    normalization step then matches it with the existing values of the same
    vocabulary field, so "korean", "Korean food" and "Kore mutfağı" end up as
    one value. Put the marker on the whole field, also when the field can be
    null or holds several values:
    `Annotated[NonEmptyText | None, Vocabulary("cuisine")]`.

    Attributes:
        field: The part of the vocabulary the values belong to, e.g. `cuisine`.
    """

    field: str


class ExtractionModel(BaseModel):
    """Shared settings for everything the text model returns.

    Unknown keys are rejected, so a reply that does not follow the schema fails
    loudly instead of being stored half-understood.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def vocabulary_fields(model: type[ExtractionModel]) -> dict[str, str]:
    """Find the fields of a schema model that are normalized against the vocabulary.

    Only the model's own fields are listed. Models nested inside it, such as
    the mentions of an `EntityExtraction`, are asked separately.

    Args:
        model: The schema model to inspect.

    Returns:
        The vocabulary field of each marked field, by the field's name.
    """
    return {
        name: marker.field
        for name, field in model.model_fields.items()
        for marker in field.metadata
        if isinstance(marker, Vocabulary)
    }


class EntityMention(ExtractionModel):
    """One real-world thing a post mentions, before a resolver has looked it up.

    Types with a resolver subclass this with the fields their resolver searches by.
    """


# A post that fits no built-in type is extracted with exactly this schema, so it keeps
# its category, tags and summary. A built-in type's schema extends it.
class Extraction(ExtractionModel):
    """What the text model extracts from every post, whatever it is about."""

    summary: NonEmptyText = Field(
        description="One or two sentences on what the post is about, in the post's language."
    )
    category: Annotated[NonEmptyText, Vocabulary(CATEGORY_VOCABULARY)] = Field(
        description=(
            "What the post is, in a few plain words of your own, e.g. 'restaurant', "
            "'car review', 'skincare routine' or 'guitar lesson'. There is no list to pick from."
        )
    )
    tags: Annotated[tuple[NonEmptyText, ...], Vocabulary(TAG_VOCABULARY)] = Field(
        description=(
            "A few short labels in plain words for what the post is about, e.g. 'rooftop', "
            "'budget' or 'beginner'. There is no list to pick from. May be empty."
        )
    )


class EntityExtraction(Extraction):
    """The extraction of a type whose posts point at real-world entities.

    Attributes:
        entities: Every entity the post mentions, in the order they appear. A
            list post ("5 movies for the weekend") has several. It is empty
            when the text did not name one.
    """

    entities: tuple[EntityMention, ...]


class FilterKind(StrEnum):
    """The kind of value a search filter takes.

    Attributes:
        VALUE: An open value, such as a cuisine or a genre. What the user typed
            is normalized to a value that exists in their own saves.
        AREA: The name of a place at any level: a country, region, city,
            district or neighbourhood, anywhere in the world.
        LOCATION: A point and a distance around it.
        BOOLEAN: True or false.
        INTEGER: A whole number.
        TEXT: Free text that is matched as it is.
    """

    VALUE = "value"
    AREA = "area"
    LOCATION = "location"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    TEXT = "text"


class FilterSource(StrEnum):
    """Where search finds what a filter is checked against.

    Attributes:
        SAVED_VALUES: What indexing stored. A `VALUE`, `INTEGER`, `BOOLEAN` or
            `TEXT` filter reads the key named after the filter: in the item's
            `fields` when the extraction schema has it as a top-level field,
            otherwise per mention in `fields.entities`, where a linked
            entity's `data` counts too. An `AREA` filter reads a linked
            entity's address levels, or, for a mention with no linked entity,
            the parts of the mention's `mention_field` (decision record 0017).
            A `LOCATION` filter reads the entities' coordinates.
        COUNTRY_AVAILABILITY: Where an entity can be streamed in the asking
            user's country (`entity_availability`). Only for `VALUE` filters.
        USER_STATUS: Whether the asking user marked the item's entity as done
            (visited, watched, read). Only for `BOOLEAN` filters.
        OPENING_HOURS: The entity's opening hours, checked against the current
            time. Only for `BOOLEAN` filters.
    """

    SAVED_VALUES = "saved_values"
    COUNTRY_AVAILABILITY = "country_availability"
    USER_STATUS = "user_status"
    OPENING_HOURS = "opening_hours"


# The filter kinds each source other than `SAVED_VALUES` can serve.
_KINDS_OF_SOURCE: dict[FilterSource, frozenset[FilterKind]] = {
    FilterSource.COUNTRY_AVAILABILITY: frozenset({FilterKind.VALUE}),
    FilterSource.USER_STATUS: frozenset({FilterKind.BOOLEAN}),
    FilterSource.OPENING_HOURS: frozenset({FilterKind.BOOLEAN}),
}


@dataclass(frozen=True)
class SearchFilter:
    """An extra filter that a content type offers to search.

    `category` and `tags` are not declared here. Search offers them for every item.

    Attributes:
        name: The key used in a search's `filters` argument, e.g. `area`.
        kind: The kind of value the filter takes.
        description: What the filter selects, written for the response agent.
        vocabulary: The vocabulary field a `VALUE` filter's values belong to,
            e.g. `cuisine`. Its aliases let a question in one language find a
            value saved in another. `None` when the values come from somewhere
            else, such as a resolver, and for every other kind.
        source: Where search finds what the filter is checked against.
        mention_field: For an `AREA` filter, the key of a mention in
            `fields.entities` that holds where the post said it is, e.g.
            `area_hint`. Read only for a mention with no linked entity; a
            linked entity's address always wins. `None` means a mention with
            no linked entity never matches the area. Other kinds read the key
            named after the filter, so they never set it.

    Raises:
        ContentTypeDefinitionError: If a filter that is not a `VALUE` names a
            vocabulary field, a filter that is not an `AREA` names a mention
            field, or its source cannot serve its kind.
    """

    name: str
    kind: FilterKind
    description: str
    vocabulary: str | None = None
    source: FilterSource = FilterSource.SAVED_VALUES
    mention_field: str | None = None

    def __post_init__(self) -> None:
        """Check that the vocabulary and the mention field fit the kind, and the source fits."""
        if self.vocabulary is not None and self.kind is not FilterKind.VALUE:
            raise ContentTypeDefinitionError(
                f"Filter '{self.name}' is not an open value, so it cannot use the vocabulary"
            )
        if self.mention_field is not None and self.kind is not FilterKind.AREA:
            raise ContentTypeDefinitionError(
                f"Filter '{self.name}' is not an area, so it cannot name a mention field"
            )
        allowed_kinds = _KINDS_OF_SOURCE.get(self.source)
        if allowed_kinds is not None and self.kind not in allowed_kinds:
            raise ContentTypeDefinitionError(
                f"Filter '{self.name}' is a {self.kind} filter, which the source "
                f"'{self.source}' cannot serve"
            )


@dataclass(frozen=True)
class ContentType:
    """Everything the system needs to know about one built-in kind of saved post.

    Attributes:
        name: The identifier stored on a saved item, e.g. `place`.
        description: What the type covers. The decision model sees it as one of
            the options when it picks a post's type, next to "none of these".
        extraction_schema: The Pydantic model the text model fills in.
        resolver: The name of the resolver in `app/resolvers/` that looks up
            this type's entities, e.g. `tmdb`. `None` means the type has no
            real-world entity, and everything it knows stays on the saved item.
        search_filters: The filters the type adds to search.

    Raises:
        ContentTypeDefinitionError: If two filters share a name, the
            resolver and the extraction schema disagree on whether the type
            has entities, or a filter's mention field is not a field of the
            type's mentions.
    """

    name: str
    description: str
    extraction_schema: type[Extraction]
    resolver: str | None
    search_filters: tuple[SearchFilter, ...]

    def __post_init__(self) -> None:
        """Check that the parts of the declaration fit together."""
        self._require_unique_filter_names()
        self._require_resolver_matches_schema()
        self._require_mention_fields_exist()

    def _require_unique_filter_names(self) -> None:
        """Check that no two filters share a name.

        Raises:
            ContentTypeDefinitionError: If a filter name appears more than once.
        """
        names = [search_filter.name for search_filter in self.search_filters]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ContentTypeDefinitionError(
                f"Duplicate filters in content type '{self.name}': {', '.join(duplicates)}"
            )

    def _require_resolver_matches_schema(self) -> None:
        """Check that a resolver is declared exactly when the schema lists entities.

        Raises:
            ContentTypeDefinitionError: If only one of the two is present.
        """
        extracts_entities = issubclass(self.extraction_schema, EntityExtraction)
        if self.resolver is not None and not extracts_entities:
            raise ContentTypeDefinitionError(
                f"Content type '{self.name}' has a resolver, so its extraction schema "
                "must extend EntityExtraction"
            )
        if self.resolver is None and extracts_entities:
            raise ContentTypeDefinitionError(
                f"Content type '{self.name}' extracts entities, so it needs a resolver"
            )

    def _require_mention_fields_exist(self) -> None:
        """Check that every filter's mention field is a field of the type's mentions.

        Raises:
            ContentTypeDefinitionError: If a filter names a mention field the
                mentions do not have, or the type has no mentions.
        """
        mention_fields = _mention_fields(self.extraction_schema)
        for search_filter in self.search_filters:
            field = search_filter.mention_field
            if field is not None and field not in mention_fields:
                raise ContentTypeDefinitionError(
                    f"Filter '{search_filter.name}' of content type '{self.name}' reads "
                    f"'{field}', which its mentions do not have"
                )


def _mention_fields(schema: type[Extraction]) -> frozenset[str]:
    """List the fields of a schema's mentions; empty if it has none."""
    entities = schema.model_fields.get(MENTIONS_FIELD)
    if entities is None:
        return frozenset()
    return frozenset(
        name
        for mention in get_args(entities.annotation)
        if isinstance(mention, type) and issubclass(mention, EntityMention)
        for name in mention.model_fields
    )
