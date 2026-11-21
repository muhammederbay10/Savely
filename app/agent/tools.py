"""The response agent's tools: the only way it reaches the user's saves (ARCHITECTURE §3.5).

The agent never queries the database. It calls these tools, and they do the
searching and the writing:

| Tool | What it does |
|---|---|
| `search_items` | Hybrid search with the type's filters, plus category and tags for every item |
| `near` | Search around a point, for the types with a location filter (places) |
| `get_item` | One item's details: what it is, where, links, streaming in the user's country |
| `update_item` | Correct one open value of one item; it never creates a synonym |
| `set_status` | Mark an item's place or title, or an item without one, as done or wanted |
| `set_profile` | Keep where the user lives (home area) and their country, as they told it |
| `request_data_deletion` | Start "delete my data": the bot asks the user to confirm |
| `confirm_data_deletion` | Delete everything of the sender, only on an explicit confirmation |

**Ownership comes from the DM, never from the model.** `Toolbox.for_user` binds
the tools to the sender's `user_id` and country. No tool takes a user id as an
argument, and an item id of another user is answered exactly like a missing
one.

**Deleting takes two steps, enforced here, not by the prompt.** The model can
never delete on its own word. `confirm_data_deletion` erases only when all of
these hold, and refuses otherwise:

- the confirmation question was delivered: `Toolbox.confirmation_delivered`
  stores the request only after the whole answer was sent;
- this turn is the user's next message: `begin_turn` takes the request out, so
  "no" or any other message uses it up, and a request of the same turn never counts;
- the caller reports an explicit confirmation with this message
  (`ToolContext.deletion_confirmed`): a tap on the question's "Yes, delete"
  button, whose fixed payload the chat worker (SAV-18) checks with
  `confirms_deletion`. A typed "yes" never counts; the question is asked again;
- the deadline (`DATA_DELETION_CONFIRM_SECONDS` after delivery) has not passed,
  checked right before erasing.

A successful erasure ends the turn (`ToolOutcome.completion`): the "done"
message was written in the user's language when the question was delivered, so
no model or tool runs after the user is gone.

**Searches are reported** with their arguments (`ToolOutcome.search`), so
short-term memory (SAV-18) can keep the filters behind the last results and a
follow-up such as "only the cheap ones" can repeat them with one more.

**Arguments are checked** against each tool's schema, and search filters
against the content type's declared filters (`plan_search`). A wrong argument
comes back to the model as an error it can correct, never as a crash. The
tools never name a content type or a filter: they read the registry.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.deletion_requests import DeletionRequests, PendingDeletion
from app.agent.replies.answers import ResultCard
from app.agent.search.fallback import FallbackLevel
from app.agent.search.normalization import NormalizationMethod
from app.agent.search.query import (
    CountryRequiredError,
    FilterValue,
    Near,
    SearchQuery,
    SearchQueryError,
)

# Re-exported, so the agent hands gaps on to the resolver without importing search itself.
from app.agent.search.retrieval import AvailabilityGap as AvailabilityGap
from app.agent.search.retrieval import HybridRetriever, SearchResult
from app.db.deletion import ErasureDeadlinePassedError, UserDataEraser
from app.db.models import (
    Entity,
    EntityAvailability,
    EntityStatus,
    ItemEntity,
    ItemStatus,
    PlatformAccount,
    SavedItem,
    User,
    UserEntity,
)
from app.db.types import GeoPoint
from app.ingestion.repository import lock_user
from app.models.base import ToolCall, ToolDefinition
from app.pipeline.vocabulary.corrections import CorrectionError, ItemCorrection
from app.pipeline.vocabulary.normalize import NormalizeOutcome
from app.pipeline.vocabulary.open_values import document_of_item, find_open_values
from app.types.base import ContentType, FilterKind, FilterSource
from app.types.registry import ContentTypeRegistry

logger = logging.getLogger(__name__)

# Storage contract with the resolvers (SAV-15): the link to the real thing, e.g.
# the Google Maps or TMDB page, is kept under this key of `entity.data`.
ENTITY_LINK_KEY: Final = "link"

SEARCH_ITEMS_TOOL: Final = "search_items"
NEAR_TOOL: Final = "near"
GET_ITEM_TOOL: Final = "get_item"
UPDATE_ITEM_TOOL: Final = "update_item"
SET_STATUS_TOOL: Final = "set_status"
REQUEST_DATA_DELETION_TOOL: Final = "request_data_deletion"
CONFIRM_DATA_DELETION_TOOL: Final = "confirm_data_deletion"
SET_PROFILE_TOOL: Final = "set_profile"

_NOT_FOUND: Final = "The user has no saved item with that id."
_ASK_USER_FOR: Final = "ask_user_for"
_COUNTRY: Final = "country"

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# ISO 3166-1 alpha-2, kept in upper case like `user.country` and the search's country.
CountryCode = Annotated[
    str, StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^[A-Za-z]{2}$")
]


class Notice(StrEnum):
    """Something the reply must tell the user about how the results were found.

    Attributes:
        BACKUP_MODE: Search ran in backup mode (L1: keyword search instead of
            meaning), so the list may miss items.
        BROADER_RESULTS: Some results were found only after relaxing the
            filters (L2), so they match less exactly, or are still being indexed.
        MAY_BE_INCOMPLETE: The search ran out of time, so results may be missing.
    """

    BACKUP_MODE = "backup_mode"
    BROADER_RESULTS = "broader_results"
    MAY_BE_INCOMPLETE = "may_be_incomplete"


@dataclass(frozen=True)
class PlatformRecipient:
    """Where the last DM goes after a user's data, platform accounts included, is deleted.

    Attributes:
        platform: e.g. `instagram`.
        platform_user_id: The user's id on that platform, e.g. the IGSID.
    """

    platform: str
    platform_user_id: str = field(repr=False)


@dataclass(frozen=True)
class SearchUsed:
    """One search the model ran: the tool and its checked arguments.

    Short-term memory keeps the searches behind the last results (the working
    set), and the agent shows them again on the next turn, so "only the cheap
    ones" can repeat the search with one more filter.

    Attributes:
        tool: `search_items` or `near`.
        arguments: The arguments as JSON values, without the ones left at
            their defaults. They hold the user's own words.
    """

    tool: str
    arguments: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True)
class DeletionConfirmation:
    """A confirmation question the bot asks; it counts only once delivered.

    Attributes:
        language: The BCP 47 tag of the language it is asked in.
        done_text: The final "done" message in that language, prepared now so
            nothing has to be written after the erasure.
    """

    language: str
    done_text: str = field(repr=False)


@dataclass(frozen=True)
class DeletionCompletion:
    """The end of a turn in which the user's data was deleted.

    Attributes:
        language: The language of `done_text`.
        done_text: The final "done" message.
        recipients: Where to send it, read before the deletion.
        saved_items: How many saved items were deleted.
    """

    language: str
    done_text: str = field(repr=False)
    recipients: tuple[PlatformRecipient, ...]
    saved_items: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ToolContext:
    """Who the tools work for. It comes from the DM, never from the model.

    Attributes:
        user_id: The sender of the DM.
        country: The sender's country (ISO 3166-1 alpha-2) from their profile;
            `None` if it is not known yet.
        deletion_confirmed: This message is a tap on the deletion question's
            "Yes, delete" button (record 0015, Q9); without it no deletion runs.
    """

    user_id: uuid.UUID
    country: str | None
    deletion_confirmed: bool = False


@dataclass(frozen=True)
class ToolOutcome:
    """What one tool call returned.

    Attributes:
        content: The JSON object the model reads.
        offered_item_ids: The items this call showed the model, best first.
            The answer may only list these and the working set.
        notices: What the reply must tell the user about the results.
        availability_gaps: Streaming availability for the resolver to fetch (SAV-15).
        asks_deletion_confirmation: The answer must be the fixed deletion
            question instead of the model's reply.
        completion: Set when the user's data was deleted; the turn ends with it.
        search: The search this call ran, when it was one and it ran.
    """

    content: Mapping[str, object] = field(repr=False)
    offered_item_ids: tuple[uuid.UUID, ...] = ()
    notices: frozenset[Notice] = frozenset()
    availability_gaps: tuple[AvailabilityGap, ...] = ()
    asks_deletion_confirmation: bool = False
    completion: DeletionCompletion | None = None
    search: SearchUsed | None = None

    def as_json(self) -> str:
        """Return the content as the JSON text the model reads."""
        return json.dumps(self.content, ensure_ascii=False, default=str)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PointArguments(_Arguments):
    """A point and a distance, as the model writes a location filter."""

    lat: float = Field(ge=-90, le=90, description="Latitude in degrees.")
    lng: float = Field(ge=-180, le=180, description="Longitude in degrees.")
    radius_m: float = Field(gt=0, description="Distance from the point, in metres.")

    def near(self) -> Near:
        """Return the search's `Near` value."""
        return Near(point=GeoPoint(latitude=self.lat, longitude=self.lng), radius_m=self.radius_m)


class SearchItemsArguments(_Arguments):
    """The arguments of `search_items`."""

    collection: NonEmptyText | None = Field(
        default=None, description="The user's own collection name, as they wrote it."
    )
    type: NonEmptyText | None = Field(
        default=None, description="A built-in content type; needed for `filters`."
    )
    category: NonEmptyText | None = Field(
        default=None, description="An open category in plain words, e.g. 'car review'."
    )
    tags: tuple[NonEmptyText, ...] = Field(
        default=(), description="Tags every result must have, in plain words."
    )
    filters: dict[str, str | int | bool | PointArguments] = Field(
        default_factory=dict,
        description="The type's own filters by name, with values in the user's words.",
    )
    text: NonEmptyText | None = Field(
        default=None, description="What to rank by meaning, e.g. 'cozy date-night place'."
    )
    platform: NonEmptyText | None = Field(
        default=None, description="Only items saved from this platform, e.g. 'instagram'."
    )
    saved_after: AwareDatetime | None = Field(
        default=None, description="Only items saved after this moment, ISO 8601 with a zone."
    )


class NearArguments(PointArguments):
    """The arguments of `near`."""

    category: NonEmptyText | None = Field(
        default=None, description="An open category in plain words, e.g. 'café'."
    )
    tags: tuple[NonEmptyText, ...] = Field(default=(), description="Tags every result must have.")
    text: NonEmptyText | None = Field(default=None, description="What to rank by meaning.")
    open_now: bool | None = Field(default=None, description="Only places open now.")


class GetItemArguments(_Arguments):
    """The arguments of `get_item`."""

    item_id: uuid.UUID = Field(description="The item's id, from a search or the working set.")


class UpdateItemArguments(_Arguments):
    """The arguments of `update_item`."""

    item_id: uuid.UUID = Field(description="The one item the user corrects.")
    path: NonEmptyText = Field(
        description=(
            "Which open value to correct, as `get_item` lists them, e.g. 'category', "
            "'tags.1', 'subject' or 'entities.0.cuisine'."
        )
    )
    value: NonEmptyText = Field(description="The right value, in the user's own words.")


class SetStatusArguments(_Arguments):
    """The arguments of `set_status`."""

    item_id: uuid.UUID = Field(description="The item whose real thing the user means.")
    status: Literal["done", "want"] = Field(
        description="'done' for visited, watched, read or done; 'want' for want to."
    )
    entity_position: int | None = Field(
        default=None,
        ge=0,
        description="Which of a list post's places or titles, as `get_item` numbers them.",
    )


class SetProfileArguments(_Arguments):
    """The arguments of `set_profile`; at least one is given."""

    home_area: NonEmptyText | None = Field(
        default=None,
        description="Where the user lives, in their own words, e.g. a district or neighbourhood.",
    )
    country: CountryCode | None = Field(
        default=None,
        description="The user's country as an ISO 3166-1 alpha-2 code, e.g. 'TR' or 'JP'.",
    )

    @model_validator(mode="after")
    def _something_to_set(self) -> "SetProfileArguments":
        if self.home_area is None and self.country is None:
            raise ValueError("give home_area, country or both")
        return self


class NoArguments(_Arguments):
    """The arguments of a tool that takes none. The user is always the DM's sender."""


class Toolbox:
    """Everything the tools need, shared by every user; `for_user` binds it to one sender."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        retriever: HybridRetriever,
        corrections: ItemCorrection,
        registry: ContentTypeRegistry,
        deletion_requests: DeletionRequests,
        eraser: UserDataEraser,
        deletion_window: timedelta,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Create the toolbox.

        Args:
            sessions: Opens a database session.
            retriever: The hybrid search.
            corrections: Applies a user's correction to one item.
            registry: The built-in content types, for filters and schemas.
            deletion_requests: Pending "delete my data" requests.
            eraser: Erases a user everywhere.
            deletion_window: How long a delivered question may be confirmed
                (`DATA_DELETION_CONFIRM_SECONDS`).
            now: The clock, for the requests' deadlines.
        """
        self._sessions = sessions
        self._retriever = retriever
        self._corrections = corrections
        self._registry = registry
        self._deletion_requests = deletion_requests
        self._eraser = eraser
        self._deletion_window = deletion_window
        self._now = now

    async def confirmation_delivered(
        self, user_id: uuid.UUID, confirmation: DeletionConfirmation
    ) -> None:
        """Make a delivered deletion question confirmable by the user's next message.

        Call it only after the whole answer was sent (`app.agent.send.send_answer`).

        Args:
            user_id: The DM's sender the question was sent to.
            confirmation: The question's language and prepared "done" message.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        now = self._now()
        pending = PendingDeletion(
            deadline=now + self._deletion_window,
            language=confirmation.language,
            done_text=confirmation.done_text,
        )
        await self._deletion_requests.save(user_id, pending, now)
        logger.info("data_deletion_confirmation_delivered")

    def for_user(self, context: ToolContext) -> "UserTools":
        """Return the tools of one sender.

        Args:
            context: The DM's sender, from the platform, never from the model.

        Returns:
            Tools that only ever read or change this user's saves.
        """
        return UserTools(
            context,
            sessions=self._sessions,
            retriever=self._retriever,
            corrections=self._corrections,
            registry=self._registry,
            deletion_requests=self._deletion_requests,
            eraser=self._eraser,
            now=self._now,
        )


class UserTools:
    """The tools of one user. Every read and write is scoped to that user."""

    def __init__(
        self,
        context: ToolContext,
        *,
        sessions: async_sessionmaker[AsyncSession],
        retriever: HybridRetriever,
        corrections: ItemCorrection,
        registry: ContentTypeRegistry,
        deletion_requests: DeletionRequests,
        eraser: UserDataEraser,
        now: Callable[[], datetime],
    ) -> None:
        """Create the tools; use `Toolbox.for_user`.

        Args:
            context: The user the tools work for.
            sessions: Opens a database session.
            retriever: The hybrid search.
            corrections: Applies a user's correction to one item.
            registry: The built-in content types.
            deletion_requests: Pending "delete my data" requests.
            eraser: Erases a user everywhere.
            now: The clock.
        """
        self._context = context
        self._sessions = sessions
        self._retriever = retriever
        self._corrections = corrections
        self._registry = registry
        self._deletion_requests = deletion_requests
        self._eraser = eraser
        self._now = now
        self._pending_deletion: PendingDeletion | None = None

    async def begin_turn(self) -> None:
        """Take the user's pending deletion request out, before the turn's first tool call.

        Only this turn can confirm it. Whatever the user wrote, the request is
        gone afterwards, so "no" or any other message cancels it.

        Raises:
            redis.exceptions.RedisError: If Redis cannot be reached.
        """
        self._pending_deletion = await self._deletion_requests.take(self._context.user_id)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Describe every tool for the model.

        Returns:
            One definition per tool, with a JSON Schema of its arguments.
        """
        return (
            ToolDefinition(
                name=SEARCH_ITEMS_TOOL,
                description=(
                    "Search the user's own saved posts. `category` and `tags` work for every "
                    "item; `filters` need `type`. Values are the user's own words; they are "
                    "matched to what the user saved. " + _type_filters_description(self._registry)
                ),
                parameters=SearchItemsArguments.model_json_schema(),
            ),
            ToolDefinition(
                name=NEAR_TOOL,
                description="Search the user's saved places around a point.",
                parameters=NearArguments.model_json_schema(),
            ),
            ToolDefinition(
                name=GET_ITEM_TOOL,
                description=(
                    "Get one saved item: what it is, its open values with their paths, where "
                    "it is, links and where it streams in the user's country."
                ),
                parameters=GetItemArguments.model_json_schema(),
            ),
            ToolDefinition(
                name=UPDATE_ITEM_TOOL,
                description=(
                    "Correct one open value of one saved item, e.g. its category. Only that "
                    "item changes; no other item and no synonym."
                ),
                parameters=UpdateItemArguments.model_json_schema(),
            ),
            ToolDefinition(
                name=REQUEST_DATA_DELETION_TOOL,
                description=(
                    "The user asks, in any words or language, to delete their data or "
                    "account. Starts the deletion; the bot then asks them to confirm."
                ),
                parameters=NoArguments.model_json_schema(),
            ),
            ToolDefinition(
                name=CONFIRM_DATA_DELETION_TOOL,
                description=(
                    "The user's message clearly confirms the deletion the bot just asked "
                    "about. Deletes everything of theirs only if the platform also reports "
                    "an explicit confirmation; otherwise nothing happens."
                ),
                parameters=NoArguments.model_json_schema(),
            ),
            ToolDefinition(
                name=SET_STATUS_TOOL,
                description=(
                    "Mark the place or title behind a saved item, or the item itself when it "
                    "has none (e.g. a study video), as done (visited, watched, read, studied) "
                    "or as something the user wants to do."
                ),
                parameters=SetStatusArguments.model_json_schema(),
            ),
            ToolDefinition(
                name=SET_PROFILE_TOOL,
                description=(
                    "Remember where the user lives (their home area) or their country, when "
                    "they tell you. Used for 'near home' and for streaming in their country."
                ),
                parameters=SetProfileArguments.model_json_schema(),
            ),
        )

    async def call(self, tool_call: ToolCall) -> ToolOutcome:
        """Run one tool call of the model.

        Args:
            tool_call: The tool and its arguments, as the model wrote them.

        Returns:
            The result, or an error the model can act on: a wrong argument, an
            unknown tool, an unknown item, or a question to ask the user.

        Raises:
            ModelError: If a model call behind a tool failed.
            sqlalchemy.exc.SQLAlchemyError: If the database failed.
        """
        # The tool's name only: a call may be the one that deletes the user.
        logger.info("agent_tool_called", extra={"tool": tool_call.name})
        handlers: dict[str, Callable[[Mapping[str, object]], Awaitable[ToolOutcome]]] = {
            SEARCH_ITEMS_TOOL: lambda arguments: self._search_items(
                SearchItemsArguments.model_validate(arguments)
            ),
            NEAR_TOOL: lambda arguments: self._near(NearArguments.model_validate(arguments)),
            GET_ITEM_TOOL: lambda arguments: self._get_item(
                GetItemArguments.model_validate(arguments)
            ),
            UPDATE_ITEM_TOOL: lambda arguments: self._update_item(
                UpdateItemArguments.model_validate(arguments)
            ),
            SET_STATUS_TOOL: lambda arguments: self._set_status(
                SetStatusArguments.model_validate(arguments)
            ),
            SET_PROFILE_TOOL: lambda arguments: self._set_profile(
                SetProfileArguments.model_validate(arguments)
            ),
            REQUEST_DATA_DELETION_TOOL: lambda arguments: self._request_data_deletion(
                NoArguments.model_validate(arguments)
            ),
            CONFIRM_DATA_DELETION_TOOL: lambda arguments: self._confirm_data_deletion(
                NoArguments.model_validate(arguments)
            ),
        }
        handler = handlers.get(tool_call.name)
        if handler is None:
            return _error(f"There is no tool named '{tool_call.name}'.")
        try:
            return await handler(tool_call.arguments)
        except ValidationError as error:
            return _error(f"Invalid arguments: {_validation_summary(error)}")

    async def result_cards(self, item_ids: Sequence[uuid.UUID]) -> tuple[ResultCard, ...]:
        """Load what a reply needs to show each item; not a tool the model calls.

        Args:
            item_ids: The items, in the order to show them.

        Returns:
            One card per item the user owns, in the given order. Ids of other
            users' items, or of deleted ones, are left out.
        """
        async with self._sessions() as session:
            return tuple(await _cards(session, self._context.user_id, item_ids))

    async def _search_items(self, arguments: SearchItemsArguments) -> ToolOutcome:
        filters: dict[str, FilterValue] = {
            name: value.near() if isinstance(value, PointArguments) else value
            for name, value in arguments.filters.items()
        }
        return await self._search(
            SearchUsed(SEARCH_ITEMS_TOOL, _json_arguments(arguments)),
            SearchQuery(
                user_id=self._context.user_id,
                country=self._context.country,
                collection=arguments.collection,
                type=arguments.type,
                category=arguments.category,
                tags=arguments.tags,
                filters=filters,
                text=arguments.text,
                platform=arguments.platform,
                saved_after=arguments.saved_after,
            ),
        )

    async def _near(self, arguments: NearArguments) -> ToolOutcome:
        located = _type_with_source(self._registry, FilterKind.LOCATION, None)
        if located is None:
            return _error("No content type can be searched by location.")
        content_type, location_filter = located
        filters: dict[str, FilterValue] = {location_filter: arguments.near()}
        if arguments.open_now is not None:
            hours = _type_with_source(
                self._registry, FilterKind.BOOLEAN, FilterSource.OPENING_HOURS
            )
            if hours is None or hours[0] is not content_type:
                return _error("Search cannot check opening hours for these items.")
            filters[hours[1]] = arguments.open_now
        return await self._search(
            SearchUsed(NEAR_TOOL, _json_arguments(arguments)),
            SearchQuery(
                user_id=self._context.user_id,
                country=self._context.country,
                type=content_type.name,
                category=arguments.category,
                tags=arguments.tags,
                filters=filters,
                text=arguments.text,
            ),
        )

    async def _search(self, used: SearchUsed, query: SearchQuery) -> ToolOutcome:
        try:
            async with self._sessions() as session:
                result = await self._retriever.search(session, query)
        except CountryRequiredError as error:
            return ToolOutcome(content={"error": str(error), _ASK_USER_FOR: _COUNTRY})
        except SearchQueryError as error:
            return _error(str(error))
        item_ids = tuple(hit.item_id for hit in result.hits)
        async with self._sessions() as session:
            cards = await _cards(session, self._context.user_id, item_ids)
        fully_indexed = {hit.item_id: hit.fully_indexed for hit in result.hits}
        notices = _notices(result)
        content: dict[str, object] = {
            "found": len(cards),
            "results": [
                {
                    "number": number,
                    "item_id": str(card.item_id),
                    "names": list(card.names),
                    "label": card.label,
                    "area": card.area,
                    "still_indexing": not fully_indexed[card.item_id],
                }
                for number, card in enumerate(cards, start=1)
            ],
            "notices": sorted(notices),
            "relaxed": list(result.relaxed),
            "matched_wording": [
                {"wrote": wording.wording, "matched": wording.value}
                for wording in result.normalized
                if wording.method is not NormalizationMethod.UNCHANGED
                and wording.wording != wording.value
            ],
        }
        return ToolOutcome(
            content=content,
            offered_item_ids=tuple(card.item_id for card in cards),
            notices=notices,
            availability_gaps=result.availability_gaps,
            search=used,
        )

    async def _get_item(self, arguments: GetItemArguments) -> ToolOutcome:
        async with self._sessions() as session:
            item = await _owned_item(session, self._context.user_id, arguments.item_id)
            if item is None:
                return _error(_NOT_FOUND)
            entities = await _linked_entities(session, [item.id])
            linked = [entity for _, entity in entities.get(item.id, [])]
            streaming = await _streaming(session, linked, self._context.country)
            statuses = await _entity_statuses(
                session, self._context.user_id, [entity.id for entity in linked]
            )
        content = {
            "item_id": str(item.id),
            "type": item.type,
            "category": item.category,
            "tags": item.tags,
            "collection": item.collection,
            "summary": item.summary,
            "fields": item.fields,
            "open_value_paths": _open_value_paths(self._registry, item),
            "post_url": item.post_url,
            "saved_at": item.saved_at.isoformat(),
            "still_indexing": item.status != ItemStatus.DONE,
            # Done or wanted, for an item without entities; see each entity otherwise.
            "status": item.user_status,
            "entities": [
                {
                    "position": position,
                    "name": entity.name,
                    "area": _area(entity),
                    "city": entity.city,
                    "country": entity.country,
                    "link": _link(entity),
                    "streaming": streaming.get(entity.id),
                    "status": statuses.get(entity.id),
                }
                for position, entity in entities.get(item.id, [])
            ],
        }
        return ToolOutcome(content=content, offered_item_ids=(item.id,))

    async def _update_item(self, arguments: UpdateItemArguments) -> ToolOutcome:
        try:
            corrected = await self._corrections.correct(
                self._context.user_id, arguments.item_id, arguments.path, arguments.value
            )
        except CorrectionError as error:
            return _error(str(error))
        return ToolOutcome(
            content={
                "item_id": str(arguments.item_id),
                "path": corrected.path,
                "previous": corrected.previous,
                "value": corrected.value,
                "kept_private": corrected.outcome is NormalizeOutcome.PRIVATE,
            },
            offered_item_ids=(arguments.item_id,),
        )

    async def _set_status(self, arguments: SetStatusArguments) -> ToolOutcome:
        async with self._sessions() as session, session.begin():
            if not await lock_user(session, self._context.user_id):
                return _error(_NOT_FOUND)
            item = await _owned_item(session, self._context.user_id, arguments.item_id)
            if item is None:
                return _error(_NOT_FOUND)
            entities = (await _linked_entities(session, [item.id])).get(item.id, [])
            if not entities:
                return _status_on_item(item, arguments)
            chosen = _chosen_entity(entities, arguments.entity_position)
            if isinstance(chosen, str):
                return _error(chosen)
            await session.execute(
                insert(UserEntity)
                .values(
                    user_id=self._context.user_id,
                    entity_id=chosen.id,
                    status=EntityStatus(arguments.status),
                )
                .on_conflict_do_update(
                    index_elements=[UserEntity.user_id, UserEntity.entity_id],
                    set_={"status": EntityStatus(arguments.status)},
                )
            )
        return ToolOutcome(
            content={"item_id": str(item.id), "name": chosen.name, "status": arguments.status},
            offered_item_ids=(item.id,),
        )

    async def _set_profile(self, arguments: SetProfileArguments) -> ToolOutcome:
        changes = arguments.model_dump(exclude_none=True)
        async with self._sessions() as session, session.begin():
            if not await lock_user(session, self._context.user_id):
                return _error("The user's profile could not be found.")
            await session.execute(
                update(User).where(User.id == self._context.user_id).values(**changes)
            )
        if arguments.country is not None:
            # A search later in this turn already uses the country the user just gave.
            self._context = replace(self._context, country=arguments.country)
        # Which fields only: where a user lives is personal data.
        logger.info("user_profile_updated", extra={"fields": sorted(changes)})
        return ToolOutcome(content={"saved": changes})

    async def _request_data_deletion(self, arguments: NoArguments) -> ToolOutcome:
        # Nothing is stored yet: the request becomes confirmable only once the
        # question was delivered (`Toolbox.confirmation_delivered`).
        # The event only, never who asked (owner update of 20 Nov 2026).
        logger.info("data_deletion_requested")
        return ToolOutcome(
            content={"question": "the bot asks the user to confirm"},
            asks_deletion_confirmation=True,
        )

    async def _confirm_data_deletion(self, arguments: NoArguments) -> ToolOutcome:
        pending, self._pending_deletion = self._pending_deletion, None
        if pending is None:
            return _deletion_refused()
        if not self._context.deletion_confirmed:
            # Typed text, even "yes", never deletes (record 0015, Q9): while the
            # question is still open, the bot asks it again, with its button.
            if self._now() < pending.deadline:
                return ToolOutcome(
                    content={"asked_again": "only the Yes, delete button confirms"},
                    asks_deletion_confirmation=True,
                )
            return _deletion_refused()
        async with self._sessions() as session:
            recipients = await _platform_recipients(session, self._context.user_id)
        # Checked right before the erasure, so time spent in this turn counts, and
        # again by the eraser once it holds the user's lock, which it may wait for.
        if self._now() >= pending.deadline:
            return _deletion_refused()
        try:
            deleted_items = await self._eraser.erase(
                self._context.user_id, deadline=pending.deadline, now=self._now
            )
        except ErasureDeadlinePassedError:
            return _deletion_refused()
        if deleted_items is None:
            return _deletion_refused()
        return ToolOutcome(
            content={"deleted": True, "saved_items": deleted_items},
            completion=DeletionCompletion(
                language=pending.language,
                done_text=pending.done_text,
                recipients=recipients,
                saved_items=deleted_items,
            ),
        )


def _deletion_refused() -> ToolOutcome:
    return _error(
        "Nothing was deleted: there is no delivered deletion question that this message "
        "explicitly confirms in time. Call request_data_deletion if the user wants to "
        "delete their data."
    )


async def _platform_recipients(
    session: AsyncSession, user_id: uuid.UUID
) -> tuple[PlatformRecipient, ...]:
    rows = await session.execute(
        select(PlatformAccount.platform, PlatformAccount.platform_user_id)
        .where(PlatformAccount.user_id == user_id)
        .order_by(PlatformAccount.platform)
    )
    return tuple(
        PlatformRecipient(platform, platform_user_id) for platform, platform_user_id in rows
    )


def _error(message: str) -> ToolOutcome:
    return ToolOutcome(content={"error": message})


def _json_arguments(arguments: BaseModel) -> dict[str, object]:
    return arguments.model_dump(mode="json", exclude_defaults=True)


def _validation_summary(error: ValidationError) -> str:
    # Only where and what went wrong; the input values are left out.
    return "; ".join(
        f"{'.'.join(str(part) for part in issue['loc']) or 'arguments'}: {issue['msg']}"
        for issue in error.errors()
    )


def _notices(result: SearchResult) -> frozenset[Notice]:
    notices: set[Notice] = set()
    if any(step.level is FallbackLevel.DEGRADED for step in result.fallbacks):
        notices.add(Notice.BACKUP_MODE)
    if any(hit.level is FallbackLevel.RELAXED for hit in result.hits):
        notices.add(Notice.BROADER_RESULTS)
    if result.incomplete:
        notices.add(Notice.MAY_BE_INCOMPLETE)
    return frozenset(notices)


def _type_filters_description(registry: ContentTypeRegistry) -> str:
    described = [
        f"Type '{content_type.name}' ({content_type.description}) filters: "
        + "; ".join(
            f"{search_filter.name} ({search_filter.kind}): {search_filter.description}"
            for search_filter in content_type.search_filters
        )
        for content_type in registry.all()
    ]
    return " ".join(described)


def _type_with_source(
    registry: ContentTypeRegistry, kind: FilterKind, source: FilterSource | None
) -> tuple[ContentType, str] | None:
    for content_type in registry.all():
        for search_filter in content_type.search_filters:
            if search_filter.kind is kind and (source is None or search_filter.source is source):
                return content_type, search_filter.name
    return None


async def _owned_item(
    session: AsyncSession, user_id: uuid.UUID, item_id: uuid.UUID
) -> SavedItem | None:
    result = await session.execute(
        select(SavedItem).where(SavedItem.id == item_id, SavedItem.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def _linked_entities(
    session: AsyncSession, item_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, list[tuple[int, Entity]]]:
    rows = await session.execute(
        select(ItemEntity.saved_item_id, ItemEntity.position, Entity)
        .join(Entity, Entity.id == ItemEntity.entity_id)
        .where(ItemEntity.saved_item_id.in_(list(item_ids)))
        .order_by(ItemEntity.saved_item_id, ItemEntity.position)
    )
    linked: dict[uuid.UUID, list[tuple[int, Entity]]] = {}
    for item_id, position, entity in rows:
        linked.setdefault(item_id, []).append((position, entity))
    return linked


async def _streaming(
    session: AsyncSession, entities: Sequence[Entity], country: str | None
) -> dict[uuid.UUID, object]:
    if country is None or not entities:
        return {}
    rows = await session.execute(
        select(EntityAvailability.entity_id, EntityAvailability.providers).where(
            EntityAvailability.entity_id.in_([entity.id for entity in entities]),
            EntityAvailability.country == country.strip().upper(),
        )
    )
    return {entity_id: providers for entity_id, providers in rows}


async def _cards(
    session: AsyncSession, user_id: uuid.UUID, item_ids: Sequence[uuid.UUID]
) -> list[ResultCard]:
    if not item_ids:
        return []
    rows = await session.execute(
        select(SavedItem).where(SavedItem.id.in_(list(item_ids)), SavedItem.user_id == user_id)
    )
    items = {item.id: item for item in rows.scalars()}
    entities = await _linked_entities(session, list(items))
    cards: list[ResultCard] = []
    for item_id in dict.fromkeys(item_ids):
        item = items.get(item_id)
        if item is None:
            continue
        linked = [entity for _, entity in entities.get(item_id, [])]
        first = linked[0] if linked else None
        cards.append(
            ResultCard(
                item_id=item.id,
                names=tuple(entity.name for entity in linked),
                label=item.category,
                area=_area(first) if first is not None else None,
                link=_link(first) if first is not None else None,
                post_url=item.post_url,
                source_message_id=item.source_message_id,
            )
        )
    return cards


def _area(entity: Entity) -> str | None:
    """Return the narrowest address level a place has, e.g. its neighbourhood or district."""
    return entity.neighbourhood or entity.district or entity.city


def _link(entity: Entity) -> str | None:
    link = entity.data.get(ENTITY_LINK_KEY)
    return link if isinstance(link, str) and link.strip() else None


def _open_value_paths(registry: ContentTypeRegistry, item: SavedItem) -> list[str]:
    """List the paths `update_item` can correct, e.g. `category` or `entities.0.cuisine`."""
    if item.category is None:
        return []
    schema = registry.extraction_schema(item.type)
    document = document_of_item(
        schema, summary=item.summary, category=item.category, tags=item.tags, fields=item.fields
    )
    return [open_value.path for open_value in find_open_values(schema, document)]


def _status_on_item(item: SavedItem, arguments: "SetStatusArguments") -> ToolOutcome:
    """Keep the status on an item without an entity, e.g. a study post (record 0015, Q6).

    The caller holds the owner's lock; the session's transaction commits it.
    """
    if arguments.entity_position is not None:
        return _error("The item is not linked to a place or title; leave entity_position out.")
    item.user_status = EntityStatus(arguments.status)
    return ToolOutcome(
        content={"item_id": str(item.id), "name": item.category, "status": arguments.status},
        offered_item_ids=(item.id,),
    )


async def _entity_statuses(
    session: AsyncSession, user_id: uuid.UUID, entity_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, str]:
    if not entity_ids:
        return {}
    rows = await session.execute(
        select(UserEntity.entity_id, UserEntity.status).where(
            UserEntity.user_id == user_id, UserEntity.entity_id.in_(list(entity_ids))
        )
    )
    return {entity_id: status for entity_id, status in rows}


def _chosen_entity(entities: Sequence[tuple[int, Entity]], position: int | None) -> Entity | str:
    """Pick the entity a status is for; a string is an error for the model."""
    if position is None:
        if len(entities) == 1:
            return entities[0][1]
        listed = ", ".join(f"{index}: {entity.name}" for index, entity in entities)
        return (
            f"The item mentions several things; ask which one and pass entity_position ({listed})."
        )
    for index, entity in entities:
        if index == position:
            return entity
    return f"The item has no entity at position {position}."
