"""The response agent: understands a DM, calls tools, and writes a short answer (ARCHITECTURE §3.5).

One answer is a loop of at most `AGENT_MAX_STEPS` model calls, plus one retry
of an answer that was cut off:

1. **Tool steps.** The text model reads the conversation with the tools and
   no answer format, and calls tools or stops. Sending the format next to
   the tools made the configured model skip the tools and answer at once,
   claiming results it never searched for (SAV-38), so a tool step never
   sends it.
2. Tool calls run through `app/agent/tools.py`, which is the only way to the
   user's saves. The agent itself never queries the database.
3. **The answer step.** When the model stops calling tools, a reply that is
   already a valid answer object is taken as it is. Otherwise the model is
   asked for its answer once more, now with the answer format and with tool
   calls forbidden. The last step is always an answer step, so the loop ends
   with an answer or an explicit error.
4. The final answer is a JSON object: the user's language, a short reply in
   it, a notice when search fell back, the ids of the items to show, and
   `outcome`, which says whether the reply shows results. It is checked
   before it is accepted: it may list only items the tools returned (or the
   user's last results), it must carry a notice when a search fell back, an
   answer that says it shows results must list them (and one that found
   nothing lists none), and it must fit in one DM. A wrong answer is sent back
   to the model with what was wrong, within the same step limit; a malformed
   one is asked for again in the answer step, any other goes back to a tool
   step, so the model can search before it answers again.
5. **A cut-off answer.** An answer that hits the output limit is no JSON
   (`SchemaOutputError`). It is asked for again in the answer step, shorter
   and with at most `AGENT_MAX_RESULTS_PER_ANSWER` items. The first such retry
   of a turn gets one call beyond the step limit, so a cut-off answer at the
   last step still recovers; another cut-off uses the normal steps. A prose
   reply cut off in a tool step is dropped, not kept as the answer's draft.

**"Delete my data".** `request_data_deletion` makes the answer the fixed
confirmation question in the user's language, and prepares the fixed "done"
message in the same language (`AgentAnswer.deletion_confirmation`); the caller
makes it confirmable only after delivering it (`app.agent.send.send_answer`).
A successful `confirm_data_deletion` ends the turn at once with that "done"
message: no model call, translation or other tool runs after the erasure.

**The cap.** An answer lists at most `AGENT_MAX_RESULTS_PER_ANSWER` items: the
answer format says so (`maxItems`), and so does the prompt, so the answer stays
short. The code never relies on it: only the first that many are shown. The
rest are kept in order for "show more" (`AgentAnswer.more_item_ids`): first the
items the answer listed beyond the cap, then the unshown results of the
searches the answer presents. Each shown item belongs to the latest search that
returned it, so a broad search that a narrower one replaced adds nothing. When
the answer continues the last results without a new search ("show more"), the
last results not shown yet. Each shown item becomes
a reply quoting the saved post (`app/agent/replies/answers.py`, decision
record 0004).

**Working set.** "Mark the second one as watched" needs the results the user
saw last. The caller passes them in `AgentTurn.working_set`, in the order they
were numbered, with how many were shown and the searches they came from, and
stores `AgentAnswer.working_set` and `AgentAnswer.searches` for the next turn.
The model sees the list by number and the searches with their arguments, so
"show more" lists the next ones and "only the cheap ones" repeats the search
with one more filter. Keeping it between messages is short-term memory
(`app/memory/session.py`, SAV-18).

**Profile.** The user's home area, when known, is shown to the model, so
"near home" needs no question; `set_profile` keeps it when the user tells it.
"""

import json
import logging
import math
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.prompts import AgentPrompts
from app.agent.replies.answers import AnswerMessages, ResultCard
from app.agent.replies.messenger import MAX_TEXT_BYTES
from app.agent.replies.templates import (
    ReplyWording,
    ReplyWordingUnavailableError,
    deletion_confirmation,
    deletion_done,
    reply_wording,
)
from app.agent.tools import (
    AvailabilityGap,
    DeletionCompletion,
    DeletionConfirmation,
    Notice,
    PlatformRecipient,
    SearchUsed,
    Toolbox,
    ToolContext,
    UserTools,
)
from app.config.agent import AgentSettings
from app.models.base import (
    GenerationRequest,
    GenerationResult,
    Generator,
    Message,
    OutputSchema,
    ToolDefinition,
)
from app.models.errors import SchemaOutputError

logger = logging.getLogger(__name__)

_ANSWER_SCHEMA_NAME = "answer"
_WORDING_SCHEMA_NAME = "reply_wording"
# The built-in wording that a translation starts from.
_SOURCE_LANGUAGE = "en"
_SECONDS_PER_MINUTE = 60
# The finish reason of a reply that hit the output limit.
_CUT_OFF_FINISH_REASON = "length"


class AgentError(Exception):
    """The agent could not produce an answer. The caller sends the error message."""


class AgentStepLimitError(AgentError):
    """The model used every step without giving a valid answer."""


class AgentWordingError(AgentError):
    """The model gave no usable wording for a language without built-in wording."""


@dataclass(frozen=True)
class AgentTurn:
    """One message to answer.

    Attributes:
        user_id: The DM's sender, from the platform. The tools only ever read
            and change this user's saves.
        country: The sender's country from their profile (ISO 3166-1
            alpha-2); `None` if not known yet, and then the agent asks.
        text: What the user wrote.
        history: The conversation before this message, oldest first.
        working_set: The items of the user's last answer, in the order they
            were numbered, so "the second one" can be resolved.
        deletion_confirmed: This message is a tap on the deletion question's
            "Yes, delete" button: the chat worker (SAV-18) sets it only when
            `confirms_deletion(message.quick_reply_payload)` holds (record 0015,
            Q9). Without it no deletion runs, whatever the model does.
        home_area: Where the user lives, from their profile; `None` if not known.
        working_set_shown: How many of `working_set` were sent to the user;
            `None` means all of them.
        working_set_searches: The searches `working_set` came from.
        message_number: The message's number among the sender's messages, from
            ingestion. A deletion question can be confirmed only by the turn of
            the message right after it; `None` confirms none.
    """

    user_id: uuid.UUID
    country: str | None
    text: str = field(repr=False)
    history: Sequence[Message] = ()
    working_set: Sequence[uuid.UUID] = ()
    deletion_confirmed: bool = False
    home_area: str | None = field(default=None, repr=False)
    working_set_shown: int | None = None
    working_set_searches: Sequence[SearchUsed] = ()
    message_number: int | None = None


@dataclass(frozen=True)
class AgentAnswer:
    """The answer to one message.

    Attributes:
        language: The BCP 47 tag of the language the answer is in.
        messages: What to send: the intro, then the items, capped.
        notices: What the intro had to say about how results were found.
        more_item_ids: Results beyond the cap, for "show more".
        availability_gaps: Streaming availability the resolver should fetch (SAV-15).
        deletion_confirmation: Set when `messages` is the deletion question;
            plan its request with `Toolbox.deletion_request` and keep it with
            `Toolbox.keep_deletion_request` once the question was sent.
        deleted_recipients: Set when the user's data was deleted in this turn:
            where to send `messages`, read before the deletion, because the
            user's platform accounts are gone now.
        searches: The searches the results came from, for the next turn:
            this turn's searches, or the last results' searches when the
            answer lists items without a new search (e.g. "show more").
            Empty when the answer lists no items.
    """

    language: str
    messages: AnswerMessages
    notices: frozenset[Notice]
    more_item_ids: tuple[uuid.UUID, ...]
    availability_gaps: tuple[AvailabilityGap, ...]
    deletion_confirmation: DeletionConfirmation | None = None
    deleted_recipients: tuple[PlatformRecipient, ...] = ()
    searches: tuple[SearchUsed, ...] = ()

    @property
    def working_set(self) -> tuple[uuid.UUID, ...]:
        """Return every result of this answer, shown or not, in order, for the next turn."""
        shown = tuple(card.item_id for card in self.messages.results)
        return shown + self.more_item_ids


class AnswerOutcome(StrEnum):
    """What the model says its reply does, so a claim can be checked against `item_ids`.

    The model reports it in the answer, in any language the user writes, so
    the check never reads the reply's words.
    """

    RESULTS = "results"
    NO_RESULTS = "no_results"
    OTHER = "other"


class _FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str = Field(min_length=1)
    reply: str = Field(min_length=1)
    notice: str | None
    item_ids: tuple[uuid.UUID, ...]
    outcome: AnswerOutcome

    def outcome_matches_items(self) -> bool:
        if self.outcome is AnswerOutcome.RESULTS:
            return bool(self.item_ids)
        if self.outcome is AnswerOutcome.NO_RESULTS:
            return not self.item_ids
        return True

    def intro(self) -> str:
        notice = (self.notice or "").strip()
        return self.reply.strip() if not notice else f"{self.reply.strip()}\n\n{notice}"


def _answer_schema(max_results: int) -> OutputSchema:
    """The final answer's format; `item_ids` holds at most `max_results` ids."""
    return OutputSchema(
        name=_ANSWER_SCHEMA_NAME,
        schema={
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "description": "BCP 47 tag of the reply's language.",
                },
                "reply": {"type": "string", "description": "The short reply, in that language."},
                "notice": {
                    "type": ["string", "null"],
                    "description": "Required when a tool result listed notices; otherwise null.",
                },
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max_results,
                    "description": (
                        f"At most {max_results} items to show, best first. The other "
                        "results are kept for 'show more'."
                    ),
                },
                "outcome": {
                    "type": "string",
                    "enum": [outcome.value for outcome in AnswerOutcome],
                    "description": (
                        "results: the reply shows the items in item_ids, which is not empty. "
                        "no_results: a search found nothing that matches; item_ids is empty. "
                        "other: anything else, such as a question to the user."
                    ),
                },
            },
            "required": ["language", "reply", "notice", "item_ids", "outcome"],
            "additionalProperties": False,
        },
    )


_WORDING_SCHEMA = OutputSchema(
    name=_WORDING_SCHEMA_NAME,
    schema={
        "type": "object",
        "properties": {wording.name: {"type": "string"} for wording in fields(ReplyWording)},
        "required": [wording.name for wording in fields(ReplyWording)],
        "additionalProperties": False,
    },
)


@dataclass
class _Turn:
    """What one answer has gathered so far."""

    tools: UserTools
    allowed_item_ids: set[uuid.UUID]
    previous_searches: tuple[SearchUsed, ...]
    # The last results the user has not seen yet, in order, for "show more".
    previous_unshown: tuple[uuid.UUID, ...]
    searches: list[SearchUsed] = field(default_factory=list)
    # Each of this turn's searches' results, in call order, each in its own order.
    search_results: list[tuple[uuid.UUID, ...]] = field(default_factory=list)
    notices: set[Notice] = field(default_factory=set)
    availability_gaps: dict[tuple[uuid.UUID, str], AvailabilityGap] = field(default_factory=dict)
    asks_deletion_confirmation: bool = False


class ResponseAgent:
    """Answers one DM with the text model and the tools."""

    def __init__(
        self,
        *,
        generator: Generator,
        toolbox: Toolbox,
        prompts: AgentPrompts,
        settings: AgentSettings,
    ) -> None:
        """Create the agent.

        Args:
            generator: The text model (DeepSeek V4.1 Flash in production).
            toolbox: The tools, bound to each sender in `answer`.
            prompts: The wording sent to the model.
            settings: The step limit, the result cap and the reply length.
        """
        self._generator = generator
        self._toolbox = toolbox
        self._prompts = prompts
        self._settings = settings
        self._answer_schema = _answer_schema(settings.agent_max_results_per_answer)

    async def answer(self, turn: AgentTurn) -> AgentAnswer:
        """Answer one message.

        Args:
            turn: The message, its sender and the conversation so far.

        Returns:
            The answer to send.

        Raises:
            AgentStepLimitError: If no valid answer came within `AGENT_MAX_STEPS`
                (and the one retry of a cut-off answer).
            AgentWordingError: If the answer's language has no wording and the
                model gave none that is usable.
            ModelError: If a model call failed.
            sqlalchemy.exc.SQLAlchemyError: If the database failed behind a tool.
        """
        tools = self._toolbox.for_user(
            ToolContext(
                user_id=turn.user_id,
                country=turn.country,
                deletion_confirmed=turn.deletion_confirmed,
                message_number=turn.message_number,
            )
        )
        await tools.begin_turn()
        state = _Turn(
            tools=tools,
            allowed_item_ids=set(turn.working_set),
            previous_searches=tuple(turn.working_set_searches),
            previous_unshown=_unshown(turn),
        )
        messages = self._opening_messages(turn)
        definitions = tools.definitions()
        max_steps = self._settings.agent_max_steps
        limit = max_steps
        answering = False
        step = 0
        while step < limit:
            step += 1
            if step == limit and not answering:
                messages.append(Message(role="system", content=self._prompts.answer_now))
                answering = True
            try:
                result = await self._generator.generate(
                    self._request(messages, definitions, answering=answering)
                )
            except SchemaOutputError:
                if not answering:
                    raise
                messages.append(Message(role="system", content=self._answer_cut_off()))
                if limit == max_steps:
                    limit += 1
                logger.warning("agent_answer_cut_off", extra={"step": step})
                continue
            if result.tool_calls:
                tool_messages, completion = await self._run_tools(result, state)
                if completion is not None:
                    answer = _completed_answer(completion)
                    self._log_answer(step, answer)
                    return answer
                messages.extend(tool_messages)
                answering = False
                continue
            final = _parse_answer(result.structured if answering else _json_or_none(result.text))
            if final is None and not answering:
                # The model is done with the tools but wrote no answer object:
                # ask for it in the answer step, which sends the format. A draft
                # cut off at the output limit is dropped, so it isn't repeated.
                cut_off = result.finish_reason == _CUT_OFF_FINISH_REASON
                messages.extend(self._answer_step_messages("" if cut_off else result.text))
                answering = True
                continue
            problem = self._prompts.invalid_answer
            if final is not None:
                rejection = self._check_answer(final, state)
                if rejection is None:
                    answer = await self._finish(final, state)
                    self._log_answer(step, answer)
                    return answer
                problem = rejection
            messages.append(Message(role="assistant", content=result.text))
            messages.append(Message(role="system", content=problem))
            # A malformed answer is asked for again with the format; any other
            # problem may need a search, so the model gets the tools back.
            answering = final is None
        logger.warning("agent_step_limit_reached", extra={"steps": step})
        raise AgentStepLimitError(f"no valid answer within {step} steps")

    def _request(
        self,
        messages: Sequence[Message],
        definitions: Sequence[ToolDefinition],
        *,
        answering: bool,
    ) -> GenerationRequest:
        """Build a tool step's request (tools, no format) or the answer step's (format, no tools).

        The answer step still lists the tools, with calls forbidden, because the
        conversation already holds tool calls and their results.
        """
        return GenerationRequest(
            messages=tuple(messages),
            output_schema=self._answer_schema if answering else None,
            tools=definitions,
            tool_choice="none" if answering else "auto",
            max_output_tokens=self._settings.agent_max_output_tokens,
        )

    def _answer_cut_off(self) -> str:
        return self._prompts.answer_cut_off.format(
            max_results=self._settings.agent_max_results_per_answer
        )

    def _answer_step_messages(self, draft: str) -> list[Message]:
        """Keep the model's prose reply, if any, and ask for it as an answer object."""
        messages = [Message(role="assistant", content=draft)] if draft.strip() else []
        messages.append(Message(role="system", content=self._prompts.answer_now))
        return messages

    def _opening_messages(self, turn: AgentTurn) -> list[Message]:
        system = self._prompts.system.format(
            max_results=self._settings.agent_max_results_per_answer
        )
        messages = [Message(role="system", content=system)]
        if turn.home_area is not None:
            messages.append(
                Message(
                    role="system", content=self._prompts.profile.format(home_area=turn.home_area)
                )
            )
        if turn.working_set:
            messages.extend(self._working_set_messages(turn))
        messages.extend(turn.history)
        messages.append(Message(role="user", content=turn.text))
        return messages

    def _working_set_messages(self, turn: AgentTurn) -> list[Message]:
        numbered = "\n".join(
            f"{number}. {item_id}" for number, item_id in enumerate(turn.working_set, start=1)
        )
        shown = len(turn.working_set) if turn.working_set_shown is None else turn.working_set_shown
        messages = [
            Message(
                role="system",
                content=self._prompts.working_set.format(items=numbered, shown=shown),
            )
        ]
        if turn.working_set_searches:
            searches = "\n".join(
                f"{search.tool} {json.dumps(search.arguments, ensure_ascii=False)}"
                for search in turn.working_set_searches
            )
            messages.append(
                Message(
                    role="system",
                    content=self._prompts.working_set_searches.format(searches=searches),
                )
            )
        return messages

    async def _run_tools(
        self, result: GenerationResult, state: _Turn
    ) -> tuple[list[Message], DeletionCompletion | None]:
        """Run the reply's tool calls in order; stop at once if one deleted the user."""
        messages = [Message(role="assistant", content=result.text, tool_calls=result.tool_calls)]
        for tool_call in result.tool_calls:
            outcome = await state.tools.call(tool_call)
            if outcome.completion is not None:
                return messages, outcome.completion
            state.allowed_item_ids.update(outcome.offered_item_ids)
            state.notices.update(outcome.notices)
            for gap in outcome.availability_gaps:
                state.availability_gaps[(gap.entity_id, gap.country)] = gap
            state.asks_deletion_confirmation |= outcome.asks_deletion_confirmation
            if outcome.search is not None:
                state.searches.append(outcome.search)
                state.search_results.append(outcome.offered_item_ids)
            messages.append(
                Message(role="tool", content=outcome.as_json(), tool_call_id=tool_call.id)
            )
        return messages, None

    def _check_answer(self, final: _FinalAnswer, state: _Turn) -> str | None:
        """Check a well-formed final answer; return what to tell the model, or `None` if valid."""
        if not set(final.item_ids) <= state.allowed_item_ids:
            return self._prompts.unknown_items
        if state.notices and not (final.notice or "").strip():
            return self._prompts.missing_notice
        # The deletion question replaces the reply, so what it claims is never sent.
        if not state.asks_deletion_confirmation and not final.outcome_matches_items():
            return self._prompts.outcome_mismatch
        if len(final.intro().encode()) > MAX_TEXT_BYTES:
            return self._prompts.answer_too_long.format(max_bytes=MAX_TEXT_BYTES)
        return None

    async def _finish(self, final: _FinalAnswer, state: _Turn) -> AgentAnswer:
        if state.asks_deletion_confirmation:
            return await self._deletion_question(final.language)
        cards = await state.tools.result_cards(final.item_ids)
        cap = self._settings.agent_max_results_per_answer
        shown: tuple[ResultCard, ...] = cards[:cap]
        wording = await self._wording(final.language) if shown else None
        searches = (tuple(state.searches) or state.previous_searches) if cards else ()
        return AgentAnswer(
            language=final.language,
            messages=AnswerMessages(intro=final.intro(), results=shown, wording=wording),
            notices=frozenset(state.notices),
            more_item_ids=_more_item_ids(shown, cards[cap:], state),
            availability_gaps=tuple(state.availability_gaps.values()),
            searches=searches,
        )

    async def _deletion_question(self, language: str) -> AgentAnswer:
        """Ask the fixed deletion question, and prepare the "done" message in the same words."""
        wording = await self._wording(language)
        minutes = math.ceil(self._settings.data_deletion_confirm_seconds / _SECONDS_PER_MINUTE)
        question = deletion_confirmation(minutes, wording)
        return AgentAnswer(
            language=language,
            messages=AnswerMessages(
                intro=question.text,
                results=(),
                wording=wording,
                intro_quick_replies=question.quick_replies,
            ),
            notices=frozenset(),
            more_item_ids=(),
            availability_gaps=(),
            deletion_confirmation=DeletionConfirmation(
                language=language, done_text=deletion_done(wording).text
            ),
        )

    async def _wording(self, language: str) -> ReplyWording:
        """Return the bot's fixed words in the answer's language, built in or translated."""
        try:
            return reply_wording(language)
        except ReplyWordingUnavailableError:
            pass
        source = asdict(reply_wording(_SOURCE_LANGUAGE))
        result = await self._generator.generate(
            GenerationRequest(
                messages=[
                    Message(role="system", content=self._prompts.wording_request),
                    Message(
                        role="user",
                        content=json.dumps(
                            {"language": language, "wording": source}, ensure_ascii=False
                        ),
                    ),
                ],
                output_schema=_WORDING_SCHEMA,
                max_output_tokens=self._settings.agent_max_output_tokens,
            )
        )
        structured = result.structured
        if structured is None:
            raise AgentWordingError(f"no wording was given for language {language!r}")
        try:
            return ReplyWording(**{name: str(value) for name, value in structured.items()})
        except (TypeError, ValueError) as error:
            raise AgentWordingError(f"unusable wording for language {language!r}") from error

    def _log_answer(self, steps: int, answer: AgentAnswer) -> None:
        # Counts and names only: a turn can end with the user deleted (record 0015).
        logger.info(
            "agent_answered",
            extra={
                "prompts_version": self._prompts.version,
                "steps": steps,
                "results": len(answer.messages.results),
                "more_results": len(answer.more_item_ids),
                "notices": sorted(answer.notices),
                "language": answer.language,
            },
        )


def _unshown(turn: AgentTurn) -> tuple[uuid.UUID, ...]:
    """Return the last results the user was not shown yet, in order."""
    shown = len(turn.working_set) if turn.working_set_shown is None else turn.working_set_shown
    return tuple(turn.working_set[shown:])


def _more_item_ids(
    shown: Sequence[ResultCard], listed_beyond_cap: Sequence[ResultCard], state: _Turn
) -> tuple[uuid.UUID, ...]:
    """Return the results kept for "show more", in order; none when nothing is shown.

    First the items the answer listed beyond the cap, then the results it did
    not show: those of the searches the answer presents (`_presented_results`),
    or, when the answer continues the last results without a new search, the
    last results not shown yet.
    """
    if not shown:
        return ()
    shown_ids = [card.item_id for card in shown]
    pool: Sequence[uuid.UUID] = _presented_results(shown_ids, state.search_results)
    if not pool and set(shown_ids) <= set(state.previous_unshown):
        pool = state.previous_unshown
    candidates = [card.item_id for card in listed_beyond_cap] + list(pool)
    return tuple(item_id for item_id in dict.fromkeys(candidates) if item_id not in shown_ids)


def _presented_results(
    shown_ids: Sequence[uuid.UUID], search_results: Sequence[tuple[uuid.UUID, ...]]
) -> tuple[uuid.UUID, ...]:
    """Return the results of the searches the answer presents, in call order.

    Each shown item belongs to the latest search that returned it, so a broad
    search that a narrower one replaced ("everything", then "restaurants")
    adds nothing; an answer drawn from two searches keeps both.
    """
    presented: set[int] = set()
    for item_id in shown_ids:
        latest = next(
            (
                index
                for index in reversed(range(len(search_results)))
                if item_id in search_results[index]
            ),
            None,
        )
        if latest is not None:
            presented.add(latest)
    ordered = (item_id for index in sorted(presented) for item_id in search_results[index])
    return tuple(dict.fromkeys(ordered))


def _json_or_none(text: str) -> object:
    """Read a tool step's reply as JSON; `None` when it is prose."""
    try:
        return json.loads(text)
    except ValueError:
        return None


def _parse_answer(candidate: object) -> _FinalAnswer | None:
    """Return the answer object if it follows the answer format, else `None`."""
    try:
        return _FinalAnswer.model_validate(candidate)
    except ValidationError:
        return None


def _completed_answer(completion: DeletionCompletion) -> AgentAnswer:
    """End a turn in which the user's data was deleted, with the prepared "done" message."""
    return AgentAnswer(
        language=completion.language,
        messages=AnswerMessages(intro=completion.done_text, results=(), wording=None),
        notices=frozenset(),
        more_item_ids=(),
        availability_gaps=(),
        deleted_recipients=completion.recipients,
    )
