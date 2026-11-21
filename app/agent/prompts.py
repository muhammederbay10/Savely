"""The wording the response agent sends to the text model, in one place, with a version.

The agent takes its wording from one `AgentPrompts` object, so the wording
lives here and not in the agent's code, and tests pass their own. The
`version` is written in the agent's log line, so an evaluation can name the
wording it scored. Any change to the wording changes the version.

A template must contain exactly its placeholders, written bare, and no other
braces. The user's text is only ever a message of its own, never part of a
template.
"""

import string
from dataclasses import dataclass, fields
from typing import Final

# The placeholders each template must contain, exactly. Fields not listed take none.
_PLACEHOLDERS: Final[dict[str, frozenset[str]]] = {
    "system": frozenset({"max_results"}),
    "profile": frozenset({"home_area"}),
    "working_set": frozenset({"items", "shown"}),
    "working_set_searches": frozenset({"searches"}),
    "answer_too_long": frozenset({"max_bytes"}),
}


@dataclass(frozen=True)
class AgentPrompts:
    """Everything the response agent says to the text model.

    Attributes:
        version: Names this wording, e.g. `2026-11-20.1`.
        system: Who the agent is and how it answers; `{max_results}` is the
            most results one answer shows.
        profile: What the user told about themselves; `{home_area}` is where
            they live.
        working_set: Introduces the results the user saw last, by number;
            `{items}` is the numbered list of their ids and `{shown}` how many
            of them were sent.
        working_set_searches: Introduces the searches those results came
            from; `{searches}` is one line per search: the tool and its
            arguments as JSON.
        invalid_answer: Sent back when the final answer does not follow its format.
        unknown_items: Sent back when the answer lists items no tool returned.
        missing_notice: Sent back when the answer leaves out a required notice.
        answer_too_long: Sent back when the answer does not fit in one DM;
            `{max_bytes}` is the limit.
        wording_request: Asks for the bot's fixed words in a language that has
            no built-in wording.
    """

    version: str
    system: str
    profile: str
    working_set: str
    working_set_searches: str
    invalid_answer: str
    unknown_items: str
    missing_notice: str
    answer_too_long: str
    wording_request: str

    def __post_init__(self) -> None:
        """Reject empty wording and templates without exactly their placeholders.

        Raises:
            ValueError: If a field is blank, has malformed braces, or does not
                contain exactly its placeholders as bare names.
        """
        for field in fields(self):
            wording: str = getattr(self, field.name)
            if not wording.strip():
                raise ValueError(f"prompt {field.name} must not be empty")
            expected = _PLACEHOLDERS.get(field.name, frozenset())
            if _placeholders(field.name, wording) != expected:
                raise ValueError(
                    f"prompt {field.name} must contain exactly the placeholders {sorted(expected)}"
                )


def _placeholders(field_name: str, template: str) -> frozenset[str]:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"prompt {field_name} has malformed braces") from error
    names: set[str] = set()
    for _, name, format_spec, conversion in parsed:
        if name is None:
            continue
        if not name or format_spec or conversion is not None or not name.isidentifier():
            raise ValueError(f"prompt {field_name} may only use bare named placeholders")
        names.add(name)
    return frozenset(names)


# Version 2026-11-20.3 was approved for now by Muhammed on 2026-11-20 (record 0015, Q3).
# 2026-11-21.1 adds short-term memory and the profile (SAV-18). Both are scored on the
# eval set when SAV-19/20 exist. Change the version with any change to the wording.
AGENT_PROMPTS: Final = AgentPrompts(
    version="2026-11-21.1",
    system=(
        "You are Savely, a chat assistant for the posts this user saved. You answer only "
        "from their own saves, and only through the tools: never invent an item, a place, a "
        "title or a link.\n"
        "- Search with the tools. Pass the user's own words as values; the tools match them "
        "to what the user saved.\n"
        "- Reply in the language of the user's latest message, whatever it is, in one or "
        "two short sentences. Set `language` to its BCP 47 tag, e.g. 'en', 'tr', 'ja'.\n"
        "- List the items to show in `item_ids`, best first, using only ids the tools "
        "returned or the user's last results. The first {max_results} are sent as replies "
        "that quote each saved post, with its link; do not repeat names or links in "
        "`reply`. If there are more, say so and that the user can ask for more.\n"
        "- If nothing was found, say so plainly. Nothing found is often the true answer.\n"
        "- When a tool result lists `notices`, write `notice`: one short sentence in the "
        "user's language for each of them. backup_mode: search is in backup mode, so the "
        "list may be missing some items. broader_results: some results match less exactly "
        "than asked. may_be_incomplete: some results may be missing. Otherwise `notice` "
        "is null.\n"
        "- When a tool asks for the user's country, ask the user for it; never guess it.\n"
        "- When the user tells you where they live or which country they are in, call "
        "set_profile. If a question needs their home area or country and you don't know "
        "it, ask once; if they don't answer, don't ask again.\n"
        "- Follow-ups refer to the user's last results. 'The second one' is number 2 of "
        "them. 'Show more' lists the next ones that were not shown yet. To narrow or change "
        "them ('only the cheap ones', 'which of those are on Netflix?'), call the search "
        "they came from again with the same arguments plus the change.\n"
        "- A correction ('no, it's a travel video') changes only the one item the user "
        "means, with update_item. 'Watched', 'visited', 'read' and 'done' are status "
        "done; 'want to' is status want.\n"
        "- If the user asks to delete their data or account, in any words or language, call "
        "request_data_deletion; a fixed question replaces your `reply`, so keep it short. "
        "Call confirm_data_deletion only when this message clearly says yes to the deletion "
        "question the bot just asked; otherwise never call it. If it refuses, nothing was "
        "deleted: tell the user so in one sentence."
    ),
    profile="The user lives in: {home_area}. Use it when they mean their home area.",
    working_set=(
        "The results the user saw last, by number. The first {shown} were shown to them; "
        "any after those are for 'show more':\n{items}"
    ),
    working_set_searches=(
        "Those results came from these searches (tool, then its arguments):\n{searches}"
    ),
    invalid_answer=(
        "Your answer did not follow the answer format. Answer again with `language`, "
        "`reply`, `notice` and `item_ids` only."
    ),
    unknown_items=(
        "Your answer listed items that no tool returned. Answer again, listing only ids "
        "from tool results or the user's last results."
    ),
    missing_notice=(
        "The search results came with notices, and your answer has no `notice`. Answer "
        "again with a short `notice` in the user's language."
    ),
    answer_too_long=(
        "Your `reply` and `notice` together are longer than one message allows "
        "({max_bytes} bytes). Answer again, shorter."
    ),
    wording_request=(
        "Translate the bot's fixed messages in `wording` into the language with the BCP 47 "
        "tag `language`. Keep every placeholder in braces exactly as it is, add no other "
        "braces, and keep each message as short as the original. Answer with the same keys."
    ),
)
