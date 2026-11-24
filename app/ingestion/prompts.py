"""The wording text routing sends to the decision model, in one place, with a version.

Deciding whether a text right after a share names a collection
(`app.ingestion.text_routing`) takes its wording from one `RoutingPrompts`
object, so the wording lives here and not in the routing code, and tests pass
their own. The `version` is written in the routing's log line, so the live
routing check can name the wording it scored. Any change to the wording
changes the version.

The input template fills in the user's text through a named placeholder. It
must contain exactly its placeholders, written bare, and no other braces; the
user's text is only ever a value, never a template.
"""

import string
from dataclasses import dataclass, fields
from typing import Final

# The placeholders each template must contain, exactly. Fields not listed take none.
_PLACEHOLDERS: Final[dict[str, frozenset[str]]] = {
    "collection_name_input": frozenset({"text"}),
}


@dataclass(frozen=True)
class RoutingPrompts:
    """Everything text routing says to the decision model.

    Attributes:
        version: Names this wording, e.g. `2026-11-24.1`.
        collection_name_question: The yes/no question: does the text only name
            a collection to file the just-shared post in?
        collection_name_input: The text as the decision model reads it; `{text}`.
    """

    version: str
    collection_name_question: str
    collection_name_input: str

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


# The examples are in several languages on purpose: the question must work in any
# language, and the code holds no word lists (decision record 0005). Change the
# version with any change to the wording, and run the live routing check
# (docs/development.md) before and after.
ROUTING_PROMPTS: Final = RoutingPrompts(
    version="2026-11-24.1",
    collection_name_question=(
        "The user just shared a post with a bot that keeps their saved posts, and then sent "
        "this message. Is the message only the name of a collection, folder or list to file "
        "that post in, in any language (for example 'Food', 'for my Movies collection', "
        "'Filmler', '勉強')? Answer no for a question, a request, a greeting, a comment about "
        "the post or anything else that expects a reply."
    ),
    collection_name_input="Message sent right after sharing a post: {text}",
)
