"""The fixed wording of the bot's DMs, in the user's language.

Eight messages have fixed wording: the save confirmation, the collection
confirmation, the "which one?" question, import progress, the error message,
one line per search result (`docs/ARCHITECTURE.md` §3.1, §3.3 and §3.5,
decision records 0004 and 0018), and the two messages of "delete my data": the
confirmation question and the final "done". Everything else
the bot says is written by the response agent (SAV-17).

**Caller contract.** Every template takes a `ReplyWording`: the bot's own words
in the user's language. Get it with `reply_wording(language, localized)`:

- `localized`, when given, is used as it is. This is how any language is
  served: the caller (the response agent, which already writes replies in the
  user's language) supplies wording for a language with no built-in wording.
- Otherwise built-in wording is used when it exists (English and Turkish).
- Otherwise `ReplyWordingUnavailableError` is raised. A user is never silently
  answered in another language (decision record 0005); the caller decides.

Every value the caller passes, such as a name, an open category, a genre, an
area, a platform, an option label or a count label, is shown exactly as given,
never translated or mapped to a fixed list (decision record 0005).
"""

import string
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Final

from app.agent.replies.messenger import MAX_QUICK_REPLIES, OutgoingMessage, QuickReply

# The payload of the "Yes, delete" button under the deletion question (record
# 0015, Q9). It is the bot's own fixed text, the same in every language; only a
# tap on that button reports it, a typed "yes" never does.
DELETION_CONFIRM_PAYLOAD: Final = "savely:delete_my_data:confirm"

# Joins the parts of one line ("Kronotrop, Beşiktaş · open now"). The middle dot
# is the bot's own layout, used in every language.
SEPARATOR: Final = " · "
_LANGUAGE_SUBTAG_SEPARATORS: Final = ("-", "_")

# The placeholders each wording field must contain, exactly. Fields not listed
# take none.
_PLACEHOLDERS: Final[dict[str, frozenset[str]]] = {
    "saved_named": frozenset({"item"}),
    "saved_as": frozenset({"category"}),
    "in_collection": frozenset({"collection"}),
    "saved_to": frozenset({"collection"}),
    "streaming_on": frozenset({"platforms"}),
    "import_progress": frozenset({"done", "total"}),
    "import_progress_with_counts": frozenset({"done", "total", "counts"}),
    "deletion_confirmation": frozenset({"minutes"}),
}


class ReplyWordingUnavailableError(LookupError):
    """There is no wording for the user's language, built in or supplied.

    Attributes:
        language: The language tag that was asked for.
    """

    def __init__(self, language: str) -> None:
        """Create the error.

        Args:
            language: The language tag that was asked for.
        """
        super().__init__(
            f"no reply wording for language {language!r}: pass localized wording for it"
        )
        self.language = language


@dataclass(frozen=True)
class ReplyWording:
    """The bot's own words for the fixed messages, in one language.

    Placeholders in braces are filled in by the templates and must appear
    exactly as listed, as bare names such as `{item}`: no format spec
    (`{item:.0}`), conversion (`{item!r}`), index or attribute, so a caller's
    value is always shown whole. No other braces are allowed. Wording that
    breaks this is refused when it is built, not when a message is rendered.

    Attributes:
        saved_named: Confirmation for an item with a name; `{item}`, e.g.
            "Saved: {item}".
        saved_as: Confirmation for an item without one; `{category}`, e.g.
            "Saved as {category}".
        in_collection: The collection, added to the save confirmation;
            `{collection}`, e.g. "in {collection}".
        saved_to: The collection confirmation, sent on its own when the save
            confirmation went out without it; `{collection}`, e.g. "Saved to {collection}".
        streaming_on: Where it streams; `{platforms}`, e.g. "on {platforms}".
        which_one: The "which one?" question, e.g. "Which one did you save?".
        none_of_these: The button for "none of the options".
        import_progress: Progress with nothing found yet; `{done}` and `{total}`.
        import_progress_with_counts: Progress with what was found; `{done}`,
            `{total}` and `{counts}`.
        error: The message sent when the bot failed to handle a message.
        open_now: Said of a place that is open now.
        closed_now: Said of a place that is closed now.
        list_separator: Joins values in a list, e.g. ", " or "、".
        deletion_confirmation: Asks the user to confirm "delete my data": it
            names what goes (all saves, stored media, the conversation, the
            profile), says it can't be undone, and that tapping the button
            confirms; `{minutes}` is how long the request waits.
        delete_button: The label of the button that confirms the deletion,
            e.g. "Yes, delete". Meta cuts labels after 20 characters.
        deletion_done: Says the data was deleted and that writing again starts fresh.
    """

    saved_named: str
    saved_as: str
    in_collection: str
    saved_to: str
    streaming_on: str
    which_one: str
    none_of_these: str
    import_progress: str
    import_progress_with_counts: str
    error: str
    open_now: str
    closed_now: str
    list_separator: str
    deletion_confirmation: str
    delete_button: str
    deletion_done: str

    def __post_init__(self) -> None:
        """Reject empty wording and wrong placeholders.

        Raises:
            ValueError: If a field is empty, has malformed braces, uses a
                placeholder with a format spec or conversion, or doesn't have
                exactly its placeholders. The message names the field.
        """
        for field in fields(self):
            text: str = getattr(self, field.name)
            # A separator may be only a space; every other field needs words.
            is_blank = not text if field.name == "list_separator" else not text.strip()
            if is_blank:
                raise ValueError(f"{field.name} must not be empty")
            expected = _PLACEHOLDERS.get(field.name, frozenset())
            if _placeholders(field.name, text) != expected:
                raise ValueError(
                    f"{field.name} must contain exactly the placeholders {sorted(expected)}"
                )


def _placeholders(field_name: str, text: str) -> frozenset[str]:
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError as error:
        raise ValueError(f"{field_name} has malformed braces") from error
    names: set[str] = set()
    for _, name, format_spec, conversion in parsed:
        if name is None:
            continue
        # A format spec or conversion can crash rendering ("{item:{x}}", "{item!z}")
        # or change the caller's value ("{item:.0}" shows nothing of it).
        if format_spec or conversion is not None:
            raise ValueError(
                f"{field_name} may only use bare placeholders such as {{{name}}}, "
                "without a format spec or conversion"
            )
        names.add(name)
    return frozenset(names)


_BUILT_IN_WORDINGS: Final[dict[str, ReplyWording]] = {
    "en": ReplyWording(
        saved_named="Saved: {item}",
        saved_as="Saved as {category}",
        in_collection="in {collection}",
        saved_to="Saved to {collection}",
        streaming_on="on {platforms}",
        which_one="Which one did you save?",
        none_of_these="None of these",
        import_progress="Imported {done}/{total}…",
        import_progress_with_counts="Imported {done}/{total}… {counts} so far",
        error="Sorry, something went wrong on my side. Please try again in a few minutes.",
        open_now="open now",
        closed_now="closed now",
        list_separator=", ",
        deletion_confirmation=(
            "This deletes all your saves, their stored media, our conversation and your "
            'profile. It can\'t be undone. To confirm, tap "Yes, delete" within {minutes} '
            "minutes; anything else cancels it."
        ),
        delete_button="Yes, delete",
        deletion_done=("Done: all your data is deleted. If you write again, you start fresh."),
    ),
    "tr": ReplyWording(
        saved_named="Kaydedildi: {item}",
        saved_as="{category} olarak kaydedildi",
        in_collection="{collection} koleksiyonunda",
        saved_to="{collection} koleksiyonuna kaydedildi",
        streaming_on="{platforms} platformunda",
        which_one="Hangisini kaydettin?",
        none_of_these="Hiçbiri",
        import_progress="{done}/{total} içe aktarıldı…",
        import_progress_with_counts="{done}/{total} içe aktarıldı… şu ana kadar {counts}",
        error="Üzgünüm, bir şeyler ters gitti. Lütfen birkaç dakika sonra tekrar dene.",
        open_now="şu an açık",
        closed_now="şu an kapalı",
        list_separator=", ",
        deletion_confirmation=(
            "Bu işlem tüm kayıtlarını, saklanan medyalarını, konuşmamızı ve profilini siler. "
            'Geri alınamaz. Onaylamak için {minutes} dakika içinde "Evet, sil" düğmesine '
            "dokun; başka bir mesaj işlemi iptal eder."
        ),
        delete_button="Evet, sil",
        deletion_done=("Tamam: tüm verilerin silindi. Tekrar yazarsan sıfırdan başlarsın."),
    ),
}


def has_built_in_wording(language: str) -> bool:
    """Tell whether a language has built-in wording.

    Args:
        language: A BCP 47 language tag, e.g. "tr" or "en-GB".

    Returns:
        True when its primary subtag has built-in wording.

    Raises:
        ValueError: If the tag is empty.
    """
    return _primary_subtag(language) in _BUILT_IN_WORDINGS


def reply_wording(language: str, localized: ReplyWording | None = None) -> ReplyWording:
    """Return the wording to use for a user.

    Args:
        language: The user's language as a BCP 47 tag, e.g. "tr" or "ja-JP".
        localized: Wording the caller supplies for that language. It is used
            when given, also for a language with built-in wording.

    Returns:
        `localized`, or the built-in wording for the language.

    Raises:
        ValueError: If the tag is empty.
        ReplyWordingUnavailableError: If there is no built-in wording for the
            language and none was supplied.
    """
    primary = _primary_subtag(language)
    if localized is not None:
        return localized
    if primary not in _BUILT_IN_WORDINGS:
        raise ReplyWordingUnavailableError(language)
    return _BUILT_IN_WORDINGS[primary]


def _primary_subtag(language: str) -> str:
    primary = language.strip().lower()
    for separator in _LANGUAGE_SUBTAG_SEPARATORS:
        primary = primary.split(separator, 1)[0]
    if not primary:
        raise ValueError("language must not be empty")
    return primary


@dataclass(frozen=True)
class SavedItemSummary:
    """What the save confirmation says about one saved item.

    Attributes:
        category: The item's open category, e.g. "car review". Shown when the
            item has no name.
        name: The real thing behind the item, e.g. "Inception", when one was
            resolved.
        year: The release year, for movies and series.
        details: Further values to show as given, e.g. ("Sci-fi",) or ("BMW M3",).
        streaming_platforms: Where the item can be streamed, e.g. ("Netflix",).
        emoji: An emoji for the item's kind, e.g. "🎬".
        collection: The collection the user named for it, e.g. "Food", shown as given.
    """

    category: str
    name: str | None = None
    year: int | None = None
    details: tuple[str, ...] = ()
    streaming_platforms: tuple[str, ...] = ()
    emoji: str | None = None
    collection: str | None = None


@dataclass(frozen=True)
class ChoiceOption:
    """One answer to a "which one?" question.

    Attributes:
        label: What the user sees, e.g. "Kronotrop Bebek".
        payload: What the webhook reports when the user taps it, e.g. the
            candidate's id for `resolve_choice`.
    """

    label: str
    payload: str


@dataclass(frozen=True)
class ImportCount:
    """How many items of one kind an import has found so far.

    Attributes:
        label: The kind in the user's language, already in plural, e.g. "places".
        count: How many.
    """

    label: str
    count: int


@dataclass(frozen=True)
class ResultLine:
    """One search result, sent as its own message.

    Attributes:
        position: The result's number, starting at 1.
        total: How many results the answer shows.
        name: The item's name, e.g. "Kronotrop"; `None` for an item with no
            name yet (e.g. one that is still being indexed).
        area: Where it is, e.g. "Beşiktaş", for places.
        details: Further values to show as given, e.g. ("Sci-fi",).
        open_now: Whether a place is open now; `None` when unknown or not a place.
        link: A link to show, e.g. a Maps, TMDB or post link.
        emoji: An emoji for the item's kind, e.g. "🍽".
    """

    position: int
    total: int
    name: str | None
    area: str | None = None
    details: tuple[str, ...] = ()
    open_now: bool | None = None
    link: str | None = None
    emoji: str | None = None


def save_confirmation(summary: SavedItemSummary, wording: ReplyWording) -> OutgoingMessage:
    """Build the DM that confirms a save and says how it was saved.

    Args:
        summary: What to say about the item.
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        E.g. "Saved: Inception (2010) · Sci-fi · on Netflix 🎬", or
        "Saved as 🚗 car review · BMW M3 · in Cars" for an item without a name
        that the user filed in a collection.
    """
    if summary.name is not None:
        item = summary.name if summary.year is None else f"{summary.name} ({summary.year})"
        parts = [wording.saved_named.format(item=item), *summary.details]
    else:
        category = _with_emoji_before(summary.emoji, summary.category)
        parts = [wording.saved_as.format(category=category), *summary.details]
    if summary.streaming_platforms:
        platforms = wording.list_separator.join(summary.streaming_platforms)
        parts.append(wording.streaming_on.format(platforms=platforms))
    if summary.collection is not None:
        parts.append(wording.in_collection.format(collection=summary.collection))
    text = SEPARATOR.join(parts)
    if summary.name is not None and summary.emoji is not None:
        text = f"{text} {summary.emoji}"
    return OutgoingMessage(text=text)


def collection_confirmation(collection: str, wording: ReplyWording) -> OutgoingMessage:
    """Build the DM that confirms the collection of an item whose save was already confirmed.

    Args:
        collection: The collection, as the user wrote it, e.g. "Food".
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        E.g. "Saved to Food".
    """
    return OutgoingMessage(text=wording.saved_to.format(collection=collection))


def which_one_question(
    options: Sequence[ChoiceOption], none_payload: str, wording: ReplyWording
) -> OutgoingMessage:
    """Build the "which one?" question, with one button per option.

    The options are numbered in the text, and each button starts with its
    number, so two options stay apart even when Meta cuts a long label.

    Args:
        options: The candidates, best first.
        none_payload: What the webhook reports when the user taps "None of these".
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        The question, with a button per option and a last "None of these" button.

    Raises:
        ValueError: If there are no options, or too many for the buttons.
    """
    max_options = MAX_QUICK_REPLIES - 1
    if not options or len(options) > max_options:
        raise ValueError(f"a 'which one?' question needs 1 to {max_options} options")
    numbered = [f"{number}. {option.label}" for number, option in enumerate(options, start=1)]
    buttons = [
        QuickReply(title=label, payload=option.payload)
        for label, option in zip(numbered, options, strict=True)
    ]
    buttons.append(QuickReply(title=wording.none_of_these, payload=none_payload))
    return OutgoingMessage(
        text="\n".join([wording.which_one, *numbered]), quick_replies=tuple(buttons)
    )


def import_progress(
    done: int, total: int, counts: Sequence[ImportCount], wording: ReplyWording
) -> OutgoingMessage:
    """Build an import progress update.

    Args:
        done: How many posts have been imported.
        total: How many posts the import has.
        counts: What was found so far, by kind. Kinds with a count of 0 are left out.
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        E.g. "Imported 230/500… 61 places, 40 movies, 25 study posts so far".

    Raises:
        ValueError: If `done` is negative or above `total`.
    """
    if not 0 <= done <= total:
        raise ValueError("done must be between 0 and total")
    found = [f"{count.count} {count.label}" for count in counts if count.count > 0]
    if not found:
        return OutgoingMessage(text=wording.import_progress.format(done=done, total=total))
    text = wording.import_progress_with_counts.format(
        done=done, total=total, counts=wording.list_separator.join(found)
    )
    return OutgoingMessage(text=text)


def error_message(wording: ReplyWording) -> OutgoingMessage:
    """Build the DM sent when the bot failed to handle the user's message.

    Args:
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        A short apology that asks the user to try again later.
    """
    return OutgoingMessage(text=wording.error)


def deletion_confirmation(minutes: int, wording: ReplyWording) -> OutgoingMessage:
    """Build the question that asks the user to confirm "delete my data", with its button.

    Args:
        minutes: How long the request waits for the confirmation.
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        What is deleted, that it can't be undone and how to confirm, with one
        "Yes, delete" button whose payload is `DELETION_CONFIRM_PAYLOAD`.
    """
    return OutgoingMessage(
        text=wording.deletion_confirmation.format(minutes=minutes),
        quick_replies=(QuickReply(title=wording.delete_button, payload=DELETION_CONFIRM_PAYLOAD),),
    )


def confirms_deletion(quick_reply_payload: str | None) -> bool:
    """Tell whether a message is a tap on the deletion question's "Yes, delete" button.

    The chat worker (SAV-18) sets `AgentTurn.deletion_confirmed` from it. The
    tools still check that a delivered question is waiting for this sender's
    next message and that it hasn't expired.

    Args:
        quick_reply_payload: `ChatMessage.quick_reply_payload` of the message.

    Returns:
        True only for the button's fixed payload; never for typed text.
    """
    return quick_reply_payload == DELETION_CONFIRM_PAYLOAD


def deletion_done(wording: ReplyWording) -> OutgoingMessage:
    """Build the last message after the user's data was deleted.

    Args:
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        That the data is gone and that writing again starts fresh.
    """
    return OutgoingMessage(text=wording.deletion_done)


def result_reply(
    line: ResultLine, reply_to_message_id: str | None, wording: ReplyWording
) -> OutgoingMessage:
    """Build the message for one search result.

    When the item has a message id, the message replies to it, so Instagram
    shows the saved reel above the line (decision record 0004).

    Args:
        line: The result to show.
        reply_to_message_id: The item's `source_message_id`, or `None` when it
            has none (e.g. an imported item); then put the post link in `line.link`.
        wording: The bot's words in the user's language, from `reply_wording`.

    Returns:
        E.g. "🍽 1/3 · Kronotrop, Beşiktaş · open now · https://maps…", as a
        reply to the saved reel.

    Raises:
        ValueError: If `position` is not between 1 and `total`.
    """
    if not 1 <= line.position <= line.total:
        raise ValueError("position must be between 1 and total")
    named = [part for part in (line.name, line.area) if part is not None]
    parts = [_with_emoji_before(line.emoji, f"{line.position}/{line.total}")]
    if named:
        parts.append(wording.list_separator.join(named))
    parts.extend(line.details)
    if line.open_now is not None:
        parts.append(wording.open_now if line.open_now else wording.closed_now)
    if line.link is not None:
        parts.append(line.link)
    return OutgoingMessage(text=SEPARATOR.join(parts), reply_to_message_id=reply_to_message_id)


def _with_emoji_before(emoji: str | None, text: str) -> str:
    return text if emoji is None else f"{emoji} {text}"
