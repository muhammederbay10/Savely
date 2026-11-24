import dataclasses

import pytest

from app.agent.replies.messenger import MAX_QUICK_REPLIES, QuickReply, build_message_body
from app.agent.replies.templates import (
    DELETION_CONFIRM_PAYLOAD,
    ChoiceOption,
    ImportCount,
    ReplyWording,
    ReplyWordingUnavailableError,
    ResultLine,
    SavedItemSummary,
    collection_confirmation,
    confirms_deletion,
    deletion_confirmation,
    deletion_done,
    error_message,
    has_built_in_wording,
    import_progress,
    reply_wording,
    result_reply,
    save_confirmation,
    which_one_question,
)
from tests.agent.support import RECIPIENT_ID, SAVED_MESSAGE_ID

INCEPTION = SavedItemSummary(
    category="movie",
    name="Inception",
    year=2010,
    details=("Sci-fi",),
    streaming_platforms=("Netflix",),
    emoji="🎬",
)
CAR_REVIEW = SavedItemSummary(category="car review", details=("BMW M3",), emoji="🚗")
ENGLISH = reply_wording("en")
TURKISH = reply_wording("tr")
# Wording a caller (the response agent, SAV-17) supplies for a language with no
# built-in wording. Japanese stands for any such language.
JAPANESE = ReplyWording(
    saved_named="保存しました: {item}",
    saved_as="{category}として保存しました",
    in_collection="{collection}に",
    saved_to="{collection}に保存しました",
    streaming_on="{platforms}で配信中",
    which_one="どれを保存しましたか",
    none_of_these="どれでもない",
    import_progress="{done}/{total}件をインポートしました…",
    import_progress_with_counts="{done}/{total}件をインポートしました…これまでに{counts}",
    error="すみません、問題が発生しました。数分後にもう一度お試しください。",
    open_now="営業中",
    closed_now="営業時間外",
    list_separator="、",
    deletion_confirmation=(
        "保存、メディア、会話、プロフィールをすべて削除します。元に戻せません。"
        "{minutes}分以内に「はい」と返信してください。"
    ),
    delete_button="はい、削除",
    deletion_done="すべてのデータを削除しました。また書けば最初から始まります。",
)
KRONOTROP = ResultLine(
    position=1,
    total=3,
    name="Kronotrop",
    area="Beşiktaş",
    open_now=True,
    link="https://maps.example/kronotrop",
    emoji="🍽",
)


# --- language --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", ENGLISH), ("en-GB", ENGLISH), ("tr", TURKISH), ("tr-TR", TURKISH), ("TR", TURKISH)],
)
def test_built_in_wording_is_found_by_the_primary_language_subtag(
    language: str, expected: ReplyWording
) -> None:
    assert has_built_in_wording(language)
    assert reply_wording(language) == expected


@pytest.mark.parametrize("language", ["ja", "ja-JP", "de", "pt_BR"])
def test_language_without_built_in_or_supplied_wording_is_an_explicit_error(
    language: str,
) -> None:
    assert not has_built_in_wording(language)

    with pytest.raises(ReplyWordingUnavailableError) as raised:
        reply_wording(language)

    assert raised.value.language == language


def test_supplied_wording_is_used_for_a_language_without_built_in_wording() -> None:
    assert reply_wording("ja-JP", localized=JAPANESE) is JAPANESE


def test_supplied_wording_takes_precedence_over_built_in_wording() -> None:
    assert reply_wording("en", localized=JAPANESE) is JAPANESE


@pytest.mark.parametrize("language", ["", "  "])
def test_empty_language_is_refused(language: str) -> None:
    with pytest.raises(ValueError, match="language"):
        reply_wording(language)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("saved_named", "保存しました"),  # {item} missing
        ("saved_as", "{category} {name}"),  # unknown placeholder
        ("import_progress", "{done} of {total"),  # malformed
        ("error", "Failed: {item}"),  # no placeholder allowed here
        ("none_of_these", "   "),
        ("list_separator", ""),
        ("deletion_confirmation", "Reply yes to delete everything."),  # {minutes} missing
        ("deletion_done", "Deleted {minutes} ago"),  # no placeholder allowed here
        ("saved_to", "Saved"),  # {collection} missing
        ("in_collection", "in {category}"),  # unknown placeholder
    ],
)
def test_supplied_wording_with_wrong_placeholders_or_empty_text_is_refused(
    field: str, value: str
) -> None:
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(JAPANESE, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("saved_named", "Saved: {item:{unknown}}"),  # nested field: KeyError when rendered
        ("saved_named", "Saved: {item!z}"),  # invalid conversion: ValueError when rendered
        ("saved_named", "Saved: {item:.0}"),  # format spec that cuts the name to nothing
        ("saved_named", "Saved: {item!r}"),  # valid conversion that still changes the value
        ("saved_as", "Saved as {category:>40}"),  # padding
        ("streaming_on", "on {platforms[0]}"),  # index: keeps one character
        ("import_progress", "{done.real}/{total}"),  # attribute access
        ("import_progress_with_counts", "{done}/{total}… {counts!s}"),
    ],
)
def test_supplied_wording_allows_only_bare_named_placeholders(field: str, value: str) -> None:
    # Refused when the wording is built, before any message is rendered.
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(JAPANESE, **{field: value})


def test_bare_placeholders_keep_the_whole_caller_value() -> None:
    wording = dataclasses.replace(ENGLISH, saved_named="Saved → {item} ✓")
    summary = SavedItemSummary(category="movie", name="Spirited Away: 千と千尋の神隠し", year=2001)

    assert save_confirmation(summary, wording).text == (
        "Saved → Spirited Away: 千と千尋の神隠し (2001) ✓"
    )


# --- save confirmation -----------------------------------------------------------------------


def test_save_confirmation_for_a_named_item_in_english() -> None:
    message = save_confirmation(INCEPTION, ENGLISH)

    assert message.text == "Saved: Inception (2010) · Sci-fi · on Netflix 🎬"
    assert message.quick_replies == ()
    assert message.reply_to_message_id is None


def test_save_confirmation_for_a_named_item_in_turkish() -> None:
    message = save_confirmation(INCEPTION, TURKISH)

    assert message.text == "Kaydedildi: Inception (2010) · Sci-fi · Netflix platformunda 🎬"


def test_save_confirmation_without_a_name_says_the_open_category() -> None:
    assert save_confirmation(CAR_REVIEW, ENGLISH).text == "Saved as 🚗 car review · BMW M3"
    assert save_confirmation(CAR_REVIEW, TURKISH).text == "🚗 car review olarak kaydedildi · BMW M3"


def test_save_confirmation_shows_open_values_exactly_as_given() -> None:
    summary = SavedItemSummary(
        category="ラーメン店",
        details=("Kore mutfağı", "Brooklyn, NY"),
        streaming_platforms=("Disney+", "MUBI"),
    )

    message = save_confirmation(summary, ENGLISH)

    assert message.text == "Saved as ラーメン店 · Kore mutfağı · Brooklyn, NY · on Disney+, MUBI"


def test_save_confirmation_with_only_a_name() -> None:
    summary = SavedItemSummary(category="restaurant", name="Kronotrop")

    assert save_confirmation(summary, ENGLISH).text == "Saved: Kronotrop"


def test_save_confirmation_names_the_collection_the_user_gave_exactly_as_written() -> None:
    summary = dataclasses.replace(CAR_REVIEW, collection="勉強")

    assert save_confirmation(summary, ENGLISH).text == "Saved as 🚗 car review · BMW M3 · in 勉強"
    assert save_confirmation(summary, TURKISH).text == (
        "🚗 car review olarak kaydedildi · BMW M3 · 勉強 koleksiyonunda"
    )
    assert (
        save_confirmation(summary, JAPANESE).text
        == "🚗 car reviewとして保存しました · BMW M3 · 勉強に"
    )


@pytest.mark.parametrize(
    ("wording", "text"),
    [
        (ENGLISH, "Saved to for my Movies collection"),
        (TURKISH, "for my Movies collection koleksiyonuna kaydedildi"),
        (JAPANESE, "for my Movies collectionに保存しました"),
    ],
    ids=["en", "tr", "supplied"],
)
def test_collection_confirmation_is_one_short_message_in_the_users_language(
    wording: ReplyWording, text: str
) -> None:
    message = collection_confirmation("for my Movies collection", wording)

    assert message.text == text
    assert message.quick_replies == ()
    assert message.reply_to_message_id is None


# --- which one? ------------------------------------------------------------------------------


def test_which_one_question_numbers_the_options_and_adds_none_of_these() -> None:
    options = [
        ChoiceOption(label="Kronotrop Bebek", payload="choice:bebek"),
        ChoiceOption(label="Kronotrop Karaköy", payload="choice:karakoy"),
    ]

    message = which_one_question(options, none_payload="choice:none", wording=ENGLISH)

    assert message.text == "Which one did you save?\n1. Kronotrop Bebek\n2. Kronotrop Karaköy"
    assert message.quick_replies == (
        QuickReply(title="1. Kronotrop Bebek", payload="choice:bebek"),
        QuickReply(title="2. Kronotrop Karaköy", payload="choice:karakoy"),
        QuickReply(title="None of these", payload="choice:none"),
    )


def test_which_one_question_in_turkish() -> None:
    options = [ChoiceOption(label="Dune (2021)", payload="tmdb:1")]

    message = which_one_question(options, none_payload="none", wording=TURKISH)

    assert message.text == "Hangisini kaydettin?\n1. Dune (2021)"
    assert message.quick_replies[-1] == QuickReply(title="Hiçbiri", payload="none")


@pytest.mark.parametrize("count", [0, MAX_QUICK_REPLIES])
def test_which_one_question_needs_room_for_every_button(count: int) -> None:
    options = [ChoiceOption(label=f"Option {n}", payload=str(n)) for n in range(count)]

    with pytest.raises(ValueError, match="options"):
        which_one_question(options, none_payload="none", wording=ENGLISH)


def test_which_one_question_accepts_as_many_options_as_buttons_allow() -> None:
    options = [ChoiceOption(label=f"Option {n}", payload=str(n)) for n in range(12)]

    message = which_one_question(options, none_payload="none", wording=ENGLISH)

    assert len(message.quick_replies) == MAX_QUICK_REPLIES


# --- import progress -------------------------------------------------------------------------


def test_import_progress_lists_what_was_found_so_far() -> None:
    counts = [
        ImportCount(label="places", count=61),
        ImportCount(label="movies", count=40),
        ImportCount(label="study posts", count=25),
        ImportCount(label="recipes", count=0),
    ]

    message = import_progress(230, 500, counts, ENGLISH)

    assert message.text == "Imported 230/500… 61 places, 40 movies, 25 study posts so far"


def test_import_progress_in_turkish_and_without_counts() -> None:
    counts = [ImportCount(label="mekan", count=3)]

    assert import_progress(5, 500, counts, TURKISH).text == (
        "5/500 içe aktarıldı… şu ana kadar 3 mekan"
    )
    assert import_progress(0, 500, [], TURKISH).text == "0/500 içe aktarıldı…"
    assert import_progress(0, 500, [], ENGLISH).text == "Imported 0/500…"


@pytest.mark.parametrize(("done", "total"), [(-1, 10), (11, 10)])
def test_import_progress_refuses_impossible_numbers(done: int, total: int) -> None:
    with pytest.raises(ValueError, match="done"):
        import_progress(done, total, [], ENGLISH)


# --- error -----------------------------------------------------------------------------------


def test_error_message_in_english_and_turkish() -> None:
    assert error_message(ENGLISH).text.startswith("Sorry, something went wrong")
    assert error_message(TURKISH).text.startswith("Üzgünüm, bir şeyler ters gitti")


# --- result lines ----------------------------------------------------------------------------


def test_result_reply_quotes_the_saved_reel_with_one_short_line() -> None:
    message = result_reply(KRONOTROP, SAVED_MESSAGE_ID, ENGLISH)

    assert message.text == "🍽 1/3 · Kronotrop, Beşiktaş · open now · https://maps.example/kronotrop"
    assert build_message_body(RECIPIENT_ID, message) == {
        "recipient": {"id": RECIPIENT_ID},
        "message": {
            "text": "🍽 1/3 · Kronotrop, Beşiktaş · open now · https://maps.example/kronotrop"
        },
        "reply_to": {"mid": SAVED_MESSAGE_ID},
    }


def test_result_reply_in_turkish_for_a_closed_place() -> None:
    line = ResultLine(position=2, total=3, name="Kronotrop", area="Karaköy", open_now=False)

    message = result_reply(line, SAVED_MESSAGE_ID, TURKISH)

    assert message.text == "2/3 · Kronotrop, Karaköy · şu an kapalı"


def test_result_reply_without_a_message_id_is_a_plain_message_with_the_post_link() -> None:
    line = ResultLine(
        position=3,
        total=3,
        name="Inception",
        details=("Sci-fi",),
        link="https://www.instagram.com/reel/SYNTHETIC/",
        emoji="🎬",
    )

    message = result_reply(line, None, ENGLISH)

    assert message.text == "🎬 3/3 · Inception · Sci-fi · https://www.instagram.com/reel/SYNTHETIC/"
    assert "reply_to" not in build_message_body(RECIPIENT_ID, message)


@pytest.mark.parametrize("position", [0, 4])
def test_result_reply_refuses_a_position_outside_the_list(position: int) -> None:
    line = ResultLine(position=position, total=3, name="Kronotrop")

    with pytest.raises(ValueError, match="position"):
        result_reply(line, None, ENGLISH)


# --- a third language, through supplied wording ----------------------------------------------


def test_every_template_uses_supplied_wording_for_a_third_language() -> None:
    wording = reply_wording("ja", localized=JAPANESE)
    movie = SavedItemSummary(
        category="映画",
        name="千と千尋の神隠し",
        year=2001,
        streaming_platforms=("Netflix", "U-NEXT"),
    )
    ramen = SavedItemSummary(category="ラーメン店", details=("一蘭",), emoji="🍜")
    options = [
        ChoiceOption(label="一蘭 渋谷店", payload="a"),
        ChoiceOption(label="一蘭 新宿店", payload="b"),
    ]
    counts = [ImportCount(label="件のお店", count=12), ImportCount(label="本の映画", count=3)]
    line = ResultLine(position=1, total=2, name="一蘭", area="渋谷区", open_now=False)

    assert save_confirmation(movie, wording).text == (
        "保存しました: 千と千尋の神隠し (2001) · Netflix、U-NEXTで配信中"
    )
    assert save_confirmation(ramen, wording).text == "🍜 ラーメン店として保存しました · 一蘭"
    question = which_one_question(options, none_payload="none", wording=wording)
    assert question.text == "どれを保存しましたか\n1. 一蘭 渋谷店\n2. 一蘭 新宿店"
    assert [button.title for button in question.quick_replies] == [
        "1. 一蘭 渋谷店",
        "2. 一蘭 新宿店",
        "どれでもない",
    ]
    assert import_progress(40, 100, counts, wording).text == (
        "40/100件をインポートしました…これまでに12 件のお店、3 本の映画"
    )
    assert error_message(wording).text == JAPANESE.error
    assert result_reply(line, SAVED_MESSAGE_ID, wording).text == "1/2 · 一蘭、渋谷区 · 営業時間外"


@pytest.mark.parametrize("language", ["en", "tr"])
def test_deletion_confirmation_names_the_deadline_in_built_in_languages(language: str) -> None:
    message = deletion_confirmation(10, reply_wording(language))

    assert "10" in message.text
    assert message.reply_to_message_id is None
    (button,) = message.quick_replies
    assert button.title == reply_wording(language).delete_button
    assert button.payload == DELETION_CONFIRM_PAYLOAD


def test_deletion_messages_in_supplied_wording() -> None:
    assert deletion_confirmation(5, JAPANESE).text.endswith("5分以内に「はい」と返信してください。")
    assert deletion_done(JAPANESE).text == JAPANESE.deletion_done


def test_the_delete_button_has_the_same_fixed_payload_in_every_language() -> None:
    (japanese,) = deletion_confirmation(5, JAPANESE).quick_replies
    (english,) = deletion_confirmation(5, reply_wording("en")).quick_replies

    assert japanese.title == "はい、削除"
    assert japanese.payload == english.payload == DELETION_CONFIRM_PAYLOAD


@pytest.mark.parametrize(
    ("payload", "confirms"),
    [
        (DELETION_CONFIRM_PAYLOAD, True),
        (None, False),
        ("yes", False),
        ("Yes, delete", False),
        (f"{DELETION_CONFIRM_PAYLOAD} ", False),
    ],
)
def test_only_the_buttons_payload_confirms_a_deletion(payload: str | None, confirms: bool) -> None:
    assert confirms_deletion(payload) is confirms
