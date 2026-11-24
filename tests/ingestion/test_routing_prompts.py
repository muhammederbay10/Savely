"""The wording text routing sends to the decision model is checked when it is built."""

from dataclasses import replace

import pytest

from app.ingestion.prompts import ROUTING_PROMPTS


def test_the_routing_prompts_fill_in_the_users_text_as_a_value() -> None:
    text = "for my {Movies} collection"

    assert ROUTING_PROMPTS.collection_name_input.format(text=text).endswith(text)


@pytest.mark.parametrize(
    ("field", "template"),
    [
        ("collection_name_input", "The message"),  # {text} missing
        ("collection_name_input", "{text} {item}"),  # unknown placeholder
        ("collection_name_input", "{text!r}"),  # conversion
        ("collection_name_question", "Is {text} a collection?"),  # no placeholder allowed
        ("collection_name_question", " "),
        ("version", ""),
    ],
)
def test_prompts_with_wrong_placeholders_or_blank_wording_are_rejected(
    field: str, template: str
) -> None:
    with pytest.raises(ValueError, match=f"prompt {field}"):
        replace(ROUTING_PROMPTS, **{field: template})
