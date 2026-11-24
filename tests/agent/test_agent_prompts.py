from dataclasses import replace

import pytest

from app.agent.agent import AnswerOutcome
from app.agent.prompts import AGENT_PROMPTS


@pytest.mark.parametrize(
    ("field", "template"),
    [
        ("system", "Answer from the user's saves."),
        ("system", "Show {max_results} of {items}."),
        ("working_set", "Last results: {max_results}"),
        ("answer_too_long", "Shorter, please."),
        ("invalid_answer", "Use {language}."),
        ("working_set", "{items!r}"),
        ("wording_request", " "),
        ("version", ""),
    ],
)
def test_prompts_with_wrong_placeholders_or_blank_wording_are_rejected(
    field: str, template: str
) -> None:
    with pytest.raises(ValueError, match=f"prompt {field}"):
        replace(AGENT_PROMPTS, **{field: template})


def test_the_system_prompt_names_the_result_cap_and_every_notice() -> None:
    system = AGENT_PROMPTS.system.format(max_results=5)

    assert "The first 5 are sent as replies" in system
    for notice in ("backup_mode", "broader_results", "may_be_incomplete"):
        assert notice in system


def test_the_system_prompt_names_every_answer_outcome() -> None:
    system = AGENT_PROMPTS.system.format(max_results=5)

    for outcome in AnswerOutcome:
        assert f"'{outcome.value}'" in system
