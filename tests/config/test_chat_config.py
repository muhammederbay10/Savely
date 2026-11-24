import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config.chat import ChatSettings

ENV_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / ".env.example"
CHAT_ENVIRONMENT = {
    "SESSION_TTL_SECONDS": "86400",
    "SESSION_HISTORY_MESSAGES": "20",
    "CHAT_CONCURRENCY": "10",
    "CHAT_MAX_TRIES": "3",
    "CHAT_RETRY_DELAY_SECONDS": "5",
    "CHAT_RUN_DEADLINE_SECONDS": "60",
    "CHAT_JOB_TIMEOUT_SECONDS": "180",
    "COLLECTION_HINT_MIN_PROBABILITY": "0.7",
}


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name, value in CHAT_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def load_from_environment_only() -> ChatSettings:
    return ChatSettings(_env_file=None)


def test_chat_settings_are_read_from_environment_variables(
    environment: pytest.MonkeyPatch,
) -> None:
    settings = load_from_environment_only()

    assert settings.session_ttl_seconds == 86400
    assert settings.session_history_messages == 20
    assert settings.chat_concurrency == 10
    assert settings.chat_max_tries == 3
    assert settings.chat_retry_delay_seconds == 5
    assert settings.chat_run_deadline_seconds == 60
    assert settings.chat_job_timeout_seconds == 180
    assert settings.collection_hint_min_probability == 0.7


@pytest.mark.parametrize("missing_variable", sorted(CHAT_ENVIRONMENT))
def test_missing_chat_variable_fails_at_load(
    environment: pytest.MonkeyPatch, missing_variable: str
) -> None:
    environment.delenv(missing_variable)

    with pytest.raises(ValidationError, match=missing_variable.lower()):
        load_from_environment_only()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("SESSION_TTL_SECONDS", "0"),
        ("SESSION_HISTORY_MESSAGES", "0"),
        ("CHAT_CONCURRENCY", "0"),
        ("CHAT_MAX_TRIES", "0"),
        ("CHAT_RETRY_DELAY_SECONDS", "0"),
        ("CHAT_RUN_DEADLINE_SECONDS", "0"),
        ("CHAT_JOB_TIMEOUT_SECONDS", "inf"),
        # No time would be left for sending.
        ("CHAT_JOB_TIMEOUT_SECONDS", "60"),
        # A probability: 0 would file every text as a collection, above 1 none.
        ("COLLECTION_HINT_MIN_PROBABILITY", "0"),
        ("COLLECTION_HINT_MIN_PROBABILITY", "1.5"),
        ("COLLECTION_HINT_MIN_PROBABILITY", "nan"),
    ],
)
def test_unusable_chat_setting_is_rejected(
    environment: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    environment.setenv(variable, value)

    with pytest.raises(ValidationError, match=variable.lower()):
        load_from_environment_only()


def test_env_example_lists_every_chat_variable_with_a_loadable_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = re.findall(r"^([A-Z_]+)=(.*)$", ENV_EXAMPLE_PATH.read_text(), flags=re.MULTILINE)
    example = dict(assignments)
    assert set(CHAT_ENVIRONMENT) <= set(example)
    for name in CHAT_ENVIRONMENT:
        monkeypatch.setenv(name, example[name])

    load_from_environment_only()
