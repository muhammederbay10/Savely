"""The live routing check never runs in CI or without the opt-in and a key (SAV-39).

Offline: the live module is run in this pytest process with `pytester`, with
building model clients and sending HTTP replaced by stubs that record any use.
Nothing here calls a model or any other external service, and the key is
synthetic.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

import app.models.factory
from tests.live.test_collection_routing_live import CASES, COLLECTION, live_run_blocker

pytest_plugins = ["pytester"]

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_MODULE = REPO_ROOT / "tests" / "live" / "test_collection_routing_live.py"
LIVE_MODULE_NAME = "tests.live.test_collection_routing_live"
SYNTHETIC_KEY = "synthetic-openrouter-key-0000"
MIN_CASES = 20
MIN_LANGUAGES = 3


class LiveClientUsedError(AssertionError):
    """A live model client or an HTTP request was used while the gate should hold."""


@pytest.fixture
def live_use(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Make building model clients and sending HTTP record the attempt and fail."""
    used: list[str] = []

    def refuse(name: str) -> object:
        def refused(*_args: object, **_kwargs: object) -> object:
            used.append(name)
            raise LiveClientUsedError(f"{name} was called")

        return refused

    monkeypatch.setattr(app.models.factory, "build_model_clients", refuse("build_model_clients"))
    monkeypatch.setattr(app.models.factory, "open_http_client", refuse("open_http_client"))
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse("httpx.AsyncClient.send"))
    # Import the live module afresh, so its gate reads this test's environment.
    monkeypatch.delitem(sys.modules, LIVE_MODULE_NAME, raising=False)
    yield used


def run_live_module(pytester: pytest.Pytester) -> pytest.RunResult:
    return pytester.runpytest_inprocess(
        str(LIVE_MODULE),
        "-c",
        str(REPO_ROOT / "pyproject.toml"),
        "--rootdir",
        str(REPO_ROOT),
        "-p",
        "no:cacheprovider",
    )


def opt_in_with_synthetic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_MODEL_TESTS", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", SYNTHETIC_KEY)


def test_ci_skips_every_case_before_any_client_or_request(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, live_use: list[str]
) -> None:
    opt_in_with_synthetic_key(monkeypatch)
    monkeypatch.setenv("CI", "true")

    result = run_live_module(pytester)

    result.assert_outcomes(skipped=len(CASES))
    result.stdout.fnmatch_lines(["*never runs in CI*"])
    assert live_use == []


def test_without_a_key_every_case_is_skipped_before_any_client_or_request(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, live_use: list[str]
) -> None:
    monkeypatch.setenv("LIVE_MODEL_TESTS", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CI", raising=False)

    result = run_live_module(pytester)

    result.assert_outcomes(skipped=len(CASES))
    assert live_use == []


def test_without_ci_the_same_settings_pass_the_gate(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, live_use: list[str]
) -> None:
    """The control: only `CI` differs, and no case is skipped; the refused client stops them."""
    opt_in_with_synthetic_key(monkeypatch)
    monkeypatch.delenv("CI", raising=False)

    result = run_live_module(pytester)

    assert result.parseoutcomes().get("skipped", 0) == 0
    assert result.ret != pytest.ExitCode.OK
    assert "httpx.AsyncClient.send" not in live_use


@pytest.mark.parametrize("ci", ["true", "1", "TRUE", "yes"])
def test_any_ci_value_blocks_the_live_check(ci: str) -> None:
    environ = {"CI": ci, "LIVE_MODEL_TESTS": "1", "OPENROUTER_API_KEY": SYNTHETIC_KEY}

    assert live_run_blocker(environ) is not None


@pytest.mark.parametrize(
    "environ",
    [
        {"OPENROUTER_API_KEY": SYNTHETIC_KEY},
        {"LIVE_MODEL_TESTS": "0", "OPENROUTER_API_KEY": SYNTHETIC_KEY},
        {"LIVE_MODEL_TESTS": "1"},
        {"LIVE_MODEL_TESTS": "1", "OPENROUTER_API_KEY": ""},
    ],
)
def test_without_the_opt_in_and_a_key_the_live_check_is_blocked(environ: dict[str, str]) -> None:
    assert live_run_blocker(environ) is not None


@pytest.mark.parametrize("ci", [None, "", "0", "false", "False"])
def test_a_local_opt_in_with_a_key_may_run(ci: str | None) -> None:
    environ = {"LIVE_MODEL_TESTS": "1", "OPENROUTER_API_KEY": SYNTHETIC_KEY}
    if ci is not None:
        environ["CI"] = ci

    assert live_run_blocker(environ) is None


def test_the_cases_are_half_collection_names_in_enough_languages() -> None:
    names = [case for case in CASES if case[1] == COLLECTION]

    assert len(CASES) >= MIN_CASES
    assert len({language for language, _, _ in CASES}) >= MIN_LANGUAGES
    assert len(names) * 2 == len(CASES)
