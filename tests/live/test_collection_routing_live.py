"""Opt-in check: on the real decision model, a text right after a share is routed right (SAV-39).

SAV-39 found that "What did i save?" sent 20 seconds after a reel was filed as
that reel's collection, because the route was decided by time alone. Now the
decision model decides (decision record 0018). The unit tests use a fake
model, so they can't tell whether the real one tells collection names from
questions. This check asks the configured decision model the production
question (`ROUTING_PROMPTS`) about 28 texts in four languages, half
collection names and half questions or small talk, and compares its
probability with the configured `COLLECTION_HINT_MIN_PROBABILITY`. The set
includes short phrases that ask about saved things without a question mark
("restaurants in Kadıköy"), which the first live check stored as
collections, and short list names ("Recipes to try"). The Spanish and
Japanese pairs of that kind are not quoted in the prompt, so they show
whether the wording generalizes.

What is real: the decision model, the wording and the threshold, from the
same settings as production. Nothing else runs: no database, no chat worker.

It is skipped unless `LIVE_MODEL_TESTS=1` and `OPENROUTER_API_KEY` are set,
and always skipped when `CI` is set: the module's skip mark is decided at
import, before any fixture builds a model client
(`tests/live/test_collection_routing_live_gate.py` proves it). It never runs
in CI or in an agent's worktree; the owner runs it locally before a push
(SAV-22, 24 Nov 2026 update). `docs/development.md` ("Live routing check")
shows how and how to read the result. A run costs a fraction of a cent.
"""

import logging
import os
from collections.abc import AsyncIterator, Mapping

import pytest

from app.config.chat import ChatSettings
from app.config.models import ModelSettings
from app.ingestion.prompts import ROUTING_PROMPTS
from app.ingestion.text_routing import collection_name_probability
from app.models.factory import ModelClients, build_model_clients, open_http_client

logger = logging.getLogger(__name__)

OPT_IN_VARIABLE = "LIVE_MODEL_TESTS"
API_KEY_VARIABLE = "OPENROUTER_API_KEY"
# Set by GitHub Actions and most CI services. Any value but these counts as CI.
CI_VARIABLE = "CI"
NOT_CI_VALUES = frozenset({"", "0", "false"})


def live_run_blocker(environ: Mapping[str, str]) -> str | None:
    """Return why the live routing check may not run in this environment, or `None` if it may."""
    if environ.get(CI_VARIABLE, "").strip().casefold() not in NOT_CI_VALUES:
        return f"the live routing check never runs in CI ({CI_VARIABLE} is set)"
    if environ.get(OPT_IN_VARIABLE) != "1" or not environ.get(API_KEY_VARIABLE):
        return f"live model tests need {OPT_IN_VARIABLE}=1 and {API_KEY_VARIABLE}"
    return None


_BLOCKER = live_run_blocker(os.environ)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_BLOCKER is not None, reason=_BLOCKER or ""),
]

COLLECTION = "collection"
QUESTION = "question"
# (language, expected route, text). Synthetic texts; half collection names, half
# questions or small talk, as a user writes them right after sharing a post.
CASES = [
    ("en", COLLECTION, "Food"),
    ("en", COLLECTION, "for my Movies collection"),
    ("en", COLLECTION, "Recipes to try"),
    ("en", QUESTION, "What did i save?"),
    ("en", QUESTION, "restaurants in Kadıköy"),
    ("en", QUESTION, "is this place open now?"),
    ("tr", COLLECTION, "Filmler"),
    ("tr", COLLECTION, "Yemek tarifleri"),
    ("tr", COLLECTION, "tatil listem için"),
    ("tr", QUESTION, "kaydettiğim tatlıcılar"),
    ("tr", QUESTION, "netflixte ne izleyebilirim"),
    ("tr", QUESTION, "bu nerede?"),
    ("es", COLLECTION, "Recetas"),
    ("es", COLLECTION, "para mi colección de viajes"),
    ("es", COLLECTION, "Películas"),
    ("es", QUESTION, "¿qué restaurantes guardé?"),
    ("es", QUESTION, "¿dónde está este lugar?"),
    ("es", QUESTION, "hola"),
    ("es", COLLECTION, "Cenas fáciles"),
    ("es", QUESTION, "cafeterías en Malasaña"),
    ("ja", COLLECTION, "勉強"),
    ("ja", COLLECTION, "旅行"),
    ("ja", COLLECTION, "映画リスト"),
    # "\uff1f" is the full-width question mark Japanese is typed with.
    ("ja", QUESTION, "保存した映画は\uff1f"),
    ("ja", QUESTION, "これはどこ\uff1f"),
    ("ja", QUESTION, "ありがとう"),
    ("ja", COLLECTION, "行きたいカフェ"),
    ("ja", QUESTION, "保存したラーメン屋"),
]


@pytest.fixture
async def clients() -> AsyncIterator[ModelClients]:
    settings = ModelSettings(_env_file=None)
    async with open_http_client(settings) as http_client:
        yield build_model_clients(settings, http_client)


@pytest.fixture
def min_probability() -> float:
    return ChatSettings(_env_file=None).collection_hint_min_probability


@pytest.mark.parametrize(("language", "expected", "text"), CASES)
async def test_the_decision_model_routes_a_text_right_after_a_share(
    clients: ModelClients, min_probability: float, language: str, expected: str, text: str
) -> None:
    probability = await collection_name_probability(clients.classifier, ROUTING_PROMPTS, text)

    names_collection = probability is not None and probability >= min_probability
    routed = COLLECTION if names_collection else QUESTION
    outcome = (
        f"language={language} expected={expected} routed={routed} probability={probability} "
        f"threshold={min_probability} prompts={ROUTING_PROMPTS.version}"
    )
    logger.info("routing_case %s", outcome)
    assert routed == expected, f"wrong route: {outcome}"
