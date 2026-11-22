"""Settings of the response agent."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Configuration of the response agent in `app/agent/` (ARCHITECTURE §3.5).

    It is separate from `Settings` so that the parts of the system that never
    answer questions (the webhook, the migrations) do not need these values.
    Like `Settings`, values come from environment variables with `.env` as a
    fallback, no field has a default, and a validation error names the field
    but never repeats its value.

    Attributes:
        agent_max_steps: The most model calls one answer may take, tool rounds
            included (`AGENT_MAX_STEPS`). The last call may not call tools, so
            the agent always ends with an answer or an explicit error.
        agent_max_results_per_answer: The most saved items one answer sends
            back as replies to the saved reels (`AGENT_MAX_RESULTS_PER_ANSWER`,
            decision record 0004). The rest are kept for "show more".
        agent_max_output_tokens: The longest reply the model may write per
            call (`AGENT_MAX_OUTPUT_TOKENS`).
        vocab_alias_review_min_users: A verified vocabulary alias is flagged
            for review once this many different users corrected a value that
            came through it (`VOCAB_ALIAS_REVIEW_MIN_USERS`, ARCHITECTURE §3.4).
            One user's repeated corrections count once.
        data_deletion_confirm_seconds: How long a "delete my data" request
            waits for the user's confirmation, counted from when sending the
            question begins (`DATA_DELETION_CONFIRM_SECONDS`).
            Only the user's next message can confirm it, and only within this time.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    agent_max_steps: int = Field(ge=2)
    agent_max_results_per_answer: int = Field(ge=1)
    agent_max_output_tokens: int = Field(ge=1)
    # "Several users", so at least two.
    vocab_alias_review_min_users: int = Field(ge=2)
    data_deletion_confirm_seconds: int = Field(ge=1)


@lru_cache
def get_agent_settings() -> AgentSettings:
    """Return the process-wide agent settings, read once from the environment.

    Returns:
        The cached `AgentSettings` instance.

    Raises:
        pydantic.ValidationError: If a required variable is missing or invalid.
    """
    return AgentSettings()
