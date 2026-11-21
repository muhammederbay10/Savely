"""Settings of the chat worker and its short-term memory."""

from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    """Configuration of the chat worker (`app/agent/worker.py`) and short-term memory.

    It is separate from `Settings` so that the parts of the system that never
    answer a DM (the webhook, the migrations) do not need these values. Like
    `Settings`, values come from environment variables with `.env` as a
    fallback, no field has a default, and a validation error names the field
    but never repeats its value.

    Attributes:
        session_ttl_seconds: How long a user's short-term memory (history and
            working set) lives after the last exchange (`SESSION_TTL_SECONDS`,
            about 24 hours, like Instagram's reply window; ARCHITECTURE §3.7).
        session_history_messages: How many of the last messages the response
            agent reads (`SESSION_HISTORY_MESSAGES`, about 20).
        chat_concurrency: How many messages one chat worker answers at once
            (`CHAT_CONCURRENCY`).
        chat_max_tries: How often one message's job runs before the bot sends
            the error message instead (`CHAT_MAX_TRIES`).
        chat_retry_delay_seconds: The wait before a failed job runs again,
            times the number of the try (`CHAT_RETRY_DELAY_SECONDS`).
        chat_run_deadline_seconds: How long one try may take to prepare an
            answer, before anything is sent (`CHAT_RUN_DEADLINE_SECONDS`).
            Sending is never cut off: a message cut off mid-way may have
            been delivered.
        chat_job_timeout_seconds: arq's own limit on one try
            (`CHAT_JOB_TIMEOUT_SECONDS`). It must be longer than the run
            deadline, to leave time for sending; it is only a backstop.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    session_ttl_seconds: int = Field(gt=0)
    session_history_messages: int = Field(ge=1)
    chat_concurrency: int = Field(ge=1)
    chat_max_tries: int = Field(ge=1)
    chat_retry_delay_seconds: float = Field(gt=0, allow_inf_nan=False)
    chat_run_deadline_seconds: float = Field(gt=0, allow_inf_nan=False)
    chat_job_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _job_timeout_leaves_time_to_send(self) -> Self:
        if self.chat_job_timeout_seconds <= self.chat_run_deadline_seconds:
            raise ValueError(
                "chat_job_timeout_seconds must be longer than chat_run_deadline_seconds"
            )
        return self


@lru_cache
def get_chat_settings() -> ChatSettings:
    """Return the process-wide chat settings, read once from the environment.

    Returns:
        The cached `ChatSettings` instance.

    Raises:
        pydantic.ValidationError: If a required variable is missing or invalid.
    """
    return ChatSettings()
