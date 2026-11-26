"""The API's settings, and the constants and types the settings groups share."""

from functools import lru_cache
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    StringConstraints,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# pgvector cannot build an HNSW index on a `vector` column with more dimensions.
MAX_EMBEDDING_DIMENSION = 2000

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Settings(BaseSettings):
    """Runtime configuration for the API.

    Values come from environment variables. A local `.env` file is read as a
    fallback, and real environment variables take priority over it. No field
    has a default: a missing variable fails at startup instead of running with
    a guessed value.

    Validation errors name the field that is wrong but never repeat its value,
    because a rejected URL can contain a password.

    Attributes:
        log_level: Minimum severity that is written to the log (`LOG_LEVEL`).
        database_url: Postgres connection URL (`DATABASE_URL`).
        redis_url: Redis connection URL (`REDIS_URL`).
        embedding_dimension: Length of the embedding vectors stored in
            `saved_item.embedding` and `vocabulary.embedding`
            (`EMBEDDING_DIMENSION`). The first migration creates both columns
            with this size, so it must match the embedding model.
        meta_app_secret: The Meta app's secret (`META_APP_SECRET`). Every
            webhook body is signed with it (`X-Hub-Signature-256`).
        meta_webhook_verify_token: The token Meta repeats in the webhook
            verification handshake (`META_WEBHOOK_VERIFY_TOKEN`). We choose it
            and enter the same value in the Meta app dashboard.
        collection_hint_window_seconds: How long after a share a text from the
            same user may still name that item's collection
            (`COLLECTION_HINT_WINDOW_SECONDS`). Only an upper bound: the chat
            worker asks the decision model about a text inside it, and a
            later text is chat without a model call (decision record 0018).
        ingestion_state_ttl_seconds: How long ingestion keeps a user's job
            payloads, text receipts and recent shares in Redis after its last
            write (`INGESTION_STATE_TTL_SECONDS`). It must cover the time Meta
            keeps delivering a failed webhook again, so a late delivery is
            still recognized, and it must be at least the hint window.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    log_level: LogLevel
    database_url: PostgresDsn
    redis_url: RedisDsn
    embedding_dimension: int = Field(gt=0, le=MAX_EMBEDDING_DIMENSION)
    meta_app_secret: SecretStr = Field(min_length=1)
    meta_webhook_verify_token: SecretStr = Field(min_length=1)
    collection_hint_window_seconds: int = Field(gt=0)
    ingestion_state_ttl_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def _state_outlives_the_hint_window(self) -> Self:
        if self.ingestion_state_ttl_seconds < self.collection_hint_window_seconds:
            raise ValueError(
                "ingestion_state_ttl_seconds must be at least collection_hint_window_seconds"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read once from the environment.

    Returns:
        The cached `Settings` instance.

    Raises:
        pydantic.ValidationError: If a required variable is missing or invalid.
    """
    return Settings()
