"""
Centralized application configuration.

WHY THIS FILE EXISTS
---------------------
Before this change, environment variables were read with scattered,
independent os.getenv() calls across ~8 different files - each with its
own (sometimes inconsistent) default value, and no validation. A typo in
an env var name would silently fall back to a default instead of failing
loudly, and there was no single place to see every configurable setting.

This file fixes that: ONE Settings object, validated at import time,
that every other module should import from instead of calling
os.getenv() directly.

USAGE
-----
    from rag_system.config import settings

    print(settings.ollama_host)
    print(settings.llm_backend)

Environment variables are still how you configure the app (e.g. in
Docker, or a .env file) - this doesn't change how you set values, only
how the code reads them: once, validated, in one place.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unrelated env vars in .env shouldn't cause errors
    )

    # --- LLM backend selection ---
    llm_backend: Literal["ollama", "watsonx"] = Field(
        default="ollama",
        validation_alias="LLM_BACKEND",
        description="Which LLM backend to use for generation.",
    )

    # --- Ollama (local LLM server) ---
    ollama_host: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_HOST",
    )

    # --- IBM Watson X (optional alternative backend) ---
    watsonx_api_key: str = Field(default="", validation_alias="WATSONX_API_KEY")
    watsonx_project_id: str = Field(default="", validation_alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        validation_alias="WATSONX_URL",
    )
    watsonx_generation_model: str = Field(
        default="ibm/granite-13b-chat-v2",
        validation_alias="WATSONX_GENERATION_MODEL",
    )
    watsonx_enrichment_model: str = Field(
        default="ibm/granite-8b-japanese",
        validation_alias="WATSONX_ENRICHMENT_MODEL",
    )

    # --- Storage paths ---
    lancedb_path: str = Field(
        default="./rag_system/index_store/lancedb",
        validation_alias="LANCEDB_PATH",
    )
    chat_db_path: str = Field(
        default="chat_data.db",
        validation_alias="CHAT_DB_PATH",
    )

    # --- Redis (optional - powers the distributed semantic cache; falls
    #     back to an in-process cache if unset or unreachable, so the app
    #     still works out of the box without Redis) ---
    redis_url: str = Field(default="", validation_alias="REDIS_URL")

    # --- Auth (Improvement #10) ---
    # SECURITY: the default below is fine for local development only.
    # Anyone deploying this beyond their own machine MUST set a real,
    # randomly-generated JWT_SECRET_KEY (e.g. `openssl rand -hex 32`) -
    # using the default in a shared/production environment would let
    # anyone forge valid auth tokens.
    jwt_secret_key: str = Field(
        default="dev-only-insecure-secret-change-me",
        validation_alias="JWT_SECRET_KEY",
    )
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_expiry_minutes: int = Field(default=60 * 24 * 7, validation_alias="JWT_EXPIRY_MINUTES")  # 7 days

    # --- HuggingFace auth (optional, for gated models) ---
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")

    # --- Server ports ---
    backend_port: int = Field(default=8000, validation_alias="BACKEND_PORT")
    rag_api_port: int = Field(default=8001, validation_alias="RAG_API_PORT")

    # --- Pipeline mode ---
    rag_config_mode: str = Field(default="default", validation_alias="RAG_CONFIG_MODE")

    # --- Logging (kept here for discoverability; logging_utils.py reads
    #     the same env vars directly since it must be configurable before
    #     Settings can be safely imported in every case) ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", validation_alias="LOG_LEVEL"
    )
    log_format: Literal["text", "json"] = Field(default="text", validation_alias="LOG_FORMAT")

    @model_validator(mode="after")
    def _validate_watsonx_credentials(self) -> "Settings":
        """Fail fast, at startup, if watsonx is selected but not configured -
        instead of failing confusingly on the first chat request."""
        if self.llm_backend == "watsonx":
            missing = [
                name
                for name, value in [
                    ("WATSONX_API_KEY", self.watsonx_api_key),
                    ("WATSONX_PROJECT_ID", self.watsonx_project_id),
                ]
                if not value
            ]
            if missing:
                raise ValueError(
                    f"LLM_BACKEND=watsonx requires the following environment "
                    f"variable(s) to be set: {', '.join(missing)}"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Cached factory - Settings is only constructed (and validated) once per
    process, then reused. Using a function instead of a bare module-level
    instance also makes it trivial to override in tests via monkeypatching
    rag_system.config.get_settings.
    """
    return Settings()


# Most call sites just want the singleton - this is the import used
# throughout the app: `from rag_system.config import settings`
settings = get_settings()
