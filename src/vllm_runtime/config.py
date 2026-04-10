"""Configuration models for the HTTP vLLM runtime."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

VLLMServerMode = Literal["managed", "external"]


class VLLMServerConfig(BaseModel):
    """Server lifecycle and connectivity configuration."""

    mode: VLLMServerMode = Field(
        default="managed",
        description="Whether the runtime manages a local server process or uses an external base URL.",
    )
    model_name: str = Field(description="vLLM model identifier served by the API.")
    base_url: str | None = Field(
        default=None,
        description="External OpenAI-compatible base URL (used for external mode).",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host bound by managed server mode.",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port bound by managed server mode.",
    )
    api_key_env_var: str = Field(
        default="VLLM_API_KEY",
        description="Environment variable containing API key for OpenAI-compatible auth.",
    )
    startup_timeout_seconds: int = Field(
        default=240,
        ge=10,
        le=3600,
        description="How long to wait for server readiness.",
    )
    request_timeout_seconds: int = Field(
        default=120,
        ge=5,
        le=3600,
        description="Per-request timeout for HTTP calls.",
    )
    allow_remote_warmup: bool = Field(
        default=False,
        description="Allow polling external servers until they become healthy.",
    )
    healthcheck_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Polling interval for startup/readiness checks.",
    )
    extra_server_args: list[str] = Field(
        default_factory=list,
        description="Additional CLI args forwarded to managed vLLM api_server.",
    )

    @property
    def resolved_base_url(self) -> str:
        """Return normalized OpenAI-compatible base URL.

        The runtime uses this value as the ownership boundary for managed-process
        lifecycle and startup locking.
        """
        if self.base_url:
            normalized = self.base_url.rstrip("/")
        else:
            normalized = f"http://{self.host}:{self.port}"
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    @property
    def healthcheck_url(self) -> str:
        """Return normalized healthcheck endpoint base URL."""
        base = self.resolved_base_url
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/health"

    @property
    def api_key(self) -> str | None:
        """Load API key from configured environment variable."""
        return os.getenv(self.api_key_env_var)


class VLLMModelConfig(BaseModel):
    """Generation and embedding model request parameters."""

    model_name: str = Field(
        description="Model identifier for OpenAI-compatible requests."
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32768,
        description="Max completion tokens for chat responses.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling parameter.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=1024,
        description="Optional top-k sampling parameter.",
    )
    min_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional min-p sampling parameter.",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Optional presence penalty for repetition control.",
    )
    repetition_penalty: float | None = Field(
        default=None,
        ge=0.1,
        le=2.0,
        description="Optional repetition penalty for vLLM sampling.",
    )


class VLLMRuntimeConfig(BaseModel):
    """Runtime transport and observability behavior."""

    max_concurrent_requests: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Maximum concurrent HTTP requests dispatched by the runtime.",
    )
    transport_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Retry attempts after transport failures.",
    )
    request_preview_chars: int = Field(
        default=300,
        ge=50,
        le=5000,
        description="Max characters logged for request previews.",
    )
    response_preview_chars: int = Field(
        default=300,
        ge=50,
        le=5000,
        description="Max characters logged for response previews.",
    )
