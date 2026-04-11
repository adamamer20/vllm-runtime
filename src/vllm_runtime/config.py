"""Configuration models for HTTP runtime families (vLLM + llama.cpp)."""

from __future__ import annotations

import os
import shutil
from typing import Literal

from pydantic import BaseModel, Field, validator

VLLMServerMode = Literal["managed", "external"]
LlamaCppServerMode = Literal["managed", "external"]


def _default_llama_server_binary() -> str:
    """Resolve the most sensible llama-server binary for managed runtime use."""
    explicit = (
        os.getenv("LLAMA_CPP_SERVER_BINARY")
        or os.getenv("LLAMA_CPP_CHAT_SERVER_BINARY")
        or os.getenv("LLAMA_CPP_EMBEDDING_SERVER_BINARY")
    )
    candidates = [
        explicit,
        os.path.expanduser("~/src/llama.cpp/build-cuda/bin/llama-server"),
        os.path.expanduser("~/.local/bin/llama-server"),
        shutil.which("llama-server"),
        "/home/linuxbrew/.linuxbrew/bin/llama-server",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "/home/linuxbrew/.linuxbrew/bin/llama-server"


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


class LlamaCppServerConfig(BaseModel):
    """Server lifecycle and connectivity configuration for llama.cpp."""

    mode: LlamaCppServerMode = Field(
        default="managed",
        description="Whether the runtime manages a local llama-server process or uses an external base URL.",
    )
    model_name: str = Field(
        description="Model identifier sent in OpenAI-compatible request payloads."
    )
    base_url: str | None = Field(
        default=None,
        description="External OpenAI-compatible base URL (used for external mode).",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host bound by managed llama-server mode.",
    )
    port: int = Field(
        default=8010,
        ge=1,
        le=65535,
        description="Port bound by managed llama-server mode.",
    )
    api_key_env_var: str = Field(
        default="LLAMA_CPP_API_KEY",
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
    model_path: str | None = Field(
        default=None,
        description="Managed-mode local GGUF model path passed to --model.",
    )
    hf_repo: str | None = Field(
        default=None,
        description="Managed-mode Hugging Face repository passed to --hf-repo.",
    )
    hf_file: str | None = Field(
        default=None,
        description="Optional Hugging Face GGUF filename passed to --hf-file.",
    )
    server_binary: str = Field(
        default_factory=_default_llama_server_binary,
        description="Managed-mode llama-server executable path.",
    )
    ctx_size: int | None = Field(
        default=None,
        ge=256,
        le=1048576,
        description="Optional llama-server context size (--ctx-size).",
    )
    parallel: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description="Optional llama-server parallel sequence count (--parallel).",
    )
    gpu_layers: int | None = Field(
        default=None,
        ge=-1,
        le=10000,
        description="Optional llama-server GPU layer count (--gpu-layers).",
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        le=1048576,
        description="Optional llama-server batch size (--batch-size).",
    )
    ubatch_size: int | None = Field(
        default=None,
        ge=1,
        le=1048576,
        description="Optional llama-server micro-batch size (--ubatch-size).",
    )
    flash_attn: bool = Field(
        default=False,
        description="Enable llama-server flash attention (--flash-attn).",
    )
    embedding: bool = Field(
        default=False,
        description="Enable embedding mode for managed llama-server (--embedding).",
    )
    pooling: str | None = Field(
        default=None,
        description="Optional llama-server embedding pooling mode (--pooling).",
    )
    extra_server_args: list[str] = Field(
        default_factory=list,
        description="Additional CLI args forwarded to managed llama-server.",
    )

    @validator("hf_repo", always=True)
    def validate_managed_model_source(
        cls, hf_repo: str | None, values: dict[str, object]
    ) -> str | None:
        """Validate managed mode model-source requirements."""
        mode = values.get("mode")
        model_path = values.get("model_path")
        if mode == "managed":
            has_model_path = bool(model_path)
            has_hf_repo = bool(hf_repo)
            if has_model_path == has_hf_repo:
                raise ValueError(
                    "Managed llama.cpp mode requires exactly one of model_path or hf_repo."
                )
        return hf_repo

    @validator("hf_file")
    def validate_hf_file_requires_hf_repo(
        cls, hf_file: str | None, values: dict[str, object]
    ) -> str | None:
        """Require hf_file to be paired with hf_repo."""
        if hf_file and not values.get("hf_repo"):
            raise ValueError("hf_file requires hf_repo to be set.")
        return hf_file

    @property
    def resolved_base_url(self) -> str:
        """Return normalized OpenAI-compatible base URL."""
        if self.base_url:
            normalized = self.base_url.rstrip("/")
        else:
            normalized = f"http://{self.host}:{self.port}"
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    @property
    def healthcheck_url(self) -> str:
        """Return llama.cpp health endpoint URL."""
        return f"{self.resolved_base_url}/health"

    @property
    def api_key(self) -> str | None:
        """Load API key from configured environment variable."""
        return os.getenv(self.api_key_env_var)


class LlamaCppModelConfig(BaseModel):
    """Generation and embedding request parameters for llama.cpp runtime."""

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
        description="Optional repetition penalty for sampling.",
    )


class LlamaCppRuntimeConfig(BaseModel):
    """Runtime transport and observability behavior for llama.cpp runtime."""

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
