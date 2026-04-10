"""HTTP-first vLLM runtime implementations for chat and embeddings."""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import re
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx

from vllm_runtime.config import VLLMModelConfig, VLLMRuntimeConfig, VLLMServerConfig

logger = logging.getLogger(__name__)

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_META_PREFIXES = (
    "the user wants",
    "the user asked",
    "the user is asking",
    "i should",
    "assistant note",
    "analysis:",
)
_SERVER_LOCKS: dict[str, asyncio.Lock] = {}
HTTP_OK = 200


def _get_server_lock(server_key: str) -> asyncio.Lock:
    lock = _SERVER_LOCKS.get(server_key)
    if lock is None:
        lock = asyncio.Lock()
        _SERVER_LOCKS[server_key] = lock
    return lock


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_PATTERN.sub("", text).strip()


def _sanitize_plain_content(text: str) -> tuple[str, bool]:
    removed_meta = False
    filtered: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped.lower().startswith(_META_PREFIXES):
            removed_meta = True
            continue
        filtered.append(line)

    sanitized = "\n".join(filtered).strip()
    if not sanitized:
        raise RuntimeError("vLLM returned no usable content after sanitation.")
    return sanitized, removed_meta


def _extract_message_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("vLLM response did not include choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_text = block.get("text")
            if isinstance(block_text, str):
                text_parts.append(block_text)
        return "\n".join(text_parts)
    return str(content)


class _ServerController:
    """Manage managed/external vLLM server lifecycle and readiness."""

    def __init__(self, config: VLLMServerConfig):
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._owns_process = False
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._recent_logs: deque[str] = deque(maxlen=80)

    async def ensure_ready(self) -> None:
        server_key = self._config.resolved_base_url
        async with _get_server_lock(server_key):
            if await self._is_healthy():
                return

            if self._config.mode == "external":
                await self._wait_for_health(
                    allow_warmup=self._config.allow_remote_warmup
                )
                return

            await self._start_managed_server()
            await self._wait_for_health(allow_warmup=True)

    async def restart_after_transport_failure(self) -> None:
        if self._config.mode != "managed":
            raise RuntimeError(
                "vLLM external runtime is unavailable after transport failure."
            )

        logger.warning("Restarting managed vLLM server after transport failure.")
        await self.shutdown()
        await self.ensure_ready()

    async def shutdown(self) -> None:
        if not self._owns_process or self._process is None:
            return

        if self._process.returncode is not None:
            self._process = None
            self._owns_process = False
            return

        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=10)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()
        finally:
            await self._stop_log_tasks()
            self._process = None
            self._owns_process = False

    async def _start_managed_server(self) -> None:
        if await self._is_healthy():
            logger.info(
                "Reusing already-running vLLM server at %s.",
                self._config.resolved_base_url,
            )
            self._owns_process = False
            return

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._config.model_name,
            "--host",
            self._config.host,
            "--port",
            str(self._config.port),
            *self._config.extra_server_args,
        ]
        if self._config.api_key:
            command.extend(["--api-key", self._config.api_key])

        logger.info(
            "Starting managed vLLM server on %s (model=%s).",
            self._config.resolved_base_url,
            self._config.model_name,
        )
        logger.info("Managed vLLM server command: %s", " ".join(command))
        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._owns_process = True
        self._recent_logs.clear()
        self._stdout_task = self._spawn_log_reader(self._process.stdout, "stdout")
        self._stderr_task = self._spawn_log_reader(self._process.stderr, "stderr")

    async def _wait_for_health(self, *, allow_warmup: bool) -> None:
        deadline = perf_counter() + self._config.startup_timeout_seconds
        while perf_counter() < deadline:
            if await self._is_healthy():
                return

            if (
                self._config.mode == "managed"
                and self._process is not None
                and self._process.returncode is not None
            ):
                raise RuntimeError(
                    "Managed vLLM server exited before becoming ready."
                    f"{self._format_recent_logs()}"
                )

            await asyncio.sleep(self._config.healthcheck_interval_seconds)

        if self._config.mode == "external" and not allow_warmup:
            raise RuntimeError(
                f"External vLLM runtime at {self._config.resolved_base_url} is unhealthy."
            )
        raise RuntimeError(
            f"Timed out waiting for vLLM runtime at {self._config.resolved_base_url}."
            f"{self._format_recent_logs() if self._config.mode == 'managed' else ''}"
        )

    async def _is_healthy(self) -> bool:
        timeout = min(10.0, float(self._config.request_timeout_seconds))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self._config.healthcheck_url)
            return response.status_code == HTTP_OK
        except Exception:
            return False

    def _spawn_log_reader(
        self,
        stream: asyncio.StreamReader | None,
        label: str,
    ) -> asyncio.Task[None] | None:
        if stream is None:
            return None
        return asyncio.create_task(self._read_stream(stream, label))

    async def _read_stream(
        self,
        stream: asyncio.StreamReader,
        label: str,
    ) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            entry = f"[{label}] {text}"
            self._recent_logs.append(entry)
            logger.info("managed-vllm %s %s", self._config.port, entry)

    async def _stop_log_tasks(self) -> None:
        tasks = [task for task in (self._stdout_task, self._stderr_task) if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._stdout_task = None
        self._stderr_task = None

    def _format_recent_logs(self) -> str:
        if not self._recent_logs:
            return ""
        return "\nRecent managed vLLM logs:\n" + "\n".join(self._recent_logs)


@dataclass(slots=True)
class VLLMChatResult:
    """Structured chat response from runtime."""

    content: str
    usage: dict[str, Any] | None
    model: str | None
    server_mode: str
    raw_response: dict[str, Any]


class _BaseRuntime:
    def __init__(
        self,
        server_config: VLLMServerConfig,
        runtime_config: VLLMRuntimeConfig | None = None,
    ):
        self.server_config = server_config
        self.runtime_config = runtime_config or VLLMRuntimeConfig()
        self._server = _ServerController(server_config)
        self._client = httpx.AsyncClient(timeout=server_config.request_timeout_seconds)
        self._semaphore = asyncio.Semaphore(self.runtime_config.max_concurrent_requests)

    async def close(self) -> None:
        await self._client.aclose()
        await self._server.shutdown()

    async def ensure_ready(self) -> None:
        """Block until runtime endpoint is healthy."""
        await self._server.ensure_ready()

    def _request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        api_key = self.server_config.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _post_json(
        self, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        await self._server.ensure_ready()
        url = f"{self.server_config.resolved_base_url}/{endpoint.lstrip('/')}"
        retries = self.runtime_config.transport_retries

        for attempt in range(retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.post(
                        url,
                        headers=self._request_headers(),
                        json=payload,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt >= retries:
                    raise RuntimeError(
                        f"vLLM transport failed after retries: {exc}"
                    ) from exc
                await self._server.restart_after_transport_failure()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"vLLM HTTP request failed ({exc.response.status_code}): {exc.response.text[:500]}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError("vLLM returned non-JSON response payload.") from exc

        raise RuntimeError("vLLM request retry loop exited unexpectedly.")


class VLLMChatRuntime(_BaseRuntime):
    """HTTP-only OpenAI-compatible chat runtime for vLLM."""

    def __init__(
        self,
        server_config: VLLMServerConfig,
        model_config: VLLMModelConfig,
        runtime_config: VLLMRuntimeConfig | None = None,
    ):
        super().__init__(server_config=server_config, runtime_config=runtime_config)
        self.model_config = model_config

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        guided_json: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        plain_content: bool = True,
    ) -> VLLMChatResult:
        """Submit a chat completion request and return sanitized content."""
        effective_template_kwargs = {"enable_thinking": False}
        if chat_template_kwargs:
            effective_template_kwargs.update(chat_template_kwargs)
        effective_template_kwargs["enable_thinking"] = False

        payload: dict[str, Any] = {
            "model": self.model_config.model_name,
            "messages": messages,
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            "top_p": self.model_config.top_p,
            "chat_template_kwargs": effective_template_kwargs,
        }
        if self.model_config.top_k is not None:
            payload["top_k"] = self.model_config.top_k
        if self.model_config.min_p is not None:
            payload["min_p"] = self.model_config.min_p
        if self.model_config.presence_penalty is not None:
            payload["presence_penalty"] = self.model_config.presence_penalty
        if self.model_config.repetition_penalty is not None:
            payload["repetition_penalty"] = self.model_config.repetition_penalty
        if response_format is not None:
            payload["response_format"] = response_format
        if guided_json is not None:
            payload["guided_json"] = guided_json

        preview = json.dumps(messages, ensure_ascii=True)[
            : self.runtime_config.request_preview_chars
        ]
        logger.info(
            "vLLM chat request (mode=%s, model=%s): %s",
            self.server_config.mode,
            self.model_config.model_name,
            preview,
        )

        started_at = perf_counter()
        response_json = await self._post_json("chat/completions", payload)
        latency_ms = (perf_counter() - started_at) * 1000
        response_preview = json.dumps(response_json, ensure_ascii=True)[
            : self.runtime_config.response_preview_chars
        ]

        usage = response_json.get("usage")
        logger.info(
            "vLLM chat response (mode=%s, model=%s, latency_ms=%.1f, usage=%s): %s",
            self.server_config.mode,
            self.model_config.model_name,
            latency_ms,
            usage,
            response_preview,
        )

        content = _extract_message_content(response_json)
        content = _strip_think_tags(content)
        if plain_content:
            content, removed_meta = _sanitize_plain_content(content)
            if removed_meta:
                logger.warning(
                    "Sanitized meta-output from vLLM response for plain-content request."
                )

        response_model = response_json.get("model")
        return VLLMChatResult(
            content=content,
            usage=usage if isinstance(usage, dict) else None,
            model=response_model if isinstance(response_model, str) else None,
            server_mode=self.server_config.mode,
            raw_response=response_json,
        )


class VLLMEmbeddingRuntime(_BaseRuntime):
    """HTTP-only embedding runtime for vLLM OpenAI-compatible server."""

    def __init__(
        self,
        server_config: VLLMServerConfig,
        model_config: VLLMModelConfig,
        runtime_config: VLLMRuntimeConfig | None = None,
    ):
        super().__init__(server_config=server_config, runtime_config=runtime_config)
        self.model_config = model_config

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        payload = {
            "model": self.model_config.model_name,
            "input": texts,
        }
        preview = json.dumps(
            {"size": len(texts), "sample": texts[:2]}, ensure_ascii=True
        )[: self.runtime_config.request_preview_chars]
        logger.info(
            "vLLM embedding request (mode=%s, model=%s): %s",
            self.server_config.mode,
            self.model_config.model_name,
            preview,
        )

        started_at = perf_counter()
        response_json = await self._post_json("embeddings", payload)
        latency_ms = (perf_counter() - started_at) * 1000
        usage = response_json.get("usage")
        logger.info(
            "vLLM embedding response (mode=%s, model=%s, latency_ms=%.1f, usage=%s)",
            self.server_config.mode,
            self.model_config.model_name,
            latency_ms,
            usage,
        )

        data = response_json.get("data")
        if not isinstance(data, list):
            raise RuntimeError("vLLM embeddings response missing data list.")

        embeddings: list[list[float]] = []
        ordered = sorted(
            (item for item in data if isinstance(item, dict)),
            key=lambda item: item.get("index", 0),
        )
        for item in ordered:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("vLLM embeddings response contained invalid vector.")
            embeddings.append([float(value) for value in embedding])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "vLLM embeddings response size mismatch for requested inputs."
            )
        return embeddings
