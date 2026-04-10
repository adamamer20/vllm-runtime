#!/usr/bin/env python3
"""Canonical unit tests for the extracted vLLM runtime package."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from vllm_runtime.config import VLLMModelConfig, VLLMRuntimeConfig, VLLMServerConfig
from vllm_runtime.runtime import (
    VLLMChatRuntime,
    VLLMEmbeddingRuntime,
    _ServerController,
    _get_server_lock,
)

HTTP_ERROR_STATUS = 400
EXPECTED_ATTEMPTS_AFTER_RETRY = 2


class _DummyResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = "ok"

    def raise_for_status(self) -> None:
        if self.status_code >= HTTP_ERROR_STATUS:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", "http://runtime"),
                response=httpx.Response(self.status_code, text=self.text),
            )

    def json(self) -> dict[str, Any]:
        return self._payload


def test_server_lock_is_keyed_by_resolved_base_url() -> None:
    chat_cfg = VLLMServerConfig(
        mode="managed",
        model_name="chat-model",
        host="127.0.0.1",
        port=8000,
    )
    embed_same_port_cfg = VLLMServerConfig(
        mode="managed",
        model_name="embed-model",
        host="127.0.0.1",
        port=8000,
    )
    embed_other_port_cfg = VLLMServerConfig(
        mode="managed",
        model_name="embed-model",
        host="127.0.0.1",
        port=8001,
    )

    lock_chat = _get_server_lock(chat_cfg.resolved_base_url)
    lock_embed_same_port = _get_server_lock(embed_same_port_cfg.resolved_base_url)
    lock_embed_other_port = _get_server_lock(embed_other_port_cfg.resolved_base_url)

    assert lock_chat is lock_embed_same_port
    assert lock_chat is not lock_embed_other_port


@pytest.mark.asyncio
async def test_managed_runtimes_with_different_ports_initialize_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_ports: set[int] = set()
    healthy_ports: set[int] = set()
    active_starts = 0
    peak_parallel_starts = 0

    async def _fake_is_healthy(self: _ServerController) -> bool:
        return self._config.port in healthy_ports  # noqa: SLF001

    async def _fake_start_managed_server(self: _ServerController) -> None:
        nonlocal active_starts, peak_parallel_starts
        active_starts += 1
        peak_parallel_starts = max(peak_parallel_starts, active_starts)
        started_ports.add(self._config.port)  # noqa: SLF001
        await asyncio.sleep(0.02)
        healthy_ports.add(self._config.port)  # noqa: SLF001
        active_starts -= 1

    async def _fake_wait_for_health(
        self: _ServerController, *, allow_warmup: bool
    ) -> None:
        _ = self
        _ = allow_warmup

    monkeypatch.setattr(_ServerController, "_is_healthy", _fake_is_healthy)
    monkeypatch.setattr(
        _ServerController, "_start_managed_server", _fake_start_managed_server
    )
    monkeypatch.setattr(_ServerController, "_wait_for_health", _fake_wait_for_health)

    chat_runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(
            mode="managed",
            model_name="chat-model",
            host="127.0.0.1",
            port=8100,
        ),
        model_config=VLLMModelConfig(model_name="chat-model"),
    )
    embed_runtime = VLLMEmbeddingRuntime(
        server_config=VLLMServerConfig(
            mode="managed",
            model_name="embed-model",
            host="127.0.0.1",
            port=8101,
        ),
        model_config=VLLMModelConfig(model_name="embed-model"),
    )

    await asyncio.gather(chat_runtime.ensure_ready(), embed_runtime.ensure_ready())
    await chat_runtime.close()
    await embed_runtime.close()

    assert started_ports == {8100, 8101}
    assert peak_parallel_starts == 2
    assert _get_server_lock(
        chat_runtime.server_config.resolved_base_url
    ) is not _get_server_lock(embed_runtime.server_config.resolved_base_url)


def test_same_base_url_different_models_is_unsupported_topology() -> None:
    chat_cfg = VLLMServerConfig(
        mode="managed",
        model_name="chat-model",
        host="127.0.0.1",
        port=8200,
    )
    embed_cfg = VLLMServerConfig(
        mode="managed",
        model_name="embed-model",
        host="127.0.0.1",
        port=8200,
    )

    assert chat_cfg.resolved_base_url == embed_cfg.resolved_base_url
    assert _get_server_lock(chat_cfg.resolved_base_url) is _get_server_lock(
        embed_cfg.resolved_base_url
    )


@pytest.mark.asyncio
async def test_managed_server_startup_and_shutdown(monkeypatch: pytest.MonkeyPatch):
    controller = _ServerController(
        VLLMServerConfig(
            mode="managed", model_name="model", host="127.0.0.1", port=9999
        )
    )
    state = {"started": 0}

    class _DummyProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = 0

        async def wait(self) -> int:
            return 0

    async def _fake_is_healthy() -> bool:
        return state["started"] > 0

    async def _fake_start_managed_server() -> None:
        state["started"] += 1
        controller._process = _DummyProcess()  # noqa: SLF001
        controller._owns_process = True  # noqa: SLF001

    async def _fake_wait_for_health(*, allow_warmup: bool) -> None:
        _ = allow_warmup

    monkeypatch.setattr(controller, "_is_healthy", _fake_is_healthy)
    monkeypatch.setattr(controller, "_start_managed_server", _fake_start_managed_server)
    monkeypatch.setattr(controller, "_wait_for_health", _fake_wait_for_health)

    await controller.ensure_ready()
    assert state["started"] == 1

    process = controller._process  # noqa: SLF001
    await controller.shutdown()
    assert process is not None
    assert process.terminated is True


@pytest.mark.asyncio
async def test_external_server_mode_waits_for_health(monkeypatch: pytest.MonkeyPatch):
    controller = _ServerController(
        VLLMServerConfig(
            mode="external",
            model_name="model",
            base_url="http://runtime.local/v1",
            allow_remote_warmup=True,
        )
    )
    called: dict[str, Any] = {"allow_warmup": None}

    async def _fake_is_healthy() -> bool:
        return False

    async def _fake_wait_for_health(*, allow_warmup: bool) -> None:
        called["allow_warmup"] = allow_warmup

    monkeypatch.setattr(controller, "_is_healthy", _fake_is_healthy)
    monkeypatch.setattr(controller, "_wait_for_health", _fake_wait_for_health)

    await controller.ensure_ready()
    assert called["allow_warmup"] is True


@pytest.mark.asyncio
async def test_chat_runtime_includes_structured_output_and_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}
    runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(
            mode="external",
            model_name="model",
            base_url="http://runtime.local/v1",
        ),
        model_config=VLLMModelConfig(model_name="model"),
    )

    async def _fake_post_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": '{"title":"ok"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "model",
        }

    monkeypatch.setattr(runtime, "_post_json", _fake_post_json)
    result = await runtime.chat(
        [{"role": "user", "content": "hello"}],
        response_format={"type": "json_object"},
        guided_json={"type": "object"},
        chat_template_kwargs={"enable_thinking": True, "foo": "bar"},
        plain_content=False,
    )
    await runtime.close()

    assert result.content == '{"title":"ok"}'
    assert captured["endpoint"] == "chat/completions"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["guided_json"] == {"type": "object"}
    assert captured["payload"]["chat_template_kwargs"]["foo"] == "bar"
    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_chat_runtime_sanitizes_reasoning_and_meta_output(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(
            mode="external",
            model_name="model",
            base_url="http://runtime.local/v1",
        ),
        model_config=VLLMModelConfig(model_name="model"),
    )

    async def _fake_post_json(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": "<think>chain of thought</think>\nThe user wants a summary.\nFinal answer"
                    }
                }
            ],
            "usage": {},
            "model": "model",
        }

    monkeypatch.setattr(runtime, "_post_json", _fake_post_json)
    result = await runtime.chat(
        [{"role": "user", "content": "hello"}], plain_content=True
    )
    await runtime.close()

    assert result.content == "Final answer"


@pytest.mark.asyncio
async def test_chat_runtime_multimodal_payload_passthrough(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}
    runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(
            mode="external",
            model_name="model",
            base_url="http://runtime.local/v1",
        ),
        model_config=VLLMModelConfig(model_name="model"),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,abc"},
                },
            ],
        }
    ]

    async def _fake_post_json(
        _endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured["messages"] = payload["messages"]
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
            "model": "model",
        }

    monkeypatch.setattr(runtime, "_post_json", _fake_post_json)
    await runtime.chat(messages, plain_content=True)
    await runtime.close()

    assert captured["messages"] == messages


@pytest.mark.asyncio
async def test_embedding_runtime_supports_batch_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = VLLMEmbeddingRuntime(
        server_config=VLLMServerConfig(
            mode="external",
            model_name="embed-model",
            base_url="http://runtime.local/v1",
        ),
        model_config=VLLMModelConfig(model_name="embed-model"),
    )

    async def _fake_post_json(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "data": [
                {"index": 1, "embedding": [2.0, 3.0]},
                {"index": 0, "embedding": [0.5, 1.5]},
            ],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }

    monkeypatch.setattr(runtime, "_post_json", _fake_post_json)
    vectors = await runtime.embed_texts(["a", "b"])
    await runtime.close()

    assert vectors == [[0.5, 1.5], [2.0, 3.0]]


@pytest.mark.asyncio
async def test_transport_retry_restarts_managed_server(monkeypatch: pytest.MonkeyPatch):
    runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(mode="managed", model_name="model"),
        model_config=VLLMModelConfig(model_name="model"),
        runtime_config=VLLMRuntimeConfig(transport_retries=1),
    )
    attempts = {"count": 0}
    restart_calls = {"count": 0}

    async def _fake_ensure_ready() -> None:
        return None

    async def _fake_restart() -> None:
        restart_calls["count"] += 1

    async def _fake_post(*_args: Any, **_kwargs: Any) -> _DummyResponse:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.TransportError("boom")
        return _DummyResponse({"ok": True})

    monkeypatch.setattr(runtime._server, "ensure_ready", _fake_ensure_ready)
    monkeypatch.setattr(
        runtime._server, "restart_after_transport_failure", _fake_restart
    )
    monkeypatch.setattr(runtime._client, "post", _fake_post)

    payload = await runtime._post_json("chat/completions", {"messages": []})  # noqa: SLF001
    await runtime.close()

    assert payload == {"ok": True}
    assert attempts["count"] == EXPECTED_ATTEMPTS_AFTER_RETRY
    assert restart_calls["count"] == 1
