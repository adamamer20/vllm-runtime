"""Exercise raw OCR content through the real HTTP client and public runtimes."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vllm_runtime import (
    LlamaCppChatRuntime,
    LlamaCppModelConfig,
    LlamaCppRuntimeConfig,
    LlamaCppServerConfig,
    VLLMChatRuntime,
    VLLMModelConfig,
    VLLMRuntimeConfig,
    VLLMServerConfig,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["vllm", "llama"])
async def test_raw_content_preserves_response_over_real_http(backend: str) -> None:
    text = "  \n<think>literal OCR markup</think>\nAssistant note: source text\n<table><tr><td>2.157,88</td></tr></table>\n "
    payload = {"model": "ocr", "choices": [{"message": {"content": text}}]}
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(json.loads(body))
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    if backend == "vllm":
        runtime = VLLMChatRuntime(
            server_config=VLLMServerConfig(
                mode="external", model_name="ocr", base_url=base_url
            ),
            model_config=VLLMModelConfig(model_name="ocr"),
            runtime_config=VLLMRuntimeConfig(transport_retries=0),
        )
    else:
        runtime = LlamaCppChatRuntime(
            server_config=LlamaCppServerConfig(
                mode="external", model_name="ocr", base_url=base_url
            ),
            model_config=LlamaCppModelConfig(model_name="ocr"),
            runtime_config=LlamaCppRuntimeConfig(transport_retries=0),
        )
    try:
        messages = [{"role": "user", "content": "OCR retained image"}]
        raw = await runtime.chat(messages, raw_content=True)
        assert raw.content == text
        assert raw.raw_response == payload
        sanitized = await runtime.chat(messages)
        assert "<think>" not in sanitized.content
        assert "Assistant note:" not in sanitized.content
        assert "2.157,88" in sanitized.content
        assert sanitized.raw_response == payload
        assert len(received) == 2
        assert all(item["messages"] == messages for item in received)
    finally:
        await runtime.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        thread.join(timeout=5)
