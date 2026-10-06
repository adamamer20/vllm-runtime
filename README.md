# vllm-runtime

`vllm-runtime` is a standalone, HTTP-first, OpenAI-compatible runtime wrapper for vLLM and llama.cpp (`llama-server`) with managed and external server modes.

Each runtime instance manages exactly one endpoint (one `resolved_base_url`). Managed server lifecycle ownership and startup locking are keyed by that resolved base URL.

If you run different models for chat and embeddings, configure different endpoints (typically different ports). Reusing the same managed `host:port` for different models is not a supported topology.

## OCR and exact generated text

Use `raw_content=True` with either chat runtime to preserve generated text,
including whitespace, markup, `<think>` tags, and model commentary. The default
chat mode continues to sanitize reasoning and meta-output. `raw_response` retains
the parsed JSON response in both modes; it is not the original HTTP byte stream.

For a resident OCR server, send concurrent requests to one external endpoint.
The server owns GPU scheduling and continuous batching; the wrapper limits
concurrent HTTP requests. Use `transport_retries=0` when an ambiguous failure must
be recorded and inspected before another inference attempt.

```python
runtime = VLLMChatRuntime(
    server_config=VLLMServerConfig(
        mode="external", model_name="ocr", base_url="http://127.0.0.1:8000/v1",
    ),
    model_config=VLLMModelConfig(
        model_name="ocr", temperature=0, top_p=1, max_tokens=8192,
    ),
    runtime_config=VLLMRuntimeConfig(
        max_concurrent_requests=4, transport_retries=0,
    ),
)
try:
    result = await runtime.chat(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
        raw_content=True,
    )
    # Retain before parsing; generated text still needs source-quality validation.
    generated_text = result.content
finally:
    await runtime.close()
```

Managed startup locking is local to one Python process. GPU leases, process
supervision, source identities and durable receipts belong to the application.

## License

[MIT](LICENSE). The vLLM and llama.cpp engines and model weights have their own
licenses and are installed separately.

## llama.cpp managed server examples

Local GGUF:

```python
import asyncio

from vllm_runtime import (
    LlamaCppChatRuntime,
    LlamaCppModelConfig,
    LlamaCppServerConfig,
)


async def main() -> None:
    runtime = LlamaCppChatRuntime(
        server_config=LlamaCppServerConfig(
            mode="managed",
            model_name="qwen-gguf",
            model_path="/models/Qwen.gguf",
            host="127.0.0.1",
            port=8010,
        ),
        model_config=LlamaCppModelConfig(model_name="qwen-gguf"),
    )
    try:
        result = await runtime.chat([{"role": "user", "content": "Hello"}])
        print(result.content)
    finally:
        await runtime.close()


asyncio.run(main())
```

Hugging Face GGUF:

```python
import asyncio

from vllm_runtime import (
    LlamaCppChatRuntime,
    LlamaCppModelConfig,
    LlamaCppServerConfig,
)


async def main() -> None:
    runtime = LlamaCppChatRuntime(
        server_config=LlamaCppServerConfig(
            mode="managed",
            model_name="qwen-gguf",
            hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
            hf_file="qwen2.5-7b-instruct-q4_k_m.gguf",
            host="127.0.0.1",
            port=8010,
        ),
        model_config=LlamaCppModelConfig(model_name="qwen-gguf"),
    )
    try:
        result = await runtime.chat([{"role": "user", "content": "Hello"}])
        print(result.content)
    finally:
        await runtime.close()


asyncio.run(main())
```

## Installation

Local editable install:

```bash
uv pip install -e .
```

Install from Git:

```bash
uv add "vllm-runtime @ git+ssh://<your-host>/<org>/vllm-runtime.git"
```

## Managed server example

```python
import asyncio

from vllm_runtime import VLLMChatRuntime, VLLMModelConfig, VLLMServerConfig


async def main() -> None:
    runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(
            mode="managed",
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            host="127.0.0.1",
            port=8000,
        ),
        model_config=VLLMModelConfig(model_name="Qwen/Qwen2.5-VL-7B-Instruct"),
    )
    try:
        result = await runtime.chat([{"role": "user", "content": "Hello"}])
        print(result.content)
    finally:
        await runtime.close()


asyncio.run(main())
```

## External server example

```python
import asyncio

from vllm_runtime import VLLMChatRuntime, VLLMModelConfig, VLLMServerConfig


async def main() -> None:
    runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(
            mode="external",
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            base_url="http://127.0.0.1:8000/v1",
        ),
        model_config=VLLMModelConfig(model_name="Qwen/Qwen2.5-VL-7B-Instruct"),
    )
    try:
        result = await runtime.chat([{"role": "user", "content": "Summarize this"}])
        print(result.content)
    finally:
        await runtime.close()


asyncio.run(main())
```

## Embeddings example

```python
import asyncio

from vllm_runtime import VLLMEmbeddingRuntime, VLLMModelConfig, VLLMServerConfig


async def main() -> None:
    runtime = VLLMEmbeddingRuntime(
        server_config=VLLMServerConfig(
            mode="external",
            model_name="intfloat/e5-large-v2",
            base_url="http://127.0.0.1:8000/v1",
        ),
        model_config=VLLMModelConfig(model_name="intfloat/e5-large-v2"),
    )
    try:
        vectors = await runtime.embed_texts(["alpha", "beta"])
        print(len(vectors), len(vectors[0]))
    finally:
        await runtime.close()


asyncio.run(main())
```

## Dual managed servers (chat + embeddings)

```python
import asyncio

from vllm_runtime import (
    VLLMChatRuntime,
    VLLMEmbeddingRuntime,
    VLLMModelConfig,
    VLLMServerConfig,
)


async def main() -> None:
    chat_runtime = VLLMChatRuntime(
        server_config=VLLMServerConfig(
            mode="managed",
            model_name="cyankiwi/Qwen3.5-9B-AWQ-4bit",
            host="127.0.0.1",
            port=8000,
        ),
        model_config=VLLMModelConfig(model_name="cyankiwi/Qwen3.5-9B-AWQ-4bit"),
    )
    embed_runtime = VLLMEmbeddingRuntime(
        server_config=VLLMServerConfig(
            mode="managed",
            model_name="Qwen/Qwen3-Embedding-0.6B",
            host="127.0.0.1",
            port=8001,
        ),
        model_config=VLLMModelConfig(model_name="Qwen/Qwen3-Embedding-0.6B"),
    )

    try:
        chat = await chat_runtime.chat([{"role": "user", "content": "Hello"}])
        vectors = await embed_runtime.embed_texts(["alpha", "beta"])
        print(chat.content, len(vectors))
    finally:
        await chat_runtime.close()
        await embed_runtime.close()


asyncio.run(main())
```

## Healthcheck CLI

```bash
vllm-runtime-health --mode external --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:8000/v1
```
