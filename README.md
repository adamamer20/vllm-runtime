# vllm-runtime

`vllm-runtime` is a standalone, HTTP-first, OpenAI-compatible runtime wrapper for vLLM with managed and external server modes.

Each runtime instance manages exactly one endpoint (one `resolved_base_url`). Managed server lifecycle ownership and startup locking are keyed by that resolved base URL.

If you run different models for chat and embeddings, configure different endpoints (typically different ports). Reusing the same managed `host:port` for different models is not a supported topology.

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
