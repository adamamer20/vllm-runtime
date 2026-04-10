"""Small helper CLI for vLLM runtime health checks."""

from __future__ import annotations

import argparse
import asyncio
import logging

from vllm_runtime import (
    VLLMChatRuntime,
    VLLMModelConfig,
    VLLMRuntimeConfig,
    VLLMServerConfig,
)


async def _run_healthcheck(args: argparse.Namespace) -> None:
    server_config = VLLMServerConfig(
        mode=args.mode,
        model_name=args.model,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        api_key_env_var=args.api_key_env_var,
    )
    model_config = VLLMModelConfig(model_name=args.model)
    runtime = VLLMChatRuntime(
        server_config=server_config,
        model_config=model_config,
        runtime_config=VLLMRuntimeConfig(),
    )
    try:
        await runtime.ensure_ready()
    finally:
        await runtime.close()


def main() -> None:
    """Run the runtime helper CLI entrypoint."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="vLLM runtime helper CLI")
    parser.add_argument("--mode", choices=["managed", "external"], default="external")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env-var", default="VLLM_API_KEY")
    args = parser.parse_args()
    asyncio.run(_run_healthcheck(args))


if __name__ == "__main__":
    main()
