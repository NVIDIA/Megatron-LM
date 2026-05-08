# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""OpenAI-compatible inference server using the Megatron high-level API.

Mirrors tools/run_dynamic_text_generation_server.py but drives the
``DynamicInferenceEngine`` through ``MegatronAsyncLLM.serve(...)`` instead
of building the coordinator/engine pipeline manually. Coordinator mode is
required (HTTP serving uses the coordinator path); ``use_coordinator=True``
is hardcoded in the script.
"""

import asyncio
import os
import sys
from argparse import ArgumentParser

import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import configure_nvtx_profiling
from megatron.inference import MegatronAsyncLLM, ServeConfig
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)
from megatron.training import get_args, initialize_megatron
from megatron.training.arguments import parse_and_validate_args


def add_serve_args(parser: ArgumentParser) -> ArgumentParser:
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title='High-level inference server')
    group.add_argument("--coordinator-host", type=str, default=None)
    group.add_argument("--coordinator-port", type=int, default=None)
    group.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind host")
    group.add_argument("--port", type=int, default=5000, help="HTTP bind port")
    group.add_argument(
        "--parsers", type=str, nargs="+", default=[], help="Response parser names"
    )
    group.add_argument(
        "--verbose", action="store_true", default=False, help="Per-request HTTP logging"
    )
    group.add_argument(
        "--frontend-replicas", type=int, default=4,
        help="Number of HTTP frontend processes spawned on the primary rank.",
    )
    return parser


async def _serve(args, model, tokenizer, inference_config):
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=True,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
    ) as llm:
        serve_config = ServeConfig(
            host=args.host,
            port=args.port,
            parsers=args.parsers,
            verbose=args.verbose,
            frontend_replicas=args.frontend_replicas,
        )
        await llm.serve(serve_config, blocking=True)


def main():
    parse_and_validate_args(
        extra_args_provider=add_serve_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )
    initialize_megatron()

    args = get_args()

    # Match the legacy tool's NVTX gating.
    if args.profile and args.nvtx_ranges:
        configure_nvtx_profiling(True)

    # Required for lm-eval loglikelihood compatibility: keeps prompt logits
    # materialized so echo=True / logprob requests work end-to-end. Matches
    # tools/run_dynamic_text_generation_server.py.
    args.return_log_probs = True

    tokenizer = build_tokenizer(args)
    model = get_model_for_inference()
    inference_config = get_inference_config_from_model_and_args(model, args)

    try:
        asyncio.run(_serve(args, model, tokenizer, inference_config))
    except KeyboardInterrupt:
        print("Server process interrupted by user.")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
