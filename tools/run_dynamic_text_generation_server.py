# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    start_text_gen_server,
    stop_text_gen_server,
)
from megatron.core.utils import configure_nvtx_profiling, trace_async_exceptions
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.initialize import initialize_megatron


def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_modelopt_args(parser)
    parser = add_inference_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
    parser.add_argument(
        "--host", type=str, default=None,
        help="Hostname or IP address to bind the server to. Defaults to 0.0.0.0 (all interfaces)."
    )
    parser.add_argument(
        "--parsers", type=str, nargs="+", default=[], help="Parsers to use for parsing the response"
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=1.0,
        help="Default top-p sampling value when a request does not specify top_p.",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=0,
        help="Default top-k sampling value when a request does not specify top_k.",
    )
    parser.add_argument(
        "--serving-mode",
        action="store_true",
        help=(
            "Optimize defaults for pure serving. In chat requests, prevent_retokenization "
            "defaults to false so prompt token IDs are not returned."
        ),
    )
    return parser


@trace_async_exceptions
async def run_text_generation_server(
    engine: DynamicInferenceEngine,
    coordinator_port: int,
    server_port: int,
    hostname: str | None = None,
    default_top_p: float = 1.0,
    default_top_k: int = 0,
    serving_mode: bool = False,
):
    """
    Runs the text generation server from rank 0 and initializes the
    DynamicInferenceEngine on all ranks.

    Args:
        engine (DynamicInferenceEngine): The dynamic inference engine.
        coordinator_port (int): The network port for the dynamic inference DP coordinator.
        server_port (int): The network for port the frontend text generation server.
        hostname (str | None): Hostname or IP address for coordinator and HTTP traffic.
        default_top_p (float): Sampling default when a request omits ``top_p``.
        default_top_k (int): Sampling default when a request omits ``top_k``.
        serving_mode (bool): Whether to use pure-serving response defaults.
    """

    rank = torch.distributed.get_rank()

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=coordinator_port, launch_inference_coordinator=True,
        hostname=hostname,
    )

    try:
        if rank == 0:
            start_text_gen_server(
                coordinator_addr=coordinator_addr,
                tokenizer=engine.controller.tokenizer,
                parsers=args.parsers,
                rank=rank,
                server_port=server_port,
                verbose=args.inference_text_gen_server_logging,
                hostname=hostname,
                default_top_p=default_top_p,
                default_top_k=default_top_k,
                serving_mode=serving_mode,
            )

        # Await the engine loop directly since the server is running in a separate process
        await engine.engine_loop_task

    finally:
        # Guarantee that the separate process is terminated when the engine loop stops or is interrupted
        if rank == 0:
            stop_text_gen_server()


if __name__ == "__main__":
    with torch.inference_mode():
        parse_and_validate_args(
            extra_args_provider=add_text_generation_server_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )
        initialize_megatron()

        args = get_args()

        # Match training's NVTX gating (training.py only flips this when both
        # --profile and --nvtx-ranges are set). Otherwise the engine-side
        # nvtx_range_push labels (bookkeeping, Decode, _ep_establish_consensus,
        # etc.) are no-ops and the inter-step gap is unattributable in nsys.
        if args.profile and args.nvtx_ranges:
            configure_nvtx_profiling(True)

        # Enable return_log_probs to allow prompt logprobs computation for echo=True requests
        # This sets materialize_only_last_token_logits=False in the inference context,
        # which is required for lm-eval compatibility (loglikelihood evaluation tasks)
        args.return_log_probs = True

        engine = get_dynamic_inference_engine()

        try:
            asyncio.run(
                run_text_generation_server(
                    engine,
                    args.inference_coordinator_port,
                    args.port,
                    args.host,
                    args.default_top_p,
                    args.default_top_k,
                    args.serving_mode,
                )
            )
        except KeyboardInterrupt:
            # Catching at the top level ensures clean stdout without spamming the traceback
            print("Server process interrupted by user.")
        finally:
            # Clean up PyTorch distributed groups properly
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
