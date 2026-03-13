# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import run_flask_server
from megatron.core.utils import trace_async_exceptions
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron


def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_modelopt_args(parser)
    parser = add_inference_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
    parser.add_argument("--parsers", type=str, nargs="+", default=[], help="Parsers to use for parsing the response")
    return parser


@trace_async_exceptions
async def run_text_generation_server(
    engine: DynamicInferenceEngine, coordinator_port: int, flask_port: int
):
    """Runs the Flask server from rank 0 and initializes the DynamicInferenceEngine on all ranks.

    Args:
        engine (DynamicInferenceEngine): The dynamic inference engine.
        coordinator_port (int): The network port for the dynamic inference DP coordinator.
        flask_port (int): The network for port the frontend Flask server.
    """

    rank = torch.distributed.get_rank()

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=coordinator_port, launch_inference_coordinator=True
    )

    server_task = None
    if rank == 0:
        server_task = asyncio.create_task(
            run_flask_server(
                coordinator_addr=coordinator_addr,
                tokenizer=engine.controller.tokenizer,
                parsers=args.parsers,
                rank=rank,
                flask_port=flask_port,
                verbose=args.inference_flask_server_logging,
            )
        )
    engine_task = engine.engine_loop_task

    tasks_to_run = [engine_task]
    if server_task:
        assert rank == 0

        tasks_to_run.append(server_task)

    await asyncio.gather(*tasks_to_run)


if __name__ == "__main__":
    with torch.inference_mode():
        initialize_megatron(
            extra_args_provider=add_text_generation_server_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        # Enable return_log_probs to allow prompt logprobs computation for echo=True requests
        # This sets materialize_only_last_token_logits=False in the inference context,
        # which is required for lm-eval compatibility (loglikelihood evaluation tasks)
        args = get_args()
        args.return_log_probs = True

        engine = get_dynamic_inference_engine()

        asyncio.run(run_text_generation_server(engine, args.inference_coordinator_port, args.port))
