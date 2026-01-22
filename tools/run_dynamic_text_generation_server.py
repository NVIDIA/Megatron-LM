# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio

import torch

from examples.inference.gpt.gpt_dynamic_inference import (
    add_dynamic_inference_args,
    get_inference_context,
    get_inference_controller,
    get_model,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import run_flask_server
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_mamba_inference_state_config_from_model, trace_async_exceptions
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args, get_tokenizer
from megatron.training.initialize import initialize_megatron


def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_modelopt_args(parser)
    parser = add_dynamic_inference_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
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

    await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=coordinator_port, launch_inference_coordinator=True
    )

    server_task = None
    if rank == 0:
        server_task = asyncio.create_task(
            run_flask_server(
                coordinator_port=coordinator_port,
                tokenizer=engine.controller.tokenizer,
                rank=rank,
                flask_port=flask_port,
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

        args = get_args()
        model = get_model()

        if args.legacy_tokenizer:
            tokenizer = get_tokenizer()
        else:
            tokenizer = build_tokenizer(args)

        mamba_inference_state_config = get_mamba_inference_state_config_from_model(model)

        # Enable return_log_probs to allow prompt logprobs computation for echo=True requests
        # This sets materialize_only_last_token_logits=False in the inference context,
        # which is required for lm-eval compatibility (loglikelihood evaluation tasks)
        args.return_log_probs = True

        context = get_inference_context(
            None,
            None,
            calculate_max_sequence_length_from_requests=False,
            mamba_inference_state_config=mamba_inference_state_config,
        )

        controller = get_inference_controller(model, context)

        engine = DynamicInferenceEngine(
            controller,
            context,
            enable_cuda_graph=args.cuda_graph_impl == "local",
            random_seed=args.seed,
            enable_chunked_prefill=not args.disable_chunked_prefill,
        )

        asyncio.run(run_text_generation_server(engine, args.inference_coordinator_port, args.port))
