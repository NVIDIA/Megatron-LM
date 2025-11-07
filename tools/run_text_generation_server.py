# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio
import os

import torch

from examples.inference.gpt.gpt_dynamic_inference import (
    add_dynamic_inference_args,
    get_inference_context,
    get_inference_controller,
    get_model,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.text_generation_server import run_text_generation_server
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_modelopt_args(parser)
    parser = add_dynamic_inference_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
    return parser


if __name__ == "__main__":
    with torch.inference_mode():
        initialize_megatron(
            extra_args_provider=add_text_generation_server_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        args = get_args()
        model = get_model()
        context = get_inference_context(None, None, calculate_max_sequence_length_from_requests=False)
        controller = get_inference_controller(model, context)

        engine = DynamicInferenceEngine(
            controller,
            context,
            termination_id=-1,
            enable_cuda_graph=args.cuda_graph_impl == "local",
            random_seed=args.seed,
        )

        default_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_log_probs=args.return_log_probs,
            num_tokens_to_generate=args.num_tokens_to_generate,
        )

        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        try:
            asyncio.run(
                run_text_generation_server(
                    engine,
                    args.inference_coordinator_port,
                    args.port,
                    default_sampling_params=default_sampling_params
                )
            )
        except KeyboardInterrupt:
            pass
        finally:
            if os.environ.get("NSIGHT_PREFIX"):
                torch.cuda.cudart().cudaProfilerStop()
