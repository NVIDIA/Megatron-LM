# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio
import os

import torch

from examples.inference.gpt.gpt_dynamic_inference import add_dynamic_inference_args
from megatron.core.inference.text_generation_server import run_text_generation_server
from megatron.training.initialize import initialize_megatron


def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_dynamic_inference_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
    return parser


if __name__ == "__main__":
    with torch.inference_mode():

        initialize_megatron(
            extra_args_provider=add_text_generation_server_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        try:
            asyncio.run(run_text_generation_server())
        except KeyboardInterrupt:
            pass
        finally:
            if os.environ.get("NSIGHT_PREFIX"):
                torch.cuda.cudart().cudaProfilerStop()
