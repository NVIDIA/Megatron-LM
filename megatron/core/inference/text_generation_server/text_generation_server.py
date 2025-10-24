# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio
from typing import Optional

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.flask_server import run_flask_server
from megatron.training import get_tokenizer


async def run_text_generation_server(
    engine: DynamicInferenceEngine,
    coordinator_port: int,
    flask_port: int,
    default_sampling_params: Optional[SamplingParams] = None,
):
    """Runs the Flask server from rank 0 and initializes the DynamicInferenceEngine on all ranks.

    Args:
        engine (DynamicInferenceEngine): The dynamic inference engine.
        coordinator_port (int): The network port for the dynamic inference DP coordinator.
        flask_port (int): The network for port the frontend Flask server.
        defualt_sampling_params (SamplingParams): The default sampling params.
            This will be deprecated once we have per-request sampling params.
    """

    tokenizer = get_tokenizer()
    rank = torch.distributed.get_rank()

    await engine.start_listening_to_data_parallel_coordinator(
        default_sampling_params,
        inference_coordinator_port=coordinator_port,
        launch_inference_coordinator=True,
    )

    server_task = None
    if rank == 0:
        server_task = asyncio.create_task(
            run_flask_server(
                coordinator_port=coordinator_port,
                tokenizer=tokenizer,
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
