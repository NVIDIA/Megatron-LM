# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import asyncio

import torch

from examples.inference.gpt.gpt_dynamic_inference import (
    get_inference_context,
    get_inference_controller,
    get_model,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_server.flask_server import run_flask_server
from megatron.training import get_args, get_tokenizer


async def run_text_generation_server():
    """Runs the Flask server from rank 0 and initializes the DynamicInferenceEngine on all ranks."""

    args = get_args()
    tokenizer = get_tokenizer()
    rank = torch.distributed.get_rank()

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

    coordinator_port = args.inference_coordinator_port

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
                flask_port=args.port,
            )
        )
    engine_task = engine.engine_loop_task

    tasks_to_run = [engine_task]
    if server_task:
        assert rank == 0

        tasks_to_run.append(server_task)

    await asyncio.gather(*tasks_to_run)
