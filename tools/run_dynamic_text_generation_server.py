# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio
import logging

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    start_text_gen_server,
    stop_text_gen_server,
)
from megatron.core.utils import configure_nvtx_profiling, get_pg_size, trace_async_exceptions
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
        "--frontend-replicas", type=int, default=-1,
        help="Number of HTTP frontend processes spawned per hosting rank. "
             "-1 (default) uses max(data parallel size, 4), or a flat 4 with "
             "--frontend-on-all-ranks, where capacity already scales with the "
             "number of ranks hosting a frontend.",
    )
    parser.add_argument(
        "--frontend-on-all-ranks", action="store_true",
        help="Run HTTP frontends on every rank instead of only rank 0, and "
             "return every rank's URL for the caller to spread requests over. "
             "Frontend work (chat template, detokenize, parsers, JSON) is "
             "CPU-bound and otherwise confined to the hosting rank's CPU "
             "allocation, which leaves the rest of the job's cores unused. "
             "Ranks still share one DP coordinator; only the HTTP tier is "
             "replicated.",
    )
    return parser


@trace_async_exceptions
async def run_text_generation_server(
    engine: DynamicInferenceEngine, coordinator_port: int, server_port: int, hostname: str | None = None,
):
    """
    Runs the text generation server from rank 0 and initializes the
    DynamicInferenceEngine on all ranks.

    Args:
        engine (DynamicInferenceEngine): The dynamic inference engine.
        coordinator_port (int): The network port for the dynamic inference DP coordinator.
        server_port (int): The network for port the frontend text generation server.
    """

    rank = torch.distributed.get_rank()

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=coordinator_port, launch_inference_coordinator=True,
        hostname=hostname,
    )

    num_replicas = args.frontend_replicas
    if num_replicas < 0:
        if args.frontend_on_all_ranks:
            # Capacity now scales with the number of ranks, so the per-rank
            # replica count stays flat rather than tracking DP size on top of it.
            num_replicas = 4
        else:
            # Each replica is a single event loop, so frontend capacity has to scale with
            # the number of engines it feeds. The floor of 4 preserves the previous default
            # for small deployments.
            num_replicas = max(get_pg_size(engine.pg_collection.dp), 4)
    if rank == 0:
        logging.info("Starting %d HTTP frontend replica(s) per hosting rank.", num_replicas)

    if args.frontend_on_all_ranks:
        # Only the DP coordinator rank learns the coordinator's address: the
        # engine broadcasts it over the DP group, which is a singleton when data
        # parallel size is 1. Every rank needs it here, since every rank's
        # frontend opens its own client.
        address = [coordinator_addr]
        torch.distributed.broadcast_object_list(address, src=0)
        coordinator_addr = address[0]
        assert coordinator_addr is not None, "no rank published a DP coordinator address"

    try:
        url = None
        if args.frontend_on_all_ranks or rank == 0:
            url = start_text_gen_server(
                coordinator_addr=coordinator_addr,
                tokenizer=engine.controller.tokenizer,
                parsers=args.parsers,
                rank=rank,
                server_port=0 if args.frontend_on_all_ranks else server_port,
                verbose=args.inference_text_gen_server_logging,
                num_replicas=num_replicas,
                hostname=hostname,
            )

        if args.frontend_on_all_ranks:
            # Unlike callers that already collect a URL per worker, this entry
            # point has to gather them itself before it can report the set.
            urls = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(urls, url)
            if rank == 0:
                for entry in [u for u in urls if u]:
                    logging.info("Frontend: %s", entry)
        elif rank == 0:
            logging.info("Frontend: %s", url)

        # Await the engine loop directly since the server is running in a separate process
        await engine.engine_loop_task

    finally:
        # Guarantee that the separate processes are terminated when the engine loop
        # stops or is interrupted. Every rank may now own frontend processes.
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
                run_text_generation_server(engine, args.inference_coordinator_port, args.port, args.host)
            )
        except KeyboardInterrupt:
            # Catching at the top level ensures clean stdout without spamming the traceback
            print("Server process interrupted by user.")
        finally:
            # Clean up PyTorch distributed groups properly
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
