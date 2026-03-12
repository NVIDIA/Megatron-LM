# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from typing import List

import torch
import torch.distributed as dist

from examples.inference.gpt.utils import Request, build_dynamic_engine_setup_prefix, build_requests
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord
from megatron.core.inference.sampling_params import SamplingParams
from megatron.inference.utils import (
    add_inference_args,
    get_dynamic_inference_engine,
    get_model_for_inference,
)
from megatron.training import get_args, get_tokenizer, initialize_megatron

# pylint: disable=line-too-long

logging.basicConfig(level=logging.INFO, force=True)


async def suspend_resume_cycle(client, engine, args, futures):
    """Wait for all in-flight requests, then suspend/train/resume."""
    await asyncio.gather(*futures)

    client.pause_engines()
    await engine.wait_until(EngineState.PAUSED)
    client.suspend_engines()
    await engine.wait_until(EngineState.SUSPENDED)
    if args.suspend_timeout > 0:
        await asyncio.sleep(args.suspend_timeout)
    client.resume_engines()
    await engine.wait_until(EngineState.RESUMED)
    client.unpause_engines()
    await engine.wait_until(EngineState.RUNNING)


async def main(
    engine: DynamicInferenceEngine,
    requests: List[Request],
    port: int | None = None,
    sampling_params: SamplingParams | None = None,
):
    if sampling_params is not None:
        warnings.warn(
            "The `sampling_params` argument is deprecated. "
            "Sampling parameters are specified per request.",
            DeprecationWarning,
        )

    # once you call engine.start_listening_to_data_parallel_coordinator,
    # the engine will start accepting requests from the data parallel coordinator.
    # and processing them in an asyncio coroutine.
    # leaving inference_coordinator_port as None will find a free port automatically.
    args = get_args()

    dp_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=port,
        launch_inference_coordinator=True,
        coordinator_schedule_output_path=args.coordinator_schedule_output_path,
    )

    # All ranks agree on the number of suspend/resume cycles from args.
    num_suspend_resume_cycles = len(requests) // args.suspend_resume_interval if args.suspend_resume_interval else 0

    # Create client and run example.
    if dist.get_rank() == 0:
        client = InferenceClient(dp_addr, deserialize=True)  # submits requests to the inference coordinator
        client.start()
        base_arrival_time = time.time_ns() / 10**9
        for request in requests:
            request.time_arrival = request.time_offset + base_arrival_time
        futures = []
        num_requests_total = len(requests)
        num_requests_added = 0
        next_suspend_at = args.suspend_resume_interval or 0
        cycles_done = 0

        while True:
            current_time = time.time_ns() / 10**9
            if args.incoming_requests_per_step is None:
                # Only add requests that have arrived at the current time.
                while (
                    num_requests_added < num_requests_total
                    and requests[num_requests_added].time_arrival <= current_time
                ):
                    request = requests[num_requests_added]
                    # These add-request calls will queue up the request on a zmq socket and return
                    # instantaneously. They will return an asyncio future which can be awaited for
                    # request completion.
                    futures.append(client.add_request(request.prompt_text, request.sampling_params))
                    num_requests_added += 1

                    if num_requests_added >= next_suspend_at and cycles_done < num_suspend_resume_cycles:
                        await suspend_resume_cycle(client, engine, args, futures)
                        cycles_done += 1
                        next_suspend_at += args.suspend_resume_interval

            else:
                # Add deterministic number of requests (generally used for debugging).
                for i in range(
                    min(args.incoming_requests_per_step, num_requests_total - num_requests_added)
                ):
                    # Change sampling parameters to force different generation lengths.
                    request = requests[num_requests_added]
                    n = request.sampling_params.num_tokens_to_generate
                    request.sampling_params.num_tokens_to_generate = n + i
                    futures.append(client.add_request(request.prompt_text, request.sampling_params))
                    num_requests_added += 1

                    if num_requests_added >= next_suspend_at and cycles_done < num_suspend_resume_cycles:
                        await suspend_resume_cycle(client, engine, args, futures)
                        cycles_done += 1
                        next_suspend_at += args.suspend_resume_interval

            if num_requests_added == num_requests_total:
                break
            # Relinquish control since there are no more requests to add at the moment. This allows the engine to run.
            await asyncio.sleep(0)

        # While we wait for the requests to complete, the engine runs in the background.
        results: List[DynamicInferenceRequestRecord] = await asyncio.gather(*futures)
    else:
        # Non-rank-0: match the suspend/resume cycles that rank 0 drives.
        for _ in range(num_suspend_resume_cycles):
            await engine.wait_until(EngineState.PAUSED)
            await engine.wait_until(EngineState.SUSPENDED)
            await engine.wait_until(EngineState.RESUMED)
            await engine.wait_until(EngineState.RUNNING)

    if dist.get_rank() == 0:
        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}
            throughputs = []

            for req in results:
                result_dict = {
                    "input_prompt": req.prompt,
                    "generated_text": req.generated_text.replace("\n", "\\n"),
                    "generated_tokens": req.generated_tokens,
                    "latency": req.latency,  # InferenceClient populates this field in the returned future.
                }
                if req.sampling_params.return_log_probs:
                    result_dict["logprobs"] = req.prompt_log_probs + req.generated_log_probs
                throughput = len(req.generated_tokens) / req.latency
                throughputs.append(throughput)
                if req.routing_indices is not None:
                    result_dict["routing_indices"] = req.routing_indices.tolist()
                                
                json_results[req.request_id] = result_dict
            throughput_dict = {"throughput": throughputs}
            if args.throughput_check_only:
                json_results = throughput_dict
            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp, indent=4)
        else:
            print("Results:")
            unique_prompt_map = defaultdict(list)
            for req in results:
                unique_prompt_map[req.prompt].append(req)
            for idx, (prompt_text, reqs) in enumerate(unique_prompt_map.items()):
                print(
                    f"%d/%d. prompt '%s' ... [%d] output '%s'."
                    % (
                        idx,
                        len(unique_prompt_map),
                        prompt_text.replace("\n", "\\n"),
                        len(reqs),
                        reqs[0].generated_text.replace("\n", "\\n"),
                    )
                )

        # Pause before stopping: STOP requires PAUSED or SUSPENDED state.
        client.pause_engines()

    await engine.wait_until(EngineState.PAUSED)

    if dist.get_rank() == 0:
        client.stop_engines()

    await engine.wait_until(EngineState.STOPPED)

    if dist.get_rank() == 0:
        client.shutdown_coordinator()
        client.stop()
    logging.info(f"Rank: {dist.get_rank()} stopped their engine instance successfully.")


if __name__ == "__main__":
    # enable inference mode in the very beginning as some fp8 optimizations
    # check for it.
    with torch.inference_mode():
        initialize_megatron(
            extra_args_provider=add_inference_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        args = get_args()
        tokenizer = get_tokenizer()

        # Sampling params.
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_log_probs=args.return_log_probs,
            num_tokens_to_generate=args.num_tokens_to_generate,
            termination_id=(
                args.termination_id if args.termination_id is not None else tokenizer.eod
            ),
        )

        model = get_model_for_inference()

        requests = build_requests(args, tokenizer, sampling_params)

        engine = get_dynamic_inference_engine(model=model)

        if dist.get_rank() == 0:
            setup_prefix = build_dynamic_engine_setup_prefix(args, model, engine.context, requests)
            print("~~~")
            print(setup_prefix)
            print("~~~")

        # Start Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        asyncio.run(main(engine, requests, args.inference_coordinator_port))

        # Stop Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStop()
