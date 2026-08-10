# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import copy
import gc
import json
import logging
import os
import time
import warnings
from collections import defaultdict
from typing import List

import torch
import torch.distributed as dist

from examples.inference.advanced.gpt_dynamic_inference import _assert_nested_close
from examples.inference.utils import Request, build_dynamic_engine_setup_prefix, build_requests
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequestRecord
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.moe.router_trace import get_moe_router_tracer, init_moe_router_tracer
from megatron.core.utils import configure_nvtx_profiling
from megatron.inference.utils import (
    add_inference_args,
    get_dynamic_inference_engine,
    get_model_for_inference,
)
from megatron.training import get_args, get_tokenizer, initialize_megatron
from megatron.training.arguments import parse_and_validate_args

# pylint: disable=line-too-long

logging.basicConfig(level=logging.INFO, force=True)


def add_async_sched_comparison_args(parser):
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="Async scheduling functional comparison")
    group.add_argument("--compare-async-sched-modes", action="store_true")
    group.add_argument("--async-sched-min-steps", type=int, default=1)
    group.add_argument("--async-sched-min-compactions", type=int, default=1)
    return parser


def _snapshot_results(results):
    return [
        {
            "prompt": result.prompt,
            "generated_text": result.generated_text,
            "generated_tokens": result.generated_tokens,
            "prompt_logprobs": result.prompt_log_probs,
            "generated_logprobs": result.generated_log_probs,
        }
        for result in results
    ]


async def suspend_resume_cycle(client, engine, args, futures):
    if not args.compare_async_sched_modes:
        await asyncio.gather(*futures)
        return await _suspend_resume(client, engine, args)

    async def wait_for_inflight_step():
        start_step = engine.context.step_count
        while not (
            any(not future.done() for future in futures) and engine.context.step_count > start_step
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_inflight_step(), timeout=60)
    assert any(not future.done() for future in futures), "suspend was not mid-flight"

    await _suspend_resume(client, engine, args)


async def _suspend_resume(client, engine, args):

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
    write_output: bool = True,
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
    num_suspend_resume_cycles = (
        len(requests) // args.suspend_resume_interval if args.suspend_resume_interval else 0
    )

    results = []

    # Create client and run example.
    if dist.get_rank() == 0:
        client = InferenceClient(
            dp_addr, deserialize=True
        )  # submits requests to the inference coordinator
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

                    if (
                        num_requests_added >= next_suspend_at
                        and cycles_done < num_suspend_resume_cycles
                    ):
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

                    if (
                        num_requests_added >= next_suspend_at
                        and cycles_done < num_suspend_resume_cycles
                    ):
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
        if args.output_path and write_output:
            json_results = {}
            throughputs = []

            for req in results if not args.compare_async_sched_modes else []:
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
    return results


if __name__ == "__main__":
    # enable inference mode in the very beginning as some fp8 optimizations
    # check for it.
    with torch.inference_mode():
        args = parse_and_validate_args(
            extra_args_provider=add_async_sched_comparison_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )
        initialize_megatron()
        configure_nvtx_profiling(True)

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

        if getattr(args, 'moe_routing_trace_path', None):
            rank = dist.get_rank()
            max_steps = getattr(args, 'moe_routing_trace_max_inference_steps', None) or 10**9
            init_moe_router_tracer(
                output_dir=args.moe_routing_trace_path,
                max_steps=max_steps,
                rank=rank,
                capture_hidden_states=getattr(
                    args, 'moe_routing_trace_capture_hidden_states', False
                ),
                capture_logits=getattr(args, 'moe_routing_trace_capture_logits', False),
                dump_router_weights=getattr(args, 'moe_routing_trace_dump_weights', False),
            )

        model = get_model_for_inference()

        tracer = get_moe_router_tracer()
        if tracer is not None:
            # When router replay is enabled, the in-pipeline recorder (RouterReplay/RoutingMetadata)
            # writes routing indices into a static buffer, and the text generation controller tees
            # that buffer into the tracer once per decode step. If router replay is not on,
            # use the forward hook method which allows for additionally saving hidden states.
            from megatron.core.utils import get_model_config

            if not get_model_config(model).moe_enable_routing_replay:
                tracer.register_hooks(model)

        requests = build_requests(args, tokenizer, sampling_params)
        if args.compare_async_sched_modes and len(requests) < 32:
            requests = [copy.deepcopy(requests[idx % len(requests)]) for idx in range(32)]

        # Start Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        legacy_outputs = None
        if args.compare_async_sched_modes:
            args.inference_dynamic_batching_async_sched_mode = "legacy"
            legacy_engine = get_dynamic_inference_engine(model=model)
            legacy_results = asyncio.run(
                main(
                    legacy_engine,
                    copy.deepcopy(requests),
                    args.inference_coordinator_port,
                    write_output=False,
                )
            )
            if dist.get_rank() == 0:
                legacy_outputs = _snapshot_results(legacy_results)
            delete_cuda_graphs()
            del legacy_engine, legacy_results
            gc.collect()
            torch.cuda.empty_cache()

        if args.compare_async_sched_modes:
            args.inference_dynamic_batching_async_sched_mode = "async"
        engine = get_dynamic_inference_engine(model=model)

        if dist.get_rank() == 0:
            setup_prefix = build_dynamic_engine_setup_prefix(args, model, engine.context, requests)
            print("~~~")
            print(setup_prefix)
            print("~~~")

        results = asyncio.run(
            main(engine, copy.deepcopy(requests), args.inference_coordinator_port)
        )

        if args.compare_async_sched_modes:
            counts = torch.tensor(
                [
                    engine.context.async_sched_step_count,
                    engine.context.async_sched_compaction_step_count,
                ],
                device="cuda",
            )
            dist.all_reduce(counts)
            assert counts[0] >= args.async_sched_min_steps
            assert counts[1] >= args.async_sched_min_compactions
            assert (
                engine.context.unified_memory_level
                == args.inference_dynamic_batching_unified_memory_level
            )
            assert engine.context.kv_cache_management_mode.value == args.rl_kv_cache_management_mode
            assert (
                engine.use_synchronous_zmq_collectives
                == args.inference_use_synchronous_zmq_collectives
            )
            assert engine.disable_ep_consensus == args.inference_disable_ep_consensus
            assert engine.context.prefix_caching_coordinator_policy.value == (
                args.inference_dynamic_batching_prefix_caching_coordinator_policy
            )
            if dist.get_rank() == 0:
                tolerance = 5e-3 if args.model_provider != "gpt" or args.fp8 else 1e-3
                _assert_nested_close(legacy_outputs, _snapshot_results(results), atol=tolerance)

        # Stop Nsight profiler.
        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStop()
