# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# pylint: disable=bad-builtin

import asyncio
import hashlib
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional

from megatron.training.arguments import parse_and_validate_args
import torch
import torch.distributed as dist

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from examples.inference.gpt.utils import (
    Request,
    build_dynamic_engine_setup_prefix,
    build_requests,
    get_curr_time,
    get_global_peak_memory_stats_bytes,
)
from megatron.core.inference.engines import DynamicInferenceEngine, EngineSuspendedError
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import configure_nvtx_profiling
from megatron.inference.utils import (
    add_inference_args,
    get_dynamic_inference_engine,
    get_model_for_inference,
)
from megatron.training import get_args, get_tokenizer, initialize_megatron

import megatron

torch.serialization.add_safe_globals([io.BytesIO])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunState])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunDiagnostic])


async def _suspend_resume_cycle(client, engine, args, futures):
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


async def _submit_requests(client, engine, requests, args):
    """Rank-0 request submission loop. Returns list of completed futures."""
    base_arrival_time = time.time_ns() / 10**9
    for request in requests:
        request.time_arrival = request.time_offset + base_arrival_time

    futures = []
    num_requests_total = len(requests)
    num_requests_added = 0
    num_suspend_resume_cycles = (
        len(requests) // args.suspend_resume_interval if args.suspend_resume_interval else 0
    )
    next_suspend_at = args.suspend_resume_interval or 0
    cycles_done = 0

    batch_ranges = None
    if args.drain_between_batches and args.batch_boundaries:
        boundaries = [int(x) for x in args.batch_boundaries.split(",")]
        batch_ranges = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else num_requests_total
            batch_ranges.append((start, end))

    def _maybe_suspend():
        nonlocal cycles_done, next_suspend_at
        if num_requests_added >= next_suspend_at and cycles_done < num_suspend_resume_cycles:
            cycles_done += 1
            next_suspend_at += args.suspend_resume_interval
            return True
        return False

    if batch_ranges is not None:
        for batch_start, batch_end in batch_ranges:
            batch_futures = []
            while num_requests_added < batch_end:
                request = requests[num_requests_added]
                batch_futures.append(
                    client.add_request(request.prompt_text, request.sampling_params)
                )
                num_requests_added += 1
            futures.extend(batch_futures)
            await asyncio.gather(*batch_futures)
    else:
        while True:
            current_time = time.time_ns() / 10**9
            if args.incoming_requests_per_step is None:
                while (
                    num_requests_added < num_requests_total
                    and requests[num_requests_added].time_arrival <= current_time
                ):
                    request = requests[num_requests_added]
                    futures.append(
                        client.add_request(request.prompt_text, request.sampling_params)
                    )
                    num_requests_added += 1
                    if _maybe_suspend():
                        await _suspend_resume_cycle(client, engine, args, futures)
            else:
                for _ in range(
                    min(args.incoming_requests_per_step, num_requests_total - num_requests_added)
                ):
                    request = requests[num_requests_added]
                    futures.append(
                        client.add_request(request.prompt_text, request.sampling_params)
                    )
                    num_requests_added += 1
                    if _maybe_suspend():
                        await _suspend_resume_cycle(client, engine, args, futures)

            if num_requests_added == num_requests_total:
                break
            await asyncio.sleep(0)

    return await asyncio.gather(*futures)


async def run_inference(
    engine: DynamicInferenceEngine,
    requests: List[Request],
) -> Optional[List[DynamicInferenceRequest]]:
    """Start the coordinator, submit requests, and return finished requests.

    Returns the list of DynamicInferenceRequest on rank 0, None on other ranks.
    """
    args = get_args()

    dp_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=args.inference_coordinator_port,
        launch_inference_coordinator=True,
        coordinator_schedule_output_path=args.coordinator_schedule_output_path,
    )

    num_suspend_resume_cycles = (
        len(requests) // args.suspend_resume_interval if args.suspend_resume_interval else 0
    )

    results = None
    if dist.get_rank() == 0:
        client = InferenceClient(dp_addr, deserialize=True)
        client.start()
        results = await _submit_requests(client, engine, requests, args)
    else:
        for _ in range(num_suspend_resume_cycles):
            await engine.wait_until(EngineState.PAUSED)
            await engine.wait_until(EngineState.SUSPENDED)
            await engine.wait_until(EngineState.RESUMED)
            await engine.wait_until(EngineState.RUNNING)

    if dist.get_rank() == 0:
        client.pause_engines()
    await engine.wait_until(EngineState.PAUSED)
    if dist.get_rank() == 0:
        client.stop_engines()
    await engine.wait_until(EngineState.STOPPED)
    if dist.get_rank() == 0:
        client.shutdown_coordinator()
        client.stop()

    return results


def run_inference_no_coordinator(
    engine: DynamicInferenceEngine,
    requests: List[Request],
) -> List[DynamicInferenceRequest]:
    """Direct-stepping inference loop without the coordinator.

    All ranks add requests locally and step the engine in lockstep. This path
    does not use EP consensus and may hang with expert parallelism when
    speculative decoding causes TP groups to diverge. Prefer the coordinator
    path (the default) for production use.
    """
    args = get_args()

    base_arrival_time = get_curr_time()
    for request in requests:
        request.time_arrival = request.time_offset + base_arrival_time

    num_requests_total = len(requests)
    num_requests_added = 0
    finished_records = []

    batch_ranges = None
    if args.drain_between_batches and args.batch_boundaries:
        boundaries = [int(x) for x in args.batch_boundaries.split(",")]
        batch_ranges = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else num_requests_total
            batch_ranges.append((start, end))

    def _add_request():
        nonlocal num_requests_added
        req = requests[num_requests_added]
        engine.add_request(num_requests_added, req.prompt_text, req.sampling_params)
        num_requests_added += 1

    def _collect_finished(result):
        if result is None:
            return
        for record in result.get("finished_request_records", []):
            finished_records.append(record.merge())

    def _step_loop(stop_condition):
        attempted_step_count = 0
        while not stop_condition():
            try:
                result = engine.step_modern()
            except EngineSuspendedError:
                result = None
            attempted_step_count += 1

            if args.suspend_resume_interval is not None:
                if attempted_step_count % args.suspend_resume_interval == 0:
                    engine.suspend()
                if (
                    attempted_step_count > 0
                    and (attempted_step_count - args.suspend_resume_interval // 2)
                    % args.suspend_resume_interval
                    == 0
                ):
                    engine.resume()

            if result is None or isinstance(result, EngineSuspendedError):
                continue
            _collect_finished(result)

    if batch_ranges is not None:
        for _, (batch_start, batch_end) in enumerate(batch_ranges):
            while num_requests_added < batch_end:
                _add_request()
            _step_loop(lambda: not engine.has_unfinished_requests())
    else:
        while True:
            add_start = get_curr_time()
            if args.incoming_requests_per_step is None:
                while num_requests_added < num_requests_total:
                    if requests[num_requests_added].time_arrival > add_start:
                        break
                    _add_request()
            else:
                for _ in range(
                    min(args.incoming_requests_per_step, num_requests_total - num_requests_added)
                ):
                    _add_request()

            try:
                result = engine.step_modern()
            except EngineSuspendedError:
                result = None
            if result is not None:
                _collect_finished(result)
            if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
                break

    engine.resume()
    return finished_records


def _print_and_save_results(results, engine, total_time, setup_prefix):
    """Print unique prompts/outputs and optionally write JSON results (rank 0 only)."""
    if dist.get_rank() != 0:
        return

    args = get_args()
    finished = sorted(results, key=lambda r: r.request_id)
    total_output_tokens = sum(len(r.generated_tokens) for r in finished)

    def escape_str(s):
        return s.replace("\n", "\\n") if s else "--"

    print("~~~~ Unique prompts + outputs. ~~~~")

    unique_prompt_map = defaultdict(list)
    for req in finished:
        unique_prompt_map[req.prompt].append(req)

    text_hashes = []
    for unique_idx, (prompt_text, reqs) in enumerate(unique_prompt_map.items()):
        prompt_len = len(reqs[0].prompt_tokens)
        print(
            f"\n{unique_idx+1}/{len(unique_prompt_map)}"
            f"[n {len(reqs)}, l {prompt_len}] {escape_str(prompt_text)}"
        )

        output_map = defaultdict(list)
        for req in reqs:
            output_map[req.generated_text].append(req)

        for output_text, output_reqs in output_map.items():
            evicted = any(
                e.type.name == "EVICT" for req in output_reqs for e in (req.events or [])
            )
            if output_text is not None:
                o_hash = hashlib.sha256(
                    (prompt_text + output_text).encode()
                ).hexdigest()[:6]
                o_len = len(output_reqs[0].generated_tokens)
                escaped_output_text = escape_str(output_text)
            else:
                o_hash = "--"
                o_len = 0
                escaped_output_text = "--"
            print(
                f"  >>>> [n {len(output_reqs)}, {o_len} tokens, hash {o_hash}"
                f"{', <evicted>' if evicted else ''}] {escaped_output_text}"
            )
            text_hashes.append(o_hash)

    if args.output_path:
        json_results = {}
        for i, req in enumerate(finished):
            if i % args.output_every_n_results == 0 or i == len(finished) - 1:
                result_dict = {
                    "input_prompt": req.prompt,
                    "generated_text": req.generated_text,
                    "generated_tokens": req.generated_tokens,
                    "latency": req.latency,
                    "ttft": req.ttft,
                    "step_count": engine.context.step_count,
                    "top_n_logprobs": req.generated_top_n_logprobs,
                    "prompt_top_n_logprobs": req.prompt_top_n_logprobs,
                }
                if req.sampling_params.return_log_probs:
                    result_dict["prompt_logprobs"] = req.prompt_log_probs
                    result_dict["generated_logprobs"] = req.generated_log_probs
                    result_dict["logprobs"] = (
                        (req.prompt_log_probs or []) + (req.generated_log_probs or [])
                    )
                if args.output_request_events:
                    result_dict["events"] = [e.serialize() for e in (req.events or [])]
                json_results[req.request_id] = result_dict

        peak_mem_stats = get_global_peak_memory_stats_bytes()
        throughput = total_output_tokens / total_time
        if args.record_throughput:
            json_results["throughput"] = [throughput]
        json_results.update(peak_mem_stats)
        json_results["lifetime_prefill_token_count"] = (
            engine.context.lifetime_prefill_token_count
        )

        print(f' Saving results to {args.output_path}')
        with open(args.output_path, "w") as fp:
            json.dump(json_results, fp, indent=1)

    stats = torch.cuda.memory_stats()
    throughput = total_output_tokens / total_time
    peak_alloc_gb = stats["allocated_bytes.all.peak"] / 1024**3
    peak_resvd_gb = stats["reserved_bytes.all.peak"] / 1024**3
    capture_str = f"{engine.capture_stats['time']:.2f} sec" if engine.capture_stats else "--"
    print("~~~")
    print(
        f"{setup_prefix} … " f"throughput: {throughput:.3f} tok/s … ",
        f"total time: {total_time:.3f}s … "
        f"mem {peak_alloc_gb:.1f}/{peak_resvd_gb:.1f} GB … "
        f"steps: {engine.context.step_count:d} … "
        f"capture {capture_str}",
    )
    print("~~~")


def _add_script_args(parser):
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title='dynamic inference script')
    group.add_argument(
        "--no-coordinator",
        action="store_true",
        default=False,
        help="Use the legacy direct-stepping loop instead of the coordinator. "
        "The coordinator (default) is required for correct EP consensus with "
        "speculative decoding.",
    )
    return parser


def main():
    """Run dynamic inference."""
    with torch.inference_mode():
        args = parse_and_validate_args(
            extra_args_provider=_add_script_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )
        initialize_megatron()

        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStart()

        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        logging.basicConfig(level=level, force=True)

        configure_nvtx_profiling(True)

        tokenizer = get_tokenizer()
        torch.cuda.reset_peak_memory_stats()

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            skip_prompt_log_probs=args.skip_prompt_log_probs,
            return_log_probs=args.return_log_probs,
            num_tokens_to_generate=args.num_tokens_to_generate,
            termination_id=(
                args.termination_id if args.termination_id is not None else tokenizer.eod
            ),
            top_n_logprobs=args.top_n_logprobs,
            stop_words=args.stop_words,
        )

        model = get_model_for_inference()
        requests = build_requests(args, tokenizer, sampling_params)
        engine = get_dynamic_inference_engine(model=model)

        setup_prefix = build_dynamic_engine_setup_prefix(args, model, engine.context, requests)
        print("~~~")
        print(setup_prefix)
        print("~~~")

        torch.cuda.reset_peak_memory_stats()
        t = get_curr_time()
        if args.no_coordinator:
            results = run_inference_no_coordinator(engine, requests)
        else:
            results = asyncio.run(run_inference(engine, requests))
        torch.cuda.synchronize()
        total_time = get_curr_time() - t

        _print_and_save_results(results, engine, total_time, setup_prefix)

        if os.environ.get("NSIGHT_PREFIX"):
            torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
