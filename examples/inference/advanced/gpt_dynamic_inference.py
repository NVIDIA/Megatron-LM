# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# pylint: disable=bad-builtin

import copy
import gc
import hashlib
import io
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from megatron.training.arguments import parse_and_validate_args

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from examples.inference.utils import (
    Request,
    build_dynamic_engine_setup_prefix,
    build_requests,
    get_curr_time,
    get_global_peak_memory_stats_bytes,
)
from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine, EngineSuspendedError
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
import logging

import megatron
from megatron.core.utils import configure_nvtx_profiling
from megatron.training import get_args, get_tokenizer, initialize_megatron

torch.serialization.add_safe_globals([io.BytesIO])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunState])
torch.serialization.add_safe_globals([megatron.core.rerun_state_machine.RerunDiagnostic])


def add_async_sched_comparison_args(parser):
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="Async scheduling functional comparison")
    group.add_argument("--compare-async-sched-modes", action="store_true")
    group.add_argument("--async-sched-min-steps", type=int, default=1)
    group.add_argument("--async-sched-min-compactions", type=int, default=1)
    group.add_argument("--async-sched-require-prefix-cache-hit", action="store_true")
    group.add_argument("--async-sched-require-cuda-graph", action="store_true")
    group.add_argument("--async-sched-require-early-stop", action="store_true")
    group.add_argument("--async-sched-add-bos", action="store_true")
    group.add_argument("--async-sched-use-total-length", action="store_true")
    group.add_argument("--async-sched-mixed-stop-retention", action="store_true")
    return parser


def _stagger_generation_lengths(requests: List[Request]) -> None:
    base_length = requests[0].sampling_params.num_tokens_to_generate
    lengths = (base_length, max(2, base_length // 4), max(3, base_length // 2))
    for request_idx, request in enumerate(requests):
        request.sampling_params.num_tokens_to_generate = lengths[request_idx % len(lengths)]


def _prompt_length(request: Request) -> int:
    return len(request.prompt_tokens) + int(request.sampling_params.add_BOS)


def _generation_budget(request: Request) -> int:
    params = request.sampling_params
    if params.num_tokens_to_generate is not None:
        return params.num_tokens_to_generate
    return params.num_tokens_total - _prompt_length(request)


def _snapshot_request_outputs(requests: List[Request]) -> list[dict]:
    return [
        {
            "prompt_tokens": request.prompt_tokens,
            "output_tokens": request.output_tokens,
            "output_text": request.output_text,
            "prompt_logprobs": getattr(request, "prompt_log_probs", None),
            "generated_logprobs": getattr(request, "generated_log_probs", None),
            "prompt_top_n_logprobs": getattr(request, "prompt_top_n_logprobs", None),
            "generated_top_n_logprobs": getattr(request, "generated_top_n_logprobs", None),
        }
        for request in requests
    ]


def _assert_nested_close(expected, actual, *, atol: float, path: str = "output") -> None:
    assert type(expected) is type(actual), f"{path}: {type(expected)} != {type(actual)}"
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), f"{path}: keys differ"
        for key in expected:
            _assert_nested_close(expected[key], actual[key], atol=atol, path=f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), f"{path}: length {len(expected)} != {len(actual)}"
        for idx, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_nested_close(expected_item, actual_item, atol=atol, path=f"{path}[{idx}]")
    elif isinstance(expected, float):
        assert math.isclose(
            expected, actual, rel_tol=0.0, abs_tol=atol
        ), f"{path}: {expected} != {actual} (atol={atol})"
    else:
        assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def _build_engine(model, tokenizer, args, requests, mode: AsyncScheduleMode):
    inference_config = get_inference_config_from_model_and_args(model, args)
    inference_config.async_sched_mode = mode
    inference_config.max_sequence_length = max(
        _prompt_length(request) + _generation_budget(request) for request in requests
    )
    context = DynamicInferenceContext(model.config, inference_config)
    if not args.enable_chunked_prefill:
        invalid_lengths = {
            idx: len(request.prompt_tokens)
            for idx, request in enumerate(requests)
            if len(request.prompt_tokens) > context.max_tokens
        }
        assert not invalid_lengths, f"prompts longer than context.max_tokens: {invalid_lengths}"
    controller = TextGenerationController(GPTInferenceWrapper(model, context), tokenizer)
    return DynamicInferenceEngine(controller, context)


def run_inference(
    requests: List[Request],
    engine: DynamicInferenceEngine,
    sampling_params: Optional[SamplingParams] = None,
) -> List[Dict[str, float]]:
    """Add requests to engine and generate tokens.

    Args:
        requests (List[Request]): Requests that are to be added and processed.
        engine (DynamicInferenceEngine): Inference engine that manages generating tokens.
        sampling_params (SamplingParams): Deprecated as of megatron-core 0.16.

    Return:
        A dictionary of step times with `prefill` and `decode` keys.
    """

    if sampling_params is not None and torch.distributed.get_rank() == 0:
        warnings.warn(
            "The `sampling_params` argument is deprecated. "
            "Sampling parameters are specified per request.",
            DeprecationWarning,
        )

    args = get_args()

    # Parse batch boundaries for batch-drain mode.
    batch_ranges = None
    if args.drain_between_batches and args.batch_boundaries:
        boundaries = [int(x) for x in args.batch_boundaries.split(",")]
        num_requests_total = len(requests)
        batch_ranges = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else num_requests_total
            batch_ranges.append((start, end))

    # Initialize request arrival times.
    base_arrival_time = get_curr_time()
    for request in requests:
        request.time_arrival = request.time_offset + base_arrival_time

    # Add and process requests.
    num_requests_total = len(requests)
    num_requests_added = 0
    num_requests_finished = 0
    step_times = {"prefill": [], "decode": []}
    add_times = []
    output_times = []
    tbar = tqdm(total=num_requests_total)
    total_output_tokens = 0
    attempted_step_count = 0
    if args.cuda_graph_impl == "local":
        cuda_graph_request_count_map = {}
    else:
        cuda_graph_request_count_map = None

    def _add_request():
        """Add request to engine.

        *Note: Using `prompt_text` instead of `prompt_tokens` for fair comparison.
        """
        nonlocal num_requests_added
        _request = requests[num_requests_added]
        engine.add_request(num_requests_added, _request.prompt_text, _request.sampling_params)
        _request.time_start = get_curr_time(do_broadcast=False)
        _request.state = "started"
        num_requests_added += 1
        tbar.update(1)

    def _process_step_result(result):
        """Process a single engine step result, updating bookkeeping state."""
        nonlocal total_output_tokens, num_requests_finished

        decode_only = engine.decode_only
        is_decode_only = (
            decode_only.launched if decode_only.launched is not None else decode_only.consumed
        )

        # Record cuda_graph_request_count.
        cuda_graph_request_count = result["cuda_graph_request_count"]
        if args.cuda_graph_impl == "local" and cuda_graph_request_count is not None:
            cuda_graph_request_count_map[cuda_graph_request_count] = (
                cuda_graph_request_count_map.get(cuda_graph_request_count, 0) + 1
            )

        # Update requests.
        active_request_ids = result["active_request_ids"]
        finished_request_records = result["finished_request_records"]
        step_time = result["step_time"]
        if len(active_request_ids) > 0 or len(finished_request_records) > 0:
            if is_decode_only:
                step_times["decode"].append(step_time)
            else:
                step_times["prefill"].append(step_time)

            # Append output tokens.
            output_start = get_curr_time(do_broadcast=False)
            for finished_request_record in finished_request_records:

                finished_request = finished_request_record.merge()

                # Update local request object.
                request = requests[finished_request.request_id]
                request.time_end = get_curr_time(do_broadcast=False)
                request.state = "finished"
                request.request_id = finished_request.request_id
                request.events = finished_request.events

                request.ttft = finished_request.ttft

                # Update prompt, in case engine has been suspended and resumed.
                request.prompt_tokens = finished_request.prompt_tokens.tolist()
                request.prompt_text = finished_request.prompt

                # Get output tokens and text.
                request.output_tokens = finished_request.generated_tokens
                request.output_text = finished_request.generated_text
                total_output_tokens += len(request.output_tokens)

                # Log probs.
                if finished_request.sampling_params.return_log_probs:
                    if not finished_request.prompt_log_probs:
                        finished_request.prompt_log_probs = []
                    request.prompt_log_probs = finished_request.prompt_log_probs
                    request.generated_log_probs = finished_request.generated_log_probs
                    request.logprobs = (
                        finished_request.prompt_log_probs + finished_request.generated_log_probs
                    )
                if finished_request.sampling_params.top_n_logprobs > 0:
                    request.generated_top_n_logprobs = finished_request.generated_top_n_logprobs
                if not finished_request.sampling_params.skip_prompt_log_probs:
                    request.prompt_top_n_logprobs = finished_request.prompt_top_n_logprobs
                num_requests_finished += 1
            output_times.append(get_curr_time(do_broadcast=False) - output_start)

    if batch_ranges is not None:
        # Batch-drain mode: add all requests in a batch, drain, then next batch.
        for batch_idx, (batch_start, batch_end) in enumerate(batch_ranges):
            # Add all requests in current batch.
            add_start = get_curr_time(do_broadcast=False)
            while num_requests_added < batch_end:
                _add_request()
            add_times.append(get_curr_time(do_broadcast=False) - add_start)

            # Step until all active requests finish (drain).
            while engine.has_unfinished_requests():
                try:
                    result = engine.step_modern()
                except EngineSuspendedError as e:
                    result = e
                attempted_step_count += 1

                if isinstance(result, EngineSuspendedError):
                    continue

                _process_step_result(result)
    else:
        # Original mode: add requests per step based on arrival time or count.
        while True:
            # Add requests.
            add_start = get_curr_time(do_broadcast=False)
            if args.incoming_requests_per_step is None:
                # Add requests with 'earlier' arrival time.
                while num_requests_added < num_requests_total:
                    if requests[num_requests_added].time_arrival > add_start:
                        break
                    _add_request()
            else:
                # Add deterministic number of requests (generally used for debugging).
                for i in range(
                    min(args.incoming_requests_per_step, num_requests_total - num_requests_added)
                ):
                    _add_request()
            add_times.append(get_curr_time(do_broadcast=False) - add_start)

            # Step inference engine (i.e., generate a token for each active request).
            # The engine reports the consumed and launched decode-only states after scheduling.
            try:
                result = engine.step_modern()
            except EngineSuspendedError as e:
                result = e
                pass  # ignore error in order to call 'engine.resume()' below.
            attempted_step_count += 1

            # Test suspending and resuming engine.
            if args.suspend_resume_interval is not None:

                # Suspend.
                if attempted_step_count % args.suspend_resume_interval == 0:
                    print(
                        "**** step %d/%d ... suspend."
                        % (engine.context.step_count, attempted_step_count)
                    )
                    engine.suspend()

                # Resume, 0+ attempted steps later.
                if (
                    attempted_step_count > 0
                    and (attempted_step_count - args.suspend_resume_interval // 2)
                    % args.suspend_resume_interval
                    == 0
                ):
                    print(
                        "**** step %d/%d ... resume."
                        % (engine.context.step_count, attempted_step_count)
                    )
                    engine.resume()

            # If engine suspended, continue to next iter.
            if isinstance(result, EngineSuspendedError):
                continue

            _process_step_result(result)

            # Check if all requests are finished.
            if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
                break

    # Resume engine (NOOP if not suspended).
    engine.resume()

    return {
        "step_times": step_times,
        "add_times": add_times,
        "output_times": output_times,
        "total_output_tokens": total_output_tokens,
        "cuda_graph_request_count_map": cuda_graph_request_count_map,
    }


@torch.inference_mode()
def main():
    """Run dynamic inference."""
    # Initialize Megatron.
    args = parse_and_validate_args(
        extra_args_provider=add_async_sched_comparison_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )
    initialize_megatron()

    # Start Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(level=level, force=True)

    configure_nvtx_profiling(True)

    # Build tokenizer
    tokenizer = build_tokenizer(args)

    # Reset peak memory stats so functional tests measure this run and not
    # whatever happened earlier during initialization.
    torch.cuda.reset_peak_memory_stats()

    # Sampling params.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        skip_prompt_log_probs=args.skip_prompt_log_probs,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=args.termination_id if args.termination_id is not None else tokenizer.eod,
        top_n_logprobs=args.top_n_logprobs,
        stop_words=args.stop_words,
        add_BOS=args.async_sched_add_bos,
    )

    model = get_model_for_inference()

    requests = build_requests(args, tokenizer, sampling_params)
    if args.compare_async_sched_modes and len(requests) < 8:
        requests = [copy.deepcopy(requests[idx % len(requests)]) for idx in range(8)]
    legacy_outputs = None
    if args.compare_async_sched_modes:
        _stagger_generation_lengths(requests)
        if args.async_sched_mixed_stop_retention:
            for request_idx, request in enumerate(requests):
                request.sampling_params.detokenize_stop_sequence = bool(request_idx % 2)
        if args.async_sched_use_total_length:
            for request in requests:
                request.sampling_params.num_tokens_total = (
                    _prompt_length(request) + request.sampling_params.num_tokens_to_generate
                )
                request.sampling_params.num_tokens_to_generate = None
        legacy_requests = copy.deepcopy(requests)
        legacy_engine = _build_engine(
            model, tokenizer, args, legacy_requests, AsyncScheduleMode.LEGACY
        )
        run_inference(legacy_requests, legacy_engine)
        assert all(request.state == "finished" for request in legacy_requests)
        legacy_outputs = _snapshot_request_outputs(legacy_requests)
        delete_cuda_graphs()
        del legacy_engine, legacy_requests
        gc.collect()
        torch.cuda.empty_cache()

    engine_mode = (
        AsyncScheduleMode.ASYNC
        if args.compare_async_sched_modes
        else AsyncScheduleMode(args.inference_dynamic_batching_async_sched_mode)
    )
    engine = _build_engine(model, tokenizer, args, requests, engine_mode)
    context = engine.context

    setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
    print("~~~")
    print(setup_prefix)
    print("~~~")

    # Run and time test, optionally `args.inference_repeat_n` times.
    throughputs = []
    for _ in range(args.inference_repeat_n):

        # Reset engine.
        engine.reset()

        torch.cuda.reset_peak_memory_stats()

        # Trial.
        t = get_curr_time()
        result = run_inference(requests, engine)
        step_times = result["step_times"]
        add_times = result["add_times"]
        output_times = result["output_times"]
        total_output_tokens = result["total_output_tokens"]
        torch.cuda.synchronize()
        total_time = get_curr_time() - t
        stats = torch.cuda.memory_stats()
        throughput = total_output_tokens / total_time
        throughputs.append(throughput)

    # Validate all requests finished.
    for request in requests:
        assert request.state == "finished", f"request.state == '{request.state}' != 'finished'."

    if args.compare_async_sched_modes:
        async_outputs = _snapshot_request_outputs(requests)
        tolerance = 5e-3 if args.model_provider != "gpt" or args.fp8 else 1e-3
        _assert_nested_close(legacy_outputs, async_outputs, atol=tolerance)
        assert context.async_sched_step_count >= args.async_sched_min_steps
        assert context.async_sched_compaction_step_count >= args.async_sched_min_compactions, (
            f"async compactions {context.async_sched_compaction_step_count} "
            f"< {args.async_sched_min_compactions}"
        )
        assert (
            len(requests) > context.max_requests
        ), f"stress requires queued requests: {len(requests)} <= max_requests {context.max_requests}"
        if args.async_sched_require_cuda_graph:
            assert result["cuda_graph_request_count_map"], "no CUDA graph replay observed"
        if args.async_sched_require_prefix_cache_hit:
            assert engine._prefix_cache_hits > 0, "no prefix-cache hit observed"
            assert engine._prefill_tokens_skipped > 0, "prefix caching skipped no prefill tokens"
        if args.async_sched_require_early_stop:
            assert any(
                len(request.output_tokens) < _generation_budget(request) for request in requests
            ), "configured stop condition never terminated a request early"

    peak_mem_stats = get_global_peak_memory_stats_bytes()

    # Print unique prompts + outputs.
    if torch.distributed.get_rank() == 0:

        def escape_str(s):
            return s.replace("\n", "\\n")

        print("~~~~ Unique prompts + outputs. ~~~~")

        # Map requests by their prompt.
        unique_prompt_map = defaultdict(list)
        for request_idx, request in enumerate(requests):
            unique_prompt_map[request.prompt_text].append(request_idx)

        # Print unique prompts + outputs.
        text_hashes = []
        for unique_idx, (prompt_text, request_idxs) in enumerate(unique_prompt_map.items()):

            # ---- Prompt summary line ----
            prompt_len = len(requests[request_idxs[0]].prompt_tokens)
            escaped_prompt_text = escape_str(prompt_text)
            print(
                f"\n{unique_idx+1}/{len(unique_prompt_map)}"
                f"[n {len(request_idxs)}, l {prompt_len}] {escaped_prompt_text}"
            )

            # ---- Group all outputs for this prompt ----
            output_map = defaultdict(list)
            for idx in request_idxs:
                req = requests[idx]
                output_map[req.output_text].append(idx)

            # ---- Print each unique output ----
            for output_text, output_request_idxs in output_map.items():
                evicted = False
                for idx in output_request_idxs:
                    for event in requests[idx].events:
                        if event.type.name == "EVICT":
                            evicted = True
                            break
                if output_text is not None:
                    # Use hash of prompt + generated text in case engine was
                    # suspended and resumed, which misaligns boundary between
                    # prompt and generated tokens.
                    o_hash = hashlib.sha256((prompt_text + output_text).encode()).hexdigest()[:6]
                    o_len = len(requests[output_request_idxs[0]].output_tokens)
                    escaped_output_text = escape_str(output_text)
                else:
                    o_hash = "--"
                    o_len = 0
                    escaped_output_text = "--"
                print(
                    f"  >>>> [n {len(output_request_idxs)}, {o_len} tokens, hash {o_hash}"
                    f"{', <evicted>' if evicted else ''}] {escaped_output_text}"
                )
                text_hashes.append(o_hash)

        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}

            # Write every 'n' requests, plus the final request.
            for i, req in enumerate(requests if not args.compare_async_sched_modes else []):
                if i % args.output_every_n_results == 0 or i == len(requests) - 1:
                    print(f' Attributes of request {i}: {req.__dict__}')
                    result_dict = {
                        "input_prompt": req.prompt_text,
                        "generated_text": req.output_text,
                        "generated_tokens": req.output_tokens,
                        "latency": req.time_end - req.time_start,
                        "ttft": req.ttft,  # Time-to-first-token in seconds
                        "cuda_graph_request_count_map": result["cuda_graph_request_count_map"],
                        "step_count": engine.context.step_count,
                        "top_n_logprobs": getattr(req, 'generated_top_n_logprobs', None),
                        "prompt_top_n_logprobs": getattr(req, 'prompt_top_n_logprobs', None),
                    }
                    if req.sampling_params.return_log_probs:
                        result_dict["prompt_logprobs"] = getattr(req, 'prompt_log_probs', None)
                        result_dict["generated_logprobs"] = getattr(
                            req, 'generated_log_probs', None
                        )
                        result_dict["logprobs"] = getattr(req, 'logprobs', None)
                    if args.output_request_events:
                        result_dict["events"] = [e.serialize() for e in req.events]
                    json_results[req.request_id] = result_dict

            # Track system-level throughput as a test / debug metric
            if args.record_throughput:
                json_results["throughput"] = throughputs
            # Attach peak memory metrics; the functional test only validates these
            # if the fields exist in the golden values.
            json_results.update(peak_mem_stats)
            json_results["lifetime_prefill_token_count"] = (
                engine.context.lifetime_prefill_token_count
            )
            json_results["async_sched_step_count"] = context.async_sched_step_count
            json_results["async_sched_compaction_step_count"] = (
                context.async_sched_compaction_step_count
            )

            print(f' Saving results to {args.output_path}')
            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp, indent=1)

        # Timing results.
        stats = torch.cuda.memory_stats()
        throughput = total_output_tokens / total_time
        print("~~~")
        peak_alloc_gb = stats["allocated_bytes.all.peak"] / 1024**3
        peak_resvd_gb = stats["reserved_bytes.all.peak"] / 1024**3

        p_times = step_times["prefill"]
        d_times = step_times["decode"]

        p_total = sum(p_times)
        d_total = sum(d_times)

        p_count = len(p_times)
        d_count = len(d_times)

        p_mean = p_total / p_count
        d_mean = d_total / d_count if d_count != 0 else 0.0

        # Commented out for now as the step/add/output times are not calculated correctly.
        # print(
        #     f"{setup_prefix} … "
        #     f"mem {peak_alloc_gb:.1f}/{peak_resvd_gb:.1f} GB … "
        #     f"total time: {step_total:.3f}s … "
        #     f"step time: total {step_total:.3f}s "
        #     f"[ p {p_total:.3f}s, d {d_total:.3f}s ], "
        #     f"mean [ p {p_mean:.3f}s, d {d_mean:.3f}s ], "
        #     f"count [ p {p_count}, d {d_count} ]."
        # )
        capture_str = f"{engine.capture_stats['time']:.2f} sec" if engine.capture_stats else "--"
        print(
            f"{setup_prefix} … " f"throughput: {throughput:.3f} tok/s … ",
            f"total time: {total_time:.3f}s … "
            f"mem {peak_alloc_gb:.1f} allocated/{peak_resvd_gb:.1f} reserved GB … "
            f"steps: {engine.context.step_count:d} … "
            f"capture {capture_str}",
        )
        print("~~~")

    # Stop Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
