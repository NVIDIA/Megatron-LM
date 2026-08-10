# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# pylint: disable=bad-builtin,protected-access

import copy
import gc
import hashlib
import io
import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import replace
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
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine, EngineSuspendedError
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
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

PREFIX_CACHE_LOGPROB_P95_ATOL = 0.048790164169432  # A 5% probability ratio in log space.
PREFIX_CACHE_LOGPROB_MAX_ATOL = 0.182321556793955  # A 20% probability ratio in log space.


def add_runner_args(parser):
    """Add dynamic-inference and prefix-cache stress arguments."""
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title="Prefix-cache stress")
    group.add_argument("--prefix-cache-compare", action="store_true")
    group.add_argument("--prefix-cache-stress-groups", type=int, default=0)
    group.add_argument("--prefix-cache-stress-copies", type=int, default=2)
    group.add_argument("--prefix-cache-stress-prompt-tokens", type=int, default=512)
    group.add_argument("--prefix-cache-stress-staged", action="store_true")
    return parser


def build_prefix_cache_stress_requests(args, tokenizer, sampling_params, requests):
    """Build repeated-prefix requests that exceed the configured KV-cache capacity."""
    if not args.prefix_cache_compare:
        return requests
    if args.inference_repeat_n < 3:
        raise ValueError("--prefix-cache-compare requires --inference-repeat-n >= 3")
    if args.prefix_cache_stress_groups < 2 or args.prefix_cache_stress_copies < 2:
        raise ValueError("prefix-cache stress requires at least two groups and two copies")
    if args.prefix_cache_stress_prompt_tokens < 1:
        raise ValueError("--prefix-cache-stress-prompt-tokens must be positive")
    if args.prefix_cache_stress_staged:
        args.incoming_requests_per_step = 1

    stress_requests = []
    for group_idx in range(args.prefix_cache_stress_groups):
        marker = f"prefix cache pressure group {group_idx:04d}; deterministic shared text. "
        prompt = marker
        while len(tokenizer.tokenize(prompt)) < args.prefix_cache_stress_prompt_tokens:
            prompt += marker
        for _ in range(args.prefix_cache_stress_copies):
            stress_requests.append(Request(prompt, -1, tokenizer, sampling_params))
    return stress_requests


def _collect_nested_numeric_pairs(reference, actual, label, pairs):
    """Collect corresponding floating-point values while checking container structure."""
    if isinstance(reference, dict):
        assert isinstance(actual, dict) and tuple(reference) == tuple(
            actual
        ), f"{label}: keys differ"
        for key in reference:
            _collect_nested_numeric_pairs(reference[key], actual[key], f"{label}.{key}", pairs)
    elif isinstance(reference, (list, tuple)):
        assert isinstance(actual, (list, tuple)), f"{label}: container type differs"
        assert len(reference) == len(actual), f"{label}: lengths differ"
        for idx, (ref_item, actual_item) in enumerate(zip(reference, actual)):
            _collect_nested_numeric_pairs(ref_item, actual_item, f"{label}[{idx}]", pairs)
    elif isinstance(reference, float):
        assert isinstance(actual, float), f"{label}: expected a float"
        pairs.append((label, reference, actual))
    else:
        assert reference == actual, f"{label}: {reference!r} != {actual!r}"


def _assert_logprob_parity(pairs):
    """Check cache-off/cache-on log-probability parity with bounded BF16 drift."""
    assert pairs, "no logprobs to compare"
    values = torch.tensor([pair[1:] for pair in pairs], dtype=torch.float64)
    assert torch.isfinite(values).all(), "non-finite logprob"
    differences = (values[:, 0] - values[:, 1]).abs()
    max_difference, worst = differences.max(dim=0)
    p95 = torch.quantile(differences, 0.95)
    stats = (
        f"count={len(pairs)}, mean={float(differences.mean()):.6g}, "
        f"p95={float(p95):.6g}, max={float(max_difference):.6g}, "
        f"worst={pairs[int(worst)]!r}"
    )
    assert p95 <= PREFIX_CACHE_LOGPROB_P95_ATOL, stats
    assert max_difference <= PREFIX_CACHE_LOGPROB_MAX_ATOL, stats


def assert_prefix_cache_parity(reference_requests, cached_requests):
    """Assert cache-off/cache-on token, text, and log-probability parity."""
    assert len(reference_requests) == len(cached_requests)
    pairs = []
    for idx, (reference, cached) in enumerate(zip(reference_requests, cached_requests)):
        assert reference.output_tokens == cached.output_tokens, f"request {idx}: token mismatch"
        assert reference.output_text == cached.output_text, f"request {idx}: text mismatch"
        for field in (
            "prompt_log_probs",
            "generated_log_probs",
            "prompt_top_n_logprobs",
            "generated_top_n_logprobs",
        ):
            _collect_nested_numeric_pairs(
                getattr(reference, field, None),
                getattr(cached, field, None),
                f"request {idx}.{field}",
                pairs,
            )
    if pairs:
        _assert_logprob_parity(pairs)


def _build_dynamic_engine(model, inference_config, tokenizer):
    """Build a fresh context and engine around an already-loaded model."""
    context = DynamicInferenceContext(model.config, inference_config)
    wrapped_model = GPTInferenceWrapper(model, context)
    controller = TextGenerationController(wrapped_model, tokenizer)
    return context, DynamicInferenceEngine(controller, context)


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
    min_kv_pool_avail = engine.context.kv_block_allocator.pool_avail
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
        nonlocal min_kv_pool_avail, total_output_tokens, num_requests_finished

        min_kv_pool_avail = min(min_kv_pool_avail, engine.context.kv_block_allocator.pool_avail)

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
                request.num_cached_tokens = finished_request.num_cached_tokens
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
        "min_kv_pool_avail": min_kv_pool_avail,
    }


@torch.inference_mode()
def main():
    """Run dynamic inference."""
    # Initialize Megatron.
    args = parse_and_validate_args(
        extra_args_provider=add_runner_args,
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
    )

    model = get_model_for_inference()

    # Requests, context, controller.
    requests = build_requests(args, tokenizer, sampling_params)
    requests = build_prefix_cache_stress_requests(args, tokenizer, sampling_params, requests)
    inference_config = get_inference_config_from_model_and_args(model, args)

    # Calculate max_sequence_length from requests
    max_gen_length = sampling_params.num_tokens_to_generate
    max_context_length = max(len(r.prompt_tokens) for r in requests)
    inference_config.max_sequence_length = max_context_length + max_gen_length
    initial_config = (
        replace(inference_config, enable_prefix_caching=False)
        if args.prefix_cache_compare
        else inference_config
    )
    context, engine = _build_dynamic_engine(model, initial_config, tokenizer)

    # Validate all context_length's <= max_tokens.
    if not args.enable_chunked_prefill:
        invalid_prompt_length_map = {}
        for request_idx, request in enumerate(requests):
            if len(request.prompt_tokens) > context.max_tokens:
                invalid_prompt_length_map[request_idx] = len(request.prompt_tokens)
        assert (
            not invalid_prompt_length_map
        ), "request idxs with prompts longer than context.max_tokens: " ", ".join(
            f"{k}({v})" for k, v in invalid_prompt_length_map.items()
        )

    throughputs = []
    memory_cycles = []
    if args.prefix_cache_compare:
        assert inference_config.enable_prefix_caching
        assert sampling_params.top_k == 1 and sampling_params.top_p == 0.0
        assert (
            min(len(request.prompt_tokens) for request in requests) >= 2 * context.block_size_tokens
        )

        group_requests = requests[:: args.prefix_cache_stress_copies]
        distinct_block_demand = sum(
            len(request.prompt_tokens) // context.block_size_tokens for request in group_requests
        )
        usable_blocks = context.kv_block_allocator.pool_size - 1
        assert distinct_block_demand > usable_blocks, (distinct_block_demand, usable_blocks)
        first_group_hashes = compute_block_hashes_batched(
            torch.tensor(group_requests[0].prompt_tokens), context.block_size_tokens
        )

        reference_requests = copy.deepcopy(requests)
        run_inference(reference_requests, engine)
        context.deallocate_inference_state_buffers()
        del engine, context
        gc.collect()
        torch.cuda.empty_cache()
        context, engine = _build_dynamic_engine(model, inference_config, tokenizer)

    setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
    print("~~~")
    print(setup_prefix)
    print("~~~")

    # Run and time test, optionally `args.inference_repeat_n` times.
    for _ in range(args.inference_repeat_n):

        # Reset engine.
        if args.prefix_cache_compare:
            requests = copy.deepcopy(requests)
        else:
            engine.reset()

        torch.cuda.reset_peak_memory_stats()
        hit_start = engine._prefix_cache_hits

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

        if args.prefix_cache_compare:
            assert_prefix_cache_parity(reference_requests, requests)
            assert engine._prefix_cache_hits > hit_start
            assert result["min_kv_pool_avail"] == 0
            allocator = context.kv_block_allocator
            if inference_config.prefix_caching_eviction_policy.value == "ref_zero":
                assert not allocator.kv_hash_to_block_id
                assert allocator.pool_avail == allocator.pool_size - 1
            else:
                assert allocator.kv_hash_to_block_id
                assert allocator.pool_avail < allocator.pool_size - 1
            allocated = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda")
            torch.distributed.all_reduce(allocated, op=torch.distributed.ReduceOp.MAX)
            memory_cycles.append(int(allocated))

    if args.prefix_cache_compare:
        assert engine._prefix_cache_blocks_matched > 0
        assert any(request.num_cached_tokens > 0 for request in requests)
        if inference_config.prefix_caching_eviction_policy.value == "lru":
            assert any(
                block_hash not in context.kv_block_allocator.kv_hash_to_block_id
                for block_hash in first_group_hashes
            )
        assert memory_cycles[-1] <= memory_cycles[-2] + 64 * 1024**2, (
            "cache-on CUDA allocation grew by more than 64 MiB after warmup: "
            f"{memory_cycles[-2:]}"
        )

    # Validate all requests finished.
    for request in requests:
        assert request.state == "finished", f"request.state == '{request.state}' != 'finished'."

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
            for i, req in enumerate(requests):
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
            json_results["async_sched_step_count"] = engine.context.async_sched_step_count
            json_results["async_sched_compaction_step_count"] = (
                engine.context.async_sched_compaction_step_count
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
