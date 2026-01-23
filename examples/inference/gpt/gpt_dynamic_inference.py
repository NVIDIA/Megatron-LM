# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# pylint: disable=bad-builtin

import hashlib
import io
import json
import os
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

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
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine, EngineSuspendedError
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
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
        _request.time_start = get_curr_time()
        _request.state = "started"
        num_requests_added += 1
        tbar.update(1)

    while True:
        # Add requests.
        add_start = get_curr_time()
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
        add_times.append(get_curr_time() - add_start)

        # Step inference engine (i.e., generate a token for each active request).
        # Before step, we haven't done the scheduling, so we cannot know the is_decode_only
        try:
            result = engine.step_modern()
        except EngineSuspendedError as e:
            result = e
            pass  # ignore error in order to call 'engine.resume()' below.
        attempted_step_count += 1

        # After step, we lost track of last iteration's is_decode_only,
        # so we need to get it from the engine
        is_decode_only = engine.is_decode_only

        # Test suspending and resuming engine.
        if args.suspend_resume_interval is not None:

            # Suspend.
            if attempted_step_count % args.suspend_resume_interval == 0:
                print("**** step %d/%d ... suspend." % (engine.step_count, attempted_step_count))
                engine.suspend()

            # Resume, 0+ attempted steps later.
            if (
                attempted_step_count > 0
                and (attempted_step_count - args.suspend_resume_interval // 2)
                % args.suspend_resume_interval
                == 0
            ):
                print("**** step %d/%d ... resume." % (engine.step_count, attempted_step_count))
                engine.resume()

        # If engine suspended, continue to next iter.
        if isinstance(result, EngineSuspendedError):
            continue

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
            output_start = get_curr_time()
            for finished_request_record in finished_request_records:

                finished_request = finished_request_record.merge()

                # Update local request object.
                request = requests[finished_request.request_id]
                request.time_end = get_curr_time()
                request.state = "finished"
                request.request_id = finished_request.request_id
                request.events = finished_request.events

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
            output_times.append(get_curr_time() - output_start)

        # Check if all requests are finished.
        if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
            break

    # Resume engine (NOOP if not suspended).
    if engine.is_suspended:
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
    initialize_megatron(
        extra_args_provider=add_inference_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )

    # Start Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(level=level, force=True)

    configure_nvtx_profiling(True)

    args = get_args()
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
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
    inference_config = get_inference_config_from_model_and_args(model, args)

    # Calculate max_sequence_length from requests
    max_gen_length = sampling_params.num_tokens_to_generate
    max_context_length = max(len(r.prompt_tokens) for r in requests)
    inference_config.max_sequence_length = max_context_length + max_gen_length
    context = DynamicInferenceContext(model.config, inference_config)
    wrapped_model = GPTInferenceWrapper(model, context)
    controller = TextGenerationController(wrapped_model, tokenizer)

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

    # Inference engine.
    engine = DynamicInferenceEngine(controller, context)

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
                        "cuda_graph_request_count_map": result["cuda_graph_request_count_map"],
                        "step_count": engine.step_count,
                        "top_n_logprobs": getattr(req, 'generated_top_n_logprobs', None),
                        "prompt_top_n_logprobs": getattr(req, 'prompt_top_n_logprobs', None),
                    }
                    if req.sampling_params.return_log_probs:
                        result_dict["prompt_logprobs"] = getattr(req, 'prompt_log_probs', None)
                        result_dict["generated_logprobs"] = getattr(
                            req, 'generated_log_probs', None
                        )
                        result_dict["logprobs"] = getattr(req, 'logprobs', None)
                    json_results[req.request_id] = result_dict

            # Track system-level throughput as a test / debug metric
            if args.record_throughput:
                json_results["throughput"] = throughputs
            # Attach peak memory metrics; the functional test only validates these
            # if the fields exist in the golden values.
            json_results.update(peak_mem_stats)

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
            f"mem {peak_alloc_gb:.1f}/{peak_resvd_gb:.1f} GB … "
            f"steps: {engine.step_count:d} … "
            f"capture {capture_str}",
        )
        print("~~~")

    # Stop Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
