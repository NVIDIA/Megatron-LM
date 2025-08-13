# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import hashlib
import os
import torch
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List
import sys
import os

from megatron.core.inference.contexts.dynamic_context import (
    ContextOverflowError,
    DynamicInferenceContext,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from megatron.training import get_args, get_model as _get_model, get_tokenizer, initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from pretrain_gpt import model_provider
import json

from examples.inference.gpt.utils import (
    add_common_inference_args,
    build_requests,
    build_dynamic_engine_setup_prefix,
    get_curr_time,
    Request,
)


def add_dynamic_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """Dynamic inference arguments."""

    add_common_inference_args(parser)

    group = parser.add_argument_group(title='Dynamic inference')
    group.add_argument(
        "--inference-ckpt-non-strict",
        action="store_true",
        help="Load checkpoint with `strict=False`.",
    )
    return parser


def get_model() -> MegatronModule:
    """Initialize model and load checkpoint."""

    args = get_args()

    # Build model.
    model = _get_model(model_provider, wrap_with_ddp=False)

    # Load checkpoint.
    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not args.inference_ckpt_non_strict,
    )

    # No virtual PP.
    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Eval mode.
    model.eval()

    return model


def get_inference_context(requests: List[Request], sampling_params: SamplingParams):
    """The inference context manages the KV cache and other inference state."""

    args = get_args()

    # Max sequence length.
    max_gen_length = sampling_params.num_tokens_to_generate
    max_context_length = max(len(r.prompt_tokens) for r in requests)
    max_sequence_length = max_context_length + max_gen_length

    # Inference context.
    context = DynamicInferenceContext(
        params_dtype=args.params_dtype,
        num_layers=args.num_layers,
        kv_channels=args.kv_channels,
        num_attention_heads=(
            args.num_query_groups if args.group_query_attention else args.num_attention_heads
        ),
        max_sequence_length=max_sequence_length,
        num_cuda_graphs=args.inference_dynamic_batching_num_cuda_graphs if args.enable_cuda_graph else None,
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        buffer_guaranteed_fraction=args.inference_dynamic_batching_buffer_guaranteed_fraction,
        chunk_size_tokens=args.inference_dynamic_batching_chunk_size,
        buffer_overflow_factor=args.inference_dynamic_batching_buffer_overflow_factor,
        max_requests_override=args.inference_dynamic_batching_max_requests_override,
        max_tokens_override=args.inference_dynamic_batching_max_tokens_override,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        materialize_only_last_token_logits=not args.return_log_probs,
    )

    return context


def get_inference_controller(
    model: MegatronModule, context: DynamicInferenceContext
) -> TextGenerationController:
    """Buid text generation controller, which manages the model inference context.

    Args:
        model (MegatronModule): Megatron GPT model.
        context (DynamicInferenceContext): Context for managing KV cache.

    Return:
        (TextGenerationController) Inference text generation controller.
    """

    args = get_args()
    tokenizer = get_tokenizer()

    # Wrap model in inference wrapper.
    model = GPTInferenceWrapper(model, args, context)

    # Note: the following is taken from AbstractModelInferenceWrapper.prep_model_for_inference().
    from megatron.core import parallel_state

    model.model_is_pipeline_parallel = not (
        parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
    )

    # Text generation controller.
    controller = TextGenerationController(model, tokenizer)

    return controller


def run_inference(
    requests: List[Request], sampling_params: SamplingParams, engine: DynamicInferenceEngine
) -> List[Dict[str, float]]:
    """Add requests to engine and generate tokens.

    Args:
        requests (List[Request]): Requests that are to be added and processed.
        sampling_params (SamplingParams): Sampling params for the logits.
        engine (DynamicInferenceEngine): Inference engine that manages generating tokens.

    Return:
        A dictionary of step times with `prefill` and `decode` keys.
    """

    # Initialize request arrival times.
    base_arrival_time = get_curr_time()
    for request in requests:
        request.time_arrival = request.time_offset + base_arrival_time

    # Add and process requests.
    num_requests_total = len(requests)
    num_requests_added = 0
    num_requests_finished = 0
    step_id = 0
    step_times = {"prefill": [], "decode": []}
    add_times = []
    output_times = []
    tbar = tqdm(total=num_requests_total)
    total_output_tokens = 0
    while True:
        curr_time = get_curr_time()

        # Add requests with 'earlier' arrival time.
        add_start = get_curr_time()
        while num_requests_added < num_requests_total:
            request = requests[num_requests_added]
            if request.time_arrival > curr_time:
                break
            try:
                # Using `prompt_text` instead of `prompt_tokens` for fair comparison.
                engine.add_request(
                    num_requests_added, request.prompt_text, sampling_params.num_tokens_to_generate
                )
                request.time_start = get_curr_time()
                request.state = "started"
                num_requests_added += 1
                tbar.update(1)
            except ContextOverflowError:
                break
        add_times.append(get_curr_time() - add_start)

        # Step inference engine (i.e., generate a token for each active request).
        is_decode_only = engine.context.is_decode_only()
        active_requests, finished_requests, step_time = engine.step(sampling_params, verbose=True)
        step_id += 1

        if len(active_requests) > 0 or len(finished_requests) > 0:
            if is_decode_only:
                step_times["decode"].append(step_time)
            else:
                step_times["prefill"].append(step_time)

            # Append output tokens.
            for finished_request in finished_requests:
                request = requests[finished_request.request_id]
                request.output_tokens = finished_request.generated_tokens
                total_output_tokens += len(request.output_tokens)
                request.time_end = get_curr_time()
                request.output_text = finished_request.generated_text
                request.state = "finished"
                request.request_id = finished_request.request_id
                if sampling_params.return_log_probs:
                    request.log_probs = (
                        finished_request.prompt_log_probs + finished_request.generated_log_probs
                    )
                num_requests_finished += 1

        # Check if all requests are finished.
        if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
            break

    return step_times, add_times, output_times, total_output_tokens


@torch.inference_mode()
def main():
    # Initialize Megatron.
    initialize_megatron(
        extra_args_provider=add_dynamic_inference_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )

    # Start Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()

    args = get_args()
    tokenizer = get_tokenizer()

    # Sampling params.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )

    # Requests, context, conroller.
    model = get_model()
    requests = build_requests(args, tokenizer)
    context = get_inference_context(requests, sampling_params)
    controller = get_inference_controller(model, context)

    # Inference engine.
    engine = DynamicInferenceEngine(
        controller,
        context,
        termination_id=tokenizer.eod,
        enable_cuda_graph=args.enable_cuda_graph,
        random_seed=args.seed,
    )

    setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
    print("~~~")
    print(setup_prefix)
    print("~~~")

    # Run and time test.
    t = get_curr_time()
    step_times, add_times, output_times, total_output_tokens = run_inference(requests, sampling_params, engine)
    torch.cuda.synchronize()
    total_time = get_curr_time() - t

    # Validate all requests finished.
    for request in requests:
        assert request.state == "finished"

    # Print unique prompts + outputs.
    if torch.distributed.get_rank() == 0:

        print("~~~~ Unique prompts + outputs. ~~~~")

        # Map requests by their prompt.
        unique_prompt_map = defaultdict(list)
        for request_idx, request in enumerate(requests):
            unique_prompt_map[request.prompt_text].append(request_idx)

        # Print unique prompts + outputs.
        for unique_idx, (prompt_text, request_idxs) in enumerate(unique_prompt_map.items()):
            request_idx = request_idxs[0]
            request = requests[request_idx]
            output_text_hash = hashlib.sha256(request.output_text.encode()).hexdigest()[:6]
            output_text_escaped = request.output_text.replace("\n", "\\n")
            print(
                f"{unique_idx}/{len(unique_prompt_map)} [n {len(request_idxs)}, hash {output_text_hash}]. "
                f"{prompt_text} ... {output_text_escaped}"
            )

        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}

            for idx, req in enumerate(requests):
                result_dict = {
                    "input_prompt": req.prompt_text,
                    "generated_text": req.output_text,
                    "generated_tokens": req.output_tokens,
                    "latency": req.time_end - req.time_start,
                }
                if sampling_params.return_log_probs:
                    response_logprobs = req.log_probs
                    result_dict["logprobs"] = response_logprobs
                json_results[req.request_id] = result_dict
            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp)

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
    d_mean = d_total / d_count

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
    print(
        f"{setup_prefix} … "
        f"mem {peak_alloc_gb:.1f}/{peak_resvd_gb:.1f} GB … "
        f"total time: {total_time:.3f}s … "
        f"throughput: {throughput:.3f} tok/s"
    )
    print("~~~")

    # Stop Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
