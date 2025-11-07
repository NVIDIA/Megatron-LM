# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
import json
import math
import os
import pickle
import sys
import torch
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm

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
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_attr_wrapped_model

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from megatron.training import get_args, get_model as _get_model, get_tokenizer, initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from model_provider import model_provider
from gpt_builders import gpt_builder
from mamba_builders import mamba_builder

from megatron.core.utils import configure_nvtx_profiling

import json

from examples.inference.gpt.utils import (
    Request,
    add_common_inference_args,
    build_dynamic_engine_setup_prefix,
    build_requests,
    get_curr_time,
)
from megatron.training import get_args
from megatron.training import get_model as _get_model
from megatron.training import get_tokenizer, initialize_megatron
from megatron.training.checkpointing import load_checkpoint

import torch
import io
import megatron


def add_dynamic_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """Dynamic inference arguments."""

    add_common_inference_args(parser)

    group = parser.add_argument_group(title='Dynamic inference')
    group.add_argument(
        "--inference-ckpt-non-strict",
        action="store_true",
        help="Load checkpoint with `strict=False`.",
    )
    group.add_argument(
        "--termination-id", type=int, default=None,
        help="Termination ID that overrides `tokenizer.eod`."
    )
    group.add_argument('--inference-repeat-n', type=int, default=1, help="Repeat inference iterations N times for benchmarking.")

    return parser


def get_model() -> MegatronModule:
    """Initialize model and load checkpoint."""

    args = get_args()

    if args.model_provider == "gpt":
        model_builder = gpt_builder
    elif args.model_provider == "mamba":
        model_builder = mamba_builder
    else:
        raise ValueError(f"Invalid model provider {args.model_provider}")

    # Build model.
    model = _get_model(
        partial(model_provider, model_builder),
        wrap_with_ddp=False
    )

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


def get_inference_context(
    requests: List[Request],
    sampling_params: Optional[SamplingParams] = None,
    calculate_max_sequence_length_from_requests: bool = True,
    layer_type_list: Optional[List[str]] = None,
    mamba_conv_states_shape: Optional[Tuple[int]] = None,
    mamba_ssm_states_shape: Optional[Tuple[int]] = None,
):
    """The inference context manages the KV cache and other inference state."""

    args = get_args()
    # Max sequence length.
    if calculate_max_sequence_length_from_requests:
        max_gen_length = sampling_params.num_tokens_to_generate    
        max_context_length = max(len(r.prompt_tokens) for r in requests)
        max_sequence_length = max_context_length + max_gen_length
    else:
        max_sequence_length = args.inference_max_seq_length

    metrics_writer = None
    if args.inference_wandb_logging_step_interval > 0:
        metrics_writer = get_wandb_writer()

    # Inference context.
    context = DynamicInferenceContext(
        params_dtype=args.params_dtype,
        num_layers=args.num_layers,
        kv_channels=args.kv_channels,
        num_attention_heads=(
            args.num_query_groups if args.group_query_attention else args.num_attention_heads
        ),
        max_sequence_length=max_sequence_length,
        num_cuda_graphs=(
            args.inference_dynamic_batching_num_cuda_graphs
            if args.cuda_graph_impl == "local"
            else None
        ),
        block_size_tokens=args.inference_dynamic_batching_block_size,
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        buffer_guaranteed_fraction=args.inference_dynamic_batching_buffer_guaranteed_fraction,
        buffer_overflow_factor=args.inference_dynamic_batching_buffer_overflow_factor,
        max_requests_override=args.inference_dynamic_batching_max_requests_override,
        max_tokens_override=args.inference_dynamic_batching_max_tokens_override,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        materialize_only_last_token_logits=not args.return_log_probs,
        layer_type_list=layer_type_list,
        mamba_conv_states_shape=mamba_conv_states_shape,
        mamba_ssm_states_shape=mamba_ssm_states_shape,
        cache_mla_latent=args.multi_latent_attention and args.cache_mla_latents,
        kv_lora_rank=args.kv_lora_rank if args.multi_latent_attention else None,
        qk_pos_emb_head_dim=args.qk_pos_emb_head_dim,
        use_cuda_graphs_for_non_decode_steps=not args.decode_only_cuda_graphs,
        use_flashinfer_fused_rope=args.use_flashinfer_fused_rope,
        unified_memory_level=args.inference_dynamic_batching_unified_memory_level,
        metrics_writer=metrics_writer,
    )

    return context


def get_inference_controller(
    model: MegatronModule, context: DynamicInferenceContext
) -> TextGenerationController:
    """Buid text generation controller, which manages the model inference context.

    Args:
        model (MegatronModule): Megatron GPT model.
        context (DynamicInferenceContext): Context for managing KV cache blocks.

    Return:
        (TextGenerationController) Inference text generation controller.
    """

    args = get_args()
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

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
    step_id = 0
    step_times = {"prefill": [], "decode": []}
    add_times = []
    output_times = []
    tbar = tqdm(total=num_requests_total)
    total_output_tokens = 0
    if args.cuda_graph_impl == "local":
        cuda_graph_request_count_map = {r:0 for r in engine.context.cuda_graph_request_counts}
    else:
        cuda_graph_request_count_map = None

    def _add_request():
        """Add request to engine.

        *Note: Using `prompt_text` instead of `prompt_tokens` for fair comparison.
        """
        nonlocal num_requests_added
        _request = requests[num_requests_added]
        engine.add_request(
            num_requests_added,
            _request.prompt_text,
            _request.sampling_params,
        )
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
            for i in range(min(
                args.incoming_requests_per_step,
                num_requests_total - num_requests_added,
            )):
                _add_request()
        add_times.append(get_curr_time() - add_start)

        # Step inference engine (i.e., generate a token for each active request).
        # Before step, we haven't done the scheduling, so we cannot know the is_decode_only
        result = engine.step_modern(verbose=True)
        # After step, we lost track of last iteration's is_decode_only, so we need to get it from the engine
        is_decode_only = engine.is_decode_only 
        step_id += 1

        # Record cuda_graph_request_count.
        cuda_graph_request_count = result["cuda_graph_request_count"]
        if args.cuda_graph_impl == "local" and cuda_graph_request_count is not None:
            cuda_graph_request_count_map[cuda_graph_request_count] += 1

        # Update requests.
        active_requests = result["active_requests"]
        finished_requests = result["finished_requests"]
        step_time = result["step_time"]
        if len(active_requests) > 0 or len(finished_requests) > 0:
            if is_decode_only:
                step_times["decode"].append(step_time)
            else:
                step_times["prefill"].append(step_time)

            # Append output tokens.
            output_start = get_curr_time()
            for finished_request in finished_requests:
                request = requests[finished_request.request_id]
                request.output_tokens = finished_request.generated_tokens
                total_output_tokens += len(request.output_tokens)
                request.time_end = get_curr_time()
                request.output_text = finished_request.generated_text
                request.state = "finished"
                request.request_id = finished_request.request_id
                if finished_request.sampling_params.return_log_probs:
                    request.log_probs = (
                        finished_request.prompt_log_probs + finished_request.generated_log_probs
                    )
                num_requests_finished += 1
            output_times.append(get_curr_time() - output_start)

        # Check if all requests are finished.
        if not (engine.has_unfinished_requests() or num_requests_added < num_requests_total):
            break

    return {
        "step_times" : step_times,
        "add_times" : add_times,
        "output_times" : output_times,
        "total_output_tokens" : total_output_tokens,
        "cuda_graph_request_count_map" : cuda_graph_request_count_map,
    }


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
    
    configure_nvtx_profiling(True)

    args = get_args()
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Sampling params.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=args.termination_id if args.termination_id is not None else tokenizer.eod,
    )

    model = get_model()

    # Layer type list for hybrid models
    decoder = get_attr_wrapped_model(model, "decoder")
    layer_type_list = getattr(decoder, "layer_type_list", None)
    if layer_type_list is not None and Symbols.MAMBA in layer_type_list:
        (mamba_conv_states_shape, mamba_ssm_states_shape) = decoder.mamba_state_shapes_per_request()
    else:
        mamba_conv_states_shape = None
        mamba_ssm_states_shape = None

    # Requests, context, controller.
    requests = build_requests(args, tokenizer, sampling_params)
    context = get_inference_context(
        requests,
        sampling_params,
        layer_type_list=layer_type_list,
        mamba_conv_states_shape=mamba_conv_states_shape,
        mamba_ssm_states_shape=mamba_ssm_states_shape,
    )
    controller = get_inference_controller(model, context)

    # Validate all context_length's <= max_tokens.
    if args.disable_chunked_prefill:
        invalid_prompt_length_map = {}
        for request_idx, request in enumerate(requests):
            if len(request.prompt_tokens) > context.max_tokens:
                invalid_prompt_length_map[request_idx] = len(request.prompt_tokens)
        assert not invalid_prompt_length_map, (
            "request idxs with prompts longer than context.max_tokens: "
            ", ".join(f"{k}({v})" for k, v in invalid_prompt_length_map.items())
        )

    # Inference engine.
    engine = DynamicInferenceEngine(
        controller,
        context,
        enable_cuda_graph=args.cuda_graph_impl == "local",
        random_seed=args.seed,
        track_paused_request_events=args.inference_dynamic_batching_track_paused_request_events,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        inference_logging_step_interval=args.inference_wandb_logging_step_interval,
    )

    setup_prefix = build_dynamic_engine_setup_prefix(args, model, context, requests)
    print("~~~")
    print(setup_prefix)
    print("~~~")

    # Run and time test, optionally `args.inference_repeat_n` times.
    throughputs = []
    for _ in range(args.inference_repeat_n):
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
        assert request.state == "finished", (
            f"request.state == '{request.state}' != 'finished'."
        )

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
        for unique_idx, (prompt_text, request_idxs) in enumerate(unique_prompt_map.items()):
            # ---- Prompt summary line ----
            prompt_len = len(requests[request_idxs[0]].prompt_tokens)
            escaped_prompt_text = escape_str(prompt_text)
            print(f"{unique_idx+1}/{len(unique_prompt_map)} [n {len(request_idxs)}, l {prompt_len}] {escaped_prompt_text}")

            # ---- Group all outputs for this prompt ----
            output_map = defaultdict(list)
            for idx in request_idxs:
                req = requests[idx]
                output_map[req.output_text].append(idx)

            # ---- Print each unique output ----
            for output_text, output_request_idxs in output_map.items():
                if output_text is not None:
                    o_hash = hashlib.sha256(output_text.encode()).hexdigest()[:6]
                    o_len = len(requests[output_request_idxs[0]].output_tokens)
                    escaped_output_text = escape_str(output_text)
                    print(f"  >>>> [n {len(output_request_idxs)}, l {o_len}, hash {o_hash}] {escaped_output_text}")
                else:
                    o_hash = "--"
                    o_len = 0
                    escaped_output_text = "--"
                    print(f"  >>>> [n {len(output_request_idxs)}, {o_len} tokens, hash {o_hash}] {escaped_output_text}")

        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            json_results = {}

            # Write every 'n' requests, plus the final request.
            for i, req in enumerate(requests):
                if i % args.output_every_n_results == 0 or i == len(requests) - 1:
                    result_dict = {
                        "input_prompt": req.prompt_text,
                        "generated_text": req.output_text,
                        "generated_tokens": req.output_tokens,
                        "latency": req.time_end - req.time_start,
                        "cuda_graph_request_count_map" : result["cuda_graph_request_count_map"],
                        "step_count" : engine.step_count,
                    }
                    if req.sampling_params.return_log_probs:
                        response_logprobs = req.log_probs
                        result_dict["logprobs"] = response_logprobs
                    json_results[req.request_id] = result_dict

            # Track system-level throughput as a test / debug metric
            json_results["throughput"] = throughputs

            with open(args.output_path, "w") as fp:
                json.dump(json_results, fp, indent=1)

    # Timing results.
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
    capture_str = (
        f"{engine.capture_stats['time']:.2f} sec"
        if engine.capture_stats else
        "--"
    )
    print(
        f"{setup_prefix} … "
        f"capture {capture_str} … "
        f"mem {peak_alloc_gb:.1f}/{peak_resvd_gb:.1f} GB … "
        f"total time: {total_time:.3f}s … "
        f"steps: {engine.step_count:d} … "
        f"throughput: {throughput:.3f} tok/s"
    )
    print("~~~")

    # Stop Nsight profiler.
    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
