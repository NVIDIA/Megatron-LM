# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline inference example using the Megatron high-level API.

Mirrors examples/inference/legacy/gpt_dynamic_inference.py but drives the
``DynamicInferenceEngine`` through ``MegatronLLM`` (sync) or
``MegatronAsyncLLM`` (async, via ``--async-mode``) instead of the manual
add_request/step_modern loop. Output format (setup prefix, unique prompt
blocks, throughput line, optional JSON dump) matches the legacy script.

Run modes are selected at the CLI:

    # sync, direct (default)
    python -m examples.inference.offline_inference --load <ckpt> ...

    # sync, coordinator
    python -m examples.inference.offline_inference --load <ckpt> --use-coordinator ...

    # async (with or without --use-coordinator)
    python -m examples.inference.offline_inference --load <ckpt> --async-mode ...
"""

import asyncio
import logging
import os
import sys
from argparse import ArgumentParser

import torch
import torch.distributed as dist

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from examples.inference.utils import (
    build_dynamic_engine_setup_prefix,
    build_requests,
    dump_inference_results_to_json,
    get_curr_time,
    get_global_peak_memory_stats_bytes,
    print_unique_prompts_and_outputs,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import configure_nvtx_profiling
from megatron.inference import MegatronAsyncLLM, MegatronLLM
from megatron.inference.utils import (
    add_inference_args,
    get_inference_config_from_model_and_args,
    get_model_for_inference,
)
from megatron.training import initialize_megatron
from megatron.training.arguments import parse_and_validate_args


def add_offline_inference_args(parser: ArgumentParser) -> ArgumentParser:
    parser = add_inference_args(parser)
    group = parser.add_argument_group(title='Offline inference (high-level API)')
    group.add_argument("--use-coordinator", action="store_true", default=False)
    group.add_argument("--coordinator-host", type=str, default=None)
    group.add_argument("--coordinator-port", type=int, default=None)
    group.add_argument(
        "--async-mode",
        action="store_true",
        default=False,
        help="Drive MegatronAsyncLLM via asyncio.run instead of MegatronLLM.",
    )
    return parser


def _validate_high_level_api_args(args):
    # engine.reset() between trials races the runtime engine loop in
    # coordinator mode (engine_loop_task runs on the runtime thread).
    if args.use_coordinator and args.inference_repeat_n > 1:
        raise ValueError(
            "--use-coordinator with --inference-repeat-n > 1 is not supported: "
            "engine.reset() races the runtime engine loop in coordinator mode."
        )
    # The high-level API takes one sampling_params per generate() call.
    if args.prompt_file and getattr(args, "num_tokens_from_file", False):
        raise ValueError(
            "--prompt-file with --num-tokens-from-file produces per-request "
            "num_tokens_to_generate, but the high-level API takes one "
            "sampling_params per generate() call. Use a uniform "
            "--num-tokens-to-generate instead."
        )


def _validate_prompt_lengths(args, llm, requests):
    # Validate prompt lengths against the resolved max_tokens (default
    # is filled in by DynamicInferenceContext during construction).
    if args.enable_chunked_prefill:
        return
    invalid = {
        idx: len(r.prompt_tokens)
        for idx, r in enumerate(requests)
        if len(r.prompt_tokens) > llm.context.max_tokens
    }
    assert not invalid, (
        "request idxs with prompts longer than context.max_tokens: "
        ", ".join(f"{k}({v})" for k, v in invalid.items())
    )


def _capture_engine_stats(llm) -> dict:
    return {
        "step_count": llm.engine.context.step_count,
        "lifetime_prefill_token_count": llm.engine.context.lifetime_prefill_token_count,
        "capture_stats": llm.engine.capture_stats,
    }


def _print_setup_prefix(setup_prefix: str) -> None:
    if dist.get_rank() == 0:
        print("~~~")
        print(setup_prefix)
        print("~~~")


def _report_results(
    args, setup_prefix, results, throughputs, total_time, peak_mem_stats, captured
):
    if dist.get_rank() != 0:
        return

    print_unique_prompts_and_outputs(results)
    dump_inference_results_to_json(
        args,
        results,
        throughputs,
        peak_mem_stats,
        captured["step_count"],
        captured["lifetime_prefill_token_count"],
    )

    stats = torch.cuda.memory_stats()
    peak_alloc_gb = stats["allocated_bytes.all.peak"] / 1024**3
    peak_resvd_gb = stats["reserved_bytes.all.peak"] / 1024**3
    throughput = throughputs[-1] if throughputs else 0.0
    capture_str = (
        f"{captured['capture_stats']['time']:.2f} sec"
        if captured["capture_stats"]
        else "--"
    )
    print("~~~")
    print(
        f"{setup_prefix} … " f"throughput: {throughput:.3f} tok/s … ",
        f"total time: {total_time:.3f}s … "
        f"mem {peak_alloc_gb:.1f}/{peak_resvd_gb:.1f} GB … "
        f"steps: {captured['step_count']:d} … "
        f"capture {capture_str}",
    )
    print("~~~")


def _run_sync(args, model, tokenizer, inference_config, requests, prompts_list, sampling_params):
    results = []
    throughputs = []
    total_time = 0.0
    captured = {"step_count": 0, "lifetime_prefill_token_count": 0, "capture_stats": None}
    setup_prefix = ""

    with MegatronLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=args.use_coordinator,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
    ) as llm:
        setup_prefix = build_dynamic_engine_setup_prefix(args, model, llm.context, requests)
        _validate_prompt_lengths(args, llm, requests)

        # Coordinator mode: only the primary rank submits work; worker ranks
        # fall through and block in __exit__ until shutdown propagates STOP.
        if llm.is_primary_rank:
            _print_setup_prefix(setup_prefix)
            for trial_idx in range(args.inference_repeat_n):
                # Skip first-trial reset; the engine is fresh post-construction.
                if trial_idx > 0:
                    llm.engine.reset()
                torch.cuda.reset_peak_memory_stats()

                t = get_curr_time()
                results = llm.generate(prompts_list, sampling_params)
                torch.cuda.synchronize()
                total_time = get_curr_time() - t

                total_output_tokens = sum(len(r.generated_tokens) for r in results)
                throughputs.append(total_output_tokens / total_time)
            captured = _capture_engine_stats(llm)

    # Engine is shut down on all ranks; safe to all-reduce peak-memory now.
    peak_mem_stats = get_global_peak_memory_stats_bytes()
    _report_results(args, setup_prefix, results, throughputs, total_time, peak_mem_stats, captured)


async def _run_async(
    args, model, tokenizer, inference_config, requests, prompts_list, sampling_params
):
    results = []
    throughputs = []
    total_time = 0.0
    captured = {"step_count": 0, "lifetime_prefill_token_count": 0, "capture_stats": None}
    setup_prefix = ""

    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=args.use_coordinator,
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
    ) as llm:
        setup_prefix = build_dynamic_engine_setup_prefix(args, model, llm.context, requests)
        _validate_prompt_lengths(args, llm, requests)

        if llm.is_primary_rank:
            _print_setup_prefix(setup_prefix)
            for trial_idx in range(args.inference_repeat_n):
                if trial_idx > 0:
                    llm.engine.reset()
                torch.cuda.reset_peak_memory_stats()

                t = get_curr_time()
                results = await llm.generate(prompts_list, sampling_params)
                torch.cuda.synchronize()
                total_time = get_curr_time() - t

                total_output_tokens = sum(len(r.generated_tokens) for r in results)
                throughputs.append(total_output_tokens / total_time)
            captured = _capture_engine_stats(llm)

    peak_mem_stats = get_global_peak_memory_stats_bytes()
    _report_results(args, setup_prefix, results, throughputs, total_time, peak_mem_stats, captured)


def main():
    args = parse_and_validate_args(
        extra_args_provider=add_offline_inference_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
    )
    initialize_megatron()
    _validate_high_level_api_args(args)

    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStart()

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, force=True)
    configure_nvtx_profiling(True)

    tokenizer = build_tokenizer(args)
    torch.cuda.reset_peak_memory_stats()

    model = get_model_for_inference()
    inference_config = get_inference_config_from_model_and_args(model, args)
    requests = build_requests(args, tokenizer, sampling_params=None)
    sampling_params = requests[0].sampling_params

    max_gen_length = sampling_params.num_tokens_to_generate
    max_context_length = max(len(r.prompt_tokens) for r in requests)
    inference_config.max_sequence_length = max_context_length + max_gen_length

    prompts_list = [r.prompt_text for r in requests]

    runner_args = (args, model, tokenizer, inference_config, requests, prompts_list, sampling_params)
    if args.async_mode:
        asyncio.run(_run_async(*runner_args))
    else:
        _run_sync(*runner_args)

    if os.environ.get("NSIGHT_PREFIX"):
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
