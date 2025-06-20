# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from pretrain_mamba import model_provider as mamba_model_provider
from pretrain_gpt import model_provider as gpt_model_provider
import torch
import sys
import time
import tqdm
import warnings
from argparse import Namespace
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
import json
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model
import asyncio
from typing import AsyncIterator, List

from examples.inference.gpt.utils import add_common_inference_args, build_requests


def add_static_inference_args(parser):
    """Static inference arguments."""

    add_common_inference_args(parser)

    group = parser.add_argument_group(title='Static inference')
    group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        dest="max_batch_size",
        help='Deprecated, use `--inference-max-requests` instead',
    )
    group.add_argument("--stream", action="store_true", default=False, help="Stream output tokens")
    group.add_argument(
        "--output-path", type=str, default=None, help="Path to save generations as JSON"
    )

    return parser


def get_inference_engine(args: Namespace, model: MegatronModule) -> StaticInferenceEngine:
    """Utility to get the relevant backend for running inference

    This function will automatically choose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. TRT LLM Backend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model .

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_requests=args.inference_max_batch_size,
        inference_max_seq_length=args.inference_max_seq_length,
        nccl_all_reduce_for_prefill=args.nccl_all_reduce_for_prefill,
    )

    inference_context = StaticInferenceContext.from_config(inference_wrapper_config)

    inference_wrapped_model = GPTInferenceWrapper(
        model, inference_wrapper_config, inference_context
    )
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return StaticInferenceEngine(text_generation_controller=text_generation_controller)


async def generate(
    inference_engine: StaticInferenceEngine, sampling_params: SamplingParams, prompts: List[str]
) -> List[InferenceRequest]:
    async def collect_stream(prompt, request_id, stream_generator):
        print(f"Request {request_id}: {prompt}", end="", flush=True)
        prev_idx = 0
        async for output in stream_generator:
            print(output.generated_text[prev_idx:], end="", flush=True)
            prev_idx = len(output.generated_text)
        print()

    request_ids: List[str] = [
        inference_engine.add_request(prompt=prompt, sampling_params=sampling_params, streaming=True)
        for prompt in prompts
    ]
    stream_generators = [
        inference_engine.get_stream_generator(request_id) for request_id in request_ids
    ]

    tasks = [
        asyncio.create_task(collect_stream(prompt, request_id, stream_generator))
        for (prompt, request_id, stream_generator) in zip(prompts, request_ids, stream_generators)
    ]

    await inference_engine.run_engine_async()
    await asyncio.gather(*tasks)

    results: List[InferenceRequest] = [
        inference_engine.scheduler.completed_request_pool[request_id] for request_id in request_ids
    ]

    return results


@torch.inference_mode()
def main():
    """Main program."""

    # Note: The default args passed here can be overwritten by using appropriate params (check arguments.py file)
    # Micro batch size is not needed to be set by user. (It is calculated based on inference-batch-times-seqlen-threshold argument)
    initialize_megatron(
        extra_args_provider=add_static_inference_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()

    if args.max_batch_size is not None:
        warnings.warn(
            f"`--max-batch-size` has been deprecated in favor of `--inference-max-requests`."
        )
        args.inference_max_batch_size = max(args.max_batch_size, args.inference_max_batch_size)

    # Set up model and load checkpoint
    if args.model_provider == "gpt":
        model_provider = gpt_model_provider
    elif args.model_provider == "mamba":
        model_provider = mamba_model_provider
    else:
        raise ValueError(f"Invalid model provider {args.model_provider}")
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=False)
    model = model[0]

    inference_engine = get_inference_engine(args, model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        top_n_logprobs=args.top_n_logprobs,
    )

    requests = build_requests(args, get_tokenizer())
    prompts = [r.prompt_text for r in requests]

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
            prompts=["warmup"], sampling_params=SamplingParams(num_tokens_to_generate=10)
        )
    start_time = time.perf_counter()
    if args.stream:
        results: List[InferenceRequest] = asyncio.run(
            generate(inference_engine, sampling_params, prompts)
        )
    else:
        results: List[InferenceRequest] = inference_engine.generate(
            prompts=prompts, sampling_params=sampling_params
        )
    end_time = time.perf_counter()
    latency = end_time - start_time

    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            result_dict = {
                'id': result.request_id,
                'input_prompt': result.prompt,
                'generated_text': result.generated_text,
                'generated_tokens': result.generated_tokens,
                'latency': latency,
            }
            if sampling_params.top_n_logprobs > 0:
                result_dict['generated_top_n_logprobs'] = result.generated_top_n_logprobs
            if sampling_params.return_log_probs:
                response_logprobs = result.prompt_log_probs + result.generated_log_probs
                result_dict["logprobs"] = response_logprobs

        # Write results to JSON. Primarily used for functional testing.
        if args.output_path:
            # Tensors cannot be serialized so we move these to CPU
            result_dict['generated_tokens'] = result_dict['generated_tokens'].cpu().numpy().tolist()
            results_as_json = json.dumps(result_dict)
            with open(args.output_path, 'w') as f:
                json.dump(results_as_json, f)

    # Print unique prompts + outputs.
    if torch.distributed.get_rank() == 0:

        print("~~~~ Unique prompts + outputs. ~~~~")

        # Map results by their prompt.
        from collections import defaultdict

        unique_prompt_map = defaultdict(list)
        for result_idx, result in enumerate(results):
            unique_prompt_map[result.prompt].append(result_idx)

        # Print unique prompts + outputs.
        for unique_idx, (prompt_text, result_idxs) in enumerate(unique_prompt_map.items()):
            result_idx = result_idxs[0]
            result = results[result_idx]
            generated_text = result.generated_text.replace("\n", "\\n")
            print(
                f"{unique_idx}/{len(unique_prompt_map)} [{len(result_idxs)}]. {prompt_text} "
                f"... {generated_text}"
            )

    stats = torch.cuda.memory_stats()
    print_rank_0(
        "static | cg %d | %s | reqs %d [ batch %d ] ... mem %.1f/%.1f ... time %.3f."
        % (
            args.enable_cuda_graph,
            (
                f"<user prompts>"
                if args.prompts
                else "<auto prompts> %s, %d, %.1e, %.1e"
                % (
                    "(%s)" % " ".join(map(str, args.num_tokens_to_prompt)),
                    args.num_tokens_to_generate,
                    args.incoming_requests_duration,
                    args.incoming_requests_per_sec,
                )
            ),
            len(requests),
            args.inference_max_batch_size,
            stats["allocated_bytes.all.peak"] / (1024**3),
            stats["reserved_bytes.all.peak"] / (1024**3),
            latency,
        )
    )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
