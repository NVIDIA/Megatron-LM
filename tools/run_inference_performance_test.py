# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import os
import random
import sys
import time

import torch

from gpt_builders import gpt_builder
from mamba_builders import mamba_builder
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine, StaticInferenceEngine
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.inference_request import (
    DynamicInferenceRequestRecord,
    InferenceRequest,
)
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from model_provider import model_provider

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from functools import partial
from typing import List

from megatron.core import mpu
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

REQUEST_ID = 0


def add_inference_benchmarking_args(parser):
    """Inference benchmarking arguments."""
    parser = add_inference_args(parser)

    group = parser.add_argument_group(title='inference_benchmarking')

    group.add_argument(
        "--num-input-tokens", type=int, default=128, help="Number of input tokens per request"
    )
    group.add_argument(
        "--engine-type", choices=["static", "dynamic"], default="static", help="Engine type"
    )
    group.add_argument(
        "--benchmark-profile", action="store_true", default=False, help="If set, profile"
    )
    return parser


def get_inference_engine(args: argparse.Namespace, model: MegatronModule) -> AbstractEngine:
    """Utility to get the relevant backend for running inference

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model .

    Returns:
        AbstractBackend: The chosen backend
    """

    if args.engine_type == "static":
        tokenizer = get_tokenizer()
        context = StaticInferenceContext(
            args.inference_max_requests, args.inference_max_sequence_length
        )
        inference_wrapped_model = GPTInferenceWrapper(model, context)
        inference_wrapped_model.model_is_pipeline_parallel = not (
            mpu.is_pipeline_first_stage() and mpu.is_pipeline_last_stage()
        )
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
        )
        return StaticInferenceEngine(text_generation_controller=text_generation_controller)
    elif args.engine_type == "dynamic":
        return get_dynamic_inference_engine(model=model)


def get_random_prompt_tokens(tokenizer, num_input_tokens) -> List[int]:
    # Get the set of special token IDs to exclude
    special_token_ids = set()
    try:
        if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
            special_token_ids.add(tokenizer.bos)
        if hasattr(tokenizer, 'eos') and tokenizer.eos is not None:
            special_token_ids.add(tokenizer.eos)
        if hasattr(tokenizer, 'eod') and tokenizer.eod is not None:
            special_token_ids.add(tokenizer.eos)
        if (
            hasattr(tokenizer, 'additional_special_tokens_ids')
            and tokenizer.additional_special_tokens_ids
        ):
            special_token_ids.update(tokenizer.additional_special_tokens_ids)
    except NotImplementedError as e:
        pass

    # Create a list of valid token IDs
    valid_token_ids = [i for i in range(tokenizer.vocab_size) if i not in special_token_ids]

    # Randomly sample tokens from the valid tokens
    prompt_tokens = random.choices(valid_token_ids, k=num_input_tokens)
    assert len(prompt_tokens) == num_input_tokens

    return prompt_tokens


def generate_dynamic(
    args: argparse.Namespace,
    inference_requests: List[InferenceRequest],
    inference_engine: DynamicInferenceEngine,
):
    global REQUEST_ID
    for request in inference_requests:
        request_id = REQUEST_ID
        REQUEST_ID += 1
        prompt_tokens = request.prompt_tokens
        inference_engine.add_request(request_id, prompt_tokens, request.inference_parameters)

    start_time = time.perf_counter()
    all_finished_requests = []
    while inference_engine.has_unfinished_requests():
        result = inference_engine.step()
        finished_requests = result["finished_requests"]
        for request in finished_requests:
            req_id = request.request_id
            latency = time.perf_counter() - start_time
            print(
                f"[{time.ctime()}] Request {req_id} finished in {latency} seconds and "
                f"generated {request.generated_length} tokens"
            )
        all_finished_requests.extend(finished_requests)

    return all_finished_requests


@torch.inference_mode()
def main():
    """Main program."""

    initialize_megatron(
        extra_args_provider=add_inference_benchmarking_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
        },
    )

    args = get_args()

    # Set up model and load checkpoint
    if args.model_provider == "gpt":
        model_builder = gpt_builder
    elif args.model_provider == "mamba":
        model_builder = mamba_builder
    else:
        raise ValueError(f"Invalid model provider {args.model_provider}")

    model = get_model(partial(model_provider, model_builder), wrap_with_ddp=False)
    tokenizer = get_tokenizer()
    load_checkpoint(model, None, None)
    model = model[0]
    model.eval()

    assert (args.prompts is None) ^ (
        args.num_input_tokens is None
    ), "Exactly one of `--prompts` and `--num-prompt-tokens` must be specified"

    inference_engine = get_inference_engine(args, model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        top_n_logprobs=args.top_n_logprobs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=-1,
    )
    sampling_params.add_attributes({"no_early_termination": True})

    requests = []
    if args.num_input_tokens is not None:
        assert args.prompts is None
        batch_size = args.inference_max_requests
        for i in range(batch_size):
            prompt_tokens = get_random_prompt_tokens(tokenizer, args.num_input_tokens)
            requests.append(
                InferenceRequest(
                    request_id=str(time.monotonic()),
                    prompt=tokenizer.detokenize(prompt_tokens),
                    prompt_tokens=prompt_tokens,
                    inference_parameters=sampling_params,
                )
            )
    else:
        assert args.prompts is not None
        for prompt in args.prompts:
            requests.append(
                InferenceRequest(
                    request_id=str(time.monotonic()),
                    prompt=prompt,
                    prompt_tokens=tokenizer.tokenize(prompt),
                    inference_parameters=sampling_params,
                )
            )

    # TODO(ksanthanam): Use a command line argument for warmup iterations
    for i in range(3):
        print(f"Running warmup iteration {i+1}...")
        warmup_sampling_params = SamplingParams(num_tokens_to_generate=10, termination_id=-1)
        inference_engine.generate(prompts=["warmup"], sampling_params=warmup_sampling_params)

    if args.benchmark_profile:
        torch.cuda.cudart().cudaProfilerStart()

    start_time = time.perf_counter()
    if args.engine_type == "static":
        results: List[InferenceRequest] = inference_engine.generate(
            prompts=args.prompts, inference_requests=requests, sampling_params=sampling_params
        )
    else:
        prompts = [request.prompt_tokens for request in requests]
        records: List[DynamicInferenceRequestRecord] = inference_engine.generate(
            prompts=prompts, sampling_params=sampling_params
        )
        results: List[InferenceRequest] = [record.merge() for record in records]

    end_time = time.perf_counter()
    latency = end_time - start_time

    memory_allocated = torch.cuda.max_memory_allocated()

    if args.benchmark_profile:
        torch.cuda.cudart().cudaProfilerStop()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            generated_log_probs = result.generated_log_probs
            result_dict = {
                'id': result.request_id,
                'num_input_tokens': len(result.prompt_tokens),
                'num_output_tokens': len(result.generated_tokens),
                'tpot': result.tpot,
                'latency': latency,
                'memory_usage_GB': memory_allocated / (1024**3),
            }
            if args.prompts is not None:
                result_dict['generated_output'] = tokenizer.detokenize(result.generated_tokens)
            print(result_dict)

    total_output_tokens = args.num_tokens_to_generate * args.inference_max_requests
    throughput = total_output_tokens / latency
    print(f"Throughput: {throughput} output tokens / second")


if __name__ == "__main__":
    main()
