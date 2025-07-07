import os
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
import argparse
from collections import OrderedDict
from pretrain_gpt import model_provider as gpt_model_provider
from pretrain_mamba import model_provider as mamba_model_provider
import random
import torch
import sys
import time
import tqdm
import warnings
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines import DynamicInferenceEngine, StaticInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model, get_tokenizer
import asyncio
from typing import AsyncIterator, List, Union

REQUEST_ID = 0


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=False,
        help='Return the log probabilities of the final output tokens',
    )
    group.add_argument(
        "--top-n-logprobs",
        type=int,
        default=0,
        help="Top-N logprobs"
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
        help='Number of tokens to generate for each prompt',
    )
    group.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        default=None,
        nargs='+',
        help='Input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--num-input-tokens", type=int, default=None, help='Number of input tokens per prompt'
    )
    group.add_argument("--stream", action="store_true", default=False, help="Stream output tokens")
    group.add_argument(
        "--model-provider", choices=["mamba", "gpt"], default="gpt", help="Model provider"
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

    This function will automatically chose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. TRT LLM Backend is not implmented yet.

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

    if args.engine_type == "static":
        inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
        inference_wrapped_model.model_is_pipeline_parallel = not (
            mpu.is_pipeline_first_stage() and mpu.is_pipeline_last_stage()
        )
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
        )
        return StaticInferenceEngine(text_generation_controller=text_generation_controller)
    elif args.engine_type == "dynamic":
        context = DynamicInferenceContext(
            params_dtype=args.params_dtype,
            num_layers=args.num_layers,
            kv_channels=args.kv_channels,
            num_attention_heads=(
                args.num_query_groups if args.group_query_attention else args.num_attention_heads
            ),
            max_sequence_length=args.inference_max_seq_length,
            buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
            buffer_guaranteed_fraction=args.inference_dynamic_batching_buffer_guaranteed_fraction,
            buffer_overflow_factor=args.inference_dynamic_batching_buffer_overflow_factor,
            max_requests_override=args.inference_dynamic_batching_max_requests_override,
            max_tokens_override=args.inference_dynamic_batching_max_tokens_override,
            chunk_size_tokens=args.inference_dynamic_batching_chunk_size,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            model, inference_wrapper_config, inference_context=context
        )
        inference_wrapped_model.model_is_pipeline_parallel = not (
            mpu.is_pipeline_first_stage() and mpu.is_pipeline_last_stage()
        )
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
        )
        return DynamicInferenceEngine(
            text_generation_controller,
            context,
            termination_id=-1,
            enable_cuda_graph=args.enable_cuda_graph,
            random_seed=args.seed,
        )


async def generate(
    inference_engine: Union[StaticInferenceEngine, DynamicInferenceEngine],
    sampling_params: SamplingParams,
    prompts: List[str],
    inference_requests: List[InferenceRequest] = None,
) -> List[InferenceRequest]:
    async def collect_stream(prompt, request_id, stream_generator):
        async for output in stream_generator:
            pass

    if inference_requests is None:
        assert prompts is not None
        inference_requests = [None for _ in range(len(prompts))]
    elif prompts is None:
        assert inference_requests is not None
        tokenizer = get_tokenizer()
        prompts = [tokenizer.detokenize(request.prompt_tokens) for request in inference_requests]

    request_ids: List[str] = [
        inference_engine.add_request(
            prompt=prompt,
            inference_request=inference_request,
            inference_parameters=sampling_params,
            streaming=True,
        )
        for prompt, inference_request in zip(prompts, inference_requests)
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
    sampling_params: SamplingParams,
):
    global REQUEST_ID
    req_data = OrderedDict()
    for request in inference_requests:
        request_id = REQUEST_ID
        REQUEST_ID += 1
        prompt_tokens = request.prompt_tokens
        inference_engine.add_request(
            request_id, prompt_tokens, num_tokens_to_generate=args.num_tokens_to_generate
        )
        cur_time = time.perf_counter()
        req_data[request_id] = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": [],
            "tpot": [],
            "prev_time": cur_time,
            "start_time": cur_time,
        }

    while inference_engine.has_unfinished_requests():
        result, _ = inference_engine.step(sampling_params, verbose=False)
        if result is not None:
            request_ids, finished_request_ids, sample = result

            request_ids = request_ids.tolist()
            sample = sample.tolist()

            cur_time = time.perf_counter()
            for req_id, token in zip(request_ids, sample):
                req_data[req_id]["output_tokens"].append(token)
                req_data[req_id]["tpot"].append(cur_time - req_data[req_id]["prev_time"])
                req_data[req_id]["prev_time"] = cur_time
                if req_id in finished_request_ids:
                    req_data[req_id]["finish_time"] = time.perf_counter()
                    latency = req_data[req_id]["finish_time"] - req_data[req_id]["start_time"]
                    print(
                        f"[{time.ctime()}] Request {req_id} finished in {latency} seconds and generated {len(req_data[req_id]['tpot'])} tokens"
                    )

    return [
        InferenceRequest(
            prompt="",
            request_id=str(request_id),
            prompt_tokens=data["prompt_tokens"],
            generated_tokens=data["output_tokens"],
        )
        for request_id, data in req_data.items()
    ]


@torch.inference_mode()
def main():
    """Main program."""

    # Note: The default args passed here can be overwritten by using appropriate params (check arguments.py file)
    # Micro batch size is not needed to be set by user. (It is calculated based on inference-batch-times-seqlen-threshold argument)
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
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
        model_provider = gpt_model_provider
    elif args.model_provider == "mamba":
        model_provider = mamba_model_provider

    model = get_model(model_provider, wrap_with_ddp=False)
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
    )

    requests = []
    if args.num_input_tokens is not None:
        assert args.prompts is None
        batch_size = args.inference_max_batch_size
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

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        if args.engine_type == "static":
            inference_engine.generate(
                prompts=None, inference_requests=requests, sampling_params=sampling_params
            )
        elif args.engine_type == "dynamic":
            generate_dynamic(args, requests, inference_engine, sampling_params)

    if args.benchmark_profile:
        torch.cuda.cudart().cudaProfilerStart()

    start_time = time.perf_counter()
    if args.stream:
        if args.engine_type == "dynamic":
            raise NotImplementedError("Streaming not supported with DynamicInferenceEngine")
        results: List[InferenceRequest] = asyncio.run(
            generate(
                inference_engine, sampling_params, prompts=args.prompts, inference_requests=requests
            )
        )
    else:
        if args.engine_type == "static":
            results: List[InferenceRequest] = inference_engine.generate(
                prompts=args.prompts, inference_requests=requests, sampling_params=sampling_params
            )
        elif args.engine_type == "dynamic":
            results: List[InferenceRequest] = generate_dynamic(
                args, requests, inference_engine, sampling_params
            )
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
                'latency': latency,
                'memory_usage_GB': memory_allocated / (1024**3),
            }
            if args.prompts is not None:
                result_dict['generated_output'] = tokenizer.detokenize(result.generated_tokens)
            print(result_dict)


if __name__ == "__main__":
    main()
