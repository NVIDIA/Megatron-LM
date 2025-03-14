import os
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
import argparse
from pretrain_gpt import model_provider as gpt_model_provider
from pretrain_mamba import model_provider as mamba_model_provider
import random
import torch
import sys
import time
import tqdm
import warnings
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
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

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model, get_tokenizer
import asyncio
from typing import AsyncIterator, List


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
    group.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        dest="inference_max_requests",
        help='Max number of prompts to process at once',
    )
    group.add_argument("--stream", action="store_true", default=False, help="Stream output tokens")
    group.add_argument(
        "--model-provider", choices=["mamba", "gpt"], default="gpt", help="Model provider"
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
        inference_max_requests=args.inference_max_requests,
        inference_max_seq_length=args.inference_max_seq_length,
    )

    inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return MCoreEngine(text_generation_controller=text_generation_controller)


async def generate(
    inference_engine: MCoreEngine, sampling_params: SamplingParams, prompts: List[str]
) -> List[InferenceRequest]:
    async def collect_stream(prompt, request_id, stream_generator):
        print(f"Request {request_id}: {prompt}", end="", flush=True)
        prev_idx = 0
        async for output in stream_generator:
            print(output.generated_text[prev_idx:], end="", flush=True)
            prev_idx = len(output.generated_text)
        print()

    request_ids: List[str] = [
        inference_engine.add_request(
            prompt=prompt, inference_parameters=sampling_params, streaming=True
        )
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

    assert (args.prompts is None) ^ (
        args.num_input_tokens is None
    ), "Exactly one of `--prompts` and `--num-prompt-tokens` must be specified"

    inference_engine = get_inference_engine(args, model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )

    requests = None
    if args.num_input_tokens is not None:
        requests = []
        batch_size = args.inference_max_requests
        for i in range(batch_size):
            prompt_tokens = get_random_prompt_tokens(tokenizer, args.num_input_tokens)
            requests.append(
                InferenceRequest(
                    request_id=inference_engine.get_new_request_id(),
                    prompt=tokenizer.detokenize(prompt_tokens),
                    prompt_tokens=prompt_tokens,
                    inference_parameters=sampling_params,
                )
            )
    assert (args.prompts is None) ^ (requests is None)

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
            prompts=args.prompts, inference_requests=requests, sampling_params=sampling_params
        )

    start_time = time.perf_counter()
    if args.stream:
        results: List[InferenceRequest] = asyncio.run(
            generate(
                inference_engine, sampling_params, prompts=args.prompts, inference_requests=requests
            )
        )
    else:
        results: List[InferenceRequest] = inference_engine.generate(
            prompts=args.prompts, inference_requests=requests, sampling_params=sampling_params
        )
    end_time = time.perf_counter()
    latency = end_time - start_time

    memory_allocated = torch.cuda.max_memory_allocated()

    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            generated_log_probs = result.generated_log_probs
            result = {
                'id': result.request_id,
                'input_prompt': result.prompt,
                'generated_text': result.generated_text,
                'generated_tokens': result.generated_tokens,
                'latency': latency,
                'memory_usage_GB': memory_allocated / (1024**3),
            }
            if args.return_log_probs:
                result['generated_log_probs'] = generated_log_probs
            print(result)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
