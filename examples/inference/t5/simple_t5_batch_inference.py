import os
import sys
from argparse import Namespace

import torch

import pretrain_t5
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.engines import AbstractEngine, StaticInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.model_inference_wrappers.t5.t5_inference_wrapper import (
    T5InferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.encoder_decoder_text_generation_controller import (
    EncoderDecoderTextGenerationController,
)
from megatron.core.transformer.module import MegatronModule
from pretrain_t5 import model_provider

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from typing import List

from megatron.core import mpu
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


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
        "--encoder-prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Encoder input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--max-batch-size", type=int, default=1, help='Max number of prompts to process at once'
    )
    return parser


def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
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
    )

    inference_wrapped_model = T5InferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = EncoderDecoderTextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return StaticInferenceEngine(
        text_generation_controller=text_generation_controller, max_batch_size=args.max_batch_size
    )


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

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]

    args = get_args()

    inference_engine = get_inference_engine(args, model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
    )

    tokenizer = get_tokenizer()
    decoder_prompts = [""] * len(
        args.encoder_prompts
    )  # for T5, the prompt is provided as encoder input, hence decoder_prompts is empty
    args.prompts = decoder_prompts

    results: List[InferenceRequest] = inference_engine.generate(
        prompts=args.prompts,
        add_BOS=True,
        encoder_prompts=args.encoder_prompts,
        sampling_params=sampling_params,
    )

    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            result = {
                'id': result.request_id,
                'input_prompt': result.prompt,
                'generated_text': result.generated_text,
                'generated_tokens': result.generated_tokens,
            }
            print(result)


if __name__ == "__main__":
    main()
