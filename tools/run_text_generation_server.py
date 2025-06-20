# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate"""
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import os
import sys
from argparse import Namespace
from contextlib import nullcontext

from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
import torch

from pretrain_gpt import model_provider as gpt_model_provider
from pretrain_mamba import model_provider as mamba_model_provider

from megatron.core.inference.engines import AbstractEngine, StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.training import get_model
from megatron.core.transformer.module import MegatronModule
from megatron.inference.text_generation import beam_search_and_post_process
from megatron.inference.text_generation.mcore_engine_server import (
    ModelInferenceWrapperServer,
    run_mcore_engine,
)
from megatron.inference.text_generation_server import MegatronServer
from megatron.training import print_rank_0

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from megatron.core import mpu
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference

    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore
    backend if the user does not specify any backends. TRTLLMBackend is not implmented yet.

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model.

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
        inference_max_seq_length=args.inference_max_seq_length,
        inference_max_requests=args.inference_max_batch_size,
        nccl_all_reduce_for_prefill=args.nccl_all_reduce_for_prefill,
    )

    inference_wrapped_model = ModelInferenceWrapperServer(model, inference_wrapper_config)
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return StaticInferenceEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=args.inference_max_batch_size,
    )


def add_text_generate_args(parser):
    """Adds text generation arguments to parser."""
    group = parser.add_argument_group(title='text generation')
    group.add_argument(
        "--port", type=int, default=5000, help='port for text generation server to run on'
    )
    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=True,
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
        nargs='+',
        help='Input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help='Deprecated in favor of `--inference-max-batch-size`',
    )
    return parser


@torch.inference_mode()
def main(model_provider: str = "gpt"):
    """Runs the text generation server with the specified model provider."""
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'exit_on_missing_checkpoint': True,
        },
    )
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text " "generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    load_context = nullcontext()
    if args.fp8:
        from transformer_engine.pytorch.fp8 import fp8_model_init

        load_context = fp8_model_init()
    with load_context:
        if model_provider == "gpt":
            model = get_model(gpt_model_provider, wrap_with_ddp=False)
        elif model_provider == "mamba":
            model = get_model(mamba_model_provider, wrap_with_ddp=False)
        else:
            raise ValueError(f"Invalid model provider {model_provider}")

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    model.eval()

    if args.max_batch_size is not None:
        assert args.inference_max_batch_size is not None
        args.inference_max_batch_size = max(args.inference_max_batch_size, args.max_batch_size)
        warnings.warn(
            "`--max-batch-size` has been deprecated in favor of `--inference-max-requests`, "
            f"setting maximum batch size to {args.inference_max_batch_size}"
        )

    inference_engine = get_inference_engine(args, model)

    if args.enable_cuda_graph:
        print(f"Running warmup for CUDA graphs...")
        inference_engine.generate(
            prompts=["Test prompt"], sampling_params=SamplingParams(num_tokens_to_generate=10)
        )

    if (
        mpu.is_pipeline_first_stage()
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_expert_model_parallel_rank() == 0
    ):
        server = MegatronServer(inference_engine, args)
        server.run("0.0.0.0", port=args.port)

    while True:
        choice = torch.tensor(1, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        if choice.item() == 0:
            try:
                run_mcore_engine(inference_engine)
            except ValueError as ve:
                pass
        elif choice.item() == 1:
            try:
                beam_search_and_post_process(
                    inference_engine.text_generation_controller.inference_wrapped_model.model
                )
            except ValueError as ve:
                pass


if __name__ == "__main__":
    main(model_provider="gpt")
