# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate"""
import os
import sys
import warnings
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import os
import sys
from argparse import Namespace
from contextlib import nullcontext

from megatron.core.utils import get_attr_wrapped_model
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
import torch

from model_provider import model_provider
from gpt_builders import gpt_builder
from mamba_builders import mamba_builder

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.engines import (
    AbstractEngine,
    DynamicInferenceEngine,
    StaticInferenceEngine,
)
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

from typing import Optional, List, Tuple


def get_dynamic_inference_context(
    args: Namespace,
    calculate_max_sequence_length_from_requests: bool = True,
    layer_type_list: Optional[List[str]] = None,
    mamba_conv_states_shape: Optional[Tuple[int]] = None,
    mamba_ssm_states_shape: Optional[Tuple[int]] = None,
):
    """The inference context manages the KV cache and other inference state."""

    # Max sequence length.
    max_sequence_length = args.inference_max_seq_length

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
            args.inference_dynamic_batching_num_cuda_graphs if args.enable_cuda_graph else None
        ),
        buffer_size_gb=args.inference_dynamic_batching_buffer_size_gb,
        buffer_guaranteed_fraction=args.inference_dynamic_batching_buffer_guaranteed_fraction,
        chunk_size_tokens=args.inference_dynamic_batching_chunk_size,
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
    )

    return context


def get_inference_controller(
    args: Namespace, model: MegatronModule, context: DynamicInferenceContext
) -> TextGenerationController:
    """Buid text generation controller, which manages the model inference context.

    Args:
        model (MegatronModule): Megatron GPT model.
        context (DynamicInferenceContext): Context for managing KV cache.

    Return:
        (TextGenerationController) Inference text generation controller.
    """

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


def get_dynamic_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
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

    # Layer type list for hybrid models
    decoder = get_attr_wrapped_model(model, "decoder")
    layer_type_list = getattr(decoder, "layer_type_list", None)
    if layer_type_list is not None and Symbols.MAMBA in layer_type_list:
        (mamba_conv_states_shape, mamba_ssm_states_shape) = decoder.mamba_state_shapes_per_request()
    else:
        mamba_conv_states_shape = None
        mamba_ssm_states_shape = None

    context = get_dynamic_inference_context(
        args,
        layer_type_list=layer_type_list,
        mamba_conv_states_shape=mamba_conv_states_shape,
        mamba_ssm_states_shape=mamba_ssm_states_shape,
    )
    controller = get_inference_controller(args, model, context)

    return DynamicInferenceEngine(
        controller,
        context,
        termination_id=tokenizer.eod,
        enable_cuda_graph=args.enable_cuda_graph,
        random_seed=args.seed,
    )


def get_static_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference

    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore
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

    inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return StaticInferenceEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=args.inference_max_batch_size,
    )


def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    if args.inference_dynamic_batching:
        return get_dynamic_inference_engine(args, model)
    else:
        return get_static_inference_engine(args, model)


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
def main(model_type: str = "gpt"):
    """Runs the text generation server with the specified model type."""
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
        if model_type == "gpt":
            model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
        elif model_type == "mamba":
            model = get_model(partial(model_provider, mamba_builder), wrap_with_ddp=False)
        else:
            raise ValueError(f"Invalid model type {model_type}")

    if args.load is not None:
        _ = load_checkpoint(model, None, None, strict=False)

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

    torch.cuda.synchronize()
    print(f"Successfully warmed up CUDA graphs")

    if (
        mpu.is_pipeline_first_stage()
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_expert_model_parallel_rank() == 0
    ):
        print(f"Rank {torch.distributed.get_rank()} starting server...")
        server = MegatronServer(inference_engine, args)
        server.run("0.0.0.0", port=args.port)

    while True:
        if torch.distributed.get_rank() == 0:
            break
        choice = torch.tensor(0, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        try:
            print(f"Rank {torch.distributed.get_rank()} calling run_mcore_engine...")
            run_mcore_engine(inference_engine)
        except ValueError as ve:
            print(f"Rank {torch.distributed.get_rank()}: Failed to run engine: {ve}")


if __name__ == "__main__":
    main(model_type="gpt")
