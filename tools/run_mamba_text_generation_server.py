# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate Mamba"""
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from argparse import Namespace

from megatron.core import mpu
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import MegatronModule
from megatron.inference.text_generation.mcore_engine_server import ModelInferenceWrapperServer, run_mcore_engine
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.inference.text_generation_server import MegatronServer
from megatron.core.transformer import TransformerConfig


def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            print_rank_0(f" - {name}: {param.numel()}")
    return num_params


# Taken from pretrain_mamba.py
def model_provider(pre_process=True, post_process=True) -> MambaModel:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        MambaModel: The returned model
    """
    args = get_args()

    print_rank_0('building Mamba model ...')
    config = core_transformer_config_from_args(args, TransformerConfig)

    assert args.use_legacy_models == False, "Mamba only supported in Mcore!"

    if args.spec is not None:
        mamba_stack_spec = import_module(args.spec)
    else:
        raise ValueError("You must provide a valid Mamba layer spec!")

    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=False,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")
    return model


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
    )

    inference_wrapped_model = ModelInferenceWrapperServer(model, inference_wrapper_config)
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )
    return MCoreEngine(
        text_generation_controller=text_generation_controller, max_batch_size=args.max_batch_size
    )


def add_text_generate_args(parser):
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
        "--max-batch-size", type=int, default=8, help='Max number of prompts to process at once'
    )
    return parser


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
        },
    )

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text " "generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    model.eval()

    inference_engine = get_inference_engine(args, model)

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
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
