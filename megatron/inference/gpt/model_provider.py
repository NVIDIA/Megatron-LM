# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""ModelOpt GPT model provider."""

import modelopt.torch.opt as mto

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.inference.ammo_support.gpt.model_specs import get_gpt_layer_ammo_spec
from megatron.core.inference.ammo_support.gpt.state_dict_hooks import (
    mcore_gpt_load_classic_state_dict_pre_hook,
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.transformer.spec_utils import import_module
from megatron.inference.checkpointing import load_modelopt_state
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args


def model_provider(pre_process=True, post_process=True, parallel_output=True) -> MCoreGPTModel:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the core GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits? This must be
            True if `model_provider` is called in text_generation_server.

    Returns:
        MCoreGPTModel: The returned model
    """
    args = get_args()

    print_rank_0("building GPT model ...")

    # ModelOpt by default assumes none homogenous layers. This affect the storage format of the sharded checkpoint.
    config = core_transformer_config_from_args(args)
    config.non_homogeneous_layers = True

    if args.use_legacy_models:
        raise ValueError(
            "ModelOpt integration only support MCore models. Use --use-mcore-modules instead."
        )

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_modelopt_spec(
            remap_te_layernorm=args.export_te_mcore_model, qk_layernorm=False,
        )

    model_type = MCoreGPTModel
    model_kwargs = {
        "config": config,
        "transformer_layer_spec": transformer_layer_spec,
        "vocab_size": args.padded_vocab_size,
        "max_sequence_length": args.max_position_embeddings,
        "pre_process": pre_process,
        "post_process": post_process,
        "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
        "parallel_output": parallel_output,
        "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
        "position_embedding_type": args.position_embedding_type,
        "rotary_percent": args.rotary_percent,
    }

    model = model_type(**model_kwargs)

    # Load modelopt_state
    modelopt_state = load_modelopt_state() if args.load else {}
    if modelopt_state:
        model = mto.restore_from_modelopt_state(model, modelopt_state)

    # Register some load_state_dict prehooks to handle some known state_dict key mismatch.
    # (legacy <-> modelopt) and (default te <-> modelopt)
    if args.export_legacy_megatron:
        model._register_load_state_dict_pre_hook(mcore_gpt_load_legacy_state_dict_pre_hook)
    if args.export_te_mcore_model:
        model._register_load_state_dict_pre_hook(mcore_gpt_load_te_state_dict_pre_hook)

    # Print models on all pp ranks.
    if get_tensor_model_parallel_rank() == 0:
        print(str(model))

    return model
