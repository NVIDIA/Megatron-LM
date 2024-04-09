# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""ModelOpt GPT model provider."""

from typing import Union

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.inference.ammo_support.gpt.model_specs import get_gpt_layer_ammo_spec
from megatron.core.inference.ammo_support.gpt.state_dict_hooks import (
    mcore_gpt_load_classic_state_dict_pre_hook,
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.models.gpt import GPTModel as MCoreGPTModel


def model_provider(
    pre_process=True, post_process=True, parallel_output=True,
) -> Union[MCoreGPTModel]:
    """Builds the GPT model.

    This model_provider only sypport use_mcore_models=True.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits? This must be
            True if `model_provider` is called in text_generation_server.

    Returns:
        Union[MCoreGPTModel]: The returned model
    """
    args = get_args()

    print_rank_0("building GPT model ...")
    config = core_transformer_config_from_args(get_args())

    if args.use_mcore_models:
        if args.spec is not None:
            raise ValueError("Custom layer specs are not supported!")
        else:
            if args.num_experts is None:
                transformer_layer_spec = get_gpt_layer_ammo_spec()
            else:
                raise ValueError("MoE is not supported for now!")

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
    else:
        raise ValueError("Classic Megatron-LM models are not supported!")

    model = model_type(**model_kwargs)
    print_rank_0(str(model))

    if args.use_mcore_models:
        if args.ammo_load_classic_megatron_to_mcore:
            model._register_load_state_dict_pre_hook(mcore_gpt_load_classic_state_dict_pre_hook)
        elif args.ammo_convert_te_to_local_spec:
            model._register_load_state_dict_pre_hook(mcore_gpt_load_te_state_dict_pre_hook)

    return model
