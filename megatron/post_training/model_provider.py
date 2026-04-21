# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""ModelOpt GPT model provider."""

import os
from argparse import Namespace
from typing import Any, Dict

import modelopt.torch.distill as mtd
import modelopt.torch.opt as mto
import yaml

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.models.mamba import MambaModel as MCoreMambaModel
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.post_training.modelopt.gpt.state_dict_hooks import (
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.post_training.algos import distillation
from megatron.post_training.checkpointing import load_modelopt_checkpoint, load_modelopt_state
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args


def _add_load_convert_hooks(model: MCoreGPTModel):
    """Register some load_state_dict prehooks to handle some known state_dict key mismatch.
    """
    args = get_args()
    if args.export_te_mcore_model:
        model._register_load_state_dict_pre_hook(mcore_gpt_load_te_state_dict_pre_hook)


def _load_teacher_model_config(checkpoint_path: str) -> Namespace:
    """Reads teacher config from a file.

    The file named ``model_config.yaml`` within the checkpoint directory should specify
    (in NEMO format) any model architecture settings which differ from the main student model's.
    This function will translate NEMO field names to MCore as needed.
    """
    required_teacher_fields = (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
    )

    config_path = os.path.join(checkpoint_path, "model_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Teacher checkpoint dir must contain a NEMO-format yaml config named 'model_config.yaml'"
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    missing_keys = [k for k in required_teacher_fields if k not in config]
    if missing_keys:
        raise ValueError(
            f"Teacher `model_config.yaml` file missing the following fields: {missing_keys}"
        )

    if "encoder_seq_length" in config:
        config["seq_length"] = config["encoder_seq_length"]
    if "bias" in config:
        config["disable_bias_linear"] = not config["bias"]
    if config.get("activation") == "swiglu":
        config["swiglu"] = True
    if config.get("position_embedding_type", False) is None:
        config["use_rotary_position_embeddings"] = config["no_position_embedding"] = True
    if "share_embeddings_and_output_weights" in config:
        config["untie_embeddings_and_output_weights"] = not config[
            "share_embeddings_and_output_weights"
        ]
    if "tokenizer" in config:
        config["tokenizer_type"] = config["tokenizer"]["type"]
        config["tokenizer_model"] = config["tokenizer"]["model"]
    if "masked_softmax_fusion" in config:
        config["no_masked_softmax_fusion"] = not config["masked_softmax_fusion"]
    if config.get("normalization") == "layernorm1p":
        config["apply_layernorm_1p"] = True
    if "precision" in config:
        config[config["precision"]] = True
    if "mcore_gpt" in config:
        config["use_mcore_models"] = config["mcore_gpt"]

    args_dict = vars(get_args()).copy()
    del args_dict["kv_channels"]  # not recalculated if present
    args_dict.update(config)

    return Namespace(**args_dict)


def _teacher_provider(config: Namespace, model_kwargs: Dict[str, Any]) -> MCoreGPTModel:
    """Teacher model factory (must be a non-local function to pickle)."""
    args = get_args()

    # Convert to `TransformerConfig` here to avoid ModelOpt pickling issues (contains local functions)
    config = core_transformer_config_from_args(config)

    teacher = MCoreGPTModel(config=config, **model_kwargs)
    _add_load_convert_hooks(teacher)

    print_rank_0("Loading teacher checkpoint...")
    # [WAR]: load checkpoint will check checkpoint's saved args and rng state if not finetune.
    # To avoid error out on loading teacher's checkpoint, we temporarily set args.finetune to
    # True while loading the teacher checkpoint.
    original_args_finetune, original_ckpt_format = args.finetune, args.ckpt_format
    args.finetune = True
    if args.export_kd_teacher_ckpt_format is not None:
        args.ckpt_format = args.export_kd_teacher_ckpt_format
    load_modelopt_checkpoint([teacher], load_arg='export_kd_teacher_load')
    args.finetune, args.ckpt_format = original_args_finetune, original_ckpt_format

    return teacher


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

    if args.use_legacy_models:
        raise ValueError(
            "ModelOpt integration only support MCore models. Use --use-mcore-modules instead."
        )
    if args.spec is not None:
        raise ValueError("ModelOpt integration does not support custom args.spec.")

    # Llama-4 Scout/Maverick support
    config.qk_l2_norm = args.export_qk_l2_norm 
    config.moe_apply_probs_on_input = args.export_moe_apply_probs_on_input 

    if args.export_model_type == "GPTModel":
        if config.heterogeneous_block_specs:
            transformer_layer_spec = get_gpt_heterogeneous_layer_spec(
                config=config,
                use_te=args.transformer_impl == "transformer_engine",
            )
        else:
            transformer_layer_spec = get_gpt_modelopt_spec(
                config=config,
                local_core_attention=args.export_force_local_attention,
                remap_te_layernorm=args.export_te_mcore_model,
                real_quant_cfg=args.export_real_quant_cfg,
                use_arbitrary_attention_mask=True,
            )
        model_kwargs = {
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
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }
        model = MCoreGPTModel(config=config, **model_kwargs)
    elif args.export_model_type == "MambaModel":
        mamba_stack_spec = get_mamba_stack_modelopt_spec(
            remap_te_layernorm=args.export_te_mcore_model
        )
        model = MCoreMambaModel(
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
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
        )
    else:
        raise ValueError("ModelOpt does not support model type {}".format(args.export_model_type))

    # import modelopt.torch.speculative as mtsp
    # config = {"eagle_num_layers": 1}
    # model = mtsp.convert(model, [("eagle", config)])

    # Load modelopt_state
    modelopt_state = load_modelopt_state(model=model) if args.load else {}
    if modelopt_state:
        model = mto.restore_from_modelopt_state(model, modelopt_state)

    _add_load_convert_hooks(model)

    # Distillation mode.
    if args.export_kd_teacher_load:
        print_rank_0("Distillation: Enabled.")

        # NOTE: Unknown memory leak occuring per fwd-bwd pass if model
        # is converted to a `modelopt.torch.opt.DynamicModule`.
        # Argument `--manual-gc` can result in an eventual OOM.
        assert (
            not args.manual_gc
        ), "ModelOpt Distillation currently incompatible with `--manual-gc` option."
        assert (
            not args.tp_comm_overlap
        ), "ModelOpt Distillation currently incompatible with `--tp-comm-overlap` option."
        if args.pipeline_model_parallel_size > 1:
            assert (
                args.virtual_pipeline_model_parallel_size is None
            ), "ModelOpt Distillation currently incompatible with interleaved pipeline schedule."

        teacher_config = _load_teacher_model_config(args.export_kd_teacher_load)
        distill_cfg = distillation.load_distillation_config(
            args.export_kd_cfg, student_cfg=config, teacher_cfg=core_transformer_config_from_args(teacher_config)
        )
        kd_config = {
            "teacher_model": (_teacher_provider, [teacher_config, model_kwargs], {}),
            "criterion": distill_cfg["criterion"],
            "loss_balancer": distill_cfg["loss_balancer"],
        }
        model = mtd.convert(model, mode=[("kd_loss", kd_config)])

        # Additional tweaks needed for MCore/Nemo.
        # NOTE: Distillation state manually removed in this function.
        # ModelOpt state restoration above will not return a `mtd.DistillationModel` for simplicity reasons.
        distillation.adjust_distillation_model_for_mcore(model, distill_cfg)

    return model
