# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""ModelOpt GPT model provider."""

import os
from argparse import Namespace
from typing import Any, Dict

import modelopt.torch.distill as mtd
import modelopt.torch.distill.plugins.megatron as mtd_mcore
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
from megatron.post_training.checkpointing import load_modelopt_checkpoint, load_modelopt_state
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args


def count_parameters_in_layer(model, layer_name):
    num_params = 0
    for name, param in model.named_parameters():
        if layer_name in name:
            num_params += param.numel()
            print_rank_0(f" - {name}: {param.numel()}")
    return num_params


def _add_load_convert_hooks(model: MCoreGPTModel):
    """Register some load_state_dict prehooks to handle some known state_dict key mismatch.
    """
    args = get_args()
    if args.export_te_mcore_model:
        model._register_load_state_dict_pre_hook(mcore_gpt_load_te_state_dict_pre_hook)


def _load_teacher_model_config(checkpoint_path: str) -> Namespace:
    """Reads teacher config from a file.

    The config provided via --teacher-model-config should specify
    (in NEMO format) any model architecture settings which differ from the main student model's.
    This function will translate NEMO field names to MCore as needed.
    """
    required_teacher_fields = (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
    )

    args = get_args()
    config_path = os.path.join(checkpoint_path, "model_config.yaml") if args.teacher_model_config is None else args.teacher_model_config
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


def _load_teacher_model(config, config_raw: Namespace, model_kwargs: Dict[str, Any]) -> MCoreGPTModel:
    """Teacher model creator."""
    args = get_args()

    if config.is_hybrid_model:
        # These parameters are not part of the TransformerConfig and need to be passed separately.
        if "hybrid_override_pattern" in config_raw:
            model_kwargs["hybrid_override_pattern"] = config_raw.hybrid_override_pattern
        if "hybrid_attention_ratio" in config_raw:
            model_kwargs["hybrid_attention_ratio"] = config_raw.hybrid_attention_ratio
        if "hybrid_mlp_ratio" in config_raw:
            model_kwargs["hybrid_mlp_ratio"] = config_raw.hybrid_mlp_ratio

        teacher = MCoreMambaModel(config=config, **model_kwargs)
    else:
        # GPT layer spec needs re-creation since it depends on number of model layers.
        if config.heterogeneous_block_specs:
            model_kwargs["transformer_layer_spec"] = get_gpt_heterogeneous_layer_spec(
                config=config,
                use_te=(args.transformer_impl == "transformer_engine"),
            )
        else:
            model_kwargs["transformer_layer_spec"] = get_gpt_modelopt_spec(
                config=config,
                local_core_attention=False if config.context_parallel_size > 1 else args.export_force_local_attention,
                remap_te_layernorm=args.export_te_mcore_model,
                real_quant_cfg=args.export_real_quant_cfg,
                use_arbitrary_attention_mask=False if config.context_parallel_size > 1 else True,
            )
        teacher = MCoreGPTModel(config=config, **model_kwargs)
    _add_load_convert_hooks(teacher)

    print_rank_0(f"Loading teacher as {type(teacher).__name__} from {args.export_kd_teacher_load} ...")
    # [WAR]: load checkpoint will check checkpoint's saved args and rng state if not finetune.
    # To avoid error out on loading teacher's checkpoint, we temporarily set args.finetune to
    # True while loading the teacher checkpoint.
    original_args_finetune, original_ckpt_format = args.finetune, args.ckpt_format
    args.finetune = True
    if args.export_kd_teacher_ckpt_format is not None:
        args.ckpt_format = args.export_kd_teacher_ckpt_format
    load_modelopt_checkpoint([teacher], load_arg='export_kd_teacher_load')
    args.finetune, args.ckpt_format = original_args_finetune, original_ckpt_format
    print_rank_0("...teacher loaded successfully.")

    return teacher


def modelopt_gpt_mamba_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
) -> MCoreGPTModel | MCoreMambaModel:
    """Builds the model.

    Args:
        args (Namespace): The arguments namespace.
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        vp_stage (int, optional): The virtual pipeline stage.
        config (TransformerConfig, optional): The configuration object.
        pg_collection (ProcessGroupCollection, optional): Collection of process groups
            used for tensor/context/pipeline/data parallelism. If provided, it will be
            attached to the returned model for downstream routing/resharding utilities.

    Returns:
        MCoreGPTModel | MCoreMambaModel: The returned model
    """
    print_rank_0("building GPT model ...")

    # ModelOpt by default assumes none homogenous layers. This affect the storage format of the sharded checkpoint.
    config = core_transformer_config_from_args(args)

    # Handle GPT-OSS mode with YaRN RoPE configuration
    if hasattr(args, 'enable_gpt_oss') and args.enable_gpt_oss:
        print_rank_0("GPT-OSS mode enabled: Configuring YaRN RoPE parameters")

        # Set GPT-OSS YaRN values directly on the config
        # These defaults are based on Huggingface GPT-OSS configurations
        config.position_embedding_type = "yarn"
        config.yarn_rotary_scaling_factor = 32.0
        config.yarn_original_max_position_embeddings = 131072
        config.yarn_beta_fast = 32.0
        config.yarn_beta_slow = 1.0
        config.yarn_mscale = 1.0
        config.yarn_mscale_all_dim = 0.0
        config.yarn_correction_range_round_to_int = False

    if vp_stage is not None:
        raise ValueError("ModelOpt integration does not currently support virtual pipeline parallel.")
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
        if args.export_offline_model:
            # Record the original num_layers. This is needed for _set_default_aux_hidden_state_layers
            config.original_num_layers = config.num_layers
            # Set num_layers to 0 for base model in offline mode
            config.num_layers = 0
            # SP is not used for offline
            # TODO: DSR1 MTP may require SP
            config.sequence_parallel = False
        if config.heterogeneous_block_specs:
            transformer_layer_spec = get_gpt_heterogeneous_layer_spec(
                config=config,
                use_te=args.transformer_impl == "transformer_engine",
            )
        else:
            local_core_attention=args.export_force_local_attention
            if config.context_parallel_size > 1:
                print_rank_0("context_parallel_size > 1! Force using TEDotProductAttention!")
                local_core_attention=False
                print_rank_0("context_parallel_size > 1! Force attention_mask_type to Causal. This can be wrong for EAGLE training!")
                use_arbitrary_attention_mask = False
            else:
                use_arbitrary_attention_mask = True

            transformer_layer_spec = get_gpt_modelopt_spec(
                config=config,
                local_core_attention=local_core_attention,
                remap_te_layernorm=args.export_te_mcore_model,
                real_quant_cfg=args.export_real_quant_cfg,
                use_arbitrary_attention_mask=use_arbitrary_attention_mask,
            )

        model_kwargs = {
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
            "pg_collection": pg_collection,
        }
        model = MCoreGPTModel(config=config, **model_kwargs)
    elif args.export_model_type == "MambaModel" or args.is_hybrid_model:
        from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec

        mamba_stack_spec = get_mamba_stack_modelopt_spec(
            remap_te_layernorm=args.export_te_mcore_model
        )
        model_kwargs = {
            "mamba_stack_spec": mamba_stack_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "hybrid_attention_ratio": args.hybrid_attention_ratio,
            "hybrid_mlp_ratio": args.hybrid_mlp_ratio,
            "hybrid_override_pattern": args.hybrid_override_pattern,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "pg_collection": pg_collection,
        }

        model = MCoreMambaModel(config=config, **model_kwargs)

        for l in range(model.decoder.num_layers_per_pipeline_rank):
            layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
            print_rank_0(f" == params layer {l}: {layer_params}")

    else:
        raise ValueError("ModelOpt does not support model type {}".format(args.export_model_type))

    # [IMPORTANT] Load modelopt_state immediately before returning the model back to `get_model()`.
    #
    # ModelOpt can create additional trainable parameters (e.g. for online speculative
    # decoding training or PEFT). Hence resuming modelopt_state during checkpoint loading is already
    # too late since Megatron created the optimizer right after calling model_provider before loading
    # the checkpoint. To ensure all trainable parameters are reigistered, we try to resume the
    # modelopt_state (which transforms the model to have additional parameters) before returning.
    if args.load is not None:
        load_modelopt_state(model=model)

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

        teacher_config_raw = _load_teacher_model_config(args.export_kd_teacher_load)
        teacher_config = core_transformer_config_from_args(teacher_config_raw)  # convert to TransformerConfig

        distill_cfg = mtd_mcore.setup_distillation_config(
            args.export_kd_cfg, student_cfg=config, teacher_cfg=teacher_config
        )
        kd_config = {
            "teacher_model": _load_teacher_model(teacher_config, teacher_config_raw, model_kwargs),
            "criterion": distill_cfg.criterion,
            "loss_balancer": distill_cfg.loss_balancer,
        }
        model = mtd.convert(model, mode=[("kd_loss", kd_config)])

        # Additional tweaks needed for MCore.
        # (accounts for sharded state, pipeline parallel, and potentially skipping LM loss)
        mtd_mcore.adjust_distillation_model_for_mcore(model, distill_cfg)
        # Also remove KD mode state to prevent issues with re-conversion after restore.
        mto.ModeloptStateManager(model).state_dict().pop()  # TODO(aanoosheh): remove once fixed in ModelOpt

    return model
