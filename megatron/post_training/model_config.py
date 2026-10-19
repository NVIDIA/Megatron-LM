# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Derive stock GPT/Hybrid model configs for ModelOpt workflows from the CLI arguments.

ModelOpt runs use the regular ``GPTModelBuilder`` / ``HybridModelBuilder``. What is specific to
ModelOpt (layer specs from the ``--export-*`` flags, a few ``TransformerConfig`` tweaks, and
the pre-wrap hooks in ``megatron.post_training.model_hooks``) is attached to the config here.
"""

import dataclasses
import functools
import logging
import os
import warnings
from argparse import Namespace
from typing import Literal

import yaml

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.post_training.modelopt.hybrid.model_specs import get_hybrid_stack_modelopt_spec
from megatron.training import print_rank_0
from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.models.distillation import DistillationConfig, load_distillation_loss_yaml
from megatron.training.models.gpt import GPTModelConfig
from megatron.training.models.hybrid import HybridModelConfig
from megatron.training.models.transforms import ModelTransformConfig
from megatron.training.utils import is_hybrid_model

logger = logging.getLogger(__name__)


def distillation_config_from_args(
    args: Namespace, student_model_config: GPTModelConfig | HybridModelConfig
) -> DistillationConfig | None:
    """Build the run's ``DistillationConfig`` from ``--export-kd-*``, or ``None`` without KD.

    The teacher architecture is resolved here from the student's model config and the teacher
    YAML, and the ``--export-kd-cfg`` YAML is read into the loss fields, so the result is
    self-contained. Incompatible settings are rejected by the run container's validation.

    Args:
        args: The parsed arguments, including those from ``add_modelopt_args``.
        student_model_config: The student config, from ``modelopt_model_config_from_args``.
    """
    if not getattr(args, "export_kd_teacher_load", None):
        return None

    loss = load_distillation_loss_yaml(args.export_kd_cfg) if args.export_kd_cfg else {}
    distillation = DistillationConfig(
        teacher_load=args.export_kd_teacher_load,
        teacher_model_config=args.export_kd_teacher_model_config,
        teacher_ckpt_format=args.export_kd_teacher_ckpt_format,
        **loss,
    )
    teacher_args = _load_teacher_model_config(distillation, args)
    overrides = {}
    if isinstance(student_model_config, HybridModelConfig):
        overrides["hybrid_layer_pattern"] = teacher_args.hybrid_layer_pattern
    distillation.teacher_model = dataclasses.replace(
        student_model_config,
        transformer=core_transformer_config_from_args(teacher_args),
        transformations=None,
        pre_wrap_hooks=[],
        post_wrap_hooks=[],
        **overrides,
    )
    return distillation


def _load_teacher_model_config(distillation: DistillationConfig, args: Namespace) -> Namespace:
    """Reads teacher config from a file.

    The config provided, either in the teacher checkpoint dir or via ``distillation.teacher_model_config``,
    should specify any model architecture settings which differ from the main student model's.
    The field names should match those returned by get_args() and not TransformerConfig.
    """
    if distillation.teacher_model_config is not None:
        config_path = distillation.teacher_model_config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Teacher model-config file ({config_path}) not found.")
    else:
        config_path = os.path.join(distillation.teacher_load, "model_config.yaml")
        if not os.path.exists(config_path):
            logger.warning(
                "No teacher config provided via --export-kd-teacher-model-config nor found at"
                f" {distillation.teacher_load}/model_config.yaml. Assuming teacher model architecture same as student's."
            )  # Useful for cases like QAD
            config_path = None

    args_dict = vars(args).copy()

    if config_path is not None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        del args_dict["kv_channels"]  # not recalculated if present
        # Setting teacher Flextron fields to false if training with Flextron, can be overridden
        if "flextron" in args_dict:
            args_dict["flextron"] = False
        if "enable_router" in args_dict:
            args_dict["enable_router"] = False
        if "freeze_model" in args_dict:
            args_dict["freeze_model"] = False

        args_dict.update(config)

    # Backward compat: old checkpoints have hybrid_override_pattern but not hybrid_layer_pattern
    if (
        args_dict.get('hybrid_override_pattern') is not None
        and args_dict.get('hybrid_layer_pattern') is None
    ):
        args_dict['hybrid_layer_pattern'] = args_dict['hybrid_override_pattern']

    return Namespace(**args_dict)


def modelopt_mtp_layer_spec(model_config: GPTModelConfig):
    """Keep MTP on the standard decoder spec, independently of ModelOpt decoder quantization."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs

    config = model_config.transformer
    return get_gpt_decoder_layer_specs(
        config, use_transformer_engine=config.transformer_impl == "transformer_engine"
    )[-1]


def modelopt_gpt_layer_spec(
    model_config: GPTModelConfig,
    *,
    default_te_spec: bool,
    force_local_attention: bool,
    remap_te_layernorm: bool,
    real_quant_cfg: str,
    disable_moe_grouped_gemm: bool,
):
    """GPT layer spec for ModelOpt workflows.

    Used as ``GPTModelConfig.transformer_layer_spec``, so it is evaluated by the builder against
    the config it is building (e.g. the teacher's, whose layer count differs from the student's).
    """
    config = model_config.transformer
    if config.heterogeneous_block_specs:
        return get_gpt_heterogeneous_layer_spec(
            config=config, use_te=config.transformer_impl == "transformer_engine"
        )
    if default_te_spec:
        # Use the canonical full Transformer Engine spec instead of the modelopt-customized
        # spec. Required by pruning, which operates on the un-customized layer graph.
        # ``disable_moe_grouped_gemm`` (set by prune.py) forces SequentialMLP so mtp.prune
        # can act on individual expert linears.
        return get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            not disable_moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            qk_l2_norm=config.qk_l2_norm,
        )
    if config.context_parallel_size > 1:
        print_rank_0("context_parallel_size > 1! Force using TEDotProductAttention!")
        local_core_attention = False
    else:
        local_core_attention = force_local_attention
    return get_gpt_modelopt_spec(
        config=config,
        local_core_attention=local_core_attention,
        remap_te_layernorm=remap_te_layernorm,
        real_quant_cfg=real_quant_cfg,
        use_arbitrary_attention_mask=False,
    )


# Values of the deprecated ``--export-model-type`` flag.
_EXPORT_MODEL_TYPES = {"GPTModel": "gpt", "HybridModel": "hybrid", "MambaModel": "hybrid"}


def _resolve_model_type(
    args: Namespace, model_type: Literal["gpt", "hybrid"] | None
) -> Literal["gpt", "hybrid"]:
    """Choose GPT or Hybrid: the caller's ``model_type``, else ``--hybrid-layer-pattern``.

    The deprecated ``--export-model-type`` is still honored (e.g. for a pure-Mamba model without a
    layer pattern), but must not contradict the entry point or the layer pattern.
    """
    flag = getattr(args, "export_model_type", None)
    if flag is None:
        if model_type is not None:
            return model_type
        return "hybrid" if is_hybrid_model(args) else "gpt"

    warnings.warn(
        "--export-model-type is deprecated: the model type now comes from the entry point "
        "(pretrain_gpt.py / pretrain_hybrid.py) or from --hybrid-layer-pattern.",
        DeprecationWarning,
        stacklevel=3,
    )
    flag_type = _EXPORT_MODEL_TYPES[flag]
    if model_type is not None and flag_type != model_type:
        raise ValueError(f"--export-model-type {flag} conflicts with this {model_type} entry point.")
    if flag_type == "gpt" and is_hybrid_model(args):
        raise ValueError(f"--export-model-type {flag} conflicts with --hybrid-layer-pattern.")
    return flag_type


def modelopt_model_config_from_args(
    args: Namespace,
    *,
    model_type: Literal["gpt", "hybrid"] | None = None,
    disable_moe_grouped_gemm: bool = False,
    vocab_size_from_tokenizer: bool = True,
) -> GPTModelConfig | HybridModelConfig:
    """Build the GPT or Hybrid model config for ModelOpt.

    Args:
        args: The parsed arguments, including those from ``add_modelopt_args``.
        model_type: ``"gpt"`` or ``"hybrid"``, for entry points that build one kind of model.
            If ``None``, a model is hybrid iff ``--hybrid-layer-pattern`` is set.
        disable_moe_grouped_gemm: Force the export spec to use SequentialMLP (per-expert
            linears) instead of the default TEGroupedMLP. Pruning sets this so
            ``mtp.prune`` can operate on individual expert linears; quantize / generate /
            finetune leave the default so MoE quantization (e.g. QuantTEGroupedMLP) works
            and TP+EP > 1 doesn't trip the QuantSequentialMLP unsupported-combo check.
        vocab_size_from_tokenizer: See ``gpt_config_from_args``.

    Returns:
        A ``GPTModelConfig`` / ``HybridModelConfig`` to build with its regular builder
        (``config.get_builder_cls()``).
    """
    if args.spec is not None:
        raise ValueError("ModelOpt integration does not support custom args.spec.")
    model_type = _resolve_model_type(args, model_type)
    if args.pipeline_model_parallel_size > 1 and args.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("ModelOpt integration does not currently support virtual pipeline parallel.")

    # ModelOpt by default assumes none homogenous layers. This affect the storage format of the sharded checkpoint.
    transformer_config = core_transformer_config_from_args(args)

    # Handle GPT-OSS mode with YaRN RoPE configuration
    if getattr(args, "enable_gpt_oss", False):
        print_rank_0("GPT-OSS mode enabled: Configuring YaRN RoPE parameters")

        # Set GPT-OSS YaRN values directly on the config
        # These defaults are based on Huggingface GPT-OSS configurations
        transformer_config.position_embedding_type = "yarn"
        transformer_config.yarn_rotary_scaling_factor = 32.0
        transformer_config.yarn_original_max_position_embeddings = 131072
        transformer_config.yarn_beta_fast = 32.0
        transformer_config.yarn_beta_slow = 1.0
        transformer_config.yarn_mscale = 1.0
        transformer_config.yarn_mscale_all_dim = 0.0
        transformer_config.yarn_correction_range_round_to_int = False

    # Llama-4 Scout/Maverick support
    transformer_config.qk_l2_norm = args.export_qk_l2_norm
    transformer_config.moe_apply_probs_on_input = args.export_moe_apply_probs_on_input

    if model_type == "gpt":
        if args.export_offline_model:
            # Record the original num_layers. This is needed for _set_default_aux_hidden_state_layers
            transformer_config.original_num_layers = transformer_config.num_layers
            # Set num_layers to 0 for base model in offline mode
            transformer_config.num_layers = 0
            # SP is not used for offline
            # TODO: DSR1 MTP may require SP
            transformer_config.sequence_parallel = False

        model_config = gpt_config_from_args(
            args, config=transformer_config, vocab_size_from_tokenizer=vocab_size_from_tokenizer
        )
        model_config.mtp_layer_spec = modelopt_mtp_layer_spec
        model_config.transformer_layer_spec = functools.partial(
            modelopt_gpt_layer_spec,
            default_te_spec=args.export_default_te_spec,
            force_local_attention=args.export_force_local_attention,
            remap_te_layernorm=args.export_te_mcore_model,
            real_quant_cfg=args.export_real_quant_cfg,
            disable_moe_grouped_gemm=disable_moe_grouped_gemm,
        )
    else:
        if args.export_default_te_spec and args.export_te_mcore_model:
            logger.warning(
                "--export-default-te-spec and --export-te-mcore-model are mutually exclusive. "
                "Since --export-default-te-spec is given, --export-te-mcore-model will be disabled."
            )
            args.export_te_mcore_model = False

        model_config = hybrid_config_from_args(
            args, config=transformer_config, vocab_size_from_tokenizer=vocab_size_from_tokenizer
        )
        # Default to grouped MLP for the export spec (matches the pre-modernization
        # behavior of get_hybrid_stack_modelopt_spec — its factory default is True).
        # ``disable_moe_grouped_gemm`` (set by prune.py) forces SequentialMLP so
        # mtp.prune can act on individual expert linears.
        model_config.hybrid_stack_spec = get_hybrid_stack_modelopt_spec(
            remap_te_layernorm=args.export_te_mcore_model,
            use_default_te_spec=args.export_default_te_spec,
            moe_grouped_gemm=not disable_moe_grouped_gemm,
        )

    qad_train_target = getattr(args, 'qad_train_target', None)
    if args.freeze_base_for_mtp:
        if qad_train_target not in (None, 'mtp'):
            raise ValueError(
                "--freeze-base-for-mtp is an alias for --qad-train-target mtp and "
                f"conflicts with --qad-train-target {qad_train_target}"
            )
        qad_train_target = 'mtp'
    model_config.transformations = ModelTransformConfig(
        qad_train_target=qad_train_target,
        lsq_scale_lr=getattr(args, "lsq_scale_lr", None),
        te_mcore_model=args.export_te_mcore_model,
    )
    return model_config


def modelopt_inference_config_from_args(
    args: Namespace,
    *,
    disable_moe_grouped_gemm: bool = False,
    device: str = "cuda",
    mixed_precision: bool = True,
):
    """Translate CLI once; build later from the populated inference run container.

    Tokenizer vocabulary is finalized after runtime services initialize. Legacy
    services can still use get_args(); model construction uses this container.
    """
    from megatron.training.argument_utils import inference_cfg_container_from_args
    from megatron.training.models.build import ModelBuildConfig

    if args.init_model_with_meta_device:
        args.use_cpu_initialization = True
    model = modelopt_model_config_from_args(
        args, disable_moe_grouped_gemm=disable_moe_grouped_gemm, vocab_size_from_tokenizer=True
    )
    cfg = inference_cfg_container_from_args(args, model_cfg=model)
    cfg.model_build = ModelBuildConfig(
        device=device, wrap_with_ddp=False, mixed_precision=mixed_precision
    )
    return cfg


def build_modelopt_models(args: Namespace) -> list:
    """Bridge legacy inference callers that initialize a run container without a model.

    New entry points should translate with modelopt_inference_config_from_args
    before initialization and call cfg.build_models directly.
    """
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.training.global_vars import get_run_config

    cfg = get_run_config()
    if cfg.model is None or cfg.model.transformations is None:
        translated = modelopt_inference_config_from_args(args)
        cfg.model = translated.model
        cfg.model_build = translated.model_build
    return cfg.build_models(ProcessGroupCollection.use_mpu_process_groups())
