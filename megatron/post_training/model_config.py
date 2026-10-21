# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Derive stock GPT/Hybrid model configs for ModelOpt workflows from the CLI arguments.

ModelOpt runs use the regular ``GPTModelBuilder`` / ``HybridModelBuilder``. What is specific to
ModelOpt (layer-spec settings from the ``--export-*`` flags, the offline speculative-decoding
tweak, and the transformations that select the pre-wrap hooks in
``megatron.post_training.model_hooks``) is applied to a stock config by
``apply_modelopt_model_config``.
"""

import dataclasses
import logging
import os
import warnings
from argparse import Namespace

import yaml

from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
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
    teacher_transformations = student_model_config.transformations
    if teacher_transformations is not None:
        teacher_transformations = dataclasses.replace(
            teacher_transformations,
            qad_train_target=None,
            lsq_scale_lr=None,
        )
    if isinstance(student_model_config, HybridModelConfig):
        teacher_model = hybrid_config_from_args(teacher_args)
        teacher_model.hybrid_stack_spec = student_model_config.hybrid_stack_spec
    else:
        teacher_model = gpt_config_from_args(teacher_args)
        teacher_model.transformer_layer_spec = student_model_config.transformer_layer_spec
        teacher_model.mtp_layer_spec = student_model_config.mtp_layer_spec
    teacher_model.vocab_size = student_model_config.vocab_size
    teacher_model.should_pad_vocab = student_model_config.should_pad_vocab
    teacher_model.transformations = teacher_transformations
    distillation.teacher_model = teacher_model
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
            config = yaml.safe_load(f) or {}

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


# Values of the deprecated ``--export-model-type`` flag.
_EXPORT_MODEL_TYPES = {"GPTModel": "gpt", "HybridModel": "hybrid", "MambaModel": "hybrid"}


def _is_hybrid_from_args(args: Namespace) -> bool:
    """For callers without a fixed model type: hybrid iff ``--hybrid-layer-pattern`` is set.

    The deprecated ``--export-model-type`` still selects the type (e.g. for a pure-Mamba model
    without a layer pattern); ``_check_model_type`` warns about it.
    """
    flag = getattr(args, "export_model_type", None)
    if flag is not None:
        return _EXPORT_MODEL_TYPES[flag] == "hybrid"
    return is_hybrid_model(args)


def _check_model_type(model_config: GPTModelConfig | HybridModelConfig, args: Namespace) -> None:
    hybrid = isinstance(model_config, HybridModelConfig)
    flag = getattr(args, "export_model_type", None)
    if flag is not None:
        warnings.warn(
            "--export-model-type is deprecated: the model type now comes from the entry point "
            "(pretrain_gpt.py / pretrain_hybrid.py) or from --hybrid-layer-pattern.",
            DeprecationWarning,
            stacklevel=3,
        )
        if (_EXPORT_MODEL_TYPES[flag] == "hybrid") != hybrid:
            raise ValueError(
                f"--export-model-type {flag} conflicts with the {type(model_config).__name__}."
            )
    if not hybrid and is_hybrid_model(args):
        raise ValueError(
            "--hybrid-layer-pattern is set but the model config is a GPTModelConfig; "
            "use pretrain_hybrid.py."
        )


def modelopt_model_config_from_args(
    args: Namespace,
    *,
    disable_moe_grouped_gemm: bool = False,
    vocab_size_from_tokenizer: bool = True,
) -> GPTModelConfig | HybridModelConfig:
    """Build the GPT or Hybrid model config for ModelOpt, for callers without a fixed type.

    A model is hybrid iff ``--hybrid-layer-pattern`` is set. Entry points that build one kind
    of model call ``gpt_config_from_args`` / ``hybrid_config_from_args`` and then
    ``apply_modelopt_model_config`` instead.

    Args:
        args: The parsed arguments, including those from ``add_modelopt_args``.
        disable_moe_grouped_gemm: See ``apply_modelopt_model_config``.
        vocab_size_from_tokenizer: See ``gpt_config_from_args``.
    """
    # Reject this before the stock translators try to import a spec that ModelOpt cannot use.
    if args.spec is not None:
        raise ValueError("ModelOpt integration does not support custom args.spec.")
    config_from_args = hybrid_config_from_args if _is_hybrid_from_args(args) else gpt_config_from_args
    model_config = config_from_args(args, vocab_size_from_tokenizer=vocab_size_from_tokenizer)
    return apply_modelopt_model_config(
        model_config, args, disable_moe_grouped_gemm=disable_moe_grouped_gemm
    )


def apply_modelopt_model_config(
    model_config: GPTModelConfig | HybridModelConfig,
    args: Namespace,
    *,
    disable_moe_grouped_gemm: bool = False,
) -> GPTModelConfig | HybridModelConfig:
    """Set declarative ModelOpt construction and transformation options on a stock model config.

    Args:
        model_config: A config from ``gpt_config_from_args`` / ``hybrid_config_from_args``,
            updated in place.
        args: The parsed arguments, including those from ``add_modelopt_args``.
        disable_moe_grouped_gemm: Force the export spec to use SequentialMLP (per-expert
            linears) instead of the default TEGroupedMLP. Pruning sets this so
            ``mtp.prune`` can operate on individual expert linears; quantize / generate /
            finetune leave the default so MoE quantization (e.g. QuantTEGroupedMLP) works
            and TP+EP > 1 doesn't trip the QuantSequentialMLP unsupported-combo check.

    Returns:
        ``model_config``, to build with its regular builder (``config.get_builder_cls()``).
    """
    if args.spec is not None:
        raise ValueError("ModelOpt integration does not support custom args.spec.")
    if args.pipeline_model_parallel_size > 1 and args.virtual_pipeline_model_parallel_size is not None:
        raise ValueError("ModelOpt integration does not currently support virtual pipeline parallel.")
    _check_model_type(model_config, args)

    if not isinstance(model_config, HybridModelConfig) and getattr(
        args, "export_offline_model", False
    ):
        transformer_config = model_config.transformer
        # Record the original num_layers. This is needed for _set_default_aux_hidden_state_layers
        transformer_config.original_num_layers = transformer_config.num_layers
        # Set num_layers to 0 for base model in offline mode
        transformer_config.num_layers = 0
        # SP is not used for offline
        # TODO: DSR1 MTP may require SP
        transformer_config.sequence_parallel = False
    qad_train_target = getattr(args, 'qad_train_target', None)
    if getattr(args, "freeze_base_for_mtp", False):
        if qad_train_target not in (None, 'mtp'):
            raise ValueError(
                "--freeze-base-for-mtp is an alias for --qad-train-target mtp and "
                f"conflicts with --qad-train-target {qad_train_target}"
            )
        qad_train_target = 'mtp'
    model_config.transformations = ModelTransformConfig(
        default_te_spec=getattr(args, "export_default_te_spec", False),
        force_local_attention=getattr(args, "export_force_local_attention", False),
        real_quant_cfg=getattr(args, "export_real_quant_cfg", "None"),
        moe_grouped_gemm=not disable_moe_grouped_gemm,
        qad_train_target=qad_train_target,
        lsq_scale_lr=getattr(args, "lsq_scale_lr", None),
    )
    return model_config


def modelopt_inference_config_from_args(args: Namespace, *, disable_moe_grouped_gemm: bool = False):
    """Translate CLI once; build later from the populated inference run container.

    Tokenizer vocabulary is finalized after runtime services initialize. Legacy
    services can still use get_args(); model construction uses this container.
    """
    from megatron.training.argument_utils import inference_cfg_container_from_args

    if args.init_model_with_meta_device:
        args.use_cpu_initialization = True
    model = modelopt_model_config_from_args(
        args, disable_moe_grouped_gemm=disable_moe_grouped_gemm, vocab_size_from_tokenizer=True
    )
    return inference_cfg_container_from_args(args, model_cfg=model)


def build_inference_models(cfg, *, device: str = "cuda", mixed_precision: bool = True) -> list:
    """Build an inference container's model on the global MPU process groups, without DDP.

    ``cfg.checkpoint`` is passed so the ModelOpt pre-wrap hooks restore its ModelOpt state.
    """
    from megatron.core.process_groups_config import ProcessGroupCollection

    kwargs = {} if mixed_precision else {"mixed_precision_wrapper": None}
    builder = cfg.model.get_builder_cls()(cfg.model)
    return builder.build_distributed_models(
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        wrap_with_ddp=False,
        device=device,
        checkpoint=cfg.checkpoint,
        **kwargs,
    )


def build_modelopt_models(args: Namespace) -> list:
    """Bridge legacy inference callers that initialize a run container without a model.

    New entry points should translate with modelopt_inference_config_from_args
    before initialization and call build_inference_models directly.
    """
    from megatron.training.global_vars import get_run_config

    cfg = get_run_config()
    if cfg.model is None or cfg.model.transformations is None:
        cfg.model = modelopt_inference_config_from_args(args).model
    return build_inference_models(cfg)
