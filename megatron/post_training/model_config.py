# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Derive stock GPT/Hybrid model configs for ModelOpt workflows from the CLI arguments.

ModelOpt runs use the regular ``GPTModelBuilder`` / ``HybridModelBuilder``. What is specific to
ModelOpt (layer specs from the ``--export-*`` flags, a few ``TransformerConfig`` tweaks, and
the pre-wrap hooks in ``megatron.post_training.model_hooks``) is attached to the config here.
"""

import functools
import logging
from argparse import Namespace

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.post_training.modelopt.hybrid.model_specs import get_hybrid_stack_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.post_training import model_hooks
from megatron.training import print_rank_0
from megatron.training.argument_utils import gpt_config_from_args, hybrid_config_from_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.models.distillation import DistillationConfig
from megatron.training.models.gpt import GPTModelConfig
from megatron.training.models.hybrid import HybridModelConfig

logger = logging.getLogger(__name__)


def distillation_config_from_args(args: Namespace) -> DistillationConfig | None:
    """Build the ``DistillationConfig`` from ``--export-kd-*``, or ``None`` if KD is not enabled."""
    if not getattr(args, "export_kd_teacher_load", None):
        return None

    # NOTE: Unknown memory leak occuring per fwd-bwd pass if model
    # is converted to a `modelopt.torch.opt.DynamicModule`.
    # Argument `--manual-gc` can result in an eventual OOM.
    assert (
        not args.manual_gc
    ), "ModelOpt Distillation currently incompatible with `--manual-gc` option."
    assert (
        not args.tp_comm_overlap
    ), "ModelOpt Distillation currently incompatible with `--tp-comm-overlap` option."
    assert (
        args.cross_entropy_fusion_impl != "te"
    ), "ModelOpt Distillation currently incompatible with TransformerEngine Cross-Entropy implementation."
    if args.pipeline_model_parallel_size > 1:
        assert (
            args.virtual_pipeline_model_parallel_size is None
        ), "ModelOpt Distillation currently incompatible with interleaved pipeline schedule."

    return DistillationConfig(
        teacher_load=args.export_kd_teacher_load,
        teacher_model_config=args.export_kd_teacher_model_config,
        teacher_ckpt_format=args.export_kd_teacher_ckpt_format,
        kd_cfg=args.export_kd_cfg,
    )


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


def modelopt_model_config_from_args(
    args: Namespace,
    *,
    disable_moe_grouped_gemm: bool = False,
    vocab_size_from_tokenizer: bool = True,
) -> GPTModelConfig | HybridModelConfig:
    """Build the GPT or Hybrid model config (dispatching on ``--export-model-type``) for ModelOpt.

    Args:
        args: The parsed arguments, including those from ``add_modelopt_args``.
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

    if args.export_model_type == "GPTModel":
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
        model_config.transformer_layer_spec = functools.partial(
            modelopt_gpt_layer_spec,
            default_te_spec=args.export_default_te_spec,
            force_local_attention=args.export_force_local_attention,
            remap_te_layernorm=args.export_te_mcore_model,
            real_quant_cfg=args.export_real_quant_cfg,
            disable_moe_grouped_gemm=disable_moe_grouped_gemm,
        )
    elif (
        args.export_model_type in ("HybridModel", "MambaModel")
        or getattr(args, "hybrid_layer_pattern", None) is not None
    ):
        if args.export_model_type == "MambaModel":
            import warnings

            warnings.warn(
                '--export-model-type "MambaModel" is deprecated. '
                'Use --export-model-type "HybridModel" instead.',
                DeprecationWarning,
                stacklevel=2,
            )

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
    else:
        raise ValueError("ModelOpt does not support model type {}".format(args.export_model_type))

    model_config.distillation = distillation_config_from_args(args)

    # Registered in dependency order; see the ``megatron.post_training.model_hooks`` docstring.
    # ``model_config.restore_modelopt_state`` is deliberately left unset: the restore hook below
    # owns restoration, and a runner that also honored the flag would restore a second time,
    # after the KD conversion, where ModelOpt silently skips it.
    init_meta = model_config.transformer.init_model_with_meta_device

    def register(hook):
        model_config.pre_wrap_hooks.append(model_hooks.on_model_init_device(hook, init_meta))

    if args.load is not None:
        register(model_hooks.restore_modelopt_state_hook(args.load))

    qad_train_target = getattr(args, 'qad_train_target', None)
    if args.freeze_base_for_mtp:
        if qad_train_target not in (None, 'mtp'):
            raise ValueError(
                "--freeze-base-for-mtp is an alias for --qad-train-target mtp and "
                f"conflicts with --qad-train-target {qad_train_target}"
            )
        qad_train_target = 'mtp'
    if qad_train_target is not None:
        register(model_hooks.qad_freeze_hook(qad_train_target))

    register(model_hooks.lsq_hook(getattr(args, "lsq_scale_lr", None)))
    if args.export_te_mcore_model:
        register(model_hooks.te_load_state_dict_hook())
    if model_config.distillation is not None:
        # Must stay last: the restore (and any freeze) has to see the un-converted student.
        register(model_hooks.kd_convert_hook(model_config, args, args.export_te_mcore_model))

    return model_config


def build_modelopt_models(
    args: Namespace, *, disable_moe_grouped_gemm: bool = False, **build_kwargs
) -> list:
    """Build the (non-DDP-wrapped by default) model stages for a ModelOpt script.

    ``build_kwargs`` are forwarded to ``ModelBuilder.build_distributed_models``.
    """
    model_config = modelopt_model_config_from_args(
        args, disable_moe_grouped_gemm=disable_moe_grouped_gemm
    )
    build_kwargs.setdefault("wrap_with_ddp", False)
    return model_config.get_builder_cls()(model_config).build_distributed_models(
        ProcessGroupCollection.use_mpu_process_groups(), **build_kwargs
    )
