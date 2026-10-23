# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.


def add_modelopt_args(parser):
    """Add additional arguments for using Model Optimizer (modelopt) features."""
    group = parser.add_argument_group(title="modelopt-generic")

    # Model and Checkpoint Compatibility
    group.add_argument(
        "--export-model-type",
        type=str,
        default=None,
        choices=["GPTModel", "HybridModel", "MambaModel"],
        help="Deprecated. The model type comes from the entry point (pretrain_gpt.py / "
        "pretrain_hybrid.py) or, for the post-training scripts, from --hybrid-layer-pattern "
        "(hybrid iff set). Only needed for a hybrid model without a layer pattern.",
    )
    group.add_argument(
        "--export-te-mcore-model",
        action="store_true",
        help="Deprecated and ignored. The ModelOpt-local spec always loads and saves "
        "layernorms under the fused Transformer-Engine checkpoint keys.",
    )
    group.add_argument(
        "--export-default-te-spec",
        action="store_true",
        help="Use the full Transformer-Engine layer spec for model building. "
        "This builds the model with TELayerNormColumnParallelLinear, TERowParallelLinear, "
        "TEGroupedMLP, TEDotProductAttention, etc., matching the canonical TE specs.",
    )
    group.add_argument(
        "--export-force-local-attention",
        action="store_true",
        help="Forcing local DotProductAttention; otherwise TEDotProductAttention is used.",
    )
    # Quantization
    group.add_argument(
        "--export-kv-cache-quant",
        help="Type of KV cache quantization to perform.",
        choices=["none", "fp8", "fp8_affine", "nvfp4", "nvfp4_affine", "nvfp4_rotate"],
        default="none",
    )
    group.add_argument(
        "--export-real-quant-cfg",
        type=str,
        default="None",
        choices=["fp8_real_quant", "fp8_blockwise_real_quant", "None"],
        help="Specify a real quantization config from the supported choices.",
    )
    group.add_argument(
        "--export-quant-cfg",
        type=str,
        default=None,
        # TODO replace choices with mtq.config.choices after deprecating the shorter aliases
        help="Specify a quantization config from mtq.config.choices.",
    )
    # Knowledge Distillation
    group.add_argument(
        '--export-kd-teacher-load',
        type=str,
        help='Path to checkpoint to load as distillation teacher. (Enables distillation mode automatically)',
    )
    group.add_argument(
        '--export-kd-teacher-model-config',
        type=str,
        default=None,
        help='Path to teacher model config for distillation, with argparse field names overriding '
        'the student\'s. If not provided, defaults to ${export_kd_teacher_load}/model_config.yaml. '
        'CLI only: YAML run configs give the teacher as DistillationConfig.teacher_model.',
    )
    group.add_argument(
        '--export-kd-teacher-ckpt-format',
        type=str,
        default=None,
        choices=['torch', 'torch_dist', 'torch_dcp'],
        help="Checkpoint format of teacher model, if different from student's.",
    )
    group.add_argument(
        '--export-kd-cfg',
        type=str,
        default=None,
        help='Path to distillation configuration yaml file, in order to use non-default settings.',
    )

    # Finetuning
    group.add_argument(
        "--finetune-hf-dataset", type=str, default=None, help="HF dataset used for finetuning."
    )
    group.add_argument(
        "--finetune-data-split", type=str, default="train", help="HF dataset split used for finetuning."
    )
    group.add_argument(
        "--finetune-data-files",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit files from an HF dataset repository used for finetuning. "
        "Listing files avoids preparing unrelated splits of large datasets.",
    )

    # MTP / base train-target selection for QAD and MTP QAT.
    group.add_argument(
        '--qad-train-target',
        type=str,
        default=None,
        choices=['base', 'mtp', 'both'],
        help='Which side of an MTP model to train during QAD / MTP QAT. '
        '"mtp": train MTP heads only, freeze the base (post-QAD two-phase recipe); '
        '"base": train the base only, freeze the MTP heads; '
        '"both": co-train the base and MTP heads together. '
        'Routers on the frozen side also have their expert_bias update skipped.',
    )
    group.add_argument(
        '--freeze-base-for-mtp',
        action='store_true',
        default=False,
        help='Deprecated alias for --qad-train-target mtp: freeze all base model '
        'parameters and only train MTP heads.',
    )

    # Scale-learning QAD (LSQ): the checkpoint is assumed to already have LSQ enabled
    group.add_argument(
        '--lsq-scale-lr',
        type=float,
        default=None,
        help='Optional separate learning rate for LSQ scale (amax) parameters. If unset, they '
        'use the base learning rate.',
    )

    # Deprecated aliases of the core architecture arguments
    group.add_argument(
        '--export-qk-l2-norm',
        action="store_true",
        dest="qk_l2_norm",
        help='Deprecated alias for --qk-l2-norm.',
    )
    group.add_argument(
        '--export-moe-apply-probs-on-input',
        action="store_true",
        dest="moe_apply_probs_on_input",
        help='Deprecated alias for --moe-apply-probs-on-input.',
    )

    # Speculative decoding
    group.add_argument(
        '--export-offline-model',
        action="store_true",
        help='If set, the base model will have no decoder layer. Only the embedding layer and output layer are initialized.',
    )

    group.add_argument(
        '--modelopt-enabled',
        action="store_true",
        help='Build the ModelOpt model even when the checkpoint has no ModelOpt state and no '
        'distillation teacher is given (both of which enable it automatically).',
    )

    return parser
