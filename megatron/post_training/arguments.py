# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


def add_modelopt_args(parser):
    """Add additional arguments for using TensorRT Model Optimizer (modelopt) features."""
    group = parser.add_argument_group(title="modelopt-generic")

    # Model and Checkpoint Compatibility
    group.add_argument(
        "--export-model-type",
        type=str,
        default="GPTModel",
        choices=["GPTModel", "MambaModel"],
        help="Model type to use in model_provider.",
    )
    group.add_argument(
        "--export-legacy-megatron",
        action="store_true",
        help="Export a legacy megatron-lm checkpoint.",
    )
    group.add_argument(
        "--export-te-mcore-model",
        action="store_true",
        help="Export a megatron-core transformer-engine checkpoint.",
    )
    group.add_argument(
        "--export-force-local-attention",
        action="store_true",
        help="Forcing local DotProductAttention; otherwise TEDotProductAttention is used.",
    )

    # Quantization
    group.add_argument(
        "--export-kv-cache-quant",
        action="store_true",
        help="Whether or not to perform KV-cache quantization.",
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
        choices=[
            "int8_sq",
            "fp8",
            "fp8_real_quant",
            "fp8_blockwise",
            "fp8_blockwise_real_quant",
            "fp8_blockwise_32",
            "int4_awq",
            "w4a8_awq",
            "nvfp4",
            "None",
        ],
        help="Specify a quantization config from the supported choices.",
    )

    # Knowledge Distillation
    group.add_argument(
        '--export-kd-cfg',
        type=str,
        default=None,
        help='Path to distillation configuration yaml file.',
    )

    group.add_argument(
        '--teacher-model-config',
        type=str,
        default=None,
        help='Path to teacher model config for distillation. If not provided, defaults to ${export_kd_teacher_load}/model_config.yaml.',
    )

    group.add_argument(
        '--export-kd-teacher-load',
        type=str,
        help='Path to checkpoint to load as distillation teacher.',
    )
    group.add_argument(
        '--export-kd-teacher-ckpt-format',
        type=str,
        default=None,
        choices=['torch', 'torch_dist', 'zarr', 'torch_dcp'],
        help="Checkpoint format of teacher model, if different from student's.",
    )

    # Finetuning
    group.add_argument(
        "--finetune-hf-dataset", type=str, default=None, help="HF dataset used for finetuning."
    )
    group.add_argument(
        "--finetune-data-split", type=str, default="train", help="HF dataset split used for finetuning."
    )

    # Special model architecture option
    group.add_argument(
        '--export-qk-l2-norm',
        action="store_true",
        help='Use Llama-4 L2Norm instead of normal LayerNorm/RMSNorm for QK normalization.',
    )
    group.add_argument(
        '--export-moe-apply-probs-on-input',
        action="store_true",
        help='Use Llama-4 expert scaling on input instead of output.',
    )

    # Speculative decoding
    group.add_argument(
        '--export-offline-model',
        action="store_true",
        help='If set, the base model will have no decoder layer. Only the embedding layer and output layer are initialized.',
    )

    # Global state
    group.add_argument(
        '--modelopt-enabled',
        action="store_true",
        help='Will be set automatically when loading a ModelOpt checkpoint.',
    )
    
    # GPT-OSS YaRN RoPE support
    group.add_argument(
        '--enable-gpt-oss',
        action="store_true",
        help='Enable GPT-OSS mode with YaRN RoPE configuration. When enabled, automatically '
             'configures all YaRN parameters with GPT-OSS defaults.',
    )

    return parser
