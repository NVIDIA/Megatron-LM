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
            "int8",
            "int8_sq",
            "fp8",
            "fp8_real_quant",
            "fp8_blockwise",
            "fp8_blockwise_real_quant",
            "int4_awq",
            "w4a8_awq",
            "int4",
            "fp4",
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
        '--export-kd-teacher-load',
        type=str,
        help='Path to checkpoint to load as distillation teacher.',
    )
    group.add_argument(
        '--export-kd-finalize',
        action="store_true",
        help='Export original student class back from a loaded distillation model.',
    )

    # Speculative decoding
    group.add_argument(
        '--export-num-medusa-heads',
        type=int,
        default=0,
        help='Number of Medusa heads for speculative decoding.',
    )
    group.add_argument(
        '--export-num-eagle-layers',
        type=int,
        default=0,
        help='Number of EAGLE layers for speculative decoding.',
    )

    # Finetuning
    group.add_argument(
        "--finetune-hf-dataset", type=str, default=None, help="HF dataset used for finetuning."
    )

    return parser
