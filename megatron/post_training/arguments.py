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

    # Speculative decoding
    group.add_argument(
        '--export-num-medusa-heads',
        type=int,
        default=0,
        help='Number of Medusa heads for speculative decoding.',
    )
    group.add_argument(
        '--export-eagle-algorithm',
        type=str,
        choices=['eagle1', 'eagle3', 'eagle-mtp'],
        default="eagle-mtp",
        help='Chosing the between different flavors of EAGLE algorithms.',
    )
    group.add_argument(
        '--export-num-eagle-layers',
        type=int,
        default=0,
        help='Number of EAGLE layers for speculative decoding.',
    )
    group.add_argument(
        '--export-draft-vocab-size',
        type=int,
        default=0,
        help='The reduced vocabulary size of the draft model.',
    )
    group.add_argument(
        '--export-num-mtp',
        type=int,
        default=0,
        help='Number of MTP modules for speculative decoding.',
    )
    group.add_argument(
        '--export-freeze-mtp',
        type=int,
        nargs="*",
        default=[],
        help='Index of MTP that will be frozen in training.',
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

    return parser


def modelopt_args_enabled(args):
    """Check if any modelopt-related arguments are provided."""
    key_args_and_defaults = {
        "export_real_quant_cfg": "None",
        "export_quant_cfg": None,
        "export_kd_teacher_load": None,
        "export_num_medusa_heads": 0,
        "export_num_eagle_layers": 0,
    }
    for key, default in key_args_and_defaults.items():
        if hasattr(args, key) and getattr(args, key) != default:
            return True
    return False
