# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


def add_modelopt_args(parser):
    """Add additional arguments for using TensorRT Model Optimizer (modelopt) features."""
    group = parser.add_argument_group(title="modelopt-generic")

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
        "--export-quant-cfg",
        type=str,
        default=None,
        choices=["int8", "int8_sq", "fp8", "int4_awq", "w4a8_awq", "int4", "None"],
        help="Specify a quantization config from the supported choices.",
    )

    return parser
