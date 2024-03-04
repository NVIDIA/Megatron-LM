# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

def add_ammo_args(parser):
    """Add additional arguments for ammo."""
    group = parser.add_argument_group(title="ammo-generic")

    group.add_argument(
        "--ammo-load-classic-megatron-to-mcore",
        action="store_true",
        help="Load a classic megatron-lm checkpoint to a new megatron-core model.",
    )
    group.add_argument(
        "--ammo-convert-te-to-local-spec",
        action="store_true",
        help="Load a megatron-core transformer-engine checkpoint to a model with local spec.",
    )
    group.add_argument(
        "--ammo-quant-cfg",
        type=str,
        default=None,
        choices=["int8_sq", "fp8", "int4_awq", "None"],
        help="Algorithms supported by atq.quantize.",
    )

    return parser
