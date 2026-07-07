# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Lite flags for miles launchers."""

from __future__ import annotations

_OPTIMIZER_BACKEND_TO_IMPL = {
    "dist_opt": "dist_opt",
    "fsdp2": "fsdp2",
}


def optimizer_backend_to_impl(backend: str) -> str:
    if backend not in _OPTIMIZER_BACKEND_TO_IMPL:
        raise ValueError(
            f"Unsupported --mlite-optimizer-backend {backend!r}; "
            f"expected one of {sorted(_OPTIMIZER_BACKEND_TO_IMPL)}."
        )
    return _OPTIMIZER_BACKEND_TO_IMPL[backend]


def add_mlite_arguments(parser):
    """Register Megatron Lite flags on a miles parser."""
    group = parser.add_argument_group(title="megatron-lite")
    group.add_argument(
        "--mlite-backend-patch",
        action="store_true",
        default=False,
        help="Patch the miles Megatron actor slot to use Megatron Lite.",
    )
    group.add_argument(
        "--mlite-model-name",
        type=str,
        default="auto",
        help="Megatron Lite model registry name; 'auto' infers it from the HF config.",
    )
    group.add_argument(
        "--mlite-impl",
        type=str,
        default="lite",
        help="Megatron Lite model implementation variant.",
    )
    group.add_argument(
        "--mlite-optimizer-backend",
        type=str,
        default="dist_opt",
        choices=sorted(_OPTIMIZER_BACKEND_TO_IMPL),
        help="Optimizer backend: dist_opt or fsdp2.",
    )
    group.add_argument(
        "--mlite-attention-backend",
        type=str,
        default=None,
        help="Override attention backend. Defaults to --attention-backend, then 'flash'.",
    )
    group.add_argument(
        "--mlite-optimizer-offload",
        action="store_true",
        default=False,
        help="Offload optimizer state to CPU via offload_fraction=1.0.",
    )
    group.add_argument(
        "--mlite-param-offload",
        action="store_true",
        default=False,
        help="Offload model parameters to CPU during training/rollout alternation.",
    )
    group.add_argument(
        "--mlite-export-dtype",
        type=str,
        default=None,
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "float"],
        help="Optional dtype cast when exporting HF-format rollout weights.",
    )
    return parser


def validate_mlite_args(args) -> None:
    if not getattr(args, "hf_checkpoint", None):
        raise ValueError("--hf-checkpoint is required for the Megatron Lite backend patch.")
    args.variable_seq_lengths = True
    if not hasattr(args, "calculate_per_token_loss"):
        args.calculate_per_token_loss = False
    if not hasattr(args, "use_rollout_logprobs"):
        args.use_rollout_logprobs = False
