# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Import-time switches for host and launch-overhead trims on the MoE router and MoK paths.

Each switch is read once, when this module is first imported, so every change can be ablated
per run without rebuilding. Every behavioural switch keeps results bit-identical to the
original code; the ``ROUTER_FAST_CAST_*`` launch values only tune the cast kernel's launch
shape. Boolean switches are parsed by :func:`megatron.core.env_flags.env_flag`: unset selects
the default, ``1``/``true``/``yes``/``on`` and ``0``/``false``/``no``/``off`` turn a switch on
or off, and any other value raises a ``ValueError``.
"""

import os

from megatron.core.env_flags import env_flag


def _env_int(name: str, default: int) -> int:
    """Integer value of the environment variable ``name``; unset selects ``default``."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(f"Environment variable {name}={value!r} is not an integer") from None


# Widen the router GEMM operands (BF16/FP16 input and weight) to FP32 with a vectorized, exact
# Triton kernel instead of PyTorch's unrolled casting copy (see router_fast_cast.py).
ROUTER_FAST_CAST = env_flag("MCORE_ROUTER_FAST_CAST", True)
# Cast the router input to FP32 in forward and save that copy for backward instead of recasting
# in backward. Moves the cast from backward to forward; costs one FP32 copy of the router input
# per MoE layer and microbatch in flight.
ROUTER_SAVE_FP32_INPUT = env_flag("MCORE_ROUTER_SAVE_FP32_INPUT", False)
# Launch shape of the cast kernel: elements per program (a power of two in [256, 65536]), warps
# per program, and the store variant (False: one flat tensor; True: 16-byte rows split into two
# 16-byte FP32 stores). An invalid block/warps pair falls back to 4096/8 with a warning.
ROUTER_FAST_CAST_BLOCK = _env_int("MCORE_ROUTER_FAST_CAST_BLOCK", 4096)
ROUTER_FAST_CAST_WARPS = _env_int("MCORE_ROUTER_FAST_CAST_WARPS", 8)
ROUTER_FAST_CAST_SPLIT = env_flag("MCORE_ROUTER_FAST_CAST_SPLIT", False)

# Pass zero_unrouted_router_grads=False to MoK's functional.backward when no top-k slot can be
# unrouted and only MCore's route adapter reads the router-weight gradient. Inactive with a MoK
# build whose functional.backward does not accept that keyword.
MOK_SKIP_UNROUTED_GRAD_MASK = env_flag("MCORE_MOK_SKIP_UNROUTED_GRAD_MASK", True)

# Use the host product num_local_tokens * prod(group sizes) for the per-token aux-loss scale
# when every rank of the static reduce groups holds the same Python-int token count and the aux
# loss is FP32/FP64, instead of an all-reduce of that count.
ROUTER_STATIC_TOKENS = env_flag("MCORE_ROUTER_STATIC_TOKENS", True)
# Debug: still all-reduce the count on the static path and assert that it equals the product
# (only outside CUDA-graph capture; adds a host sync per call).
ROUTER_CHECK_STATIC_TOKENS = env_flag("MCORE_ROUTER_CHECK_STATIC_TOKENS", False)

# Small autograd cleanups.
# RoutingMapToIndices: do not materialize a zero gradient for its non-differentiable indices.
MOK_ROUTE_NO_MATERIALIZE_GRADS = env_flag("MCORE_MOK_ROUTE_NO_MATERIALIZE_GRADS", True)
# MoEAuxLossAutoScaler: form the 0-dim aux-loss gradient directly instead of building a
# [1]-shaped one that autograd then reduces onto the 0-dim aux loss.
AUX_LOSS_SCALE_0DIM = env_flag("MCORE_AUX_LOSS_SCALE_0DIM", True)
# RandomSTE: draw the random logits into torch.empty_like(logits) instead of logits.clone().
RANDOM_STE_EMPTY_LIKE = env_flag("MCORE_RANDOM_STE_EMPTY_LIKE", True)


def summary() -> str:
    """Return the effective value of every switch, for logs and tests."""
    names = (
        "ROUTER_FAST_CAST",
        "ROUTER_SAVE_FP32_INPUT",
        "ROUTER_FAST_CAST_BLOCK",
        "ROUTER_FAST_CAST_WARPS",
        "ROUTER_FAST_CAST_SPLIT",
        "MOK_SKIP_UNROUTED_GRAD_MASK",
        "ROUTER_STATIC_TOKENS",
        "ROUTER_CHECK_STATIC_TOKENS",
        "MOK_ROUTE_NO_MATERIALIZE_GRADS",
        "AUX_LOSS_SCALE_0DIM",
        "RANDOM_STE_EMPTY_LIKE",
    )
    return " ".join(f"{name}={globals()[name]}" for name in names)
