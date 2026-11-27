# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Which Gated DeltaNet kernel backends deterministic mode accepts.

Kept free of torch imports at module level so ``TransformerConfig`` can validate at construction
time and the GDN module can validate again at build time (``deterministic_mode`` may be set after
``__post_init__``, as Megatron-Bridge's determinism overrides do).

The ``transformer_engine`` backend is deterministic only inside the envelope where cuDNN serves
Gated DeltaNet from its FROST engine: q/k and v head dims of 64 or 128, an SM100 / SM103 / SM107
device and the CuTe DSL (``nvidia-cutlass-dsl``) installed. Outside it cuDNN falls back to its
cuTile engine, whose autotune pick moves bits until NVIDIA/cudnn-frontend#1735 lands. The head
dims are known from the config; the device and the DSL are checked at build time only, because
configs are routinely constructed on CPU hosts.
"""

import importlib.util

DETERMINISTIC_GDN_BACKENDS = frozenset({"torch", "transformer_engine"})
"""Backends whose kernels replay bit-exactly: the torch reference and TE over cuDNN (FROST)."""

FROST_GDN_HEAD_DIMS = frozenset({64, 128})
"""q/k and v head dims cuDNN's FROST GDN engine serves; other dims go to the cuTile engine."""

FROST_COMPUTE_CAPABILITY_MAJOR = 10
"""FROST runs on SM100 / SM103 / SM107; Hopper and earlier go to the cuTile engine."""

_CUTILE_GUIDANCE = (
    "cuDNN would serve Gated DeltaNet from its cuTile engine, whose autotune pick is not verified "
    "deterministic (NVIDIA/cudnn-frontend#1735). Use gdn_kernel_backend='torch' instead."
)


def check_deterministic_gdn_backend(config, *, check_device: bool = False) -> None:
    """Raise when ``config`` pairs ``deterministic_mode`` with a GDN kernel that may not replay.

    Args:
        config: A ``TransformerConfig`` (or a stand-in with ``deterministic_mode``,
            ``gdn_kernel_backend``, ``linear_key_head_dim`` and ``linear_value_head_dim``).
        check_device: Also require the current CUDA device and the CuTe DSL to be inside the
            FROST envelope. Set at module build time, where a device is at hand; the config-time
            check covers the head dims only.
    """
    if not config.deterministic_mode:
        return
    backend = config.gdn_kernel_backend
    if backend not in DETERMINISTIC_GDN_BACKENDS:
        accepted = ", ".join(repr(name) for name in sorted(DETERMINISTIC_GDN_BACKENDS))
        raise ValueError(
            f"deterministic_mode=True requires gdn_kernel_backend in {{{accepted}}} for "
            f"Gated DeltaNet; got {backend!r}, whose kernel is not deterministic."
        )
    if backend != "transformer_engine":
        return

    head_dims = (config.linear_key_head_dim, config.linear_value_head_dim)
    if any(dim not in FROST_GDN_HEAD_DIMS for dim in head_dims):
        allowed = sorted(FROST_GDN_HEAD_DIMS)
        raise ValueError(
            "deterministic_mode=True with gdn_kernel_backend='transformer_engine' requires "
            f"linear_key_head_dim and linear_value_head_dim in {allowed}; got {head_dims}. "
            f"For these dims {_CUTILE_GUIDANCE}"
        )
    if not check_device:
        return

    import torch

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] != FROST_COMPUTE_CAPABILITY_MAJOR:
            raise ValueError(
                "deterministic_mode=True with gdn_kernel_backend='transformer_engine' requires "
                f"an SM100 / SM103 / SM107 device; found compute capability "
                f"{capability[0]}.{capability[1]}. On this device {_CUTILE_GUIDANCE}"
            )
    if importlib.util.find_spec("cutlass") is None:
        raise ValueError(
            "deterministic_mode=True with gdn_kernel_backend='transformer_engine' requires the "
            "CuTe DSL (nvidia-cutlass-dsl), which cuDNN's FROST GDN engine is built with; without "
            f"it {_CUTILE_GUIDANCE}"
        )
