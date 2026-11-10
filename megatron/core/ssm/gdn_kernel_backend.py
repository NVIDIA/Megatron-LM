# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Which Gated DeltaNet kernel backends deterministic mode accepts.

Kept free of torch imports so ``TransformerConfig`` can validate at construction time and the
GDN module can validate again at build time (``deterministic_mode`` may be set after
``__post_init__``, as Megatron-Bridge's determinism overrides do).
"""

DETERMINISTIC_GDN_BACKENDS = frozenset({"torch", "transformer_engine"})
"""Backends whose kernels replay bit-exactly: the torch reference and TE over cuDNN."""


def check_deterministic_gdn_backend(config) -> None:
    """Raise when ``config`` pairs ``deterministic_mode`` with a non-deterministic GDN backend."""
    backend = config.gdn_kernel_backend
    if config.deterministic_mode and backend not in DETERMINISTIC_GDN_BACKENDS:
        accepted = ", ".join(repr(name) for name in sorted(DETERMINISTIC_GDN_BACKENDS))
        raise ValueError(
            f"deterministic_mode=True requires gdn_kernel_backend in {{{accepted}}} for "
            f"Gated DeltaNet; got {backend!r}, whose kernel is not deterministic."
        )
