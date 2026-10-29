# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
__all__ = [
    "DSAIndexShareState",
    "DynamicSparseAttention",
    "MagiAttentionConfig",
    "MagiDotProductAttention",
    "MultiLatentAttention",
    "RMSNorm",
    "build_rope_cache",
    "build_rotary_embeddings",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = (
        "magi" if name.startswith("Magi") else "mla" if name == "MultiLatentAttention" else "dsa"
    )
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
