from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]


def build_fp8_recipe(train_config=None):
    """Build the standard TE FP8 recipe (DelayedScaling, HYBRID format, H100)."""
    from transformer_engine.common.recipe import DelayedScaling, Format

    return DelayedScaling(margin=0, fp8_format=Format.HYBRID)


def ensure_divisible(numerator: int, denominator: int, msg: str = "") -> int:
    if numerator % denominator != 0:
        detail = f" ({msg})" if msg else ""
        raise ValueError(f"{numerator} is not divisible by {denominator}{detail}")
    return numerator // denominator


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[Megatron Lite] {msg}", flush=True)


__all__ = [
    "build_fp8_recipe",
    "ensure_divisible",
    "log_rank0",
]
