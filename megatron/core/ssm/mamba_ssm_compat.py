# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility patches for the installed mamba_ssm package.

mamba_ssm 2.2.6.post3's ``rearrange_and_update_stride`` (ops/triton/
ssd_combined.py) densifies its input with a plain ``.contiguous()`` whenever
the rearranged tensor's ``stride(dim)`` is not a multiple of 8. For the
``b s d -> b d s`` transposed views used around the causal-conv1d call this
converts a channel-last view into a channels-FIRST tensor, after which
causal-conv1d's C++ binding rejects any ``seq_idx`` with
"seq_idx is only supported for channel last layout" (csrc/causal_conv1d.cpp,
stride-only check). Whether densification triggers depends on the per-rank
zxBCdt concat width W: at TP2, CP4 gives W=1288 (aligned, works) while CP8
gives W=772 (misaligned, crashes) — the offending term is the dt block,
nheads/(tp*cp).

The patch below replaces the densification with a channel-last-preserving
one (``transpose(1, 2).contiguous().transpose(1, 2)``), which satisfies both
the %8 stride-alignment goal (the resulting stride(dim) equals the channel
count, a multiple of 8 for all our conv widths) and the conv kernel's
channel-last requirement. Monkeypatching the module-level name covers both
the forward and backward internal uses.

Validated empirically (4-leg single-GPU repro, 2026-07-20): W=772 with the
patch passes where unpatched raises the exact production error.
"""

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_mamba_ssm_channel_last_patch():
    """Idempotently patch mamba_ssm's rearrange_and_update_stride.

    Safe no-op when mamba_ssm is absent or has no such function. The wrapper
    preserves upstream behaviour exactly except for the densification layout.
    """
    global _PATCHED
    if _PATCHED:
        return
    try:
        import mamba_ssm
        import mamba_ssm.ops.triton.ssd_combined as _ssd
        from einops import rearrange as _rearrange
    except ImportError:
        return
    if not hasattr(_ssd, 'rearrange_and_update_stride'):
        return

    def _channel_last_rearrange_and_update_stride(tensor, pattern=None, dim=2):
        t = _rearrange(tensor, pattern) if pattern is not None else tensor
        if t.stride(dim) % 8 != 0:
            if t.dim() == 3 and t.stride(1) == 1:
                # Densify while preserving channel-last: materialize in
                # (b, s, d) order, then re-view as (b, d, s). stride(dim)
                # becomes the channel count, which is 8-aligned for all
                # supported conv widths; fall back if it is not.
                t = t.transpose(1, 2).contiguous().transpose(1, 2)
                if t.stride(dim) % 8 != 0:
                    t = t.contiguous()
            else:
                t = t.contiguous()
        return t

    _ssd.rearrange_and_update_stride = _channel_last_rearrange_and_update_stride
    _PATCHED = True
    logger.info(
        "mamba_ssm_compat: patched rearrange_and_update_stride for "
        "channel-last-preserving densification (mamba_ssm %s)",
        getattr(mamba_ssm, '__version__', 'unknown'),
    )
