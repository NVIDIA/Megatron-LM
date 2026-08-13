# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Availability of the CuteDSL varlen SSD backend.

Kept free of Triton and CuteDSL imports so the inference layer can ask whether
the kernel could run here without pulling either runtime in. Which backend is
actually used is a configuration choice (`InferenceConfig.mamba_prefill_backend`),
not something this module decides.
"""

import torch

_CUTEDSL_SSD_AVAILABLE = None


def cutedsl_ssd_available():
    """Whether the CuteDSL SSD kernel can run on this system.

    True on Blackwell (SM 10.0+) with an importable CuteDSL runtime. The result
    is cached in `_CUTEDSL_SSD_AVAILABLE` (tests may override that global
    directly to force a backend).
    """
    global _CUTEDSL_SSD_AVAILABLE
    if _CUTEDSL_SSD_AVAILABLE is not None:
        return _CUTEDSL_SSD_AVAILABLE

    available = False
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            from .cutedsl_mamba2_ssd import is_cutedsl_ssd_available

            available = is_cutedsl_ssd_available()
    except Exception:
        available = False
    _CUTEDSL_SSD_AVAILABLE = available
    return available
