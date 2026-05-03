# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU peak FLOPS lookup for MFU (Model FLOPs Utilization) calculation.

Peak BF16 dense tensor core FLOPS for supported GPU architectures.
These values are per-GPU and do NOT include structured sparsity.
"""

import logging
import torch

logger = logging.getLogger(__name__)

# BF16 dense tensor core peak TFLOP/s per GPU
# Dense BF16 tensor core peak (no structured sparsity).
# Sparse (2:4) peak is 2× these values.
_GPU_PEAK_TFLOPS_BF16 = {
    "A100-SXM": 312,
    "A100-PCIE": 312,
    "H100-SXM": 989,
    "H100-PCIE": 756,
    "H200": 989,
    "B100": 1750,
    "B200": 2250,
    "GB200": 2250,
}


def get_gpu_peak_tflops(dtype=torch.bfloat16):
    """Detect GPU model and return peak TFLOP/s for the given dtype.

    Currently only supports BF16. Returns the dense (non-sparse) tensor core peak.
    Falls back to a conservative estimate based on GPU architecture generation
    if the specific model is not found.

    Returns:
        float: Peak TFLOP/s for one GPU, or 0.0 if detection fails.
    """
    if dtype != torch.bfloat16:
        logger.warning(f"MFU peak lookup only supports BF16, got {dtype}. Returning 0.")
        return 0.0

    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        logger.warning("Could not detect GPU. MFU will not be reported.")
        return 0.0

    name_upper = device_name.upper()

    for key, tflops in _GPU_PEAK_TFLOPS_BF16.items():
        if key.upper().replace("-", "") in name_upper.replace("-", "").replace(" ", ""):
            logger.info(f"GPU detected: {device_name} -> peak BF16: {tflops} TFLOP/s")
            return float(tflops)

    # Fallback by architecture generation
    try:
        major = torch.cuda.get_device_properties(0).major
        fallback = {8: 312, 9: 989, 10: 2250}  # Ampere, Hopper, Blackwell
        if major in fallback:
            tflops = fallback[major]
            logger.warning(
                f"GPU '{device_name}' not in lookup table. "
                f"Using arch-generation fallback (sm_{major}0): {tflops} TFLOP/s"
            )
            return float(tflops)
    except Exception:
        pass

    logger.warning(f"Unknown GPU '{device_name}'. MFU will not be reported.")
    return 0.0
