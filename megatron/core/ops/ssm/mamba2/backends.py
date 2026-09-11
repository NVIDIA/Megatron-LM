# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Construction-time loading of Mamba scan kernels and their capabilities."""

import inspect
from dataclasses import dataclass
from typing import Callable

from megatron.core.ops.kernel_metadata import DeterminismPolicy, validate_kernel, validate_kernels
from megatron.core.ops.ssm.common.kernel_metadata import CAUSAL_CONV
from megatron.core.ops.ssm.mamba2.kernel_metadata import MAMBA_SCAN, MAMBA_SPLIT_SCAN


@dataclass(frozen=True)
class MambaKernels:
    """Direct targets for ordinary scan and optional memory-efficient training."""

    scan: Callable
    split_scan: Callable | None
    has_state_dtype: bool
    causal_conv1d: Callable | None


def select_mamba_kernels(use_mem_eff_path: bool, deterministic: bool = False) -> MambaKernels:
    """Load the scan needed by prefill and the explicitly enabled fused training scan."""
    policy = DeterminismPolicy.WARN if deterministic else DeterminismPolicy.IGNORE
    validate_kernels(
        (MAMBA_SCAN, MAMBA_SPLIT_SCAN) if use_mem_eff_path else (MAMBA_SCAN,), determinism=policy
    )
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    split_scan = None
    targets = [mamba_chunk_scan_combined]
    if use_mem_eff_path:
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

        split_scan = mamba_split_conv1d_scan_combined
        targets.append(split_scan)
    try:
        validate_kernel(CAUSAL_CONV, determinism=policy)
    except ImportError as exc:
        # Mamba's ordinary path historically uses Torch convolution when the package is absent.
        # A broken installation or missing export is not an alternative implementation choice.
        if (
            use_mem_eff_path
            or not isinstance(exc.__cause__, ModuleNotFoundError)
            or exc.__cause__.name != "causal_conv1d"
        ):
            raise
        conv = None
    else:
        from causal_conv1d import causal_conv1d_fn

        conv = causal_conv1d_fn
    return MambaKernels(
        scan=mamba_chunk_scan_combined,
        split_scan=split_scan,
        has_state_dtype=all(
            "state_dtype" in inspect.signature(target).parameters for target in targets
        ),
        causal_conv1d=conv,
    )
