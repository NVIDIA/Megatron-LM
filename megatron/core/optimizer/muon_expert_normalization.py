# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Keep native per-matrix normalization while screening batched NS GEMMs."""

import inspect

import torch
from emerging_optimizers.orthogonalized_optimizers import muon_utils

_SIGNATURE = inspect.signature(muon_utils.newton_schulz)
_EPS = _SIGNATURE.parameters['eps'].default
if 'normalize_in_double' in _SIGNATURE.parameters:
    if _SIGNATURE.parameters['normalize_in_double'].default is not False:
        raise RuntimeError('Expert batching requires native single-precision normalization.')


def reference_norm_batched_ns(
    grad,
    steps,
    coefficient_type,
    tp_group,
    partition_dim=None,
    tp_mode='duplicated',
    use_syrk=False,
):
    """Use the same 2D normalization operations before native batched NS steps."""
    if (
        grad.ndim != 3
        or grad.dtype != torch.float32
        or torch.get_float32_matmul_precision() != 'medium'
        or partition_dim is not None
        or tp_mode != 'duplicated'
        or use_syrk
        or steps != 16
        or coefficient_type != 'polar_express'
        or tp_group.size() != 1
    ):
        raise ValueError('Only original local FP32-input/BF16-compute NS16 is covered')
    transpose = grad.size(-2) > grad.size(-1)
    matrices = [value.mT if transpose else value for value in grad.unbind(0)]
    # Exactly match original 2D input shape, strides, dtype, eps, and reduction.
    normalized = [
        torch.nn.functional.normalize(value, p=2, dim=(-2, -1), eps=_EPS) for value in matrices
    ]
    # Preserve each native matrix's post-normalization transpose layout too.
    packed = torch.stack([value.mT if transpose else value for value in normalized])
    packed = packed.to(torch.bfloat16)
    value = packed.mT if transpose else packed
    coefficients = muon_utils.get_coefficient_iterator(
        steps, muon_utils._COEFFICIENT_SETS[coefficient_type], mode='repeat_last'
    )
    for a, b, c in coefficients:
        value = muon_utils.batched_newton_schulz_step(value, a, b, c, tp_group=None)
    value = value.to(torch.float32)
    return value.mT if transpose else value
