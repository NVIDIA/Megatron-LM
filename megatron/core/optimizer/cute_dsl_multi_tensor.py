# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Route multi-tensor Adam, L2 norm and zero count to persistent CuTe DSL kernels.

Transformer Engine's ``multi_tensor_adam`` and ``multi_tensor_l2norm`` run
``multi_tensor_apply``, which caps one launch at 320 chunks of 65,536 elements: an
optimizer call over a few billion fp32 elements issues hundreds of launches, each paying a
fixed launch and tail cost. The kernels in ``cute_dsl_multi_tensor_kernels`` cover a whole
tensor list in one launch (a persistent grid pulling chunks from an atomic counter) with the
same fp32 Adam terms, betas, eps and bias correction, and the same per-chunk partial sum of
squares then reduction for the norm.

``adam_impl`` and ``l2norm_impl`` wrap TE's callables with drop-ins of the same signature.
They use the kernels only after ``enable`` succeeded (``OptimizerConfig.
use_cute_dsl_multi_tensor``); a call the kernels do not cover (a non-fp32 tensor,
per-tensor norms, L2-mode Adam, another chunk size, or a call under CUDA graph capture, since
the kernels upload their tensor table from a host buffer) falls through to TE unchanged.
``count_zeros`` counts the zeros of the ``--log-num-zeros-in-grad`` statistic in one launch per
dtype; it returns None where the caller keeps its ``torch.count_nonzero`` path.
"""

import logging
from types import ModuleType
from typing import Callable, List, Optional

import torch

logger = logging.getLogger(__name__)

_kernels: Optional[ModuleType] = None


def enable() -> bool:
    """Load the CuTe DSL kernels so the wrappers start using them.

    Returns:
        True when the kernels are in use: CUDA is available, the GPU is SM100 or newer and
        ``nvidia-cutlass-dsl`` imports. Otherwise logs why and returns False, and the
        wrappers keep calling Transformer Engine.
    """
    global _kernels
    if _kernels is not None:
        return True
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        logger.warning("CuTe DSL multi-tensor kernels need an SM100+ GPU; using TE kernels")
        return False
    try:
        from megatron.core.optimizer import cute_dsl_multi_tensor_kernels
    except ImportError as e:
        logger.warning("CuTe DSL multi-tensor kernels unavailable (%s); using TE kernels", e)
        return False
    _kernels = cute_dsl_multi_tensor_kernels
    return True


def adam_impl(te_adam: Callable) -> Callable:
    """Return a drop-in for ``transformer_engine_torch.multi_tensor_adam``.

    Args:
        te_adam: TE's ``multi_tensor_adam``, called for anything the kernel does not cover.

    Returns:
        A callable with ``te_adam``'s signature.
    """

    def multi_tensor_adam(
        chunk_size,
        noop_flag,
        tensor_lists,
        lr,
        beta1,
        beta2,
        eps,
        step,
        mode,
        bias_correction,
        weight_decay,
    ):
        if (
            _kernels is not None
            and not torch.cuda.is_current_stream_capturing()
            and len(tensor_lists) == 4
            and chunk_size == _kernels.CHUNK
            and _kernels.multi_tensor_adam(
                *tensor_lists, lr, beta1, beta2, eps, step, mode, bias_correction, weight_decay
            )
        ):
            return None
        return te_adam(
            chunk_size,
            noop_flag,
            tensor_lists,
            lr,
            beta1,
            beta2,
            eps,
            step,
            mode,
            bias_correction,
            weight_decay,
        )

    return multi_tensor_adam


def l2norm_impl(te_l2norm: Callable) -> Callable:
    """Return a drop-in for ``multi_tensor_l2norm`` (TE, Apex or the local fallback).

    Args:
        te_l2norm: The original ``multi_tensor_l2norm``, called for anything the kernel does
            not cover.

    Returns:
        A callable with ``te_l2norm``'s signature.
    """

    def multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor=False):
        if (
            _kernels is not None
            and not torch.cuda.is_current_stream_capturing()
            and not per_tensor
            and len(tensor_lists) == 1
            and chunk_size == _kernels.CHUNK
        ):
            ret = _kernels.multi_tensor_l2norm(tensor_lists[0], noop_flag)
            if ret is not None:
                return ret, torch.empty(0, dtype=torch.float32, device=ret.device)
        return te_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

    return multi_tensor_l2norm


def count_zeros(base: int, tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """Count zeros with one persistent launch per dtype.

    Args:
        base: The number of counted elements; the result is ``base`` minus the nonzeros.
        tensors: Contiguous bf16, fp16 or fp32 CUDA tensors to count.

    Returns:
        ``base`` minus the nonzero elements of ``tensors`` as a 0-d int64 tensor, or None
        when the kernels are not enabled, under CUDA graph capture, or for tensors they do not
        cover (the caller then counts with ``torch.count_nonzero``).
    """
    if _kernels is None or torch.cuda.is_current_stream_capturing():
        return None
    return _kernels.multi_tensor_count_zeros(tensors, base)
