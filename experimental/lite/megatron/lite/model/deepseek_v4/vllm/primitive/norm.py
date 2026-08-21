from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from .recompute import visible_functional_vjp


def _norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(value.float(), (value.shape[-1],), weight.float(), eps).to(
        value.dtype
    )


def rms_norm(visible_op: Callable, value, weight, eps):
    return visible_functional_vjp(
        lambda value_, weight_: visible_op(value_, weight_, eps),
        lambda value_, weight_: _norm(value_, weight_, eps),
        (value, weight),
        version_indices=(1,),
    )


def fused_qkv_rms_norm(visible_op: Callable, q, kv, q_weight, kv_weight, eps):
    return visible_functional_vjp(
        lambda q_, kv_, qw_, kvw_: visible_op(q_, kv_, qw_, kvw_, eps),
        lambda q_, kv_, qw_, kvw_: (
            _norm(q_, qw_, eps),
            _norm(kv_, kvw_, eps),
        ),
        (q, kv, q_weight, kv_weight),
        version_indices=(2, 3),
    )
