# Copyright (c) 2026, ETH Zurich.

"""Aurora optimizer (Tilde Research, 2026).

Spectral update with row-balance: alternating projection onto the
intersection of the column-orthogonal Stiefel manifold and the row-uniform
oblique manifold. Drop-in for Muon on tall matrices, where Muon's polar
update produces persistent row-norm imbalance ("dead neurons").

References:
- Blog: https://blog.tilderesearch.com/blog/aurora
- Reference impl: https://github.com/tilde-research/aurora-release
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size

try:
    from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object


logger = logging.getLogger(__name__)


# Tilde defaults; match Keller Jordan modded-nanogpt track-3 byte-for-byte.
# Cubic-rate convergence with sigma=1 super-attracting; 12 iters drives all
# singular values in (0, sqrt(2)) to 1 within bf16 precision.
_NS_COEFFS: tuple[float, float, float] = (2.0, -1.5, 0.5)
_NS_STEPS: int = 12


@torch.no_grad()
def polar(
    G: torch.Tensor,
    precision: Literal["bf16", "fp32"] = "bf16",
    num_ns_steps: int = _NS_STEPS,
    coeffs: tuple[float, float, float] = _NS_COEFFS,
) -> torch.Tensor:
    """Polar factor U V^T of G via simple-quintic Newton-Schulz.

    Returns a tensor in the requested precision and the same shape as G;
    all non-zero singular values of G are mapped to ~1. Tall matrices are
    transposed internally so the inner X @ X.mT stays at [n, n] rather than
    [m, m] when m >> n.
    """
    assert G.ndim >= 2
    target_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    X = G.to(target_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = coeffs
    for _ in range(num_ns_steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.no_grad()
def aurora_orthogonalize(
    grad: torch.Tensor,
    precision: Literal["bf16", "fp32"] = "bf16",
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    num_ns_steps: int = _NS_STEPS,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Aurora's leverage-uniform polar update.

    Joint constraint: U^T U = I_n AND ||U_i:||^2 = n/m. Approximated by
    pp_iterations passes of polar(D * G), updating the diagonal preconditioner
    D between passes to drive row sq-norms toward target n/m. For square
    matrices reduces to plain polar (no row-balance freedom to exploit).
    """
    if grad.ndim != 2:
        raise ValueError(f"Aurora requires 2D grad, got shape {tuple(grad.shape)}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if not (0.0 < pp_beta <= 1.0):
        raise ValueError(f"pp_beta must be in (0, 1], got {pp_beta}")

    m, n = grad.size(-2), grad.size(-1)
    if m == n:
        return polar(grad, precision=precision, num_ns_steps=num_ns_steps)

    transposed = m < n
    if transposed:
        grad = grad.mT
        m, n = n, m

    G32 = grad.to(torch.float32)
    target_row_sq = n / m
    row_norm = G32.norm(dim=-1, keepdim=True).clamp_min(eps)
    D = 1.0 / row_norm

    U: torch.Tensor | None = None
    for k in range(pp_iterations):
        U = polar(D * G32, precision=precision, num_ns_steps=num_ns_steps)
        if k < pp_iterations - 1:
            row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps)
            D = D * (target_row_sq / row_sq).pow(pp_beta)
    assert U is not None
    return U.mT if transposed else U


def _aurora_scale_factor(m: int, n: int) -> float:
    """Spectral aspect-ratio scale (Muon convention: max(1, m/n)^0.5)."""
    return max(1.0, m / n) ** 0.5


class TensorParallelAurora(OrthogonalizedOptimizer):
    """Aurora optimizer wired through emerging_optimizers' OrthogonalizedOptimizer parent.

    The parent handles SGD-momentum + Nesterov + decoupled weight decay; this
    class supplies the orthogonalize step (Aurora's alternating polar+rebalance
    plus the spectral aspect-ratio scale). v1 supports tp_mode='duplicated'
    only; with TP > 1 and partition_dim set, the polar runs on each rank's
    local shard (matches Muon's existing duplicated-mode behavior).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.025,
        use_decoupled_weight_decay: bool = True,
        polar_precision: Literal["bf16", "fp32"] = "bf16",
        pp_iterations: int = 2,
        pp_beta: float = 0.5,
        num_ns_steps: int = _NS_STEPS,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["duplicated"] = "duplicated",
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
    ) -> None:
        if not HAVE_EMERGING_OPTIMIZERS:
            raise ImportError(
                "emerging_optimizers is required for Aurora's parent class "
                "(OrthogonalizedOptimizer handles momentum + WD)."
            )
        if scale_mode != "spectral":
            raise ValueError(
                f"Aurora hardcodes the spectral aspect-ratio scale; "
                f"got scale_mode={scale_mode!r}. Use --muon-scale-mode spectral."
            )
        # Match Muon's pattern: 'blockwise' reduces to 'duplicated' at the
        # orthogonalize call site (each rank polars its full local matrix).
        # 'distributed' (TP-coordinated polar) is not supported in v1.
        if tp_mode not in ("duplicated", "blockwise"):
            raise ValueError(
                f"Aurora v1 supports tp_mode in ('duplicated', 'blockwise'), got {tp_mode!r}."
            )

        def scaled_orthogonalize_fn(
            g: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup | None,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            size = [g.size(-2), g.size(-1)]
            if partition_dim is not None and tp_group is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            update = aurora_orthogonalize(
                g,
                precision=polar_precision,
                pp_iterations=pp_iterations,
                pp_beta=pp_beta,
                num_ns_steps=num_ns_steps,
            )
            scale = _aurora_scale_factor(size[0], size[1])
            return update * scale * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self._aurora_polar_precision = polar_precision
        self._aurora_pp_iterations = pp_iterations
        self._aurora_pp_beta = pp_beta
        self._aurora_num_ns_steps = num_ns_steps

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply Aurora's leverage-uniform polar to the momentum tensor."""
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, "expert_tp", False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            grad_shape = grad.shape
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)  # type: ignore[arg-type]
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),  # type: ignore[arg-type]
                self.qkv_split_shapes,  # type: ignore[arg-type]
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]
            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad
