# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core._rank_utils import log_single_rank
from megatron.core.component_nvtx import component
from megatron.core.env_flags import FALSE_VALUES, TRUE_VALUES
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.residual_recompute_plan import (
    build_recompute_block_end_plan,
    build_recompute_layer_managers,
    finalize_recompute_block,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import ensure_params_ready, nvtx_decorator

_MHC_SINKHORN_EPS = 1e-6
_MHC_COMPUTE_H_EPS = 1e-6


@dataclass
class SinglePassMHCState:
    """Previous sublayer's input mix, owned by one forward invocation.

    A fresh state is required for every microbatch. The final sublayer's mix
    contracts the stack output; no separate learned output head is needed.
    """

    pre_mix: Tensor | None = None

    @component("mhc")
    def contract(
        self,
        hidden_states: Tensor,
        n: int,
        *,
        aggregate: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        """Contract residual streams in FP32, selecting stream zero at stack entry."""
        if n < 1 or hidden_states.ndim != 3 or hidden_states.shape[-1] % n:
            raise ValueError("Expected [sequence, batch, n * hidden] residual streams")
        streams = hidden_states.unflatten(-1, (n, -1))
        if self.pre_mix is None:
            return streams[..., 0, :].contiguous()
        if self.pre_mix.shape != hidden_states.shape[:2] + (n,):
            raise ValueError("The preceding mHC mix must have shape [sequence, batch, n]")
        if aggregate is not None:
            return aggregate(streams, self.pre_mix).to(hidden_states.dtype)
        return (
            (streams.float() * self.pre_mix.float().unsqueeze(-1)).sum(-2).to(hidden_states.dtype)
        )


def get_mhc_aggregate(config: TransformerConfig) -> Callable[[Tensor, Tensor], Tensor] | None:
    """Bind the configured fused aggregation for the stack's final contraction."""
    if not config.use_fused_mhc:
        return None
    from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate

    return partial(fused_h_aggregate, backend=config.mhc_fused_backend)


def build_mhc_recompute_layer_plan(
    num_layers: int, mhc_recompute_layer_num: Optional[int], use_mhc_recompute: bool
) -> Tuple[list[Optional[CheckpointWithoutOutputManager]], list[bool]]:
    """Build per-layer mHC recompute managers and recompute-block end markers."""
    if not use_mhc_recompute or num_layers == 0:
        return [None] * num_layers, [False] * num_layers

    block_ends = build_recompute_block_end_plan(num_layers, mhc_recompute_layer_num)
    return build_recompute_layer_managers(block_ends), list(block_ends)


def finalize_mhc_recompute_layer(
    mhc_manager: Optional[CheckpointWithoutOutputManager],
    hidden_states: Tensor,
    is_last_in_recompute_block: bool,
) -> None:
    """Finalize mHC recompute state when the current recompute block ends."""
    finalize_recompute_block(mhc_manager, hidden_states, is_last_in_recompute_block)


@torch.compile
def _sinkhorn_iterations(input_logits: Tensor, num_iterations: int, eps: float) -> Tensor:
    M = input_logits.softmax(dim=-1) + eps
    M = M / (M.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iterations - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class SinkhornKnopp(torch.autograd.Function):
    """Sinkhorn-Knopp projection to doubly stochastic matrix.

    This is an autograd.Function because the iterative forward is re-executed
    during backward (under torch.enable_grad) so that PyTorch's autograd can
    differentiate through it without storing all intermediate iteration states.
    """

    @staticmethod
    def forward(ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
        """Run Sinkhorn iterations and save inputs for backward recomputation."""
        M = _sinkhorn_iterations(input_logits, num_iterations, eps)
        ctx.save_for_backward(input_logits)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return M

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Recompute forward under enable_grad and back-propagate."""
        (input_logits,) = ctx.saved_tensors
        with torch.enable_grad():
            logits = input_logits.detach().requires_grad_(True)
            M = _sinkhorn_iterations(logits, ctx.num_iterations, ctx.eps)
            M.backward(grad_output)
        return logits.grad, None, None


def native_sinkhorn(input_logits: Tensor, num_iterations: int, eps: float = 1e-6) -> Tensor:
    """Native Sinkhorn-Knopp (autograd.Function wrapper)."""
    return SinkhornKnopp.apply(input_logits, num_iterations, eps)


@torch.compile
def native_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Native n-stream weighted aggregation: out = sum_j(h_pre_j * x_j)."""
    return (x * h_pre.unsqueeze(-1)).sum(dim=2)


@torch.compile
def native_h_post_bda(
    h_res: Tensor, original_residual: Tensor, h_post: Tensor, x: Tensor, bias: Optional[Tensor]
) -> Tensor:
    """Native H_res.T @ residual + H_post * (x [+ bias])."""
    s, b, n, C = original_residual.shape
    h_res_batched = h_res.view(s * b, n, n)
    residual_batched = original_residual.view(s * b, n, C)
    mixed = torch.bmm(h_res_batched.transpose(1, 2), residual_batched).view(s, b, n, C)
    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)
    if bias is not None:
        bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
        return x_expanded + bias_expanded + mixed
    return x_expanded + mixed


@torch.compile
def native_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Native fused projection + RMS normalization."""
    proj = torch.matmul(x, weight.t())
    norm = x.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    v = norm / math.sqrt(K) + eps
    r = 1.0 / v
    return proj, r


@torch.compile
def native_fused_add_3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Native 3-way elementwise add (torch.compile fuses into single kernel)."""
    return a + b + c


class BroadcastTensorFused(torch.autograd.Function):
    """Split one tensor into 3 autograd-graph children sharing the same storage.

    During backward the three incoming gradients are summed with a caller-
    supplied fused-add function (cuTile or torch.compile fallback) instead of
    PyTorch's default sequential accumulation.
    """

    @staticmethod
    def forward(ctx, x, fused_add_3_fn):
        """Return three view aliases and save the fused gradient combiner."""
        ctx.fused_add_3_fn = fused_add_3_fn
        return x.view_as(x), x.view_as(x), x.view_as(x)

    @staticmethod
    def backward(ctx, grad1, grad2, grad3):
        """Combine gradients from the three broadcast aliases."""
        grads = [g for g in (grad1, grad2, grad3) if g is not None]
        if len(grads) == 0:
            return None, None
        if len(grads) == 1:
            return grads[0], None
        if len(grads) == 2:
            return grads[0] + grads[1], None
        return ctx.fused_add_3_fn(grad1, grad2, grad3), None


def _parse_single_pass_fused_grad_add(value: Optional[str]) -> Optional[str]:
    """Return the MCORE_MHC_SP_FUSED_GRAD_ADD mode: None (off), "exact" or "fp32".

    Unset selects "exact". ``0``, ``false``, ``no`` and ``off`` turn the fused sum off;
    ``1``, ``true``, ``yes``, ``on`` and ``exact`` select the bit-exact sum, and ``fp32`` the
    FP32 one (the on and off spellings are those of ``env_flag``). Matching ignores case and
    surrounding whitespace; any other value raises.
    """
    if value is None:
        return "exact"
    mode = value.strip().lower()
    if mode in FALSE_VALUES:
        return None
    if mode in TRUE_VALUES or mode == "exact":
        return "exact"
    if mode == "fp32":
        return "fp32"
    raise ValueError(
        f"Environment variable MCORE_MHC_SP_FUSED_GRAD_ADD={value!r} is not a mode; use exact "
        "(or 1/true/yes/on), fp32 or 0/false/no/off, or unset it for exact."
    )


# MCORE_MHC_SP_FUSED_GRAD_ADD (read once at import). The single-pass forward reads each
# sublayer's residual input in three places: the mappings, the aggregate and the
# residual-mixing BDA, so autograd sums three full-size gradients with two sequential adds.
# Unless it is off, BroadcastTensorFused splits the input of callers that take the residual
# alias (return_residual=True), and one kernel sums the three gradients:
#   exact (default): (residual + aggregate) + mappings, rounded after each add exactly as
#                    autograd accumulates them in that arrival order (bit-identical gradients);
#   fp32:            native_fused_add_3 (torch.compile), rounding the FP32 sum once (opt-in,
#                    not bit-identical).
_SINGLE_PASS_FUSED_GRAD_ADD = _parse_single_pass_fused_grad_add(
    os.environ.get("MCORE_MHC_SP_FUSED_GRAD_ADD")
)
_SINGLE_PASS_FUSED_GRAD_ADD_LOGGED: set[str] = set()


def _ordered_single_pass_grad_add_3(
    add_3_sequential: Callable[[Tensor, Tensor, Tensor], Tensor],
    grad_mappings: Tensor,
    grad_aggregate: Tensor,
    grad_residual: Tensor,
) -> Tensor:
    """Sum in the order autograd accumulates the unsplit single-pass residual's gradients.

    The engine runs ready nodes latest-created first. The BDA ends the sublayer, so its residual
    gradient arrives first; the aggregate's arrives after the branch backward; the mapping nodes
    were created before the aggregate's checkpoint node, so their gradient arrives last.
    """
    return add_3_sequential(grad_residual, grad_aggregate, grad_mappings)


def _single_pass_grad_add_3_op(mode: str) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Return BroadcastTensorFused's combiner of (mappings, aggregate, residual) gradients."""
    if mode not in _SINGLE_PASS_FUSED_GRAD_ADD_LOGGED:
        _SINGLE_PASS_FUSED_GRAD_ADD_LOGGED.add(mode)
        log_single_rank(
            logging.getLogger(__name__),
            logging.INFO,
            f"[mHC] MCORE_MHC_SP_FUSED_GRAD_ADD={mode}: one fused sum per single-pass residual",
        )
    if mode == "fp32":
        return native_fused_add_3
    from megatron.core.fusions.fused_mhc_kernels import fused_add_3_sequential

    return partial(_ordered_single_pass_grad_add_3, fused_add_3_sequential)


@torch.compile
def learned_output_contract(
    hidden_states: Tensor, head_fn: Tensor, base: Tensor, scale: Tensor, n: int, eps: float
) -> Tensor:
    """Learned output contraction: n-stream → 1-stream via sigmoid-gated weighted sum."""
    dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    head_fn = head_fn.to(torch.float32)
    base = base.to(torch.float32)
    scale = scale.to(torch.float32)
    rsqrt = torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(hidden_states, head_fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + eps
    y = torch.sum(pre.unsqueeze(-1) * hidden_states.view(*hidden_states.shape[:-1], n, -1), dim=-2)
    return y.to(dtype)


@torch.compile
def _single_pass_projection_rms_forward(
    hidden_states: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute FP32 mapping and epsilon-inside-sqrt RMS without saving the wide cast."""
    x = hidden_states.float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        projected = F.linear(x, weight.float())
        inv_rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return projected * inv_rms, projected, inv_rms


@torch.compile
def _single_pass_projection_rms_backward(
    grad_output: Tensor,
    hidden_states: Tensor,
    weight: Optional[Tensor],
    projected: Optional[Tensor],
    inv_rms: Tensor,
    need_hidden: bool,
    need_weight: bool,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Recast original storage only in backward; merge both dX paths before rounding."""
    x = hidden_states.float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        grad = grad_output.float()
        grad_projected = grad * inv_rms
        grad_hidden = None
        if need_hidden:
            grad_hidden = F.linear(grad_projected, weight.float().t())
            grad_rms = (grad * projected).sum(-1, keepdim=True)
            grad_mean = grad_rms * -0.5 * inv_rms.pow(3)
            grad_square = grad_mean.expand_as(x) / x.shape[-1]
            grad_hidden = (grad_hidden + grad_square * (2 * x)).to(hidden_states.dtype)
        grad_weight = None
        if need_weight:
            grad_weight = (
                grad_projected.reshape(-1, grad_projected.shape[-1])
                .t()
                .matmul(x.reshape(-1, x.shape[-1]))
            )
    return grad_hidden, grad_weight


class _SinglePassProjectionRMS(torch.autograd.Function):
    """Retain original activation storage and only narrow FP32 mapping statistics.

    This boundary must stay outside the coefficient activation's compiled graph:
    jointly partitioning forward/backward can otherwise retain the wide FP32
    cast again. Saved tensors keep their original StorageImpl so checkpoint
    output storage can be discarded and rebound before backward reads it.
    """

    @staticmethod
    def forward(ctx, hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        normalized, projected, inv_rms = _single_pass_projection_rms_forward(
            hidden_states, weight, eps
        )
        ctx.need_hidden, ctx.need_weight = ctx.needs_input_grad[:2]
        ctx.weight_dtype = weight.dtype
        ctx.set_materialize_grads(False)
        if ctx.need_hidden or ctx.need_weight:
            ctx.save_for_backward(
                hidden_states,
                weight if ctx.need_hidden else None,
                projected if ctx.need_hidden else None,
                inv_rms,
            )
        return normalized

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: Optional[Tensor]):
        if grad_output is None:
            return None, None, None
        hidden_states, weight, projected, inv_rms = ctx.saved_tensors
        grad_hidden, grad_weight = _single_pass_projection_rms_backward(
            grad_output, hidden_states, weight, projected, inv_rms, ctx.need_hidden, ctx.need_weight
        )
        if grad_weight is not None:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None


@torch.compile
def _single_pass_coefficient_activations(
    projected: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply the original coefficient activations to the narrow FP32 projection."""
    pre = (projected[..., :n] * alpha_pre + bias[:n]).sigmoid()
    post = (projected[..., n : 2 * n] * alpha_post + bias[n : 2 * n]).sigmoid()
    residual = (projected[..., 2 * n :] * alpha_res + bias[2 * n :]).unflatten(-1, (n, n))
    return pre + eps, 2 * post, residual


# ============================================================================
# HyperConnectionModule
# ============================================================================


# TODO: keep hyper connection in fp32 computation
class HyperConnectionModule(MegatronModule):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.

    Implements the complete mHC propagation:
        x_{l+1} = H_res^T @ x_l + H_post^T @ F(H_pre @ x_l)

    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream → 1-stream (H_pre @ x)
    3. Expansion: 1-stream → n-stream (H_post^T @ output)
    4. Residual merge: H_res^T @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries

    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.mhc_num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.single_pass = config.mhc_single_pass
        self.sinkhorn_eps = config.mhc_epsilon if self.single_pass else _MHC_SINKHORN_EPS
        self.compute_h_eps = config.mhc_epsilon if self.single_pass else _MHC_COMPUTE_H_EPS

        # Projection weights for dynamic mappings
        # Input: [s, b, n*C] -> Output: n^2 + 2n values per token
        # - H_pre: n values
        # - H_post: n values
        # - H_res: n^2 values (before Sinkhorn projection)
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size, self.n * self.n + 2 * self.n, bias=False
        )

        init_alpha = config.mhc_init_gating_factor
        # Learnable scaling factors (Eq. 5 in paper)
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))

        # Static bias terms
        self.bias = nn.Parameter(torch.zeros(self.n * self.n + 2 * self.n))
        mark_keep_in_fp32(self.mapping_proj.weight)
        mark_keep_in_fp32(self.alpha_pre)
        mark_keep_in_fp32(self.alpha_post)
        mark_keep_in_fp32(self.alpha_res)
        mark_keep_in_fp32(self.bias)
        self.norm_eps = config.layernorm_epsilon if self.single_pass else 1e-6

        # Choose implementation: unified fused kernels vs reference modules.
        # The fused public API selects the backend per operation internally.
        # fused_add_3 always uses torch.compile (native_fused_add_3) regardless
        # of the kernel backend. cuTile's register overhead (56 regs/thread for
        # a trivial a+b+c) is not worth it for a pure memory-bound elementwise op.
        self._fused_add_3_op = native_fused_add_3

        # The fused path computes the projection and compute_h in one op, so
        # _projection_and_get_norm — and therefore _proj_rms_op — is only ever
        # reached on the unfused path.
        self._proj_rms_op = native_proj_rms

        if config.use_fused_mhc:
            from megatron.core.fusions.fused_mhc_kernels import (
                fused_h_aggregate,
                fused_h_post_bda,
                fused_proj_rms_compute_h,
                fused_sinkhorn,
                log_fused_mhc_backend_once,
            )

            backend = config.mhc_fused_backend
            log_fused_mhc_backend_once(backend)
            self._sinkhorn_op = partial(fused_sinkhorn, backend=backend)
            self._h_aggregate_op = partial(fused_h_aggregate, backend=backend)
            self._h_post_bda_op = partial(fused_h_post_bda, backend=backend)
            self._proj_rms_compute_h_op = partial(fused_proj_rms_compute_h, backend=backend)
        else:
            self._sinkhorn_op = native_sinkhorn
            self._h_aggregate_op = native_h_aggregate
            self._h_post_bda_op = native_h_post_bda
            self._proj_rms_compute_h_op = None

        # MCORE_MHC_SP_FUSED_GRAD_ADD: one fused sum of the single-pass residual's gradients.
        self._single_pass_grad_add_3_op = (
            _single_pass_grad_add_3_op(_SINGLE_PASS_FUSED_GRAD_ADD)
            if self.single_pass and _SINGLE_PASS_FUSED_GRAD_ADD is not None
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # Honor the mcore convention: skip weight init when the caller will load a
        # checkpoint over these parameters anyway (e.g. meta-device construction).
        if self.config.perform_initialization:
            nn.init.xavier_uniform_(self.mapping_proj.weight)

        # Set sequence_parallel attribute on parameters for gradient synchronization
        # across TP ranks when sequence_parallel is enabled.
        # This is required because HyperConnectionModule uses non-TP-aware layers
        # (nn.Linear, nn.RMSNorm) whose gradients need to be all-reduced.
        if self.config.sequence_parallel:
            setattr(self.mapping_proj.weight, 'sequence_parallel', True)
            setattr(self.alpha_pre, 'sequence_parallel', True)
            setattr(self.alpha_post, 'sequence_parallel', True)
            setattr(self.alpha_res, 'sequence_parallel', True)
            setattr(self.bias, 'sequence_parallel', True)

    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Projection + RMS normalization.

        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        # The mHC mapping computation runs in FP32: the parameters are kept in
        # FP32 and the activations are upcast here, then compute_mappings casts
        # the bounded mixing weights back to the activation dtype.
        x_2d = x.reshape(s * b, nC).to(torch.float32)
        weight = self.mapping_proj.weight.to(torch.float32)
        proj, r = self._proj_rms_op(x_2d, weight, self.norm_eps)
        return proj.view(s, b, -1), r.view(s, b, 1)

    @torch.compile
    def _compute_h(self, proj: Tensor, r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute h from projected hidden states and scaling factors.

        Args:
            proj: [s, b, n^2 + 2n] - projected hidden states
            r: [s, b, 1] - scaling factors

        Returns:
            h_pre: [s, b, n] - aggregation weights
            h_post: [s, b, n] - expansion weights
            h_res: [s, b, n^2] - residual mixing logits
        """
        alpha_ = torch.cat(
            [
                self.alpha_pre.expand(self.n),
                self.alpha_post.expand(self.n),
                self.alpha_res.expand(self.n * self.n),
            ],
            dim=-1,
        )

        h = r * proj * alpha_ + self.bias
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., : self.n].sigmoid() + self.compute_h_eps  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n : 2 * self.n].sigmoid() * 2
        h_res = h[..., 2 * self.n :]
        return h_pre, h_post, h_res

    def _single_pass_coefficients(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Keep the original input for backward instead of its wide FP32 cast."""
        projected = _SinglePassProjectionRMS.apply(
            hidden_states, self.mapping_proj.weight, self.norm_eps
        )
        return _single_pass_coefficient_activations(
            projected,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            self.bias,
            self.n,
            self.compute_h_eps,
        )

    def _compute_single_pass_mappings(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict FP32 coefficients, then apply the configured Sinkhorn kernel."""
        pre, post, residual = self._single_pass_coefficients(hidden_states)
        residual = self._sinkhorn_op(residual, self.sinkhorn_iterations, self.sinkhorn_eps)
        return pre, post, residual

    @nvtx_decorator(message="HyperConnection::compute_mappings")
    def compute_mappings(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute mHC mappings from input hidden states.

        Reference: Eq. (5) and (8) in mHC paper

        Args:
            x: [s, b, n*C] - n-stream hidden states

        Returns:
            h_pre: [s, b, n] - aggregation weights (sigmoid activated)
            h_post: [s, b, n] - expansion weights (2*sigmoid activated)
            h_res: [s, b, n, n] - residual mixing matrix (doubly stochastic)
        """
        # Mapping kernels read parameters directly, bypassing their owning-module pre-hooks.
        ensure_params_ready(
            (self.mapping_proj.weight, self.alpha_pre, self.alpha_post, self.alpha_res, self.bias)
        )
        if self.single_pass:
            return self._compute_single_pass_mappings(x)

        s, b, _ = x.shape

        if self._proj_rms_compute_h_op is not None:
            # Fused path: proj_rms + compute_h in one kernel launch sequence
            x_2d = x.reshape(s * b, self.n * self.hidden_size)
            with torch.cuda.nvtx.range("HyperConnection::fused_proj_rms_compute_h"):
                h_pre, h_post, h_res, _ = self._proj_rms_compute_h_op(
                    x_2d,
                    self.mapping_proj.weight,
                    self.alpha_pre,
                    self.alpha_post,
                    self.alpha_res,
                    self.bias,
                    self.n,
                    self.norm_eps,
                    self.compute_h_eps,
                )
            h_pre = h_pre.view(s, b, self.n)
            h_post = h_post.view(s, b, self.n)
            h_res = h_res.view(s, b, self.n, self.n)
        else:
            # Native path: separate proj_rms + _compute_h
            with torch.cuda.nvtx.range("HyperConnection::projection_and_get_norm"):
                proj, r = self._projection_and_get_norm(x)
            with torch.cuda.nvtx.range("HyperConnection::compute_h"):
                h_pre, h_post, h_res = self._compute_h(proj, r)
            h_res = h_res.view(s, b, self.n, self.n)

        h_res = self._sinkhorn_op(
            h_res, self.sinkhorn_iterations, self.sinkhorn_eps
        )  # [s, b, n, n]

        # The mixing weights are bounded (sigmoid outputs / doubly stochastic
        # matrix), so after the FP32 computation they are safe to apply to the
        # streams in the activation dtype.
        dtype = x.dtype
        return h_pre.to(dtype), h_post.to(dtype), h_res.to(dtype)

    @torch.compile
    def _apply_h_post(self, x: Tensor, h_post: Tensor) -> Tensor:
        """
        Core implementation of H_post application to a single tensor.

        Computes: H_post^T @ x

        Args:
            x: Input tensor, can be either:
               - [s, b, C] - standard hidden states
               - [C] - bias tensor (will be broadcast)
            h_post: [s, b, n] - expansion weights

        Returns:
            output: [s, b, n*C] - expanded tensor
        """
        n = self.n
        s, b, _ = h_post.shape

        if x.dim() == 1:
            # x is bias with shape [C], need to broadcast to [s, b, 1, C]
            C = x.shape[0]
            x_expanded = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(s, b, 1, C)
        else:
            # x is [s, b, C]
            C = x.shape[-1]
            x_expanded = x.unsqueeze(2)  # [s, b, 1, C]

        # h_post^T @ x : [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C]
        # Using broadcast multiply instead of einsum
        result = h_post.unsqueeze(-1) * x_expanded
        return result.view(s, b, n * C)

    @nvtx_decorator(message="HyperConnection::apply_h_post")
    def apply_h_post(
        self,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        h_post: Tensor,
        manager: Optional[CheckpointWithoutOutputManager] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post to x and optionally bias, with optional checkpointing.

        This is the unified entry point that handles both normal execution
        and checkpoint-based execution for memory efficiency.

        Args:
            x_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - hidden states
                - bias: [C] or None - optional bias tensor
            h_post: [s, b, n] - expansion weights
            manager: Optional CheckpointWithoutOutputManager for checkpoint management.
                When provided, wraps _apply_h_post with CheckpointWithoutOutput.

        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [s, b, n*C] - expanded hidden states
                - bias_out: [s, b, n*C] or None - expanded bias if input bias was not None
        """
        x, bias = x_with_bias

        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            # Checkpoint _apply_h_post to discard the output
            x_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._apply_h_post, x, h_post
            )

            # Checkpoint _apply_h_post for bias if not None
            if bias is not None:
                bias_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                    self._apply_h_post, bias, h_post
                )
            else:
                bias_out = None
        else:
            # Normal execution without checkpoint
            x_out = self._apply_h_post(x, h_post)
            bias_out = self._apply_h_post(bias, h_post) if bias is not None else None

        return x_out, bias_out

    def aggregate(self, x: Tensor, h_pre: Tensor) -> Tensor:
        """
        Aggregate n-stream to 1-stream.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights

        Returns:
            aggregated: [s, b, C] - single stream hidden states
        """
        s, b, _ = x.shape
        C = self.hidden_size
        x_streams = x.view(s, b, self.n, C)
        return self._h_aggregate_op(x_streams, h_pre)

    @torch.compile
    def apply_h_res(self, h_res: Tensor, residual: Tensor) -> Tensor:
        """
        Apply H_res to residual using H_res weights.

        Computes: H_res.T @ residual

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            residual: [s, b, n*C] - n-stream hidden states
        """
        s, b, _ = residual.shape
        n = self.n
        C = self.hidden_size

        # Reshape for bmm: [s, b, n, n] -> [s*b, n, n]
        h_res_batched = h_res.view(s * b, n, n)
        # [s, b, n*C] -> [s, b, n, C] -> [s*b, n, C]
        residual_batched = residual.view(s, b, n, C).view(s * b, n, C)

        # Batch matrix multiply: [s*b, n, n].T @ [s*b, n, C] -> [s*b, n, C]
        mixed = torch.bmm(h_res_batched.transpose(1, 2), residual_batched)

        return mixed.view(s, b, n * C)

    @component("mhc")
    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager: Optional[CheckpointWithoutOutputManager] = None,
        return_residual: bool = False,
        mhc_state: SinglePassMHCState | None = None,
    ) -> Tuple[Tensor, ...]:
        """
        Full mHC forward pass.

        Uses BroadcastTensorFused to split hidden_states into 3 autograd-graph
        children so that gradient accumulation from the 3 consumers
        (compute_mappings, aggregate, fused_h_res_h_post_bda) is handled by a
        single fused add instead of PyTorch's default sequential accumulation.
        The single-pass branch splits only with MCORE_MHC_SP_FUSED_GRAD_ADD on and
        ``return_residual=True``; 3-tuple callers feed their own unsplit residual to the
        BDA and keep autograd's accumulation.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            mhc_recompute_manager: Optional CheckpointWithoutOutputManager for checkpoint
                management.
                When provided, uses _forward_with_checkpoint for memory-efficient execution.

        Returns:
            The compatible 3-tuple ``(aggregated, h_res, h_post)`` by default.
            HybridModel callers set ``return_residual=True`` to also receive the
            residual branch created by ``BroadcastTensorFused``.
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            residual: [s, b, n*C] - residual view for fused_h_res_h_post_bda
        """
        if self.single_pass:
            if mhc_state is None:
                raise ValueError("Single-pass mHC requires a forward-local SinglePassMHCState")
            hs_for_mappings = hs_for_aggregate = hs_for_residual = hidden_states
            if (
                self._single_pass_grad_add_3_op is not None
                and return_residual
                and torch.is_grad_enabled()
                and hidden_states.requires_grad
            ):
                # One fused sum replaces autograd's two sequential full-size gradient adds. Only
                # callers that take the residual alias split: a 3-tuple caller passes its own
                # unsplit residual to the BDA, so the unused alias would get a zero-filled
                # gradient and the BDA gradient an extra add (neither exact nor faster).
                # Exact mode matches autograd only while these three are the input's only
                # consumers, each alias feeds a single node, and the gradients arrive as BDA,
                # aggregate, mappings (see _ordered_single_pass_grad_add_3). Another consumer
                # or a reordering changes bits silently; MCORE_MHC_SP_FUSED_GRAD_ADD=0 restores
                # autograd's own sum and is the first switch to flip when bisecting a mismatch.
                hs_for_mappings, hs_for_aggregate, hs_for_residual = BroadcastTensorFused.apply(
                    hidden_states, self._single_pass_grad_add_3_op
                )
            h_pre, h_post, h_res = self.compute_mappings(hs_for_mappings)
            if mhc_recompute_manager is None:
                aggregated = self._single_pass_aggregate(hs_for_aggregate, mhc_state.pre_mix)
            else:
                from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

                # Replay receives the incoming mix explicitly, never the mutable state.
                aggregated = CheckpointWithoutOutput(ckpt_manager=mhc_recompute_manager).checkpoint(
                    self._single_pass_aggregate, hs_for_aggregate, mhc_state.pre_mix
                )
            mhc_state.pre_mix = h_pre
            result = aggregated, h_res, h_post, hs_for_residual
        elif mhc_recompute_manager is not None:
            result = self._forward_with_checkpoint(hidden_states, mhc_recompute_manager)
        else:
            result = self._forward_normal(hidden_states)
        return result if return_residual else result[:3]

    def _single_pass_aggregate(self, hidden_states: Tensor, pre_mix: Tensor | None) -> Tensor:
        """Aggregate with an explicit predecessor, safe for checkpoint replay."""
        aggregate = self._h_aggregate_op if self.config.use_fused_mhc else None
        output = SinglePassMHCState(pre_mix).contract(hidden_states, self.n, aggregate=aggregate)
        # Entry selection can alias residual storage for one token or one stream.
        return output.clone() if pre_mix is None else output

    def _forward_normal(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Normal forward pass without checkpointing.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            residual: [s, b, n*C] - residual view for fused_h_res_h_post_bda
        """
        # Split into 3 views to avoid extra grad accumulations in backward
        hs_for_mappings, hs_for_aggregate, hs_for_residual = BroadcastTensorFused.apply(
            hidden_states, self._fused_add_3_op
        )

        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hs_for_mappings)

        # Aggregate for layer input
        with torch.cuda.nvtx.range("HyperConnection::aggregate"):
            aggregated = self.aggregate(hs_for_aggregate, h_pre)

        return aggregated, h_res, h_post, hs_for_residual

    def _forward_with_checkpoint(
        self, hidden_states: Tensor, manager: CheckpointWithoutOutputManager
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass with checkpointing for memory efficiency.

        compute_mappings is called directly (not checkpointed) since its outputs
        (h_pre, h_post, h_res) are needed downstream. Only aggregate is wrapped with
        CheckpointWithoutOutput and auto-registered to the manager.
        apply_h_res is deferred to fused_h_res_h_post_bda for kernel fusion.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            manager: CheckpointWithoutOutputManager for unified recomputation

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            residual: [s, b, n*C] - residual view for fused_h_res_h_post_bda
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        # Split into 3 views to avoid extra grad accumulations in backward
        hs_for_mappings, hs_for_aggregate, hs_for_residual = BroadcastTensorFused.apply(
            hidden_states, self._fused_add_3_op
        )

        h_pre, h_post, h_res = self.compute_mappings(hs_for_mappings)

        # Checkpoint aggregate - auto-registers to manager
        aggregated = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.aggregate, hs_for_aggregate, h_pre
        )

        return aggregated, h_res, h_post, hs_for_residual

    # ==================== Block-level utilities ====================

    @staticmethod
    @component("mhc")
    def input_expand(x: Tensor, n: int) -> Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.

        Simple replication strategy: each stream initialized as a copy of input.

        Args:
            x: [s, b, C] - single stream hidden states
            n: Number of residual streams

        Returns:
            expanded: [s, b, n*C] - n-stream hidden states
        """
        s, b, C = x.shape
        # Replicate input to n streams
        expanded = x.unsqueeze(2).expand(s, b, n, C).contiguous()
        return expanded.view(s, b, n * C)

    @staticmethod
    @component("mhc")
    def output_contract(x: Tensor, n: int) -> Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.

        Simple averaging strategy: average all streams.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            n: Number of residual streams

        Returns:
            contracted: [s, b, C] - single stream hidden states
        """
        s, b, nC = x.shape
        C = nC // n
        # Average all streams
        x_streams = x.view(s, b, n, C)
        contracted = x_streams.mean(dim=2)
        return contracted

    # ==================== Fused kernel placeholder ====================

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda")
    @component("mhc")
    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional[CheckpointWithoutOutputManager] = None,
    ) -> Tensor:
        """
        Fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.

        This is a placeholder for future kernel fusion optimization.
        Currently implements the operations sequentially using native PyTorch.

        The computation flow is:
            1. mixed = H_res.T @ original_residual (apply_h_res)
            2. expanded = H_post^T @ layer_output (apply_h_post)
            3. output = dropout(expanded + bias) + mixed (bias-dropout-add)

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states (before H_res applied)
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - layer output (attention or MLP output)
                - bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: Optional CheckpointWithoutOutputManager for checkpoint management.
                When provided, each operation is wrapped with CheckpointWithoutOutput.

        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        if self.single_pass:
            args = (
                h_res,
                original_residual,
                h_post,
                *layer_output_with_bias,
                dropout_prob,
                training,
            )
            if manager is None:
                return self._single_pass_bda(*args)
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            return CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._single_pass_bda, *args
            )
        if manager is not None:
            return self._fused_h_res_h_post_bda_with_checkpoint(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
                manager,
            )
        else:
            return self._fused_h_res_h_post_bda_native(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
            )

    def _single_pass_bda(
        self, h_res, original_residual, h_post, x, bias, dropout_prob, training
    ) -> Tensor:
        """Apply residual mixing and branch expansion once, preserving FP32 coefficients."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            streams = original_residual.unflatten(-1, (self.n, self.hidden_size))
            if self.config.use_fused_mhc and (not training or dropout_prob == 0.0):
                return self._h_post_bda_op(
                    h_res,
                    streams,
                    h_post,
                    # Bias gradients reduce across tokens; retain FP32 partials.
                    x if bias is None else x.float(),
                    None if bias is None else bias.float(),
                    output_dtype=x.dtype,
                ).flatten(-2)
            mixed = torch.einsum("sbij,sbic->sbjc", h_res.float(), streams.float())
            branch = x.float() if bias is None else x.float() + bias.float()
            expanded = h_post.float().unsqueeze(-1) * branch.unsqueeze(-2)
            expanded = F.dropout(expanded, p=dropout_prob, training=training)
            return (mixed + expanded).flatten(-2).to(x.dtype)

    def _fused_h_res_h_post_bda_native(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
    ) -> Tensor:
        """
        h_res, h_post and bda.

        When dropout is zero (or inference), uses a single fused/reference kernel
        for H_res.T @ residual + H_post * (x + bias). Falls back to unfused
        implementation when dropout is needed.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation

        Returns:
            output: [s, b, n*C] - final output
        """
        x, bias = layer_output_with_bias

        if dropout_prob == 0.0 or not training:
            s, b, _ = original_residual.shape
            n = self.n
            C = self.hidden_size
            orig_reshaped = original_residual.view(s, b, n, C)
            output = self._h_post_bda_op(h_res, orig_reshaped, h_post, x, bias)
            return output.view(s, b, n * C)

        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        with torch.cuda.nvtx.range("HyperConnection::apply_h_res"):
            mixed = self.apply_h_res(h_res, original_residual)
        with torch.cuda.nvtx.range("HyperConnection::apply_h_post"):
            x_expanded = self._apply_h_post(x, h_post)
            bias_expanded = self._apply_h_post(bias, h_post) if bias is not None else None
        bda_func = get_bias_dropout_add(training, fused)
        with torch.cuda.nvtx.range("HyperConnection::bda"):
            output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
        return output

    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda_with_checkpoint")
    def _fused_h_res_h_post_bda_with_checkpoint(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: CheckpointWithoutOutputManager,
    ) -> Tensor:
        """
        Checkpointed variant of _fused_h_res_h_post_bda_native.

        Wraps compute in CheckpointWithoutOutput for activation memory savings.
        Cannot reuse _native directly because checkpoint requires all args to be
        positional Tensors; tuple/Optional/scalar args are unpacked or captured
        via closure instead.

        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: CheckpointWithoutOutputManager for checkpoint management

        Returns:
            output: [s, b, n*C] - final output
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        x, bias = layer_output_with_bias
        n = self.n
        C = self.hidden_size

        # Fast path: no dropout — use fused/reference h_post_bda kernel (same as _native)
        if dropout_prob == 0.0 or not training:

            def _fused_wrapper(h_res, original_residual, h_post, x, *optional_bias):
                s, b, _ = original_residual.shape
                orig_reshaped = original_residual.view(s, b, n, C)
                b_arg = optional_bias[0] if optional_bias else None
                return self._h_post_bda_op(h_res, orig_reshaped, h_post, x, b_arg).view(s, b, n * C)

            ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
            if bias is not None:
                output = ckpt.checkpoint(_fused_wrapper, h_res, original_residual, h_post, x, bias)
            else:
                output = ckpt.checkpoint(_fused_wrapper, h_res, original_residual, h_post, x)

        # Slow path: dropout required — fused kernel does not support dropout,
        # fall back to sequential apply_h_res + apply_h_post + bda
        else:
            from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

            bda_func = get_bias_dropout_add(training, fused)
            has_bias = bias is not None

            def _native_wrapper(h_res, original_residual, h_post, x, *optional_bias):
                with torch.cuda.nvtx.range("HyperConnection::apply_h_res"):
                    mixed = self.apply_h_res(h_res, original_residual)
                with torch.cuda.nvtx.range("HyperConnection::apply_h_post"):
                    x_expanded = self._apply_h_post(x, h_post)
                    if has_bias:
                        bias_expanded = self._apply_h_post(optional_bias[0], h_post)
                    else:
                        bias_expanded = None
                with torch.cuda.nvtx.range("HyperConnection::bda"):
                    output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
                return output

            ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
            if has_bias:
                output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x, bias)
            else:
                output = ckpt.checkpoint(_native_wrapper, h_res, original_residual, h_post, x)

        return output
