# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator

_MHC_SINKHORN_EPS = 1e-6
_MHC_COMPUTE_H_EPS = 1e-6


def build_mhc_recompute_layer_plan(
    num_layers: int, mhc_recompute_layer_num: Optional[int], use_mhc_recompute: bool
) -> Tuple[list[Optional[CheckpointWithoutOutputManager]], list[bool]]:
    """Build per-layer mHC recompute managers and recompute-block end markers."""
    layer_managers: list[Optional[CheckpointWithoutOutputManager]] = [None] * num_layers
    is_recompute_block_end = [False] * num_layers

    if not use_mhc_recompute or num_layers == 0:
        return layer_managers, is_recompute_block_end

    mhc_manager = CheckpointWithoutOutputManager()
    for layer_index in range(num_layers):
        is_last_in_transformer_block = layer_index == num_layers - 1
        is_last_in_recompute_block = is_last_in_transformer_block
        if mhc_recompute_layer_num is not None:
            is_last_in_recompute_block = is_last_in_transformer_block or (
                (layer_index + 1) % mhc_recompute_layer_num == 0
            )

        layer_managers[layer_index] = mhc_manager
        is_recompute_block_end[layer_index] = is_last_in_recompute_block

        if is_last_in_recompute_block and not is_last_in_transformer_block:
            mhc_manager = CheckpointWithoutOutputManager()

    return layer_managers, is_recompute_block_end


def finalize_mhc_recompute_layer(
    mhc_manager: Optional[CheckpointWithoutOutputManager],
    hidden_states: Tensor,
    is_last_in_recompute_block: bool,
) -> None:
    """Finalize mHC recompute state when the current recompute block ends."""
    if mhc_manager is not None and is_last_in_recompute_block:
        mhc_manager.discard_all_outputs_and_register_unified_recompute(hidden_states)


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
        self.sinkhorn_eps = _MHC_SINKHORN_EPS
        self.compute_h_eps = _MHC_COMPUTE_H_EPS

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
        self.norm_eps = 1e-6

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

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager: Optional[CheckpointWithoutOutputManager] = None,
        return_residual: bool = False,
    ) -> Tuple[Tensor, ...]:
        """
        Full mHC forward pass.

        Uses BroadcastTensorFused to split hidden_states into 3 autograd-graph
        children so that gradient accumulation from the 3 consumers
        (compute_mappings, aggregate, fused_h_res_h_post_bda) is handled by a
        single fused add instead of PyTorch's default sequential accumulation.

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
        if mhc_recompute_manager is not None:
            result = self._forward_with_checkpoint(hidden_states, mhc_recompute_manager)
        else:
            result = self._forward_normal(hidden_states)
        return result if return_residual else result[:3]

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


# ============================================================================
# Gated Residual (Qwen4-Exp / Qwen3.8-Flash-Next hyper-connections)
# ============================================================================


def gated_residual_group_rmsnorm(x: Tensor, weight: Tensor, n: int, eps: float) -> Tensor:
    """Zero-centered RMSNorm applied independently to each of the ``n`` residual streams.

    Args:
        x: [..., n*C] n-stream hidden states.
        weight: [n*C] zero-centered gain (the effective scale is ``1 + weight``).
        n: Number of residual streams.
        eps: RMSNorm epsilon.

    Returns:
        [..., n*C] normalized streams in ``x.dtype`` (computed in fp32).
    """
    orig_dtype = x.dtype
    x_streams = x.float().unflatten(-1, (n, -1))
    normed = x_streams * torch.rsqrt(x_streams.pow(2).mean(-1, keepdim=True) + eps)
    normed = normed.flatten(-2) * (1.0 + weight.float())
    return normed.to(orig_dtype)


class GatedResidualHyperConnection(MegatronModule):
    """Gated Residual hyper-connection of Qwen4-Exp (Qwen3.8-Flash-Next).

    Every sub-layer (attention / MLP) of a decoder layer is wrapped by one instance. With ``n``
    residual streams of width ``C`` travelling between layers as ``[s, b, n*C]``:

        xn      = GroupRMSNorm(x)                                # per-stream, zero-centered gain
        mix     = sigmoid(W_up silu(W_down xn / n))              # [.., n*C] element-wise read gate
        block_in = mean_j(mix_j * xn_j)                          # [.., C]   sub-layer input
        inject  = 2 * sigmoid(W_inject xn / n)                   # [.., n]   per-stream write gate
        x_new   = x + inject_j * F(block_in)  for every stream j

    The interface matches :class:`HyperConnectionModule` so that
    :class:`~megatron.core.transformer.transformer_layer.HyperConnectionTransformerLayer` drives
    both formulations: ``forward`` returns ``(block_input, h_res, h_post)`` where ``h_res`` is
    ``None`` (the residual mixing is the identity) and ``h_post`` is ``inject``, and
    :meth:`fused_h_res_h_post_bda` applies the write gate. With ``use_combine=False`` the module
    is the block-level output mixer: it only returns ``block_in`` (see
    :class:`GatedResidualOutputMixer`).

    Parameters are replicated across tensor-parallel ranks and kept in ``params_dtype`` like the
    reference checkpoints; the norm, gates and stream reductions run in fp32.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = 1, use_combine: bool = True):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.mhc_num_residual_streams
        self.hidden_size = config.hidden_size
        self.rank = config.mhc_gated_residual_rank
        self.use_combine = use_combine
        self.norm_eps = config.layernorm_epsilon
        hc_hidden_size = self.n * self.hidden_size
        dtype = config.params_dtype

        self.hc_norm = nn.Module()
        self.hc_norm.weight = nn.Parameter(torch.zeros(hc_hidden_size, dtype=dtype))
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, self.rank, bias=False, dtype=dtype)
        self.input_mix_weight_up = nn.Linear(self.rank, hc_hidden_size, bias=False, dtype=dtype)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden_size, self.n, bias=False, dtype=dtype)
        else:
            self.block_inject_weight = None

        if self.config.perform_initialization:
            nn.init.normal_(self.input_mix_weight_down.weight, mean=0.0, std=config.init_method_std)
            nn.init.normal_(self.input_mix_weight_up.weight, mean=0.0, std=config.init_method_std)
            if self.block_inject_weight is not None:
                nn.init.normal_(
                    self.block_inject_weight.weight, mean=0.0, std=config.init_method_std
                )

        # Replicated (non tensor-parallel) parameters: their gradients must be all-reduced
        # across the TP group when sequence parallelism shards the tokens.
        if self.config.sequence_parallel:
            for param in self.parameters():
                setattr(param, 'sequence_parallel', True)

    def _mix(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, b, nC = hidden_states.shape
        n, C = self.n, self.hidden_size
        xn = gated_residual_group_rmsnorm(hidden_states, self.hc_norm.weight, n, self.norm_eps)
        down = F.silu(self.input_mix_weight_down(xn).float() / n)
        gate = torch.sigmoid(self.input_mix_weight_up(down.to(xn.dtype)).float())
        block_input = (gate.view(s, b, n, C) * xn.float().view(s, b, n, C)).mean(dim=2)
        block_input = block_input.to(hidden_states.dtype)
        if self.block_inject_weight is None:
            return block_input, None
        inject = 2.0 * torch.sigmoid(self.block_inject_weight(xn).float() / n)  # [s, b, n]
        return block_input, inject

    def forward(
        self, hidden_states: Tensor, mhc_recompute_manager=None, return_residual: bool = False
    ):
        """Read the streams.

        Args:
            hidden_states: [s, b, n*C] n-stream hidden states.
            mhc_recompute_manager: Accepted for interface compatibility; not used.
            return_residual: Also return the (unchanged) n-stream residual as a 4th element,
                like :class:`HyperConnectionModule`.

        Returns:
            ``(block_input [s, b, C], h_res=None, h_post [s, b, n])``. ``h_post`` is ``None``
            for the output mixer (``use_combine=False``).
        """
        block_input, inject = self._mix(hidden_states)
        if return_residual:
            return block_input, None, inject, hidden_states
        return block_input, None, inject

    def apply_h_res(self, h_res: Optional[Tensor], residual: Tensor) -> Tensor:
        """The gated residual keeps the streams untouched (identity residual mixing)."""
        return residual

    @staticmethod
    def _inject(
        residual: Tensor, inject: Tensor, x: Tensor, bias: Optional[Tensor], n: int
    ) -> Tensor:
        s, b, nC = residual.shape
        out = x.float()
        if bias is not None:
            out = out + bias.float()
        update = inject.unsqueeze(-1) * out.unsqueeze(2)  # [s, b, n, C]
        return (residual.float().view(s, b, n, -1) + update).view(s, b, nC).to(residual.dtype)

    @nvtx_decorator(message="GatedResidual::fused_h_res_h_post_bda")
    def fused_h_res_h_post_bda(
        self,
        h_res: Optional[Tensor],
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager=None,
    ) -> Tensor:
        """Write the sub-layer output back to every stream: ``x + inject_j * (F(x) + bias)``.

        Args:
            h_res: Unused (identity residual mixing).
            original_residual: [s, b, n*C] n-stream hidden states before the sub-layer.
            h_post: [s, b, n] per-stream write gates returned by :meth:`forward`.
            layer_output_with_bias: ``(x [s, b, C], bias [C] or None)``.
            dropout_prob / training / fused: bias-dropout-add settings; dropout is applied to the
                broadcast update when non-zero.
            manager: Accepted for interface compatibility; not used.

        Returns:
            [s, b, n*C] updated streams.
        """
        x, bias = layer_output_with_bias
        if dropout_prob == 0.0 or not training:
            return self._inject(original_residual, h_post, x, bias, self.n)
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        s, b, nC = original_residual.shape
        x_expanded = (h_post.unsqueeze(-1) * x.float().unsqueeze(2)).view(s, b, nC).to(x.dtype)
        bias_expanded = (
            (h_post.unsqueeze(-1) * bias.float().view(1, 1, 1, -1)).view(s, b, nC).to(x.dtype)
            if bias is not None
            else None
        )
        bda_func = get_bias_dropout_add(training, fused)
        return bda_func((x_expanded, bias_expanded), original_residual, dropout_prob)


class GatedResidualOutputMixer(GatedResidualHyperConnection):
    """Block-level learned contraction of the Qwen4-Exp residual streams.

    Replaces the final layer norm of a
    :class:`~megatron.core.transformer.transformer_block.TransformerBlock`
    built with ``enable_mhc_connections`` and ``mhc_variant="gated_residual"``: it consumes the
    ``[s, b, n*C]`` streams and returns the ``[s, b, C]`` gated mean of their normalized values
    (``hyper_connection_mixer`` in the HF checkpoint). The block skips its unweighted
    ``output_contract`` because ``contracts_mhc_streams`` is set.

    The ``(config, hidden_size, eps)`` signature matches the layer-norm builders used in block
    specs.
    """

    contracts_mhc_streams: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: Optional[int] = None,
        eps: Optional[float] = None,
    ):
        super().__init__(config, layer_number=0, use_combine=False)
        if hidden_size is not None:
            assert (
                hidden_size == config.hidden_size
            ), "GatedResidualOutputMixer mixes full hidden streams."
        if eps is not None:
            self.norm_eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:  # type: ignore[override]
        """[s, b, n*C] streams -> [s, b, C] mixed output."""
        block_input, _ = self._mix(hidden_states)
        return block_input
