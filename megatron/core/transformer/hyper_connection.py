# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointManager


@torch.compile
def _sinkhorn_iterations(input_logits: Tensor, num_iterations: int, eps: float) -> Tensor:
    # Stabilization strategy aligned with the cuTile fused kernel
    # (`_ct_sinkhorn_fwd_kernel` uses `row_sum + eps`). Both paths therefore
    # produce bit-similar results for well-conditioned inputs, and any future
    # divergence at near-zero sums is bounded by the same `eps` regularization.
    output_dtype = input_logits.dtype
    input_logits_fp32 = input_logits.float()
    row_max = input_logits_fp32.max(dim=-1, keepdim=True).values
    M = torch.exp(input_logits_fp32 - row_max)
    for _ in range(num_iterations):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M.to(output_dtype)


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
        """Recompute forward under enable_grad for memory-efficient backward."""
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
    """Native H_res @ residual + H_post * (x [+ bias])."""
    s, b, n, C = original_residual.shape
    h_res_batched = h_res.view(s * b, n, n)
    residual_batched = original_residual.view(s * b, n, C)
    mixed = torch.bmm(h_res_batched, residual_batched).view(s, b, n, C)
    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(2)
    if bias is not None:
        bias_expanded = h_post.unsqueeze(-1) * bias.view(1, 1, 1, C)
        return x_expanded + bias_expanded + mixed
    return x_expanded + mixed


@torch.compile
def native_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Native fused projection + RMS normalization."""
    input_dtype = x.dtype
    x_float = x.float()
    weight_float = weight.float()
    proj = torch.matmul(x_float, weight_float.t()).to(dtype=input_dtype)
    norm = x_float.norm(dim=-1, keepdim=True)
    K = x.shape[-1]
    v = norm / math.sqrt(K) + eps
    r = (1.0 / v).to(dtype=input_dtype)
    return proj, r


# ============================================================================
# HyperConnectionModule
# ============================================================================


# TODO: keep hyper connection in fp32 computation
class HyperConnectionModule(MegatronModule):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.

    Implements the complete mHC propagation:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream → 1-stream (H_pre @ x)
    3. Expansion: 1-stream → n-stream (H_post^T @ output)
    4. Residual merge: H_res @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries

    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations

        # Stream-mapping projection that produces the per-token H_pre, H_post, and
        # H_res logits used by the hyper connection. Note the strong asymmetry: the
        # input is wide (n * hidden_size, e.g. 16384 for n=4 / C=4096) but the
        # output is tiny (n^2 + 2n, e.g. 24 for n=4). The output slices are:
        #   - first  n        : H_pre  logits (aggregation weights)
        #   - next   n        : H_post logits (expansion weights)
        #   - last   n*n      : H_res  logits (residual mixing, fed into Sinkhorn)
        # Kept named `mapping_proj` to preserve checkpoint state_dict keys.
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
        self.norm_eps = 1e-6

        # Choose implementation: fused cuTile kernels vs reference modules.
        # Both paths expose the same call signatures so the rest of the code
        # is implementation-agnostic.
        if config.use_fused_mhc:
            from megatron.core.fusions.fused_mhc_kernels import (
                fused_h_aggregate,
                fused_h_post_bda,
                fused_proj_rms,
                fused_sinkhorn,
            )

            self._sinkhorn_op = fused_sinkhorn
            self._h_aggregate_op = fused_h_aggregate
            self._h_post_bda_op = fused_h_post_bda
            self._proj_rms_op = fused_proj_rms
        else:
            self._sinkhorn_op = native_sinkhorn
            self._h_aggregate_op = native_h_aggregate
            self._h_post_bda_op = native_h_post_bda
            self._proj_rms_op = native_proj_rms

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.mapping_proj.weight)

        # Set sequence_parallel attribute on parameters for gradient synchronization
        # across TP ranks when sequence_parallel is enabled.
        # This is required because HyperConnectionModule uses non-TP-aware layers
        # (nn.Linear, nn.RMSNorm) whose gradients need to be all-reduced.
        if self.config.sequence_parallel:
            self.mapping_proj.weight.sequence_parallel = True
            self.alpha_pre.sequence_parallel = True
            self.alpha_post.sequence_parallel = True
            self.alpha_res.sequence_parallel = True
            self.bias.sequence_parallel = True

    def _projection_and_get_norm(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Projection + RMS normalization.

        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        x_2d = x.reshape(s * b, nC)
        proj, r = self._proj_rms_op(x_2d, self.mapping_proj.weight, self.norm_eps)
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
        # `alpha_` is rebuilt each call from three learnable scalars rather than
        # cached as a derived buffer because @torch.compile fuses the expand+cat
        # into the surrounding fused multiply, leaving no measurable overhead in
        # the compiled graph. The scalar `alpha_*` parameters remain the source of
        # truth for optimizer/state_dict.
        alpha_ = torch.cat(
            [
                self.alpha_pre.expand(self.n),
                self.alpha_post.expand(self.n),
                self.alpha_res.expand(self.n * self.n),
            ],
            dim=-1,
        )
        h = r.float() * proj.float() * alpha_.float() + self.bias.float()
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., : self.n].sigmoid().to(dtype=proj.dtype)  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = (h[..., self.n : 2 * self.n].sigmoid() * 2).to(dtype=proj.dtype)  # [s, b, n]
        h_res = h[..., 2 * self.n :].to(dtype=proj.dtype)
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
        with torch.cuda.nvtx.range("HyperConnection::projection_and_get_norm"):
            proj, r = self._projection_and_get_norm(x)
        with torch.cuda.nvtx.range("HyperConnection::compute_h"):
            h_pre, h_post, h_res = self._compute_h(proj, r)
        h_res = self._sinkhorn_op(
            h_res.view(s, b, self.n, self.n), self.sinkhorn_iterations, self.norm_eps
        )  # [s, b, n, n]

        return h_pre, h_post, h_res

    @torch.compile
    def _apply_h_post_hidden(self, x: Tensor, h_post: Tensor) -> Tensor:
        """
        Apply H_post to hidden states.

        Computes: H_post^T @ x

        Args:
            x: [s, b, C] - standard hidden states
            h_post: [s, b, n] - expansion weights

        Returns:
            output: [s, b, n*C] - expanded tensor
        """
        n = self.n
        s, b, _ = h_post.shape
        C = x.shape[-1]
        x_expanded = x.unsqueeze(2)  # [s, b, 1, C]

        # h_post^T @ x : [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C]
        result = h_post.unsqueeze(-1) * x_expanded
        return result.view(s, b, n * C)

    @torch.compile
    def _apply_h_post_bias(self, bias: Tensor, h_post: Tensor) -> Tensor:
        """
        Apply H_post to a bias vector.

        Args:
            bias: [C] - bias tensor broadcast across sequence and batch
            h_post: [s, b, n] - expansion weights

        Returns:
            output: [s, b, n*C] - expanded bias
        """
        n = self.n
        s, b, _ = h_post.shape
        C = bias.shape[0]
        bias_expanded = bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(s, b, 1, C)

        # h_post^T @ x : [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C]
        result = h_post.unsqueeze(-1) * bias_expanded
        return result.view(s, b, n * C)

    @nvtx_decorator(message="HyperConnection::apply_h_post")
    def apply_h_post(
        self,
        x_with_bias: Tuple[Tensor, Optional[Tensor]],
        h_post: Tensor,
        manager: Optional['CheckpointManager'] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post to x and optionally bias, with optional checkpointing.

        This is the unified entry point that handles both normal execution
        and checkpoint-based execution for memory efficiency. The TransformerLayer
        path currently uses h_res_h_post_bda directly, but this helper is
        kept for callers that need standalone H_post expansion.

        Args:
            x_with_bias: Tuple of (x, bias) where:
                - x: [s, b, C] - hidden states
                - bias: [C] or None - optional bias tensor
            h_post: [s, b, n] - expansion weights
            manager: Optional CheckpointManager for checkpoint management.
                When provided, wraps _apply_h_post with CheckpointWithoutOutput.

        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [s, b, n*C] - expanded hidden states
                - bias_out: [s, b, n*C] or None - expanded bias if input bias was not None
        """
        x, bias = x_with_bias

        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

            # Checkpoint H_post application to discard the output
            x_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._apply_h_post_hidden, x, h_post
            )

            # Checkpoint H_post bias expansion if not None
            if bias is not None:
                bias_out = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                    self._apply_h_post_bias, bias, h_post
                )
            else:
                bias_out = None
        else:
            # Normal execution without checkpoint
            x_out = self._apply_h_post_hidden(x, h_post)
            bias_out = self._apply_h_post_bias(bias, h_post) if bias is not None else None

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

        Computes: H_res @ residual

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

        # Batch matrix multiply: [s*b, n, n] @ [s*b, n, C] -> [s*b, n, C]
        mixed = torch.bmm(h_res_batched, residual_batched)

        return mixed.view(s, b, n * C)

    def forward(
        self, hidden_states: Tensor, mhc_recompute_manager: Optional['CheckpointManager'] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            mhc_recompute_manager: Optional CheckpointManager for checkpoint management.
                When provided, uses _forward_with_checkpoint for memory-efficient execution.

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        if mhc_recompute_manager is not None:
            return self._forward_with_checkpoint(hidden_states, mhc_recompute_manager)
        else:
            return self._forward_normal(hidden_states)

    def _forward_normal(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normal forward pass without checkpointing.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)

        # Aggregate for layer input
        with torch.cuda.nvtx.range("HyperConnection::aggregate"):
            aggregated = self.aggregate(hidden_states, h_pre)

        return aggregated, h_res, h_post

    def _forward_with_checkpoint(
        self, hidden_states: Tensor, manager: 'CheckpointManager'
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with checkpointing for memory efficiency.

        compute_mappings is called directly (not checkpointed) since its outputs
        (h_pre, h_post, h_res) are needed downstream. Only aggregate is wrapped with
        CheckpointWithoutOutput and auto-registered to the manager.
        apply_h_res is deferred to h_res_h_post_bda for kernel fusion.

        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            manager: CheckpointManager for unified recomputation

        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

        h_pre, h_post, h_res = self.compute_mappings(hidden_states)

        # Checkpoint aggregate - auto-registers to manager
        aggregated = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.aggregate, hidden_states, h_pre
        )

        return aggregated, h_res, h_post

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
        assert nC % n == 0, (
            f"output_contract: n-stream input dim {nC} is not a multiple of "
            f"num_residual_streams={n}"
        )
        C = nC // n
        # Average all streams
        x_streams = x.view(s, b, n, C)
        contracted = x_streams.mean(dim=2)
        return contracted

    # ==================== Combined H_res + H_post + BDA path ====================

    @nvtx_decorator(message="HyperConnection::h_res_h_post_bda")
    def h_res_h_post_bda(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional['CheckpointManager'] = None,
    ) -> Tensor:
        """
        Combine apply_h_res, apply_h_post and bias-dropout-add.

        This is a reference implementation that uses native PyTorch for the
        dropout path. Actual fused kernels are selected through _h_post_bda_op
        when dropout is disabled or training is off.

        The computation flow is:
            1. mixed = H_res @ original_residual (apply_h_res)
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
            manager: Optional CheckpointManager for checkpoint management.
                When provided, each operation is wrapped with CheckpointWithoutOutput.

        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        if manager is not None:
            return self._h_res_h_post_bda_with_checkpoint(
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
            return self._h_res_h_post_bda_native(
                h_res,
                original_residual,
                h_post,
                layer_output_with_bias,
                dropout_prob,
                training,
                fused,
            )

    def _h_res_h_post_bda_native(
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
        for H_res @ residual + H_post * (x + bias). Falls back to unfused
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
            x_expanded = self._apply_h_post_hidden(x, h_post)
            bias_expanded = self._apply_h_post_bias(bias, h_post) if bias is not None else None
        bda_func = get_bias_dropout_add(training, fused)
        with torch.cuda.nvtx.range("HyperConnection::bda"):
            output = bda_func((x_expanded, bias_expanded), mixed, dropout_prob)
        return output

    @nvtx_decorator(message="HyperConnection::h_res_h_post_bda_with_checkpoint")
    def _h_res_h_post_bda_with_checkpoint(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: 'CheckpointManager',
    ) -> Tensor:
        """
        Checkpointed variant of _h_res_h_post_bda_native.

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
            manager: CheckpointManager for checkpoint management

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
                    x_expanded = self._apply_h_post_hidden(x, h_post)
                    if has_bias:
                        bias_expanded = self._apply_h_post_bias(optional_bias[0], h_post)
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
