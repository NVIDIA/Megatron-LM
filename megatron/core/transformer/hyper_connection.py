from typing import Tuple, Optional, TYPE_CHECKING
import math
import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_decorator, nvtx_range_push, nvtx_range_pop

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import MHCBlockRecomputeManager


#TODO: kernel fusion
class SinkhornKnopp(torch.autograd.Function):
    """
    Differentiable Sinkhorn-Knopp algorithm for doubly stochastic projection.
    
    Projects a positive matrix onto the Birkhoff polytope (doubly stochastic matrices)
    via iterative row and column normalization.
    
    Reference: Eq. (9) in mHC paper - M^{(t)} = T_c(T_r(M^{(t-1)}))
    """
    
    @staticmethod
    def forward(ctx, H_res_logits: Tensor, num_iterations: int) -> Tensor:
        """
        Project to doubly stochastic matrix via iterative row/col normalization.
        
        Args:
            H_res_logits: [s, b, n, n] - raw logits for residual mixing matrix
            num_iterations: Number of Sinkhorn iterations (paper uses 20)
        
        Returns:
            H_res: [s, b, n, n] - doubly stochastic matrix
        """
        # Use no_grad to avoid creating unnecessary computation graph in forward.
        # Gradients are computed explicitly in backward via recomputation.
            # M^{(0)} = exp(H_res_logits) - save initial M for backward recomputation
        M_init = torch.exp(H_res_logits)
        M = M_init.clone()
            
        with torch.no_grad():
            for _ in range(num_iterations):
                # T_r: Row normalization
                M = M / M.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                # T_c: Column normalization
                M = M / M.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        
        # Save initial M for backward recomputation
        ctx.save_for_backward(M_init)
        ctx.num_iterations = num_iterations
        return M
    
    @staticmethod  
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        """
        Backward through Sinkhorn-Knopp iterations using recomputation.
        
        Recomputes the forward pass with gradient tracking to obtain accurate gradients.
        """
        M_init, = ctx.saved_tensors
        num_iterations = ctx.num_iterations
        
        # Recompute forward with autograd enabled
        with torch.enable_grad():
            # Leaf for recomputation
            M_input = M_init.detach().requires_grad_(True)

            M_current = M_input
            for _ in range(num_iterations):
                # T_r: Row normalization
                M_current = M_current / M_current.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                # T_c: Column normalization
                M_current = M_current / M_current.sum(dim=-2, keepdim=True).clamp(min=1e-8)

            # Compute dL/dM_input (i.e., dL/dM_init) via autograd
            grad_M_init, = torch.autograd.grad(
                outputs=M_current,
                inputs=M_input,
                grad_outputs=grad_output,
                create_graph=False,   # typically what you want here
                retain_graph=False,
            )
        # Apply chain rule: dL/dH = dL/dM_init * dM_init/dH = dL/dM_init * M_init
        # Since M_init = exp(H_res_logits), we have d(exp(x))/dx = exp(x) = M_init
        grad_input = grad_M_init * M_init
        
        return grad_input, None


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
        
        # Projection weights for dynamic mappings
        # Input: [s, b, n*C] -> Output: n^2 + 2n values per token
        # - H_pre: n values
        # - H_post: n values  
        # - H_res: n^2 values (before Sinkhorn projection)
        self.norm = nn.RMSNorm(self.hidden_size * self.n)
        
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size, 
            self.n * self.n + 2 * self.n,
            bias=False
        )
        
        init_alpha = config.mhc_init_gating_factor
        # Learnable scaling factors (Eq. 5 in paper)
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))
        
        # Static bias terms
        self.bias = nn.Parameter(torch.zeros(self.n * self.n + 2 * self.n))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.mapping_proj.weight)
        
        # Set sequence_parallel attribute on parameters for gradient synchronization
        # across TP ranks when sequence_parallel is enabled.
        # This is required because HyperConnectionModule uses non-TP-aware layers
        # (nn.Linear, nn.RMSNorm) whose gradients need to be all-reduced.
        if self.config.sequence_parallel:
            setattr(self.mapping_proj.weight, 'sequence_parallel', True)
            setattr(self.norm.weight, 'sequence_parallel', True)
            setattr(self.alpha_pre, 'sequence_parallel', True)
            setattr(self.alpha_post, 'sequence_parallel', True)
            setattr(self.alpha_res, 'sequence_parallel', True)
            setattr(self.bias, 'sequence_parallel', True)
    

    # TODO: Kernel fusion
    @torch.compile
    @nvtx_decorator(message="HyperConnection::projection_and_rms")
    def _projection_and_rms(self, x : Tensor) -> Tuple[Tensor, Tensor]:
        """
        Project input hidden states to mapping space and apply RMS normalization.
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
        """
        s, b, nC = x.shape
        n = self.n
        r = x.norm(dim=-1, keepdim=True) / math.sqrt(nC) # shape: [s, b, 1]
        r = 1.0 / (r + 1e-8) # shape: [s, b, 1]
        proj = self.mapping_proj(x)  # [s, b, n^2 + 2n]
        return proj, r
    
    #TODO: kernel fusion
    @torch.compile
    @nvtx_decorator(message="HyperConnection::compute_h")
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
        s, b, _ = proj.shape
        alpha_ = torch.cat([self.alpha_pre.expand(self.n), self.alpha_post.expand(self.n), self.alpha_res.expand(self.n * self.n)], dim = -1)
        h = r * proj * alpha_ + self.bias
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., :self.n].sigmoid()  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n:2*self.n].sigmoid() * 2 # [s, b, n]
        h_res = h[..., 2*self.n:]
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
        proj, r = self._projection_and_rms(x)
        h_pre, h_post, h_res = self._compute_h(proj, r)
        h_res = SinkhornKnopp.apply(h_res.view(s, b, self.n, self.n), self.sinkhorn_iterations) # [s, b, n, n] 
        
        return h_pre, h_post, h_res
    
    @torch.compile
    @nvtx_decorator(message="HyperConnection::apply_h_post_inner")
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
        manager: Optional['MHCBlockRecomputeManager'] = None,
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
            manager: Optional MHCBlockRecomputeManager for checkpoint management.
                When provided, wraps _apply_h_post with CheckpointWithoutOutput.
        
        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [s, b, n*C] - expanded hidden states
                - bias_out: [s, b, n*C] or None - expanded bias if input bias was not None
        """
        x, bias = x_with_bias
        
        if manager is not None:
            from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
            
            # Checkpoint _apply_h_post for x
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

    
    #TODO: Kernel fusion
    @torch.compile
    @nvtx_decorator(message="HyperConnection::aggregate")
    def aggregate(self, x: Tensor, h_pre: Tensor) -> Tensor:
        """
        Aggregate n-stream to 1-stream using H_pre weights.
        
        Computes: sum_i(h_pre_i * x_stream_i)
        
        Args:
            x: [s, b, n*C] - n-stream hidden states
            h_pre: [s, b, n] - aggregation weights
        
        Returns:
            aggregated: [s, b, C] - single stream hidden states
        """
        s, b, _ = x.shape
        C = self.hidden_size
        
        # Reshape to [s, b, n, C]
        x_streams = x.view(s, b, self.n, C)
        
        # Weighted sum: [s, b, n, C] * [s, b, n, 1] -> sum over n -> [s, b, C]
        aggregated = (x_streams * h_pre.unsqueeze(-1)).sum(dim=2)
        
        return aggregated

    @torch.compile
    @nvtx_decorator(message="HyperConnection::apply_h_res")
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
        self,
        hidden_states: Tensor,
        residual: Tensor, 
        training: bool = True,
        mhc_recompute_manager: Optional['MHCBlockRecomputeManager'] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
            training: Whether in training mode
            mhc_recompute_manager: Optional MHCBlockRecomputeManager for checkpoint management.
                When provided, uses _forward_with_checkpoint for memory-efficient execution.
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
            
        """
        if mhc_recompute_manager is not None:
            return self._forward_with_checkpoint(
                hidden_states, residual, mhc_recompute_manager
            )
        else:
            return self._forward_normal(hidden_states, residual)
    
    def _forward_normal(
        self, hidden_states: Tensor, residual: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Normal forward pass without checkpointing.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)
        
        # Aggregate for layer input
        aggregated = self.aggregate(hidden_states, h_pre)

        return aggregated, h_res, h_post
    
    def _forward_with_checkpoint(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        manager: 'MHCBlockRecomputeManager',
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with checkpointing for memory efficiency.
        
        Operations (compute_mappings, aggregate) are wrapped with
        CheckpointWithoutOutput and auto-registered to the manager.
        apply_h_res is deferred to fused_h_res_h_post_bda for kernel fusion.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
            manager: MHCBlockRecomputeManager for unified recomputation
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            h_res: [s, b, n, n] - residual mixing matrix (for fused kernel)
            h_post: [s, b, n] - expansion weights
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        
        nvtx_range_push("HyperConnection::compute_mappings")
        # Checkpoint compute_mappings - auto-registers to manager via ckpt_manager parameter
        h_pre, h_post, h_res = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.compute_mappings, hidden_states
        )
        nvtx_range_pop("HyperConnection::compute_mappings")
        # Checkpoint aggregate - auto-registers to manager
        nvtx_range_push("HyperConnection::aggregate")
        aggregated = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.aggregate, hidden_states, h_pre
        )
        nvtx_range_pop("HyperConnection::aggregate")
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
        manager: Optional['MHCBlockRecomputeManager'] = None,
    ) -> Tensor:
        """
        Fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.
        
        This is a placeholder for future kernel fusion optimization.
        Currently implements the operations sequentially using native PyTorch.
        
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
            manager: Optional MHCBlockRecomputeManager for checkpoint management.
                When provided, each operation is wrapped with CheckpointWithoutOutput.
        
        Returns:
            output: [s, b, n*C] - final output after all operations
        """
        if manager is not None:
            return self._fused_h_res_h_post_bda_with_checkpoint(
                h_res, original_residual, h_post, layer_output_with_bias,
                dropout_prob, training, fused, manager
            )
        else:
            return self._fused_h_res_h_post_bda_native(
                h_res, original_residual, h_post, layer_output_with_bias,
                dropout_prob, training, fused
            )
    
    @nvtx_decorator(message="HyperConnection::fused_h_res_h_post_bda_native")
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
        Native implementation of fused h_res, h_post and bda operations.
        
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
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
        
        # Step 1: Apply H_res to original residual
        mixed = self.apply_h_res(h_res, original_residual)
        
        # Step 2: Apply H_post to layer output
        x, bias = layer_output_with_bias
        x_expanded = self._apply_h_post(x, h_post)
        bias_expanded = self._apply_h_post(bias, h_post) if bias is not None else None
        
        # Step 3: Bias-dropout-add
        bda_func = get_bias_dropout_add(training, fused)
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
        manager: 'MHCBlockRecomputeManager',
    ) -> Tensor:
        """
        Checkpointed implementation of fused h_res, h_post and bda operations.
        
        Each operation is wrapped with CheckpointWithoutOutput for memory efficiency.
        
        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            original_residual: [s, b, n*C] - n-stream hidden states
            h_post: [s, b, n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias)
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation
            manager: MHCBlockRecomputeManager for checkpoint management
        
        Returns:
            output: [s, b, n*C] - final output
        """
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
        
        # Step 1: Checkpoint apply_h_res
        nvtx_range_push("HyperConnection::apply_h_res")
        mixed = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self.apply_h_res, h_res, original_residual
        )
        nvtx_range_pop("HyperConnection::apply_h_res")
        # Step 2: Checkpoint apply_h_post for x
        x, bias = layer_output_with_bias
        nvtx_range_push("HyperConnection::apply_h_post")
        x_expanded = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            self._apply_h_post, x, h_post
        )   
        nvtx_range_pop("HyperConnection::apply_h_post")
        # Checkpoint apply_h_post for bias if not None
        if bias is not None:
            bias_expanded = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
                self._apply_h_post, bias, h_post
            )
        else:
            bias_expanded = None
        
        # Step 3: Checkpoint bias-dropout-add
        # Get the underlying BDA function
        if fused:
            from megatron.core.fusions.fused_bias_dropout import (
                bias_dropout_add_fused_train,
                bias_dropout_add_fused_inference,
            )
            if training:
                bda_func = bias_dropout_add_fused_train
            else:
                bda_func = bias_dropout_add_fused_inference
        else:
            from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_unfused
            bda_func = bias_dropout_add_unfused(training)
        
        # Wrapper function that re-packs the tuple for the actual BDA function
        def _bda_wrapper(output, bias, res, dropout):
            return bda_func((output, bias), res, dropout)
        
        ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
        nvtx_range_push("HyperConnection::bda_wrapper")
        output = ckpt.checkpoint(_bda_wrapper, x_expanded, bias_expanded, mixed, dropout_prob)
        nvtx_range_pop("HyperConnection::bda_wrapper")
        
        return output


# ==================== Checkpoint utilities for mHC ====================

class HyperConnectionCheckpoint:
    """
    Checkpoint utility for mHC intermediate activations.
    
    Implements the paper's "recomputing strategy" to reduce memory footprint
    by discarding intermediate n-stream activations and recomputing on-the-fly.
    """
    
    @staticmethod
    def compute_optimal_block_size(num_layers: int, num_streams: int) -> int:
        """
        Compute optimal recomputation block size.
        
        From paper Eq. (20): L_r^* ≈ sqrt(nL/(n+2))
        
        Args:
            num_layers: Total number of transformer layers
            num_streams: Number of residual streams (n)
        
        Returns:
            block_size: Optimal block size for checkpointing
        """
        block_size = int(math.sqrt(num_streams * num_layers / (num_streams + 2)))
        return max(1, block_size)
