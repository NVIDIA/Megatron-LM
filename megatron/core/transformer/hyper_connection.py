from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


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
        # M^{(0)} = exp(H_res_logits) - save initial M for backward recomputation
        M_init = torch.exp(H_res_logits)
        M = M_init.clone()
        
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
        s, b, nC = x.shape
        n = self.n 
        # Todo: kernel fusion
        # x_norm = self.norm(x)  # [s, b, n*C]
        r = x.norm(dim=-1, keepdim=True) / math.sqrt(nC) # shape: [s, b, 1]

        # Project to mapping space
        proj = self.mapping_proj(x)  # [s, b, n^2 + 2n]
        
        alpha_ = torch.cat([self.alpha_pre.expand(n), self.alpha_post.expand(n), self.alpha_res.expand(n * n)], dim = -1)

        h = r * proj * alpha_ + self.bias

        # Split projections
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., :self.n].sigmoid()  # [s, b, n]

        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n:2*self.n].sigmoid() * 2 # [s, b, n]
        
        # H_res = Sinkhorn-Knopp(exp(α_res * (θ_res @ x̃) + b_res))
        h_res = SinkhornKnopp.apply(h[..., 2*self.n:].view(s, b, self.n, self.n), self.sinkhorn_iterations) # [s, b, n, n] 

        # h_res = torch.dia
        
        return h_pre, h_post, h_res
    
    def apply_h_post(self, x_with_bias: Tuple[Tensor, Optional[Tensor]], h_post: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply H_post to x using H_post weights.
        
        Computes: H_post^T @ x_with_bias
        
        Here x_with_bias is a tuple of (x, bias), with shape
        [s, b, C] and [C], respectively.
        
        h_post: [s, b, n] - expansion weights

        Return the tuple of (x, bias) after applying H_post.
        """
        x, bias = x_with_bias
        s, b, C = x.shape 
        n = self.n

        # x : h_post^T @ x : [s, b, n, 1] @ [s, b, 1, C] -> [s, b, n, C] 
        x = torch.einsum('sbij,sbjc->sbic', h_post.unsqueeze(-1), x.unsqueeze(2)) # [s, b, n, C]
        x = x.view(s, b, n * C)

        if bias is not None : 
            assert bias.shape == (C,), "Bias must be of shape [C]"
            # bias : h_post^T @ bias : [s, b, n, 1] @ [s, b, 1, C] -> [s, b, n, C]
            bias = torch.einsum('sbij,sbjc->sbic', h_post.unsqueeze(-1), bias.unsqueeze(0).expand(s, b, -1, C)) # [s, b, n, C]
            bias = bias.view(s, b, n * C)
            return x, bias
        else:
            return x, None

    
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

    def apply_h_res(self, h_res: Tensor, residual: Tensor) -> Tensor:
        """
        Apply H_res to residual using H_res weights.
        
        Computes: H_res @ residual
        
        Args:
            h_res: [s, b, n, n] - residual mixing matrix
            residual: [s, b, n*C] - n-stream hidden states
        """
        s, b, _ = residual.shape 
        C = self.hidden_size
        residual_streams = residual.view(s, b, self.n, C)
        mixed = torch.einsum('sbij,sbjc->sbic', h_res, residual_streams) # [s, b, n, C]
        return mixed.view(s, b, self.n * C)
    
    def expand(self, layer_output: Tensor, h_post: Tensor) -> Tensor:
        """
        Expand 1-stream layer output to n-stream using H_post weights.
        
        Computes: H_post^T @ layer_output (broadcast to n streams)
        
        Args:
            layer_output: [s, b, C] - output from attention/MLP
            h_post: [s, b, n] - expansion weights
        
        Returns:
            expanded: [s, b, n*C] - n-stream expanded output
        """
        s, b, C = layer_output.shape
        
        # Expand: [s, b, C] * [s, b, n, 1] -> [s, b, n, C] -> [s, b, n*C]
        expanded = layer_output.unsqueeze(2) * h_post.unsqueeze(-1)  # [s, b, n, C]
        expanded = expanded.view(s, b, self.n * C)
        
        return expanded
    
    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor, 
        training: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.
        
        Args:
            hidden_states: [s, b, n*C] - n-stream hidden states
            residual: [s, b, n*C] - n-stream hidden states (x_l)
            dropout_p: Dropout probability for residual merge
            training: Whether in training mode
        
        Returns:
            aggregated: [s, b, C] - aggregated input for layer computation
            mixed: [s, b, n*C] - mixed output (H_res @ x_l)
            h_post: [s, b, n] - expansion weights
            
        """
        # Compute mappings
        h_pre, h_post, h_res = self.compute_mappings(hidden_states)
        
        # Aggregate for layer input
        aggregated = self.aggregate(hidden_states, h_pre)

        mixed = self.apply_h_res(h_res, residual)

        return aggregated, mixed, h_post
    
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
