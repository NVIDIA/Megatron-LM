import torch

from .tilelang_sparse_mla_bwd import sparse_mla_bwd
from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface


class SparseMLA(torch.autograd.Function):
    """Autograd wrapper around tilelang sparse-MLA forward/backward kernels."""

    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        """
        Args:
            q: Query tensor (seq_len, heads, dim_plus_tail_dim)
            kv: Key-Value tensor (seq_len_kv, kv_group, dim_plus_tail_dim)
            indices: Sparse indices tensor (seq_len, kv_group, topk)

        Returns:
            out: Output tensor (seq_len, heads, dim)
        """
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)

        # Save tensors for backward pass
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        tl_dq, tl_dkv = sparse_mla_bwd(
            q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling
        )

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None
