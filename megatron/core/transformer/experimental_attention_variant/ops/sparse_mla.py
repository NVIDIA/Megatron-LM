import torch

from .tilelang_sparse_mla_bwd import sparse_mla_bwd, sparse_mla_delta
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

        # Do not save tl_out/tl_lse: backward recomputes them just long enough to form
        # delta and run the kernel. Saved inputs still go through autograd's saved-tensor
        # hooks/offload path and retain_graph can recompute these tensors again.
        ctx.save_for_backward(q, kv, indices)

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        q, kv, indices = ctx.saved_tensors
        scaling = ctx.scaling
        grad_output = grad_output.contiguous()
        with torch.no_grad():
            tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
        delta = sparse_mla_delta(tl_out, grad_output)
        del tl_out

        tl_dq, tl_dkv = sparse_mla_bwd(
            q, kv, None, grad_output, indices, tl_lse, sm_scale=scaling, delta=delta
        )
        del tl_lse

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None
