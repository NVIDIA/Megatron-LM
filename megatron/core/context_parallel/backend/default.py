from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import nn

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)

from ._base import ContextParallelHandler

if TYPE_CHECKING:
    from megatron.core.transformer import TransformerConfig

try:
    # Register the TE CUDA kernels
    import transformer_engine  # pylint: disable=unused-import

    # Alias the PyTorch wrapper so we can call tex.* APIs
    import transformer_engine_torch as tex
except ImportError:
    # TE isn’t installed or the torch wrapper is missing
    tex = None


# =============================================================================
# Helper Functions
# =============================================================================


def _get_interleaved_indices(rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    """
    Generate interleaved indices for the specific rank using a zigzag pattern.

    Pattern: [rank, 2*world_size - rank - 1]
    This ensures a balanced distribution of tokens, often used for Ring-Attention.

    Args:
        rank (int): The current context parallel rank.
        world_size (int): The total context parallel size.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: A tensor containing the indices assigned to this rank.
    """
    return torch.tensor([rank, 2 * world_size - rank - 1], device=device, dtype=torch.long)


def _view_as_chunks(tensor: torch.Tensor, seq_dim: int, cp_size: int) -> torch.Tensor:
    """
    Reshape the tensor to view it as chunks along the sequence dimension.

    Transformation: [..., seq_len, ...] -> [..., 2*cp, chunk_len, ...]

    Args:
        tensor (torch.Tensor): The input tensor.
        seq_dim (int): The dimension representing the sequence length.
        cp_size (int): The context parallel size.

    Returns:
        torch.Tensor: The reshaped tensor with an extra dimension for chunks.

    Raises:
        ValueError: If seq_len is not divisible by 2 * cp_size.
    """
    shape = list(tensor.shape)
    # Ensure seq_len is divisible by 2*cp_size, a prerequisite for the Interleaved strategy
    if shape[seq_dim] % (2 * cp_size) != 0:
        raise ValueError(
            f"Sequence length {shape[seq_dim]} must be divisible by 2 * cp_size {2 * cp_size}"
        )

    num_chunks = 2 * cp_size
    chunk_len = shape[seq_dim] // num_chunks
    new_shape = shape[:seq_dim] + [num_chunks, chunk_len] + shape[seq_dim + 1 :]
    return tensor.view(new_shape)


def _flatten_chunks(tensor: torch.Tensor, seq_dim: int) -> torch.Tensor:
    """
    Flatten the chunk dimensions back into the sequence dimension.

    Transformation: [..., chunks, chunk_len, ...] -> [..., seq_len, ...]

    Args:
        tensor (torch.Tensor): The chunked tensor.
        seq_dim (int): The original sequence dimension index.

    Returns:
        torch.Tensor: The flattened tensor.
    """
    shape = list(tensor.shape)
    new_shape = shape[:seq_dim] + [-1] + shape[seq_dim + 2 :]
    return tensor.view(new_shape)


def _get_packed_indices(
    cu_seqlens: torch.Tensor, total_len: int, rank: int, world_size: int, device: torch.device
) -> torch.Tensor:
    """
    Wrapper for THD/Packed index calculation using Transformer Engine.

    Args:
        cu_seqlens (torch.Tensor): Cumulative sequence lengths.
        total_len (int): Total sequence length (padded).
        rank (int): Current CP rank.
        world_size (int): Total CP world size.
        device (torch.device): Device for the output indices.

    Returns:
        torch.Tensor: The indices belonging to the current rank for variable length sequences.
    """
    return tex.thd_get_partitioned_indices(cu_seqlens, total_len, world_size, rank).to(
        device=device, dtype=torch.long
    )


def _copy_compact_to_padded(
    compact_tensor: torch.Tensor,
    padded_shape: List[int],
    seqlens_list: List[int],
    seqlens_with_padded_list: List[int],
    seq_dim: int,
) -> torch.Tensor:
    """
    Helper: Copy data from Packed (Compact) format to Padded format.
    Used to align sequences before splitting for Context Parallelism.

    Args:
        compact_tensor (torch.Tensor): The source compact tensor.
        padded_shape (List[int]): The target shape of the padded tensor.
        seqlens_list (List[int]): List of actual sequence lengths.
        seqlens_with_padded_list (List[int]): List containing [actual_len, pad_len] pairs flattened.
        seq_dim (int): The sequence dimension.

    Returns:
        torch.Tensor: The padded tensor.
    """
    padded_tensor = torch.zeros(
        padded_shape, device=compact_tensor.device, dtype=compact_tensor.dtype
    )
    foreach_srcs = torch.split(compact_tensor, seqlens_list, dim=seq_dim)
    # seqlens_with_padded_list contains [actual_len, padding_len],
    # we select every 2nd element (actual_len) for destination
    foreach_dsts = torch.split(padded_tensor, seqlens_with_padded_list, dim=seq_dim)[::2]
    torch._foreach_copy_(foreach_dsts, foreach_srcs)
    return padded_tensor


def _copy_padded_to_compact(
    padded_tensor: torch.Tensor,
    compact_shape: List[int],
    seqlens_list: List[int],
    seqlens_with_padded_list: List[int],
    seq_dim: int,
) -> torch.Tensor:
    """
    Helper: Copy data from Padded format back to Packed (Compact) format.

    Args:
        padded_tensor (torch.Tensor): The source padded tensor.
        compact_shape (List[int]): The target shape of the compact tensor.
        seqlens_list (List[int]): List of actual sequence lengths.
        seqlens_with_padded_list (List[int]): List containing [actual_len, pad_len] pairs flattened.
        seq_dim (int): The sequence dimension.

    Returns:
        torch.Tensor: The compact tensor.
    """
    compact_tensor = torch.zeros(
        compact_shape, device=padded_tensor.device, dtype=padded_tensor.dtype
    )
    # Select actual data segments from padded tensor
    foreach_srcs = torch.split(padded_tensor, seqlens_with_padded_list, dim=seq_dim)[::2]
    foreach_dsts = torch.split(compact_tensor, seqlens_list, dim=seq_dim)
    torch._foreach_copy_(foreach_dsts, foreach_srcs)
    return compact_tensor


# =============================================================================
# Strategy 1: Interleaved (Equivalent to SBHD/BSHD)
# Suitable for Dense Tensors with regular dimensions.
# =============================================================================


class InterleavedDispatchFwdCombineBwd(torch.autograd.Function):
    """
    Universal Interleaved Dispatcher.

    Implements a Scatter-Gather pattern optimized for Context Parallelism.
    """

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, seq_dim: int, cp_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Forward pass: Dispatch logic.

        1. Input is a Global Tensor (full sequence).
        2. View as chunks.
        3. Select specific chunks for the current rank based on interleaved indices.

        Args:
            ctx (Any): Autograd context.
            tensor (torch.Tensor): Global input tensor.
            seq_dim (int): Sequence dimension.
            cp_group (dist.ProcessGroup): Context parallel process group.

        Returns:
            torch.Tensor: Local tensor chunk for this rank.
        """
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        ctx.seq_dim = seq_dim
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.global_shape = tensor.shape

        tensor_view = _view_as_chunks(tensor, seq_dim, cp_size)
        indices = _get_interleaved_indices(cp_rank, cp_size, tensor.device)
        local_view = tensor_view.index_select(seq_dim, indices)
        return _flatten_chunks(local_view, seq_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        """
        Backward pass: Combine logic.

        1. Input is Local Gradient.
        2. Place Local Gradient into a zero-filled Global Gradient tensor.
        3. AllReduce (Sum) to synchronize gradients across all ranks.

        Args:
            ctx (Any): Autograd context.
            grad_output (torch.Tensor): Gradient w.r.t the local output.

        Returns:
            Tuple: Gradient w.r.t input, None, None.
        """
        seq_dim = ctx.seq_dim
        cp_size = ctx.cp_size

        # 1. Prepare Global Grad container
        global_grad = torch.zeros(
            ctx.global_shape, device=grad_output.device, dtype=grad_output.dtype
        )
        global_grad_view = _view_as_chunks(global_grad, seq_dim, cp_size)

        # 2. Reshape Local Grad to [..., 2, chunk_len, ...]
        local_grad_shape = list(grad_output.shape)
        chunk_len = local_grad_shape[seq_dim] // 2
        view_shape = local_grad_shape[:seq_dim] + [2, chunk_len] + local_grad_shape[seq_dim + 1 :]
        grad_output_view = grad_output.view(view_shape)

        # 3. Scatter local gradients into the global view
        indices = _get_interleaved_indices(ctx.cp_rank, cp_size, grad_output.device)
        global_grad_view.index_add_(seq_dim, indices, grad_output_view)

        # 4. AllReduce to sum gradients from all CP ranks
        dist.all_reduce(global_grad, op=dist.ReduceOp.SUM, group=ctx.cp_group)

        return global_grad, None, None


class InterleavedCombineFwdDispatchBwd(torch.autograd.Function):
    """
    Universal Interleaved Combiner.

    Inverse of the Dispatcher, gathering local chunks into a global tensor.
    """

    @staticmethod
    def forward(
        ctx: Any, local_tensor: torch.Tensor, seq_dim: int, cp_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """
        Forward pass: Combine logic.

        1. Input is Local Tensor.
        2. Place into zero-filled Global Tensor at correct indices.
        3. AllReduce (Sum) to gather full data from all ranks.

        Args:
            ctx (Any): Autograd context.
            local_tensor (torch.Tensor): Local input tensor chunk.
            seq_dim (int): Sequence dimension.
            cp_group (dist.ProcessGroup): Context parallel process group.

        Returns:
            torch.Tensor: The gathered Global tensor.
        """
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        ctx.seq_dim = seq_dim
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank

        local_shape = list(local_tensor.shape)
        global_shape = list(local_shape)
        global_shape[seq_dim] = local_shape[seq_dim] * cp_size
        ctx.global_shape = global_shape

        # 1. Initialize Global Tensor
        global_tensor = torch.zeros(
            global_shape, device=local_tensor.device, dtype=local_tensor.dtype
        )
        global_view = _view_as_chunks(global_tensor, seq_dim, cp_size)

        # 2. View Local Tensor
        chunk_len = local_shape[seq_dim] // 2
        local_view_shape = local_shape[:seq_dim] + [2, chunk_len] + local_shape[seq_dim + 1 :]
        local_view = local_tensor.view(local_view_shape)

        # 3. Fill specific indices and AllReduce to aggregate
        indices = _get_interleaved_indices(cp_rank, cp_size, local_tensor.device)
        global_view.index_copy_(seq_dim, indices, local_view)
        dist.all_reduce(global_tensor, op=dist.ReduceOp.SUM, group=cp_group)

        return global_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        """
        Backward pass: Dispatch logic.

        1. Input is Global Gradient.
        2. AllReduce to ensure consistency (standard CP backward flow).
        3. Slice/Dispatch chunks relevant to this rank.

        Args:
            ctx (Any): Autograd context.
            grad_output (torch.Tensor): Gradient w.r.t the global output.

        Returns:
            Tuple: Gradient w.r.t local input, None, None.
        """
        grad_view = _view_as_chunks(grad_output, ctx.seq_dim, ctx.cp_size)
        indices = _get_interleaved_indices(ctx.cp_rank, ctx.cp_size, grad_output.device)
        local_grad_view = grad_view.index_select(ctx.seq_dim, indices)

        return _flatten_chunks(local_grad_view, ctx.seq_dim), None, None


# =============================================================================
# Strategy 2: Packed (Equivalent to THD)
# Suitable for Variable Length Tensors, 1D Packed format.
# =============================================================================


class PackedDispatchFwdCombineBwd(torch.autograd.Function):
    """
    Dispatch logic for Packed/THD format (Variable Sequence Length).
    Handles padding and unpadding to deal with CP distribution.
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        seq_dim: int,
        seqlens_list: List[int],
        seqlens_with_padded_list: List[int],
        cu_seqlens_padded: torch.Tensor,
        total_seqlen_padded: int,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass:
        1. Compact (Full) -> Pad to align boundaries.
        2. Slice specific indices for Local Rank.

        Args:
            ctx (Any): Autograd context.
            tensor (torch.Tensor): Global packed tensor.
            seq_dim (int): Dimension to split.
            seqlens_list (List[int]): Original sequence lengths.
            seqlens_with_padded_list (List[int]): Layout for padding.
            cu_seqlens_padded (torch.Tensor): Cumulative seq lengths (padded).
            total_seqlen_padded (int): Total length after padding.
            cp_group (dist.ProcessGroup): Process group.

        Returns:
            torch.Tensor: Local tensor part (padded view sliced).
        """
        # Save Tensors for backward
        ctx.save_for_backward(cu_seqlens_padded)

        # Save Metadata
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        ctx.meta = {
            "seq_dim": seq_dim,
            "cp_group": cp_group,
            "cp_size": cp_size,
            "cp_rank": cp_rank,
            "total_seqlen_padded": total_seqlen_padded,
            "seqlens_list": seqlens_list,
            "seqlens_with_padded_list": seqlens_with_padded_list,
            "compact_shape": tensor.shape,
        }

        # 1. Compact -> Padded conversion
        padded_shape = list(tensor.shape)
        padded_shape[seq_dim] = total_seqlen_padded
        padded_tensor = _copy_compact_to_padded(
            tensor, padded_shape, seqlens_list, seqlens_with_padded_list, seq_dim
        )

        # 2. Slice for current rank
        local_indices = _get_packed_indices(
            cu_seqlens_padded, total_seqlen_padded, cp_rank, cp_size, tensor.device
        )
        return padded_tensor.index_select(seq_dim, local_indices)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None]:
        """
        Backward pass:
        1. Local Grad -> Pad into full shape.
        2. Unpad back to Compact (Full) shape.
        3. AllReduce to sum gradients.

        Args:
            ctx (Any): Autograd context.
            grad_output (torch.Tensor): Gradient w.r.t local output.

        Returns:
            Tuple: Gradients for inputs.
        """
        (cu_seqlens_padded,) = ctx.saved_tensors
        meta = ctx.meta
        seq_dim = meta["seq_dim"]

        # 1. Local Grad -> Padded Grad
        padded_shape = list(meta["compact_shape"])
        padded_shape[seq_dim] = meta["total_seqlen_padded"]
        padded_grad = torch.zeros(padded_shape, device=grad_output.device, dtype=grad_output.dtype)

        local_indices = _get_packed_indices(
            cu_seqlens_padded,
            meta["total_seqlen_padded"],
            meta["cp_rank"],
            meta["cp_size"],
            grad_output.device,
        )
        padded_grad.index_add_(seq_dim, local_indices, grad_output)

        # 2. Unpad -> Compact Grad
        compact_grad = _copy_padded_to_compact(
            padded_grad,
            meta["compact_shape"],
            meta["seqlens_list"],
            meta["seqlens_with_padded_list"],
            seq_dim,
        )

        # 3. AllReduce (on Compact tensor)
        dist.all_reduce(compact_grad, op=dist.ReduceOp.SUM, group=meta["cp_group"])

        return compact_grad, None, None, None, None, None, None


class PackedCombineFwdDispatchBwd(torch.autograd.Function):
    """
    Combine logic for Packed/THD format.
    Gathers distributed packed tensors into a single global packed tensor.
    """

    @staticmethod
    def forward(
        ctx: Any,
        local_tensor: torch.Tensor,
        seq_dim: int,
        seqlens_list: List[int],
        seqlens_with_padded_list: List[int],
        cu_seqlens_padded: torch.Tensor,
        total_seqlen_padded: int,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass:
        1. Local Tensor -> Pad to global alignment.
        2. Unpad to Compact format (resulting in a sparse global tensor).
        3. AllReduce to aggregate sparse tensors into a dense Compact Tensor.

        Args:
            ctx (Any): Autograd context.
            local_tensor (torch.Tensor): Local input tensor part.
            seq_dim (int): Sequence dimension.
            seqlens_list (List[int]): Original sequence lengths.
            seqlens_with_padded_list (List[int]): Layout for padding.
            cu_seqlens_padded (torch.Tensor): Cumulative seq lengths (padded).
            total_seqlen_padded (int): Total length after padding.
            cp_group (dist.ProcessGroup): Process group.

        Returns:
            torch.Tensor: Aggregated Global Packed Tensor.
        """
        ctx.save_for_backward(cu_seqlens_padded)
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        padded_shape = list(local_tensor.shape)
        padded_shape[seq_dim] = total_seqlen_padded

        ctx.meta = {
            "seq_dim": seq_dim,
            "cp_group": cp_group,
            "cp_size": cp_size,
            "cp_rank": cp_rank,
            "total_seqlen_padded": total_seqlen_padded,
            "seqlens_list": seqlens_list,
            "seqlens_with_padded_list": seqlens_with_padded_list,
            "padded_shape": padded_shape,
        }

        # 1. Local -> Padded
        padded_tensor = torch.zeros(
            padded_shape, device=local_tensor.device, dtype=local_tensor.dtype
        )
        local_indices = _get_packed_indices(
            cu_seqlens_padded, total_seqlen_padded, cp_rank, cp_size, local_tensor.device
        )
        padded_tensor.index_copy_(seq_dim, local_indices, local_tensor)

        # 2. Padded -> Compact (Sparse: mostly zeros except for local data)
        compact_shape = list(local_tensor.shape)
        compact_shape[seq_dim] = sum(seqlens_list)
        compact_tensor = _copy_padded_to_compact(
            padded_tensor, compact_shape, seqlens_list, seqlens_with_padded_list, seq_dim
        )

        # 3. AllReduce to aggregate
        dist.all_reduce(compact_tensor, op=dist.ReduceOp.SUM, group=cp_group)
        return compact_tensor

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None]:
        """
        Backward pass:
        1. Compact Grad (Full) -> AllReduce.
        2. Pad -> Slice specific indices to get Local Grad.

        Args:
            ctx (Any): Autograd context.
            grad_output (torch.Tensor): Gradient w.r.t global output.

        Returns:
            Tuple: Gradients for inputs.
        """
        (cu_seqlens_padded,) = ctx.saved_tensors
        meta = ctx.meta
        seq_dim = meta["seq_dim"]

        # 1. Compact -> Padded
        padded_grad = _copy_compact_to_padded(
            grad_output,
            meta["padded_shape"],
            meta["seqlens_list"],
            meta["seqlens_with_padded_list"],
            seq_dim,
        )

        # 2. Slice Local part
        local_indices = _get_packed_indices(
            cu_seqlens_padded,
            meta["total_seqlen_padded"],
            meta["cp_rank"],
            meta["cp_size"],
            grad_output.device,
        )
        grad_input = padded_grad.index_select(seq_dim, local_indices)

        return grad_input, None, None, None, None, None, None


@dataclass
class DefaultContextParallelHandler(ContextParallelHandler):
    """
    Default implementation of the Context Parallel Handler.
    Supports both Interleaved (Dense/SBHD) and Packed (Variable Length/THD) formats.
    """

    def __post_init__(self) -> None:
        """
        Post-initialization to set up sequence lengths and padding information
        required for THD/Packed format context parallelism.
        """
        super().__post_init__()

        if self.qkv_format == "thd":
            assert self.cu_seqlens_q is not None and self.cu_seqlens_kv is not None

            # Reference existing tensors
            self.cu_seqlens_q = self.cu_seqlens_q
            self.cu_seqlens_kv = self.cu_seqlens_kv

            # Calculate individual sequence lengths from cumulative lengths
            seqlens_q = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
            seqlens_kv = self.cu_seqlens_kv[1:] - self.cu_seqlens_kv[:-1]

            self.seqlens_q_list = seqlens_q.tolist()
            self.seqlens_kv_list = seqlens_kv.tolist()

            # Calculate padded sequence lengths to be divisible by (2 * cp_size)
            # This is often required for specific balanced splitting strategies
            align_factor = 2 * self._cp_size
            seqlens_q_padded = (seqlens_q + align_factor - 1) // align_factor * align_factor
            seqlens_kv_padded = (seqlens_kv + align_factor - 1) // align_factor * align_factor

            seqlens_q_padded_size = seqlens_q_padded - seqlens_q
            seqlens_kv_padded_size = seqlens_kv_padded - seqlens_kv

            self.seqlens_q_padded = seqlens_q_padded
            self.seqlens_kv_padded = seqlens_kv_padded

            max_seqlen_q = torch.max(seqlens_q_padded).item()
            max_seqlen_kv = torch.max(seqlens_kv_padded).item()
            self.max_seqlen_q = max_seqlen_q
            self.max_seqlen_kv = max_seqlen_kv

            # Create a flattened list of [actual_len, pad_len] for internal C++ or CUDA kernels
            self.seqlens_q_with_padded_list = (
                torch.stack([seqlens_q, seqlens_q_padded_size], dim=1).flatten().tolist()
            )
            self.seqlens_kv_with_padded_list = (
                torch.stack([seqlens_kv, seqlens_kv_padded_size], dim=1).flatten().tolist()
            )

            self.total_seqlen_padded_q = torch.sum(seqlens_q_padded).item()
            self.total_seqlen_padded_kv = torch.sum(seqlens_kv_padded).item()

            # Re-calculate cumulative sequence lengths based on padded values
            cu_seqlens_q_padded = torch.zeros(
                seqlens_q_padded.size(0) + 1,
                device=seqlens_q_padded.device,
                dtype=seqlens_q_padded.dtype,
            )
            cu_seqlens_kv_padded = torch.zeros(
                seqlens_kv_padded.size(0) + 1,
                device=seqlens_kv_padded.device,
                dtype=seqlens_kv_padded.dtype,
            )

            torch.cumsum(seqlens_q_padded, dim=0, out=cu_seqlens_q_padded[1:])
            torch.cumsum(seqlens_kv_padded, dim=0, out=cu_seqlens_kv_padded[1:])
            self.cu_seqlens_q_padded = cu_seqlens_q_padded
            self.cu_seqlens_kv_padded = cu_seqlens_kv_padded

            self._post_initialized = True

    def dispatch(
        self, seq_dim: int, tensor: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Splits a global tensor into local chunks for the current Context Parallel rank.

        Args:
            seq_dim (int): The dimension to split along.
            tensor (torch.Tensor): The global tensor.

        Returns:
            torch.Tensor: The local tensor chunk.
        """
        if self._cp_size == 1:
            return tensor

        # Strategy 1: Interleaved (Corresponds to SBHD, BSHD etc., dense formats)
        if self.qkv_format in ["sbhd", "bshd"]:
            return InterleavedDispatchFwdCombineBwd.apply(tensor, seq_dim, self.cp_group)

        # Strategy 2: Packed (Corresponds to THD, variable length formats)
        if self.qkv_format == "thd":
            return PackedDispatchFwdCombineBwd.apply(
                tensor,
                seq_dim,
                self.seqlens_q_list,
                self.seqlens_q_with_padded_list,
                self.cu_seqlens_q_padded,
                self.total_seqlen_padded_q,
                self.cp_group,
            )

        return tensor

    def combine(
        self, seq_dim: int, tensor: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Gathers local tensor chunks from all Context Parallel ranks into a global tensor.

        Args:
            seq_dim (int): The dimension to combine along.
            tensor (torch.Tensor): The local tensor chunk.

        Returns:
            torch.Tensor: The gathered global tensor.
        """
        if self._cp_size == 1:
            return tensor

        # Strategy 1: Interleaved
        if self.qkv_format in ["sbhd", "bshd"]:
            return InterleavedCombineFwdDispatchBwd.apply(tensor, seq_dim, self.cp_group)

        # Strategy 2: Packed
        if self.qkv_format == "thd":
            return PackedCombineFwdDispatchBwd.apply(
                tensor,
                seq_dim,
                self.seqlens_q_list,
                self.seqlens_q_with_padded_list,
                self.cu_seqlens_q_padded,
                self.total_seqlen_padded_q,
                self.cp_group,
            )

        return tensor

    def apply_rotary_pos_emb(
        self,
        tensor: torch.Tensor,
        freq: torch.Tensor,
        config: "TransformerConfig",
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Applies Rotary Positional Embeddings (RoPE), accounting for Context Parallel offsets.

        Args:
            tensor (torch.Tensor): Input tensor.
            freq (torch.Tensor): RoPE frequencies.
            config (TransformerConfig): Configuration object.

        Returns:
            torch.Tensor: Tensor with RoPE applied.
        """
        if self.qkv_format in ["sbhd", "bshd"]:
            return apply_rotary_pos_emb(
                t=tensor,
                freqs=freq,
                config=config,
                cu_seqlens=None,
                mscale=_yarn_get_concentration_factor_from_config(config),
                cp_group=self.cp_group,
            )
        if self.qkv_format == "thd":
            return apply_rotary_pos_emb(
                t=tensor,
                freqs=freq,
                config=config,
                cu_seqlens=self.cu_seqlens_q_padded,
                mscale=_yarn_get_concentration_factor_from_config(config),
                cp_group=self.cp_group,
            )
        return tensor

    def get_emb_on_this_cp_rank(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Extracts the embedding portion relevant to the current CP rank.

        Args:
            emb (torch.Tensor): Global embedding tensor.

        Returns:
            torch.Tensor: Local embedding tensor.
        """
        if self.qkv_format in ["sbhd", "bshd"]:
            return self.dispatch(seq_dim=0, tensor=emb)

        if self.qkv_format == "thd":
            return emb

        return emb

    def roll_tensor(
        self, tensor: torch.Tensor, shifts: int = -1, dims: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a roll operation on the tensor, handling distributed data exchange.
        Crucial for Ring-Attention mechanisms where KV blocks need to shift between ranks.

        For Dense (sbhd/bshd):
            - Splits the tensor into two chunks.
            - Communicates boundary elements between ranks (Send/Recv).
            - Re-assembles the tensor.

        For Packed (thd):
            - Iterates through packed sequences.
            - Handles boundary communication per sequence.

        Args:
            tensor (torch.Tensor): Input tensor.
            shifts (int): Number of places by which elements are shifted (default -1).
            dims (int): Dimension along which to roll (default -1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The rolled tensor.
                - The sum of the rolled tensor (useful for correctness checks/gradient flow).
        """
        if self.qkv_format in ["sbhd", "bshd"]:
            if self._cp_size == 1:
                if self.qkv_format in ["sbhd", "bshd"]:
                    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
                    rolled_tensor.select(dims, shifts).fill_(0)
                    return rolled_tensor, rolled_tensor.sum()

            # CP-enabled rolling: Split tensor into chunks and handle boundary communication
            # This matches the batch splitting logic in get_batch_on_this_cp_rank() function
            tensor_list = tensor.chunk(2, dim=dims)
            rolled_tensor_list = []
            for i in range(len(tensor_list)):
                rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

            # Prepare tensors for communication between CP ranks
            # Each CP rank needs to send boundary elements to adjacent ranks
            tensor_send_list = []
            tensor_recv_list = []
            for i in range(len(rolled_tensor_list)):
                tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
                empty_tensor = torch.empty(
                    tensor_send_list[i].shape,
                    dtype=tensor_send_list[i].dtype,
                    device=torch.cuda.current_device(),
                )
                tensor_recv_list.append(empty_tensor)

            # Get the global rank of next and prev process in the cp group
            global_ranks = dist.get_process_group_ranks(group=self.cp_group)
            local_rank = dist.get_rank(group=self.cp_group)

            # Note: global_ranks uses list indexing, so we compute neighbors via modulo arithmetic
            next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
            prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

            # Start send and recv ops
            ops = []
            if local_rank != 0:
                req_send_first_part = dist.isend(tensor=tensor_send_list[0], dst=prev_rank)
                ops.append(req_send_first_part)
                req_recv_second_part = dist.irecv(tensor=tensor_recv_list[1], src=prev_rank)
                ops.append(req_recv_second_part)
            else:
                # Inserted elements at the very start are set to 0.0
                tensor_recv_list[1] = 0

            if local_rank != len(global_ranks) - 1:
                req_recv_first_part = dist.irecv(tensor=tensor_recv_list[0], src=next_rank)
                ops.append(req_recv_first_part)
                req_send_second_part = dist.isend(tensor=tensor_send_list[1], dst=next_rank)
                ops.append(req_send_second_part)
            else:
                # For the last CP rank, the removed elements of second part go into the first part
                tensor_recv_list[0] = tensor_send_list[1]

            # Wait for all communication operations to complete
            for op in ops:
                op.wait()

            # Splicing: Replace boundary elements with received elements from adjacent ranks
            # This ensures proper sequence continuity across CP boundaries
            index = [slice(None)] * rolled_tensor_list[0].dim()
            index[dims] = shifts
            for i in range(len(rolled_tensor_list)):
                rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

            # Concatenate the processed chunks back into a single tensor
            rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)

            return rolled_tensor, rolled_tensor.sum()

        if self.qkv_format == "thd":
            # Notice: This is a naive implementation to test the correctness.
            # A better solution would only sync the boundary tokens once.
            assert (
                dims == -1 or dims == tensor.dim() - 1
            ), "Packed sequence roll only supports the last dimension."
            assert shifts == -1, "Packed sequence roll only supports a single-token left shift."

            cu_seqlens = self.cu_seqlens_q
            rolled_tensor = tensor.clone()

            if self._cp_size == 1:
                # CP disabled: roll each packed sequence independently within its boundaries
                for i in range(len(cu_seqlens) - 1):
                    start_idx = cu_seqlens[i]
                    end_idx = cu_seqlens[i + 1]
                    seq_slice = tensor[..., start_idx:end_idx]
                    rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dims)
                    # Zero out the last position(s) that would cross sequence boundaries
                    rolled_seq[..., shifts:] = 0
                    rolled_tensor[..., start_idx:end_idx] = rolled_seq
                return rolled_tensor, rolled_tensor.sum()

            """Roll tensor with packed sequence support (CP enabled).
            This function handles rolling for packed sequences by respecting sequence boundaries
            distributed across ranks.
            """

            # CP enabled: each rank owns two chunks per sequence (front and mirrored tail).
            local_rank = dist.get_rank(group=self.cp_group)
            global_ranks = dist.get_process_group_ranks(group=self.cp_group)
            next_rank = global_ranks[(local_rank + 1) % self._cp_size]
            prev_rank = global_ranks[(local_rank - 1) % self._cp_size]

            # Iterate over each sequence individually
            for i in range(len(cu_seqlens) - 1):
                start_idx = cu_seqlens[i]
                end_idx = cu_seqlens[i + 1]

                # The index has been multiplied by cp_size,
                # need to divide it by cp_size to get the local idx
                local_start_idx = start_idx // self._cp_size
                local_end_idx = end_idx // self._cp_size
                tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()

                # The following code is very similar to the SBHD logic above
                local_chunks = tensor_slice.chunk(2, dim=dims)
                rolled_chunks = [
                    torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks
                ]

                tensor_send_list = []
                tensor_recv_list = []
                for chunk in rolled_chunks:
                    boundary = chunk.select(dims, shifts).contiguous().clone()
                    tensor_send_list.append(boundary)
                    tensor_recv_list.append(torch.empty_like(boundary))

                ops = []
                if local_rank != 0:
                    ops.append(dist.isend(tensor=tensor_send_list[0], dst=prev_rank))
                    ops.append(dist.irecv(tensor=tensor_recv_list[1], src=prev_rank))
                else:
                    tensor_recv_list[1].zero_()

                if local_rank != self._cp_size - 1:
                    ops.append(dist.irecv(tensor=tensor_recv_list[0], src=next_rank))
                    ops.append(dist.isend(tensor=tensor_send_list[1], dst=next_rank))
                else:
                    tensor_recv_list[0].copy_(tensor_send_list[1])

                for op in ops:
                    op.wait()

                index = [slice(None)] * rolled_chunks[0].dim()
                index[dims] = shifts
                for chunk, recv in zip(rolled_chunks, tensor_recv_list):
                    chunk[tuple(index)] = recv

                seq_result = torch.cat(rolled_chunks, dim=dims)

                # Update the rolled tensor
                rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

            return rolled_tensor, rolled_tensor.sum()

        # Fallback for unexpected formats
        return tensor, tensor.sum()

    def core_attn(self, attn_mod: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the core attention module, injecting the context parallel handler.

        Args:
            attn_mod (nn.Module): The attention module to execute.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The output of the attention module.
        """
        return attn_mod(cp_handler=self, *args, **kwargs)


@dataclass
class TEDynamicContextParallelHandler(DefaultContextParallelHandler):
    """
    Placeholder for Transformer Engine Dynamic Context Parallel Handler.
    Inherits from DefaultContextParallelHandler.
    """

    pass
