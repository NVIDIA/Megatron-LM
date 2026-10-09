# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Chunk-aligned Mamba forests with shared prefix state and ragged branches.

Only the unaligned prefix tail and the convolution halo are copied to siblings.
Every continuation has its own chunk padding; no longest-sibling rectangle is
materialized. The surrounding projections, gate and distributed layout remain
in the canonical shared-token order.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl
from causal_conv1d import causal_conv1d_fn
from einops import rearrange

from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)
from megatron.core.ssm.mamba_mixer import MAMBA_HAS_STATE_DTYPE, MambaMixer


@triton.jit
def _gather_forward(X, INDEX, Y, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(INDEX + row)
    value = tl.load(X + source * WIDTH + col, (source >= 0) & (col < WIDTH), other=0)
    tl.store(Y + row * WIDTH + col, value, col < WIDTH)


@triton.jit
def _gather_backward(
    DY,
    CONTRIBUTORS,
    DX,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    COPIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), tl.float32)
    # One owner per input element: deterministic FP32 accumulation, no atomics.
    for copy in range(COPIES - 1, -1, -1):
        source = tl.load(CONTRIBUTORS + copy * ROWS + row)
        value += tl.load(DY + source * WIDTH + col, (source >= 0) & (col < WIDTH), other=0).to(
            tl.float32
        )
    tl.store(DX + row * WIDTH + col, value, col < WIDTH)


class _RaggedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices, contributors):
        """Gather ragged token rows using the explicit ownership indices."""
        value = value.contiguous()
        output = value.new_empty((indices.numel(), 1, value.shape[-1]))
        ctx.save_for_backward(contributors)
        ctx.input_shape = value.shape
        _gather_forward[(indices.numel(), triton.cdiv(value.shape[-1], 256))](
            value, indices, output, value.shape[-1], 256
        )
        return output

    @staticmethod
    def backward(ctx, gradient):
        """Sum logical copy gradients with one owner for every physical input element."""
        (contributors,) = ctx.saved_tensors
        gradient = gradient.contiguous()
        output = gradient.new_empty(ctx.input_shape)
        _gather_backward[(output.shape[0], triton.cdiv(output.shape[-1], 256))](
            gradient,
            contributors,
            output,
            output.shape[0],
            output.shape[-1],
            contributors.shape[0],
            256,
        )
        return output, None, None


@dataclass(frozen=True)
class RaggedMambaLayout:
    """Cached token maps and chunk topology; these tensors carry no gradients."""

    convolution_indices: torch.Tensor
    convolution_sequences: torch.Tensor
    contributors: torch.Tensor
    scan_indices: torch.Tensor
    output_indices: torch.Tensor
    segment_chunks: torch.Tensor
    root_segments: torch.Tensor
    input_tokens: int
    scan_tokens: int
    replayed_tail_tokens: int
    padding_tokens: int


@lru_cache(maxsize=16)
def ragged_mamba_layout(
    roots: tuple[tuple[int, tuple[int, ...]], ...],
    chunk_size: int,
    convolution_width: int,
    device: torch.device,
) -> RaggedMambaLayout:
    """Build a forest whose prefix ends at the established scan chunk boundary."""
    if not roots or chunk_size < 1 or chunk_size & (chunk_size - 1):
        raise ValueError("Ragged Mamba requires roots and a power-of-two chunk size")
    if convolution_width < 1:
        raise ValueError("Ragged Mamba requires a positive convolution width")
    convolution_indices, convolution_sequences, scan_indices, output_indices = ([], [], [], [])
    segment_chunks, root_segments = [0], [0]
    input_start = scan_start = replayed = padding = 0
    halo = convolution_width - 1

    for prefix, lengths in roots:
        if prefix < 1 or not lengths or min(lengths) < 0:
            raise ValueError("Ragged Mamba requires a nonempty prefix and valid siblings")
        head = prefix // chunk_size * chunk_size
        tail = prefix - head
        segments = [(list(range(input_start, input_start + head)), [-1] * halo)]
        branch_start = input_start + prefix
        context = [input_start + i if i >= 0 else -1 for i in range(head - halo, head)]
        for length in lengths:
            tokens = list(range(input_start + head, input_start + prefix))
            tokens.extend(range(branch_start, branch_start + length))
            segments.append((tokens, context))
            branch_start += length
        replayed += tail * (len(lengths) - 1)

        for segment, (tokens, context) in enumerate(segments):
            real_length = len(tokens)
            padded = (real_length + chunk_size - 1) // chunk_size * chunk_size
            padding += padded - real_length
            if padded:
                convolution_start = len(convolution_indices)
                convolution_indices.extend(context)
                convolution_indices.extend(tokens)
                convolution_indices.extend([-1] * (padded - real_length))
                convolution_sequences.extend([len(segment_chunks) - 1] * (halo + padded))
                scan_indices.extend(
                    range(convolution_start + halo, convolution_start + halo + padded)
                )
            if segment == 0:
                output_indices.extend(range(scan_start, scan_start + head))
            else:
                if segment == 1:
                    output_indices.extend(range(scan_start, scan_start + tail))
                output_indices.extend(range(scan_start + tail, scan_start + real_length))
            scan_start += padded
            segment_chunks.append(scan_start // chunk_size)
        root_segments.append(len(segment_chunks) - 1)
        input_start = branch_start

    # Preserve the original contributor order without allocating one Python
    # list per token or iterating over every token/copy pair in Python. Stable
    # sorting keeps destinations in ascending order within each source token;
    # the backward kernel still accumulates them in exactly the reverse order.
    sources = np.asarray(convolution_indices, dtype=np.int32)
    destinations = np.flatnonzero(sources >= 0)
    sources = sources[destinations]
    order = np.argsort(sources, kind="stable")
    counts = np.bincount(sources, minlength=input_start)
    starts = np.cumsum(counts) - counts
    copy_indices = np.arange(sources.size) - np.repeat(starts, counts)
    inverse = np.full((int(counts.max()), input_start), -1, dtype=np.int32)
    inverse[copy_indices, sources[order]] = destinations[order]
    if len(output_indices) != input_start:
        raise RuntimeError("Ragged Mamba output map does not cover the canonical input")
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device=device)
    return RaggedMambaLayout(
        convolution_indices=tensor(convolution_indices),
        convolution_sequences=tensor(convolution_sequences)[None],
        contributors=tensor(inverse),
        scan_indices=tensor(scan_indices),
        output_indices=tensor(output_indices),
        segment_chunks=tensor(segment_chunks),
        root_segments=tensor(root_segments),
        input_tokens=input_start,
        scan_tokens=scan_start,
        replayed_tail_tokens=replayed,
        padding_tokens=padding,
    )


def scan_mamba_ragged_forest(
    mixer: MambaMixer,
    recurrent: torch.Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
) -> torch.Tensor:
    """Scan one canonical forest, retaining autograd through every state fork.

    NRL_SP_MAMBA_SAVE_INTERMEDIATES=1 retains scan intermediates during training.
    It defaults to 0; the scan also disables retention for no-grad forwards.
    """
    save_intermediates = os.environ.get("NRL_SP_MAMBA_SAVE_INTERMEDIATES", "0")
    if save_intermediates not in ("0", "1"):
        raise ValueError("NRL_SP_MAMBA_SAVE_INTERMEDIATES must be exactly '0' or '1'")

    from megatron.core.ssm.mamba_ragged_scan import mamba_chunk_scan_forest

    if recurrent.ndim != 3 or recurrent.shape[1] != 1:
        raise ValueError("Ragged Mamba requires [sequence, 1, recurrent channels]")
    if recurrent.shape[0] < layout.total_len:
        raise ValueError("Ragged Mamba input is shorter than its forest")
    roots = []
    for offset, root in layout.iter_roots():
        lengths = list(root.completion_lens)
        if offset + root.total_len == layout.total_len:
            lengths[-1] += recurrent.shape[0] - layout.total_len
        roots.append((root.prefix_len, tuple(lengths)))
    metadata = ragged_mamba_layout(tuple(roots), mixer.chunk_size, mixer.d_conv, recurrent.device)
    cp = mixer.cp
    dim, groups = cp.d_inner_local_tpcp, cp.ngroups_local_tpcp
    fields = _RaggedGather.apply(
        recurrent, metadata.convolution_indices, metadata.contributors
    ).transpose(0, 1)
    xbc, dt = fields.split([dim + 2 * groups * mixer.d_state, cp.nheads_local_tpcp], -1)
    xbc = (
        causal_conv1d_fn(
            xbc.contiguous().transpose(1, 2),
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            seq_idx=metadata.convolution_sequences,
            activation=mixer.activation,
        )
        .transpose(1, 2)
        .index_select(1, metadata.scan_indices)
        .contiguous()
    )
    dt = dt.index_select(1, metadata.scan_indices).contiguous()
    x, b, c = xbc.split([dim, groups * mixer.d_state, groups * mixer.d_state], -1)
    # The scan kernels accept these strided views: each field's last dimension
    # is contiguous. Retain the shared xbc storage rather than copying fields.
    y = mamba_chunk_scan_forest(
        rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim),
        dt,
        -torch.exp(cp.get_A_log().float()),
        rearrange(b, "b l (g n) -> b l g n", n=mixer.d_state),
        rearrange(c, "b l (g n) -> b l g n", n=mixer.d_state),
        mixer.chunk_size,
        metadata.segment_chunks,
        metadata.root_segments,
        D=(
            rearrange(cp.get_D().float(), "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else cp.get_D()
        ),
        dt_bias=cp.get_dt_bias().float(),
        dt_softplus=True,
        state_dtype=mixer.mamba_training_ssm_states_dtype if MAMBA_HAS_STATE_DTYPE else None,
        save_intermediates=save_intermediates == "1" and mixer.training,
    )
    return (
        rearrange(y, "b l h p -> l b (h p)").contiguous().index_select(0, metadata.output_indices)
    )
