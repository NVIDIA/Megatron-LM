# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tensors of one shared-prefix forward, built from ordinary padded rows.

A row is one sequence of a padded batch: a prompt followed by a completion. A tree is a set of
rows with the same prompt. A pack is the trees of one forward. The packed sequence of a forward
stores, for each tree, the prompt once and then the completion of each row. Each completion is
followed by row padding: zero tokens that extend the row, prompt included, to a multiple of the
padding multiple. Pack padding is the padding after the last tree. It extends the whole packed
sequence to a multiple of the padding multiple.

:func:`build_shared_prefix_inputs` returns the layout of a forward, its packed tokens and
positions, and the index maps between the packed sequence and the rows. The other functions
gather further per-token tensors into the packed order and split a forward over
context-parallel ranks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.datasets.shared_prefix.layout import (
    SharedPrefixPack,
    SharedPrefixTree,
    shared_prefix_topology_multiple,
)


@dataclass(frozen=True, slots=True)
class SharedPrefixIndexMaps:
    """Index tensors of one forward, on the device of its ``input_ids``.

    A packed position is an index into the packed sequence ``[0, layout.total_len)``. A row is
    a row of the forward's ``input_ids``. A real token is a token of a row. Row padding is not
    real. ``C`` is the number of real completion tokens of the forward.

    Attributes:
        token_gather_rows: Row that the token of each packed position is read from. A prompt is
            read from the first row of its tree. Shape ``[layout.total_len]``.
        token_gather_columns: Column that the token of each packed position is read from. It is
            also the position id of the token. Shape ``[layout.total_len]``.
        completion_positions: Packed position of each real completion token. Shape ``[C]``.
        predecessor_positions: Packed position whose output predicts each real completion
            token. For the first completion token of a row it is the last prompt token of the
            tree. Shape ``[C]``.
        completion_scatter_rows: Row of each real completion token. Shape ``[C]``.
        completion_scatter_columns: Column of each real completion token in a next-token
            tensor of shape ``[rows, width - 1]``, where column ``c`` holds the value of token
            ``c + 1``. Shape ``[C]``.
        padding_positions: Packed positions of all row padding.
    """

    token_gather_rows: Tensor
    token_gather_columns: Tensor
    completion_positions: Tensor
    predecessor_positions: Tensor
    completion_scatter_rows: Tensor
    completion_scatter_columns: Tensor
    padding_positions: Tensor


@dataclass(frozen=True, slots=True)
class SharedPrefixForwardInputs:
    """The layout, packed tokens, positions and index maps of one forward.

    The tensors have no pack padding. Their length is ``layout.total_len``.

    Attributes:
        layout: Token layout that the model forward takes as ``shared_prefix_layout``.
        tree_sizes: Number of rows of each tree. The rows of a tree are consecutive.
        packed_input_ids: Packed tokens, in the dtype of ``input_ids``. Row padding is 0.
        position_ids: Position of each packed token. A completion continues the positions of
            its prompt.
        indices: Index maps between the packed sequence and the rows.
    """

    layout: SharedPrefixTree | SharedPrefixPack
    tree_sizes: tuple[int, ...]
    packed_input_ids: Tensor
    position_ids: Tensor
    indices: SharedPrefixIndexMaps


@dataclass(frozen=True, slots=True)
class SharedPrefixContextParallelShard:
    """The part of a padded forward that one context-parallel rank owns.

    Attributes:
        packed_input_ids: Tokens of this rank. Shape ``[padded_total_length / cp_size]``.
        position_ids: Position ids of this rank's tokens. Pack padding has position 0.
        global_token_indices: Index of each of this rank's tokens in the padded packed sequence.
        padded_total_length: Length of the packed sequence with its pack padding.
        padding_multiple: Padding multiple of the forward.
    """

    packed_input_ids: Tensor
    position_ids: Tensor
    global_token_indices: Tensor
    padded_total_length: int
    padding_multiple: int


def _round_up(value: int, multiple: int) -> int:
    """Return the smallest multiple of ``multiple`` that is at least ``value``."""
    return ((value + multiple - 1) // multiple) * multiple


def _is_int(value: object) -> bool:
    """Return whether ``value`` is a Python integer. A boolean is not accepted."""
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_shared_prefix_padding_multiple(
    *, tp_size: int, cp_size: int, padding_multiple: int | None
) -> int:
    """Check a padding multiple against the parallel topology.

    Args:
        tp_size: Tensor-parallel size.
        cp_size: Context-parallel size.
        padding_multiple: Padding multiple chosen by the caller. None selects the multiple of
            the topology, see :func:`shared_prefix_topology_multiple`.

    Returns:
        The padding multiple to use.

    Raises:
        ValueError: If ``padding_multiple`` is neither None nor a positive integer multiple of
            the topology multiple.
    """
    topology_multiple = shared_prefix_topology_multiple(tp_size=tp_size, cp_size=cp_size)
    if padding_multiple is None:
        return topology_multiple
    if not _is_int(padding_multiple):
        raise ValueError(
            f"shared-prefix padding_multiple must be an integer or None, got {padding_multiple!r}"
        )
    if padding_multiple < 1 or padding_multiple % topology_multiple:
        raise ValueError(
            "shared-prefix padding_multiple must be a positive multiple of the topology "
            f"multiple {topology_multiple} (TP={tp_size}, CP={cp_size}), got {padding_multiple}"
        )
    return padding_multiple


def _to_device(values: Tensor, device: torch.device) -> Tensor:
    """Copy a small host tensor to ``device`` without waiting for queued device work."""
    if device.type == "cuda":
        return values.pin_memory().to(device, non_blocking=True)
    return values.to(device)


def _validate_input_ids(input_ids: Tensor) -> None:
    """Check that ``input_ids`` is an integer tensor of shape ``[rows, width]``."""
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [rows, width], got {getattr(input_ids, 'shape', None)}"
        )
    if input_ids.is_floating_point() or input_ids.is_complex() or input_ids.dtype == torch.bool:
        raise ValueError(f"input_ids must have an integer dtype, got {input_ids.dtype}")


def _host_lengths(lengths: Tensor, *, name: str, rows: int, width: int) -> list[int]:
    """Check one per-row length tensor and return its values as Python integers."""
    if not isinstance(lengths, Tensor) or lengths.ndim != 1 or lengths.numel() != rows:
        raise ValueError(f"{name} must have shape [{rows}], got {getattr(lengths, 'shape', None)}")
    if lengths.is_floating_point() or lengths.is_complex() or lengths.dtype == torch.bool:
        raise ValueError(f"{name} must have an integer dtype, got {lengths.dtype}")
    values = lengths.detach().cpu().to(torch.long).tolist()
    if any(value < 0 or value > width for value in values):
        raise ValueError(f"{name} must be within the input_ids width [0, {width}]")
    return values


def build_shared_prefix_inputs(
    input_ids: Tensor,
    *,
    input_lengths: Tensor,
    prompt_lengths: Tensor,
    tree_sizes: Sequence[int],
    padding_multiple: int,
    mtp_loss_group_tree_counts: Sequence[int] = (),
) -> SharedPrefixForwardInputs:
    """Pack the rows of one forward so that the prompt of each tree is stored once.

    The work on the host grows with the number of rows, not with the number of tokens. The
    index maps and the packed tokens are computed on the device of ``input_ids``.

    Args:
        input_ids: Tokens of the rows. Shape ``[rows, width]``, integer dtype.
        input_lengths: Number of tokens of each row, prompt included. Shape ``[rows]``.
        prompt_lengths: Number of prompt tokens of each row. Shape ``[rows]``.
        tree_sizes: Number of rows of each tree. The rows of tree ``t`` are the next
            ``tree_sizes[t]`` consecutive rows. A tree may have one row.
        padding_multiple: Each row, prompt included, is stored with a length that is a multiple
            of this number. :func:`shard_shared_prefix_inputs_for_context_parallel` pads the
            packed sequence to the same multiple and checks it against the parallel topology.
        mtp_loss_group_tree_counts: Sizes of consecutive groups of trees. The loss of the
            multi-token prediction (MTP) layers is normalized per group. When empty, each tree
            is its own group.

    Returns:
        The inputs of the forward. Their ``layout`` is a ``SharedPrefixTree`` when the forward
        has one tree and ``mtp_loss_group_tree_counts`` is empty. Otherwise it is a
        ``SharedPrefixPack``.

    Raises:
        ValueError: If an argument has a wrong shape, dtype or value, if ``tree_sizes`` does
            not cover every row once, if a row has an empty prompt or an empty completion, or
            if two rows of a tree differ in their prompt length or prompt tokens. The message
            names the rows.
    """
    _validate_input_ids(input_ids)
    rows, width = input_ids.shape
    lengths = _host_lengths(input_lengths, name="input_lengths", rows=rows, width=width)
    prompts = _host_lengths(prompt_lengths, name="prompt_lengths", rows=rows, width=width)
    if not _is_int(padding_multiple) or padding_multiple < 1:
        raise ValueError(f"padding_multiple must be a positive integer, got {padding_multiple!r}")
    tree_sizes = tuple(tree_sizes)
    if not tree_sizes or any(not _is_int(size) or size < 1 for size in tree_sizes):
        raise ValueError(f"tree_sizes must hold one positive integer per tree, got {tree_sizes!r}")
    if sum(tree_sizes) != rows:
        raise ValueError(
            f"tree_sizes must cover every row once: they sum to {sum(tree_sizes)}, "
            f"input_ids has {rows} rows"
        )

    # A node is the prompt of a tree or the completion of one row. The unpacking of the node
    # table below names the nine values that each node records.
    nodes: list[tuple[int, ...]] = []
    trees: list[SharedPrefixTree] = []
    num_completion_tokens = num_padding = num_prompt_checks = 0
    first_row = tree_start = 0
    for size in tree_sizes:
        tree_rows = range(first_row, first_row + size)
        prompt = prompts[first_row]
        for row in tree_rows:
            if prompts[row] != prompt:
                raise ValueError(
                    f"rows {first_row} and {row} are in one tree but have prompt lengths "
                    f"{prompt} and {prompts[row]}"
                )
            if not 0 < prompt < lengths[row]:
                raise ValueError(
                    f"row {row} needs a non-empty prompt and a non-empty completion, got "
                    f"prompt length {prompt} and input length {lengths[row]}"
                )
        real_lengths = tuple(lengths[row] - prompt for row in tree_rows)
        stored_lengths = tuple(
            _round_up(lengths[row], padding_multiple) - prompt for row in tree_rows
        )
        trees.append(
            SharedPrefixTree(
                prefix_len=prompt,
                completion_lens=stored_lengths,
                logical_completion_lens=real_lengths,
                padding_multiple=padding_multiple,
            )
        )
        nodes.append((tree_start, prompt, prompt, first_row, 0, tree_start, prompt, first_row, 0))
        start = tree_start + prompt
        for row, stored, real in zip(tree_rows, stored_lengths, real_lengths, strict=True):
            nodes.append((start, stored, real, row, prompt, tree_start, prompt, first_row, 1))
            start += stored
        num_completion_tokens += sum(real_lengths)
        num_padding += sum(stored_lengths) - sum(real_lengths)
        num_prompt_checks += prompt * (size - 1)
        tree_start = start
        first_row += size
    total_length = tree_start
    mtp_loss_group_tree_counts = tuple(mtp_loss_group_tree_counts)
    layout: SharedPrefixTree | SharedPrefixPack
    if len(trees) == 1 and not mtp_loss_group_tree_counts:
        layout = trees[0]
    else:
        layout = SharedPrefixPack(
            tuple(trees), mtp_loss_group_root_counts=mtp_loss_group_tree_counts
        )

    device = input_ids.device
    (
        node_start,  # first packed position of the node
        node_stored,  # number of packed tokens, row padding included
        node_real,  # number of packed tokens without row padding
        node_row,  # row of ``input_ids`` that the tokens are read from
        node_column,  # column of the first token
        node_tree_start,  # first packed position of the node's tree
        node_prompt,  # prompt length of the node's tree
        node_first_row,  # first row of the node's tree
        is_completion,  # 1 for a completion, 0 for a prompt
    ) = _to_device(torch.tensor(nodes, dtype=torch.long), device).unbind(1)
    node_ids = torch.arange(len(nodes), device=device)

    def spans(span_lengths: Tensor, total: int) -> tuple[Tensor, Tensor]:
        """Lay spans of the given per-node lengths end to end.

        Returns, for each of the ``total`` elements, its node and its offset inside that
        node's span.
        """
        owner = torch.repeat_interleave(node_ids, span_lengths, output_size=total)
        first = torch.cumsum(span_lengths, 0) - span_lengths
        return owner, torch.arange(total, device=device) - first[owner]

    # A prompt starts at column 0 and a completion at its prompt length, so the source column
    # of a token is also its position id.
    token_node, token_offset = spans(node_stored, total_length)
    token_gather_columns = node_column[token_node] + token_offset
    completion, completion_offset = spans(node_real * is_completion, num_completion_tokens)
    completion_positions = node_start[completion] + completion_offset
    padding, padding_offset = spans(node_stored - node_real, num_padding)
    indices = SharedPrefixIndexMaps(
        token_gather_rows=node_row[token_node],
        token_gather_columns=token_gather_columns,
        completion_positions=completion_positions,
        predecessor_positions=torch.where(
            completion_offset == 0,
            node_tree_start[completion] + node_prompt[completion] - 1,
            completion_positions - 1,
        ),
        completion_scatter_rows=node_row[completion],
        completion_scatter_columns=node_prompt[completion] + completion_offset - 1,
        padding_positions=node_start[padding] + node_real[padding] + padding_offset,
    )

    # The packed prompt of a tree is read from its first row, so every other row of the tree
    # must hold the same prompt tokens. One comparison on the device costs one host sync.
    is_later_row = is_completion * (node_row != node_first_row)
    checked, column = spans(node_prompt * is_later_row, num_prompt_checks)
    same_prompt = (
        input_ids[node_row[checked], column] == input_ids[node_first_row[checked], column]
    ).all()
    if not bool(same_prompt):
        _raise_prompt_mismatch(input_ids, prompts=prompts, tree_sizes=tree_sizes)

    return SharedPrefixForwardInputs(
        layout=layout,
        tree_sizes=tree_sizes,
        packed_input_ids=_gather_token_aligned(input_ids, indices=indices, padding_value=0),
        position_ids=token_gather_columns.clone(),
        indices=indices,
    )


def _raise_prompt_mismatch(
    input_ids: Tensor, *, prompts: Sequence[int], tree_sizes: Sequence[int]
) -> None:
    """Raise a ValueError that names the first row whose prompt tokens differ from its tree.

    This is the slow path of the prompt check. It compares row by row on the host.
    """
    first_row = 0
    for size in tree_sizes:
        prompt = prompts[first_row]
        expected = input_ids[first_row, :prompt]
        for row in range(first_row + 1, first_row + size):
            if not torch.equal(input_ids[row, :prompt], expected):
                raise ValueError(
                    f"rows {first_row} and {row} are in one tree but their first {prompt} "
                    "tokens, the prompt, differ"
                )
        first_row += size
    raise AssertionError("the prompt check failed but no row differs from its tree")


def _gather_token_aligned(
    source: Tensor, *, indices: SharedPrefixIndexMaps, padding_value: int | float
) -> Tensor:
    """Gather ``source`` into packed order and set row padding to ``padding_value``.

    The caller has checked that every real token lies inside ``source``. Only row padding can
    point past the last column. Its column is clamped for the gather and its value is then
    overwritten.
    """
    columns = indices.token_gather_columns.clamp_max(source.shape[1] - 1)
    packed = source[indices.token_gather_rows, columns]
    return packed.index_fill_(0, indices.padding_positions, padding_value)


def gather_shared_prefix_token_aligned(
    source: Tensor, *, inputs: SharedPrefixForwardInputs, padding_value: int | float = 0
) -> Tensor:
    """Gather a per-token tensor of the rows into the packed order of a forward.

    ``source`` holds one value per token of the rows, for example a loss mask. The result
    follows the packed tokens: the values of a prompt are read from the first row of its tree,
    and row padding gets ``padding_value``. The result has no pack padding.

    Args:
        source: Per-token values of the rows of ``inputs``. Shape ``[rows, source_width]``.
            ``source_width`` must be at least the number of tokens of the longest row.
        inputs: The forward, as built by :func:`build_shared_prefix_inputs`.
        padding_value: Value of row padding.

    Returns:
        The packed values. Shape ``[inputs.layout.total_len]``, in the dtype of ``source``.

    Raises:
        ValueError: If ``source`` is not 2-D, has a different number of rows than the forward,
            or is narrower than the longest row.
    """
    if not isinstance(source, Tensor) or source.ndim != 2:
        raise ValueError(
            "a token-aligned source must have shape [rows, width], got "
            f"{getattr(source, 'shape', None)}"
        )
    rows = sum(inputs.tree_sizes)
    if source.shape[0] != rows:
        raise ValueError(f"a token-aligned source needs {rows} rows, got {source.shape[0]}")
    required_width = max(
        tree.prefix_len + max(tree.logical_completion_lens)
        for _, tree in inputs.layout.iter_roots()
    )
    if source.shape[1] < required_width:
        raise ValueError(
            f"a token-aligned source has width {source.shape[1]}, but the forward reads real "
            f"tokens up to column {required_width - 1}"
        )
    return _gather_token_aligned(source, indices=inputs.indices, padding_value=padding_value)


def _pad_and_select(
    packed: Tensor,
    *,
    padded_total_length: int,
    global_token_indices: Tensor,
    padding_value: int | float,
) -> Tensor:
    """Append pack padding to a packed 1-D tensor and select one rank's tokens."""
    pad_length = padded_total_length - packed.shape[0]
    if pad_length:
        packed = F.pad(packed, (0, pad_length), value=padding_value)
    return packed.index_select(0, global_token_indices)


def shard_shared_prefix_inputs_for_context_parallel(
    inputs: SharedPrefixForwardInputs,
    *,
    cp_rank: int,
    cp_size: int,
    tp_size: int,
    padding_token_id: int = 0,
) -> SharedPrefixContextParallelShard:
    """Pad a forward to its padding multiple and select one context-parallel rank's tokens.

    The padding multiple is the one of ``inputs.layout``. The pack padding follows the last
    tree, so a causal model lets no real token attend to it. The layout of the forward does
    not include it. Rank ``r`` owns chunks ``r`` and ``2 * cp_size - 1 - r`` of the
    ``2 * cp_size`` equal chunks of the padded sequence. This is the split that the model uses
    for every context-parallel forward.

    Args:
        inputs: The forward, as built by :func:`build_shared_prefix_inputs`.
        cp_rank: Context-parallel rank whose tokens are returned.
        cp_size: Context-parallel size.
        tp_size: Tensor-parallel size.
        padding_token_id: Token id of the pack padding. Row padding keeps token id 0.

    Returns:
        The tokens, position ids and global token indices of the rank.

    Raises:
        ValueError: If ``cp_rank`` is not a rank of ``cp_size``, or if the padding multiple of
            the forward is not a multiple of the topology multiple, see
            :func:`shared_prefix_topology_multiple`.
    """
    layout = inputs.layout
    padding_multiple = resolve_shared_prefix_padding_multiple(
        tp_size=tp_size, cp_size=cp_size, padding_multiple=layout.padding_multiple
    )
    if not _is_int(cp_rank) or not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be an integer in [0, {cp_size}), got {cp_rank!r}")
    padded_total_length = _round_up(layout.total_len, padding_multiple)
    global_token_indices = layout.cp_local_indices(
        padded_total_length, cp_size, cp_rank, inputs.packed_input_ids.device
    )
    return SharedPrefixContextParallelShard(
        packed_input_ids=_pad_and_select(
            inputs.packed_input_ids,
            padded_total_length=padded_total_length,
            global_token_indices=global_token_indices,
            padding_value=padding_token_id,
        ),
        position_ids=_pad_and_select(
            inputs.position_ids,
            padded_total_length=padded_total_length,
            global_token_indices=global_token_indices,
            padding_value=0,
        ),
        global_token_indices=global_token_indices,
        padded_total_length=padded_total_length,
        padding_multiple=padding_multiple,
    )


def shard_shared_prefix_token_aligned_for_context_parallel(
    packed: Tensor, *, shard: SharedPrefixContextParallelShard, padding_value: int | float = 0
) -> Tensor:
    """Pad a packed per-token tensor like its forward and select the tokens of a shard.

    Args:
        packed: Per-token values in packed order, for example the result of
            :func:`gather_shared_prefix_token_aligned`. Shape ``[layout.total_len]``. A tensor
            that already has the pack padding is accepted.
        shard: Shard of the same forward, from
            :func:`shard_shared_prefix_inputs_for_context_parallel`.
        padding_value: Value of the pack padding.

    Returns:
        The values of the shard's tokens, in the order of ``shard.global_token_indices``.

    Raises:
        ValueError: If ``packed`` is not 1-D, or if its length cannot be the length of the
            forward that ``shard`` was cut from.
    """
    if not isinstance(packed, Tensor) or packed.ndim != 1:
        raise ValueError(
            f"a packed tensor must have shape [tokens], got {getattr(packed, 'shape', None)}"
        )
    pad_length = shard.padded_total_length - packed.shape[0]
    if not 0 <= pad_length < shard.padding_multiple:
        raise ValueError(
            f"a packed tensor of {packed.shape[0]} tokens does not belong to a forward that is "
            f"padded to {shard.padded_total_length} tokens with multiple {shard.padding_multiple}"
        )
    return _pad_and_select(
        packed,
        padded_total_length=shard.padded_total_length,
        global_token_indices=shard.global_token_indices,
        padding_value=padding_value,
    )
