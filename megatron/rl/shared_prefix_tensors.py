# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor materialization for backend-neutral shared-prefix plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixRow,
    _round_up,
)


@dataclass(frozen=True, slots=True)
class SharedPrefixTensorIndices:
    """Device tensors for token gather and completion logprob fan-out/scatter."""

    token_gather_rows: torch.Tensor
    token_gather_columns: torch.Tensor
    completion_positions: torch.Tensor
    predecessor_positions: torch.Tensor
    completion_scatter_columns: torch.Tensor
    physical_padding_positions: torch.Tensor


@dataclass(frozen=True, slots=True)
class SharedPrefixTensorBin:
    """One unpadded shared-prefix star or forest ready for a backend adapter.

    Backends lower ``layout`` to structured attention directly; no dense
    ``[tokens, tokens]`` mask is built.
    """

    layout: SharedPrefixLayout | SharedPrefixForestLayout
    packed_input_ids: torch.Tensor
    position_ids: torch.Tensor
    indices: SharedPrefixTensorIndices


@dataclass(frozen=True, slots=True)
class SharedPrefixContextParallelShard:
    """One standard zigzag CP shard of a padded shared-prefix star.

    ``global_token_indices`` maps the local sequence dimension back to the
    padded global physical order. Branch-physical tokens occupy
    ``[0, tensor_bin.layout.physical_total_length)``; this includes each ordinary
    sequence's zero tail. Any later positions are topology-only padding.
    """

    packed_input_ids: torch.Tensor
    position_ids: torch.Tensor
    global_token_indices: torch.Tensor
    padded_total_length: int
    padding_multiple: int


def resolve_shared_prefix_parallel_topology(
    *, tp_size: object, cp_size: object, sequence_parallel: object
) -> tuple[int, int, bool]:
    """Strictly validate the TP/CP/SP shared-prefix topology.

    Configuration values must already have their semantic Python types. This
    deliberately rejects coercible strings, floats, and booleans for world
    sizes, plus truthy/falsy non-booleans for sequence parallelism.
    """
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
        raise ValueError(
            "shared-prefix tensor_model_parallel_size must be a positive integer, "
            f"got {tp_size!r}"
        )
    if isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size < 1:
        raise ValueError(
            "shared-prefix context_parallel_size must be a positive integer, " f"got {cp_size!r}"
        )
    if not isinstance(sequence_parallel, bool):
        raise ValueError(
            "shared-prefix sequence_parallel must be a boolean, got " f"{sequence_parallel!r}"
        )
    if sequence_parallel != (tp_size > 1):
        raise ValueError(
            "shared-prefix train mode requires sequence_parallel=true exactly "
            f"when TP>1; got TP={tp_size}, sequence_parallel={sequence_parallel}"
        )
    return tp_size, cp_size, sequence_parallel


def get_shared_prefix_physical_alignment(*, tp_size: int, cp_size: int) -> int:
    """Return the topology-minimum physical token quantum for one star.

    TP1 retains the original contract: CP1 is unpadded and CP>1 uses the
    standard two-chunk CP quantum. TP>1 shared-prefix execution requires SP and
    uses the stricter ``2 * TP * CP`` integration quantum so every CP-local
    zigzag sequence can be split evenly across TP sequence-parallel ranks.

    The ``2 * TP`` quantum also applies at CP1, which is stricter than the
    ``TP`` alignment of conventional sequence packing: a TP-aligned
    ``make_sequence_length_divisible_by`` is rejected unless it is a multiple
    of ``2 * TP``. This mirrors the MCore shared-prefix validator
    (``_validate_shared_prefix_physical_length``); relaxing CP1 to ``TP``
    requires changing both together.
    """
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
        raise ValueError(f"tp_size must be a positive integer, got {tp_size!r}")
    if isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size < 1:
        raise ValueError(f"cp_size must be a positive integer, got {cp_size!r}")
    if tp_size == 1:
        return 1 if cp_size == 1 else 2 * cp_size
    return 2 * tp_size * cp_size


def resolve_shared_prefix_physical_padding_multiple(
    *, tp_size: int, cp_size: int, padding_multiple: int | None
) -> int:
    """Resolve explicit packing alignment against the topology minimum.

    Older configs may omit ``make_sequence_length_divisible_by``; those resolve
    to the topology quantum. Explicit values are never interpreted by
    truthiness, so zero and booleans fail instead of silently selecting a
    fallback.
    """
    topology_alignment = get_shared_prefix_physical_alignment(tp_size=tp_size, cp_size=cp_size)
    if padding_multiple is None:
        return topology_alignment
    if isinstance(padding_multiple, bool) or not isinstance(padding_multiple, int):
        raise ValueError(
            "shared-prefix padding_multiple must be an integer or None, got "
            f"{padding_multiple!r}"
        )
    if padding_multiple < 1 or padding_multiple % topology_alignment:
        raise ValueError(
            "shared-prefix padding_multiple must be a positive multiple of "
            f"the TP/CP topology alignment {topology_alignment}, got "
            f"{padding_multiple}"
        )
    return padding_multiple


def get_shared_prefix_context_parallel_indices(
    padded_total_length: int,
    *,
    cp_rank: int,
    cp_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return standard two-chunk causal CP indices in rank-local order.

    Rank ``r`` owns global chunk ``r`` followed by chunk ``2*CP-r-1``.  This
    is the same zigzag sequence ownership used by NeMo-RL's conventional MCore
    path, expressed explicitly so the shared-prefix output fan-out can route
    only scalar log-probabilities instead of gathering vocabulary logits.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if padded_total_length < 1:
        raise ValueError(f"padded_total_length must be positive, got {padded_total_length}")
    if cp_size == 1:
        return torch.arange(padded_total_length, dtype=torch.long, device=device)
    alignment = 2 * cp_size
    if padded_total_length % alignment != 0:
        raise ValueError(
            "shared-prefix CP length must be divisible by 2 * cp_size: "
            f"{padded_total_length} % {alignment} != 0"
        )

    chunk_length = padded_total_length // alignment
    first_chunk = cp_rank
    second_chunk = alignment - cp_rank - 1
    first = torch.arange(
        first_chunk * chunk_length,
        (first_chunk + 1) * chunk_length,
        dtype=torch.long,
        device=device,
    )
    second = torch.arange(
        second_chunk * chunk_length,
        (second_chunk + 1) * chunk_length,
        dtype=torch.long,
        device=device,
    )
    return torch.cat((first, second), dim=0)


def shard_shared_prefix_tensor_bin_for_context_parallel(
    tensor_bin: SharedPrefixTensorBin,
    *,
    cp_rank: int,
    cp_size: int,
    tp_size: int = 1,
    padding_token_id: int = 0,
    padding_multiple: int | None = None,
) -> SharedPrefixContextParallelShard:
    """Pad and shard a materialized star for MCore context parallelism.

    The logical layout remains unpadded.  Padding is appended after all real
    branches, so causality guarantees that it cannot influence a real token; the
    MCore forest adapter masks padded queries/keys and NeMo-RL never selects
    their logits for the loss.
    """
    total_length = tensor_bin.layout.physical_total_length
    resolved_padding_multiple = resolve_shared_prefix_physical_padding_multiple(
        tp_size=tp_size, cp_size=cp_size, padding_multiple=padding_multiple
    )
    padded_total_length = _round_up(total_length, resolved_padding_multiple)
    pad_length = padded_total_length - total_length

    packed_input_ids = tensor_bin.packed_input_ids
    position_ids = tensor_bin.position_ids
    if pad_length:
        packed_input_ids = torch.nn.functional.pad(
            packed_input_ids, (0, pad_length), value=padding_token_id
        )
        position_ids = torch.nn.functional.pad(position_ids, (0, pad_length), value=0)

    global_token_indices = get_shared_prefix_context_parallel_indices(
        padded_total_length, cp_rank=cp_rank, cp_size=cp_size, device=packed_input_ids.device
    )
    return SharedPrefixContextParallelShard(
        packed_input_ids=packed_input_ids.index_select(0, global_token_indices),
        position_ids=position_ids.index_select(0, global_token_indices),
        global_token_indices=global_token_indices,
        padded_total_length=padded_total_length,
        padding_multiple=resolved_padding_multiple,
    )


def materialize_shared_prefix_layout(
    input_ids: torch.Tensor,
    *,
    input_lengths: torch.Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
) -> SharedPrefixTensorBin:
    """Gather a planned star or forest from padded ``input_ids`` rows.

    Host work is proportional to the number of layout nodes, not tokens: one
    small node table, copied once (pinned and non-blocking on CUDA), drives
    every token-level index map on ``input_ids.device``. Source-row validation
    also runs on that device and costs a single synchronization.

    Args:
        input_ids: Integer token tensor with shape ``[batch, sequence]``.
        input_lengths: Unpadded source-row lengths with shape ``[batch]``.
        layout: Planner output whose row indices address ``input_ids``.

    Returns:
        Unpadded packed tokens, prefix-continued positions, and tensorized
        gather/fan-out/scatter indices on ``input_ids.device``.

    Raises:
        ValueError: If a source row's unpadded length differs from the planned
            prompt plus completion, or its prompt tokens no longer match the
            planned exact prompt.
    """
    _validate_input_ids(input_ids)
    batch_size, sequence_width = input_ids.shape
    _validate_length_vector_metadata(input_lengths, name="input_lengths", batch_size=batch_size)

    # One entry per tree node: packed start, physical length, logical length,
    # source row, first source column, root packed start, root prompt length,
    # offset of the root prompt in ``prompt_tokens``, and whether it is a branch.
    nodes: list[tuple[int, ...]] = []
    prompt_tokens: list[int] = []
    num_padding = num_prompt_checks = 0
    for offset, root in layout.iter_roots():
        prompt_length = root.prompt_length
        if prompt_length > sequence_width:
            raise ValueError(
                f"shared-prefix prompt of {prompt_length} tokens exceeds the input_ids "
                f"width {sequence_width}"
            )
        prompt_offset = len(prompt_tokens)
        prompt_tokens.extend(root.prompt_token_ids)
        root_row = root.row_indices[0]
        nodes.append(
            (offset, prompt_length, prompt_length, root_row, 0, offset, prompt_length, 0, 0)
        )
        for row_index, start, physical, logical in zip(
            root.row_indices,
            root.branch_starts,
            root.physical_completion_lengths,
            root.completion_lengths,
            strict=True,
        ):
            if row_index >= batch_size:
                raise ValueError(
                    f"shared-prefix layout references row {row_index} outside input_ids"
                )
            nodes.append(
                (
                    offset + start,
                    physical,
                    logical,
                    row_index,
                    prompt_length,
                    offset,
                    prompt_length,
                    prompt_offset,
                    1,
                )
            )
            num_padding += physical - logical
            num_prompt_checks += prompt_length

    device = input_ids.device
    (
        node_start,
        node_physical,
        node_logical,
        node_row,
        node_column,
        root_start,
        root_prompt,
        prompt_offset,
        is_branch,
    ) = _to_device(torch.tensor(nodes, dtype=torch.long), device).unbind(1)
    node_ids = torch.arange(len(nodes), device=device)

    def spans(lengths: torch.Tensor, total: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the owning node and local offset of every concatenated span element."""
        owner = torch.repeat_interleave(node_ids, lengths, output_size=total)
        first = torch.cumsum(lengths, 0) - lengths
        return owner, torch.arange(total, device=device) - first[owner]

    # Prompt nodes have position offset 0 and branch nodes continue from the
    # prompt, so a node's first source column is also its position offset.
    segment, local = spans(node_physical, layout.physical_total_length)
    token_gather_columns = node_column[segment] + local
    completion, completion_offset = spans(node_logical * is_branch, sum(layout.completion_lengths))
    completion_positions = node_start[completion] + completion_offset
    padding, padding_offset = spans(node_physical - node_logical, num_padding)
    indices = SharedPrefixTensorIndices(
        token_gather_rows=node_row[segment],
        token_gather_columns=token_gather_columns,
        completion_positions=completion_positions,
        predecessor_positions=torch.where(
            completion_offset == 0,
            root_start[completion] + root_prompt[completion] - 1,
            completion_positions - 1,
        ),
        completion_scatter_columns=root_prompt[completion] + completion_offset - 1,
        physical_padding_positions=(node_start[padding] + node_logical[padding] + padding_offset),
    )

    # Every source row must still hold its planned prompt and exact length.
    # Gathers stay in bounds even for invalid sources, which then fail below.
    lengths = input_lengths.to(device=device, dtype=torch.long)
    checked, column = spans(root_prompt * is_branch, num_prompt_checks)
    expected_prompts = _to_device(torch.tensor(prompt_tokens, dtype=input_ids.dtype), device)
    valid = torch.stack(
        (
            ((lengths >= 0) & (lengths <= sequence_width)).all(),
            ((lengths[node_row] == root_prompt + node_logical) | (is_branch == 0)).all(),
            (
                input_ids[node_row[checked], column]
                == expected_prompts[prompt_offset[checked] + column]
            ).all(),
        )
    )
    if not bool(valid.all()):
        _raise_source_mismatch(input_ids, input_lengths=input_lengths, layout=layout)

    return SharedPrefixTensorBin(
        layout=layout,
        packed_input_ids=_gather_token_aligned_tensor(input_ids, indices=indices, padding_value=0),
        position_ids=token_gather_columns.clone(),
        indices=indices,
    )


def _raise_source_mismatch(
    input_ids: torch.Tensor,
    *,
    input_lengths: torch.Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
) -> None:
    """Report the first source row that no longer satisfies ``layout`` (slow path)."""
    batch_size, sequence_width = input_ids.shape
    lengths_cpu = _validate_length_vector(
        input_lengths, name="input_lengths", batch_size=batch_size, sequence_width=sequence_width
    ).tolist()
    for _, root in layout.iter_roots():
        expected_prompt = list(root.prompt_token_ids)
        for row_index, completion_length in zip(
            root.row_indices, root.completion_lengths, strict=True
        ):
            required_length = root.prompt_length + completion_length
            if required_length != lengths_cpu[row_index]:
                raise ValueError(
                    f"layout completion length differs from source row {row_index}: "
                    f"layout requires {required_length} tokens, row has "
                    f"{lengths_cpu[row_index]}"
                )
            if input_ids[row_index, : root.prompt_length].tolist() != expected_prompt:
                raise ValueError(
                    f"source prompt for row {row_index} differs from the planned exact prompt"
                )
    raise AssertionError("shared-prefix source validation failed without a mismatching row")


def materialize_shared_prefix_token_aligned_tensor(
    source: torch.Tensor, *, tensor_bin: SharedPrefixTensorBin, padding_value: int | float = 0
) -> torch.Tensor:
    """Gather a token-aligned source tensor into the star's physical order.

    This is used for metadata such as MTP loss masks that must follow exactly
    the same prompt deduplication, branch padding, and row permutation as token
    IDs.  The returned tensor is global and unsharded; topology-only padding is
    applied by the caller together with the model input.
    """
    if source.ndim != 2:
        raise ValueError("shared-prefix token-aligned source must have shape [batch, sequence]")
    if source.shape[0] <= max(tensor_bin.layout.row_indices):
        raise ValueError("shared-prefix token-aligned source is missing a referenced row")
    return _gather_token_aligned_tensor(
        source, indices=tensor_bin.indices, padding_value=padding_value
    )


def _gather_token_aligned_tensor(
    source: torch.Tensor, *, indices: SharedPrefixTensorIndices, padding_value: int | float
) -> torch.Tensor:
    """Apply the same physical gather and tail-padding rule to tokens and metadata.

    Columns past the source width (only branch padding tails reach them) are
    clamped for the gather and then overwritten, so no padded copy of the
    source is made.
    """
    columns = indices.token_gather_columns
    width = source.shape[1]
    packed = source[indices.token_gather_rows, columns.clamp_max(width - 1)]
    packed.masked_fill_(columns >= width, padding_value)
    packed[indices.physical_padding_positions] = padding_value
    return packed


def _to_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a small host tensor without waiting for queued device work."""
    if device.type == "cuda":
        return values.pin_memory().to(device, non_blocking=True)
    return values.to(device)


def _validate_input_ids(input_ids: torch.Tensor) -> None:
    """Validate the source token tensor without changing its device or dtype."""
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {getattr(input_ids, "shape", None)}"
        )
    if input_ids.is_floating_point() or input_ids.is_complex() or input_ids.dtype == torch.bool:
        raise ValueError(f"input_ids must have an integer dtype, got {input_ids.dtype}")


def _validate_batch_inputs(
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: Sequence[str | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate conventional batch metadata and return CPU integer lengths."""
    _validate_input_ids(input_ids)
    batch_size, sequence_width = input_ids.shape
    if isinstance(group_ids, (str, bytes)):
        raise TypeError("group_ids must be a sequence of per-row IDs, not one string")
    if len(group_ids) != batch_size:
        raise ValueError(f"group_ids must have {batch_size} entries, got {len(group_ids)}")

    lengths_cpu = _validate_length_vector(
        input_lengths, name="input_lengths", batch_size=batch_size, sequence_width=sequence_width
    )
    prompt_lengths_cpu = _validate_length_vector(
        prompt_lengths, name="prompt_lengths", batch_size=batch_size, sequence_width=sequence_width
    )
    if bool(torch.any(prompt_lengths_cpu > lengths_cpu).item()):
        raise ValueError("prompt_lengths must satisfy 0 <= prompt_length <= input_length")
    return lengths_cpu, prompt_lengths_cpu


def _validate_length_vector_metadata(lengths: torch.Tensor, *, name: str, batch_size: int) -> None:
    """Validate one source-length vector's shape and dtype without a device sync."""
    if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1 or lengths.numel() != batch_size:
        raise ValueError(
            f"{name} must have shape [{batch_size}], got {getattr(lengths, "shape", None)}"
        )
    if lengths.is_floating_point() or lengths.is_complex() or lengths.dtype == torch.bool:
        raise ValueError(f"{name} must have an integer dtype, got {lengths.dtype}")


def _validate_length_vector(
    lengths: torch.Tensor, *, name: str, batch_size: int, sequence_width: int
) -> torch.Tensor:
    """Validate and copy one source-length vector to CPU long."""
    _validate_length_vector_metadata(lengths, name=name, batch_size=batch_size)
    lengths_cpu = lengths.detach().cpu().to(torch.long)
    if bool(torch.any(lengths_cpu < 0).item()) or bool(
        torch.any(lengths_cpu > sequence_width).item()
    ):
        raise ValueError(f"{name} must be within input_ids width [0, {sequence_width}]")
    return lengths_cpu


def build_shared_prefix_rows(
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: Sequence[str | None],
) -> list[SharedPrefixRow]:
    """Build validated CPU planner rows from conventional batch tensors."""
    lengths_cpu, prompt_lengths_cpu = _validate_batch_inputs(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
    )
    input_length_values = lengths_cpu.tolist()
    prompt_length_values = prompt_lengths_cpu.tolist()
    # Only prompt columns are read, so copy just those to the host.
    prompt_tokens = input_ids[:, : max(prompt_length_values, default=0)].detach().cpu().tolist()
    rows: list[SharedPrefixRow] = []
    for row_index, (input_length, prompt_length) in enumerate(
        zip(input_length_values, prompt_length_values, strict=True)
    ):
        group_id = group_ids[row_index]
        if group_id is not None and not isinstance(group_id, str):
            raise TypeError(f"shared-prefix group ID for row {row_index} must be str or None")
        rows.append(
            SharedPrefixRow(
                row_index=row_index,
                group_id=group_id,
                prompt_token_ids=tuple(prompt_tokens[row_index][:prompt_length]),
                completion_length=input_length - prompt_length,
            )
        )
    return rows
