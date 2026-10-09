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

"""Quadratic reference oracles and planning helpers for shared-prefix tests.

Production code lowers layouts to structured attention and never builds these
``[tokens, tokens]`` masks.
"""

from collections.abc import Sequence

import torch

from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    plan_shared_prefix_bins,
)
from megatron.rl.shared_prefix_tensors import (
    SharedPrefixTensorBin,
    build_shared_prefix_rows,
    materialize_shared_prefix_layout,
)
from megatron.rl.tree_layout import PackedTreeLayout


def build_tree_attention_allow_mask(
    layout: PackedTreeLayout, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Reference causal attention for arbitrary-depth token-span trees.

    Queries can see their own causal node history and real tokens in strict
    ancestor nodes. Ancestor padding, sibling branches, and other roots are
    invisible. Each physical padding query retains self-attention, avoiding an
    empty row. Peak memory is one boolean ``[T, T]`` mask plus a same-size
    temporary.
    """
    node_ids = torch.tensor(layout.segment_ids(), dtype=torch.long, device=device)
    ancestry = torch.zeros((layout.num_nodes, layout.num_nodes), dtype=torch.bool, device=device)
    for node in range(layout.num_nodes):
        ancestry[node, list(layout.ancestors(node))] = True
    real_keys = torch.ones(layout.total_len, dtype=torch.bool, device=device)
    real_keys[list(layout.padding_positions())] = False
    mask = ancestry.index_select(0, node_ids).index_select(1, node_ids)
    mask &= real_keys
    positions = torch.arange(layout.total_len, device=device)
    mask |= (node_ids[:, None] == node_ids[None, :]) & (positions[None, :] <= positions[:, None])
    return mask


def build_star_attention_allow_mask(
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Dense reference mask for a planned star or forest."""
    return build_tree_attention_allow_mask(layout.tree_layout, device=device)


def plan_and_materialize(
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: Sequence[str | None],
    bin_capacity: int,
    max_completions_per_bin: int = 16,
    sequence_length_pad_multiple: int = 1,
) -> tuple[tuple[SharedPrefixTensorBin, ...], tuple[int, ...]]:
    """Plan exact-prompt stars from a padded batch and materialize each one."""
    rows = build_shared_prefix_rows(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
    )
    plan = plan_shared_prefix_bins(
        rows,
        bin_capacity=bin_capacity,
        max_completions_per_bin=max_completions_per_bin,
        sequence_length_pad_multiple=sequence_length_pad_multiple,
    )
    tensor_bins = tuple(
        materialize_shared_prefix_layout(input_ids, input_lengths=input_lengths, layout=layout)
        for layout in plan.shared_bins
    )
    return tensor_bins, plan.fallback_row_indices
