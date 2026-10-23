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

"""Quadratic reference oracles and shared builders for shared-prefix tests.

Production code lowers layouts to structured attention and never builds these
``[tokens, tokens]`` masks. Import this module by its full package path, so that
pytest does not load a second copy under another name.
"""

import contextlib
from collections.abc import Mapping

import torch

from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixPlan,
    SharedPrefixRow,
    pack_shared_prefix_groups,
)
from megatron.rl.shared_prefix_tensors import build_shared_prefix_rows
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


def row(index, *, group="group", prompt=(10, 11, 12), completion=2) -> SharedPrefixRow:
    """One planner row; the defaults share a prompt with every other default row."""
    return SharedPrefixRow(index, group, prompt, completion)


def pack(layouts, bin_capacity, dense_capacity=None, padding_multiple=1):
    kwargs = dict(dense_capacity=dense_capacity, padding_multiple=padding_multiple)
    return pack_shared_prefix_groups(layouts, bin_capacity=bin_capacity, **kwargs)


def nested_tree() -> PackedTreeLayout:
    """Root -> shared intermediate -> two leaves, with padding in the intermediate."""
    return PackedTreeLayout((0, 2, 6, 8), (2, 4, 2, 3), (-1, 0, 1, 1), (2, 2, 2, 3))


def batch_rows(input_ids, lengths, prompts, group_ids):
    """``build_shared_prefix_rows`` on tensor-like inputs; ``group_ids`` may be a string."""
    kwargs = dict(input_lengths=torch.as_tensor(lengths), prompt_lengths=torch.as_tensor(prompts))
    return build_shared_prefix_rows(input_ids=input_ids, group_ids=list(group_ids), **kwargs)


def covered_rows(plan_or_units) -> list[int]:
    """Sorted rows a plan executes; equal to ``range(n)`` only if each row appears once."""
    if isinstance(plan_or_units, SharedPrefixPlan):
        shared = [index for layout in plan_or_units.shared_bins for index in layout.row_indices]
        return sorted(shared + list(plan_or_units.fallback_row_indices))
    return sorted(index for unit in plan_or_units for index in unit.row_indices)


def check_layout_matches_tree(layout) -> None:
    """A planned layout's index maps must agree with its generic tree descriptor."""
    tree, prev = layout.tree_layout, layout.tree_layout.prev_token_index()
    assert layout.position_ids == tree.position_ids()
    assert layout.physical_total_length == tree.total_len
    assert layout.predecessor_positions == tuple(prev[i] for i in layout.completion_positions)
    assert layout.physical_padding_positions == tree.padding_positions()


@contextlib.contextmanager
def note(**context):
    """Name the generated case (seed, iteration, layout) in any failure raised inside."""
    try:
        yield
    except Exception as error:
        error.add_note(f"case: {context}")
        raise


def observe(value, path: str):
    """Follow a dotted path of ``len``, integer indices, mapping keys and attributes.

    ``"a, b"`` observes both paths as a tuple.
    """
    if "," in path:
        return tuple(observe(value, part.strip()) for part in path.split(","))
    for part in filter(None, path.split(".")):
        if part == "len":
            value = len(value)
        elif part.isdigit():
            value = value[int(part)]
        elif isinstance(value, Mapping):
            value = value[part]
        else:
            value = getattr(value, part)
    return value


def assert_observed(value, expected) -> None:
    """Check each ``path: expected`` pair, tensors exactly with dtype, and report every mismatch.

    A non-mapping ``expected`` is compared with ``value`` itself.
    """
    errors = []
    for path, want in (expected if isinstance(expected, Mapping) else {"": expected}).items():
        try:
            got = observe(value, path)
            if isinstance(want, torch.Tensor):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            else:
                assert got == want, f"got {got!r}, expected {want!r}"
        except Exception as error:
            errors.append(f"{path!r}: {type(error).__name__}: {error}")
    assert not errors, "\n".join(errors)
