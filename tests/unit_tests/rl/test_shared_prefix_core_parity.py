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

"""The planner-side padding quantum and CP ownership agree with MCore execution.

megatron.rl must stay importable without megatron.core, so the two sides keep
separate implementations; this test pins them together instead of an import.
"""

import random

import pytest
import torch

from megatron.rl.shared_prefix_packing import pack_shared_prefix_groups, plan_shared_prefix_bins
from megatron.rl.shared_prefix_tensors import (
    build_shared_prefix_rows,
    get_shared_prefix_context_parallel_indices,
    materialize_shared_prefix_layout,
    resolve_shared_prefix_physical_padding_multiple,
    shard_shared_prefix_tensor_bin_for_context_parallel,
)

core_shared_prefix = pytest.importorskip("megatron.core.models.hybrid.shared_prefix")
core_layouts = pytest.importorskip("megatron.core.models.hybrid.shared_prefix_layout")


def _lower(layout, padding_multiple):
    """Lower a planner layout exactly as the NeMo-RL adapter does."""
    tree = layout.tree_layout
    roots = tuple(
        core_layouts.SharedPrefixLayout(
            prefix_len=tree.node_len[root],
            completion_lens=tuple(tree.node_len[child] for child in children),
            logical_completion_lens=tuple(tree.logical_node_len[child] for child in children),
            padding_multiple=padding_multiple,
        )
        for root, children in tree.iter_star_roots()
    )
    return roots[0] if len(roots) == 1 else core_layouts.SharedPrefixForestLayout(roots)


def _layouts(rng, multiple):
    for _ in range(10):
        rows, prompt_lengths, group_ids = [], [], []
        for group in range(rng.randint(1, 3)):
            prompt = [rng.randrange(1, 50) for _ in range(rng.randint(1, 9))]
            for _ in range(rng.randint(2, 4)):
                rows.append(prompt + [rng.randrange(1, 50) for _ in range(rng.randint(1, 13))])
                prompt_lengths.append(len(prompt))
                group_ids.append(f"g{group}")
        input_ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
        input_lengths = torch.tensor(list(map(len, rows)))
        planner_rows = build_shared_prefix_rows(
            input_ids=input_ids,
            input_lengths=input_lengths,
            prompt_lengths=torch.tensor(prompt_lengths),
            group_ids=group_ids,
        )
        plan = plan_shared_prefix_bins(
            planner_rows, bin_capacity=64 * multiple, sequence_length_pad_multiple=multiple
        )
        packed = pack_shared_prefix_groups(
            plan.shared_bins,
            bin_capacity=512 * multiple,
            dense_capacity=None,
            padding_multiple=multiple,
        )
        for layout in (*plan.shared_bins, *packed):
            yield input_ids, input_lengths, layout


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_planner_padding_and_cp_ownership_match_mcore(tp_size, cp_size):
    multiple = resolve_shared_prefix_physical_padding_multiple(
        tp_size=tp_size, cp_size=cp_size, padding_multiple=None
    )
    for input_ids, input_lengths, layout in _layouts(
        random.Random(tp_size * 10 + cp_size), multiple
    ):
        tensor_bin = materialize_shared_prefix_layout(
            input_ids, input_lengths=input_lengths, layout=layout
        )
        core_layout = _lower(layout, multiple)
        assert core_layout.total_len == layout.physical_total_length
        assert torch.equal(core_layout.position_ids("cpu"), tensor_bin.position_ids)
        shard = shard_shared_prefix_tensor_bin_for_context_parallel(
            tensor_bin, cp_rank=0, cp_size=cp_size, tp_size=tp_size, padding_multiple=multiple
        )
        # MCore accepts exactly the padded length the planner side produces.
        core_shared_prefix._validate_shared_prefix_physical_length(
            core_layout,
            shard.padded_total_length,
            tp_size=tp_size,
            cp_size=cp_size,
            sequence_parallel=tp_size > 1,
        )
        if cp_size > 1:
            for rank in range(cp_size):
                assert torch.equal(
                    get_shared_prefix_context_parallel_indices(
                        shard.padded_total_length, cp_rank=rank, cp_size=cp_size
                    ),
                    core_layouts.SharedPrefixLayout.cp_local_indices(
                        shard.padded_total_length, cp_size, rank, "cpu"
                    ),
                )
