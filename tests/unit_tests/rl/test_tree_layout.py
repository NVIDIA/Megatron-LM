# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from dataclasses import FrozenInstanceError

import pytest
import torch

from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    SharedPrefixRow,
    build_shared_prefix_layout,
)
from megatron.rl.shared_prefix_tensors import build_tree_attention_allow_mask
from megatron.rl.tree_layout import PackedTreeLayout


def _nested() -> PackedTreeLayout:
    # root -> shared intermediate -> two leaves, with padding in the intermediate.
    return PackedTreeLayout((0, 2, 6, 8), (2, 4, 2, 3), (-1, 0, 1, 1), (2, 2, 2, 3))


def test_nested_paths_positions_and_predecessors_skip_ancestor_padding():
    tree = _nested()
    assert tree.depth() == 2
    assert tree.roots() == (0,)
    assert tree.leaf_nodes() == (2, 3)
    assert tree.ancestors(3) == (0, 1)
    assert tree.is_ancestor(1, 3)
    assert not tree.is_ancestor(2, 3)
    assert not tree.is_ancestor(3, 3)
    assert tree.node_pos_offset() == (0, 2, 4, 4)
    assert tree.position_ids() == (0, 1, 2, 3, 4, 5, 4, 5, 4, 5, 6)
    assert tree.first_predecessors() == (-1, 1, 3, 3)
    assert tree.prev_token_index() == (-1, 0, 1, 2, 3, 4, 3, 6, 3, 8, 9)
    assert tree.padding_positions() == (4, 5)
    assert tree.path_token_indices(2) == (0, 1, 2, 3, 6, 7)
    assert tree.path_token_indices(3) == (0, 1, 2, 3, 8, 9, 10)
    assert tree.total_len == 11
    assert tree.logical_total_len == 9


def test_tree_input_snapshots_are_immutable_and_pickleable():
    import pickle

    starts, lengths, parents = [0, 2], [2, 1], [-1, 0]
    tree = PackedTreeLayout(starts, lengths, parents)
    starts[0], lengths[0], parents[1] = 99, 99, -1
    assert tree.node_start == (0, 2)
    assert tree.node_len == (2, 1)
    assert tree.node_parent == (-1, 0)
    assert pickle.loads(pickle.dumps(tree)) == tree
    with pytest.raises(FrozenInstanceError):
        tree.node_len = (99, 1)


@pytest.mark.parametrize(
    "starts,lengths,parents,logical",
    [
        ((), (), (), ()),
        ((0,), (2,), (), ()),
        ((0, 1), (2, 1), (-1, 0), ()),  # overlap
        ((0, 3), (2, 1), (-1, 0), ()),  # gap
        ((0,), (0,), (-1,), ()),
        ((0, 2), (2, 1), (-1, 1), ()),  # self parent
        ((0, 2), (2, 1), (1, 0), ()),  # forward edge/cycle
        ((0,), (2,), (-2,), ()),
        ((0,), (True,), (-1,), ()),
        ((0,), (2,), (-1.0,), ()),
        ((0,), (2,), (-1,), (3,)),
        ((0,), (2,), (-1,), (0,)),
        ((0, 2), (2, 1), (-1, 0), (1,)),
    ],
)
def test_malformed_trees_are_rejected(starts, lengths, parents, logical):
    with pytest.raises(ValueError):
        PackedTreeLayout(starts, lengths, parents, logical)


def test_concat_rebases_nested_and_independent_parent_links():
    nested = _nested()
    star = PackedTreeLayout.from_shared_prefix(2, (1, 2))
    tree = PackedTreeLayout.concat((nested, star, nested))
    assert tree.roots() == (0, 4, 7)
    assert tree.node_parent == (-1, 0, 1, 1, -1, 4, 4, -1, 7, 8, 8)
    assert tree.node_start == (0, 2, 6, 8, 11, 13, 14, 16, 18, 22, 24)
    assert tree.ancestors(10) == (7, 8)
    assert tree.path_token_indices(9) == tuple(16 + i for i in nested.path_token_indices(2))
    assert tree.prev_token_index()[16] == -1
    assert tree.prev_token_index()[22] == 19  # real intermediate end, not padding
    assert tree.position_ids() == (
        *nested.position_ids(),
        *star.position_ids(),
        *nested.position_ids(),
    )


def test_star_lowering_rejects_deeper_trees_and_retains_root_only_trajectories():
    with pytest.raises(NotImplementedError, match="forest of stars"):
        tuple(_nested().iter_star_roots())
    star = PackedTreeLayout.from_shared_prefix(2, (1, 2))
    dense = PackedTreeLayout((0,), (3,), (-1,))
    assert tuple(PackedTreeLayout.concat((star, dense)).iter_star_roots()) == ((0, (1, 2)), (3, ()))


@pytest.mark.parametrize("node", [-1, 4, True, 1.0])
def test_invalid_node_queries_fail_instead_of_using_python_negative_indexing(node):
    with pytest.raises(ValueError, match="node index"):
        _nested().path_token_indices(node)


def test_attention_isolates_siblings_roots_and_ancestor_padding():
    tree = PackedTreeLayout.concat((_nested(), PackedTreeLayout((0,), (2,), (-1,))))
    mask = build_tree_attention_allow_mask(tree)
    assert mask.dtype == torch.bool
    # Every logical root-to-leaf path is exactly ordinary causal attention.
    for leaf in tree.leaf_nodes():
        path = torch.tensor(tree.path_token_indices(leaf))
        selected = mask[path[:, None], path[None, :]]
        assert torch.equal(selected, torch.ones_like(selected).tril())
    assert mask[8].nonzero().flatten().tolist() == [0, 1, 2, 3, 8]
    assert mask[7].nonzero().flatten().tolist() == [0, 1, 2, 3, 6, 7]
    assert not mask[6:11, 4:6].any()  # intermediate padding is never inherited
    assert not mask[:11, 11:].any() and not mask[11:, :11].any()
    assert mask.diagonal().all()  # physical padding queries have a valid self key


def test_nested_attention_outputs_and_gradients_match_expanded_leaf_paths():
    tree = _nested()
    generator = torch.Generator().manual_seed(17)
    qkv = torch.randn((3, tree.total_len, 4), dtype=torch.float64, generator=generator)
    shared = qkv.clone().requires_grad_()
    dense = qkv.clone().requires_grad_()

    def attend(values, mask):
        q, k, v = values
        logits = (q @ k.T) / 2
        return logits.masked_fill(~mask, -torch.inf).softmax(-1) @ v

    shared_output = attend(shared, build_tree_attention_allow_mask(tree))
    shared_loss, dense_loss = [], []
    for leaf in tree.leaf_nodes():
        path = torch.tensor(tree.path_token_indices(leaf))
        expanded = attend(
            dense[:, path], torch.ones((len(path), len(path)), dtype=torch.bool).tril()
        )
        leaf_positions = torch.arange(
            tree.node_start[leaf], tree.node_start[leaf] + tree.logical_node_len[leaf]
        )
        torch.testing.assert_close(shared_output[leaf_positions], expanded[-len(leaf_positions) :])
        shared_loss.append(shared_output[leaf_positions].square().sum())
        dense_loss.append(expanded[-len(leaf_positions) :].square().sum())
    sum(shared_loss).backward()
    sum(dense_loss).backward()
    torch.testing.assert_close(shared.grad, dense.grad, rtol=1e-12, atol=1e-12)
    assert shared.grad[:, :4].abs().sum() > 0  # both leaves accumulate into ancestors
    assert not shared.grad[:, 4:6].any()


def test_current_star_and_forest_plans_expose_and_use_the_tree_descriptor():
    def star(group, rows):
        return build_shared_prefix_layout(
            [SharedPrefixRow(row, group, (1, 2, 3), length) for row, length in rows],
            sequence_length_pad_multiple=4,
        )

    a, b = star("a", ((0, 2), (1, 4))), star("b", ((2, 1), (3, 3)))
    forest = SharedPrefixForestLayout((a, b), mtp_loss_group_root_counts=(2,))
    assert a.tree_layout.node_parent == (-1, 0, 0)
    assert a.tree_layout.logical_node_len == (3, 2, 4)
    assert forest.tree_layout.node_parent == (-1, 0, 0, -1, 3, 3)
    assert forest.position_ids == forest.tree_layout.position_ids()
    assert forest.physical_total_length == forest.tree_layout.total_len
    predecessors = forest.tree_layout.prev_token_index()
    assert forest.predecessor_positions == tuple(
        predecessors[i] for i in forest.completion_positions
    )
    assert forest.physical_padding_positions == forest.tree_layout.padding_positions()
    assert forest.mtp_loss_group_root_counts == (2,)


def test_legacy_backend_lowering_rejects_interleaved_stars_and_padded_roots():
    interleaved = PackedTreeLayout((0, 2, 4, 5), (2, 2, 1, 1), (-1, -1, 0, 1))
    padded_root = PackedTreeLayout((0, 4), (4, 1), (-1, 0), (2, 1))
    with pytest.raises(NotImplementedError, match="contiguous star storage"):
        tuple(interleaved.iter_star_roots())
    with pytest.raises(NotImplementedError, match="unpadded prompt roots"):
        tuple(padded_root.iter_star_roots())
    # Both remain valid generic tree descriptors and reference attention inputs.
    assert interleaved.path_token_indices(2) == (0, 1, 4)
    assert build_tree_attention_allow_mask(padded_root)[4].nonzero().flatten().tolist() == [0, 1, 4]
