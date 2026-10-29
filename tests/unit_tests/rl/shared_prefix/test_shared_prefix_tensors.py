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

"""Materialized stars and forests: tensor examples, dense-attention parity and CP shards."""

import random
from functools import partial

import pytest
import torch
from torch import tensor

from megatron.rl.shared_prefix_dense_bins import share_prefixes_in_dense_training_bins
from megatron.rl.shared_prefix_execution import SharedPrefixExecutionUnit
from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    build_shared_prefix_layout,
    plan_shared_prefix_bins,
)
from megatron.rl.shared_prefix_tensors import (
    get_shared_prefix_context_parallel_indices,
    get_shared_prefix_physical_alignment,
    materialize_shared_prefix_layout,
    materialize_shared_prefix_token_aligned_tensor,
    resolve_shared_prefix_physical_padding_multiple,
    shard_shared_prefix_tensor_bin_for_context_parallel,
)
from megatron.rl.tree_layout import PackedTreeLayout
from tests.unit_tests.rl.shared_prefix.shared_prefix_oracles import (
    assert_observed,
    batch_rows,
    build_star_attention_allow_mask,
    build_tree_attention_allow_mask,
    check_layout_matches_tree,
    covered_rows,
    nested_tree,
    note,
    pack,
    row,
)

VOCAB, WIDTH, MAX_POSITION = 50, 8, 128


def _cp_shards(tensor_bin, cp_size, **kwargs):
    shard = shard_shared_prefix_tensor_bin_for_context_parallel
    return [shard(tensor_bin, cp_rank=rank, cp_size=cp_size, **kwargs) for rank in range(cp_size)]


def _unshard(shards):
    """Scatter every rank's tokens and positions back to global packed order."""
    tokens, positions = torch.empty((2, shards[0].padded_total_length), dtype=torch.long)
    for shard in shards:
        tokens[shard.global_token_indices] = shard.packed_input_ids
        positions[shard.global_token_indices] = shard.position_ids
    return tokens, positions


def _mask(text):
    """Square bool mask from space-separated rows of ``0``/``1``, right-padded with zeros."""
    rows = text.split()
    return tensor([[char == "1" for char in line.ljust(len(rows), "0")] for line in rows])


def _star_bins(input_ids, lengths, prompts, group_ids, capacity, multiple=1, cp_size=1, **cp):
    """Plan and materialize stars from a padded batch; mask and CP-shard the first one."""
    input_ids, lengths = tensor(input_ids), tensor(lengths)
    kwargs = dict(bin_capacity=capacity, sequence_length_pad_multiple=multiple)
    plan = plan_shared_prefix_bins(batch_rows(input_ids, lengths, prompts, group_ids), **kwargs)
    materialize = partial(materialize_shared_prefix_layout, input_ids, input_lengths=lengths)
    bins = [materialize(layout=layout) for layout in plan.shared_bins]
    mask = build_star_attention_allow_mask(bins[0].layout)
    result = dict(bins=bins, bin=bins[0], fallback=plan.fallback_row_indices, mask=mask)
    return result | dict(shards=_cp_shards(bins[0], cp_size, **cp))


def _padded_branch_tails():
    batch = [[10, 11, 12, 20, 21, 91, 92], [10, 11, 12, 30, 31, 32, 33]]
    result = _star_bins(batch, [5, 7], [3, 3], "gg", 16, 4, cp_size=2, padding_multiple=8)
    tensor_bin = result["bin"]
    # A route surrogate over the router's token and RoPE position inputs. Prompt routes
    # count twice; every per-row physical tail route counts once.
    star_routes = (tensor_bin.packed_input_ids + tensor_bin.position_ids) % 4
    weights = tensor([2] * 3 + [1] * 10)
    star_counts = torch.zeros(4, dtype=torch.long).scatter_add_(0, star_routes, weights)
    dense_tokens = tensor([[10, 11, 12, 20, 21, 0, 0, 0], [10, 11, 12, 30, 31, 32, 33, 0]])
    dense_counts = torch.bincount(((dense_tokens + torch.arange(8)) % 4).flatten(), minlength=4)
    source = tensor([[0, 0, 0, 1, 1, 7, 7], [0, 0, 0, 1, 1, 1, 1]])
    mtp_mask = materialize_shared_prefix_token_aligned_tensor(source, tensor_bin=tensor_bin)
    return result | dict(routes=(star_counts, dense_counts), mtp_mask=mtp_mask)


def _tp_sp_cp_padding():
    """M-aligned branches and the final star tail remain distinct metadata."""
    batch = [[10, 11, 12, 13, 20, 21, 22, 0, 0], [10, 11, 12, 13, 30, 31, 32, 33, 0]]
    batch.append([10, 11, 12, 13, 40, 41, 42, 43, 44])
    cp = dict(cp_size=2, tp_size=2, padding_multiple=16)
    result = _star_bins(batch, [7, 8, 9], [4, 4, 4], "ggg", 64, 16, **cp)
    shards = result["shards"]
    summary = [(s.padding_multiple, s.padded_total_length, len(s.packed_input_ids)) for s in shards]
    union = torch.cat([shard.global_token_indices for shard in shards]).sort().values
    return result | dict(summary=summary, union=union, topology_tail=_unshard(shards)[0][40:])


def _singleton_fallback_forest():
    pad4 = dict(sequence_length_pad_multiple=4)
    star = [row(i, group="s", prompt=(10, 11)) for i in (0, 1)]
    shared = build_shared_prefix_layout(star, **pad4)
    single = [row(2, group="f", prompt=(40, 41), completion=1)]
    fallback = build_shared_prefix_layout(single, allow_singleton=True, **pad4)
    forest = SharedPrefixForestLayout((shared, fallback), mtp_loss_group_root_counts=(1, 1))
    input_ids = tensor([[10, 11, 20, 21], [10, 11, 30, 31], [40, 41, 42, 0]])
    lengths = tensor([4, 4, 3])
    tensor_bin = materialize_shared_prefix_layout(input_ids, input_lengths=lengths, layout=forest)
    return {"mask": build_star_attention_allow_mask(forest), "bin": tensor_bin}


def _dense_bin_reconstruction():
    input_ids, lengths = tensor([[1, 2, 3, 8], [4, 5, 6, 9], [1, 2, 3, 10]]), tensor([4, 4, 4])
    dense = (SharedPrefixExecutionUnit((0, 1, 2), None, 12),)
    kwargs = dict(costs=(4, 4, 4), padding_multiple=1, bin_capacity=12)
    planned_rows = batch_rows(input_ids, lengths, [3, 3, 3], "aba")
    (unit,) = share_prefixes_in_dense_training_bins(planned_rows, dense, **kwargs)
    layout = unit.shared_layout
    tensor_bin = materialize_shared_prefix_layout(input_ids, input_lengths=lengths, layout=layout)
    return {"unit": unit, "mask": build_star_attention_allow_mask(tensor_bin.layout)}


_padding_multiple = partial(resolve_shared_prefix_physical_padding_multiple, tp_size=2, cp_size=2)
_PADDED_IDS = [[10, 11, 12, 20, 21, -1], [10, 11, 12, 30, 31, 32], [40, 41, 50, 51, -1, -1]]

TENSOR_EXAMPLES = {
    "star-from-padded-batch": (
        lambda: _star_bins(_PADDED_IDS, [5, 6, 4], [3, 3, 2], ["s", "s", "singleton"], 16),
        {
            "bins.len": 1,
            "fallback": (2,),
            "bin.layout.row_indices": (1, 0),
            "bin.layout.completion_scatter_rows": (1, 1, 1, 0, 0),
            "bin.packed_input_ids": tensor([10, 11, 12, 30, 31, 32, 20, 21]),
            "bin.position_ids": tensor([0, 1, 2, 3, 4, 5, 3, 4]),
            "bin.indices.token_gather_rows": tensor([1, 1, 1, 1, 1, 1, 0, 0]),
            "bin.indices.token_gather_columns": tensor([0, 1, 2, 3, 4, 5, 3, 4]),
            "bin.indices.completion_positions": tensor([3, 4, 5, 6, 7]),
            "bin.indices.predecessor_positions": tensor([2, 3, 4, 2, 6]),
            "bin.indices.completion_scatter_columns": tensor([2, 3, 4, 2, 3]),
            # Row 1's branch (3-5) and row 0's branch (6-7) see the prompt (0-2), never each other.
            "mask": _mask("1 11 111 1111 11111 111111 1110001 11100011"),
        },
    ),
    "group-id-and-exact-prompt-both-required": (
        lambda: _star_bins(
            [[1, 2, 3], [1, 2, 4], [1, 9, 5], [1, 2, 6]], [3] * 4, [2] * 4, "abaa", 8
        ),
        {"bins.len": 1, "bin.layout.row_indices": (0, 3), "fallback": (1, 2)},
    ),
    "cp-zigzag-indices": (
        lambda: [
            get_shared_prefix_context_parallel_indices(16, cp_rank=r, cp_size=2) for r in (0, 1)
        ],
        {"0": tensor([0, 1, 2, 3, 12, 13, 14, 15]), "1": tensor([4, 5, 6, 7, 8, 9, 10, 11])},
    ),
    "cp-shard-pads-after-real-tokens": (
        lambda: _star_bins(
            [[10, 11, 12, 20, 0], [10, 11, 12, 30, 31]], [4, 5], [3, 3], "gg", 8, cp_size=2
        ),
        {
            "bin.layout.total_length": 6,
            "shards.0.padded_total_length": 8,
            "shards.1.padded_total_length": 8,
            "shards.0.global_token_indices": tensor([0, 1, 6, 7]),
            "shards.1.global_token_indices": tensor([2, 3, 4, 5]),
            "shards.0.packed_input_ids": tensor([10, 11, 0, 0]),
            "shards.1.packed_input_ids": tensor([12, 30, 31, 20]),
        },
    ),
    "padded-branch-tails": (
        _padded_branch_tails,
        {
            "bin.layout.completion_lengths": (4, 2),
            "bin.layout.physical_completion_lengths": (5, 5),
            "bin.layout.physical_total_length": 13,
            "bin.packed_input_ids": tensor([10, 11, 12, 30, 31, 32, 33, 0, 20, 21, 0, 0, 0]),
            "bin.position_ids": tensor([0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7]),
            "bin.indices.physical_padding_positions": tensor([7, 10, 11, 12]),
            "routes.0": tensor([2, 4, 5, 5]),  # star routes equal dense routes
            "routes.1": tensor([2, 4, 5, 5]),
            # Branch tails 7 and 10-12 are padding; each branch sees only the prompt and itself.
            "mask": _mask(
                "1 11 111 1111 11111 111111 1111111 11111111"
                " 111000001 1110000011 11100000111 111000001111 1110000011111"
            ),
            # The token-aligned MTP mask follows star order and zeroes physical padding.
            "mtp_mask": tensor([0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0]),
            "shards.0.padded_total_length": 16,
        },
    ),
    "tp-sp-cp-padding": (
        _tp_sp_cp_padding,
        {
            "bin.layout.completion_lengths": (5, 4, 3),
            "bin.layout.physical_completion_lengths": (12, 12, 12),
            "bin.layout.physical_total_length": 40,
            "bin.indices.physical_padding_positions.len": 24,
            "summary": [(16, 48, 24), (16, 48, 24)],
            "union": torch.arange(48),
            # The final eight positions are topology-only padding, separate from the
            # 24 per-branch native padding positions retained in the logical layout.
            "topology_tail": torch.zeros(8, dtype=torch.long),
        },
    ),
    "forest-with-singleton-fallback-root": (
        _singleton_fallback_forest,
        {
            # A causal fallback root beside the star: the forest mask is block diagonal.
            "mask": torch.block_diag(_mask("1 11 111 1111 11001 110011"), _mask("1 11 111 1111")),
            "bin.layout.mtp_loss_group_root_counts": (1, 1),
            "bin.packed_input_ids": tensor([10, 11, 20, 21, 30, 31, 40, 41, 42, 0]),
        },
    ),
    **{
        f"physical-alignment-tp{tp}-cp{cp}": (
            partial(get_shared_prefix_physical_alignment, tp_size=tp, cp_size=cp),
            quantum,
        )
        for tp, cp, quantum in ((1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 8), (4, 4, 32))
    },
    "padding-multiple-default": (partial(_padding_multiple, padding_multiple=None), 8),
    "padding-multiple-explicit": (partial(_padding_multiple, padding_multiple=32), 32),
    # Group "a" (rows 0, 2) shares one prompt; row 1 is its own causal root, isolated from "a".
    "dense-bins-reconstruction-causal": (
        _dense_bin_reconstruction,
        {
            "unit.row_indices": (0, 2, 1),
            "unit.physical_length": 9,
            "unit.shared_layout.mtp_loss_group_root_counts": (2,),
            "mask": torch.block_diag(_mask("1 11 111 1111 11101"), _mask("1 11 111 1111")),
        },
    ),
    # Nested tree (11 tokens) plus a 2-token root. Queries see real ancestor tokens and their
    # own node causally, never ancestor padding (4-5), sibling leaves or the other root.
    "tree-attention-isolation": (
        lambda: build_tree_attention_allow_mask(
            PackedTreeLayout.concat((nested_tree(), PackedTreeLayout((0,), (2,), (-1,))))
        ),
        torch.block_diag(
            _mask("1 11 111 1111 11111 111111 1111001 11110011 111100001 1111000011 11110000111"),
            _mask("1 11"),
        ),
    ),
    # The child (4) sees the root's two real tokens and itself, never the root's padding (2-3).
    "tree-padded-root-mask": (
        lambda: build_tree_attention_allow_mask(PackedTreeLayout((0, 4), (4, 1), (-1, 0), (2, 1))),
        _mask("1 11 111 1111 11001"),
    ),
}


@pytest.mark.parametrize("build, expected", TENSOR_EXAMPLES.values(), ids=list(TENSOR_EXAMPLES))
def test_tensor_examples(build, expected):
    assert_observed(build(), expected)


class _ToyModel:
    """Two layers of masked attention over token and RoPE-like position inputs."""

    def __init__(self):
        generator = torch.Generator().manual_seed(1234)

        def weight(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float64) * 0.4

        self.embedding, self.position = weight(VOCAB, WIDTH), weight(MAX_POSITION, WIDTH)
        self.layers = [tuple(weight(WIDTH, WIDTH) for _ in range(3)) for _ in range(2)]

    def __call__(self, tokens, positions, allow):
        hidden = self.embedding[tokens] + self.position[positions]
        for wq, wk, wv in self.layers:
            scores = ((hidden @ wq) @ (hidden @ wk).T).masked_fill(~allow, float("-inf"))
            hidden = hidden + torch.softmax(scores, dim=-1) @ (hidden @ wv)
        return hidden

    def dense(self, tokens):
        causal = torch.ones(len(tokens), len(tokens), dtype=torch.bool).tril()
        return self(tokens, torch.arange(len(tokens)), causal)


def _random_batch(rng):
    batch = []
    for group in range(rng.randint(1, 4)):
        prompt = [rng.randrange(1, VOCAB) for _ in range(rng.randint(1, 6))]
        group_id = None if rng.random() < 0.1 else f"g{group}"
        for _ in range(rng.randint(1, 5)):
            row_prompt = list(prompt)
            if rng.random() < 0.1:
                row_prompt[-1] = rng.randrange(1, VOCAB)
            completion = [rng.randrange(1, VOCAB) for _ in range(rng.randint(0, 8))]
            batch.append((group_id, row_prompt, completion))
    rng.shuffle(batch)
    lengths = [len(prompt) + len(completion) for _, prompt, completion in batch]
    input_ids = torch.full((len(batch), max(lengths) + rng.randint(0, 3)), VOCAB - 1)
    for index, (_, prompt, completion) in enumerate(batch):
        input_ids[index, : lengths[index]] = tensor(prompt + completion)
    prompt_lengths = [len(prompt) for _, prompt, _ in batch]
    return input_ids, tensor(lengths), prompt_lengths, [group_id for group_id, _, _ in batch]


def _planned_layouts(rng):
    """Yield (input_ids, input_lengths, layout) for stars, forests and singleton roots."""
    for _ in range(40):
        input_ids, input_lengths, prompt_lengths, group_ids = _random_batch(rng)
        multiple = rng.choice([1, 2, 4, 8])
        planned = batch_rows(input_ids, input_lengths, prompt_lengths, group_ids)
        kwargs = dict(bin_capacity=multiple * rng.choice([4, 8, 16]))
        kwargs |= dict(max_completions_per_bin=rng.choice([2, 3, 16]))
        plan = plan_shared_prefix_bins(planned, sequence_length_pad_multiple=multiple, **kwargs)
        assert covered_rows(plan) == list(range(len(planned)))
        pad = dict(sequence_length_pad_multiple=multiple, allow_singleton=True)
        fallback = [planned[index] for index in plan.fallback_row_indices]
        eligible = [r for r in fallback if r.group_id and r.prompt_length and r.completion_length]
        singletons = tuple(build_shared_prefix_layout([r], **pad) for r in eligible)
        for layout in plan.shared_bins:
            yield input_ids, input_lengths, layout
        for packed in pack(plan.shared_bins, 1024, None, multiple) if plan.shared_bins else ():
            roots = packed.roots if isinstance(packed, SharedPrefixForestLayout) else (packed,)
            yield input_ids, input_lengths, SharedPrefixForestLayout(roots + singletons)
            singletons = ()


def _check_mask_definition(layout):
    tree = layout.tree_layout
    nodes, padding = tree.segment_ids(), set(tree.padding_positions())

    def allowed(query, key):
        own_node = nodes[query] == nodes[key] and key <= query
        return own_node or (tree.is_ancestor(nodes[key], nodes[query]) and key not in padding)

    expected = [[allowed(q, k) for k in range(tree.total_len)] for q in range(tree.total_len)]
    assert torch.equal(build_star_attention_allow_mask(layout), tensor(expected))


def _check_packed_forward(model, input_ids, lengths, layout):
    tensor_bin = materialize_shared_prefix_layout(input_ids, input_lengths=lengths, layout=layout)
    indices, scatter_columns = tensor_bin.indices, tensor_bin.indices.completion_scatter_columns
    padding = set(indices.physical_padding_positions.tolist())
    # Loss positions never touch physical padding.
    assert not padding & set(indices.completion_positions.tolist())
    assert not padding & set(indices.predecessor_positions.tolist())
    mask = build_star_attention_allow_mask(layout)
    packed = model(tensor_bin.packed_input_ids, tensor_bin.position_ids, mask)
    dense = {r: model.dense(input_ids[r, : int(lengths[r])]) for r in layout.row_indices}
    for offset, root in layout.iter_roots():
        prompt = slice(0, root.prompt_length)
        for r in root.row_indices:
            torch.testing.assert_close(packed[offset:][prompt], dense[r][prompt])
    for predecessor, target, r, column in zip(
        indices.predecessor_positions.tolist(),
        indices.completion_positions.tolist(),
        layout.completion_scatter_rows,
        scatter_columns.tolist(),
        strict=True,
    ):
        torch.testing.assert_close(packed[predecessor], dense[r][column])
        torch.testing.assert_close(packed[target], dense[r][column + 1])
        assert int(tensor_bin.packed_input_ids[target]) == int(input_ids[r, column + 1])
    # Every completion token of every row is predicted exactly once.
    expected = sorted(
        (r, column)
        for _, root in layout.iter_roots()
        for r, length in zip(root.row_indices, root.completion_lengths)
        for column in range(root.prompt_length - 1, root.prompt_length + length - 1)
    )
    assert sorted(zip(layout.completion_scatter_rows, scatter_columns.tolist())) == expected
    # Token-aligned metadata follows the same gather and zeroes padding.
    source = torch.rand(input_ids.shape, generator=torch.Generator().manual_seed(1)) > 0.5
    gathered = materialize_shared_prefix_token_aligned_tensor(source, tensor_bin=tensor_bin)
    real = torch.ones_like(gathered).index_fill_(0, indices.physical_padding_positions, False)
    columns = indices.token_gather_columns.clamp_max(source.shape[1] - 1)
    assert torch.equal(gathered, source[indices.token_gather_rows, columns] & real)


def _planned_attention(seed):
    """Planned stars and forests: descriptor, oracle mask, then packed forward vs dense rows."""
    model, forests, layouts = _ToyModel(), 0, _planned_layouts(random.Random(seed))
    for index, (input_ids, input_lengths, layout) in enumerate(layouts):
        with note(seed=seed, layout_index=index, rows=layout.row_indices):
            check_layout_matches_tree(layout)
            # The oracle is checked first so that an oracle bug is not reported as a packing bug.
            _check_mask_definition(layout)
            _check_packed_forward(model, input_ids, input_lengths, layout)
        forests += isinstance(layout, SharedPrefixForestLayout)
    assert forests > 0


def _nested_tree_attention():
    """Deeper trees: every root-to-leaf path is causal attention, outputs and gradients match."""
    tree = PackedTreeLayout.concat((nested_tree(), PackedTreeLayout((0,), (2,), (-1,))))
    mask = build_tree_attention_allow_mask(tree)
    paths = {leaf: tensor(tree.path_token_indices(leaf)) for leaf in tree.leaf_nodes()}
    for leaf, path in paths.items():
        selected = mask[path[:, None], path[None, :]]
        assert torch.equal(selected, torch.ones_like(selected).tril()), f"leaf {leaf}"

    generator = torch.Generator().manual_seed(17)
    qkv = torch.randn((3, tree.total_len, 4), dtype=torch.float64, generator=generator)
    shared, dense = qkv.clone().requires_grad_(), qkv.clone().requires_grad_()

    def attend(values, allow):
        q, k, v = values
        return ((q @ k.T) / 2).masked_fill(~allow, -torch.inf).softmax(-1) @ v

    shared_output, losses = attend(shared, mask), []
    for leaf, path in paths.items():
        expanded = attend(dense[:, path], torch.ones((len(path),) * 2, dtype=torch.bool).tril())
        start, length = tree.node_start[leaf], tree.logical_node_len[leaf]
        leaf_output, expanded_leaf = shared_output[start : start + length], expanded[-length:]
        torch.testing.assert_close(leaf_output, expanded_leaf)
        losses.append(leaf_output.square().sum() + expanded_leaf.square().sum())
    sum(losses).backward()
    torch.testing.assert_close(shared.grad, dense.grad, rtol=1e-12, atol=1e-12)
    assert shared.grad[:, :4].abs().sum() > 0  # both leaves accumulate into ancestors
    assert not shared.grad[:, 4:6].any()


@pytest.mark.parametrize(
    "check",
    [
        pytest.param(partial(_planned_attention, 0), id="planned-seed0"),
        pytest.param(partial(_planned_attention, 3), id="planned-seed3"),
        pytest.param(_nested_tree_attention, id="nested-tree"),
    ],
)
def test_attention_matches_dense(check):
    check()


def _mcore_zigzag(length, cp_rank, cp_size):
    """Independent reference for MCore's two-chunk context-parallel ownership."""
    if cp_size == 1:
        return torch.arange(length)
    chunks = torch.arange(length).view(2 * cp_size, -1)
    return chunks[[cp_rank, 2 * cp_size - cp_rank - 1]].reshape(-1)


@pytest.mark.parametrize(
    "tp_size, cp_size",
    [pytest.param(tp, cp, id=f"tp{tp}-cp{cp}") for tp in (1, 2) for cp in (1, 2, 4)],
)
def test_context_parallel_shards(tp_size, cp_size):
    """CP shards of every planned layout reassemble it and pad only after real tokens."""
    quantum = get_shared_prefix_physical_alignment(tp_size=tp_size, cp_size=cp_size)
    layouts = _planned_layouts(random.Random(cp_size + 10 * tp_size))
    for index, (ids, lengths, layout) in enumerate(layouts):
        tensor_bin = materialize_shared_prefix_layout(ids, input_lengths=lengths, layout=layout)
        total = layout.physical_total_length
        for multiple in (None, quantum, 3 * quantum):
            with note(layout_index=index, padding_multiple=multiple):
                shards = _cp_shards(tensor_bin, cp_size, tp_size=tp_size, padding_multiple=multiple)
                padded, aligned = shards[0].padded_total_length, shards[0].padding_multiple
                assert padded % aligned == 0 and total <= padded < total + aligned
                for rank, shard in enumerate(shards):
                    # Sequence parallelism splits each CP-local sequence evenly over TP.
                    assert shard.packed_input_ids.numel() == padded // cp_size
                    assert (padded // cp_size) % tp_size == 0
                    reference = _mcore_zigzag(padded, rank, cp_size)
                    assert torch.equal(shard.global_token_indices, reference)
                    cp = dict(cp_rank=rank, cp_size=cp_size)
                    assert torch.equal(
                        reference, get_shared_prefix_context_parallel_indices(padded, **cp)
                    )
                tokens, positions = _unshard(shards)
                assert torch.equal(tokens[:total], tensor_bin.packed_input_ids)
                assert torch.equal(positions[:total], tensor_bin.position_ids)
                assert not tokens[total:].any() and not positions[total:].any()
