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

"""Seeded properties of materialized stars and forests against dense per-row evaluation."""

import random

import pytest
import torch

from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    build_shared_prefix_layout,
    pack_shared_prefix_groups,
    plan_shared_prefix_bins,
)
from megatron.rl.shared_prefix_tensors import (
    build_shared_prefix_rows,
    get_shared_prefix_context_parallel_indices,
    get_shared_prefix_physical_alignment,
    materialize_shared_prefix_layout,
    materialize_shared_prefix_token_aligned_tensor,
    shard_shared_prefix_tensor_bin_for_context_parallel,
)
from tests.unit_tests.rl.shared_prefix_oracles import build_star_attention_allow_mask

VOCAB, WIDTH, MAX_POSITION = 50, 8, 128


class _ToyModel:
    """Two layers of masked attention over token and RoPE-like position inputs."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(1234)

        def weight(*shape: int) -> torch.Tensor:
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
        length = tokens.numel()
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        return self(tokens, torch.arange(length), causal)


def _random_batch(rng: random.Random):
    rows = []
    for group in range(rng.randint(1, 4)):
        prompt = [rng.randrange(1, VOCAB) for _ in range(rng.randint(1, 6))]
        group_id = None if rng.random() < 0.1 else f"g{group}"
        for _ in range(rng.randint(1, 5)):
            row_prompt = list(prompt)
            if rng.random() < 0.1:
                row_prompt[-1] = rng.randrange(1, VOCAB)
            completion = [rng.randrange(1, VOCAB) for _ in range(rng.randint(0, 8))]
            rows.append((group_id, row_prompt, completion))
    rng.shuffle(rows)
    lengths = [len(prompt) + len(completion) for _, prompt, completion in rows]
    input_ids = torch.full((len(rows), max(lengths) + rng.randint(0, 3)), VOCAB - 1)
    for index, (_, prompt, completion) in enumerate(rows):
        input_ids[index, : lengths[index]] = torch.tensor(prompt + completion)
    return (
        input_ids,
        torch.tensor(lengths),
        torch.tensor([len(prompt) for _, prompt, _ in rows]),
        [group_id for group_id, _, _ in rows],
    )


def _planned_layouts(rng: random.Random):
    """Yield (input_ids, input_lengths, layout) for stars, forests and singleton roots."""
    for _ in range(40):
        input_ids, input_lengths, prompt_lengths, group_ids = _random_batch(rng)
        multiple = rng.choice([1, 2, 4, 8])
        rows = build_shared_prefix_rows(
            input_ids=input_ids,
            input_lengths=input_lengths,
            prompt_lengths=prompt_lengths,
            group_ids=group_ids,
        )
        plan = plan_shared_prefix_bins(
            rows,
            bin_capacity=multiple * rng.choice([4, 8, 16]),
            max_completions_per_bin=rng.choice([2, 3, 16]),
            sequence_length_pad_multiple=multiple,
        )
        covered = [row for layout in plan.shared_bins for row in layout.row_indices]
        assert sorted(covered + list(plan.fallback_row_indices)) == list(range(len(rows)))
        singletons = tuple(
            build_shared_prefix_layout(
                [rows[index]], sequence_length_pad_multiple=multiple, allow_singleton=True
            )
            for index in plan.fallback_row_indices
            if rows[index].group_id and rows[index].prompt_length and rows[index].completion_length
        )
        for layout in plan.shared_bins:
            yield input_ids, input_lengths, layout
        if plan.shared_bins:
            for packed in pack_shared_prefix_groups(
                plan.shared_bins, bin_capacity=1024, dense_capacity=None, padding_multiple=multiple
            ):
                roots = packed.roots if isinstance(packed, SharedPrefixForestLayout) else (packed,)
                yield input_ids, input_lengths, SharedPrefixForestLayout(roots + singletons)
                singletons = ()


def test_packed_forward_matches_dense_rows_at_every_planned_position():
    model = _ToyModel()
    forests = 0
    for input_ids, input_lengths, layout in _planned_layouts(random.Random(0)):
        tensor_bin = materialize_shared_prefix_layout(
            input_ids, input_lengths=input_lengths, layout=layout
        )
        indices = tensor_bin.indices
        padding = set(indices.physical_padding_positions.tolist())
        # Loss positions never touch physical padding.
        assert not padding & set(indices.completion_positions.tolist())
        assert not padding & set(indices.predecessor_positions.tolist())
        packed = model(
            tensor_bin.packed_input_ids,
            tensor_bin.position_ids,
            build_star_attention_allow_mask(layout),
        )
        dense = {
            row: model.dense(input_ids[row, : int(input_lengths[row])])
            for row in layout.row_indices
        }
        for offset, root in layout.iter_roots():
            for row in root.row_indices:
                prompt = packed[offset : offset + root.prompt_length]
                torch.testing.assert_close(prompt, dense[row][: root.prompt_length])
        for predecessor, target, row, column in zip(
            indices.predecessor_positions.tolist(),
            indices.completion_positions.tolist(),
            layout.completion_scatter_rows,
            indices.completion_scatter_columns.tolist(),
            strict=True,
        ):
            torch.testing.assert_close(packed[predecessor], dense[row][column])
            torch.testing.assert_close(packed[target], dense[row][column + 1])
            assert int(tensor_bin.packed_input_ids[target]) == int(input_ids[row, column + 1])
        # Every completion token of every row is predicted exactly once.
        expected = sorted(
            (row, column)
            for _, root in layout.iter_roots()
            for row, length in zip(root.row_indices, root.completion_lengths)
            for column in range(root.prompt_length - 1, root.prompt_length + length - 1)
        )
        scattered = zip(layout.completion_scatter_rows, indices.completion_scatter_columns.tolist())
        assert sorted(scattered) == expected
        # Token-aligned metadata follows the same gather and zeroes padding.
        source = torch.rand(input_ids.shape, generator=torch.Generator().manual_seed(1)) > 0.5
        gathered = materialize_shared_prefix_token_aligned_tensor(source, tensor_bin=tensor_bin)
        rows, columns = indices.token_gather_rows, indices.token_gather_columns
        real = torch.ones_like(gathered)
        real[indices.physical_padding_positions] = False
        expected_values = source[rows, columns.clamp_max(source.shape[1] - 1)] & real
        assert torch.equal(gathered, expected_values)
        forests += isinstance(layout, SharedPrefixForestLayout)
    assert forests > 0


def test_reference_mask_matches_its_definition():
    for _, _, layout in _planned_layouts(random.Random(3)):
        tree = layout.tree_layout
        segments, padding = tree.segment_ids(), set(tree.padding_positions())
        expected = torch.tensor(
            [
                [
                    (segments[query] == segments[key] and key <= query)
                    or (tree.is_ancestor(segments[key], segments[query]) and key not in padding)
                    for key in range(tree.total_len)
                ]
                for query in range(tree.total_len)
            ]
        )
        assert torch.equal(build_star_attention_allow_mask(layout), expected)


def _mcore_zigzag(length: int, cp_rank: int, cp_size: int) -> torch.Tensor:
    """Independent reference for MCore's two-chunk context-parallel ownership."""
    if cp_size == 1:
        return torch.arange(length)
    chunks = torch.arange(length).view(2 * cp_size, -1)
    return chunks[[cp_rank, 2 * cp_size - cp_rank - 1]].reshape(-1)


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_context_parallel_shards_partition_stars_and_forests(tp_size, cp_size):
    quantum = get_shared_prefix_physical_alignment(tp_size=tp_size, cp_size=cp_size)
    for input_ids, input_lengths, layout in _planned_layouts(random.Random(cp_size + 10 * tp_size)):
        tensor_bin = materialize_shared_prefix_layout(
            input_ids, input_lengths=input_lengths, layout=layout
        )
        total = layout.physical_total_length
        for padding_multiple in (None, quantum, 3 * quantum):
            shards = [
                shard_shared_prefix_tensor_bin_for_context_parallel(
                    tensor_bin,
                    cp_rank=rank,
                    cp_size=cp_size,
                    tp_size=tp_size,
                    padding_multiple=padding_multiple,
                )
                for rank in range(cp_size)
            ]
            padded = shards[0].padded_total_length
            assert padded % shards[0].padding_multiple == 0
            assert total <= padded < total + shards[0].padding_multiple
            tokens = torch.empty(padded, dtype=torch.long)
            positions = torch.empty(padded, dtype=torch.long)
            for rank, shard in enumerate(shards):
                # Sequence parallelism splits each CP-local sequence evenly over TP.
                assert shard.packed_input_ids.numel() == padded // cp_size
                assert (padded // cp_size) % tp_size == 0
                assert torch.equal(shard.global_token_indices, _mcore_zigzag(padded, rank, cp_size))
                assert torch.equal(
                    shard.global_token_indices,
                    get_shared_prefix_context_parallel_indices(
                        padded, cp_rank=rank, cp_size=cp_size
                    ),
                )
                tokens[shard.global_token_indices] = shard.packed_input_ids
                positions[shard.global_token_indices] = shard.position_ids
            assert torch.equal(tokens[:total], tensor_bin.packed_input_ids)
            assert torch.equal(positions[:total], tensor_bin.position_ids)
            assert not tokens[total:].any() and not positions[total:].any()
