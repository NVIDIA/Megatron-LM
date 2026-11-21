# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared-prefix forward inputs: examples, a packed forward against dense rows, CP shards (CPU)."""

import random
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch import tensor

from megatron.core.datasets.shared_prefix.inputs import (
    SharedPrefixIndexMaps,
    build_shared_prefix_inputs,
    gather_shared_prefix_token_aligned,
    resolve_shared_prefix_padding_multiple,
    shard_shared_prefix_inputs_for_context_parallel,
    shard_shared_prefix_token_aligned_for_context_parallel,
    shared_prefix_topology_multiple,
)
from megatron.core.datasets.shared_prefix.layout import SharedPrefixPack, SharedPrefixTree

VOCAB, WIDTH, MAX_POSITION = 50, 8, 128


def _build(input_ids, lengths, prompts, tree_sizes, multiple=1, **kwargs):
    """``build_shared_prefix_inputs`` on nested lists or tensors."""
    return build_shared_prefix_inputs(
        torch.as_tensor(input_ids),
        input_lengths=torch.as_tensor(lengths),
        prompt_lengths=torch.as_tensor(prompts),
        tree_sizes=tree_sizes,
        padding_multiple=multiple,
        **kwargs,
    )


def _shards(inputs, cp_size, tp_size=1, **kwargs):
    shard = partial(shard_shared_prefix_inputs_for_context_parallel, inputs, **kwargs)
    return [shard(cp_rank=rank, cp_size=cp_size, tp_size=tp_size) for rank in range(cp_size)]


def _unshard(shards, values):
    """Scatter the per-rank ``values`` back to the padded packed order."""
    merged = torch.empty(shards[0].padded_total_length, dtype=values[0].dtype)
    for shard, value in zip(shards, values, strict=True):
        merged[shard.global_token_indices] = value
    return merged


def _reference_mask(layout):
    """Quadratic reference attention mask of a tree or pack. The model never builds it.

    A token sees the earlier tokens of its own prompt or row. A row also sees the prompt of its
    tree. Rows do not see each other, and trees do not see each other. Row padding follows the
    real tokens of its row, so no real token sees it, and it still sees its own row.
    """
    causal = torch.ones(layout.total_len, layout.total_len, dtype=torch.bool).tril()
    mask = torch.zeros_like(causal)
    for offset, tree in layout.iter_roots():
        prompt = slice(offset, offset + tree.prefix_len)
        rows = [slice(offset + row.start, offset + row.stop) for row in tree.completion_slices()]
        for span in (prompt, *rows):
            mask[span, span] = causal[span, span]
        for row in rows:
            mask[row, prompt] = True
    return mask


def _observe(inputs, *, cp_size=1, tp_size=1, padding_token_id=0, source=None):
    """Everything an example can pin, with tensors as lists and the mask as text.

    The mask text has one word of 0/1 per query, cut after the last key the query sees.
    """
    shards = _shards(inputs, cp_size, tp_size, padding_token_id=padding_token_id)
    mask = _reference_mask(inputs.layout).int().tolist()
    observed = {
        "layout": inputs.layout,
        "tree_sizes": inputs.tree_sizes,
        "tokens": inputs.packed_input_ids.tolist(),
        "positions": inputs.position_ids.tolist(),
        "mask": " ".join("".join("01"[key] for key in query).rstrip("0") for query in mask),
        "padded_total_length": [shard.padded_total_length for shard in shards],
        "shard_token_indices": [shard.global_token_indices.tolist() for shard in shards],
        "shard_tokens": [shard.packed_input_ids.tolist() for shard in shards],
        "shard_positions": [shard.position_ids.tolist() for shard in shards],
    }
    for name in SharedPrefixIndexMaps.__dataclass_fields__:
        observed[name] = getattr(inputs.indices, name).tolist()
    if source is not None:
        aligned = gather_shared_prefix_token_aligned(tensor(source), inputs=inputs)
        observed["token_aligned"] = aligned.tolist()
    return observed


def _tree(prompt, stored, real, multiple):
    """A tree from its prompt length and its stored and real completion lengths."""
    return SharedPrefixTree(prompt, stored, logical_completion_lens=real, padding_multiple=multiple)


_ONE_ROW = dict(input_ids=[[1, 2, 3]], lengths=[3], prompts=[1], tree_sizes=(1,))

EXAMPLES = {
    # Rows 0 and 1 share the prompt (10, 11, 12). The -1 lies past the length of row 0.
    "tree-from-padded-rows": (
        lambda: _observe(
            _build([[10, 11, 12, 20, 21, -1], [10, 11, 12, 30, 31, 32]], [5, 6], [3, 3], (2,))
        ),
        {
            "layout": _tree(3, (2, 3), (2, 3), 1),
            "tokens": [10, 11, 12, 20, 21, 30, 31, 32],
            "positions": [0, 1, 2, 3, 4, 3, 4, 5],
            "token_gather_rows": [0, 0, 0, 0, 0, 1, 1, 1],
            "token_gather_columns": [0, 1, 2, 3, 4, 3, 4, 5],
            "completion_positions": [3, 4, 5, 6, 7],
            "predecessor_positions": [2, 3, 2, 5, 6],
            "completion_scatter_rows": [0, 0, 1, 1, 1],
            "completion_scatter_columns": [2, 3, 2, 3, 4],
            "padding_positions": [],
            # Row 0 (3-4) and row 1 (5-7) see the prompt (0-2) and themselves, never each other.
            "mask": "1 11 111 1111 11111 111001 1110011 11100111",
        },
    ),
    # Multiple 4 stores the rows of 5 and 7 tokens with 8 tokens each. Tokens 91, 92 and the
    # source values 7 lie past the length of row 0 and are not copied.
    "row-padding": (
        lambda: _observe(
            _build(
                [[10, 11, 12, 20, 21, 91, 92], [10, 11, 12, 30, 31, 32, 33]],
                [5, 7],
                [3, 3],
                (2,),
                4,
            ),
            cp_size=2,
            source=[[0, 0, 0, 1, 1, 7, 7], [0, 0, 0, 1, 1, 1, 1]],
        ),
        {
            "layout": _tree(3, (5, 5), (2, 4), 4),
            "tokens": [10, 11, 12, 20, 21, 0, 0, 0, 30, 31, 32, 33, 0],
            "positions": [0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7],
            "completion_positions": [3, 4, 8, 9, 10, 11],
            "predecessor_positions": [2, 3, 2, 8, 9, 10],
            "padding_positions": [5, 6, 7, 12],
            # Positions 5-7 and 12 are row padding. Each row sees only the prompt and itself.
            "mask": (
                "1 11 111 1111 11111 111111 1111111 11111111"
                " 111000001 1110000011 11100000111 111000001111 1110000011111"
            ),
            "token_aligned": [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
            "padded_total_length": [16, 16],
        },
    ),
    # The 9-token tree is padded to 12. Pack padding (token 99) follows the last row. Row
    # padding keeps token 0.
    "cp-shards-pad-after-the-last-row": (
        lambda: _observe(
            _build([[10, 11, 12, 20, 0], [10, 11, 12, 30, 31]], [4, 5], [3, 3], (2,), 4),
            cp_size=2,
            padding_token_id=99,
        ),
        {
            "tokens": [10, 11, 12, 20, 30, 31, 0, 0, 0],
            "padded_total_length": [12, 12],
            "shard_token_indices": [[0, 1, 2, 9, 10, 11], [3, 4, 5, 6, 7, 8]],
            "shard_tokens": [[10, 11, 12, 99, 99, 99], [20, 30, 31, 0, 0, 0]],
            "shard_positions": [[0, 1, 2, 0, 0, 0], [3, 3, 4, 5, 6, 7]],
        },
    ),
    # TP 2 with CP 2 needs a multiple of 8, and the caller chose 16. Each row is stored with 16
    # tokens. Eight tokens of pack padding extend the 40-token tree to 48.
    "tp-cp-padding": (
        lambda: _observe(
            _build(
                [
                    [10, 11, 12, 13, 20, 21, 22, 0, 0],
                    [10, 11, 12, 13, 30, 31, 32, 33, 0],
                    [10, 11, 12, 13, 40, 41, 42, 43, 44],
                ],
                [7, 8, 9],
                [4, 4, 4],
                (3,),
                16,
            ),
            cp_size=2,
            tp_size=2,
        ),
        {
            "layout": _tree(4, (12, 12, 12), (3, 4, 5), 16),
            "padding_positions": [*range(7, 16), *range(20, 28), *range(33, 40)],
            "padded_total_length": [48, 48],
            "shard_tokens": [
                [10, 11, 12, 13, 20, 21, 22] + [0] * 17,
                [0] * 4 + [30, 31, 32, 33] + [0] * 8 + [40, 41, 42, 43, 44] + [0] * 3,
            ],
        },
    ),
    # Two trees in one forward. The second tree has one row, and its mask block is causal.
    "pack-with-a-one-row-tree": (
        lambda: _observe(
            _build(
                [[10, 11, 20, 21], [10, 11, 30, 31], [40, 41, 42, 0]],
                [4, 4, 3],
                [2, 2, 2],
                [2, 1],
                4,
                mtp_loss_group_tree_counts=[1, 1],
            )
        ),
        {
            "layout": SharedPrefixPack(
                (_tree(2, (2, 2), (2, 2), 4), _tree(2, (2,), (1,), 4)),
                mtp_loss_group_root_counts=(1, 1),
            ),
            "tree_sizes": (2, 1),
            "tokens": [10, 11, 20, 21, 30, 31, 40, 41, 42, 0],
            "positions": [0, 1, 2, 3, 2, 3, 0, 1, 2, 3],
            "token_gather_rows": [0, 0, 0, 0, 1, 1, 2, 2, 2, 2],
            "completion_positions": [2, 3, 4, 5, 8],
            "predecessor_positions": [1, 2, 1, 4, 7],
            "completion_scatter_rows": [0, 0, 1, 1, 2],
            "completion_scatter_columns": [1, 2, 1, 2, 1],
            "padding_positions": [9],
            "mask": "1 11 111 1111 11001 110011 0000001 00000011 000000111 0000001111",
        },
    ),
    "one-tree-without-loss-groups-is-a-tree": (
        lambda: _observe(_build(**_ONE_ROW)),
        {"layout": _tree(1, (2,), (2,), 1)},
    ),
    "one-tree-with-a-loss-group-is-a-pack": (
        lambda: _observe(_build(**_ONE_ROW, mtp_loss_group_tree_counts=(1,))),
        {"layout": SharedPrefixPack((_tree(1, (2,), (2,), 1),), mtp_loss_group_root_counts=(1,))},
    ),
}


@pytest.mark.parametrize("observe, expected", EXAMPLES.values(), ids=list(EXAMPLES))
def test_examples(observe, expected):
    observed = observe()
    assert {name: observed[name] for name in expected} == expected


def test_tensor_dtypes():
    """Packed tokens keep the dtype of ``input_ids``. Positions and index maps are long."""
    inputs = build_shared_prefix_inputs(
        tensor([[1, 2, 3]], dtype=torch.int32),
        input_lengths=tensor([3], dtype=torch.int32),
        prompt_lengths=tensor([1], dtype=torch.int16),
        tree_sizes=[1],
        padding_multiple=2,
    )
    assert inputs.packed_input_ids.dtype == torch.int32
    assert inputs.packed_input_ids.tolist() == [1, 2, 3, 0]
    assert inputs.position_ids.dtype == torch.long
    for name in SharedPrefixIndexMaps.__dataclass_fields__:
        assert getattr(inputs.indices, name).dtype == torch.long, name


@pytest.mark.parametrize(
    "tp_size, cp_size, multiple", [(1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 8), (4, 4, 32)]
)
def test_topology_multiple(tp_size, cp_size, multiple):
    assert shared_prefix_topology_multiple(tp_size=tp_size, cp_size=cp_size) == multiple
    resolve = partial(resolve_shared_prefix_padding_multiple, tp_size=tp_size, cp_size=cp_size)
    assert resolve(padding_multiple=None) == multiple
    assert resolve(padding_multiple=4 * multiple) == 4 * multiple


def test_token_aligned_source_must_cover_every_real_token():
    """A source narrower than the longest row raises instead of reading a wrong column."""
    inputs = _build(
        [[10, 11, 12, 20, 21, 22, 0], [10, 11, 12, 30, 31, 32, 33]], [6, 7], [3, 3], (2,), 4
    )
    full = torch.ones(2, 7, dtype=torch.long)
    gathered = gather_shared_prefix_token_aligned(full, inputs=inputs)
    padding = inputs.indices.padding_positions
    assert torch.equal(gathered, torch.ones_like(gathered).index_fill_(0, padding, 0))
    # Width 6 cuts the last token of row 1, as a [:, :-1] slice does. Width 2 cuts a prompt token.
    for width in (6, 2):
        with pytest.raises(ValueError, match="width"):
            gather_shared_prefix_token_aligned(full[:, :width], inputs=inputs)


class _ToyModel:
    """Two layers of masked attention over token and position embeddings, in float64."""

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


def _random_forward(rng):
    """Rows of one forward: 1 to 4 trees, each one prompt with 1 to 5 completions.

    Columns past the length of a row hold token ``VOCAB - 1``, which no forward may read.
    """
    rows, prompts, tree_sizes = [], [], []
    for _ in range(rng.randint(1, 4)):
        prompt = [rng.randrange(1, VOCAB - 1) for _ in range(rng.randint(1, 9))]
        size = rng.randint(1, 5)
        tree_sizes.append(size)
        for _ in range(size):
            completion = [rng.randrange(1, VOCAB - 1) for _ in range(rng.randint(1, 13))]
            rows.append(prompt + completion)
            prompts.append(len(prompt))
    input_ids = torch.full((len(rows), max(map(len, rows)) + rng.randint(0, 3)), VOCAB - 1)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = tensor(row)
    return input_ids, tensor(list(map(len, rows))), tensor(prompts), tree_sizes


def _check_packed_forward(model, input_ids, lengths, prompts, inputs):
    """The packed forward gives every row the outputs of its own dense causal forward."""
    layout, indices = inputs.layout, inputs.indices
    rows, columns = indices.completion_scatter_rows, indices.completion_scatter_columns
    padding = set(indices.padding_positions.tolist())
    # No loss position is row padding.
    assert not padding & set(indices.completion_positions.tolist())
    assert not padding & set(indices.predecessor_positions.tolist())

    packed = model(inputs.packed_input_ids, inputs.position_ids, _reference_mask(layout))
    dense = torch.zeros(*input_ids.shape, WIDTH, dtype=torch.float64)
    for row, length in enumerate(lengths.tolist()):
        dense[row, :length] = model.dense(input_ids[row, :length])
    first_row = 0
    for (offset, tree), size in zip(layout.iter_roots(), inputs.tree_sizes, strict=True):
        prompt = packed[offset : offset + tree.prefix_len]
        for row in range(first_row, first_row + size):
            torch.testing.assert_close(prompt, dense[row, : tree.prefix_len])
        first_row += size
    # The predecessor's output predicts the completion token at the scatter row and column.
    torch.testing.assert_close(packed[indices.predecessor_positions], dense[rows, columns])
    torch.testing.assert_close(packed[indices.completion_positions], dense[rows, columns + 1])
    targets = inputs.packed_input_ids[indices.completion_positions]
    assert torch.equal(targets, input_ids[rows, columns + 1])
    # Every completion token of every row is predicted exactly once, in row order.
    expected = [
        (row, column)
        for row, (prompt, length) in enumerate(zip(prompts.tolist(), lengths.tolist()))
        for column in range(prompt - 1, length - 1)
    ]
    assert list(zip(rows.tolist(), columns.tolist())) == expected

    # The layout's dense rows, which MTP uses, read each row back with its row padding. A
    # token-aligned tensor follows the same order.
    multiple = layout.padding_multiple
    source = input_ids + 100
    aligned = gather_shared_prefix_token_aligned(source, inputs=inputs, padding_value=-1)
    for row, branch in enumerate(layout.dense_branch_indices("cpu")):
        length = int(lengths[row])
        tail = (0, len(branch) - length)
        assert len(branch) % multiple == 0 and 0 <= tail[1] < multiple
        assert torch.equal(inputs.packed_input_ids[branch], F.pad(input_ids[row, :length], tail))
        assert torch.equal(aligned[branch], F.pad(source[row, :length], tail, value=-1))


@pytest.mark.parametrize("seed", [0, 3])
def test_packed_forward_matches_dense_rows(seed):
    rng, model, packs = random.Random(seed), _ToyModel(), 0
    for _ in range(40):
        input_ids, lengths, prompts, tree_sizes = _random_forward(rng)
        inputs = _build(input_ids, lengths, prompts, tree_sizes, rng.choice([1, 2, 4, 8]))
        _check_packed_forward(model, input_ids, lengths, prompts, inputs)
        packs += isinstance(inputs.layout, SharedPrefixPack)
    assert 0 < packs < 40  # both a single tree and a pack were checked


def _mcore_zigzag(length, cp_rank, cp_size):
    """Independent reference for the two chunks that a context-parallel rank owns."""
    if cp_size == 1:
        return torch.arange(length)
    chunks = torch.arange(length).view(2 * cp_size, -1)
    return chunks[[cp_rank, 2 * cp_size - cp_rank - 1]].reshape(-1)


@pytest.mark.parametrize(
    "tp_size, cp_size",
    [pytest.param(tp, cp, id=f"tp{tp}-cp{cp}") for tp in (1, 2) for cp in (1, 2, 4)],
)
def test_context_parallel_shards(tp_size, cp_size):
    """The shards of a forward reassemble it and pad it only after its last token."""
    quantum = shared_prefix_topology_multiple(tp_size=tp_size, cp_size=cp_size)
    rng = random.Random(cp_size + 10 * tp_size)
    for _ in range(40):
        input_ids, lengths, prompts, tree_sizes = _random_forward(rng)
        multiple = quantum * rng.choice([1, 3])
        inputs = _build(input_ids, lengths, prompts, tree_sizes, multiple)
        layout, total = inputs.layout, inputs.layout.total_len
        assert torch.equal(inputs.position_ids, layout.position_ids("cpu"))

        shards = _shards(inputs, cp_size, tp_size, padding_token_id=VOCAB)
        padded = shards[0].padded_total_length
        assert padded % multiple == 0 and total <= padded < total + multiple
        for rank, shard in enumerate(shards):
            assert (shard.padded_total_length, shard.padding_multiple) == (padded, multiple)
            # Sequence parallelism splits the tokens of a CP rank evenly over the TP ranks.
            assert shard.packed_input_ids.numel() == padded // cp_size
            assert (padded // cp_size) % tp_size == 0
            assert torch.equal(shard.global_token_indices, _mcore_zigzag(padded, rank, cp_size))

        tokens = _unshard(shards, [shard.packed_input_ids for shard in shards])
        positions = _unshard(shards, [shard.position_ids for shard in shards])
        assert torch.equal(tokens[:total], inputs.packed_input_ids)
        assert torch.equal(positions[:total], inputs.position_ids)
        assert (tokens[total:] == VOCAB).all() and not positions[total:].any()

        aligned = gather_shared_prefix_token_aligned(input_ids % 2 == 0, inputs=inputs)
        shard_aligned = partial(shard_shared_prefix_token_aligned_for_context_parallel, aligned)
        merged = _unshard(shards, [shard_aligned(shard=shard, padding_value=1) for shard in shards])
        assert torch.equal(merged[:total], aligned) and merged[total:].all()


_VALID = dict(
    input_ids=[[2, 3, 4, 5], [2, 3, 6, 7]], lengths=[4, 4], prompts=[2, 2], tree_sizes=(2,)
)


def _changed(**changes):
    """A call that builds the valid two-row tree ``_VALID`` with ``changes`` applied."""
    return partial(_build, **(_VALID | changes))


def _with_dtype(field, dtype):
    """``_VALID`` with one tensor cast to ``dtype``. Float values are made non-integral."""
    value = tensor(_VALID[field]).to(getattr(torch, dtype))
    return _changed(**{field: value + 0.25 if dtype == "float32" else value})


def _shard_valid(multiple=4, **kwargs):
    kwargs = dict(cp_rank=0, cp_size=2, tp_size=1) | kwargs
    return shard_shared_prefix_inputs_for_context_parallel(_changed(multiple=multiple)(), **kwargs)


def _token_aligned(source):
    return gather_shared_prefix_token_aligned(source, inputs=_changed()())


def _shard_token_aligned(packed):
    return shard_shared_prefix_token_aligned_for_context_parallel(packed, shard=_shard_valid())


def _topology(tp_size, cp_size):
    return partial(shared_prefix_topology_multiple, tp_size=tp_size, cp_size=cp_size)


_resolve_tp2_cp2 = partial(resolve_shared_prefix_padding_multiple, tp_size=2, cp_size=2)
_THREE_ROWS = dict(
    input_ids=[[2, 3, 4], [5, 6, 7], [5, 9, 7]],
    lengths=[3, 3, 3],
    prompts=[2, 2, 2],
    tree_sizes=(1, 2),
)

REJECTED = [
    ("topology-tp-zero", _topology(0, 1), "tp_size must be a positive integer"),
    ("topology-tp-bool", _topology(True, 1), "tp_size must be a positive integer"),
    ("topology-cp-float", _topology(1, 2.0), "cp_size must be a positive integer"),
    *[
        (f"padding-multiple-{name}", partial(_resolve_tp2_cp2, padding_multiple=raw), match)
        for name, raw, match in [
            ("true", True, "integer or None"),
            ("false", False, "integer or None"),
            ("float", 8.0, "integer or None"),
            ("str", "8", "integer or None"),
            ("zero", 0, "positive multiple"),
            ("negative", -8, "positive multiple"),
            ("not-a-topology-multiple", 12, "topology multiple 8"),
        ]
    ],
    *[
        (f"build-padding-multiple-{raw!r}", _changed(multiple=raw), "padding_multiple must be a")
        for raw in (0, -4, True, 4.0, "4", None)
    ],
    ("rows-1d-input", _changed(input_ids=[1, 2, 3]), "input_ids must have shape"),
    # Integral float tokens are rejected by dtype, not by value.
    ("rows-float-input", _changed(input_ids=torch.zeros(2, 4)), "integer dtype"),
    *[
        (f"rows-dtype-{field}-{dtype}", _with_dtype(field, dtype), "integer dtype")
        for field in ("input_ids", "lengths", "prompts")
        for dtype in ("float32", "complex64", "bool")
    ],
    ("rows-length-count", _changed(lengths=[4, 4, 4]), r"input_lengths must have shape \[2\]"),
    ("rows-length-over-width", _changed(lengths=[4, 5]), "input_lengths must be within"),
    ("rows-prompt-over-width", _changed(prompts=[5, 5]), "prompt_lengths must be within"),
    ("rows-prompt-over-length", _changed(lengths=[4, 1]), "row 1 needs a non-empty prompt"),
    ("rows-empty-completion", _changed(lengths=[4, 2]), "row 1 needs .* a non-empty completion"),
    ("rows-empty-prompt", _changed(prompts=[0, 0]), "row 0 needs a non-empty prompt"),
    ("tree-two-prompt-lengths", _changed(prompts=[2, 1]), "rows 0 and 1 .* prompt lengths 2 and 1"),
    (
        "tree-prompt-tokens-differ",
        _changed(input_ids=[[2, 3, 4, 5], [2, 9, 6, 7]]),
        "rows 0 and 1 are in one tree but their first 2 tokens",
    ),
    # Rows 1 and 2 form the second tree. Row 0 is not compared with them.
    ("tree-prompt-tokens-differ-in-second-tree", _changed(**_THREE_ROWS), "rows 1 and 2 are"),
    ("tree-sizes-too-few", _changed(tree_sizes=(1,)), "cover every row once"),
    ("tree-sizes-too-many", _changed(tree_sizes=(2, 1)), "cover every row once"),
    ("tree-sizes-empty", _changed(tree_sizes=()), "one positive integer per tree"),
    ("tree-sizes-zero", _changed(tree_sizes=(2, 0)), "one positive integer per tree"),
    ("tree-sizes-float", _changed(tree_sizes=(2.0,)), "one positive integer per tree"),
    ("mtp-groups-miss-a-tree", _changed(mtp_loss_group_tree_counts=(2,)), "partition every root"),
    ("token-aligned-1d", partial(_token_aligned, torch.zeros(4)), "must have shape"),
    ("token-aligned-row-count", partial(_token_aligned, torch.zeros(3, 4)), "needs 2 rows"),
    (
        "shard-cp-rank-out-of-range",
        partial(_shard_valid, cp_rank=2),
        r"cp_rank must be .* \[0, 2\)",
    ),
    ("shard-cp-rank-bool", partial(_shard_valid, cp_rank=True), "cp_rank must be an integer"),
    ("shard-cp-size-zero", partial(_shard_valid, cp_size=0), "cp_size must be a positive integer"),
    # The tree was built with multiple 2, and CP 2 needs a multiple of 4.
    ("shard-multiple-below-topology", partial(_shard_valid, multiple=2), "topology multiple 4"),
    ("shard-token-aligned-2d", partial(_shard_token_aligned, torch.zeros(1, 6)), "must have shape"),
    # The valid tree has 6 tokens and is padded to 8. A 3-token tensor is not one of its tensors.
    ("shard-token-aligned-short", partial(_shard_token_aligned, torch.zeros(3)), "does not belong"),
    ("shard-token-aligned-long", partial(_shard_token_aligned, torch.zeros(9)), "does not belong"),
]


@pytest.mark.parametrize(
    "call, match", [pytest.param(call, match, id=name) for name, call, match in REJECTED]
)
def test_rejects_invalid_inputs(call, match):
    with pytest.raises(ValueError, match=match):
        call()
