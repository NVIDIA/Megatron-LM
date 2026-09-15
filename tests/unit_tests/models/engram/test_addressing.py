# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import pytest
import torch

from megatron.core.transformer.engram import (
    CompressedTokenizer,
    EngramConfig,
    NgramHashMapping,
    RowShardedTableLayout,
    row_shard_bounds,
    row_shard_owner,
)
from megatron.core.transformer.engram.tokenizer import (
    build_engram_tokenizer_lookup,
    get_engram_tokenizer_pad_id,
)


def make_config(**kwargs):
    values = dict(
        enabled=True,
        hash_table_min_sizes=(17, 19),
        max_ngram_size=3,
        embedding_dim_per_ngram=8,
        num_hash_heads_per_ngram=2,
        layer_ids=(0, 2),
        pad_id=2,
        seed=7,
        kernel_size=4,
        hidden_size=8,
    )
    values.update(kwargs)
    return EngramConfig(**values)


def make_mapping(config=None):
    config = config or make_config()
    lookup = torch.arange(64)
    lookup[10] = 1
    return NgramHashMapping(
        config.hash_table_min_sizes,
        config.max_ngram_size,
        config.num_hash_heads_per_ngram,
        config.layer_ids,
        config.pad_id,
        config.seed,
        compressed_tokenizer=CompressedTokenizer(lookup),
    )


def numpy_reference(mapping, raw_ids, layer_id):
    compressed = mapping.compressed_tokenizer(raw_ids).numpy()
    batch, sequence = compressed.shape
    multipliers = getattr(mapping, f"layer_multipliers_{layer_id}").numpy()
    shifts = []
    for shift in range(mapping.max_ngram_size):
        shifts.append(
            compressed
            if shift == 0
            else np.pad(
                compressed, ((0, 0), (shift, 0)), mode="constant", constant_values=mapping.pad_id
            )[:, :sequence]
        )
    output = []
    for ngram in range(2, mapping.max_ngram_size + 1):
        mix = shifts[0] * multipliers[0]
        for index in range(1, ngram):
            mix = np.bitwise_xor(mix, shifts[index] * multipliers[index])
        for size in mapping.hash_moduli_by_layer[layer_id][ngram - 2]:
            output.append((mix % size).astype(np.int64, copy=False))
    return np.stack(output, axis=2)


class _TokenizerWithoutPad:
    @property
    def pad_id(self):
        raise NotImplementedError

    @property
    def eod(self):
        return 1


def test_engram_pad_id_uses_override_or_eod_fallback():
    tokenizer = _TokenizerWithoutPad()
    assert get_engram_tokenizer_pad_id(tokenizer, configured_pad_id=7) == 7
    assert get_engram_tokenizer_pad_id(tokenizer) == 1


@pytest.mark.parametrize(
    ("rows", "world_size", "bounds"),
    [
        (10, 3, [(0, 3), (3, 6), (6, 10)]),
        (2, 4, [(0, 0), (0, 1), (1, 1), (1, 2)]),
        (0, 3, [(0, 0), (0, 0), (0, 0)]),
    ],
)
def test_row_owner_formula_covers_each_row_once(rows, world_size, bounds):
    assert [row_shard_bounds(rows, rank, world_size) for rank in range(world_size)] == bounds
    for rank, (start, end) in enumerate(bounds):
        assert all(row_shard_owner(rows, row, world_size) == rank for row in range(start, end))
    assert sum(end - start for start, end in bounds) == rows


def test_row_layout_maps_multiple_tables_and_empty_shards():
    layout = RowShardedTableLayout((2, 7, 3), rank=1, world_size=4)
    assert layout.local_bounds == ((0, 1), (1, 3), (0, 1))
    assert layout.local_offsets == (0, 1, 3)
    global_rows = torch.tensor([0, 3, 9])
    assert layout.owners(global_rows).tolist() == [1, 1, 1]
    assert layout.to_local_rows(global_rows).tolist() == [0, 1, 3]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"table_backend": "row_a2a", "params_dtype": torch.float16}, "FP32 and BF16"),
        ({"table_backend": "unknown"}, "Unsupported Engram table backend"),
    ],
)
def test_row_a2a_configuration_fails_fast(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


def test_compressed_tokenizer_matches_normalization_rules():
    pytest.importorskip("tokenizers")
    decoded = ["A", "a", " ", "é", "E", "�", ""]
    raw_tokens = ["A", "a", "SPACE", "accent", "E", "<0xFF>", "EMPTY"]

    class FakeTokenizer:
        def __len__(self):
            return len(decoded)

        def decode(self, token_ids, skip_special_tokens=False):
            return decoded[token_ids[0]]

        def convert_ids_to_tokens(self, token_id):
            return raw_tokens[token_id]

    lookup = build_engram_tokenizer_lookup(FakeTokenizer())
    torch.testing.assert_close(lookup, torch.tensor([0, 0, 1, 2, 2, 3, 4]))


def test_runtime_hf_added_special_token_is_in_lookup():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"A": 0, "a": 1, "[UNK]": 2}, unk_token="[UNK]")
    )
    runtime_hf = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

    class HuggingFaceWrapper:
        tokenizer = runtime_hf

    class MegatronTokenizerTextWrapper:
        _tokenizer = HuggingFaceWrapper()

    before = build_engram_tokenizer_lookup(MegatronTokenizerTextWrapper())
    added = runtime_hf.add_special_tokens({"additional_special_tokens": ["<RUNTIME_ADDED>"]})
    assert added == 1
    added_id = runtime_hf.convert_tokens_to_ids("<RUNTIME_ADDED>")
    lookup = build_engram_tokenizer_lookup(MegatronTokenizerTextWrapper())

    assert before.numel() + 1 == len(runtime_hf)
    assert lookup.numel() == len(runtime_hf)
    assert 0 <= added_id < lookup.numel()
    assert 0 <= int(lookup[added_id]) <= int(lookup.max())


def test_runtime_tokenizer_requires_hf_lookup_capabilities():
    class UnsupportedTokenizer:
        pass

    with pytest.raises(TypeError, match=r"len\(\), decode\(\)"):
        build_engram_tokenizer_lookup(UnsupportedTokenizer())


def test_compressed_tokenizer_range_handling():
    tokenizer = CompressedTokenizer(lookup_table=np.array([0, 0, 2, 3], dtype=np.int64))
    assert torch.equal(tokenizer(torch.tensor([[-1, 0, 1, 2]])), torch.tensor([[-1, 0, 0, 2]]))
    with pytest.raises(ValueError, match="outside"):
        tokenizer(torch.tensor([[4]]))


def test_deterministic_hashing_unique_primes_and_reference_equations():
    config = make_config()
    mapping = make_mapping(config)
    all_sizes = [
        size
        for layer_id in config.layer_ids
        for order_sizes in mapping.hash_moduli_by_layer[layer_id]
        for size in order_sizes
    ]
    assert len(all_sizes) == len(set(all_sizes))
    raw_ids = torch.tensor([[1, 2, 3, 4], [10, 6, 7, 8]])
    for layer_id in config.layer_ids:
        np.testing.assert_array_equal(
            mapping(raw_ids, layer_id).numpy(), numpy_reference(mapping, raw_ids, layer_id)
        )


def test_multiplier_generation_matches_official_numpy_abi():
    config = make_config()
    mapping = NgramHashMapping(
        config.hash_table_min_sizes,
        config.max_ngram_size,
        config.num_hash_heads_per_ngram,
        config.layer_ids,
        config.pad_id,
        config.seed,
        compressed_tokenizer=CompressedTokenizer(torch.arange(64)),
    )
    half_bound = max(1, int(np.iinfo(np.int64).max // 64) // 2)
    golden = {
        0: [90085750735096467, 129302135670983891, 111788089137436145],
        2: [30741515212205435, 45358833932237951, 28612392822150695],
    }
    for layer_id in config.layer_ids:
        official = np.random.default_rng(config.seed + 10007 * layer_id).integers(
            low=0, high=half_bound, size=(config.max_ngram_size,), dtype=np.int64
        )
        expected = torch.from_numpy(official * 2 + 1)
        actual = getattr(mapping, f"layer_multipliers_{layer_id}")
        assert actual.dtype == torch.int64
        torch.testing.assert_close(actual, expected)
        assert actual.tolist() == golden[layer_id]


def test_invalid_engram_config_fails_fast():
    with pytest.raises(ValueError, match="one size"):
        make_config(hash_table_min_sizes=(17,))
    with pytest.raises(ValueError, match="divisible"):
        make_config(embedding_dim_per_ngram=7)


def test_compressed_hashes_preserve_layer_identity_and_repeat_exactly():
    mapping = make_mapping()
    raw_ids = torch.tensor([[1, 2, 3, 4], [10, 6, 7, 8]])
    compressed = mapping.compressed_tokenizer(raw_ids)
    hashes = {}
    for layer_id in mapping.layer_ids:
        hashes[layer_id] = mapping.forward_compressed(compressed, layer_id)
        assert hashes[layer_id].dtype == torch.int64
        torch.testing.assert_close(hashes[layer_id], mapping(raw_ids, layer_id))
        torch.testing.assert_close(
            hashes[layer_id], mapping.forward_compressed(compressed, layer_id)
        )
    assert not torch.equal(hashes[0], hashes[2])
