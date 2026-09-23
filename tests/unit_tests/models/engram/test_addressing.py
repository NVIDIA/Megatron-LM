# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.engram import (
    CompressedTokenizer,
    Engram,
    EngramConfig,
    MultiHeadEmbedding,
    NgramHashMapping,
    RowShardedMultiHeadEmbedding,
    RowShardedTableLayout,
    ShortConv,
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


def _lookup():
    return torch.arange(64)


def test_multi_head_embedding_uses_layer_local_offsets():
    memory = MultiHeadEmbedding([5, 7, 11, 13], D=3)
    parameter = memory.embedding.weight
    metadata = parameter.engram_table_metadata
    memory.bfloat16().float()
    assert memory.embedding.weight is parameter
    assert parameter.engram_table_metadata is metadata
    assert torch.equal(memory.offsets, torch.tensor([0, 5, 12, 23]))
    ids = torch.tensor([[[1, 2, 3, 4]]])
    expected = F.embedding(ids + memory.offsets, memory.embedding.weight)
    torch.testing.assert_close(memory(ids), expected)
    assert memory.embedding.num_embeddings == 36
    assert not memory.embedding.weight.engram_table_metadata.row_parallel
    assert not hasattr(memory.embedding.weight.engram_table_metadata, "lr_multiplier")
    assert not hasattr(memory.embedding.weight.engram_table_metadata, "weight_decay")


def test_short_conv_matches_padding_crop_norm_and_activation():
    module = ShortConv(hidden_size=4, kernel_size=3, dilation=2)
    values = torch.randn(2, 5, 4)
    normalized = module.norms[0](values)
    expected = F.silu(module.conv(normalized.transpose(1, 2))[..., :5].transpose(1, 2))
    torch.testing.assert_close(module(values), expected)
    assert module.conv.bias is None


def test_checkpoint_restores_lookup_and_validates_architecture():
    config = make_config()
    source = Engram(engram_config=config, tokenizer_lookup=_lookup())
    state = copy.deepcopy(source.state_dict())
    raw_ids = torch.tensor([[60, 61, 62, 63]])
    source_compressed = source.compress_input_ids(raw_ids)
    source_hashes = source.hash_mapping.forward_compressed(source_compressed, 0)

    changed_lookup = _lookup()
    changed_lookup[-1] = 0
    config.perform_initialization = False
    rng = torch.get_rng_state().clone()
    restored = Engram(engram_config=config, tokenizer_lookup=changed_lookup)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    result = restored.load_state_dict(copy.deepcopy(state), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    torch.testing.assert_close(restored.compressed_tokenizer.lookup_table, _lookup())
    restored_hashes = restored.hash_mapping.forward_compressed(
        restored.compress_input_ids(raw_ids), 0
    )
    torch.testing.assert_close(restored_hashes, source_hashes)

    mismatched_state = copy.deepcopy(state)
    mismatched_state["_extra_state"]["seed"] += 1
    with pytest.raises(ValueError, match="checkpoint architecture mismatch"):
        restored.load_state_dict(mismatched_state, strict=True)


def test_engram_keeps_continuous_sequence_history():
    config = make_config()
    module = Engram(engram_config=config, tokenizer_lookup=_lookup())
    raw_ids = torch.tensor([[1, 2, 3, 4, 5]])
    hidden = torch.randn(5, 1, config.hidden_size)
    continuous = module(hidden, 0, module.compress_input_ids(raw_ids))
    separate = torch.cat(
        (
            module(hidden[:2], 0, module.compress_input_ids(raw_ids[:, :2])),
            module(hidden[2:], 0, module.compress_input_ids(raw_ids[:, 2:])),
        ),
        dim=0,
    )
    assert not torch.allclose(continuous[2:], separate[2:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_biases_initialization_and_selected_layer_gradients(dtype, device):
    config = make_config(params_dtype=dtype)
    module = Engram(engram_config=config, tokenizer_lookup=_lookup()).to(device)
    assert all(p.dtype == dtype for p in module.parameters())
    layer = module.layers["0"]
    assert layer.value_proj.bias is not None
    assert all(projection.bias is not None for projection in layer.key_projs)
    assert torch.count_nonzero(layer.short_conv.conv.weight).item() == 0

    hidden = torch.randn(4, 2, config.hidden_size, requires_grad=True, device=device, dtype=dtype)
    output = module(
        hidden,
        0,
        module.compress_input_ids(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)),
    )
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )
    assert all(parameter.grad is None for parameter in module.layers["2"].parameters())


@pytest.mark.parametrize(
    "invalid", ["missing", "missing_all", "unexpected", "shape", "dtype", "object"]
)
def test_row_factory_validates_all_entries_before_copy(invalid):
    layout = RowShardedTableLayout((3, 5), rank=0, world_size=1)
    owner = SimpleNamespace(layout=layout, replica_rank=0)
    target = torch.full((8, 4), -9.0)
    factory = RowShardedMultiHeadEmbedding._row_sharded_factory(owner, "engram.weight", target)
    loaded = {
        "_restore_target": target,
        "_restore_slices": factory.build()["_restore_slices"].unwrap(),
        "0": torch.full((12,), 1.0),
        "1": torch.full((20,), 2.0),
    }
    if invalid == "missing":
        del loaded["1"]
    elif invalid == "missing_all":
        del loaded["0"], loaded["1"]
    elif invalid == "unexpected":
        loaded["2"] = torch.ones(1, 4)
    elif invalid == "shape":
        loaded["1"] = torch.ones(4, 4)
    elif invalid == "dtype":
        loaded["1"] = loaded["1"].to(torch.bfloat16)
    else:
        loaded["1"] = object()
    message = (
        "required state"
        if invalid in ("missing", "missing_all", "unexpected")
        else (
            "shape/dtype mismatch"
            if invalid in ("shape", "dtype")
            else "checkpoint entry must be a tensor"
        )
    )
    with pytest.raises((ValueError, TypeError), match=message):
        factory.merge_fn(loaded)
    torch.testing.assert_close(target, torch.full_like(target, -9.0))


def test_row_factory_accepts_empty_owner_without_persisted_tables():
    owner = SimpleNamespace(
        layout=RowShardedTableLayout((1, 1), rank=0, world_size=2), replica_rank=0
    )
    target = torch.empty((0, 4))
    factory = RowShardedMultiHeadEmbedding._row_sharded_factory(owner, "engram.weight", target)
    assert set(factory.build()) == {"_restore_target", "_restore_slices"}
    assert factory.merge_fn({"_restore_target": target, "_restore_slices": ()}) is target


@pytest.mark.parametrize("start,stop", [(0, 32), (1, 31), (10, 18), (12, 12), (31, 32)])
def test_row_factory_flattened_state_restores_its_own_storage(start, stop, monkeypatch):
    from dataclasses import replace

    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensor

    owner = SimpleNamespace(layout=RowShardedTableLayout((3, 5), 0, 1), replica_rank=1)
    model = torch.full((8, 4), 9.0, dtype=torch.bfloat16)
    template = RowShardedMultiHeadEmbedding._row_sharded_factory(owner, "weight", model)
    target = torch.arange(start, stop, dtype=torch.float32)
    expected = target.clone()
    storage = target.untyped_storage().data_ptr()
    factory = replace(
        template, data=target, flattened_range=slice(start, stop), replica_id=(0, 0, 0)
    )

    def reject_cat(*args, **kwargs):
        raise AssertionError("Factory must not concatenate packed state")

    monkeypatch.setattr(torch, 'cat', reject_cat)
    built = factory.build()
    restored = {}
    covered = 0
    for key, value in built.items():
        if isinstance(value, LocalNonpersistentObject):
            restored[key] = value.unwrap()
        else:
            assert isinstance(value, ShardedTensor)
            assert value.data.untyped_storage().data_ptr() == storage
            assert value.data.dtype == torch.float32
            assert value.replica_id == (0, 0, 0)
            assert value.flattened_range is None
            table = int(value.key.rsplit('_', 1)[1])
            (offset,) = value.global_offset
            (length,) = value.local_shape
            logical = (
                torch.arange(32).view(8, 4)[0:3] if table == 0 else torch.arange(32).view(8, 4)[3:]
            )
            torch.testing.assert_close(
                value.data, logical.flatten()[offset : offset + length].float()
            )
            restored[key] = value.data.clone()
            covered += value.data.numel()
    assert covered == stop - start
    target.fill_(-1)
    result = factory.merge_fn(restored)
    assert result is target and result.untyped_storage().data_ptr() == storage
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(model, torch.full_like(model, 9.0), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_stream_matches_previous_layout_and_all_gradients(dtype):
    """Compare the former four-dimensional formula using identical nonzero weights."""
    module = Engram(engram_config=make_config(), tokenizer_lookup=_lookup()).cuda().to(dtype)
    layer = module.layers["0"]
    torch.nn.init.normal_(layer.short_conv.conv.weight, std=0.05)
    reference = copy.deepcopy(layer)
    hidden = torch.randn(5, 2, 8, device="cuda", dtype=dtype, requires_grad=True)
    embeddings = torch.randn(2, 5, 16, device="cuda", dtype=dtype, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_embeddings = embeddings.detach().clone().requires_grad_()
    actual = layer(hidden, embeddings)
    streams = ref_hidden.transpose(0, 1).unsqueeze(2)
    key = reference.key_projs[0](ref_embeddings)
    score = (reference.norm1[0](key) * reference.norm2[0](streams[:, :, 0])).sum(-1)
    score = score / math.sqrt(8)
    gate = (score.abs().clamp_min(1e-6).sqrt() * score.sign()).sigmoid().unsqueeze(-1)
    value = torch.stack([gate], dim=2) * reference.value_proj(ref_embeddings).unsqueeze(2)
    normalized = torch.cat([reference.short_conv.norms[0](value[:, :, 0])], dim=-1)
    conv = reference.short_conv.conv(normalized.transpose(1, 2))[..., :5]
    conv = F.silu(conv.transpose(1, 2).view(2, 5, 1, 8).contiguous())
    expected = ref_hidden + (value + conv).squeeze(2).transpose(0, 1).contiguous()
    sensitivity = torch.randn_like(actual)
    actual.backward(sensitivity)
    expected.backward(sensitivity)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(embeddings.grad, ref_embeddings.grad, rtol=0, atol=0)
    assert list(layer.state_dict()) == list(reference.state_dict())
    assert [n for n, _ in layer.named_parameters()] == [n for n, _ in reference.named_parameters()]
    for (name, param), (_, ref_param) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        if param.grad is None:
            assert ref_param.grad is None, name
        else:
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=0, atol=0, msg=name)
