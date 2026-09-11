# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.config import (
    EngramConfig,
    _validate_packed_sequence_args,
    allocate_table_sizes,
)
from megatron.core.models.engram.hashing import (
    build_ngram_hashes,
    compress_token_ids,
    slice_hashes_for_sequence_parallel,
)
from tools.engram.generate_tokenizer_map import build_remap

from ._test_utils import write_tokenizer_map


class _FakeTokenizer:
    def __init__(self, decoded_tokens, raw_tokens):
        self.decoded_tokens = decoded_tokens
        self.raw_tokens = raw_tokens

    def __len__(self):
        return len(self.decoded_tokens)

    def decode(self, token_ids, skip_special_tokens=False):
        assert not skip_special_tokens
        assert len(token_ids) == 1
        return self.decoded_tokens[token_ids[0]]

    def convert_ids_to_tokens(self, token_id):
        return self.raw_tokens[token_id]


def test_official_tokenizer_compression_uses_first_normalized_occurrence():
    tokenizer = _FakeTokenizer(
        decoded_tokens=["É", "e\u0301", " \t\n", " ", "", "�"],
        raw_tokens=["tok0", "tok1", "tok2", "tok3", "tok4", "raw-invalid"],
    )

    remap, compressed_vocab_size = build_remap(tokenizer)

    assert remap == [0, 0, 1, 1, 2, 3]
    assert compressed_vocab_size == 4


def test_official_fixed_hash_vector():
    input_ids = torch.tensor([[1, 4, 2, 7]], dtype=torch.int64)
    hashes = build_ngram_hashes(
        input_ids=input_ids,
        tokenizer_remap=torch.arange(16, dtype=torch.int64),
        multipliers=torch.tensor([13, 17, 19], dtype=torch.int64),
        table_sizes=torch.tensor([11, 13, 17, 19], dtype=torch.int64),
        max_ngram_order=3,
        num_hash_heads=2,
        boundary_token_id=3,
    )
    expected = torch.tensor([[[7, 10, 7, 7], [4, 11, 11, 9], [6, 3, 9, 1], [0, 4, 2, 15]]])
    torch.testing.assert_close(hashes, expected)


def test_compression_preserves_negative_sentinels():
    remap = torch.tensor([0, 4, 1, 3, 2], dtype=torch.int64)
    input_ids = torch.tensor([[1, -100, 4, 2]], dtype=torch.int64)
    torch.testing.assert_close(
        compress_token_ids(input_ids, remap), torch.tensor([[4, -100, 2, 1]])
    )


def test_prime_allocation_is_global_and_distinct():
    sizes = allocate_table_sizes((10, 10), (1, 5), 2)
    assert sizes == {1: (11, 13, 17, 19), 5: (23, 29, 31, 37)}
    assert len(set(sizes[1] + sizes[5])) == 8


def test_full_hash_then_sequence_parallel_slice(monkeypatch):
    hashes = torch.arange(2 * 8 * 3).view(2, 8, 3)
    monkeypatch.setattr("megatron.core.models.engram.hashing.get_pg_size", lambda _: 4)
    monkeypatch.setattr("megatron.core.models.engram.hashing.get_pg_rank", lambda _: 2)
    local = slice_hashes_for_sequence_parallel(hashes, local_sequence_length=2, tp_group=object())
    torch.testing.assert_close(local, hashes[:, 4:6])


def _packed_args(**overrides):
    values = dict(
        sft=False,
        use_varlen_dataset=False,
        varlen_sbhd_validation=False,
        sequence_packing_scheduler=None,
        pipeline_model_parallel_size=1,
        max_seqlen_per_dp_cp_rank=0,
        seq_length=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_packed_sequence_detection_matches_pretrain_gpt():
    assert not _validate_packed_sequence_args(_packed_args())
    assert _validate_packed_sequence_args(_packed_args(sft=True))
    # SBHD validation reads the varlen dataset but keeps the dense layout, so it is not packed.
    assert not _validate_packed_sequence_args(
        _packed_args(use_varlen_dataset=True, varlen_sbhd_validation=True)
    )
    with pytest.raises(ValueError, match="sequence-packing schedulers"):
        _validate_packed_sequence_args(_packed_args(sequence_packing_scheduler="first_fit"))
    with pytest.raises(ValueError, match="max-seqlen-per-dp-cp-rank"):
        _validate_packed_sequence_args(
            _packed_args(
                sft=True,
                pipeline_model_parallel_size=2,
                max_seqlen_per_dp_cp_rank=256,
                seq_length=512,
            )
        )


def _startup_transformer_config(**overrides):
    values = dict(
        num_layers=8,
        context_parallel_size=1,
        bf16=True,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
        recompute_granularity=None,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
        transformer_impl="transformer_engine",
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        overlap_moe_expert_parallel_comm=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_invalid_configuration_messages(tmp_path):
    artifact = write_tokenizer_map(tmp_path / "map.json", vocab_size=16, layer_ids=(1,))
    with pytest.raises(ValueError, match="exactly one value"):
        EngramConfig(
            global_vocab_sizes=(17,),
            layer_ids=(1,),
            max_ngram_order=3,
            num_hash_heads=2,
            memory_dim=8,
            kernel_size=4,
            hash_seed=0,
            boundary_token_id=0,
            tokenizer_map_path=str(artifact),
        )

    config = EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
    )
    with pytest.raises(ValueError, match="context_parallel_size == 1"):
        config.validate_startup(_startup_transformer_config(context_parallel_size=2), 16)
    with pytest.raises(ValueError, match="expert_tensor_parallel_size == tensor_model_parallel"):
        config.validate_startup(
            _startup_transformer_config(
                tensor_model_parallel_size=2, expert_tensor_parallel_size=1
            ),
            16,
        )
    with pytest.raises(ValueError, match="BF16 parameters"):
        config.validate_startup(_startup_transformer_config(bf16=False), 16)
    with pytest.raises(ValueError, match="FP4"):
        config.validate_startup(_startup_transformer_config(fp4="e2m1"), 16)
    # FP8/MXFP8 recipes and MTP layers are permitted (Engram modules stay in BF16;
    # MTP layers never build Engram).
    config.validate_startup(_startup_transformer_config(fp8="mxfp8"), 16)
    config.validate_startup(_startup_transformer_config(mtp_num_layers=1), 16)
    with pytest.raises(ValueError, match="Packed sequences are not supported by the deepseek"):
        config.validate_startup(_startup_transformer_config(), 16, packed_sequences=True)
    with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm"):
        config.validate_startup(
            _startup_transformer_config(overlap_moe_expert_parallel_comm=True), 16
        )
    # The convolution history must fit in one sequence-parallel slice.
    with pytest.raises(ValueError, match="at least 9 positions"):
        config.validate_startup(
            _startup_transformer_config(
                sequence_parallel=True, tensor_model_parallel_size=4, expert_tensor_parallel_size=4
            ),
            16,
            sequence_length=16,
        )
    config.validate_startup(
        _startup_transformer_config(
            sequence_parallel=True, tensor_model_parallel_size=4, expert_tensor_parallel_size=4
        ),
        16,
        sequence_length=64,
    )
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        config.validate_startup(_startup_transformer_config(), 17)
