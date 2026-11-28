# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed metadata-only stages must plan the same buffers as data-owning stages."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import PipelinePayloadPlan
from megatron.training.datasets.v41_pipeline import (
    batch_shape,
    normalize_v41_batch,
    prepare_v41_pipeline_inputs,
)


@pytest.mark.parametrize("scheduled", [False, True])
def test_packed_plans_preserve_logical_lengths_and_consume_batches_once(scheduled):
    logical = torch.tensor([0, 5, 11], dtype=torch.int32)
    padded = torch.tensor([0, 8, 16], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=padded,
        cu_seqlens_kv=padded,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        total_tokens=16,
    )
    args = SimpleNamespace(
        context_parallel_size=2,
        sft=not scheduled,
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler='dp_balanced' if scheduled else None,
    )
    batches = []
    for owns_data in (True, False):
        batch = ContextParallelBatch.from_single_layout(
            'contiguous',
            dict(
                cu_seqlens=logical, tokens=torch.ones(1, 8, dtype=torch.long) if owns_data else None
            ),
            params,
        )
        batches.append(normalize_v41_batch(batch))
    assert batch_shape(batches[0], args) == batch_shape(batches[1], args)
    assert torch.equal(batches[0].get_packed_seq_params().cu_seqlens_q, logical)
    model = SimpleNamespace(
        vp_stage=0,
        config=SimpleNamespace(pipeline_model_parallel_size=2),
        pipeline_stage=SimpleNamespace(
            plan=lambda shapes: PipelinePayloadPlan(tuple(shapes), tuple(shapes))
        ),
    )
    reads = []

    def get_batch(iterator, stage):
        reads.append(stage)
        return next(iterator)

    prepared = prepare_v41_pipeline_inputs(iter(batches), model, 2, args=args, get_batch=get_batch)
    assert reads == [0, 0]
    assert prepared.pipeline_payload_plan.incoming[0] == prepared.pipeline_payload_plan.incoming[1]
    assert list(prepared) == batches
    assert reads == [0, 0]


def test_released_map_roundtrip_preserves_layer_keys_and_padding(tmp_path):
    """The CLI artifact and the legacy tensor address the same rows after vocab padding."""
    from megatron.core.models.deepseek_v41.engram import released_engram_config
    from tools.engram.convert_v41_tokenizer_map import convert

    options = dict(
        layer_ids=[0, 3],
        num_embeddings=[60, 120],
        max_ngram_size=3,
        vocab_size=11,
        n_heads=2,
        head_dim=4,
        pad_token_id=2,
        compressed_vocab_size=32,
    )
    tensor, output = tmp_path / "tokens.pt", tmp_path / "tokens-v41.json"
    torch.save(torch.arange(32), tensor)
    legacy = released_engram_config(options, str(tensor))
    convert(options, tensor, output, padded_vocab_size=40)
    shared = released_engram_config(options, str(output))
    legacy.validate_vocabulary(40)
    assert shared.layer_ids == (1, 7)
    assert shared.multipliers(7) == legacy.multipliers(7)
    torch.testing.assert_close(shared.tokenizer_remap, legacy.tokenizer_remap)
    assert shared.tokenizer_remap[32:].tolist() == [2] * 8


def test_builder_and_transformer_share_engram_config():
    """Defaults and explicit configs must reach the common distributed builder."""
    from megatron.training.models.deepseek_v41 import DeepSeekV41ModelConfig

    memory = object()
    transformer = SimpleNamespace(engram_config=memory)
    inherited = DeepSeekV41ModelConfig(transformer=transformer)
    assert inherited.engram_config is transformer.engram_config is memory
    replacement = object()
    explicit = DeepSeekV41ModelConfig(transformer=transformer, engram_config=replacement)
    assert explicit.engram_config is transformer.engram_config is replacement
