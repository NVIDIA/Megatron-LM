# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model assembly boundaries: pipeline source ownership and release loading."""
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
from megatron.lite.model.deepseek_v41.lite.carrier import pack_sources, unpack_sources
from megatron.lite.model.deepseek_v41.lite.checkpoint import source_name
from megatron.lite.model.deepseek_v41.lite.engram import NgramHash
from megatron.lite.model.deepseek_v41.lite.protocol import ImplConfig, _validate, forward_step
from megatron.lite.model.deepseek_v41.lite.state import KVSourceState
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig


def test_pipeline_source_carrier_preserves_values_and_owner_gradients():
    hidden = torch.randn(5, 1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    pre = torch.sigmoid(torch.randn(5, 1, 4)).requires_grad_()
    owner = torch.randn(5, 8, requires_grad=True)
    prefix, new_prefix = torch.tensor([0, 2, 5], dtype=torch.int32), torch.tensor(
        [0, 2, 5], dtype=torch.int32
    )
    group, new_group = object(), object()
    state = KVSourceState()
    key = (3, ("source", prefix.data_ptr(), id(group), torch.bfloat16, torch.device("cpu")))
    pool = torch.tensor([[True, False], [False, True]])
    state.values = {key: (owner, prefix), ("candidate_pool", 3): pool}
    carried = pack_sources(hidden, pre, state, prefix, group)
    h, p, values = unpack_sources(carried, new_prefix, new_group)
    new_key = (
        3,
        ("source", new_prefix.data_ptr(), id(new_group), torch.bfloat16, torch.device("cpu")),
    )
    recovered, recovered_prefix = values[new_key]
    assert torch.equal(h, hidden) and torch.equal(p, pre)
    assert torch.equal(recovered, owner) and torch.equal(recovered_prefix, prefix)
    assert torch.equal(values[("candidate_pool", 3)], pool)
    # Two downstream readers contribute once each to the original owner.
    (h.float().sum() + p.sum() + recovered.sum() * 2 + recovered.sum() * 3).backward()
    assert torch.equal(owner.grad, torch.full_like(owner, 5))
    assert torch.equal(hidden.grad, torch.ones_like(hidden))
    assert torch.equal(pre.grad, torch.ones_like(pre))


def test_hash_resets_history_at_each_packed_document():
    hash_fn = NgramHash(
        torch.arange(16), 0, torch.tensor([[3, 5, 7]]), torch.tensor([[[17], [19]]])
    )
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    prefix = torch.tensor([0, 2, 5], dtype=torch.int32)
    packed = hash_fn(ids, cu_seqlens=prefix)
    expected = torch.cat([hash_fn(ids[:, :2]), hash_fn(ids[:, 2:])], dim=1)
    assert torch.equal(packed, expected)


@pytest.mark.parametrize(
    "option", [dict(quantized=True), dict(w4a8_experts=True), dict(dtype=torch.float32)]
)
def test_quantized_execution_rejected_before_model_allocation(option):
    with pytest.raises(ValueError, match="BF16"):
        _validate(DeepseekV41Config(num_hidden_layers=4), ImplConfig(**option))


@pytest.mark.parametrize("rank", [0, 1])
def test_protocol_real_core_metadata_contiguous_rows_and_document_end_mask(rank):
    from megatron.core.packed_seq_params import PackedSeqParams

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = SimpleNamespace(cp_size=2, cp_rank=rank, cp_group=object())

        def forward(self, input_ids, **kwargs):
            self.kwargs = kwargs
            self.ids = input_ids
            return {"log_probs": torch.zeros_like(input_ids, dtype=torch.float32)}

    model = Capture()
    ids = torch.arange(12)
    forward_step(model, PackedBatch(ids, ids, torch.tensor([5, 1, 6])))
    args = model.kwargs
    assert isinstance(args["packed_seq_params"], PackedSeqParams)
    assert args["packed_seq_params"].cp_partition_mode == "contiguous"
    assert args["packed_seq_params"].cp_group is model.ps.cp_group
    assert model.ids.numel() == 10  # native padding: 8 + 4 + 8, then contiguous CP
    assert args["full_input_ids"].numel() == 20
    # One-document tail and every physical padding row contribute no loss.
    mask = args["loss_mask"]
    expected = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] if rank == 0 else [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    )
    assert torch.equal(mask, expected)


def test_release_source_mapping_owner_shapes():
    assert (
        source_name("layers.3.attention.core_attention.main_projection.weight")
        == "layers.3.attn.compressor.wkv.weight"
    )
    assert (
        source_name("layers.3.attention.core_attention.native.indexer.compressor.wk.weight")
        == "layers.3.attn.indexer.wk.weight"
    )
    assert source_name("layers.3.ffn_mixes.alpha_pre") == "layers.3.hc_ffn_scale"
    with pytest.raises(KeyError, match="Unmapped"):
        source_name("unowned.weight")
