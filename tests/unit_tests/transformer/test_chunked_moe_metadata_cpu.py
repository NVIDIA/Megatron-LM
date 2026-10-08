# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer import transformer_layer


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_chunked_moe_preserves_aligned_token_metadata(sequence_parallel, monkeypatch):
    local_length, batch_size = 5, 2
    group = SimpleNamespace(size=lambda: 2)
    config = SimpleNamespace(
        mlp_chunks_for_prefill=1,
        mlp_chunks_for_training=3,
        transformer_impl="transformer_engine",
        inference_fuse_tp_communication=False,
        sequence_parallel=sequence_parallel,
    )
    full_length = local_length * (2 if sequence_parallel else 1)
    ids = torch.arange(batch_size * full_length).view(batch_size, full_length)
    padding = ids.remainder(3).eq(0)
    expected_ids = ids[:, local_length:] if sequence_parallel else ids
    expected_padding = padding[:, local_length:] if sequence_parallel else padding
    packed = object()
    calls, scatters = [], []

    def scatter(value, group):
        assert group is layer.pg_collection.tp
        scatters.append(value.dtype)
        return value[local_length:]

    class RecordingMoE(torch.nn.Module):
        def forward(self, chunk, padding_mask, input_ids, packed_seq_params):
            assert packed_seq_params is packed
            assert input_ids.shape == padding_mask.shape == (batch_size, chunk.shape[0])
            calls.append((input_ids.clone(), padding_mask.clone()))
            result = chunk + input_ids.T.unsqueeze(-1)
            return result.masked_fill(padding_mask.T.unsqueeze(-1), 0), None

    layer = SimpleNamespace(
        config=config,
        training=True,
        is_moe_layer=True,
        recompute_mlp=False,
        mlp=RecordingMoE(),
        pg_collection=SimpleNamespace(tp=group),
    )
    monkeypatch.setattr(
        transformer_layer.tensor_parallel, "scatter_to_sequence_parallel_region", scatter
    )
    hidden = (
        torch.arange(local_length * batch_size * 3)
        .view(local_length, batch_size, 3)
        .float()
        .requires_grad_()
    )
    output, bias = transformer_layer.TransformerLayer._run_mlp(
        layer, hidden, hidden, padding, None, input_ids=ids, packed_seq_params=packed
    )
    expected = (hidden + expected_ids.T.unsqueeze(-1)).masked_fill(
        expected_padding.T.unsqueeze(-1), 0
    )
    torch.testing.assert_close(output, expected)
    assert bias is None
    assert [x.shape[1] for x, _ in calls] == [2, 2, 1]
    torch.testing.assert_close(torch.cat([x for x, _ in calls], dim=1), expected_ids)
    torch.testing.assert_close(torch.cat([x for _, x in calls], dim=1), expected_padding)
    assert len(scatters) == (2 if sequence_parallel else 0)
    output.sum().backward()
    torch.testing.assert_close(
        hidden.grad, (~expected_padding).T.unsqueeze(-1).expand_as(hidden).float()
    )


def test_chunked_moe_rejects_incompatible_partial_graph_output_contract():
    from megatron.core.transformer.transformer_config import TransformerConfig

    with pytest.raises(ValueError, match="Chunked MoE training does not support"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=2,
            mlp_chunks_for_training=2,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=["moe_router"],
            moe_token_dispatcher_type="alltoall",
        )
