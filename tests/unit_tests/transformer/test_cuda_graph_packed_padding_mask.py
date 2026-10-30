# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""THD padding masks across the TE per-layer CUDA graph boundary.

Under static packed (THD) inputs a graphed layer rebuilds ``PackedSeqParams`` from tensor
inputs. Attention that reads ``PackedSeqParams.padding_mask`` (QSA's indexer loss keeps THD
padding rows out of its KL) must therefore get the mask as a graph input, or the captured
layer computes a different loss than the eager one.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.transformer_layer import TransformerLayer


def _layer(*, router=False, attn=True, thd=True, sp=False, consumes=True):
    return SimpleNamespace(
        _moe_router_in_cuda_graph=lambda: router,
        _attention_in_cuda_graph=lambda: attn,
        _is_thd_cuda_graph=lambda: thd,
        config=SimpleNamespace(
            sequence_parallel=sp, max_seqlen_per_dp_cp_rank=16, context_parallel_size=1
        ),
        self_attention=SimpleNamespace(consumes_packed_padding_mask=consumes),
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(), True),
        (dict(router=True, attn=False, consumes=False), True),
        (dict(consumes=False), False),
        (dict(thd=False), False),
        (dict(attn=False), False),
        (dict(sp=True), False),
    ],
)
def test_padding_mask_graph_input_predicate(kwargs, expected):
    assert TransformerLayer._padding_mask_in_cuda_graph(_layer(**kwargs)) is expected


@pytest.mark.parametrize("sp", [False, True])
def test_rebuilt_packed_seq_params_carry_the_mask(sp):
    cu = torch.tensor([0, 10, 16], dtype=torch.int32)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[:, 10:] = True
    kwargs = dict(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        padding_mask=mask,
    )
    TransformerLayer._reconstruct_packed_seq_params_from_kwargs(_layer(sp=sp), kwargs)
    params = kwargs["packed_seq_params"]
    if sp:
        # The layer's mask is SP-scattered there, not the whole local sequence.
        assert params.padding_mask is None
    else:
        assert params.padding_mask is mask
    # The layer-level kwarg stays for the MoE router.
    assert kwargs["padding_mask"] is mask


def test_input_layouts_are_recorded_once_from_any_path():
    """The hybrid hyper-connection wrapper calls the layer's raw branch directly; it records
    the graph-input layouts through the same method as _forward_attention."""
    layer = SimpleNamespace(training=True, _padding_mask_shape_seen=None, _rotary_pos_emb_seen=None)
    rotary = torch.zeros(3, 1, 64, 16, dtype=torch.float32)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    TransformerLayer._record_cuda_graph_input_layouts(layer, rotary, mask)
    assert layer._rotary_pos_emb_seen == ((3, 1, 64, 16), torch.float32)
    assert layer._padding_mask_shape_seen == (1, 64)
    # Later calls do not overwrite the first layouts.
    TransformerLayer._record_cuda_graph_input_layouts(layer, torch.zeros(2, 2), None)
    assert layer._rotary_pos_emb_seen == ((3, 1, 64, 16), torch.float32)
