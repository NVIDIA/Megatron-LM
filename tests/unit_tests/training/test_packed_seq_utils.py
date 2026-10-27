# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batch entrypoints must prepare attention routes at the physical pack capacity."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.utils import packed_seq_utils


class _Group:
    def size(self):
        return 4

    def rank(self):
        return 0


@pytest.mark.parametrize(
    ("boundaries", "dynamic", "capacity"),
    [([0, 1024, 4096], False, None), ([0, 1024, 3000], False, 4096), ([0, 1024, 4096], True, None)],
    ids=["scheduler", "middle_pp_raw", "dynamic_graph_scheduler"],
)
def test_prepare_attention_routes_uses_physical_capacity(
    monkeypatch, boundaries, dynamic, capacity
):
    group = _Group()
    cu = torch.tensor(boundaries, dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    calls = []

    def finalize(params):
        calls.append(params)
        params.cp_group = group
        return params

    monkeypatch.setattr(packed_seq_utils, "finalize_packed_seq_params", finalize)
    config = SimpleNamespace(
        dsa_cp_balance_indexer=True,
        dsa_cp_balance_indexer_graph_dynamic_packs=dynamic,
        max_seqlen_per_dp_cp_rank=1024,
        context_parallel_size=4,
        pad_packed_seq_alignment="max",
        cuda_graph_impl="transformer_engine" if dynamic else "none",
        cuda_graph_modules=["attn"],
    )
    result = packed_seq_utils.prepare_packed_seq_params(packed, config, capacity=capacity)
    assert result is packed and calls == [packed]
    if dynamic:
        from megatron.core.transformer.experimental_attention_variant.cp_balanced_indexer import (
            get_graph_dynamic_plan,
        )

        assert get_graph_dynamic_plan(packed)["half"] == 512
        assert not hasattr(packed, "_dsa_cp_balance_layout_cache")
    else:
        assert packed._dsa_cp_balance_layout_cache["zz_pack_ok"] == (1024, True)
        assert packed._dsa_cp_balance_layout_cache[("zigzag", 0)]["half"] == 512


def test_unpacked_batch_needs_no_attention_config(monkeypatch):
    monkeypatch.setattr(packed_seq_utils, "finalize_packed_seq_params", lambda params: params)
    assert packed_seq_utils.prepare_packed_seq_params(None, SimpleNamespace()) is None
