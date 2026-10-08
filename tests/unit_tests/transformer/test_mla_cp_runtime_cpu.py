# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the real MLA forward boundary before GPU-dependent projections."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_latent_attention as mla_module
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)


@pytest.mark.parametrize(
    "attention_cls", [mla_module.MultiLatentAttention, AbsorbedMLASelfAttention]
)
def test_forward_binds_dynamic_cp_singleton_before_projection(attention_cls, monkeypatch):
    static_group = SimpleNamespace(size=lambda: 4, rank=lambda: 0)
    singleton = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    packed = PackedSeqParams(local_cp_size=1, cp_singleton_group=singleton)
    layer = SimpleNamespace(
        training=True,
        cache_mla_latents=False,
        config=SimpleNamespace(cache_mla_latents=False),
        pg_collection=SimpleNamespace(cp=static_group),
        _build_time_cp_group=static_group,
        offload_qkv_linear=False,
    )

    class ProjectionReached(Exception):
        pass

    def project(*args, **kwargs):
        assert layer.pg_collection.cp is singleton
        assert packed.cp_group is None
        raise ProjectionReached

    layer.get_query_key_value_tensors = project
    monkeypatch.setattr(
        mla_module, "off_interface", lambda enabled, value, name: nullcontext(value)
    )
    with pytest.raises(ProjectionReached):
        attention_cls.forward(layer, torch.zeros(2, 1, 4), None, packed_seq_params=packed)
