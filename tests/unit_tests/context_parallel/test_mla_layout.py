# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.multi_latent_attention import MultiLatentAttention


class _ReachedProjection(Exception):
    """Stop after the production layout guard, before expensive MLA computation."""


@pytest.mark.parametrize("mla", [MultiLatentAttention, AbsorbedMLASelfAttention])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("input_layout", ["zigzag", "contiguous"])
def test_mla_checks_actual_input_layout_before_rope(mla, packed, cp_size, input_layout):
    def projection(*args, **kwargs):
        raise _ReachedProjection

    group = SimpleNamespace(size=lambda: cp_size)
    model = SimpleNamespace(
        config=SimpleNamespace(cp_partition_mode="contiguous", cache_mla_latents=False),
        cache_mla_latents=False,
        training=False,
        pg_collection=SimpleNamespace(cp=group),
        get_query_key_value_tensors=projection,
    )
    # The manager annotation takes precedence over config for SBHD. Packed input's
    # metadata takes precedence over the module annotation for THD.
    if not packed:
        model._cp_input_partition_mode = input_layout
    else:
        model._cp_input_partition_mode = "zigzag"
    params = PackedSeqParams(qkv_format="thd", cp_partition_mode=input_layout) if packed else None
    error = ValueError if cp_size > 1 and input_layout == "contiguous" else _ReachedProjection
    with pytest.raises(error):
        mla.forward(model, torch.zeros(4, 1, 8), None, packed_seq_params=params)


@pytest.mark.parametrize("mla", [MultiLatentAttention, AbsorbedMLASelfAttention])
def test_mla_sbhd_uses_config_without_manager_annotation(mla):
    model = SimpleNamespace(
        config=SimpleNamespace(cp_partition_mode="contiguous", cache_mla_latents=False),
        cache_mla_latents=False,
        training=False,
        pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 2)),
    )
    with pytest.raises(ValueError, match="requires cp_partition_mode='zigzag'"):
        mla.forward(model, torch.zeros(4, 1, 8), None)
