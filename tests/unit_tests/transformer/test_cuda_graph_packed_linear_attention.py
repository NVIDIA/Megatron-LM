# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests: packed-THD GDN-family layers stay out of per-layer TE CUDA graphs.

With a sequence packing scheduler, a GDN / KDA layer checks its boundaries on the host and FLA
derives its chunk indices on the host, so the ``attn`` scope leaves those layers eager and still
captures the other attention layers. Layers are built without ``__init__``: the selection only
reads their type and their attention submodules.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.ssm.gated_delta_net import KimiDeltaAttention
from megatron.core.transformer.cuda_graphs import (
    _attention_runs_eager_under_packing,
    _layer_is_graphable,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import TransformerLayer


def _bare(cls, **attributes):
    obj = object.__new__(cls)
    for name, value in attributes.items():
        object.__setattr__(obj, name, value)
    return obj


def _attention_layer(attention):
    return _bare(
        TransformerLayer, self_attention=attention, cross_attention=IdentityOp(), mlp=IdentityOp()
    )


def _config(scheduler):
    return SimpleNamespace(
        cuda_graph_modules=[CudaGraphModule.attn], sequence_packing_scheduler=scheduler
    )


@pytest.mark.parametrize("wrapped", [False, True], ids=["plain", "mhc"])
@pytest.mark.parametrize("scheduler", [None, "dp_balanced", "default_dynamic_cp"])
def test_packed_kda_layers_stay_eager_and_other_attention_is_captured(wrapped, scheduler):
    kda = _bare(KimiDeltaAttention)
    other = torch.nn.Linear(1, 1)  # stands in for DSA / MLA attention
    config = _config(scheduler)

    def layer(attention):
        inner = _attention_layer(attention)
        return _bare(HyperConnectionHybridLayer, inner_layer=inner) if wrapped else inner

    assert _attention_runs_eager_under_packing(kda, config) is (scheduler is not None)
    assert not _attention_runs_eager_under_packing(other, config)
    assert _layer_is_graphable(layer(kda), config) is (scheduler is None)
    assert _layer_is_graphable(layer(other), config)
