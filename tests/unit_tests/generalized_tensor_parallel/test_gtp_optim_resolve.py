# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-level tests for the GTP fallback in the fully-reshardable optimizer save path.

``sharded_param_state_fully_reshardable`` matches grad-buffer parameters to model
ShardedTensors by object identity. GTP breaks that identity in two ways — a dequantized
BF16 copy for native-FP8 weights, and a factory exposing the gathered tensor for fused
projections — and ``_resolve_gtp_sharded_metadata`` is the resolver for both. These tests
pin its contract without GPUs or process groups; the end-to-end save/load is covered by
the distributed GTP DCP suite.
"""

import types

import pytest
import torch

pytest.importorskip("transformer_engine")

from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory  # noqa: E402
from megatron.core.optimizer.distrib_optimizer import _resolve_gtp_sharded_metadata  # noqa: E402


def _gtp_param(shape=(4, 2)):
    param = torch.nn.Parameter(torch.zeros(shape))
    # What is_gtp_param() keys off; set directly to avoid building process groups.
    param.is_gtp_weight_remat = True
    return param


def _factory(key):
    return ShardedTensorFactory(
        key, torch.zeros(8, 2), build_fn=lambda *a: {}, merge_fn=lambda sd: sd, replica_id=0
    )


def test_returns_none_for_a_plain_parameter():
    """A non-GTP param is not this resolver's business; the caller raises its own error."""
    param = torch.nn.Parameter(torch.zeros(4, 2))
    assert _resolve_gtp_sharded_metadata(param, {"w": _factory("w")}) is None


def test_follows_the_dequant_backlink():
    """A dequantized BF16 copy resolves through its _gtp_dequant_src backlink."""
    param = _gtp_param()
    dequant_copy = torch.zeros(4, 2, dtype=torch.bfloat16)
    dequant_copy._gtp_dequant_src = param
    entry = types.SimpleNamespace(data=dequant_copy)
    sharded = {"decoder.weight": entry, "other": _factory("other")}

    assert _resolve_gtp_sharded_metadata(param, sharded) is entry


def test_matches_a_factory_by_stripped_name():
    """A factory-backed weight (GDN/Mamba in_proj) resolves by name, module. prefix stripped."""
    param = _gtp_param()
    param._debug_name = "module.decoder.layers.0.mixer.in_proj.weight"
    factory = _factory("decoder.layers.0.mixer.in_proj.weight")
    sharded = {"in_proj": factory}

    assert _resolve_gtp_sharded_metadata(param, sharded) is factory


def test_refuses_the_rebuild_for_an_expert_parallel_param():
    """The per-shard rebuild is EP-unaware; an expert param must fail loudly, not corrupt."""
    param = _gtp_param()
    param._debug_name = "decoder.layers.0.mlp.experts.weight1"
    param.allreduce = False

    with pytest.raises(ValueError, match="expert-parallel"):
        _resolve_gtp_sharded_metadata(param, {})
