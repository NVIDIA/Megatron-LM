# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compose the n-gram memory into a hybrid stack spec."""

from types import SimpleNamespace

import pytest

from megatron.core.models.engram import apply_engram_to_hybrid_stack_spec
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

from ._test_utils import write_tokenizer_map


def _engram_config(tmp_path, layer_ids=(2,)):
    artifact = write_tokenizer_map(tmp_path / "map.json", vocab_size=16, layer_ids=layer_ids)
    return EngramConfig(
        global_vocab_sizes=(17, 19),
        layer_ids=layer_ids,
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
    )


def _apply(tmp_path, pattern, layer_ids=(2,), enable_hyper_connections=False):
    transformer_config = SimpleNamespace(enable_hyper_connections=enable_hyper_connections)
    return apply_engram_to_hybrid_stack_spec(
        hybrid_stack_spec, _engram_config(tmp_path, layer_ids), pattern, transformer_config
    )


def test_engram_is_attached_to_the_selected_layer_type_only(tmp_path):
    # "M*M-": layer 2 is attention, layer 4 is MLP.
    spec = _apply(tmp_path, "M*M-", layer_ids=(2,))
    assert spec.submodules.attention_layer.submodules.engram.module is Engram
    # Mamba layers have no composition point at all, and untouched types keep theirs unset.
    assert not hasattr(spec.submodules.mamba_layer.submodules, "engram")
    assert (
        spec.submodules.mlp_layer.submodules.engram
        is not spec.submodules.attention_layer.submodules.engram
    )

    spec = _apply(tmp_path, "M*M-", layer_ids=(2, 4))
    assert spec.submodules.attention_layer.submodules.engram.module is Engram
    assert spec.submodules.mlp_layer.submodules.engram.module is Engram


def test_shared_stack_spec_singleton_is_not_mutated(tmp_path):
    before = hybrid_stack_spec.submodules.attention_layer.submodules.engram
    _apply(tmp_path, "M*M*", layer_ids=(2,))
    # Attaching in place would carry a stale EngramConfig into every later model in the process.
    assert hybrid_stack_spec.submodules.attention_layer.submodules.engram is before


def test_layer_id_on_a_type_that_cannot_carry_the_memory_is_rejected(tmp_path):
    # Mamba exposes no composition point.
    with pytest.raises(ValueError, match="assigns to a 'M' layer"):
        _apply(tmp_path, "M*M*", layer_ids=(1,))
    # 'W' (DSv4 sliding window) is IdentityOp in this stack spec, so it would build nothing.
    # Window layers cannot coexist with '*' attention, hence the Mamba pairing.
    with pytest.raises(ValueError, match="assigns to a 'W' layer"):
        _apply(tmp_path, "MWMW", layer_ids=(2,))


def test_pipeline_separators_and_mtp_do_not_shift_layer_numbering(tmp_path):
    # Pipe symbols are stage boundaries, not layers: layer 3 is still Mamba.
    with pytest.raises(ValueError, match="assigns to a 'M' layer"):
        _apply(tmp_path, "M*|M*", layer_ids=(3,))
    _apply(tmp_path, "M*|M*", layer_ids=(4,))

    with pytest.raises(ValueError, match="describes only 4 layers"):
        _apply(tmp_path, "M*M*", layer_ids=(5,))
    with pytest.raises(ValueError, match="requires --hybrid-layer-pattern"):
        _apply(tmp_path, None, layer_ids=(2,))


def test_multi_token_prediction_is_rejected(tmp_path):
    # The nested MTP stack is built from these same submodules, and HybridStack drops
    # is_mtp_layer for '-', 'G' and 'K', so an MTP layer would build a second memory.
    with pytest.raises(ValueError, match="multi-token prediction on the hybrid path"):
        _apply(tmp_path, "M*M*/M-", layer_ids=(2,))


def test_hyper_connections_are_rejected(tmp_path):
    # HyperConnectionHybridLayer's fast path calls the layer's attention/MLP helpers directly and
    # never reaches _forward_attention, so the memory would be built but never applied.
    with pytest.raises(ValueError, match="hyper-connections on the hybrid path"):
        _apply(tmp_path, "M*M*", layer_ids=(2,), enable_hyper_connections=True)
