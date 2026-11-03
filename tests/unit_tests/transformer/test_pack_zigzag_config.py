# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Config and CLI coverage for the DSv4 pack-level zigzag context-parallel layout."""

from argparse import ArgumentParser

import pytest


def _make_pack_zigzag_config(monkeypatch, **overrides):
    """Build a CPU DSv4 config that passes the pack-zigzag startup checks by default."""
    from megatron.core.transformer import transformer_config as transformer_config_module
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    # Same isolation as test_balanced_dynamic_pack_config.py: the fused-backend predicate is
    # imported lazily inside __post_init__, so patching the module attribute is enough, and a
    # real 'cudnn' backend would need a CUDA device at config time.
    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda *_a, **_k: True)
    kwargs = dict(
        context_parallel_size=2,
        cp_partition_mode="contiguous",
        dsa_cp_pack_zigzag=True,
        dsa_indexer_use_sparse_loss=True,
    )
    kwargs.update(overrides)
    return _make_config(**kwargs)


def test_pack_zigzag_cli_flag_is_generated_from_the_config_field():
    from megatron.training.arguments import _add_network_size_args

    parser = _add_network_size_args(ArgumentParser(exit_on_error=False))
    assert parser.parse_args([]).dsa_cp_pack_zigzag is False
    assert parser.parse_args(["--dsa-cp-pack-zigzag"]).dsa_cp_pack_zigzag is True


@pytest.mark.internal
def test_pack_zigzag_config_is_accepted_with_contiguous_static_cp(monkeypatch):
    config = _make_pack_zigzag_config(monkeypatch)

    assert config.dsa_cp_pack_zigzag
    assert config.cp_partition_mode == "contiguous"
    assert config.attention_cp_layout == "contiguous"
    assert not config.dsa_cp_balance_indexer


@pytest.mark.internal
def test_pack_zigzag_requires_dsv4_hybrid():
    from megatron.core.transformer import TransformerConfig

    with pytest.raises(ValueError, match="requires experimental_attention_variant='dsv4_hybrid'"):
        TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, dsa_cp_pack_zigzag=True
        )


@pytest.mark.internal
def test_pack_zigzag_requires_fused_dsa_backend(monkeypatch):
    """Without the fused predicate patched, _make_config's dsa_kernel_backend='none' is rejected."""
    from megatron.core.transformer import transformer_config as transformer_config_module
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda *_a, **_k: True)
    with pytest.raises(ValueError, match="requires the fused DSA kernel backend"):
        _make_config(
            context_parallel_size=2,
            cp_partition_mode="contiguous",
            dsa_cp_pack_zigzag=True,
            dsa_indexer_use_sparse_loss=True,
        )


@pytest.mark.internal
@pytest.mark.parametrize(
    "overrides,match",
    [
        # CP=1 so dev's earlier "DSv4 Hybrid with context parallelism requires
        # cp_partition_mode='contiguous'" check does not fire first.
        (
            {"context_parallel_size": 1, "cp_partition_mode": "zigzag"},
            "keeps the data split contiguous",
        ),
        ({"dsa_cp_balance_indexer": True}, "disable dsa_cp_balance_indexer"),
        # dynamic CP auto-selects the default_dynamic_cp scheduler, which needs a capacity and
        # per-token loss before the pack-zigzag block runs; CP=1 so no other check fires first.
        (
            {
                "context_parallel_size": 1,
                "dynamic_context_parallel": True,
                "sequence_packing_scheduler": "default_dynamic_cp",
                "max_seqlen_per_dp_cp_rank": 4096,
                "calculate_per_token_loss": True,
            },
            "does not support dynamic context parallelism",
        ),
        ({"context_parallel_size": 1}, "requires context_parallel_size > 1"),
        ({"cuda_graph_impl": "full_iteration"}, "incompatible with full-iteration CUDA graphs"),
        ({"dsa_indexer_use_sparse_loss": False}, "sparse indexer loss only"),
    ],
)
def test_pack_zigzag_startup_rejections(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_pack_zigzag_config(monkeypatch, **overrides)
