# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Retained dev validation coverage for inferred balanced dynamic-pack routes."""

import pytest
import torch


def _make_balanced_dynamic_pack_config(monkeypatch, **overrides):
    """Build a config that infers dynamic-pack routing while isolating backend probes."""
    from megatron.core.transformer import transformer_config as transformer_config_module
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels
    from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
        _make_config,
    )

    monkeypatch.setattr(dsa_kernels, "use_fused_dsa_kernels", lambda _config: True)
    monkeypatch.setattr(transformer_config_module, "HAVE_PACKAGING", True)
    monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
    kwargs = dict(
        context_parallel_size=2,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="dp_balanced",
        max_seqlen_per_dp_cp_rank=128,
        pad_packed_seq_alignment="max",
        thd_max_packed_sequences=8,
        csa_compress_ratios=[4, 4, 4, 4],
        csa_dense_mode=False,
        dsa_kernel_backend="none",
        dsa_cp_balance_indexer=True,
        cuda_graph_impl="transformer_engine",
        cuda_graph_dynamic_microbatches=True,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return _make_config(**kwargs)


@pytest.mark.internal
@pytest.mark.parametrize("virtual_pipeline_size", [None, 2])
def test_balanced_dynamic_packs_are_inferred_for_attention_graph_with_pp_vpp(
    monkeypatch, virtual_pipeline_size
):
    config = _make_balanced_dynamic_pack_config(
        monkeypatch, virtual_pipeline_model_parallel_size=virtual_pipeline_size
    )

    assert config.pipeline_model_parallel_size == 2
    assert config.virtual_pipeline_model_parallel_size == virtual_pipeline_size
    assert config.dsa_cp_balance_indexer_graph_dynamic_packs


@pytest.mark.internal
def test_balanced_dynamic_packs_allows_verified_safe_row_limit_boundary(monkeypatch):
    """Each half-call may contain exactly 32768 rows; only larger calls are unsafe."""
    config = _make_balanced_dynamic_pack_config(monkeypatch, max_seqlen_per_dp_cp_rank=2 * 32768)

    assert config.max_seqlen_per_dp_cp_rank // 2 == 32768


@pytest.mark.internal
def test_balanced_static_pack_graph_keeps_pp_rejection(monkeypatch):
    with pytest.raises(ValueError, match="dynamic-pack routing is inferred"):
        _make_balanced_dynamic_pack_config(monkeypatch, cuda_graph_impl="local")


@pytest.mark.internal
def test_balanced_static_pack_pp_allows_graph_scope_outside_attention(monkeypatch):
    config = _make_balanced_dynamic_pack_config(monkeypatch, cuda_graph_modules=["mlp"])

    assert config.pipeline_model_parallel_size == 2
    assert not config.dsa_cp_balance_indexer_graph_dynamic_packs


@pytest.mark.internal
@pytest.mark.parametrize(
    "overrides",
    [
        {"dsa_cp_balance_indexer": False},
        {"cuda_graph_impl": "local"},
        {"cuda_graph_modules": ["mlp"]},
        {"sequence_packing_scheduler": "default_dynamic_cp"},
    ],
)
def test_balanced_dynamic_packs_require_all_inferred_conditions(monkeypatch, overrides):
    config = _make_balanced_dynamic_pack_config(
        monkeypatch,
        pipeline_model_parallel_size=1,
        cuda_graph_dynamic_microbatches=False,
        **overrides,
    )

    assert not config.dsa_cp_balance_indexer_graph_dynamic_packs


@pytest.mark.internal
def test_balanced_dynamic_packs_cannot_be_set_as_a_constructor_option(monkeypatch):
    with pytest.raises(TypeError, match="dsa_cp_balance_indexer_graph_dynamic_packs"):
        _make_balanced_dynamic_pack_config(
            monkeypatch, dsa_cp_balance_indexer_graph_dynamic_packs=True
        )


@pytest.mark.internal
@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"dynamic_context_parallel": True},
            "Dynamic context parallelism requires sequence_packing_scheduler=default_dynamic_cp",
        ),
        (
            {"cuda_graph_dynamic_microbatches": False},
            "with PP/VPP requires cuda_graph_dynamic_microbatches=True",
        ),
        (
            {"max_seqlen_per_dp_cp_rank": None, "pad_packed_seq_alignment": None},
            "max_seqlen_per_dp_cp_rank must be set when sequence_packing_scheduler",
        ),
        ({"max_seqlen_per_dp_cp_rank": 127}, "requires an even max_seqlen_per_dp_cp_rank"),
        ({"max_seqlen_per_dp_cp_rank": 0}, "requires a positive max_seqlen_per_dp_cp_rank"),
        ({"max_seqlen_per_dp_cp_rank": -2}, "requires a positive max_seqlen_per_dp_cp_rank"),
        ({"max_seqlen_per_dp_cp_rank": 65538}, "above the verified-safe limit"),
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not yet support overlap_moe_expert_parallel_comm or delay_wgrad_compute",
        ),
        (
            {"delay_wgrad_compute": True},
            "does not yet support overlap_moe_expert_parallel_comm or delay_wgrad_compute",
        ),
    ],
)
def test_balanced_dynamic_packs_validate_inferred_contract(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _make_balanced_dynamic_pack_config(monkeypatch, **overrides)
