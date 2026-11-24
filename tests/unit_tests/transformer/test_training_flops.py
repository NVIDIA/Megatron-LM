# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward and backward FLOPs reflect the layers constructed for training."""

from types import SimpleNamespace

import pytest

from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, DSAttention
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from tests.unit_tests.test_utilities import make_flops_test_layer

BATCH_STATS = dict(total_real_tokens_in_batch=4, seqlen_squared_sum_in_batch=16)


def _config(loss_coeff=0.01):
    return SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_rank=3,
        kv_lora_rank=4,
        qk_head_dim=2,
        qk_pos_emb_head_dim=2,
        v_head_dim=3,
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=2,
        dsa_indexer_topk=16,
        dsa_indexer_kpool=1,
        dsa_indexer_kpool_always_select_tail=True,
        dsa_indexer_loss_coeff=loss_coeff,
        dsa_indexer_use_sparse_loss=False,
    )


@pytest.mark.parametrize(
    "loss_coeff,include_backward,expected",
    [(0.0, False, 416), (0.01, False, 416), (0.0, True, 416), (0.01, True, 896)],
)
def test_indexer_obeys_its_backward_rule(loss_coeff, include_backward, expected):
    indexer = make_flops_test_layer(DSAIndexer, _config(loss_coeff))
    # Forward costs 352 projection + 64 scoring FLOPs. Loss doubles projections
    # and triples scoring, rather than applying x3 to the whole indexer.
    assert indexer.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


@pytest.mark.parametrize(
    "computes_index,include_backward,expected",
    [(False, False, 1848), (True, False, 2264), (False, True, 5544), (True, True, 6440)],
)
def test_sparse_attention_counts_only_its_own_indexer(computes_index, include_backward, expected):
    config = _config()
    indexer = make_flops_test_layer(DSAIndexer, config) if computes_index else None
    attention = make_flops_test_layer(DSAttention, config, indexer=indexer)
    # Attention forward: 1,528 projections + 320 absorbed-core FLOPs. A sharing
    # layer still executes attention, while only its source layer pays the 896 indexer FLOPs.
    assert attention.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


@pytest.mark.parametrize(
    "module_type,include_backward,expected",
    [
        (MLASelfAttention, False, 2168),
        (AbsorbedMLASelfAttention, False, 2264),
        (MLASelfAttention, True, 6152),
        (AbsorbedMLASelfAttention, True, 6440),
    ],
)
def test_mla_composes_its_sparse_core_once(module_type, include_backward, expected):
    config = _config()
    indexer = make_flops_test_layer(DSAIndexer, config)
    core = make_flops_test_layer(DSAttention, config, indexer=indexer)
    attention = make_flops_test_layer(module_type, config, core_attention=core)
    # Expanded MLA core costs 224 FLOPs; absorbed core costs 320. Both include
    # projections exactly once and the same exceptional indexer training cost.
    assert attention.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


def test_mamba_flops_use_resolved_mixer_dimensions():
    config = SimpleNamespace(
        hidden_size=8, mamba_head_dim=2, mamba_num_heads=3, mamba_state_dim=4, mamba_num_groups=1
    )
    # Three heads give inner width 6. Each token costs 368 + 168 + 96 FLOPs.
    assert MambaMixer.estimate_flops(config, **BATCH_STATS) == 2528

    # A nondefault constructor expansion of 1.5 gives six heads and inner width 12.
    config.mamba_num_heads = None
    mixer = make_flops_test_layer(MambaMixer, config, d_model=8, d_inner=12, nheads=6)
    assert mixer.estimate_flops(**BATCH_STATS) == 4544


@pytest.mark.parametrize(
    "attention_backend,packing_scheduler,loss_coeff,expected_training",
    [
        ("auto", None, 0.0, 7664),
        ("auto", None, 0.1, 7728),
        ("unfused", None, 0.0, 7728),
        ("unfused", "dp_balanced", 0.0, 7664),
    ],
)
def test_csa_counts_score_backward_for_the_configured_kernel(
    attention_backend, packing_scheduler, loss_coeff, expected_training
):
    config = _config(loss_coeff)
    config.output_projection_lora_rank = 2
    config.output_projection_groups = 1
    config.csa_window_size = 2
    config.csa_dense_mode = False
    config.attention_backend = attention_backend
    config.dsa_kernel_backend = "cudnn"
    config.sequence_packing_scheduler = packing_scheduler
    config.context_parallel_size = 1
    attention = make_flops_test_layer(DSv4HybridSelfAttention, config, _dsv4_compress_ratio=4)

    # The resolved ratio is sufficient for legacy configs that lack compress_ratio.
    # Sparse ratio 4 costs 2,544 projection/window/attention FLOPs and 32 indexer score FLOPs.
    assert attention.estimate_flops(**BATCH_STATS) == 2576
    config.compress_ratio = 0
    assert attention.estimate_flops(**BATCH_STATS, include_backward=True) == expected_training
    assert DSv4HybridSelfAttention.estimate_flops(config, **BATCH_STATS) == 992
