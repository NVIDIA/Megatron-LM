# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward and backward FLOPs use the sparse core and indexer actually constructed."""

from types import SimpleNamespace

import pytest

from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, DSAttention
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_latent_attention import MLASelfAttention

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


def _bare_module(module_type, config, **children):
    # Accounting needs config and composition only, without CUDA parameters/process groups.
    module = module_type.__new__(module_type)
    MegatronModule.__init__(module, config)
    for name, child in children.items():
        setattr(module, name, child)
    return module


@pytest.mark.parametrize(
    "loss_coeff,include_backward,expected",
    [(0.0, False, 416), (0.01, False, 416), (0.0, True, 416), (0.01, True, 896)],
)
def test_indexer_obeys_its_backward_rule(loss_coeff, include_backward, expected):
    indexer = _bare_module(DSAIndexer, _config(loss_coeff))
    # Forward costs 352 projection + 64 scoring FLOPs. Loss doubles projections
    # and triples scoring, rather than applying x3 to the whole indexer.
    assert indexer.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected


@pytest.mark.parametrize(
    "computes_index,include_backward,expected",
    [(False, False, 1848), (True, False, 2264), (False, True, 5544), (True, True, 6440)],
)
def test_sparse_attention_counts_only_its_own_indexer(computes_index, include_backward, expected):
    config = _config()
    indexer = _bare_module(DSAIndexer, config) if computes_index else None
    attention = _bare_module(DSAttention, config, indexer=indexer)
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
    indexer = _bare_module(DSAIndexer, config)
    core = _bare_module(DSAttention, config, indexer=indexer)
    attention = _bare_module(module_type, config, core_attention=core)
    # Expanded MLA core costs 224 FLOPs; absorbed core costs 320. Both include
    # projections exactly once and the same exceptional indexer training cost.
    assert attention.estimate_flops(**BATCH_STATS, include_backward=include_backward) == expected
