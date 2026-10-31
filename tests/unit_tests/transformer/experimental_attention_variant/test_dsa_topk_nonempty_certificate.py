# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A top-k-computing DSA layer certifies that every top-k row selects a key, which lets the
sparse-attention backend skip its empty-row masking, only when its selection guarantees it."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal

_SEQ_LEN = 4


def _make_config() -> MLATransformerConfig:
    """A small one-layer DSA config without index sharing: the layer computes its own top-k."""
    return MLATransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=8,
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_head_dim=8,
        qk_pos_emb_head_dim=4,
        v_head_dim=8,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=False,
        dsa_kernel_backend="none",
        attention_backend=AttnBackend.unfused,
        add_bias_linear=False,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_base=10000,
        gradient_accumulation_fusion=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 2, multiplier=2.0),
        use_cpu_initialization=False,
        perform_initialization=True,
    )


@pytest.fixture
def cpu_dsa_layers(monkeypatch):
    """Build DSA layers on CPU without Transformer Engine indexer modules."""
    monkeypatch.setattr(dsa_module, "build_module", lambda *_args, **_kwargs: nn.Identity())


@pytest.fixture
def certificates(monkeypatch):
    """Record the certificate and top-k tensor each sparse-attention call receives."""
    seen = []
    original = dsa_module._run_sparse_attention

    def recording_sparse_attention(**kwargs):
        seen.append(
            SimpleNamespace(
                certificate=kwargs["all_topk_rows_nonempty"], topk=kwargs["topk_indices"]
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(dsa_module, "_run_sparse_attention", recording_sparse_attention)
    return seen


def _fake_forward_before_topk(x, _qr, _packed_seq_params):
    generator = torch.Generator().manual_seed(7)
    rows, batch = x.shape[:2]
    q = torch.randn(rows, batch, 2, 8, generator=generator)
    k = torch.randn(rows, batch, 8, generator=generator)
    weights = torch.rand(rows, batch, 2, generator=generator) + 0.5
    return q, k, weights


def _make_cpu_layer(config, layer_number: int):
    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=None),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=None),
    )
    attention.indexer.forward_before_topk = _fake_forward_before_topk
    return attention


def _cpu_inputs(config):
    generator = torch.Generator().manual_seed(11)
    heads = config.num_attention_heads
    absorbed_dim = config.kv_lora_rank + config.qk_pos_emb_head_dim
    return {
        "query": torch.randn(_SEQ_LEN, 1, heads, absorbed_dim, generator=generator),
        "key": torch.randn(_SEQ_LEN, 1, 1, absorbed_dim, generator=generator),
        "x": torch.randn(_SEQ_LEN, 1, config.hidden_size, generator=generator),
        "qr": torch.randn(_SEQ_LEN, 1, config.q_lora_rank, generator=generator),
        "up_v_weight": torch.randn(
            heads, config.v_head_dim, config.kv_lora_rank, generator=generator
        ),
    }


def _cpu_forward(attention, inputs, attention_mask, attn_mask_type):
    return attention(
        query=inputs["query"],
        key=inputs["key"],
        value=None,
        attention_mask=attention_mask,
        x=inputs["x"],
        qr=inputs["qr"],
        up_v_weight=inputs["up_v_weight"],
        attn_mask_type=attn_mask_type,
    )


def _all_rows_nonempty(topk: torch.Tensor) -> bool:
    return bool((topk >= 0).any(dim=-1).all())


@pytest.mark.parametrize("always_select_tail", [True, False], ids=["tail", "no-tail"])
def test_kpool_producer_certifies_rows_only_with_tail_selection(
    cpu_dsa_layers, certificates, always_select_tail
):
    """Without its tail, KPool leaves the rows before the first complete pool empty."""
    config = _make_config()
    producer = _make_cpu_layer(config, 1)
    assert not producer.skip_topk
    pool_size = 2
    indexer = producer.indexer
    indexer.index_kpool = pool_size
    indexer.index_kpool_always_select_tail = always_select_tail
    indexer.index_kpool_use_quantization = False
    indexer.index_kpool_compress_ape = torch.zeros(pool_size, 8)

    def kpool_forward_before_topk(x, qr, packed_seq_params):
        q, k, weights = _fake_forward_before_topk(x, qr, packed_seq_params)
        indexer._kpool_gate_score = torch.zeros_like(k)
        return q, k, weights

    indexer.forward_before_topk = kpool_forward_before_topk
    # No explicit mask: a plain causal layout is the case the certificate can prove.
    _cpu_forward(producer, _cpu_inputs(config), None, AttnMaskType.causal)

    # The plain causal layout alone would prove every row non-empty.
    assert [call.certificate for call in certificates] == [always_select_tail]
    assert _all_rows_nonempty(certificates[0].topk) is always_select_tail
