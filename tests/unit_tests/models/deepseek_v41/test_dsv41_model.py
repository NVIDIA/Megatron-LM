# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GPU tests: tiny DeepSeek-V4.1 HybridModel forward / backward on one device."""

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.deepseek_v41.hybrid_stack import DSv41HybridStack
from megatron.core.models.deepseek_v41.hyper_connection import SinglePassHyperConnectionHybridLayer
from megatron.core.models.deepseek_v41.layer_specs import (
    build_dsv41_hybrid_layer_pattern,
    dsv41_config_kwargs_from_model_layers,
    hybrid_dsv41_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.csa2.attention import CSA2Attention
from megatron.core.transformer.experimental_attention_variant.csa2.roles import CSA2LayerMode
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

TINY_RATIOS = [0, 0, 2, 2, 1, 1]
TINY_KV = [2, 4]
TINY_INDEX = [2, 3, 4, 5]  # layers 3 and 5 are REINDEX
VOCAB = 128
SEQ = 32


def make_tiny_config(engram: bool = False, **overrides):
    kwargs = dict(
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        q_lora_rank=32,
        qk_pos_emb_head_dim=16,
        v_head_dim=64,
        o_groups=2,
        o_lora_rank=32,
        rope_type="rope",
        rotary_base=10000,
        rotary_scaling_factor=16,
        original_max_position_embeddings=64,
        mscale=1.0,
        mscale_all_dim=1.0,
        multi_latent_attention=True,
        qk_layernorm=True,
        experimental_attention_variant="dsv4_hybrid",
        dsv4_version="v4.1",
        enable_hyper_connections=True,
        num_residual_streams=4,
        csa_window_size=8,
        csa_compress_rotary_base=160000.0,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=32,
        dsa_indexer_topk=4,
        dsa_indexer_loss_coeff=0.0,
        dsa_kernel_backend="none",
        csa2_kv_source_layers=TINY_KV,
        csa2_index_source_layers=TINY_INDEX,
        csa2_candidate_source_layer=4,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        moe_shared_expert_intermediate_size=64,
        moe_router_topk=2,
        moe_router_score_function="sqrtsoftplus",
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=1.5,
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="alltoall",
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=10.0,
        normalization="RMSNorm",
        gated_linear_unit=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.02,
    )
    if engram:
        kwargs.update(
            engram_layer_ids=[1, 2],
            engram_num_embeddings=[2048, 2048],
            engram_max_ngram_size=3,
            engram_bucket_size=300,
            engram_n_heads=2,
            engram_head_dim=16,
            engram_compressed_vocab_size=VOCAB,
        )
    kwargs.update(dsv41_config_kwargs_from_model_layers(TINY_RATIOS))
    kwargs.update(overrides)
    return MLATransformerConfig(**kwargs)


def build_model(config):
    return HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_dsv41_stack_spec(config),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        hybrid_layer_pattern=build_dsv41_hybrid_layer_pattern(TINY_RATIOS),
        position_embedding_type="none",
    ).cuda()


def batch(b=2, s=SEQ):
    input_ids = torch.randint(0, VOCAB, (b, s), device="cuda")
    position_ids = torch.arange(s, device="cuda").unsqueeze(0).expand(b, s)
    labels = torch.randint(0, VOCAB, (b, s), device="cuda")
    return input_ids, position_ids, labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestTinyDSv41Model:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_structure(self):
        model = build_model(make_tiny_config())
        assert isinstance(model.decoder, DSv41HybridStack)
        assert len(model.decoder.layers) == 12
        assert all(
            isinstance(layer, SinglePassHyperConnectionHybridLayer)
            for layer in model.decoder.layers
        )
        assert not hasattr(model.decoder, "hc_head_fn")
        cores = [m for m in model.modules() if isinstance(m, CSA2Attention)]
        assert [c.plan.mode for c in cores] == [
            CSA2LayerMode.WINDOW,
            CSA2LayerMode.WINDOW,
            CSA2LayerMode.FULL,
            CSA2LayerMode.REINDEX,
            CSA2LayerMode.FULL,
            CSA2LayerMode.REINDEX,
        ]
        assert cores[2].compressor is not None and cores[2].indexer is not None
        assert cores[3].compressor is None and cores[3].indexer is not None
        assert cores[5].plan.uses_candidates and cores[4].plan.is_candidate_source
        # indexer parameters are frozen in this phase (no gradient path without the loss)
        for core in cores[2:]:
            assert all(not p.requires_grad for p in core.indexer.parameters())
        # ratio-1 layers rotate with the compressed base + YaRN
        attn_modules = [c for c in model.modules() if hasattr(c, "_dsv4_uses_yarn_rope")]
        assert [a._dsv4_uses_yarn_rope for a in attn_modules] == [
            False,
            False,
            True,
            True,
            True,
            True,
        ]

    def test_forward_backward_finite(self):
        model = build_model(make_tiny_config())
        input_ids, position_ids, labels = batch()
        loss = model(input_ids, position_ids, attention_mask=None, labels=labels)
        assert torch.isfinite(loss).all()
        loss.mean().backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
        # the shared compressed KV must receive gradient from the reuse layers: the
        # compressor weights of both KV sources have non-zero gradients
        cores = [m for m in model.modules() if isinstance(m, CSA2Attention)]
        for core in (cores[2], cores[4]):
            assert core.compressor.linear_wkv.weight.grad.abs().sum() > 0

    def test_consumers_contribute_to_source_gradient(self):
        """Detaching the shared KV inside the consumers must change the source gradient."""
        from megatron.core.transformer.experimental_attention_variant.csa2 import state as st

        model = build_model(make_tiny_config())
        torch.manual_seed(7)
        input_ids, position_ids, labels = batch()
        cores = [m for m in model.modules() if isinstance(m, CSA2Attention)]

        def source_grad():
            model.zero_grad(set_to_none=True)
            loss = model(input_ids, position_ids, attention_mask=None, labels=labels)
            loss.mean().backward()
            return cores[2].compressor.linear_wkv.weight.grad.detach().clone()

        reference = source_grad()
        original = st.DSv41SharedState.get_compressed

        def detached_get(self, source_layer):
            rec = original(self, source_layer)
            return st.CompressedKVRecord(
                rec.source_layer,
                rec.compress_ratio,
                rec.n_compressed,
                rec.kv.detach(),
                rec.index_keys.detach(),
            )

        st.DSv41SharedState.get_compressed = detached_get
        try:
            detached = source_grad()
        finally:
            st.DSv41SharedState.get_compressed = original
        assert not torch.allclose(reference, detached), "consumers add nothing to the source grad"

    def test_engram_frozen_forward(self):
        model = build_model(make_tiny_config(engram=True))
        engrams = [layer.engram for layer in model.decoder.layers if layer.engram is not None]
        assert len(engrams) == 2
        assert model.decoder.engram_hasher is not None
        for engram in engrams:
            assert all(not p.requires_grad for p in engram.parameters())
        input_ids, position_ids, labels = batch()
        loss = model(input_ids, position_ids, attention_mask=None, labels=labels)
        assert torch.isfinite(loss).all()
        loss.mean().backward()
        for engram in engrams:
            assert all(p.grad is None for p in engram.parameters())

    def test_two_steps_no_nan(self):
        model = build_model(make_tiny_config())
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        for _ in range(3):
            input_ids, position_ids, labels = batch()
            loss = model(input_ids, position_ids, attention_mask=None, labels=labels).mean()
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestFullRecompute:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "method,num_layers", [("uniform", 1), ("uniform", 3), ("uniform", 5), ("block", 4)]
    )
    def test_matches_eager(self, method, num_layers):
        """Loss and every gradient must match the eager forward; the shared state crosses
        checkpoint boundaries as explicit tensors."""
        torch.manual_seed(3)
        model_parallel_cuda_manual_seed(3)
        eager = build_model(make_tiny_config(engram=True))
        torch.manual_seed(3)
        model_parallel_cuda_manual_seed(3)
        ckpt = build_model(
            make_tiny_config(
                engram=True,
                recompute_granularity="full",
                recompute_method=method,
                recompute_num_layers=num_layers,
            )
        )
        ckpt.load_state_dict(eager.state_dict())
        gen = torch.Generator(device="cuda").manual_seed(9)
        input_ids = torch.randint(0, VOCAB, (2, SEQ), device="cuda", generator=gen)
        position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(2, SEQ)
        labels = torch.randint(0, VOCAB, (2, SEQ), device="cuda", generator=gen)

        losses = []
        for m in (eager, ckpt):
            m.train()
            loss = m(input_ids, position_ids, attention_mask=None, labels=labels).mean()
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-3, atol=1e-3)
        for (name, p_e), (_, p_c) in zip(eager.named_parameters(), ckpt.named_parameters()):
            if p_e.grad is None:
                assert p_c.grad is None, name
                continue
            torch.testing.assert_close(
                p_e.grad.float(), p_c.grad.float(), rtol=2e-2, atol=2e-2, msg=name
            )
