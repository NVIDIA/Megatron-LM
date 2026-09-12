# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the Transformer Engine wrappers in ``megatron/core/extensions/``.

These are the kernels that carry almost every FLOP of a training step: cuBLASLt GEMMs (with
heuristic algorithm selection under the pinned workspace), fused LayerNorm/RMSNorm forward
and backward (cross-row dgamma reduction), grouped GEMMs over uneven expert loads, fused
attention with GQA (dK/dV accumulate across query groups), and fused RoPE. Each wrapper is
replayed standalone under side-stream contention; the bare TE modules are what the model
level suite composes, so a regression here localises to one kernel family.
"""

import os

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    seeded,
)
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TE), reason="needs a GPU and Transformer Engine"
)

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TEGroupedLinear,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

HIDDEN, FFN, TOKENS = 2048, 8192, 8192


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        num_attention_heads=16,
        num_query_groups=4,
        kv_channels=128,
        use_cpu_initialization=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        deterministic_mode=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestTEWrappers:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_te_column_parallel_linear_replays(self):
        seeded()
        module = TEColumnParallelLinear(
            HIDDEN,
            FFN,
            config=_config(),
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()
        x = torch.randn(
            TOKENS // 2, 2, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        assert_module_replays_bit_exact(
            module, (x,), replays=3, contention=True, what="TEColumnParallelLinear"
        )

    def test_te_row_parallel_linear_replays(self):
        seeded()
        module = TERowParallelLinear(
            FFN,
            HIDDEN,
            config=_config(),
            init_method=init_method_normal(0.02),
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()
        x = torch.randn(
            TOKENS // 2, 2, FFN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        assert_module_replays_bit_exact(
            module, (x,), replays=3, contention=True, what="TERowParallelLinear"
        )

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    def test_te_layernorm_column_parallel_linear_replays(self, normalization, zero_centered_gamma):
        seeded()
        config = _config(
            normalization=normalization, layernorm_zero_centered_gamma=zero_centered_gamma
        )
        module = TELayerNormColumnParallelLinear(
            HIDDEN,
            3 * HIDDEN,
            config=config,
            init_method=init_method_normal(0.02),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()
        # 16k rows stress the cross-row dgamma/dbeta reduction of the norm backward.
        x = torch.randn(TOKENS, 2, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        assert_module_replays_bit_exact(
            module,
            (x,),
            replays=3,
            contention=True,
            what=f"TELayerNormColumnParallelLinear[{normalization}]",
        )

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_te_norm_replays(self, normalization):
        seeded()
        module = TENorm(_config(normalization=normalization), HIDDEN, eps=1e-5).cuda()
        x = torch.randn(TOKENS, 2, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        assert_module_replays_bit_exact(
            module, (x,), replays=3, contention=True, what=f"TENorm[{normalization}]"
        )

    def test_te_grouped_linear_replays_on_uneven_splits(self):
        seeded()
        module = TEGroupedLinear(
            8,
            HIDDEN,
            FFN,
            parallel_mode=None,
            config=_config(),
            init_method=init_method_normal(0.02),
            bias=False,
            skip_bias_add=False,
            is_expert=True,
        ).cuda()
        m_splits = [4096, 13, 0, 2048, 1, 8191, 33, 1999]
        x = torch.randn(
            sum(m_splits), HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        assert_module_replays_bit_exact(
            module, (x, m_splits), replays=3, contention=True, what="TEGroupedLinear"
        )

    @pytest.mark.parametrize("backend", ["fused", "flash"])
    def test_te_dot_product_attention_replays(self, backend, monkeypatch):
        """GQA causal attention; TE must pick a deterministic backward (NVTE_ALLOW_NONDETERMINISTIC_ALGO=0)."""
        seeded()
        # conftest forces both TE attention backends off per test; pick one explicitly.
        clear_nvte_env_vars()
        monkeypatch.setenv("NVTE_FUSED_ATTN", "1" if backend == "fused" else "0")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1" if backend == "flash" else "0")
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        # TE caches its backend choice per AttentionParams and only re-selects when the
        # params change, so an env flip alone would replay the previously chosen backend.
        from transformer_engine.pytorch.attention.dot_product_attention import (
            dot_product_attention as te_dpa,
        )

        monkeypatch.setattr(
            te_dpa,
            "_attention_backends",
            {
                "attention_params": None,
                "use_flash_attention": None,
                "flash_attention_backend": None,
                "use_fused_attention": None,
                "fused_attention_backend": None,
                "use_unfused_attention": None,
                "backend_selection_requires_update": False,
            },
        )
        config = _config()
        module = TEDotProductAttention(
            config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
        ).cuda()
        s, b, h, hkv, d = 4096, 2, 16, 4, 128
        q = torch.randn(s, b, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(s, b, hkv, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(s, b, hkv, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        def fn(q, k, v):
            return module(q, k, v, None, AttnMaskType.causal)

        try:
            assert_replays_bit_exact(
                fn, (q, k, v), replays=4, contention=True, what=f"TEDotProductAttention[{backend}]"
            )
        except (RuntimeError, AssertionError) as error:
            if "backend" in str(error).lower() and "avail" in str(error).lower():
                pytest.skip(f"TE has no {backend} attention backend here: {error}")
            raise


# --- fused RoPE --------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
def test_te_fused_rope_replays_fwd_bwd(layout):
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    if (
        rope_utils.fused_apply_rotary_pos_emb is None
        and rope_utils.fused_apply_rotary_pos_emb_thd is None
    ):
        pytest.skip("TE fused RoPE unavailable")
    Utils.initialize_model_parallel()
    try:
        seeded()
        config = _config(apply_rope_fusion=True)
        s, b, h, d = 4096, 2, 32, 128
        freqs = RotaryEmbedding(kv_channels=d, rotary_percent=1.0)(s)
        if layout == "sbhd":
            t = torch.randn(s, b, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            cu_seqlens = None
        else:
            t = torch.randn(s, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            cu_seqlens = torch.tensor([0, 1000, 1500, 2048, 4096], dtype=torch.int32, device="cuda")

        def fn(t):
            return rope_utils.apply_rotary_pos_emb(t, freqs, config, cu_seqlens=cu_seqlens)

        assert_replays_bit_exact(fn, (t,), replays=3, what=f"TE fused RoPE[{layout}]")
    finally:
        Utils.destroy_model_parallel()
