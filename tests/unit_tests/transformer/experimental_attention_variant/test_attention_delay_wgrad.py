# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Deferred weight-gradient (``delay_wgrad_compute``) flush tests for the CSA and DSA
experimental attention variants.

Under ``delay_wgrad_compute=True`` every TE linear built from the shared config defers
its weight gradient until an explicit ``backward_dw()`` call.  These tests verify that

* ``DSv4HybridSelfAttention.backward_dw()`` traverses ``core_attention``
  (``CompressedSparseAttention`` -> ``Compressor`` / ``CSAIndexer``) so the six CSA
  compressor/indexer linears of a ratio-4 layer are flushed, and
* ``MLASelfAttention.backward_dw()`` traverses ``core_attention`` (``DSAttention`` ->
  ``DSAIndexer``) so the three DSA indexer linears are flushed,

i.e. after ``loss.backward()`` the deferred weights still have ``.grad is None`` and only
after ``attn.backward_dw()`` do all of them (nested ones included) receive gradients.
"""

import operator
from unittest.mock import patch

import pytest
import torch

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None

_SEED = 42


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Patch hadamard_transform in dsa/csa modules if the library is not installed."""
    if not HAVE_HADAMARD:
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                _mock_hadamard_transform,
            ),
            patch(
                'megatron.core.transformer.experimental_attention_variant.csa.rotate_activation',
                lambda x: x * (x.size(-1) ** -0.5),
            ),
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Config / build helpers
# ---------------------------------------------------------------------------


def _enable_delay_wgrad_compute(config):
    """Turn on delayed wgrad on an already-constructed config.

    Set AFTER construction on purpose: ``TransformerConfig.__post_init__`` couples
    ``delay_wgrad_compute`` to ``overlap_moe_expert_parallel_comm`` (EP pipelining),
    which is irrelevant for a single-GPU attention-module test, so we bypass that
    validation by mutating the finished dataclass.  This still happens BEFORE
    ``build_module``: TELinear reads the flag at ``__init__``
    (megatron/core/extensions/transformer_engine.py:797-801, TE >= 2.3.0 only) and
    ``TELinear.backward_dw`` gates on it (transformer_engine.py:986-989).
    """
    config.delay_wgrad_compute = True
    return config


def _make_csa_config(delay_wgrad=True):
    """MLATransformerConfig for the dsv4_hybrid (CSA) delayed-wgrad test.

    Mirrors ``_make_config`` in test_dsv4_hybrid_attention.py.
    """
    config = MLATransformerConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        q_lora_rank=64,
        kv_lora_rank=32,  # v_head_dim - qk_pos_emb_head_dim
        qk_head_dim=32,  # v_head_dim - qk_pos_emb_head_dim
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        o_groups=8,
        o_lora_rank=64,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        csa_compress_ratios=[0, 4, 128, 4],
        csa_window_size=8,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=8,
        # Indexer top-k is non-differentiable: the indexer linears ONLY receive
        # gradients through the indexer KL loss.  coeff=0 would make this test vacuous.
        dsa_indexer_loss_coeff=1.0,
        # Deferred wgrads must land in param.grad, not main_grad.
        gradient_accumulation_fusion=False,
    )
    return _enable_delay_wgrad_compute(config) if delay_wgrad else config


def _make_dsa_config(delay_wgrad=True):
    """MLATransformerConfig for the dsa delayed-wgrad test.

    Mirrors the configs in test_attention_variant_dsa.py (TestDSAttention /
    TestDSAModuleSpecDispatch) plus ``experimental_attention_variant='dsa'``.
    """
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=16,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        # MLA specific configs
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant='dsa',
        # Sparse attention specific configs
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        # Indexer top-k is non-differentiable: the indexer linears ONLY receive
        # gradients through the indexer KL loss.  coeff=0 would make this test vacuous.
        dsa_indexer_loss_coeff=1.0,
        dsa_indexer_use_sparse_loss=False,
        # Deferred wgrads must land in param.grad, not main_grad.
        gradient_accumulation_fusion=False,
    )
    return _enable_delay_wgrad_compute(config) if delay_wgrad else config


def _build_csa_attention(config, layer_number, pg_collection):
    """Instantiate a DSv4HybridSelfAttention from config (mirrors
    test_dsv4_hybrid_attention.py::_build_attention)."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsv4_hybrid_module_spec_for_backend,
    )
    from megatron.core.transformer.spec_utils import build_module

    spec = get_dsv4_hybrid_module_spec_for_backend(config=config, backend=TESpecProvider())
    return build_module(spec, config=config, layer_number=layer_number, pg_collection=pg_collection)


def _build_dsa_attention(config, layer_number, pg_collection):
    """Instantiate an MLASelfAttention with a DSAttention core from config."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsa_module_spec_for_backend,
    )
    from megatron.core.transformer.spec_utils import build_module

    spec = get_dsa_module_spec_for_backend(config=config, backend=TESpecProvider())
    return build_module(spec, config=config, layer_number=layer_number, pg_collection=pg_collection)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_flushed_grads_match_eager(
    delayed_module,
    delayed_named_linears,
    make_config,
    build_attention,
    layer_number,
    hidden,
    pg_collection,
):
    """Delay-vs-eager gradient equality: the flush must not change the math.

    Rebuild the same attention with delay_wgrad_compute off, copy the delayed
    module's exact weights (state_dict, so construction-time RNG consumption is
    irrelevant), replay a clone of the identical input, and require value-level
    agreement of every flushed weight gradient with the inline (eager) wgrads.
    This pins that backward_dw() only changes WHEN the wgrad GEMM runs, never
    the gradient definition.
    """
    delayed_grads = {
        name: module.weight.grad.detach().clone() for name, module in delayed_named_linears
    }

    torch.manual_seed(_SEED)
    model_parallel_cuda_manual_seed(_SEED)
    eager_attn = build_attention(
        make_config(delay_wgrad=False), layer_number=layer_number, pg_collection=pg_collection
    ).cuda()
    eager_attn.train()
    eager_attn.load_state_dict(delayed_module.state_dict())

    eager_hidden = hidden.detach().clone().requires_grad_(True)
    output, _ = eager_attn(hidden_states=eager_hidden, attention_mask=None)
    output.sum().backward()

    for name, _ in delayed_named_linears:
        eager_linear = operator.attrgetter(name)(eager_attn)
        assert eager_linear.weight.grad is not None, f"eager reference has no grad for {name}"
        torch.testing.assert_close(
            delayed_grads[name],
            eager_linear.weight.grad,
            msg=f"delayed-flushed wgrad differs from eager inline wgrad for {name}",
        )


def _assert_all_grads_none(named_linears, stage):
    for name, linear in named_linears:
        assert linear.weight.grad is None, (
            f"{stage}: {name}.weight.grad should still be deferred (None) under "
            f"delay_wgrad_compute, got a tensor"
        )


def _assert_all_grads_present(named_linears, stage):
    for name, linear in named_linears:
        assert (
            linear.weight.grad is not None
        ), f"{stage}: {name}.weight.grad is None — backward_dw() did not flush this linear"
        assert torch.isfinite(linear.weight.grad).all(), f"{stage}: non-finite grad on {name}"


def _assert_any_grad_nonzero(named_linears, stage):
    assert any(
        linear.weight.grad.abs().sum().item() > 0 for _, linear in named_linears
    ), f"{stage}: all flushed indexer grads are exactly zero — indexer loss did not contribute"


# ===========================================================================
# TEST 1: dsv4_hybrid / CSA — six nested compressor/indexer linears
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
@pytest.mark.skipif(
    not is_te_min_version("2.3.0"),
    reason="delay_wgrad_compute requires TE >= 2.3.0 (older TE ignores the flag)",
)
class TestDSv4HybridCSADelayedWgradFlush:
    """DSv4HybridSelfAttention.backward_dw() must flush the CSA nested linears."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_csa_deferred_wgrads_flushed_through_core_attention(self):
        seq_len = 256
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_csa_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        # layer_number=2 -> csa_compress_ratios[1] == 4: compressor AND indexer built.
        attn = _build_csa_attention(config, layer_number=2, pg_collection=pg).cuda()
        attn.train()

        core = attn.core_attention
        assert core.compressor is not None, "ratio-4 layer must build the compressor"
        assert core.indexer is not None, "ratio-4 layer must build the indexer"

        # The six CSA linears the HEAD commit flushes through core_attention.
        csa_nested_linears = [
            ("core_attention.compressor.linear_wkv", core.compressor.linear_wkv),
            ("core_attention.compressor.linear_wgate", core.compressor.linear_wgate),
            ("core_attention.indexer.linear_wq_b", core.indexer.linear_wq_b),
            ("core_attention.indexer.linear_weights_proj", core.indexer.linear_weights_proj),
            ("core_attention.indexer.compressor.linear_wkv", core.indexer.compressor.linear_wkv),
            (
                "core_attention.indexer.compressor.linear_wgate",
                core.indexer.compressor.linear_wgate,
            ),
        ]
        # The attention-level TE linears also defer (flushed by backward_dw even
        # before the core_attention traversal was added).
        attention_level_linears = [
            ("linear_q_down_proj", attn.linear_q_down_proj),
            ("linear_q_up_proj", attn.linear_q_up_proj),
            ("linear_kv_proj", attn.linear_kv_proj),
            ("linear_proj", attn.linear_proj),
        ]

        hidden = (
            torch.randn(seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        output, bias = attn(hidden_states=hidden, attention_mask=None)
        assert output.shape == (seq_len, batch_size, config.hidden_size)
        # The indexer KL loss is attached to `output` via DSAIndexerLossAutoScaler,
        # so backward through the output alone reaches the indexer subgraph.
        output.sum().backward()

        # dgrads flow normally; only the TE-linear wgrads are deferred.
        assert hidden.grad is not None, "no dgrad on hidden_states"
        # linear_o_group_proj is a raw nn.Parameter (einsum), not a TE linear: it
        # receives its gradient through plain autograd, proving backward ran.
        assert attn.linear_o_group_proj.grad is not None, "no grad on linear_o_group_proj"

        _assert_all_grads_none(csa_nested_linears + attention_level_linears, "pre-flush")

        attn.backward_dw()

        _assert_all_grads_present(csa_nested_linears + attention_level_linears, "post-flush")
        # The four indexer-side linears receive gradients exclusively through the
        # indexer loss (x/qr are detached); make sure that path is non-trivial.
        _assert_any_grad_nonzero(csa_nested_linears[2:], "post-flush")

        # Same math, different timing: flushed wgrads must equal eager wgrads.
        _assert_flushed_grads_match_eager(
            attn,
            csa_nested_linears + attention_level_linears,
            _make_csa_config,
            _build_csa_attention,
            2,
            hidden,
            pg,
        )


# ===========================================================================
# TEST 2: dsa — three DSA indexer linears under MLASelfAttention
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
@pytest.mark.skipif(
    not is_te_min_version("2.3.0"),
    reason="delay_wgrad_compute requires TE >= 2.3.0 (older TE ignores the flag)",
)
class TestDSADelayedWgradFlush:
    """MLASelfAttention.backward_dw() must flush the DSA indexer linears."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_dsa_deferred_wgrads_flushed_through_core_attention(self):
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_dsa_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_dsa_attention(config, layer_number=1, pg_collection=pg).cuda()
        attn.train()

        indexer = attn.core_attention.indexer
        # The three DSA indexer linears the HEAD commit flushes through core_attention.
        dsa_indexer_linears = [
            ("core_attention.indexer.linear_wq_b", indexer.linear_wq_b),
            ("core_attention.indexer.linear_wk", indexer.linear_wk),
            ("core_attention.indexer.linear_weights_proj", indexer.linear_weights_proj),
        ]
        # The MLA-level TE linears also defer (already flushed before the fix).
        attention_level_linears = [
            ("linear_q_down_proj", attn.linear_q_down_proj),
            ("linear_q_up_proj", attn.linear_q_up_proj),
            ("linear_kv_down_proj", attn.linear_kv_down_proj),
            ("linear_kv_up_proj", attn.linear_kv_up_proj),
            ("linear_proj", attn.linear_proj),
        ]

        hidden = (
            torch.randn(seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # attn_mask_type=causal comes from the spec params; DSAttention then builds
        # its own causal float mask, so attention_mask=None is valid here.
        output, bias = attn(hidden_states=hidden, attention_mask=None)
        assert output.shape == (seq_len, batch_size, config.hidden_size)
        # The indexer KL loss is attached to `output` via DSAIndexerLossAutoScaler,
        # so backward through the output alone reaches the (detached-x) indexer.
        output.sum().backward()

        assert hidden.grad is not None, "no dgrad on hidden_states"
        # k_norm is a TE norm (not a TE linear): its weight grad arrives through
        # plain autograd during the indexer-loss backward, proving that backward
        # reached the indexer while the linears' wgrads stayed deferred.
        assert (
            indexer.k_norm.weight.grad is not None
        ), "indexer loss backward did not reach the indexer (k_norm has no grad)"

        _assert_all_grads_none(dsa_indexer_linears + attention_level_linears, "pre-flush")

        attn.backward_dw()

        _assert_all_grads_present(dsa_indexer_linears + attention_level_linears, "post-flush")
        # The indexer linears receive gradients exclusively through the indexer
        # loss (x/qr are detached in DSAttention.forward); ensure it contributed.
        _assert_any_grad_nonzero(dsa_indexer_linears, "post-flush")

        # Same math, different timing: flushed wgrads must equal eager wgrads.
        _assert_flushed_grads_match_eager(
            attn,
            dsa_indexer_linears + attention_level_linears,
            _make_dsa_config,
            _build_dsa_attention,
            1,
            hidden,
            pg,
        )
