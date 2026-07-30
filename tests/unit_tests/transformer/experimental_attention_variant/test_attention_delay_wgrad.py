# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Deferred weight-gradient (``delay_wgrad_compute``) tests for the absorbed MLA
attention variant.

Under ``delay_wgrad_compute=True`` TE linears that execute their forward defer their
weight gradient until an explicit ``backward_dw()`` call. The absorbed K/V up projection
is an exception: absorption consumes ``linear_kv_up_proj.weight`` directly instead of
running the module's forward, so no delayed-wgrad closure is ever enqueued for it and it
must stay on the plain autograd path. These tests verify that, after
``loss.backward()``,

* the MLA linears that do execute a GEMM still have ``.grad is None`` and only receive
  gradients once ``attn.backward_dw()`` runs, while
* ``linear_kv_up_proj.weight.grad`` is already present and is left untouched by
  ``backward_dw()``,

and that the flushed gradients match an eager (``delay_wgrad_compute=False``) reference.
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
    """Patch hadamard_transform in the dsa module if the library is not installed."""
    if not HAVE_HADAMARD:
        with patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
            _mock_hadamard_transform,
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
    ``build_module``: TELinear reads the flag at ``__init__`` (TE >= 2.3.0 only) and
    ``TELinear.backward_dw`` gates on it.
    """
    config.delay_wgrad_compute = True
    return config


def _make_dsa_config(delay_wgrad=True):
    """MLATransformerConfig for the absorbed-MLA delayed-wgrad test.

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


def _build_dsa_attention(config, layer_number, pg_collection):
    """Instantiate an AbsorbedMLASelfAttention with a DSAttention core from config."""
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
    agreement of every weight gradient with the inline (eager) wgrads.  This pins
    that backward_dw() only changes WHEN the wgrad GEMM runs, never the gradient
    definition — and that routing linear_kv_up_proj through plain autograd yields
    the same gradient as the eager path.
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
            msg=f"delayed wgrad differs from eager inline wgrad for {name}",
        )


def _assert_all_grads_none(named_linears, stage):
    for name, linear in named_linears:
        assert linear.weight.grad is None, (
            f"{stage}: {name}.weight.grad should still be deferred (None) under "
            f"delay_wgrad_compute, got a tensor"
        )


def _assert_all_grads_present(named_linears, stage):
    for name, linear in named_linears:
        assert linear.weight.grad is not None, f"{stage}: {name}.weight.grad is None"
        assert torch.isfinite(linear.weight.grad).all(), f"{stage}: non-finite grad on {name}"


# ===========================================================================
# The absorbed K/V up projection must bypass the delayed-wgrad queue
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
@pytest.mark.skipif(
    not is_te_min_version("2.3.0"),
    reason="delay_wgrad_compute requires TE >= 2.3.0 (older TE ignores the flag)",
)
class TestAbsorbedMLADelayedWgrad:
    """AbsorbedMLASelfAttention must exempt linear_kv_up_proj from delayed wgrad."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_absorbed_kv_up_proj_keeps_plain_autograd_wgrad(self):
        seq_len = 64
        batch_size = 2

        torch.manual_seed(_SEED)
        model_parallel_cuda_manual_seed(_SEED)

        config = _make_dsa_config()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        attn = _build_dsa_attention(config, layer_number=1, pg_collection=pg).cuda()
        attn.train()

        # MLA linears that execute their forward defer and are flushed by backward_dw().
        attention_level_linears = [
            ("linear_q_down_proj", attn.linear_q_down_proj),
            ("linear_q_up_proj", attn.linear_q_up_proj),
            ("linear_kv_down_proj", attn.linear_kv_down_proj),
            ("linear_proj", attn.linear_proj),
        ]
        # Absorption reads the K/V up-projection weight directly instead of executing
        # the module's forward, so it uses plain autograd rather than TE's delayed-wgrad
        # queue.  The exemption must come from a private config copy — the shared config
        # keeps delayed wgrad on for every other linear.
        absorbed_kv_up_linears = [("linear_kv_up_proj", attn.linear_kv_up_proj)]
        assert config.delay_wgrad_compute is True
        assert attn.linear_kv_up_proj.config is not config
        assert attn.linear_kv_up_proj.config.delay_wgrad_compute is False

        hidden = (
            torch.randn(seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # attn_mask_type=causal comes from the spec params; DSAttention then builds
        # its own causal float mask, so attention_mask=None is valid here.
        output, bias = attn(hidden_states=hidden, attention_mask=None)
        assert output.shape == (seq_len, batch_size, config.hidden_size)
        output.sum().backward()

        assert hidden.grad is not None, "no dgrad on hidden_states"

        _assert_all_grads_none(attention_level_linears, "pre-flush")
        _assert_all_grads_present(absorbed_kv_up_linears, "pre-flush")
        absorbed_kv_up_grads = {
            name: linear.weight.grad.detach().clone() for name, linear in absorbed_kv_up_linears
        }

        # Without the exemption this raises: backward_dw() would pop a delayed-wgrad
        # closure that the never-executed kv-up-proj forward never enqueued.
        attn.backward_dw()

        _assert_all_grads_present(attention_level_linears, "post-flush")
        for name, linear in absorbed_kv_up_linears:
            torch.testing.assert_close(
                linear.weight.grad,
                absorbed_kv_up_grads[name],
                msg=f"backward_dw() changed plain-autograd wgrad for {name}",
            )

        # Same math, different timing: flushed wgrads must equal eager wgrads.
        _assert_flushed_grads_match_eager(
            attn,
            attention_level_linears + absorbed_kv_up_linears,
            _make_dsa_config,
            _build_dsa_attention,
            1,
            hidden,
            pg,
        )
