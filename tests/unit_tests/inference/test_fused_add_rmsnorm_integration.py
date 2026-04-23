# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for the cross-layer deferred-add + fused RMSNorm
protocol.

The layer-level protocol:

* ``native_defer_mode()`` -- :class:`DeferMode` the layer would use if
  asked to skip its exit residual-add.
* ``native_absorb_mode()`` -- :class:`AbsorbMode` the layer would use if
  asked to fold an incoming ``DeferredAdd`` into its entry RMSNorm.

These tests cover the env-var opt-out and the ``fp32_residual_connection``
disabling path on a real inference-optimized ``TransformerLayer``.  The
main end-to-end correctness guarantee comes from
``tests/unit_tests/inference/engines/test_dynamic_engine.py::test_simple``
(the mamba variants) which drives a full hybrid forward.
"""

import pytest
import torch

from megatron.core.fusions.deferred_add import AbsorbMode, DeferMode
from megatron.core.fusions.fused_add_rmsnorm import HAVE_TRITON
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(),
    reason="Requires Triton and CUDA",
)


NANO_BASE = dict(
    num_layers=1,
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=4,
    moe_ffn_hidden_size=128,
    moe_router_topk=2,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_topk_scaling_factor=1.0,
    moe_router_dtype='fp32',
    moe_token_dispatcher_type="alltoall",
    moe_grouped_gemm=True,
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    transformer_impl="inference_optimized",
    normalization="RMSNorm",
    attention_backend='flash',
    sequence_parallel=False,
)


def _build_layer(**overrides):
    from megatron.core.transformer.transformer_layer import TransformerLayer

    config = TransformerConfig(**{**NANO_BASE, **overrides})
    spec = get_gpt_decoder_layer_specs(config, use_transformer_engine=True)[0]
    layer = TransformerLayer(config=config, submodules=spec.submodules, layer_number=1)
    return layer.cuda().to(torch.bfloat16)


@pytest.mark.internal
class TestDeferredAddProtocol:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_defer_mode_is_enum(self):
        layer = _build_layer()
        assert isinstance(layer.native_defer_mode(), DeferMode)

    def test_absorb_mode_is_enum(self):
        layer = _build_layer()
        assert isinstance(layer.native_absorb_mode(), AbsorbMode)

    def test_env_var_disables_both(self, monkeypatch):
        monkeypatch.setenv("MEGATRON_DISABLE_CROSS_LAYER_ADD_FUSION", "1")
        layer = _build_layer()
        assert layer.native_defer_mode() is DeferMode.NONE
        assert layer.native_absorb_mode() is AbsorbMode.NONE

    def test_fp32_residual_disables_both(self):
        layer = _build_layer(fp32_residual_connection=True)
        assert layer.native_defer_mode() is DeferMode.NONE
        assert layer.native_absorb_mode() is AbsorbMode.NONE
