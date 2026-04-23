# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Integration tests for the cross-layer RMSNorm + residual-add fusion
protocol.

The layer-level protocol:

* ``native_fusion_modes()`` -- returns ``(AbsorbMode, DeferMode)`` for
  the entry and exit sites this layer would use if paired with a
  compatible neighbour.

These tests cover the env-var opt-out and the ``fp32_residual_connection``
disabling path on a real inference-optimized ``TransformerLayer``. The
main end-to-end correctness guarantee comes from
``tests/unit_tests/inference/engines/test_dynamic_engine.py::test_simple``
(the mamba variants) which drives a full hybrid forward.
"""

import pytest
import torch

from megatron.core.fusions.fused_add_rmsnorm import HAVE_TRITON
from megatron.core.fusions.rmsnorm_residual_fusion import AbsorbMode, DeferMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not HAVE_TRITON or not torch.cuda.is_available(), reason="Requires Triton and CUDA"
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

    def test_modes_are_enums(self):
        layer = _build_layer()
        absorb, defer = layer.native_fusion_modes()
        assert isinstance(absorb, AbsorbMode)
        assert isinstance(defer, DeferMode)

    def test_env_var_disables_both(self, monkeypatch):
        monkeypatch.setenv("MEGATRON_DISABLE_CROSS_LAYER_ADD_FUSION", "1")
        layer = _build_layer()
        absorb, defer = layer.native_fusion_modes()
        assert absorb is AbsorbMode.NONE
        assert defer is DeferMode.NONE

    def test_fp32_residual_disables_both(self):
        layer = _build_layer(fp32_residual_connection=True)
        absorb, defer = layer.native_fusion_modes()
        assert absorb is AbsorbMode.NONE
        assert defer is DeferMode.NONE
