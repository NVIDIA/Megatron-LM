# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDN module integration with optional output-norm recomputation."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import gdn_fusion
from megatron.core.ssm.gated_delta_net import gdn as gdn_module
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_gdn_fused_module(monkeypatch, model_parallel, packed, recompute):
    pytest.importorskip("transformer_engine.pytorch")
    if gdn_fusion._LINEAR_BWD is None:
        pytest.skip("FLA convolution backward is unavailable")
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321)
    config = TransformerConfig(
        hidden_size=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=16,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        num_attention_heads=4,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["gdn_norm_out"] if recompute else [],
    )
    spec = get_experimental_attention_variant_module_spec(config=config)
    groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    model = spec.module(config, submodules=spec.submodules, layer_number=1, pg_collection=groups)
    model = model.cuda().bfloat16()
    hidden = torch.randn(257, 1, 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(hidden)
    cu = torch.tensor([0, 1, 128, 257], device="cuda", dtype=torch.int32)
    metadata = (
        PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu) if packed else None
    )

    def run(fused):
        monkeypatch.setenv("MCORE_GDN_FUSION", str(int(fused)))
        model.zero_grad(set_to_none=True)
        x = hidden.detach().clone().requires_grad_()
        out, _ = model(x, None, packed_seq_params=metadata)
        out.backward(dy)
        grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        grads["input"] = x.grad.detach().clone()
        return out.detach(), grads

    expected, reference_grads = run(False)
    with (
        patch.object(gdn_module, "fused_prepare", wraps=gdn_module.fused_prepare) as prepare,
        patch.object(gdn_module, "fused_gated_norm", wraps=gdn_module.fused_gated_norm) as norm,
    ):
        actual, grads = run(True)
        assert prepare.call_count == 1
        assert norm.call_count == (2 if recompute else 1)
    assert actual.shape == expected.shape
    assert grads.keys() == reference_grads.keys()
    for got, ref, tolerance in [
        (actual, expected, 0.02),
        *((grads[name], reference_grads[name], 0.04) for name in grads),
    ]:
        assert torch.isfinite(got).all()
        relative_l2 = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        assert relative_l2 < tolerance
