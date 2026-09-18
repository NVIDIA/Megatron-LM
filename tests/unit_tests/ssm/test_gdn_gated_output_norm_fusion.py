# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN output fusion integration, adapted from Layali Rashid's PR #7368."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.fusions import fused_gated_norm as gated_norm
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    kwargs = dict(
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
        experimental_attention_variant="gdn",
        linear_attention_freq=[1],
        linear_cp_mode="headwise",
        transformer_impl="transformer_engine",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_post_fusion_defaults_to_disabled(monkeypatch):
    monkeypatch.setenv("MCORE_GDN_FUSION", "1")
    assert not _config().gdn_gated_output_norm_fusion


@pytest.mark.parametrize("variant", ["gdn", "gated_delta_net"])
@pytest.mark.parametrize("pre_fusion", [False, True])
def test_post_fusion_is_independent_of_pre_fusion(variant, pre_fusion):
    config = _config(
        experimental_attention_variant=variant,
        gdn_pre_gated_delta_rule_fusion=pre_fusion,
        gdn_gated_output_norm_fusion=True,
    )
    assert config.gdn_gated_output_norm_fusion
    assert config.gdn_pre_gated_delta_rule_fusion == pre_fusion


@pytest.mark.parametrize("variant", [None, "gdn2"])
def test_post_fusion_requires_gdn(variant):
    with pytest.raises(ValueError, match="gdn_gated_output_norm_fusion"):
        _config(
            experimental_attention_variant=variant,
            linear_attention_freq=None if variant is None else [1],
            gdn_gated_output_norm_fusion=True,
        )


@pytest.fixture
def model_parallel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("fla")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321)
    yield
    Utils.destroy_model_parallel()


def _model(pre_fusion, recompute=False, **overrides):
    if pre_fusion:
        pytest.importorskip("causal_conv1d.cpp_functions")
    config = _config(
        gdn_pre_gated_delta_rule_fusion=pre_fusion,
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["gdn_norm_out"] if recompute else [],
        **overrides,
    )
    spec = get_experimental_attention_variant_module_spec(config=config)
    groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    model = spec.module(config, submodules=spec.submodules, layer_number=1, pg_collection=groups)
    return model.cuda().to(config.params_dtype)


def _inputs(packed):
    hidden = torch.randn(257, 1, 256, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(hidden)
    cu = torch.tensor([0, 1, 128, 257], device="cuda", dtype=torch.int32)
    metadata = (
        PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu) if packed else None
    )
    return hidden, dy, metadata


def _run(model, hidden, dy, metadata):
    model.zero_grad(set_to_none=True)
    x = hidden.detach().clone().requires_grad_()
    out, _ = model(x, None, packed_seq_params=metadata)
    out.backward(dy)
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    grads["input"] = x.grad.detach().clone()
    return out.detach(), grads


def _assert_close(actual, expected, grads, reference_grads):
    assert actual.shape == expected.shape
    assert grads.keys() == reference_grads.keys()
    for got, ref, tolerance in [
        (actual, expected, 0.02),
        *((grads[name], reference_grads[name], 0.04) for name in grads),
    ]:
        assert torch.isfinite(got).all()
        relative_l2 = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        assert relative_l2 < tolerance


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("pre_fusion", [False, True])
def test_post_fusion_module_parity(model_parallel, packed, recompute, pre_fusion):
    model = _model(pre_fusion, recompute)
    hidden, dy, metadata = _inputs(packed)
    expected, reference_grads = _run(model, hidden, dy, metadata)
    model.config.gdn_gated_output_norm_fusion = True
    with (
        patch.object(
            gated_norm, "validate_gated_norm", wraps=gated_norm.validate_gated_norm
        ) as check,
        patch.object(gated_norm, "fused_gated_norm", wraps=gated_norm.fused_gated_norm) as norm,
    ):
        actual, grads = _run(model, hidden, dy, metadata)
        assert norm.call_count == (2 if recompute else 1)
        assert check.call_count == norm.call_count
    _assert_close(actual, expected, grads, reference_grads)


@pytest.mark.parametrize("pre_fusion", [False, True])
def test_post_fusion_revalidates_each_forward(model_parallel, pre_fusion):
    model = _model(pre_fusion)
    model.config.gdn_gated_output_norm_fusion = True
    hidden = torch.randn(17, 1, 256, device="cuda", dtype=torch.bfloat16)
    with (
        torch.no_grad(),
        patch.object(gated_norm, "fused_gated_norm", wraps=gated_norm.fused_gated_norm) as norm,
    ):
        model(hidden, None)
        assert norm.call_count == 1
        model(hidden.expand(-1, 2, -1).contiguous(), None)
        assert norm.call_count == 2
        model.out_norm = torch.nn.Identity()
        with pytest.raises(ValueError, match="RMSNorm output"):
            model(hidden, None)
        assert norm.call_count == 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch,heads,dim", [(2, 8, 64), (3, 4, 256)])
@pytest.mark.parametrize("pre_fusion", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_dense_batch_and_head_layout_parity(
    model_parallel, batch, heads, dim, pre_fusion, recompute, dtype
):
    model = _model(
        pre_fusion,
        recompute,
        linear_num_value_heads=heads,
        linear_key_head_dim=dim,
        linear_value_head_dim=dim,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        params_dtype=dtype,
    )
    hidden = torch.randn(129, batch, 256, device="cuda", dtype=dtype)
    dy = torch.randn_like(hidden)
    expected, reference_grads = _run(model, hidden, dy, None)
    model.config.gdn_gated_output_norm_fusion = True
    with patch.object(gated_norm, "fused_gated_norm", wraps=gated_norm.fused_gated_norm) as norm:
        actual, grads = _run(model, hidden, dy, None)
        assert norm.call_count == (2 if recompute else 1)
    _assert_close(actual, expected, grads, reference_grads)


def test_pre_fusion_rejects_deterministic_mode(model_parallel):
    with pytest.raises(ValueError, match="Pre-GDR fusion is non-deterministic"):
        _model(True, deterministic_mode=True)


@pytest.mark.parametrize("value_heads", [16, 64])
@pytest.mark.parametrize(
    "tp,cp,sequence_parallel", [(1, 2, False), (1, 4, False), (2, 2, True), (4, 1, True)]
)
@pytest.mark.parametrize("batch,packed", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("pre_fusion", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_post_fusion_distributed_layout(
    tp, cp, sequence_parallel, batch, packed, pre_fusion, recompute, value_heads
):
    """Enable the post fusion through CP redistribution and the complete backward."""
    if torch.distributed.get_world_size() < tp * cp:
        pytest.skip("This layout requires four distributed GPU ranks")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, context_parallel_size=cp
    )
    try:
        torch.manual_seed(321)
        model_parallel_cuda_manual_seed(321)
        model = _model(
            pre_fusion,
            recompute,
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            sequence_parallel=sequence_parallel,
            linear_num_key_heads=value_heads // 4,
            linear_num_value_heads=value_heads,
            num_attention_heads=value_heads // 4,
        )
        length = 512
        local_length = length // cp // (tp if sequence_parallel else 1)
        hidden = torch.randn(local_length, batch, 256, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(hidden)
        cu = torch.tensor([0, 256, length], device="cuda", dtype=torch.int32)
        metadata = (
            PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu) if packed else None
        )
        expected, reference_grads = _run(model, hidden, dy, metadata)
        model.config.gdn_gated_output_norm_fusion = True
        with patch.object(
            gated_norm, "fused_gated_norm", wraps=gated_norm.fused_gated_norm
        ) as norm:
            actual, grads = _run(model, hidden, dy, metadata)
            assert norm.call_count == (2 if recompute else 1)
        _assert_close(actual, expected, grads, reference_grads)
    finally:
        Utils.destroy_model_parallel()
