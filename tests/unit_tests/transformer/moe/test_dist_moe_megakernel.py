# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Ownership and weight-layout contracts independent of full model training."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.moe.megakernel.dist_moe import runtime
from megatron.core.transformer.moe.megakernel.dist_moe.backend import DistMoEMegakernel
from megatron.core.transformer.moe.megakernel.dist_moe.weights import bf16_view, prepare_weight
from megatron.core.transformer.moe.megakernel.mok.weights import _swizzle_mxfp8_scale
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**kwargs):
    options = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        moe_grouped_gemm=True,
        moe_use_grouped_tensor=True,
        moe_single_grouped_weight=True,
        gradient_accumulation_fusion=True,
        moe_megakernel_backend="dist_moe",
        expert_model_parallel_size=4,
    )
    options.update(kwargs)
    return TransformerConfig(**options)


def _module():
    experts = torch.nn.Module()
    for name, shape in (("linear_fc1", (2, 512, 256)), ("linear_fc2", (2, 256, 256))):
        linear = torch.nn.Module()
        linear.single_grouped_weight = True
        linear.register_parameter(
            "weight", torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
        )
        experts.add_module(name, linear)
    module = DistMoEMegakernel(_config(), object(), experts, None, 2)
    return experts, module


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moe_shared_expert_intermediate_size": None},
        {"moe_shared_expert_intermediate_size": 512},
        {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param": True},
        {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param": False},
    ],
)
def test_supported_precision_and_shared_configs(kwargs):
    _config(**kwargs)


@pytest.mark.parametrize("fp8_primary", [False, True])
def test_hybrid_mxfp8_is_rejected_before_collectives(fp8_primary):
    # Hybrid requests E5M2 gradient operands; Dist kernels consume E4M3.
    with pytest.raises(ValueError, match="E4M3"):
        _config(fp8="hybrid", fp8_recipe="mxfp8", fp8_param=fp8_primary)


def test_layer_precision_override_is_rejected_before_collectives():
    with pytest.raises(ValueError, match="first_last_layers_bf16"):
        _config(fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, first_last_layers_bf16=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moe_single_grouped_weight": False},
        {"moe_mlp_glu_interleave_size": 32},
        {"activation_func_clamp_value": 4.0},
        {"activation_func_clamp_value": 0.0},
        {"cuda_graph_impl": "full_iteration"},
        {"moe_token_dispatcher_type": "flex"},
        {"add_bias_linear": True},
        {"moe_megakernel_backend_config": {"unknown_option": 1}},
    ],
)
def test_unsupported_configuration_fails_before_collectives(kwargs):
    with pytest.raises((ValueError, TypeError)):
        _config(**kwargs)


def test_constructor_does_not_allocate_runtime_or_require_main_grad():
    experts, module = _module()
    assert module.context is None
    assert module.owners == (experts.linear_fc1.weight, experts.linear_fc2.weight)
    assert module.sharded_state_dict() == {}
    assert module.state_dict() == {}
    with pytest.raises(RuntimeError, match="initialize_dist_moe_runtime"):
        module(torch.empty(2, 256), torch.empty(2, 8), torch.empty(2, 8))


def test_runtime_rejects_expert_parameters_tied_across_layers():
    experts, first = _module()
    second = DistMoEMegakernel(_config(), object(), experts, None, 2)
    with pytest.raises(ValueError, match="tied expert parameters across layers"):
        runtime.initialize_dist_moe_runtime(
            [torch.nn.ModuleList([first, second])],
            ddp_config=SimpleNamespace(),
            max_num_local_input_tokens=256,
        )


def _checkpoint_parent():
    experts, backend = _module()
    parent = torch.nn.Module()
    parent.add_module("experts", experts)
    parent.add_module("megakernel_experts", backend)
    checkpoint = {name: torch.full_like(value, 0.25) for name, value in parent.state_dict().items()}
    return parent, backend, checkpoint


def test_idle_checkpoint_load_preserves_owners_grad_buffers_and_context():
    parent, backend, checkpoint = _checkpoint_parent()
    owners = backend.owners
    for owner in owners:
        owner.main_grad = torch.zeros_like(owner, dtype=torch.float32)
    grads = tuple(owner.main_grad for owner in owners)
    context = backend.context = object()
    backend._weight_cache = backend._prepared = object()
    backend.is_first_microbatch = False
    parent.load_state_dict(checkpoint, strict=True)
    assert backend.context is context and backend._prepared is None
    assert backend._weight_cache is None and backend.is_first_microbatch
    for before, owner, grad in zip(owners, backend.owners, grads):
        assert before is owner and grad is owner.main_grad
        torch.testing.assert_close(owner, torch.full_like(owner, 0.25))


@pytest.mark.parametrize("live,assign", [(True, False), (False, True)])
def test_checkpoint_load_rejects_live_or_replaced_owners_before_expert_mutation(live, assign):
    parent, backend, checkpoint = _checkpoint_parent()
    owners = backend.owners
    backend._live = live
    with pytest.raises(RuntimeError, match="unfinished backward|assign=False"):
        parent.load_state_dict(checkpoint, strict=True, assign=assign)
    for before, owner in zip(owners, backend.owners):
        assert before is owner
        torch.testing.assert_close(owner, torch.zeros_like(owner))


def test_completion_reuses_buffers_and_rejects_live_storage_changes():
    _, module = _module()
    for owner in module.owners:
        owner.main_grad = torch.full_like(owner, 0.25, dtype=torch.float32)
    module._gradient_binding = tuple(runtime.storage_signature(p.main_grad) for p in module.owners)
    module._weight_binding = module._weight_signature()
    module._live = True
    sentinels = module._complete(module.owners)
    assert not module._live
    for owner, sentinel in zip(module.owners, sentinels):
        assert owner.grad_added_to_main_grad and sentinel.data_ptr() == owner.data_ptr()
        torch.testing.assert_close(owner.main_grad, torch.full_like(owner.main_grad, 0.25))
    module.owners[0].main_grad = torch.zeros_like(module.owners[0].main_grad)
    with pytest.raises(RuntimeError, match="main_grad storage changed"):
        module._complete(module.owners)


def test_shared_expert_runs_once_and_adds_to_routed_output():
    from megatron.core.transformer.moe import moe_layer
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    layer = moe_layer.MoELayer.__new__(moe_layer.MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        moe_megakernel_backend="dist_moe",
        moe_megakernel_backend_config=None,
        sequence_parallel=False,
    )
    layer.attn_tp_group = SimpleNamespace(size=lambda: 1)
    layer.moe_layer_recompute = False
    layer.route = Mock(return_value=(torch.ones(4, 8), torch.ones(4, 8, dtype=torch.bool)))
    hidden = torch.ones(4, 1, 256, device="cuda", requires_grad=True)
    layer.megakernel_experts = torch.nn.Identity()
    layer.megakernel_experts.handles_shared_expert = False
    layer.megakernel_experts.forward = Mock(side_effect=lambda x, *_: x * 2)
    layer.shared_experts_compute = Mock(side_effect=lambda x: x * 3)
    output, bias = layer(hidden)
    torch.testing.assert_close(output, hidden * 5)
    assert bias is None
    layer.shared_experts_compute.assert_called_once_with(hidden)
    _, grads = assert_replays_bit_exact(
        layer,
        (hidden,),
        grad_outputs={"out[0]": torch.ones_like(hidden)},
        replays=3,
        contention=True,
        what="megakernel routed plus native shared expert",
    )
    torch.testing.assert_close(grads["in[0]"], torch.full_like(hidden, 5))
    assert layer.shared_experts_compute.call_count == 4


def test_bf16_grouped_view_aliases_native_storage():
    storage = torch.randn((2, 512, 256), device="cuda", dtype=torch.bfloat16)
    weight = SimpleNamespace(rowwise_data=storage.flatten(), shape=storage.shape)
    actual = bf16_view(weight)
    assert actual.data_ptr() == storage.data_ptr()
    torch.testing.assert_close(actual, storage, rtol=0, atol=0)


@pytest.mark.parametrize("rows,columns", [(128, 256), (512, 1024)])
def test_scale_bytes_match_dist_blocked_layout(rows, columns):
    pytest.importorskip("dist_moe")
    from dist_moe.kernels.scale_factor_layout import natural_to_cublas_blocked

    logical = torch.randint(0, 255, (2, rows, columns // 32), device="cuda", dtype=torch.uint8)
    expected = natural_to_cublas_blocked(logical.view(torch.float8_e8m0fnu), is_a=False)
    actual = _swizzle_mxfp8_scale(logical, rows=rows, columns=columns)
    torch.testing.assert_close(
        actual.flatten(), expected.view(torch.uint8).flatten(), rtol=0, atol=0
    )


@pytest.mark.parametrize("fp8_primary", [False, True])
def test_independent_te_operands_alias_and_refresh_in_place(fp8_primary):
    pytest.importorskip("dist_moe")
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    from megatron.core.fp8_utils import get_grouped_quantized_members

    torch.manual_seed(711)
    source = torch.randn((2, 256, 256), device="cuda", dtype=torch.bfloat16)
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = False
    quantizer.internal = False
    if fp8_primary:
        workspace = tex.group_quantize(source.flatten(0, 1), quantizer, 2, None)
        parameter = torch.nn.Parameter(
            GroupedTensor(
                shape=(512, 256),
                dtype=source.dtype,
                num_tensors=2,
                shapes=[(256, 256)] * 2,
                quantizer=quantizer,
                data=workspace.rowwise_data,
                columnwise_data=workspace.columnwise_data,
                scale_inv=workspace.scale_inv,
                columnwise_scale_inv=workspace.columnwise_scale_inv,
            )
        )
    else:
        parameter = torch.nn.Parameter(source.clone())
    prepared, cache = prepare_weight(parameter, use_mxfp8=True)
    operands = parameter if fp8_primary else cache["workspace"]
    assert prepared.fprop_data.data_ptr() == operands.rowwise_data.data_ptr()
    assert prepared.dgrad_data.data_ptr() == operands.columnwise_data.data_ptr()
    assert prepared.fprop_data.data_ptr() != prepared.dgrad_data.data_ptr()
    assert torch.any(prepared.fprop_data.view(torch.uint8) != prepared.dgrad_data.view(torch.uint8))
    pointers = tuple(tensor.data_ptr() for tensor in prepared.storage_tensors())
    old_scale = prepared.fprop_scale.view(torch.uint8).clone()
    with torch.no_grad():
        if fp8_primary:
            for member, value in zip(get_grouped_quantized_members(parameter), source * 4):
                member.quantize_(value)
        else:
            parameter.mul_(4)
    refreshed, same_cache = prepare_weight(parameter, use_mxfp8=True, cached=cache)
    assert same_cache is cache
    assert tuple(tensor.data_ptr() for tensor in refreshed.storage_tensors()) == pointers
    assert torch.any(old_scale != refreshed.fprop_scale.view(torch.uint8))
