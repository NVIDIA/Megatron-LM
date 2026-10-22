# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Ownership and weight-layout contracts independent of full model training."""

from contextlib import nullcontext
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
        {"cuda_graph_impl": "local", "cuda_graph_modules": []},
        {
            "fp8": "e4m3",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "cuda_graph_impl": "local",
            "cuda_graph_modules": [],
        },
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


def test_mxfp8_graph_requires_fp8_primary_parameters():
    with pytest.raises(ValueError, match="MXFP8 CUDA graphs require fp8_param=True"):
        _config(
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fp8_param=False,
            cuda_graph_impl="local",
            cuda_graph_modules=[],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moe_single_grouped_weight": False},
        {"moe_mlp_glu_interleave_size": 32},
        {"activation_func_clamp_value": 4.0},
        {"activation_func_clamp_value": 0.0},
        {"cuda_graph_impl": "full_iteration"},
        {"cuda_graph_impl": "local", "cuda_graph_modules": ["moe_router"]},
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


def _functional_bridge(monkeypatch):
    import dist_moe

    _, module = _module()
    for owner in module.owners:
        owner.main_grad = torch.full_like(owner, 0.25, dtype=torch.float32)
    module._gradient_binding = tuple(runtime.storage_signature(p.main_grad) for p in module.owners)
    module._weight_binding = module._weight_signature()
    module._prepared = tuple(owner.detach() for owner in module.owners)
    module.context = object()
    module._live = True

    def forward(x, ids, scores, w13, w2, context, *, save_for_backward):
        assert w13 is module._prepared[0] and w2 is module._prepared[1]
        assert context is module.context
        assert not torch.is_grad_enabled()
        return torch.ones_like(x), (x, scores) if save_for_backward else None

    def backward(state, dy, *, grad_w13, grad_w2):
        assert grad_w13 is module.owners[0].main_grad
        assert grad_w2 is module.owners[1].main_grad
        grad_w13.add_(1)
        grad_w2.add_(2)
        return dy * 2, torch.full_like(state[1], 0.5)

    functional = SimpleNamespace(
        forward=Mock(side_effect=forward), backward=Mock(side_effect=backward)
    )
    monkeypatch.setattr(dist_moe, "functional", functional, raising=False)
    return module, functional


@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("pack_saved_tensors", [False, True])
def test_autograd_bridge_accumulates_main_grad_and_completes_each_owner(
    monkeypatch, input_grad, pack_saved_tensors
):
    module, functional = _functional_bridge(monkeypatch)
    hooks, counts = [], [0, 0]
    pointers = [owner.main_grad.data_ptr() for owner in module.owners]
    for index, owner in enumerate(module.owners):

        def finish(param, index=index):
            assert param is module.owners[index] and param.grad_added_to_main_grad
            counts[index] += 1
            # Native DDP discards the hook-only gradient without adding its values.
            param.grad = None

        hooks.append(owner.register_post_accumulate_grad_hook(finish))
    for microbatch in range(3):
        module._live = True
        for owner in module.owners:
            owner.grad_added_to_main_grad = False
        x = torch.ones(2, 256, requires_grad=input_grad)
        scores = torch.ones(2, 2, requires_grad=input_grad)
        ids = torch.zeros(2, 2, dtype=torch.int32)
        with (
            torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor.detach(), lambda x: x)
            if pack_saved_tensors
            else nullcontext()
        ):
            output = runtime._DistMoEAutograd.apply(module, x, ids, scores, *module.owners)
        assert output.requires_grad  # Native owners alone must retain the backward edge.
        output.sum().backward(retain_graph=microbatch == 2)
        assert not module._live and counts == [microbatch + 1] * 2
        if input_grad:
            torch.testing.assert_close(x.grad, torch.full_like(x, 2))
            torch.testing.assert_close(scores.grad, torch.full_like(scores, 0.5))
        for index, owner in enumerate(module.owners):
            assert owner.grad is None and owner.main_grad.data_ptr() == pointers[index]
            torch.testing.assert_close(
                owner.main_grad,
                torch.full_like(owner.main_grad, 0.25 + (index + 1) * (microbatch + 1)),
            )
        if microbatch == 2:
            with pytest.raises(RuntimeError, match="one backward"):
                output.sum().backward()
    assert functional.forward.call_count == functional.backward.call_count == 3
    for hook in hooks:
        hook.remove()


def test_autograd_bridge_no_grad_does_not_retain_training_state(monkeypatch):
    module, functional = _functional_bridge(monkeypatch)
    module._live = False
    with torch.no_grad():
        output = runtime._DistMoEAutograd.apply(
            module,
            torch.ones(2, 256),
            torch.zeros(2, 2, dtype=torch.int32),
            torch.ones(2, 2),
            *module.owners,
        )
    assert not output.requires_grad and not module._live
    assert functional.forward.call_args.kwargs == {"save_for_backward": False}
    functional.backward.assert_not_called()


@pytest.mark.parametrize("mutation", ["weight", "owner", "main_grad", "input"])
def test_autograd_bridge_rejects_mutations_before_kernel_backward(monkeypatch, mutation):
    module, functional = _functional_bridge(monkeypatch)
    x = torch.ones(2, 256, requires_grad=True)
    output = runtime._DistMoEAutograd.apply(
        module, x, torch.zeros(2, 2, dtype=torch.int32), torch.ones(2, 2), *module.owners
    )
    with torch.no_grad():
        if mutation == "weight":
            module.owners[0].add_(1)
        elif mutation == "owner":
            module.routed_fc1_weight = torch.nn.Parameter(module.owners[0].clone())
        elif mutation == "main_grad":
            module.owners[0].main_grad = torch.zeros_like(module.owners[0].main_grad)
        else:
            x.add_(1)
    with pytest.raises(RuntimeError, match="storage|version"):
        output.sum().backward()
    functional.backward.assert_not_called()
    assert module._live


def test_autograd_bridge_rejects_higher_order_before_kernel_backward(monkeypatch):
    module, functional = _functional_bridge(monkeypatch)
    x = torch.ones(2, 256, requires_grad=True)
    output = runtime._DistMoEAutograd.apply(
        module, x, torch.zeros(2, 2, dtype=torch.int32), torch.ones(2, 2), *module.owners
    )
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(output.sum(), x, create_graph=True)
    functional.backward.assert_not_called()


@pytest.mark.parametrize("overlap", [False, True])
def test_shared_expert_runs_once_and_adds_to_routed_output(overlap):
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
    layer.use_shared_expert = True
    layer.shared_expert_overlap = overlap
    layer.route = Mock(return_value=(torch.ones(4, 8), torch.ones(4, 8, dtype=torch.bool)))
    hidden = torch.ones(4, 1, 256, device="cuda", requires_grad=True)
    layer.megakernel_experts = torch.nn.Identity()
    layer.megakernel_experts.handles_shared_expert = False
    layer.megakernel_experts.forward = Mock(side_effect=lambda x, *_: x * 2)
    layer.shared_experts_compute = Mock(side_effect=lambda x: x * 3)
    layer.shared_experts = torch.nn.Module()
    layer.shared_experts.stream = torch.cuda.Stream()
    layer.shared_experts.wait_current_stream = lambda: layer.shared_experts.stream.wait_stream(
        torch.cuda.current_stream()
    )
    layer.shared_experts.forward = layer.shared_experts_compute
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


def test_graph_runtime_pins_storage_and_checks_replay_ownership(monkeypatch):
    from megatron.core.transformer.cuda_graphs import (
        _CudagraphGlobalRecord,
        _CudaGraphRunner,
        delete_cuda_graphs,
    )

    _, module = _module()
    module.cuda()
    for owner in module.owners:
        owner.main_grad = torch.zeros_like(owner, dtype=torch.float32)
    module.context = object()
    module._gradient_binding = tuple(runtime.storage_signature(p.main_grad) for p in module.owners)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    module.cuda_graph_capture_prepare()
    with pytest.raises(RuntimeError, match="delete_cuda_graphs"):
        module.load_state_dict({})
    module.cuda_graph_prepare_replay(True, True)
    with pytest.raises(RuntimeError, match="unfinished backward"):
        module.cuda_graph_release()
    module.cuda_graph_finish_replay()
    with torch.no_grad():
        module.owners[0].add_(1)
    module.cuda_graph_prepare_replay(True, True)
    module.cuda_graph_finish_replay()
    runner = _CudaGraphRunner.__new__(_CudaGraphRunner)
    torch.nn.Module.__init__(runner)
    runner.megakernel_backends = (module,)
    runner.grad_enabled = True
    with torch.no_grad(), pytest.raises(RuntimeError, match="no_grad"):
        runner.replay_graph_capture(False, (), {})
    assert not module._live
    module.cuda_graph_prepare_replay(True, True)
    module.cuda_graph_finish_replay()
    runner.cudagraph_created = True
    runner.fwd_graph = runner.bwd_graph = object()
    # Training capture clears its schedule; deletion must still find its runner.
    monkeypatch.setattr(_CudagraphGlobalRecord, "cudagraph_record", [])
    monkeypatch.setattr(_CudagraphGlobalRecord, "cudagraph_inference_record", [])
    monkeypatch.setattr(_CudagraphGlobalRecord, "created_runners", {runner})
    delete_cuda_graphs()
    assert runner.fwd_graph is None and not runner.cudagraph_created
    assert not module._graph_bound
    # Teardown also visits a runner that was only recorded, or already released.
    module.cuda_graph_release()


@pytest.mark.parametrize("mutation", ["weight", "main_grad"])
def test_graph_backward_rejects_changed_storage_before_gpu_work(mutation):
    from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode, _GraphStatus

    _, module = _module()
    for owner in module.owners:
        owner.main_grad = torch.zeros_like(owner, dtype=torch.float32)
    module._gradient_binding = tuple(runtime.storage_signature(p.main_grad) for p in module.owners)
    module._weight_binding = module._weight_signature()
    module._live = True
    grad_output, graph = Mock(), Mock()
    runner = SimpleNamespace(
        megakernel_backends=(module,),
        bwd_graph=graph,
        status=_GraphStatus.BWD_READY,
        static_grad_outputs=(grad_output,),
    )
    with torch.no_grad():
        if mutation == "weight":
            module.owners[0].data = torch.ones_like(module.owners[0])
        else:
            module.owners[0].main_grad = torch.zeros_like(module.owners[0].main_grad)
    with pytest.raises(RuntimeError, match="storage"):
        _CudagraphReplayNode.backward(SimpleNamespace(runner=runner), torch.zeros(1))
    grad_output.copy_.assert_not_called()
    graph.replay.assert_not_called()
    assert module._live


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
