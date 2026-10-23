# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Ownership and weight-layout contracts independent of full model training."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.moe.megakernel.dist_moe import backend as dist_backend
from megatron.core.transformer.moe.megakernel.dist_moe import runtime
from megatron.core.transformer.moe.megakernel.dist_moe.backend import DistMoEMegakernel
from megatron.core.transformer.moe.megakernel.dist_moe.pipeline import DistMoEPipelineSchedule
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
        pipeline_dtype=torch.bfloat16,
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


def _module(**config):
    experts = torch.nn.Module()
    for name, shape in (("linear_fc1", (2, 512, 256)), ("linear_fc2", (2, 256, 256))):
        linear = torch.nn.Module()
        linear.single_grouped_weight = True
        linear.register_parameter(
            "weight", torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
        )
        experts.add_module(name, linear)
    module = DistMoEMegakernel(_config(**config), object(), experts, None, 2)
    return experts, module


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moe_shared_expert_intermediate_size": None},
        {"moe_shared_expert_intermediate_size": 512},
        {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param": True},
        {"fp8": "e4m3", "fp8_recipe": "mxfp8", "fp8_param": False},
        {"cuda_graph_impl": "local", "cuda_graph_modules": []},
        {"num_layers": 4, "pipeline_model_parallel_size": 2},
        {
            "num_layers": 4,
            "pipeline_model_parallel_size": 2,
            "cuda_graph_impl": "local",
            "cuda_graph_modules": [],
        },
        {
            "num_layers": 4,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 2,
            "cuda_graph_impl": "local",
            "cuda_graph_modules": [],
            "fp8": "e4m3",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
        },
        {
            "num_layers": 4,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 2,
        },
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


def test_pipeline_runtime_defers_allocation_and_sizes_each_chunk(monkeypatch):
    import dist_moe

    monkeypatch.setattr(runtime, "_POOLS", [])
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    contexts = []

    def create_context(*, config, **kwargs):
        context = SimpleNamespace(
            config=config, memory_plan=SimpleNamespace(explain=lambda: "mock plan"), close=Mock()
        )
        contexts.append(context)
        return context

    monkeypatch.setattr(dist_moe, "create_context", create_context)
    group = Mock()
    group.size.return_value = 4
    chunks = []
    for _ in range(2):
        layers = []
        for _ in range(2):
            _, module = _module(
                num_layers=8, pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2
            )
            module.ep_group = group
            for owner in module.owners:
                owner.main_grad = torch.zeros_like(owner, dtype=torch.float32)
            layers.append(module)
        chunks.append(torch.nn.ModuleList(layers))
    runtime.initialize_dist_moe_runtime(
        chunks, ddp_config=SimpleNamespace(), max_num_local_input_tokens=256
    )
    assert not contexts and all(module.context is None for chunk in chunks for module in chunk)
    with pytest.raises(RuntimeError, match="already initialized"):
        runtime.initialize_dist_moe_runtime(
            chunks, ddp_config=SimpleNamespace(), max_num_local_input_tokens=256
        )
    runtime.configure_dist_moe_activation_slots(chunks, 5)
    assert len(contexts) == 1
    assert contexts[0].config.num_activation_slots == 5
    assert contexts[0].config.max_moe_layers_per_activation_slot == 2
    runtime.configure_dist_moe_activation_slots(chunks, 5)
    assert len(contexts) == 1
    for chunk in chunks:
        for module in chunk:
            module._graph_bound = True
    # Evaluation and a smaller schedule must not invalidate captured slot views.
    for capacity in (1, 3, 5):
        runtime.configure_dist_moe_activation_slots(chunks, capacity)
        assert all(module.context is contexts[0] for chunk in chunks for module in chunk)
    contexts[0].close.assert_not_called()
    with pytest.raises(RuntimeError, match="delete_cuda_graphs"):
        runtime.configure_dist_moe_activation_slots(chunks, 6)
    for chunk in chunks:
        for module in chunk:
            module.cuda_graph_release()
    chunks[0][0]._live = 1
    with pytest.raises(RuntimeError, match="unfinished backward"):
        runtime.configure_dist_moe_activation_slots(chunks, 6)
    chunks[0][0]._live = 0
    runtime.configure_dist_moe_activation_slots(chunks, {id(chunks[0][0]._runtime_pool): 6})
    assert len(contexts) == 2
    contexts[0].close.assert_called_once()
    runtime.close_dist_moe_runtime(chunks)
    contexts[1].close.assert_called_once()
    assert not runtime._POOLS
    assert all(module._runtime_pool is None for chunk in chunks for module in chunk)


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


def _bridge_call(module, x, ids, scores, *owners, ticket=None):
    call = runtime._DistMoECall(module._weight_binding, ticket) if module._live else None
    return runtime._DistMoEAutograd.apply(module, call, x, ids, scores, *owners)


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
            output = _bridge_call(module, x, ids, scores, *module.owners)
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
        output = _bridge_call(
            module,
            torch.ones(2, 256),
            torch.zeros(2, 2, dtype=torch.int32),
            torch.ones(2, 2),
            *module.owners,
        )
    assert not output.requires_grad and not module._live
    assert functional.forward.call_args.kwargs == {"save_for_backward": False}
    functional.backward.assert_not_called()


def test_multiple_live_calls_complete_independently_and_keep_prepared_weights(monkeypatch):
    module, functional = _functional_bridge(monkeypatch)
    inputs, outputs, tickets = [], [], []
    hooks = [
        owner.register_post_accumulate_grad_hook(lambda param: setattr(param, "grad", None))
        for owner in module.owners
    ]
    prepare = Mock(side_effect=AssertionError("must not overwrite borrowed operands"))
    monkeypatch.setattr(
        "megatron.core.transformer.moe.megakernel.dist_moe.backend.prepare_weight", prepare
    )
    for index in range(3):
        module._live = index + 1
        module._prepare_weights(first_microbatch=True)
        x = torch.full((2, 256), float(index), requires_grad=True)
        ticket = SimpleNamespace(before_backward=Mock(), complete=Mock())
        inputs.append(x)
        tickets.append(ticket)
        outputs.append(
            _bridge_call(
                module,
                x,
                torch.zeros(2, 2, dtype=torch.int32),
                torch.ones(2, 2),
                *module.owners,
                ticket=ticket,
            )
        )
    # Different activation slots permit FIFO or other non-LIFO completion.
    for remaining, index in zip((2, 1, 0), (0, 2, 1)):
        outputs[index].sum().backward()
        assert module._live == remaining
        assert functional.backward.call_args.args[0][0] is inputs[index]
        tickets[index].before_backward.assert_called_once()
        tickets[index].complete.assert_called_once()
        torch.testing.assert_close(inputs[index].grad, torch.full_like(inputs[index], 2))
    prepare.assert_not_called()
    for index, owner in enumerate(module.owners):
        torch.testing.assert_close(
            owner.main_grad, torch.full_like(owner.main_grad, 0.25 + 3 * (index + 1))
        )
    for hook in hooks:
        hook.remove()


def test_live_weights_cannot_refresh_and_call_keeps_its_original_binding(monkeypatch):
    module, functional = _functional_bridge(monkeypatch)
    output = _bridge_call(
        module,
        torch.ones(2, 256, requires_grad=True),
        torch.zeros(2, 2, dtype=torch.int32),
        torch.ones(2, 2),
        *module.owners,
    )
    with torch.no_grad():
        module.owners[0].add_(1)
    with pytest.raises(RuntimeError, match="live forwards"):
        module._prepare_weights(first_microbatch=True)
    # Even accidental replacement of module-wide bookkeeping cannot conceal a
    # changed owner from the already-live call.
    module._weight_binding = module._weight_signature()
    with pytest.raises(RuntimeError, match="storage/version"):
        output.sum().backward()
    functional.backward.assert_not_called()


def test_pipeline_order_checked_before_kernel_backward(monkeypatch):
    module, functional = _functional_bridge(monkeypatch)
    ticket = SimpleNamespace(
        before_backward=Mock(side_effect=RuntimeError("wrong pipeline layer order")),
        complete=Mock(),
    )
    output = _bridge_call(
        module,
        torch.ones(2, 256),
        torch.zeros(2, 2, dtype=torch.int32),
        torch.ones(2, 2),
        *module.owners,
        ticket=ticket,
    )
    with pytest.raises(RuntimeError, match="pipeline layer order"):
        output.sum().backward()
    functional.backward.assert_not_called()
    ticket.complete.assert_not_called()
    assert module._live


@pytest.mark.parametrize("mutation", ["weight", "owner", "main_grad", "input"])
def test_autograd_bridge_rejects_mutations_before_kernel_backward(monkeypatch, mutation):
    module, functional = _functional_bridge(monkeypatch)
    x = torch.ones(2, 256, requires_grad=True)
    output = _bridge_call(
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
    output = _bridge_call(
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
    call = module.cuda_graph_prepare_replay(True, True)
    with pytest.raises(RuntimeError, match="unfinished backward"):
        module.cuda_graph_release()
    with pytest.raises(RuntimeError, match="drained pipeline schedule"):
        with module.cuda_graph_capture_scope(None):
            pytest.fail("capture entered a live graph replay")
    module.cuda_graph_prepare_backward(call)
    module.cuda_graph_finish_replay(call)
    with torch.no_grad():
        module.owners[0].add_(1)
    call = module.cuda_graph_prepare_replay(True, True)
    module.cuda_graph_prepare_backward(call)
    module.cuda_graph_finish_replay(call)
    runner = _CudaGraphRunner.__new__(_CudaGraphRunner)
    torch.nn.Module.__init__(runner)
    runner.megakernel_backends = (module,)
    runner.grad_enabled = True
    with torch.no_grad(), pytest.raises(RuntimeError, match="no_grad"):
        runner.replay_graph_capture(False, (), {})
    assert not module._live
    call = module.cuda_graph_prepare_replay(True, True)
    module.cuda_graph_prepare_backward(call)
    module.cuda_graph_finish_replay(call)
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
def test_graph_backward_rejects_changed_storage_before_gpu_work(monkeypatch, mutation):
    from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode, _GraphStatus

    module = _ready_graph_backend(monkeypatch)
    call = module.cuda_graph_prepare_replay(True, True)
    grad_output, graph = Mock(), Mock()
    runner = SimpleNamespace(
        megakernel_backends=(module,),
        megakernel_calls=(call,),
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


def _ready_graph_backend(monkeypatch, **config):
    """Exercise ownership on CPU; the numeric suite tests actual graph execution."""
    _, module = _module(**config)
    for owner in module.owners:
        owner.main_grad = torch.zeros_like(owner, dtype=torch.float32)
    module._gradient_binding = tuple(runtime.storage_signature(p.main_grad) for p in module.owners)
    module.context = SimpleNamespace(select_activation_slot=Mock())
    monkeypatch.setattr(
        dist_backend, "prepare_weight", lambda owner, **kwargs: (owner.detach(), None)
    )
    module.cuda_graph_capture_prepare()
    return module


def test_pipeline_graph_replay_owns_each_call_until_native_backward_returns(monkeypatch):
    modules = [
        _ready_graph_backend(monkeypatch, num_layers=4, pipeline_model_parallel_size=2)
        for _ in range(2)
    ]
    for module in modules:
        module._chunk_index = 0
        module.context = modules[0].context
        module._runtime_pool = modules[0].context
    monkeypatch.setattr(runtime, "configure_dist_moe_activation_slots", Mock())
    scope = DistMoEPipelineSchedule([torch.nn.ModuleList(modules)], [(0, 0), (1, 0)], 2)
    calls = []
    for microbatch in range(2):
        with scope._step("forward", 0, microbatch):
            calls.append(
                tuple(
                    module.cuda_graph_prepare_replay(microbatch == 0, True, module.cuda_graph_key())
                    for module in modules
                )
            )
    assert all(module._live == 2 for module in modules)
    for microbatch in range(2):
        with scope._step("backward", 0, microbatch):
            first, last = calls[microbatch]
            if microbatch == 0:
                # Reject a future microbatch and the wrong layer before marking either started.
                with pytest.raises(RuntimeError, match="reverse layer order"):
                    modules[-1].cuda_graph_prepare_backward(calls[1][-1])
                with pytest.raises(RuntimeError, match="reverse layer order"):
                    modules[0].cuda_graph_prepare_backward(first)
                with pytest.raises(RuntimeError, match="completion"):
                    modules[-1].cuda_graph_finish_replay(last)
                copied = runtime._DistMoECall(last.weight_binding, last.ticket)
                with pytest.raises(RuntimeError, match="live forward"):
                    modules[-1].cuda_graph_prepare_backward(copied)
            # A whole-layer graph can contain multiple backends: validate all before launch.
            for module, call in reversed(tuple(zip(modules, calls[microbatch]))):
                module.cuda_graph_prepare_backward(call)
                with pytest.raises(RuntimeError, match="live forward"):
                    module.cuda_graph_prepare_backward(call)
            for module, call in reversed(tuple(zip(modules, calls[microbatch]))):
                module.cuda_graph_finish_replay(call)
                with pytest.raises(RuntimeError, match="completion"):
                    module.cuda_graph_finish_replay(call)
            # Canonical owner hooks still execute inside this native backward interval.
            assert (0, microbatch) in scope.leases
            assert all(module._live == 1 - microbatch for module in modules)
        assert (0, microbatch) not in scope.leases
    scope.finish()
    assert all(not module._graph_calls for module in modules)


@pytest.mark.parametrize("mutation", ["slot", "depth", "schedule"])
def test_graph_binding_mismatch_is_rejected_before_forward_launch(monkeypatch, mutation):
    from megatron.core.transformer.cuda_graphs import _CudagraphReplayNode, _CudaGraphRunner

    module = _ready_graph_backend(monkeypatch, num_layers=2, pipeline_model_parallel_size=2)
    module._chunk_index = 0
    module._runtime_pool = module.context
    monkeypatch.setattr(runtime, "configure_dist_moe_activation_slots", Mock())
    scope = DistMoEPipelineSchedule([torch.nn.ModuleList([module])], [(0, 0)], 0)
    runner = _CudaGraphRunner.__new__(_CudaGraphRunner)
    torch.nn.Module.__init__(runner)
    runner.megakernel_backends = (module,)
    runner.grad_enabled = True
    runner.get_mismatch_errors = lambda *args: []
    runner._gtp_fwd_params_to_ensure_ready = ()
    runner.get_tensors = Mock()
    launch = Mock()
    monkeypatch.setattr(_CudagraphReplayNode, "apply", launch)
    with scope._step("forward", 0, 0):
        events, slot, depth = module.cuda_graph_key()
        invalid_key = {
            "slot": (events, slot + 1, depth),
            "depth": (events, slot, depth + 1),
            "schedule": (events + events, slot, depth),
        }[mutation]
        runner.megakernel_graph_keys = (invalid_key,)
        with pytest.raises(RuntimeError, match="schedule/slot changed"):
            runner.replay_graph_capture(True, (), {})
        launch.assert_not_called()
        runner.get_tensors.assert_not_called()
        module.context.select_activation_slot.assert_not_called()
        assert not module._live
        # A rejected preflight leaves the valid invocation available.
        call = module.cuda_graph_prepare_replay(True, True, module.cuda_graph_key())
    with scope._step("backward", 0, 0):
        module.cuda_graph_prepare_backward(call)
        module.cuda_graph_finish_replay(call)
    scope.finish()


def test_capture_scope_keeps_outstanding_frames_and_warmup_independent(monkeypatch):
    module, functional = _functional_bridge(monkeypatch)
    hooks = [
        owner.register_post_accumulate_grad_hook(lambda param: setattr(param, "grad", None))
        for owner in module.owners
    ]
    module.config.pipeline_model_parallel_size = 2
    module._live = 0
    module._stream = 7
    module.context = SimpleNamespace(
        config=SimpleNamespace(max_num_local_input_tokens=2), select_activation_slot=Mock()
    )
    monkeypatch.setattr(
        dist_backend, "prepare_weight", lambda owner, **kwargs: (owner.detach(), None)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        dist_backend,
        "routing_map_to_indices",
        lambda probs, routing, topk: (probs[:, :topk], torch.zeros(2, topk, dtype=torch.long)),
    )

    def forward():
        return module(
            torch.ones(2, 256, requires_grad=True),
            torch.ones(2, 8, requires_grad=True),
            torch.ones(2, 8, dtype=torch.bool),
        )

    key0, key1 = ((), 0, 2), ((), 1, 2)
    with module.cuda_graph_capture_scope(key0):
        first = forward()
    assert module._live == 1
    with module.cuda_graph_capture_scope(key1):
        # Native capture interposes a local warmup F+B while another runner is outstanding.
        forward().sum().backward()
        assert module._live == 1
        second = forward()
        assert module._live == 2
    assert not module._capturing_graph and module._graph_capture_key is None
    first.sum().backward()
    assert module._live == 1
    second.sum().backward()
    assert module._live == 0
    assert [call.args for call in module.context.select_activation_slot.call_args_list] == [
        (0, 2),
        (1, 2),
        (1, 2),
    ]
    assert functional.backward.call_count == 3
    with pytest.raises(ValueError, match="capture failure"):
        with module.cuda_graph_capture_scope(key0):
            raise ValueError("capture failure")
    assert not module._capturing_graph and module._graph_capture_key is None
    module._pipeline_schedule = object()
    with pytest.raises(RuntimeError, match="drained pipeline schedule"):
        with module.cuda_graph_capture_scope(key0):
            pytest.fail("capture entered an active native schedule")
    for hook in hooks:
        hook.remove()


def test_pipeline_recapture_discards_deleted_runners_and_recreates_pool(monkeypatch):
    from megatron.core.transformer import cuda_graphs

    manager = cuda_graphs.CudaGraphManager.__new__(cuda_graphs.CudaGraphManager)
    torch.nn.Module.__init__(manager)
    deleted = SimpleNamespace(mempool=None)
    manager.cudagraph_runners = [deleted]
    manager.custom_cudagraphs_lookup_table = {"old": deleted}
    manager.need_backward = True
    manager.func = object()
    manager._num_warmup_steps = None
    pool = object()
    fresh = SimpleNamespace(mempool=pool, status=cuda_graphs._GraphStatus.FWD_READY)
    constructor = Mock(return_value=fresh)
    observer = Mock()
    monkeypatch.setattr(cuda_graphs, "_CudaGraphRunner", constructor)
    monkeypatch.setattr(cuda_graphs, "HAVE_GTP", False)
    monkeypatch.setattr(cuda_graphs.CudaGraphManager, "global_mempool", None)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", Mock(return_value=pool))
    monkeypatch.setattr(cuda_graphs._CudagraphGlobalRecord, "cudagraph_created", False)
    monkeypatch.setattr(
        cuda_graphs._CudagraphGlobalRecord, "_enable_saved_tensors_observer", observer
    )
    model = object()
    assert manager.get_cudagraph_runner(model, (), {}, False) is fresh
    assert manager.cudagraph_runners == [fresh]
    assert not manager.custom_cudagraphs_lookup_table
    constructor.assert_called_once_with(model, pool, (), {}, manager.func, True)
    observer.assert_called_once()
    monkeypatch.setattr(cuda_graphs._CudagraphGlobalRecord, "cudagraph_created", True)
    assert manager.get_cudagraph_runner(model, (), {}, False) is fresh
    torch.cuda.graph_pool_handle.assert_called_once()


def test_no_grad_eval_preserves_training_runners_and_runtime(monkeypatch):
    from megatron.core.transformer import cuda_graphs

    module = _ready_graph_backend(monkeypatch, num_layers=2, pipeline_model_parallel_size=2)
    manager = cuda_graphs.CudaGraphManager.__new__(cuda_graphs.CudaGraphManager)
    torch.nn.Module.__init__(manager)
    manager.need_backward = True
    manager.func = Mock(return_value="eager output")
    manager.get_cudagraph_runner = Mock()
    context = module.context
    monkeypatch.setattr(cuda_graphs._CudagraphGlobalRecord, "cudagraph_created", True)
    with torch.no_grad():
        assert manager(module, ("input",), {}) == "eager output"
    manager.func.assert_called_once_with("input")
    manager.get_cudagraph_runner.assert_not_called()
    assert module.context is context and module._graph_bound
    assert not module._live and not module._graph_calls


@pytest.mark.parametrize("has_megakernel", [False, True])
def test_capture_restores_megakernel_execution_stream(monkeypatch, has_megakernel):
    from megatron.core.transformer import cuda_graphs

    record = cuda_graphs._CudagraphGlobalRecord
    execution_stream, capture_stream = Mock(), Mock()
    current_stream = Mock(side_effect=[execution_stream, capture_stream])
    set_stream = Mock()
    cleanup = Mock()
    runner = SimpleNamespace(megakernel_backends=(object(),) if has_megakernel else ())
    monkeypatch.setattr(record, "cudagraph_record", [(runner, "fwd")])
    monkeypatch.setattr(record, "_create_cudagraphs", Mock(return_value="capture result"))
    monkeypatch.setattr(record, "_disable_saved_tensors_observer", cleanup)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(torch.cuda, "set_stream", set_stream)
    assert record.create_cudagraphs() == "capture result"
    cleanup.assert_called_once()
    if has_megakernel:
        execution_stream.wait_stream.assert_called_once_with(capture_stream)
        set_stream.assert_called_once_with(execution_stream)
    else:
        current_stream.assert_not_called()
        set_stream.assert_not_called()


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
