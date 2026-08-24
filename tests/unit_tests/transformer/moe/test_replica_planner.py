# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for deterministic replica planning and compact HybridEP routing."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.replica_planner import (
    ReplicaPlan,
    _collect_replica_projection_specs,
    map_replica_plan_to_hybridep,
    start_replica_grad_reduce_after_expert_backward,
    start_replica_weight_prefetch_before_combine_backward,
    wait_replica_grad_reduce_after_dispatch_backward,
    wait_replica_weight_prefetch_before_expert_backward,
)


def test_replica_hybridep_rank_layout_requires_equal_shapes(monkeypatch):
    from megatron.core.transformer.moe.token_dispatcher import (
        _validate_replica_rank_layout,
    )

    group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fake_all_gather_object(outputs, value, group):
        outputs[:] = [value, (value[0] + 1, value[1])]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    with pytest.raises(ValueError, match="equal local token counts"):
        _validate_replica_rank_layout(
            group, num_tokens=8, hidden_dim=16, backend_name="Replica-HybridEP"
        )


def test_replica_hybridep_binds_the_cutedsl_bridge(monkeypatch):
    from megatron.core.transformer.moe import token_dispatcher
    from megatron.core.transformer.moe.token_dispatcher import _ReplicaHybridEPManager

    captured = {}
    bridge = object()

    def fake_bridge(**kwargs):
        captured.update(kwargs)
        return bridge

    class FakeExperts:
        def set_replica_weight_bridge(self, value):
            self.bound_bridge = value

    monkeypatch.setattr(token_dispatcher, "HybridEPReplicaWeightBridge", fake_bridge)
    manager = _ReplicaHybridEPManager.__new__(_ReplicaHybridEPManager)
    manager.group = object()
    manager.semantic_num_experts = 8
    manager.num_owned_experts = 2
    manager.config = SimpleNamespace(
        moe_flex_dispatcher_num_sms=12,
        moe_hybridep_num_blocks_permute=7,
        moe_hybridep_num_blocks_unpermute=5,
        moe_hybridep_num_sms_preprocessing=9,
    )
    experts = FakeExperts()
    manager.bind_experts(experts)

    assert manager._bridge is bridge
    assert experts.bound_bridge is bridge
    assert captured["num_experts"] == 8
    assert captured["num_local_experts"] == 2


def test_replica_hybridep_metadata_uses_routing_map_for_zero_probability_routes():
    """A selected zero-probability expert must not be replaced by a tied zero."""
    from megatron.core.transformer.moe.token_dispatcher import _ReplicaHybridEPManager

    manager = _ReplicaHybridEPManager.__new__(_ReplicaHybridEPManager)
    manager.semantic_num_experts = 4
    manager.router_topk = 2
    routing_map = torch.tensor([[False, True, False, True], [True, False, True, False]])
    probs = torch.tensor(
        [[0.0, 0.75, 0.0, 0.0], [0.6, 0.0, 0.4, 0.0]], requires_grad=True
    )

    manager.setup_metadata(routing_map, probs)

    actual_routes = [set(row) for row in manager.semantic_token_indices.tolist()]
    assert actual_routes == [{1, 3}, {0, 2}]
    torch.testing.assert_close(
        manager.semantic_token_probs.sum(dim=-1), torch.tensor([0.75, 1.0])
    )
    manager.semantic_token_probs.sum().backward()
    torch.testing.assert_close(probs.grad, routing_map.to(probs.dtype))


def test_replica_hybridep_keeps_virtual_routes_compact():
    plan = ReplicaPlan(
        virtual_experts=torch.tensor([[1, 6], [3, 4]], dtype=torch.int64),
        experts_to_copy=torch.empty((0,), dtype=torch.int32),
    )
    probs = torch.tensor([[0.75, 0.25], [0.6, 0.4]])
    routing_map, dense_probs = map_replica_plan_to_hybridep(plan, probs, num_experts=8)

    assert routing_map.shape == (2, 8)
    assert dense_probs.shape == (2, 8)
    assert routing_map[0, 1] and routing_map[0, 6]
    assert dense_probs[1, 3] == 0.6


def test_hybridep_compact_router_gradient_is_gathered_from_dense_result(monkeypatch):
    dense_prob_grad = torch.tensor([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])

    class FakeHybridEPBuffer:
        def dispatch_with_permute(self, **kwargs):
            return (
                kwargs["hidden"].clone(),
                kwargs["topk_weights"].clone(),
                None,
                torch.ones(2, dtype=torch.int32),
                (torch.tensor(0),),
            )

        def combine_with_unpermute(self, **kwargs):
            return kwargs["hidden"], dense_prob_grad

    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", FakeHybridEPBuffer())
    hidden = torch.randn(2, 3, requires_grad=True)
    topk_idx = torch.tensor([[1, 3], [0, -1]], dtype=torch.int64)
    topk_weights = torch.randn(2, 2, requires_grad=True)
    outputs = fused_a2a.HybridEPDispatch.apply(
        hidden,
        None,
        None,
        object(),
        2,
        None,
        None,
        None,
        None,
        False,
        4,
        None,
        108,
        topk_idx,
        topk_weights,
        4,
    )
    (outputs[0].sum() + outputs[1].sum()).backward()

    torch.testing.assert_close(
        topk_weights.grad, torch.tensor([[11.0, 13.0], [20.0, 0.0]])
    )


def test_replica_async_collectives_span_transport_backward():
    events = []
    plan = object()

    class FakeBridge:
        source_parameters = (
            torch.nn.Parameter(torch.ones(())),
            torch.nn.Parameter(torch.ones(())),
        )

        def start_prefetch(self, current_plan, *, retain_for_grad=False):
            assert current_plan is plan and retain_for_grad
            events.append("start_prefetch")

        def wait_prefetch(self, current_plan):
            assert current_plan is plan
            events.append("wait_prefetch")

        def start_grad_reduce(self, current_plan):
            assert current_plan is plan
            events.append("start_grad_reduce")

        def wait_grad_reduce(self, current_plan):
            assert current_plan is plan
            events.append("wait_grad_reduce")

    class BackwardMarker(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, label):
            ctx.label = label
            return value

        @staticmethod
        def backward(ctx, grad):
            events.append(ctx.label)
            return grad, None

    bridge = FakeBridge()
    hidden = torch.ones((), requires_grad=True)
    hidden = wait_replica_grad_reduce_after_dispatch_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "dispatch_backward")
    hidden = start_replica_grad_reduce_after_expert_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "expert_backward")
    hidden = wait_replica_weight_prefetch_before_expert_backward(hidden, bridge, plan)
    hidden = BackwardMarker.apply(hidden, "combine_backward")
    hidden = start_replica_weight_prefetch_before_combine_backward(hidden, bridge, plan)
    hidden.backward()

    assert events == [
        "start_prefetch",
        "combine_backward",
        "wait_prefetch",
        "expert_backward",
        "start_grad_reduce",
        "dispatch_backward",
        "wait_grad_reduce",
    ]
    assert all(parameter.grad is None for parameter in bridge.source_parameters)


def _set_main_grad(parameter):
    parameter.main_grad = torch.zeros(
        parameter.shape, dtype=torch.float32, device=parameter.device
    )
    parameter.grad_added_to_main_grad = False
    parameter.overwrite_main_grad = True


def _set_main_grads(layer):
    for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
        if getattr(linear, "single_grouped_weight", False):
            parameters = (linear.get_parameter("weight"),)
        else:
            parameters = tuple(
                linear.get_parameter(f"weight{i}") for i in range(linear.num_gemms)
            )
        for parameter in parameters:
            _set_main_grad(parameter)
    if layer.config.moe_latent_size is not None:
        _set_main_grad(layer.fc1_latent_proj.weight)
        _set_main_grad(layer.fc2_latent_proj.weight)


def _stack_linear_main_grad(linear):
    if getattr(linear, "single_grouped_weight", False):
        return linear.get_parameter("weight").main_grad.detach().clone()
    return torch.stack(
        tuple(
            linear.get_parameter(f"weight{i}").main_grad.detach()
            for i in range(linear.num_gemms)
        )
    )


def _weight_storage_ptrs(weight):
    if hasattr(weight, "_rowwise_data"):
        return (
            weight._rowwise_data.data_ptr(),
            weight._rowwise_scale_inv.data_ptr(),
            weight._columnwise_data.data_ptr(),
            weight._columnwise_scale_inv.data_ptr(),
        )
    return (weight.data_ptr(),)


def _assert_replica_mxfp8_prefetch_exact(bridge, orientation):
    """Check every active virtual MXFP8 component against its owning rank."""
    assert orientation in ("rowwise", "columnwise")
    component_names = (
        ("_rowwise_data", "_rowwise_scale_inv")
        if orientation == "rowwise"
        else ("_columnwise_data", "_columnwise_scale_inv")
    )
    local_errors = []
    for projection_index, projection in enumerate(bridge.projections):
        for component_name in component_names:
            local_sources = torch.stack(
                tuple(
                    getattr(source, component_name)
                    for source in projection.source_tensors
                )
            )
            gathered_sources = [
                torch.empty_like(local_sources) for _ in range(bridge.world_size)
            ]
            torch.distributed.all_gather(
                gathered_sources, local_sources, group=bridge.group
            )
            for slot, global_expert in enumerate(
                bridge.last_plan.experts_to_copy[bridge.rank].tolist()
            ):
                if global_expert < 0:
                    continue
                owner_rank = global_expert // bridge.num_local_experts
                owner_expert = global_expert % bridge.num_local_experts
                actual = getattr(projection.virtual_weight[slot], component_name)
                expected = gathered_sources[owner_rank][owner_expert]
                if not torch.equal(actual, expected):
                    local_errors.append(
                        f"projection={projection_index} component={component_name} "
                        f"slot={slot} expert={global_expert}"
                    )
    any_error = torch.tensor(
        int(bool(local_errors)), dtype=torch.int32, device=bridge.device
    )
    torch.distributed.all_reduce(
        any_error, op=torch.distributed.ReduceOp.MAX, group=bridge.group
    )
    assert (
        not any_error.item()
    ), f"rank {bridge.rank} {orientation} MXFP8 prefetch mismatch: " + (
        ", ".join(local_errors) if local_errors else "reported by another rank"
    )


def _run_replica_hybridep_full_layer_parity(
    monkeypatch,
    backend,
    activation_func,
    gated_linear_unit,
    weighted_squared_relu,
    glu_interleave,
    moe_latent_size,
    single_grouped_weight=True,
    mxfp8=False,
    gtp=False,
    reference_dispatcher="alltoall",
    verify_hybridep_contract=False,
):
    """Compare full expert/router fwd+bwd and grouped main_grads on 4 NVLink GPUs."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip(
            "replica_hybridep distributed coverage requires a 4-rank torchrun launch"
        )

    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv(
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1" if single_grouped_weight else "0"
    )
    expert_model_parallel_size = 2 if gtp else 4
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        expert_gtp_remat_size=2 if gtp else 1,
    )
    if gtp:
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        # This test isolates the bridge's explicit materialize-before-exchange
        # dependency. The production-script test covers linked async GTP chains.
        update_gtp_config(weight_prefetch=False, async_reduction=False)
    torch.manual_seed(1234)

    common = {
        "num_layers": 1,
        "hidden_size": 1024,
        "ffn_hidden_size": 1024,
        "moe_ffn_hidden_size": 1024,
        "num_attention_heads": 8,
        "num_moe_experts": 4,
        "expert_model_parallel_size": expert_model_parallel_size,
        "expert_tensor_parallel_size": 1,
        "expert_tensor_parallel_num_weight_shards": 2 if gtp else 1,
        "moe_router_topk": 2,
        "moe_router_load_balancing_type": "none",
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "moe_single_grouped_weight": single_grouped_weight,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_cpu_initialization": False,
        "activation_func": activation_func,
        "gated_linear_unit": gated_linear_unit,
        "use_fused_weighted_squared_relu": weighted_squared_relu,
        "moe_mlp_glu_interleave_size": glu_interleave,
        "moe_latent_size": moe_latent_size,
    }
    if mxfp8:
        common.update(
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fp8_param=True,
            moe_router_padding_for_quantization=True,
        )
    if reference_dispatcher == "alltoall":
        reference_config = TransformerConfig(
            **common, moe_token_dispatcher_type="alltoall"
        )
    elif reference_dispatcher == "hybridep":
        reference_config = TransformerConfig(
            **common,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
        )
    else:
        raise ValueError(f"Unsupported reference dispatcher {reference_dispatcher!r}.")
    backend_config = {}
    if backend == "replica_hybridep":
        backend_config["moe_expert_rank_capacity_factor"] = 2.0
    replica_config = TransformerConfig(
        **common,
        **backend_config,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend=backend,
    )
    mlp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=4, moe_grouped_gemm=True
    ).submodules.mlp
    submodules = get_submodules(mlp_spec)

    try:
        if mxfp8:
            from transformer_engine.common.recipe import MXFP8BlockScaling
            from transformer_engine.pytorch import fp8_model_init

            with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                ref_layer = MoELayer(reference_config, submodules).cuda()
            with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                replica_layer = MoELayer(replica_config, submodules).cuda()
        else:
            ref_layer = MoELayer(reference_config, submodules).cuda()
            replica_layer = MoELayer(replica_config, submodules).cuda()
        if mxfp8 and moe_latent_size is not None:
            # In production DDP exposes an MXFP8 parameter's main-grad buffer
            # through its distributed-weight wrapper. This focused MoELayer
            # test has no DDP wrapper, so let the two ordinary latent linears
            # return wgrads through autograd; expert wgrads remain fused and
            # exercise replica reduction.
            for layer in (ref_layer, replica_layer):
                layer.fc1_latent_proj.fuse_wgrad_accumulation = False
                layer.fc2_latent_proj.fuse_wgrad_accumulation = False
        replica_layer.load_state_dict(ref_layer.state_dict())
        assert replica_layer.state_dict().keys() == ref_layer.state_dict().keys()
        _set_main_grads(ref_layer)
        _set_main_grads(replica_layer)
        if backend.startswith("replica_"):
            bridge = replica_layer.token_dispatcher._comm_manager._bridge
            bridge.prepare_runtime_parameters()
            if mxfp8:
                reference_specs, _ = _collect_replica_projection_specs(
                    ref_layer.experts,
                    num_local_experts=bridge.num_local_experts,
                    backend_name="test-alltoall",
                )
                for reference, replica in zip(reference_specs, bridge.projections):
                    for reference_weight, replica_weight in zip(
                        reference.source_tensors, replica.source_tensors
                    ):
                        for component_name in (
                            "_rowwise_data",
                            "_rowwise_scale_inv",
                            "_columnwise_data",
                            "_columnwise_scale_inv",
                        ):
                            getattr(replica_weight, component_name).copy_(
                                getattr(reference_weight, component_name)
                            )
                            torch.testing.assert_close(
                                getattr(replica_weight, component_name),
                                getattr(reference_weight, component_name),
                                rtol=0,
                                atol=0,
                            )
            if mxfp8:
                with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                    second_layer = MoELayer(replica_config, submodules).cuda()
            else:
                second_layer = MoELayer(replica_config, submodules).cuda()
            second_bridge = second_layer.token_dispatcher._comm_manager._bridge
            assert second_bridge.workspace is bridge.workspace
            second_bridge.destroy()
            del second_layer
            for projection, runtime_weights in zip(
                bridge.projections,
                (bridge.runtime_fc1_weights, bridge.runtime_fc2_weights),
            ):
                assert len(runtime_weights) == bridge.num_runtime_experts
                native_weights = projection.source_tensors
                if projection.packed_runtime_grad is not None:
                    native_grads = tuple(
                        projection.packed_runtime_grad[: bridge.num_local_experts]
                    )
                elif projection.gtp_leader is not None:
                    native_grads = (
                        tuple(projection.gtp_native_grad)
                        if projection.gtp_native_grad is not None
                        else projection.gtp_wgrad_tensors
                    )
                elif len(projection.parameters) == 1:
                    native_grads = tuple(
                        projection.parameters[0].main_grad.view(
                            bridge.num_local_experts, *projection.member_shape
                        )
                    )
                else:
                    native_grads = tuple(
                        parameter.main_grad for parameter in projection.parameters
                    )
                for index, runtime_weight in enumerate(runtime_weights):
                    if index < bridge.num_local_experts:
                        expected_weight = native_weights[index]
                        expected_grad = native_grads[index]
                    else:
                        slot = index - bridge.num_local_experts
                        expected_weight = projection.virtual_weight[slot]
                        expected_grad = projection.virtual_grad[slot]
                    assert _weight_storage_ptrs(runtime_weight) == _weight_storage_ptrs(
                        expected_weight
                    )
                    assert (
                        runtime_weight.main_grad.data_ptr() == expected_grad.data_ptr()
                    )

        torch.manual_seed(1234)
        test_input = torch.randn(2, 4, 1024, device="cuda", dtype=torch.bfloat16)

        def run(layer, *, replica_bridge=None):
            hidden = test_input.detach().clone().requires_grad_(True)
            output, _ = layer(hidden)
            if replica_bridge is not None and mxfp8:
                _assert_replica_mxfp8_prefetch_exact(replica_bridge, "rowwise")
            output.float().sum().backward()
            if replica_bridge is not None:
                assert all(
                    parameter.grad is None
                    for parameter in replica_bridge.source_parameters
                )
                assert all(
                    runtime_parameter.grad is None
                    for projection in replica_bridge.projections
                    for runtime_parameter in projection.runtime_parameters
                )
            if replica_bridge is not None and mxfp8:
                _assert_replica_mxfp8_prefetch_exact(replica_bridge, "columnwise")
            values = [
                output.detach(),
                hidden.grad.detach(),
                layer.router.weight.grad.detach().clone(),
                _stack_linear_main_grad(layer.experts.linear_fc1),
                _stack_linear_main_grad(layer.experts.linear_fc2),
            ]
            if layer.config.moe_latent_size is not None:
                latent_grads = []
                for latent_projection in (
                    layer.fc1_latent_proj,
                    layer.fc2_latent_proj,
                ):
                    gradient = (
                        latent_projection.weight.main_grad
                        if latent_projection.fuse_wgrad_accumulation
                        else latent_projection.weight.grad
                    )
                    latent_grads.append(gradient.detach().clone())
                values.extend(latent_grads)
            return values

        ref_values = run(ref_layer)
        if reference_dispatcher == "hybridep":
            # HybridEP owns one process-global buffer for a fixed local-expert
            # count. Baseline and replica layouts use N and 2N respectively,
            # so reinitialize it between the two sequential comparisons.
            torch.cuda.synchronize()
            torch.distributed.barrier()
            fused_a2a.reset_hybrid_ep_buffer()
            torch.distributed.barrier()
        replica_values = run(replica_layer, replica_bridge=bridge)
        if reference_dispatcher == "hybridep":
            active_replica = torch.any(bridge.last_plan.experts_to_copy >= 0).to(
                torch.int32
            )
            torch.distributed.all_reduce(
                active_replica, op=torch.distributed.ReduceOp.MAX
            )
            assert (
                active_replica.item()
            ), "HybridEP parity must exercise an active replica"
        if backend.startswith("replica_") and not mxfp8:
            for projection in bridge.projections:
                torch.testing.assert_close(
                    projection.virtual_grad,
                    torch.zeros_like(projection.virtual_grad),
                    rtol=0,
                    atol=0,
                )
        value_names = [
            "output",
            "input grad",
            "router grad",
            "FC1 main_grad",
            "FC2 main_grad",
        ]
        if moe_latent_size is not None:
            value_names.extend(["latent FC1 main_grad", "latent FC2 main_grad"])
        for value_name, actual, expected in zip(
            value_names, replica_values, ref_values
        ):
            if verify_hybridep_contract and value_name not in (
                "FC1 main_grad",
                "FC2 main_grad",
            ):
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0,
                    atol=0,
                    msg=lambda msg: f"{value_name} must be bitwise equal: {msg}",
                )
                continue
            if verify_hybridep_contract:
                # A replicated expert's wgrad is accumulated from independently
                # rounded FP32 partials. This changes addition order but not the
                # mathematical gradient.
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-7,
                    atol=2e-6,
                    msg=lambda msg: f"{value_name} reduction-order bound: {msg}",
                )
                continue
            if mxfp8:
                # Replica placement changes the token population of each MX
                # quantization block. Per-element absolute tolerances are not
                # stable under those different block boundaries. Bound
                # execution-level noise while the raw weights and scales
                # remain byte-exact above; these limits are far below the
                # corruption produced by a wrong expert or scale mapping.
                if "main_grad" in value_name:
                    atol = 32.0
                elif value_name == "router grad":
                    atol = 16.0
                else:
                    atol = 0.75
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0.2,
                    atol=atol,
                    msg=lambda msg: f"{value_name}: {msg}",
                )
                continue
            # Different dispatchers may change BF16 reduction ordering even
            # when replica planning leaves the mathematical result unchanged.
            rtol = 2e-2
            atol = 2e-2
            torch.testing.assert_close(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                msg=lambda msg: f"{value_name}: {msg}",
            )
        if (
            backend.startswith("replica_")
            and not mxfp8
            and not gtp
            and not verify_hybridep_contract
        ):
            # Weight transfer runs at the eager prefetch boundary outside the
            # configured expert graph. Capture consumers of the stable native-
            # then-virtual wrappers, refresh virtual slots from changed native
            # weights, and prove replay observes unchanged wrapper addresses.
            plan = bridge.last_plan
            bridge.prefetch(plan)
            runtime_weight_tuples = (
                bridge.runtime_fc1_weights,
                bridge.runtime_fc2_weights,
            )
            stable_pointers = tuple(
                tuple(weight.data_ptr() for weight in runtime_weights)
                for runtime_weights in runtime_weight_tuples
            )
            captured_weights = tuple(
                torch.empty(
                    (bridge.num_runtime_experts, *projection.member_shape),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for projection in bridge.projections
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for captured, runtime_weights in zip(
                    captured_weights, runtime_weight_tuples
                ):
                    for index, runtime_weight in enumerate(runtime_weights):
                        captured[index].copy_(runtime_weight)
            before_source_update = tuple(
                torch.stack(
                    tuple(runtime_weight.detach() for runtime_weight in runtime_weights)
                )
                for runtime_weights in runtime_weight_tuples
            )
            for projection in bridge.projections:
                for parameter in projection.parameters:
                    rowwise_data = getattr(parameter, "rowwise_data", parameter.data)
                    rowwise_data.add_(1000)
            for runtime_weights, previous in zip(
                runtime_weight_tuples, before_source_update
            ):
                current = torch.stack(
                    tuple(runtime_weight.detach() for runtime_weight in runtime_weights)
                )
                torch.testing.assert_close(
                    current[: bridge.num_local_experts],
                    previous[: bridge.num_local_experts] + 1000,
                    rtol=0,
                    atol=0,
                    msg=lambda msg: f"rank {bridge.rank} native alias update check: {msg}",
                )
                torch.testing.assert_close(
                    current[bridge.num_local_experts :],
                    previous[bridge.num_local_experts :],
                    rtol=0,
                    atol=0,
                    msg=lambda msg: f"rank {bridge.rank} virtual isolation check: {msg}",
                )
            plan_snapshot = plan.experts_to_copy.clone()
            # The normal training path may cache this exact plan in the shared
            # workspace.  Evict that cache here so this check exercises an
            # actual refresh from the modified optimizer-owned weights.
            bridge.workspace.resident_bridge = None
            bridge.workspace.resident_plan = None
            bridge.prefetch(plan)
            torch.testing.assert_close(
                plan.experts_to_copy,
                plan_snapshot,
                rtol=0,
                atol=0,
                msg=lambda msg: f"rank {bridge.rank} plan stability check: {msg}",
            )
            after_prefetch = tuple(
                torch.stack(
                    tuple(runtime_weight.detach() for runtime_weight in runtime_weights)
                )
                for runtime_weights in runtime_weight_tuples
            )
            graph.replay()
            torch.cuda.synchronize()
            active_slots = plan.experts_to_copy[bridge.rank] >= 0
            for pointers, captured, previous, current, runtime_weights in zip(
                stable_pointers,
                captured_weights,
                before_source_update,
                after_prefetch,
                runtime_weight_tuples,
            ):
                torch.testing.assert_close(captured, current, rtol=0, atol=0)
                if torch.any(active_slots):
                    assert torch.any(
                        current[bridge.num_local_experts :][active_slots]
                        != previous[bridge.num_local_experts :][active_slots]
                    )
                torch.testing.assert_close(
                    current[bridge.num_local_experts :][~active_slots],
                    previous[bridge.num_local_experts :][~active_slots],
                    rtol=0,
                    atol=0,
                )
                assert (
                    tuple(weight.data_ptr() for weight in runtime_weights) == pointers
                )
    finally:
        fused_a2a.reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()
        if gtp:
            update_gtp_config(weight_prefetch=True, async_reduction=True)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
@pytest.mark.parametrize("single_grouped_weight", [True, False])
def test_replica_hybridep_full_layer_gradients_match_alltoall(
    monkeypatch, single_grouped_weight
):
    """Check replica-planned HybridEP output and all training gradients."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        single_grouped_weight,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_gtp_gradients_match_alltoall(monkeypatch):
    """Check discrete GTP-sharded experts through replica weight and grad exchange."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        single_grouped_weight=False,
        gtp=True,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
@pytest.mark.parametrize("single_grouped_weight", [True, False])
def test_replica_hybridep_mxfp8_gradients_match_alltoall(
    monkeypatch, single_grouped_weight
):
    """Check native MXFP8 replica outputs and gradients against alltoall."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        single_grouped_weight,
        mxfp8=True,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
@pytest.mark.parametrize("mxfp8", [False, True])
def test_replica_hybridep_full_layer_squared_relu_matches_alltoall(monkeypatch, mxfp8):
    """Check the squared-ReLU configuration used by main_cg_debug_int.sh."""
    try:
        from transformer_engine.pytorch.ops import ScaledSReLU  # noqa: F401
    except ImportError:
        pytest.skip("Transformer Engine ScaledSReLU is required")
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        squared_relu,
        False,
        True,
        None,
        640,
        mxfp8=mxfp8,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_bf16_semantics_match_hybridep(monkeypatch):
    """Require bitwise semantics and tightly bounded expert wgrad reduction noise."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        reference_dispatcher="hybridep",
        verify_hybridep_contract=True,
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_HYBRIDEP,
    reason="CUDA and HybridEP are required",
)
def test_replica_hybridep_mxfp8_matches_hybridep(monkeypatch):
    """Bound MX execution noise while requiring byte-exact weight transport."""
    _run_replica_hybridep_full_layer_parity(
        monkeypatch,
        "replica_hybridep",
        F.silu,
        True,
        False,
        None,
        None,
        mxfp8=True,
        reference_dispatcher="hybridep",
    )
