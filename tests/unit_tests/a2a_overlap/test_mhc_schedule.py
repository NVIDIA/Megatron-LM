# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.pipeline_parallel.utils import NoopScheduleNode, get_comp_stream, set_streams
from megatron.core.tensor_parallel.random import (
    CheckpointManager,
    CheckpointWithoutOutput,
    initialize_rng_tracker,
)
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    build_gpt_model,
    build_input_data,
    deterministic_mode,
    get_test_config,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def _initialized_model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _make_valid_mhc_overlap_config(**overrides):
    """Build the smallest config satisfying the mHC overlap prerequisites."""
    kwargs = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        enable_hyper_connections=True,
        num_residual_streams=4,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
        overlap_moe_expert_parallel_comm=True,
        expert_model_parallel_size=2,
        num_moe_experts=4,
        moe_token_dispatcher_type="alltoall",
        bf16=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_mhc_post_on_compute_stream_defaults_to_false():
    config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)

    assert config.mhc_post_on_compute_stream is False


def test_mhc_post_on_compute_stream_does_not_require_recompute():
    config = _make_valid_mhc_overlap_config(
        mhc_post_on_compute_stream=True, recompute_granularity=None, recompute_modules=[]
    )

    assert config.mhc_post_on_compute_stream is True


def test_mhc_post_on_compute_stream_requires_hyper_connections():
    with pytest.raises(ValueError, match="enable_hyper_connections"):
        _make_valid_mhc_overlap_config(
            mhc_post_on_compute_stream=True,
            enable_hyper_connections=False,
            recompute_granularity=None,
            recompute_modules=[],
        )


def test_mhc_post_on_compute_stream_requires_ep_overlap():
    with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm"):
        _make_valid_mhc_overlap_config(
            mhc_post_on_compute_stream=True, overlap_moe_expert_parallel_comm=False
        )


@pytest.mark.parametrize(
    "cuda_graph_kwargs",
    ({"cuda_graph_impl": "local"}, {"enable_cuda_graph": True}, {"external_cuda_graph": True}),
)
def test_mhc_overlap_recompute_rejects_cuda_graphs(cuda_graph_kwargs):
    with pytest.raises(ValueError, match="eager-only"):
        _make_valid_mhc_overlap_config(**cuda_graph_kwargs)


class _RecordingNode:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    def forward(self, value=None):
        self.calls.append(f"{self.name}.forward")
        return value

    def backward(self, value):
        self.calls.append(f"{self.name}.backward")
        return value

    def backward_dw(self):
        self.calls.append(f"{self.name}.backward_dw")


class _RecordingLayer:
    def __init__(self, calls, prefix, separate_mhc_post=False):
        self.calls = calls
        self.config = SimpleNamespace(ep_overlap_early_attn_memory_release=False)
        self.attn = _RecordingNode(calls, f"{prefix}.attn")
        self.moe_dispatch = _RecordingNode(calls, f"{prefix}.moe_dispatch")
        self.mlp = _RecordingNode(calls, f"{prefix}.mlp")
        self.moe_combine = _RecordingNode(calls, f"{prefix}.moe_combine")
        self.mhc_post = (
            _RecordingNode(calls, f"{prefix}.mhc_post") if separate_mhc_post else NoopScheduleNode()
        )
        self.mhc_recompute = None
        self.mtp_post_process = _RecordingNode(calls, f"{prefix}.mtp_post_process")

    def get_fp8_context(self):
        return nullcontext()

    def release_state(self):
        self.calls.append(f"{self.attn.name.split('.')[0]}.release_state")


class _RecordingChunk:
    def __init__(self, calls, layers):
        self.calls = calls
        self.layers = layers
        self.pre_process = _RecordingNode(calls, "chunk.pre_process")
        self.post_process = None
        self.vp_stage = 0

    def record_current_stream(self):
        self.calls.append("chunk.record_current_stream")

    def wait_current_stream(self):
        self.calls.append("chunk.wait_current_stream")

    def num_layers(self):
        return len(self.layers)

    def pop_layer(self):
        return self.layers.pop()

    def release_state(self):
        self.calls.append("chunk.release_state")


def _assert_called_before(calls, first, second):
    assert calls.count(first) == 1
    assert calls.count(second) == 1
    assert calls.index(first) < calls.index(second), f"Expected {first} before {second}: {calls}"


@pytest.mark.parametrize("separate_mhc_post", (False, True), ids=("merged", "compute-stream"))
@pytest.mark.parametrize("explicit_recompute", (False, True), ids=("eager", "recompute"))
def test_layer_schedule_orders_mhc_post_and_recompute(separate_mhc_post, explicit_recompute):
    calls = []
    forward_layer = _RecordingLayer(calls, "forward", separate_mhc_post)
    backward_layer = _RecordingLayer(calls, "backward", separate_mhc_post)
    if explicit_recompute:
        backward_layer.mhc_recompute = _RecordingNode(calls, "backward.mhc_recompute")

    TransformerLayerSchedulePlan.run(
        forward_layer, backward_layer, f_input=object(), b_grad=object(), is_last_layer_in_bwd=True
    )

    if separate_mhc_post:
        _assert_called_before(calls, "forward.moe_combine.forward", "forward.mhc_post.forward")
        _assert_called_before(calls, "backward.mhc_post.backward", "backward.moe_combine.backward")
    else:
        assert not any(".mhc_post." in call for call in calls)

    if explicit_recompute:
        next_backward = (
            "backward.mhc_post.backward" if separate_mhc_post else "backward.moe_combine.backward"
        )
        _assert_called_before(calls, "backward.mhc_recompute.forward", next_backward)
    else:
        assert "backward.mhc_recompute.forward" not in calls


@pytest.mark.parametrize("mhc_post_on_compute_stream", (False, True), ids=("merged", "separate"))
def test_mhc_post_node_uses_configured_stream(monkeypatch, mhc_post_on_compute_stream):
    import megatron.core.models.gpt.fine_grained_callables as fine_grained_callables
    import megatron.core.transformer.moe.moe_layer as moe_layer
    import megatron.core.transformer.transformer_layer as transformer_layer

    calls = []

    class _Value:
        requires_grad = False

        def __init__(self, name):
            self.name = name

        def record_stream(self, stream):
            pass

    combined_value = _Value("combined")
    post_value = _Value("post")

    class _FakeMoE:
        num_local_experts = 1

        def combine(self, output):
            calls.append("combine")
            return output

        def postprocess(self, output, shared_expert_output):
            calls.append("postprocess")
            return combined_value

    class _FakeHyperConnectionLayer:
        def __init__(self):
            self.config = SimpleNamespace(
                delay_wgrad_compute=False,
                mhc_post_on_compute_stream=mhc_post_on_compute_stream,
                moe_flex_dispatcher_backend=None,
                moe_token_dispatcher_type="alltoall",
            )
            self.layer_number = 1
            self.mlp = _FakeMoE()
            self.backward_dw_wrapper = object()

        def _forward_mlp(self, hidden_states):
            return hidden_states

        def _forward_mhc_mlp_post(self, output, mlp_h_res, residual, mlp_hc_h_post, manager):
            calls.append("mhc_post")
            return post_value

        def init_backward_dw_wrapper(self):
            pass

    class _CapturedNode:
        def __init__(
            self,
            stream,
            event,
            layer_state,
            chunk_state,
            submodule,
            name="default",
            bwd_dw_callables=None,
            extra_args=None,
        ):
            extra_args = extra_args or {}
            self.stream = stream
            self.layer_state = layer_state
            self.chunk_state = chunk_state
            self.submodule = submodule
            self.name = name
            self.is_mtp = extra_args.get("is_mtp", False)
            self.is_last_layer = extra_args.get("is_last_layer", False)
            self.mhc_recompute_manager = extra_args.get("mhc_recompute_manager")
            self.is_last_layer_in_mhc_recompute_group = extra_args.get(
                "is_last_layer_in_mhc_recompute_group", False
            )

        def set_post_forward_hook(self, hook):
            self.post_forward_hook = hook

        def set_post_backward_hook(self, hook):
            self.post_backward_hook = hook

    monkeypatch.setattr(fine_grained_callables, "MoELayer", _FakeMoE)
    monkeypatch.setattr(
        fine_grained_callables, "HyperConnectionTransformerLayer", _FakeHyperConnectionLayer
    )
    monkeypatch.setattr(fine_grained_callables, "TransformerLayer", _FakeHyperConnectionLayer)
    monkeypatch.setattr(fine_grained_callables, "TransformerLayerNode", _CapturedNode)
    monkeypatch.setattr(
        fine_grained_callables, "make_viewless_tensor", lambda inp, requires_grad, keep_graph: inp
    )
    monkeypatch.setattr(fine_grained_callables.torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(moe_layer, "MoELayer", _FakeMoE)
    monkeypatch.setattr(
        transformer_layer, "HyperConnectionTransformerLayer", _FakeHyperConnectionLayer
    )
    monkeypatch.setattr(transformer_layer, "TransformerLayer", _FakeHyperConnectionLayer)

    layer = _FakeHyperConnectionLayer()
    plan = TransformerLayerSchedulePlan.__new__(TransformerLayerSchedulePlan)
    plan.config = layer.config
    plan.layer = layer
    plan.layer_state = SimpleNamespace(
        residual=_Value("residual"),
        mlp_h_res=_Value("mlp_h_res"),
        mlp_hc_h_post=_Value("mlp_hc_h_post"),
    )
    plan.chunk_state = SimpleNamespace()

    def comp_stream():
        return "compute"

    def comm_stream():
        return "communication"

    recompute_manager = CheckpointManager()
    plan._build_callable_nodes(
        event=object(),
        comp_stream=comp_stream,
        comm_stream=comm_stream,
        extra_args={
            "mhc_recompute_manager": recompute_manager,
            "is_last_layer_in_mhc_recompute_group": True,
            "mhc_recompute_group_index": 0,
        },
    )

    assert plan.moe_combine.stream is comm_stream
    assert plan.mhc_recompute.stream is comp_stream
    combine_output = plan.moe_combine.submodule(plan.moe_combine, _Value("experts"))
    if mhc_post_on_compute_stream:
        assert isinstance(plan.mhc_post, _CapturedNode)
        assert plan.mhc_post.stream is comp_stream
        assert combine_output is combined_value
        assert calls == ["combine", "postprocess"]
        assert plan.mhc_post.submodule(plan.mhc_post, combine_output) is post_value
        assert calls == ["combine", "postprocess", "mhc_post"]
    else:
        assert isinstance(plan.mhc_post, NoopScheduleNode)
        assert combine_output is post_value
        assert calls == ["combine", "postprocess", "mhc_post"]

    plan.set_fsdp_reshard_hooks(lambda _: None, lambda _: None)
    expected_last_node = plan.mhc_post if mhc_post_on_compute_stream else plan.moe_combine
    previous_node = plan.moe_combine if mhc_post_on_compute_stream else plan.mhc_post
    assert hasattr(expected_last_node, "post_forward_hook")
    assert not hasattr(previous_node, "post_forward_hook")


def test_mtp_preserves_optional_mhc_post_callable(monkeypatch):
    import megatron.core.models.gpt.fine_grained_callables as fine_grained_callables

    class _FakeMoE:
        pass

    marker = object()

    def mhc_post_forward(node, hidden_states):
        return marker

    transformer_callables = [object(), object(), object(), object(), mhc_post_forward, None]
    backward_dw = {"attn": object(), "mlp": object()}
    monkeypatch.setattr(fine_grained_callables, "MoELayer", _FakeMoE)
    monkeypatch.setattr(
        fine_grained_callables,
        "build_transformer_layer_callables",
        lambda _: (transformer_callables, backward_dw),
    )
    layer = SimpleNamespace(
        mtp_model_layer=SimpleNamespace(mlp=_FakeMoE()),
        config=SimpleNamespace(sequence_parallel=False),
        eh_proj=object(),
    )

    forward_callables, _ = fine_grained_callables.build_mtp_layer_callables(layer)

    assert forward_callables[4](object(), object()) is marker


def test_model_chunk_recompute_groups_trigger_in_reverse_order():
    calls = []
    layers = [_RecordingLayer(calls, f"layer_{index}") for index in range(5)]
    # Forward groups are [0, 1], [2, 3], [4], so their explicit replay nodes
    # live on the final forward layer of each group.
    for group_end in (1, 3, 4):
        layers[group_end].mhc_recompute = _RecordingNode(calls, f"layer_{group_end}.mhc_recompute")
    chunk = _RecordingChunk(calls, layers)

    TransformerModelChunkSchedulePlan.run(None, chunk, b_grad=object())

    recompute_calls = [call for call in calls if ".mhc_recompute.forward" in call]
    assert recompute_calls == [
        "layer_4.mhc_recompute.forward",
        "layer_3.mhc_recompute.forward",
        "layer_1.mhc_recompute.forward",
    ]
    for group_end in (4, 3, 1):
        _assert_called_before(
            calls,
            f"layer_{group_end}.mhc_recompute.forward",
            f"layer_{group_end}.moe_combine.backward",
        )


def test_model_chunk_builds_independent_two_layer_recompute_groups(monkeypatch):
    captured_extra_args = []

    class _CapturedLayerPlan:
        def __init__(self, layer, event, state, comp_stream, comm_stream, extra_args):
            captured_extra_args.append(dict(extra_args))

    monkeypatch.setattr(
        "megatron.core.models.common.model_chunk_schedule_plan.TransformerLayerSchedulePlan",
        _CapturedLayerPlan,
    )
    plan = TransformerModelChunkSchedulePlan.__new__(TransformerModelChunkSchedulePlan)
    plan._event = object()
    plan._model_chunk_state = object()
    plan._transformer_layers = []
    config = SimpleNamespace(
        enable_hyper_connections=True,
        recompute_granularity="selective",
        recompute_modules=["mhc"],
        mhc_recompute_layer_num=2,
    )
    module = SimpleNamespace(config=config, layers=[object() for _ in range(5)], training=True)

    plan._build_layer_schedule_plan(module, get_comp_stream, lambda: None)

    managers = [extra_args["mhc_recompute_manager"] for extra_args in captured_extra_args]
    assert managers[0] is managers[1]
    assert managers[2] is managers[3]
    assert managers[0] is not managers[2]
    assert managers[4] is not managers[2]
    assert [
        extra_args["is_last_layer_in_mhc_recompute_group"] for extra_args in captured_extra_args
    ] == [False, True, False, True, True]
    assert [extra_args["mhc_recompute_group_index"] for extra_args in captured_extra_args] == [
        0,
        0,
        1,
        1,
        2,
    ]


def test_checkpoint_manager_explicit_recompute_is_idempotent_and_restores_gradients(
    _initialized_model_parallel,
):
    def run_function(value):
        return torch.sin(value) * value

    initialize_rng_tracker(force_reset=True)
    input_tensor = torch.randn(32, device="cuda", requires_grad=True)
    reference_input = input_tensor.detach().clone().requires_grad_(True)

    reference_output = run_function(reference_input)
    reference_loss = reference_output.square().sum()
    reference_loss.backward()

    manager = CheckpointManager()
    checkpoint = CheckpointWithoutOutput(ckpt_manager=manager)
    output = checkpoint.checkpoint(run_function, input_tensor)
    expected_output = output.detach().clone()
    loss = output.square().sum()

    manager.discard_all_outputs()
    assert output.untyped_storage().nbytes() == 0

    manager.recompute_now()
    torch.testing.assert_close(output, expected_output)

    # A second explicit trigger must be an observable no-op.
    manager.recompute_now()
    torch.testing.assert_close(output, expected_output)

    loss.backward()
    torch.testing.assert_close(input_tensor.grad, reference_input.grad)


def _run_schedule_and_capture(model, data):
    schedule_plan = model.build_schedule_plan(**data)
    output = TransformerModelChunkSchedulePlan.run(schedule_plan, None)
    output_value = output.detach().clone()
    TransformerModelChunkSchedulePlan.run(None, schedule_plan, b_grad=torch.ones_like(output))
    torch.cuda.synchronize()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output_value, gradients


def _run_eager_and_capture(model, data):
    output = float16_to_fp32(model.forward(**data))
    output_value = output.detach().clone()
    output.backward(torch.ones_like(output))
    torch.cuda.synchronize()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output_value, gradients


def _run_interleaved_schedule_and_capture(model, batches):
    first_plan = model.build_schedule_plan(**batches[0])
    first_output = TransformerModelChunkSchedulePlan.run(first_plan, None)
    first_output_value = first_output.detach().clone()
    second_plan = model.build_schedule_plan(**batches[1])
    second_output = TransformerModelChunkSchedulePlan.run(
        second_plan, first_plan, b_grad=torch.ones_like(first_output)
    )
    second_output_value = second_output.detach().clone()
    TransformerModelChunkSchedulePlan.run(None, second_plan, b_grad=torch.ones_like(second_output))
    torch.cuda.synchronize()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return [first_output_value, second_output_value], gradients


def _run_eager_batches_and_capture(model, batches):
    outputs = [float16_to_fp32(model.forward(**data)) for data in batches]
    output_values = [output.detach().clone() for output in outputs]
    for output in outputs:
        output.backward(torch.ones_like(output))
    torch.cuda.synchronize()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output_values, gradients


def _make_mhc_numerical_config(overlap=True, recompute=True, mhc_post_on_compute_stream=False):
    recompute_kwargs = (
        {
            "recompute_granularity": "selective",
            "recompute_modules": ["mhc"],
            "mhc_recompute_layer_num": 2,
        }
        if recompute
        else {"recompute_granularity": None, "recompute_modules": []}
    )
    return get_test_config(
        num_layers=2,
        extra_kwargs={
            "moe_token_dispatcher_type": "alltoall",
            "overlap_moe_expert_parallel_comm": overlap,
            "enable_hyper_connections": True,
            "mhc_sinkhorn_iterations": 5,
            "mhc_post_on_compute_stream": mhc_post_on_compute_stream,
            **recompute_kwargs,
        },
    )


class TestMhcA2AOverlapNumerics:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize(
        "mhc_post_on_compute_stream", (False, True), ids=("merged", "compute-stream")
    )
    @pytest.mark.parametrize("recompute", (False, True), ids=("without-recompute", "recompute"))
    def test_two_layer_alltoall_schedule_matches_eager_hook_path(
        self, recompute, mhc_post_on_compute_stream
    ):
        reference_config = _make_mhc_numerical_config(overlap=False, recompute=recompute)
        overlap_config = _make_mhc_numerical_config(
            recompute=recompute, mhc_post_on_compute_stream=mhc_post_on_compute_stream
        )
        with deterministic_mode():
            data = build_input_data(seq_len=16)
            reference_model = build_gpt_model(reference_config)
            initial_parameters = reset_model(reference_model)
            reference_output, reference_gradients = _run_eager_and_capture(reference_model, data)
            del reference_model

            overlap_model = build_gpt_model(overlap_config)
            reset_model(overlap_model, initial_parameters)
            overlap_output, overlap_gradients = _run_schedule_and_capture(overlap_model, data)

        torch.testing.assert_close(overlap_output, reference_output, rtol=5e-3, atol=5e-3)
        assert overlap_gradients.keys() == reference_gradients.keys()
        for name in reference_gradients:
            torch.testing.assert_close(
                overlap_gradients[name],
                reference_gradients[name],
                rtol=5e-3,
                atol=5e-3,
                msg=f"Gradient mismatch for {name}",
            )

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize(
        "mhc_post_on_compute_stream", (False, True), ids=("merged", "compute-stream")
    )
    def test_two_inflight_plans_keep_recompute_groups_independent(self, mhc_post_on_compute_stream):
        reference_config = _make_mhc_numerical_config(overlap=False)
        overlap_config = _make_mhc_numerical_config(
            mhc_post_on_compute_stream=mhc_post_on_compute_stream
        )
        with deterministic_mode():
            batches = [build_input_data(seq_len=16) for _ in range(2)]
            reference_model = build_gpt_model(reference_config)
            initial_parameters = reset_model(reference_model)
            reference_outputs, reference_gradients = _run_eager_batches_and_capture(
                reference_model, batches
            )
            del reference_model

            overlap_model = build_gpt_model(overlap_config)
            reset_model(overlap_model, initial_parameters)
            overlap_outputs, overlap_gradients = _run_interleaved_schedule_and_capture(
                overlap_model, batches
            )

        for overlap_output, reference_output in zip(overlap_outputs, reference_outputs):
            torch.testing.assert_close(overlap_output, reference_output, rtol=5e-3, atol=5e-3)
        assert overlap_gradients.keys() == reference_gradients.keys()
        for name in reference_gradients:
            torch.testing.assert_close(
                overlap_gradients[name],
                reference_gradients[name],
                rtol=5e-3,
                atol=5e-3,
                msg=f"Gradient mismatch for {name}",
            )
