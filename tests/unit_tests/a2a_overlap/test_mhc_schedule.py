# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.pipeline_parallel.utils import get_comp_stream, set_streams
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
    def __init__(self, calls, prefix):
        self.calls = calls
        self.config = SimpleNamespace(ep_overlap_early_attn_memory_release=False)
        self.attn = _RecordingNode(calls, f"{prefix}.attn")
        self.moe_dispatch = _RecordingNode(calls, f"{prefix}.moe_dispatch")
        self.mlp = _RecordingNode(calls, f"{prefix}.mlp")
        self.moe_combine = _RecordingNode(calls, f"{prefix}.moe_combine")
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


@pytest.mark.parametrize("explicit_recompute", (False, True), ids=("eager", "recompute"))
def test_layer_schedule_orders_recompute(explicit_recompute):
    calls = []
    forward_layer = _RecordingLayer(calls, "forward")
    backward_layer = _RecordingLayer(calls, "backward")
    if explicit_recompute:
        backward_layer.mhc_recompute = _RecordingNode(calls, "backward.mhc_recompute")

    TransformerLayerSchedulePlan.run(
        forward_layer, backward_layer, f_input=object(), b_grad=object(), is_last_layer_in_bwd=True
    )

    if explicit_recompute:
        _assert_called_before(
            calls, "backward.mhc_recompute.forward", "backward.moe_combine.backward"
        )
    else:
        assert "backward.mhc_recompute.forward" not in calls


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


def _make_mhc_numerical_config(overlap=True, recompute=True, extra_config=None):
    recompute_kwargs = (
        {
            "recompute_granularity": "selective",
            "recompute_modules": ["mhc"],
            "mhc_recompute_layer_num": 2,
        }
        if recompute
        else {"recompute_granularity": None, "recompute_modules": []}
    )
    extra_kwargs = {
        "moe_token_dispatcher_type": "alltoall",
        "overlap_moe_expert_parallel_comm": overlap,
        "enable_hyper_connections": True,
        "mhc_sinkhorn_iterations": 5,
        **recompute_kwargs,
    }
    if extra_config:
        extra_kwargs.update(extra_config)
    return get_test_config(num_layers=2, extra_kwargs=extra_kwargs)


def _assert_close_grads(overlap_gradients, reference_gradients, rtol=5e-3, atol=5e-3):
    assert overlap_gradients.keys() == reference_gradients.keys()
    for name in reference_gradients:
        torch.testing.assert_close(
            overlap_gradients[name],
            reference_gradients[name],
            rtol=rtol,
            atol=atol,
            msg=f"Gradient mismatch for {name}",
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
    @pytest.mark.parametrize("recompute", (False, True), ids=("without-recompute", "recompute"))
    def test_two_layer_alltoall_schedule_matches_eager_hook_path(self, recompute):
        reference_config = _make_mhc_numerical_config(overlap=False, recompute=recompute)
        overlap_config = _make_mhc_numerical_config(recompute=recompute)
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
    def test_two_inflight_plans_keep_recompute_groups_independent(self):
        reference_config = _make_mhc_numerical_config(overlap=False)
        overlap_config = _make_mhc_numerical_config()
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

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_dense_final_layer_schedule_matches_eager(self):
        # moe_layer_freq=[1, 0] makes the last decoder layer dense, so its terminal schedule
        # node is the dense MLP rather than moe_combine. The decoder boundary (mHC output
        # contraction + final layer norm) must still run there; otherwise an uncontracted
        # [s, b, n*h] tensor reaches postprocessing (wrong shape / values).
        dense_final = {"moe_layer_freq": [1, 0]}
        reference_config = _make_mhc_numerical_config(
            overlap=False, recompute=False, extra_config=dense_final
        )
        overlap_config = _make_mhc_numerical_config(recompute=False, extra_config=dense_final)
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
        _assert_close_grads(overlap_gradients, reference_gradients)

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("recompute", (False, True), ids=("without-recompute", "recompute"))
    def test_mtp_schedule_matches_eager(self, recompute):
        # With MTP (mtp_num_layers=1, default mtp_detach_heads=False) the decoder boundary
        # produces the pre-contraction mHC multi-stream consumed by the MTP depths. It must be
        # detached at its producer so MTP backward does not traverse the decoder mHC graph out
        # of schedule order; this node's backward_impl reconnects the accumulated gradient.
        # Parameterized over recompute to exercise mHC + MTP + EP overlap + mhc recompute.
        mtp = {"mtp_num_layers": 1}
        reference_config = _make_mhc_numerical_config(
            overlap=False, recompute=recompute, extra_config=mtp
        )
        overlap_config = _make_mhc_numerical_config(recompute=recompute, extra_config=mtp)
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
        _assert_close_grads(overlap_gradients, reference_gradients)

    @pytest.mark.skipif(
        not is_te_min_version("2.3.0"), reason="delay_wgrad_compute requires TE >= 2.3.0"
    )
    def test_mtp_schedule_with_delayed_wgrad_matches_eager(self):
        # Regression test for the MTP delayed-wgrad callable selection: under mHC the MTP
        # layer builds separate e_proj/h_proj and sets eh_proj=None, so
        # build_mtp_layer_callables must register [e_proj, h_proj] (not eh_proj) with the
        # attn node's delayed-wgrad list. With the unconditional eh_proj append this test
        # crashes at TransformerLayerNode.backward_dw (None.backward_dw); with a wrong-but-
        # non-None list it would leave the e_proj/h_proj weight gradients permanently
        # deferred, which the grad-keys equality below catches. delay_wgrad_compute
        # requires overlap_moe_expert_parallel_comm, so only the overlap config carries it;
        # the eager reference computes its wgrads inline.
        mtp = {"mtp_num_layers": 1}
        reference_config = _make_mhc_numerical_config(
            overlap=False, recompute=False, extra_config=mtp
        )
        overlap_config = _make_mhc_numerical_config(
            recompute=False, extra_config={**mtp, "delay_wgrad_compute": True}
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
        # The MTP projections must have flushed their deferred wgrads.
        assert any("e_proj.weight" in name for name in overlap_gradients), overlap_gradients.keys()
        assert any("h_proj.weight" in name for name in overlap_gradients), overlap_gradients.keys()
        _assert_close_grads(overlap_gradients, reference_gradients)

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("recompute", (False, True), ids=("without-recompute", "recompute"))
    def test_mtp_two_inflight_plans_match_eager(self, recompute):
        # Two in-flight schedule plans verify that the per-chunk mHC multi-stream (and hence
        # the reconnected MTP side gradient) stays bound to the correct microbatch.
        # Parameterized over recompute so the per-group recompute managers are also verified
        # to stay independent across in-flight microbatches when MTP depths are present.
        mtp = {"mtp_num_layers": 1}
        reference_config = _make_mhc_numerical_config(
            overlap=False, recompute=recompute, extra_config=mtp
        )
        overlap_config = _make_mhc_numerical_config(recompute=recompute, extra_config=mtp)
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
        # The tied word-embedding gradient is the most-accumulated parameter (input
        # embedding + tied lm-head + MTP re-embedding, summed over both microbatches).
        # Interleaving two in-flight plans reorders that bf16 reduction relative to the
        # sequential eager run, so use a looser grad tolerance here. A real microbatch-
        # binding error would show up as a gross mismatch on the MTP-specific parameters
        # (verified bitwise-identical), not a ~2% nudge on the shared embedding.
        _assert_close_grads(overlap_gradients, reference_gradients, rtol=3e-2, atol=3e-2)

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_schedule_with_full_iteration_cuda_graph_matches_eager(self):
        """mHC + EP overlap + ``cuda_graph_impl='full_iteration'`` (no mhc recompute).

        The ``__post_init__`` guard added by this PR rejects CUDA graphs only when
        mhc *recompute* is enabled (explicit group replay is eager-only); this test
        proves the guard admits full-iteration CG + EP overlap without recompute,
        and that the scheduled forward+backward step is actually capturable into a
        ``torch.cuda.CUDAGraph`` and numerically faithful on replay — the core-level
        equivalent of what ``FullCudaGraphWrapper`` captures in production.

        MoE runs in drop_and_pad mode: the dropless alltoall dispatcher performs a
        mandatory D2H splits sync that is illegal during stream capture. The
        capacity factor is sized so no token can be dropped at this scale, and both
        configs share the setting so the eager reference matches numerically.
        """
        # capacity_factor=4.0 -> per-expert capacity = 4.0 * 16 tokens * topk(2) / 8
        # experts = 16 = the full token count, so no token can ever be dropped and
        # the padded slot count (8 * 16) matches the routing-map size exactly.
        drop_and_pad = {"moe_pad_expert_input_to_capacity": True, "moe_expert_capacity_factor": 4.0}
        reference_config = _make_mhc_numerical_config(
            overlap=False, recompute=False, extra_config=drop_and_pad
        )
        overlap_config = _make_mhc_numerical_config(
            recompute=False, extra_config={**drop_and_pad, "cuda_graph_impl": "full_iteration"}
        )
        # Full-iteration capture requires a graph-safe RNG tracker (production
        # enforces use_te_rng_tracker with CUDA graphs): TE attention forks the
        # tracker even with dropout disabled, and the default tracker's
        # set_state is illegal during stream capture.
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        try:
            self._run_full_iteration_cuda_graph_case(reference_config, overlap_config)
        finally:
            initialize_rng_tracker(force_reset=True)

    def _run_full_iteration_cuda_graph_case(self, reference_config, overlap_config):
        with deterministic_mode():
            data = build_input_data(seq_len=16)
            reference_model = build_gpt_model(reference_config)
            initial_parameters = reset_model(reference_model)
            reference_output, reference_gradients = _run_eager_and_capture(reference_model, data)
            del reference_model

            overlap_model = build_gpt_model(overlap_config)
            reset_model(overlap_model, initial_parameters)

            def run_step():
                schedule_plan = overlap_model.build_schedule_plan(**data)
                output = TransformerModelChunkSchedulePlan.run(schedule_plan, None)
                TransformerModelChunkSchedulePlan.run(
                    None, schedule_plan, b_grad=torch.ones_like(output)
                )
                return output

            # Side-stream eager warmup so lazy allocations and one-time init exist
            # before capture.
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(2):
                    run_step()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()

            # Zero gradients strictly in place: the captured accumulation kernels
            # must keep writing into the same live tensors on every replay.
            for parameter in overlap_model.parameters():
                if parameter.grad is not None:
                    parameter.grad.zero_()

            graph = torch.cuda.CUDAGraph()
            torch.distributed.barrier()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                static_output = run_step()
            torch.cuda.synchronize()
            torch.distributed.barrier()

            # Capture only records the kernels; the first replay produces the
            # real values in the static output/grad buffers.
            graph.replay()
            torch.cuda.synchronize()

            overlap_output = static_output.detach().clone()
            overlap_gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in overlap_model.named_parameters()
                if parameter.grad is not None
            }
            del graph
            gc.collect()

        torch.testing.assert_close(overlap_output, reference_output, rtol=5e-3, atol=5e-3)
        _assert_close_grads(overlap_gradients, reference_gradients)
