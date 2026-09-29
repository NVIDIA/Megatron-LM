# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 graph boundaries, compact workspaces and native Hybrid autograd parity.

CPU graph callables execute the capture body eagerly on every invocation. These
tests verify the production capture/replay interface and state ownership, not
CUDA replay or Transformer Engine kernels. Conditional CUDA tests exercise real
Transformer Engine and compact-indexer replay.
"""

import inspect
from copy import deepcopy
from types import MethodType

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.transformer.cuda_graphs import _set_capture_end, _set_capture_start
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
    prepare_csa2_indexer_inputs,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCState
from megatron.core.transformer.mhc_recompute import MHCRecomputeArenaSlot, MHCRecomputeSlotMetadata
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    _kernel_indexer,
    _packed,
    _record_losses,
    _require_sparse_kernels,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _FFN,
    _config,
    _run_chunks,
    _stack,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    _assert_gradient,
    _assert_parameter_gradients,
    _recompute_config,
    _stacks,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    checkpoints as checkpoints,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_recompute import (
    cpu_checkpoint_rng as cpu_checkpoint_rng,
)


def _graph_config(config, layout):
    config = deepcopy(config)
    # Native CPU modules need no TE installation. Test the graph boundary after
    # constructing the normal model config, independently of dependency gates.
    config.cuda_graph_impl = "transformer_engine"
    if layout == "thd":
        config.sequence_packing_scheduler = "pack_by_seq"
        config.max_seqlen_per_dp_cp_rank = 14
    return config


class _JointGraphBackward(torch.autograd.Function):
    """Model TE's one autograd node and zero gradients for unused graph outputs.

    The inner graph is ordinary native PyTorch. Its backward deliberately visits
    every differentiable graph output, including outputs unused by the caller.
    """

    @staticmethod
    def forward(ctx, function, input_count, *tensors):
        inputs = tuple(
            value.detach().requires_grad_(value.requires_grad) for value in tensors[:input_count]
        )
        parameters = tensors[input_count:]
        with torch.enable_grad():
            outputs = tuple(function(*inputs))
        ctx.inputs = (*inputs, *parameters)
        ctx.outputs = outputs
        ctx.set_materialize_grads(False)
        return tuple(output.detach() for output in outputs)

    @staticmethod
    def backward(ctx, *output_gradients):
        outputs, gradients = [], []
        for output, gradient in zip(ctx.outputs, output_gradients):
            if output.requires_grad:
                outputs.append(output)
                gradients.append(torch.zeros_like(output) if gradient is None else gradient)
        targets = tuple(value for value in ctx.inputs if value.requires_grad)
        # TE builds the joint backward while MCore's capture flag is set.
        # CheckpointWithoutOutput supports autograd.grad in that context.
        _set_capture_start()
        try:
            result = torch.autograd.grad(outputs, targets, gradients, allow_unused=True)
        finally:
            _set_capture_end()
        result = iter(result)
        return None, None, *(next(result) if value.requires_grad else None for value in ctx.inputs)


def _install_eager_graph_callables(stacks, *, joint_backward=False, num_microbatches=2):
    calls = []
    for stack in stacks:
        for layer in stack.layers:
            adapter = layer._te_cuda_graph_adapter
            static = _static_inputs(layer)
            adapter.finalize_sample_inputs(
                (static["hidden_states"],),
                {name: tensor for name, tensor in static.items() if name != "hidden_states"},
            )
            split = getattr(layer, "_uses_mhc_recompute_cuda_graph_split", lambda: False)()
            hidden_slots = [
                static["hidden_states"].detach().clone() for _ in range(num_microbatches)
            ]
            # TransformerLayer's override imports TE to support old versions.
            # CPU fakes exercise the current tensor-kwargs contract directly.
            layer._get_te_cuda_graph_replay_args = MethodType(
                GraphableMegatronModule._get_te_cuda_graph_replay_args, layer
            )

            def make_callable(layer, adapter, microbatch):
                def graph(*args, **kwargs):
                    kwargs.pop("is_first_microbatch", None)
                    assert all(isinstance(value, torch.Tensor) for value in args)
                    assert all(
                        value is None or isinstance(value, torch.Tensor)
                        for value in kwargs.values()
                    ), "mutable CSA2/mHC objects must not reach the TE callable"
                    calls.append((layer.layer_number, microbatch))
                    if getattr(layer, "_uses_mhc_recompute_cuda_graph_split", lambda: False)():
                        slot = layer.get_te_cuda_graph_static_hidden_input(microbatch)
                        assert args[0].data_ptr() == slot.data_ptr(), "aggregate must direct-write"
                        assert args[0].shape[-1] == layer.config.hidden_size
                        assert "mhc_graph_pre_mix" not in kwargs
                    if joint_backward:
                        names = tuple(name for name, value in kwargs.items() if value is not None)
                        flat_inputs = (*args, *(kwargs[name] for name in names))

                        def capture(*inputs):
                            captured_kwargs = {**kwargs, **dict(zip(names, inputs[len(args) :]))}
                            return adapter.capture(
                                layer._te_cuda_graph_capture,
                                *inputs[: len(args)],
                                **captured_kwargs,
                            )

                        return _JointGraphBackward.apply(
                            capture, len(flat_inputs), *flat_inputs, *layer.parameters()
                        )
                    return adapter.capture(layer._te_cuda_graph_capture, *args, **kwargs)

                return graph

            layer.cuda_graphs = [make_callable(layer, adapter, i) for i in range(num_microbatches)]
            if split:
                layer.set_te_cuda_graph_static_hidden_inputs(hidden_slots)
    return calls


def _static_inputs(layer):
    """CPU-capable sample allocation, followed by the production state schema."""
    config = layer.config
    packed = config.sequence_packing_scheduler is not None
    seq, batch = (14, 1) if packed else (9, 2)
    split = getattr(layer, "_uses_mhc_recompute_cuda_graph_split", lambda: False)()
    width = config.hidden_size * (
        config.num_residual_streams if config.enable_hyper_connections and not split else 1
    )
    device = next(layer.parameters()).device
    static = {
        "hidden_states": torch.ones(
            seq, batch, width, dtype=config.params_dtype, device=device, requires_grad=True
        )
    }
    adapter = layer._te_cuda_graph_adapter
    feature = next((c for c in adapter.components if hasattr(c, "is_attention")), None)
    if packed and feature is not None and feature.is_attention:
        prefix = torch.tensor([0, seq, seq, seq, seq], dtype=torch.int32, device=device)
        for suffix in ("q", "kv", "q_padded", "kv_padded"):
            static["cu_seqlens_" + suffix] = prefix.clone()
    return layer._te_cuda_graph_adapter.get_static_inputs(static)


@pytest.fixture
def cpu_graph_slots(monkeypatch):
    """Relax only CUDA allocation gates; retain arena address/discard/replay logic."""
    # Native boundary tests intentionally remain on CPU even on a GPU host.
    original_init = MHCRecomputeArenaSlot.__init__
    original_set = GraphableMegatronModule.set_te_cuda_graph_static_hidden_inputs

    def initialize(slot, key, tensor):
        if tensor.is_cuda:
            return original_init(slot, key, tensor)
        assert tensor.is_contiguous()
        slot.key, slot.consumer = key, tensor
        slot.metadata = MHCRecomputeSlotMetadata(
            tensor.shape, tensor.dtype, tensor.device, tensor.layout, tensor.data_ptr()
        )

    def set_inputs(layer, inputs):
        inputs = tuple(inputs)
        if any(tensor.is_cuda for tensor in inputs):
            return original_set(layer, inputs)
        assert len(inputs) == len(layer.cuda_graphs)
        layer._te_cuda_graph_static_hidden_inputs = inputs
        layer._te_cuda_graph_static_hidden_input_ptrs = tuple(t.data_ptr() for t in inputs)

    monkeypatch.setattr(MHCRecomputeArenaSlot, "__init__", initialize)
    monkeypatch.setattr(
        GraphableMegatronModule, "set_te_cuda_graph_static_hidden_inputs", set_inputs
    )


def _packed_microbatch(index):
    logical, physical = ([1, 4, 0, 2], [2, 5, 0, 4]) if index == 0 else ([2, 1, 0, 3], [3, 3, 0, 5])
    params, _, valid = _packed(logical, physical, tail=3)
    params.max_seqlen_q = params.max_seqlen_kv = 14
    return params, valid


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("mhc", [False, True])
@pytest.mark.parametrize("coefficient", [None, 0.0, 0.3])
@pytest.mark.parametrize("cuts", [(), (1, 5, 7, 8, 10)])
def test_graph_boundary_preserves_shared_state_and_two_outstanding_microbatches(
    monkeypatch, layout, mhc, coefficient, cuts
):
    """SWA/Full/Reindex/Reuse and an FFN relay preserve all consumer gradients."""
    torch.manual_seed(922)
    records = _record_losses(monkeypatch)
    reference = _stack(_config(coefficient=coefficient, enable_hyper_connections=mhc))
    graph_config = _graph_config(reference.config, layout)
    stacks, plan = _stacks(reference, graph_config, cuts, layout)
    calls = _install_eager_graph_callables(stacks)
    pairs = []
    for microbatch in range(2):
        params, valid = _packed_microbatch(microbatch) if layout == "thd" else (None, None)
        shape = (14, 1) if layout == "thd" else (9, 2)
        x = torch.randn(*shape, graph_config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        expected = reference(ref_x, None, packed_seq_params=params)
        expected_losses = [record["loss"].clone() for record in records]
        records.clear()
        for stack in stacks:
            for layer in stack.layers:
                layer.current_microbatch = microbatch
        actual, payloads = _run_chunks(stacks, plan, x, params)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert len(records) == len(expected_losses)
        for actual_loss, expected_loss in zip(records, expected_losses):
            torch.testing.assert_close(actual_loss["loss"], expected_loss)
        records.clear()
        pairs.append((actual, expected, x, ref_x, payloads))
    assert len(calls) == 2 * graph_config.num_layers
    assert {microbatch for _, microbatch in calls} == {0, 1}
    for first, second in zip(pairs[0][-1], pairs[1][-1]):
        for old, new in zip(first.tensors, second.tensors):
            assert old is not new, "one microbatch cannot retain another's graph output"
    for actual, expected, x, ref_x, _ in reversed(pairs):
        probe = torch.randn_like(actual) * 0.1
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        _assert_gradient(x.grad, ref_x.grad)
    _assert_parameter_gradients(reference, stacks, plan)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize(
    "mhc,field",
    [
        (False, "hidden"),
        (False, "global_kv"),
        (False, "indexer_k"),
        (True, "hidden"),
        (True, "global_kv"),
        (True, "indexer_k"),
        (True, "pre_mix"),
    ],
)
def test_joint_graph_backward_does_not_inject_auxiliary_loss_for_unused_hidden(
    monkeypatch, layout, mhc, field
):
    """Zero-filling an unused hidden output must not activate its indexer autoscaler."""
    torch.manual_seed(219)
    _record_losses(monkeypatch)
    config = _config(coefficient=0.3, enable_hyper_connections=mhc)
    # The boundary exports values with a declared external reader. Make L4 a
    # Reindex consumer so L2's indexer K is a live side output for this objective.
    config.csa2_index_source_layers = [2, 4, 6, 8]
    reference = _stack(config)
    actual = _stack(_graph_config(config, layout))
    actual.load_state_dict(reference.state_dict())
    _install_eager_graph_callables([actual], joint_backward=True)
    params, _ = _packed_microbatch(0) if layout == "thd" else (None, None)
    shape = (14, 1) if layout == "thd" else (9, 2)
    width = config.hidden_size * (config.num_residual_streams if mhc else 1)
    hidden = torch.randn(*shape, width, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    state, ref_state = CSA2State(), CSA2State()
    mix = torch.rand(*shape, config.num_residual_streams, requires_grad=True) if mhc else None
    ref_mix = mix.detach().clone().requires_grad_() if mhc else None
    mhc_state, ref_mhc_state = (
        (SinglePassMHCState(mix), SinglePassMHCState(ref_mix)) if mhc else (None, None)
    )
    kwargs = {"packed_seq_params": params, "cross_layer_state": state}
    ref_kwargs = {"packed_seq_params": params, "cross_layer_state": ref_state}
    if mhc:
        kwargs["mhc_state"], ref_kwargs["mhc_state"] = mhc_state, ref_mhc_state
    # Attention index 2 is the first Full owner. Its graph emits the ordinary
    # hidden output and independently consumable KV/indexer/mHC side tensors.
    output = actual.layers[2](hidden, attention_mask=None, **kwargs)[0]
    expected = reference.layers[2](ref_hidden, attention_mask=None, **ref_kwargs)[0]
    if field == "pre_mix":
        target, ref_target = mhc_state.pre_mix, ref_mhc_state.pre_mix
    elif field == "hidden":
        target, ref_target = output, expected
    else:
        target, ref_target = getattr(state, field), getattr(ref_state, field)
    torch.testing.assert_close(target, ref_target, atol=0, rtol=0)
    target.square().sum().backward()
    ref_target.square().sum().backward()
    _assert_gradient(hidden.grad, ref_hidden.grad)
    if mhc:
        _assert_gradient(mix.grad, ref_mix.grad)
    for parameter, ref_parameter in zip(
        actual.layers[2].parameters(), reference.layers[2].parameters()
    ):
        _assert_gradient(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("group_size", [None, 1, 3])
@pytest.mark.parametrize("cuts", [(), (1, 5, 7, 8, 10)])
@pytest.mark.parametrize("coefficient", [0.0, 0.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mhc_recompute_graph_slots_preserve_two_outstanding_microbatches(
    monkeypatch, checkpoints, cpu_graph_slots, layout, group_size, cuts, coefficient, dtype
):
    """Graph consumers retain fixed aggregate addresses through discard and replay."""
    torch.manual_seed(704)
    records = _record_losses(monkeypatch)
    reference = _stack(_config(dtype=dtype, coefficient=coefficient))
    config = _graph_config(_recompute_config(reference.config, group_size), layout)
    stacks, plan = _stacks(reference, config, cuts, layout)
    calls = _install_eager_graph_callables(stacks, joint_backward=True)
    pairs = []
    for microbatch in range(2):
        params, _ = _packed_microbatch(microbatch) if layout == "thd" else (None, None)
        shape = (14, 1) if layout == "thd" else (9, 2)
        hidden = torch.randn(*shape, config.hidden_size, dtype=dtype, requires_grad=True)
        ref_hidden = hidden.detach().clone().requires_grad_()
        expected = reference(ref_hidden, None, packed_seq_params=params)
        expected_losses = [record["loss"].clone() for record in records]
        records.clear()
        for stack in stacks:
            for layer in stack.layers:
                layer.current_microbatch = microbatch
        actual, payloads = _run_chunks(stacks, plan, hidden, params)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert len(records) == len(expected_losses)
        for actual_loss, expected_loss in zip(records, expected_losses):
            torch.testing.assert_close(actual_loss["loss"], expected_loss)
        records.clear()
        assert checkpoints and all(manager._outputs_discarded for manager in checkpoints)
        for manager in checkpoints:
            for checkpoint in manager.checkpoints:
                for output in checkpoint.outputs:
                    if checkpoint.output_slot is None:
                        assert output.untyped_storage().nbytes() == 0
                    else:
                        checkpoint.output_slot.validate_output(output)
        for payload in payloads:
            assert all(t.numel() == 0 or t.untyped_storage().nbytes() for t in payload.tensors)
        pairs.append((actual, actual.detach().clone(), expected, hidden, ref_hidden))
    # Each outstanding graph's input has independent storage. Deliberately
    # corrupt it without a version bump; the manager must restore the bytes the
    # graph backward reads before using them for input/parameter gradients.
    for stack in stacks:
        for layer in stack.layers:
            slots = layer._te_cuda_graph_static_hidden_inputs
            assert len({slot.data_ptr() for slot in slots}) == 2
            assert layer._mhc_recompute_manager is None
            for slot in slots:
                slot.data.fill_(float("nan"))
    for actual, snapshot, expected, hidden, ref_hidden in reversed(pairs):
        torch.testing.assert_close(actual, snapshot, atol=0, rtol=0)
        probe = torch.randn_like(actual) * 0.1
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        _assert_gradient(hidden.grad, ref_hidden.grad, dtype=dtype)
    _assert_parameter_gradients(reference, stacks, plan, dtype=dtype)
    assert len(calls) == 2 * config.num_layers, "mHC replay must not rerun the inner graph forward"
    assert all(manager._recomputed for manager in checkpoints)
    assert all(
        checkpoint.ctx is None for manager in checkpoints for checkpoint in manager.checkpoints
    )


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize(
    "field,explicit_zero_hidden",
    [("pre_mix", False), ("global_kv", False), ("indexer_k", False), ("indexer_k", True)],
)
@pytest.mark.parametrize("graph_mode", ["disabled", "capture_body", "joint"])
def test_mhc_recompute_side_output_only_backward(
    monkeypatch,
    request,
    checkpoints,
    cpu_graph_slots,
    layout,
    field,
    explicit_zero_hidden,
    graph_mode,
):
    """Side-output backward preserves mHC checkpoints, including the eager fallback."""
    if graph_mode == "joint" and field == "indexer_k" and not explicit_zero_hidden:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    "Isolated indexer K objectives with aux loss across joint layer graphs "
                    "are unsupported: zero hidden gradients activate upstream aux loss. "
                    "Use cuda_graph_impl='none' for this objective."
                ),
            )
        )
    torch.manual_seed(971)
    _record_losses(monkeypatch)
    config = _config(coefficient=0.3 if field == "indexer_k" else 0.0)
    reference = _stack(config)
    expected_stacks, _ = _stacks(reference, config, (7,), layout)
    actual_config = _recompute_config(config, 3)
    if graph_mode != "disabled":
        actual_config = _graph_config(actual_config, layout)
    stacks, _ = _stacks(reference, actual_config, (7,), layout)
    if graph_mode == "disabled":
        assert actual_config.cuda_graph_impl == "none"
    else:
        _install_eager_graph_callables(stacks, joint_backward=graph_mode == "joint")
    params, _ = _packed_microbatch(0) if layout == "thd" else (None, None)
    shape = (14, 1) if layout == "thd" else (9, 2)
    hidden = torch.randn(*shape, config.hidden_size, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    actual = stacks[0](hidden, None, packed_seq_params=params)
    expected = expected_stacks[0](ref_hidden, None, packed_seq_params=params)
    index = tuple(spec.name for spec in actual.tensor_specs).index(field)
    target, ref_target = actual.tensors[index], expected.tensors[index]
    torch.testing.assert_close(target, ref_target, atol=0, rtol=0)
    loss, ref_loss = target.square().sum(), ref_target.square().sum()
    if explicit_zero_hidden:
        # An actual zero hidden gradient must still activate the ordinary aux
        # scaler. Only an absent hidden objective can suppress its graph edge.
        hidden_index = tuple(spec.name for spec in actual.tensor_specs).index("hidden_states")
        loss = loss + actual.tensors[hidden_index].sum() * 0
        ref_loss = ref_loss + expected.tensors[hidden_index].sum() * 0
    loss.backward()
    ref_loss.backward()
    _assert_gradient(hidden.grad, ref_hidden.grad)
    for (name, parameter), ref_parameter in zip(
        stacks[0].named_parameters(), expected_stacks[0].parameters()
    ):
        try:
            _assert_gradient(parameter.grad, ref_parameter.grad)
        except AssertionError as error:
            raise AssertionError(name) from error
    if field == "indexer_k":
        query_gradient = dict(stacks[0].named_parameters())[
            "layers.2.inner_layer.self_attention.core.indexer.linear_wq_b.weight"
        ].grad
        if explicit_zero_hidden:
            assert query_gradient is not None and torch.count_nonzero(query_gradient)
        else:
            assert query_gradient is None or not torch.count_nonzero(query_gradient)
    assert any(manager._recomputed for manager in checkpoints)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("mhc_recompute", [False, True])
def test_graph_state_is_restored_before_partial_moe_continuation(
    monkeypatch, checkpoints, cpu_graph_slots, layout, mhc_recompute
):
    """Mock router/experts while retaining the production mHC prefix and eager BDA."""
    original_capture = TransformerLayer._te_cuda_graph_capture_impl
    continuations = []

    def capture_prefix(layer, hidden_states, **kwargs):
        if isinstance(layer.mlp, _FFN):
            return (layer.pre_mlp_layernorm(hidden_states),)
        return original_capture(layer, hidden_states, **kwargs)

    def resume_experts(layer, outputs):
        assert len(outputs) == 1, "CSA2/mHC side outputs must be removed before expert replay"
        continuations.append(layer.layer_number)
        return layer.mlp(outputs[0])

    monkeypatch.setattr(
        HyperConnectionHybridLayer,
        "_inner_is_partial_moe_capture",
        lambda layer: isinstance(layer.inner_layer.mlp, _FFN),
    )
    monkeypatch.setattr(TransformerLayer, "_te_cuda_graph_capture_impl", capture_prefix)
    monkeypatch.setattr(
        TransformerLayer, "resume_moe_experts_after_partial_cudagraph", resume_experts
    )
    if mhc_recompute:
        test_mhc_recompute_graph_slots_preserve_two_outstanding_microbatches(
            monkeypatch,
            checkpoints,
            cpu_graph_slots,
            layout,
            3,
            (1, 5, 7, 8, 10),
            0.3,
            torch.float32,
        )
    else:
        test_graph_boundary_preserves_shared_state_and_two_outstanding_microbatches(
            monkeypatch, layout, True, 0.3, (1, 5, 7, 8, 10)
        )
    assert len(continuations) == 12


class _NativeCaptureBody(torch.nn.Module):
    """Give TE ownership of the real layer parameters and tensor capture body."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        kwargs.setdefault("attention_mask", None)
        return self.layer._te_cuda_graph_adapter.capture(
            self.layer._te_cuda_graph_capture, *args, **kwargs
        )


def _inputs(layout, ratio, device, *, rows=65):
    params = None
    heads, dim = 32, 128
    if layout == "sbhd":
        q = torch.randn(rows, 2, heads, dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(rows // ratio, 2, dim, dtype=torch.bfloat16, device=device)
        weights = torch.randn(rows, 2, heads, dtype=torch.bfloat16, device=device)
    else:
        params, _, valid = _packed([17, 31], [19, 34], tail=12, device=device)
        token_layout = build_csa2_thd_layout(params, valid.numel())
        capacity = token_layout.for_compression(ratio).capacity
        q = torch.randn(valid.numel(), heads, dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(capacity, dim, dtype=torch.bfloat16, device=device)
        weights = torch.randn(valid.numel(), heads, dtype=torch.bfloat16, device=device)
    return q, k, weights, params


def _select(indexer, q, k, weights, params):
    layout = build_csa2_thd_layout(params, q.shape[0]) if params is not None else None
    compressed = layout.for_compression(indexer.compress_ratio) if layout is not None else None
    inputs = prepare_csa2_indexer_inputs(
        q, k, weights, indexer.compress_ratio, thd_layout=layout, compressed_layout=compressed
    )
    return indexer._fused_topk(inputs)


@pytest.mark.parametrize("layout", ["sbhd", "thd"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_compact_selection_cuda_graph_replays_changed_inputs(layout, precision, ratio):
    """Real compact Top-K replays values and packed boundaries at fixed capacity."""
    _require_sparse_kernels()
    frontend = pytest.importorskip("cudnn")
    wrapper = getattr(frontend.DSA, "indexer_forward_top_k_wrapper", None)
    parameters = set(inspect.signature(wrapper).parameters) if callable(wrapper) else set()
    required = {"deterministic"}
    if precision == "mxfp8":
        required.add("q_scale")
    helper = "compress_topk_cand_buffer_size" + ("_thd" if layout == "thd" else "")
    if required - parameters or not hasattr(frontend.DSA, helper):
        pytest.skip("installed cuDNN Frontend lacks the required compact indexer support")
    indexer = _kernel_indexer(ratio, precision, 32, "cuda")
    indexer.config.cuda_graph_impl = "local"
    tensors = _inputs(layout, ratio, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _select(indexer, *tensors)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = _select(indexer, *tensors)

    for iteration in range(2):
        q, k, weights, params = tensors
        q.normal_()
        k.normal_()
        weights.normal_()
        if params is not None and iteration:
            replacement, _, _ = _packed([13, 29], [23, 30], tail=12, device="cuda")
            for name in (
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_q_padded",
                "cu_seqlens_kv_padded",
            ):
                getattr(params, name).copy_(getattr(replacement, name))
        output.fill_(-123)
        graph.replay()
        expected = _select(indexer, *tensors)
        assert torch.equal(output, expected)
        assert (output != -123).all()
