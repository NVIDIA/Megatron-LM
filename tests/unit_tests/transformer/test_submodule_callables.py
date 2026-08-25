# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.common import fine_grained_callables as common_callables
from megatron.core.models.common import model_chunk_schedule_plan
from megatron.core.models.common.fine_grained_callables import build_layer_callables
from megatron.core.models.gpt import fine_grained_callables as gpt_callables
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    DummyNode,
    DummyState,
    build_data,
    compare_captures,
    deterministic_mode,
    get_test_config,
    get_valid_flex_dispatcher_backend,
    get_valid_token_dispatcher_types,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


def test_repeated_mtp_te_graph_owns_backward_dw_wrapper_per_logical_depth():
    """Each logical use of a repeated MTP layer updates its own wgrad wrapper."""

    class FakeBackwardDWWrapper:
        def __init__(self):
            self.graphed_backward_dw_callable = None

        def set_graphed_backward_dw_callable(self, callable_):
            self.graphed_backward_dw_callable = callable_

    class FakeGraphableLayer(gpt_callables.GraphableMegatronModule):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(
                moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None
            )
            self.mlp = object()
            self.cuda_graphs = [object()]
            self.current_microbatch = 0
            self.created_wrappers = []
            self.graphed_dw_microbatches = []

        def init_backward_dw_wrapper(self):
            self.backward_dw_wrapper = FakeBackwardDWWrapper()
            self.created_wrappers.append(self.backward_dw_wrapper)

        def _te_cuda_graph_backward_dw_graph(self, microbatch):
            self.graphed_dw_microbatches.append(microbatch)

        @staticmethod
        def _te_cuda_graph_replay(hidden_states, **_kwargs):
            return hidden_states, None, None, None

    layer = FakeGraphableLayer()
    logical_callables = []
    owned_wrappers = []
    for _ in range(3):
        forward_callables, backward_dw = gpt_callables.build_transformer_layer_callables(layer)
        logical_callables.append(forward_callables[0])
        owned_wrappers.append(backward_dw["pre_dispatch_computation"])

    assert len({id(wrapper) for wrapper in owned_wrappers}) == 3
    assert layer.backward_dw_wrapper is owned_wrappers[-1]

    chunk_state = SimpleNamespace(
        padding_mask=None,
        attention_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    )
    hidden_states = torch.ones(2, 1, 4)
    for logical_depth, callable_ in enumerate(logical_callables):
        layer.current_microbatch = logical_depth
        node = SimpleNamespace(chunk_state=chunk_state, layer_state=SimpleNamespace())
        assert callable_(node, hidden_states) is hidden_states

    assert all(wrapper.graphed_backward_dw_callable is not None for wrapper in owned_wrappers)
    for wrapper in owned_wrappers:
        wrapper.graphed_backward_dw_callable()
    assert layer.graphed_dw_microbatches == [0, 1, 2]


def run_model_ref_with_capture(model, input_tensors, iterations):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """

    output_tensors = []
    for i in range(iterations):
        output = model(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_model_submodules_with_capture(model, input_tensors, microbatches):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """

    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()

    output_tensors = []
    # get callables
    callables, dw = build_layer_callables(model)
    attn, dispatch, moe, combine, post_process = callables
    assert post_process is None
    dummy_model = DummyState()
    dummy_model.decoder = DummyState()
    dummy_model.decoder.final_layernorm = None
    for i in range(microbatches):
        # build mock func/state
        node = DummyNode()
        node.is_mtp = False
        node.chunk_state.model = dummy_model

        # attn fwd
        local_tokens, probs = attn(node, input_tensors[i])

        # dispatch fwd
        dispatched_tokens = dispatch(node, local_tokens, probs)

        # moe fwd
        expert_output = moe(node, dispatched_tokens)

        # combine fwd
        hidden_states = combine(node, expert_output)

        # loss
        output_tensors.append(hidden_states)
        hidden_states.backward(torch.ones_like(hidden_states))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def test_mtp_pre_dispatch_applies_hybrid_empty_decoder_final_norm(monkeypatch):
    """Covers the HybridModel empty-decoder MTP pre-dispatch final_norm path."""

    from megatron.core.models.hybrid.hybrid_model import HybridModel

    def inner_pre_dispatch(_node, hidden_states, padding_mask=None):
        del padding_mask
        return hidden_states

    def unused_forward(*_args, **_kwargs):
        raise AssertionError("only MTP pre-dispatch should run in this test")

    def fake_build_layer_callables(_layer):
        return (
            [inner_pre_dispatch, unused_forward, unused_forward, unused_forward, None],
            {"pre_dispatch_computation": object()},
        )

    class FakeMTPConfig:
        sequence_parallel = False
        mtp_num_layers = 1

    class FakeMTPLayer:
        config = FakeMTPConfig()
        cp_group = None
        tp_group = None
        eh_proj = object()
        mtp_model_layer = object()

        def _get_embeddings(
            self,
            input_ids,
            position_ids,
            embedding,
            hidden_states,
            packed_seq_params,
            padding_mask,
            **_kwargs,
        ):
            return input_ids, position_ids, padding_mask, None, hidden_states

        def _concat_embeddings(self, hidden_states, decoder_input):
            return hidden_states

        def _postprocess(self, hidden_states):
            return hidden_states

    monkeypatch.setattr(common_callables, "build_layer_callables", fake_build_layer_callables)
    monkeypatch.setattr(common_callables, "get_layer_moe_metadata", lambda _layer: (True, 1))
    monkeypatch.setattr(common_callables, "get_mtp_layer_offset", lambda _config, _vp_stage: 0)
    monkeypatch.setattr(common_callables, "prepare_mtp_sequence_roll_context", lambda *_args: None)

    model = HybridModel.__new__(HybridModel)
    torch.nn.Module.__init__(model)
    model.decoder = DummyState()
    model.decoder.layers = []
    model.decoder.final_norm = lambda hidden_states: hidden_states + 4.0
    model.embedding = object()
    model.post_process = False
    model.vp_stage = None

    node = DummyNode()
    node.chunk_state = DummyState()
    node.chunk_state.model = model
    node.chunk_state.context = None
    node.chunk_state.input_ids = torch.zeros(1, 3, dtype=torch.long)
    node.chunk_state.position_ids = torch.zeros(1, 3, dtype=torch.long)
    node.chunk_state.labels = None
    node.chunk_state.loss_mask = None
    node.chunk_state.padding_mask = None
    node.chunk_state.mtp_padding_mask = None
    node.chunk_state.mtp_materialized_roll_rows = None
    node.chunk_state.packed_seq_params = None
    node.is_first_layer = True
    node.mtp_absolute_depth = 1

    hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2).requires_grad_()
    expected = hidden_states + 4.0
    forward_funcs, _ = common_callables.build_mtp_layer_callables(FakeMTPLayer())

    output = forward_funcs[0](node, hidden_states)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(node.chunk_state.mtp_hidden_states[0], expected)


def test_fine_grained_repeated_mtp_reuses_prepared_rows_by_absolute_depth(monkeypatch):
    """One physical MTP layer consumes immutable rows for three logical depths."""
    source_rows = {
        "input_ids": tuple(torch.tensor([[depth, depth + 1, 0]]) for depth in (1, 2, 3)),
        "position_ids": tuple(torch.tensor([[10 + depth, 11 + depth, 0]]) for depth in (1, 2, 3)),
        "padding_mask": tuple(torch.tensor([[False, False, depth == 3]]) for depth in (1, 2, 3)),
    }
    materialize_calls = []

    class FakeSequenceRollContext:
        max_offset = 3
        keys = tuple(source_rows)

        @staticmethod
        def is_prepared_for_fields(fields):
            return {field.key for field in fields} == {
                "input_ids",
                "position_ids",
                "labels",
                "loss_mask",
                "padding_mask",
            }

        def prepare_fields(self, fields, *, max_offset):
            raise AssertionError("An already prepared context must be reused.")

        @staticmethod
        def materialize_all(key):
            materialize_calls.append(key)
            return source_rows[key]

    sequence_roll_context = FakeSequenceRollContext()
    prepare_calls = []

    def fake_prepare(*args):
        prepare_calls.append(args)
        return sequence_roll_context

    attn_calls = []

    def fake_attn(node, hidden_states, padding_mask=None):
        del node
        attn_calls.append(padding_mask)
        return hidden_states

    def unused_callable(*args, **kwargs):
        raise AssertionError("Only MTP attention/postprocess should run in this test.")

    def fake_build_layer_callables(layer):
        del layer
        return [fake_attn, unused_callable, unused_callable, unused_callable, None], {
            "pre_dispatch_computation": []
        }

    monkeypatch.setattr(common_callables, "build_layer_callables", fake_build_layer_callables)
    monkeypatch.setattr(common_callables, "get_layer_moe_metadata", lambda _layer: (True, 1))
    monkeypatch.setattr(common_callables, "prepare_mtp_sequence_roll_context", fake_prepare)
    monkeypatch.setattr(common_callables, "resolve_cp_group", lambda group, packed: group)

    embedding_calls = []

    def get_embeddings(**kwargs):
        embedding_calls.append(kwargs)
        return (
            kwargs["input_ids"],
            kwargs["position_ids"],
            kwargs["padding_mask"],
            torch.zeros_like(kwargs["hidden_states"]),
            kwargs["hidden_states"],
        )

    config = SimpleNamespace(sequence_parallel=False, mtp_num_layers=3)
    layer = SimpleNamespace(
        config=config,
        cp_group=object(),
        tp_group=object(),
        layer_number=1,
        mtp_model_layer=object(),
        _get_embeddings=get_embeddings,
        _concat_embeddings=lambda hidden_states, decoder_input: hidden_states + decoder_input,
        _postprocess=lambda hidden_states: hidden_states,
    )
    callables = common_callables.build_mtp_layer_callables(layer)[0]
    mtp_attn, mtp_postprocess = callables[0], callables[4]

    input_ids = torch.tensor([[0, 1, 2]])
    position_ids = torch.tensor([[10, 11, 12]])
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    packed_seq_params = SimpleNamespace(qkv_format="thd")
    chunk_state = SimpleNamespace(
        model=SimpleNamespace(
            embedding=SimpleNamespace(add_position_embedding=True), post_process=True, vp_stage=None
        ),
        input_ids=input_ids,
        position_ids=position_ids,
        labels=input_ids + 1,
        loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
        padding_mask=padding_mask,
        mtp_padding_mask=padding_mask,
        packed_seq_params=packed_seq_params,
        context=None,
        mtp_hidden_states=None,
        mtp_sequence_roll_context=None,
        mtp_materialized_roll_rows=None,
    )
    hidden_states = torch.ones(3, 1, 4)
    current = hidden_states
    for depth in range(1, 4):
        node = SimpleNamespace(
            is_first_layer=depth == 1,
            is_last_layer=depth == 3,
            mtp_absolute_depth=depth,
            chunk_state=chunk_state,
        )
        current = mtp_attn(node, current)
        current = mtp_postprocess(node, current)

    assert current.shape[0] == 4 * hidden_states.shape[0]
    assert len(prepare_calls) == 1
    assert prepare_calls[0] == (input_ids, layer.cp_group, packed_seq_params)
    assert materialize_calls == ["input_ids", "position_ids", "padding_mask"]
    assert len(embedding_calls) == 3
    for depth, call in enumerate(embedding_calls, 1):
        assert call["input_ids"] is source_rows["input_ids"][depth - 1]
        assert call["position_ids"] is source_rows["position_ids"][depth - 1]
        assert call["padding_mask"] is source_rows["padding_mask"][depth - 1]
        assert call["_inputs_pre_aligned"]
        assert call["roll_depth"] == depth - 1
        assert call["packed_seq_params"] is packed_seq_params
    assert all(
        actual is expected for actual, expected in zip(attn_calls, source_rows["padding_mask"])
    )
    assert chunk_state.input_ids is input_ids
    assert chunk_state.position_ids is position_ids
    assert chunk_state.padding_mask is padding_mask


def test_repeated_mtp_schedule_expands_one_physical_layer_to_three_depths(monkeypatch):
    """Fine-grained scheduling carries absolute depth without copying parameters."""
    records = []

    class RecordingLayerPlan:
        def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args):
            del event, chunk_state, comp_stream, comm_stream
            records.append((layer, dict(extra_args)))

    monkeypatch.setattr(
        model_chunk_schedule_plan, "TransformerLayerSchedulePlan", RecordingLayerPlan
    )
    config = SimpleNamespace(mtp_num_layers=3)
    physical_layer = SimpleNamespace(layer_number=1)
    module = SimpleNamespace(
        layers=[physical_layer], config=config, mtp_use_repeated_layer=True, mtp_num_depths=0
    )
    owner = model_chunk_schedule_plan.TransformerModelChunkSchedulePlan.__new__(
        model_chunk_schedule_plan.TransformerModelChunkSchedulePlan
    )
    owner._transformer_layers = []
    owner._event = object()
    owner._model_chunk_state = SimpleNamespace(model=SimpleNamespace(mtp=module))

    owner._build_layer_schedule_plan(module, "compute", "communication")

    assert len(records) == 3
    assert all(layer is physical_layer for layer, _ in records)
    assert [extra["mtp_absolute_depth"] for _, extra in records] == [1, 2, 3]
    assert [extra["is_first_layer"] for _, extra in records] == [True, False, False]
    assert [extra["is_last_layer"] for _, extra in records] == [False, False, True]


class TestTransformerLayerSubmoduleCallables:
    """
    Test class for transformer layer submodule callables.

    This class contains tests to verify that the transformer layer submodule callables
    provide the same results as the reference implementation.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("permute_fusion", [True, False])
    def test_1f1b_overlap(self, dispatcher_type, grouped_gemm, permute_fusion):
        """
        Tests the 1-forward-1-backward overlap optimization.

        This test verifies that the all-to-all overlap optimization produces
        the same results as the reference implementation.
        """

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        qk_layernorm = True
        extra_kwargs = {
            "moe_token_dispatcher_type": dispatcher_type,
            "moe_permute_fusion": permute_fusion,
            "qk_layernorm": qk_layernorm,
        }
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()
        config = get_test_config(extra_kwargs=extra_kwargs, moe_grouped_gemm=grouped_gemm)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
                num_experts=8,
                moe_grouped_gemm=grouped_gemm,
                qk_layernorm=qk_layernorm,
                multi_latent_attention=True,
            )
            model = TransformerLayer(config, transformer_layer_submodules)

            params = reset_model(model)
            input_tensors = [build_data() for _ in range(microbatches)]

            capture_ref = run_model_ref_with_capture(model, input_tensors, microbatches)
            reset_model(model, params)
            capture_callables = run_model_submodules_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_callables, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
            Utils.destroy_model_parallel()
