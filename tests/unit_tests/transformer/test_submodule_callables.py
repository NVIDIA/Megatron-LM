# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import pytest
import torch

from megatron.core.models.common import fine_grained_callables as common_callables
from megatron.core.models.common.fine_grained_callables import build_layer_callables
from megatron.core.models.gpt import fine_grained_callables as gpt_callables
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
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
    # Six slots: mHC post-processing is its own node. This model has no hyper
    # connections, so that slot is None -- but unpack it explicitly so a future
    # width change fails here rather than silently dropping the last callable.
    attn, dispatch, moe, combine, post_process, mhc_post = callables
    assert post_process is None
    assert mhc_post is None
    dummy_model = DummyState()
    dummy_model.decoder = DummyState()
    dummy_model.decoder.final_layernorm = None
    for i in range(microbatches):
        # build mock func/state
        node = DummyNode()
        node.is_mtp = False
        node.is_last_layer = False
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

    def inner_pre_dispatch(_node, hidden_states):
        return hidden_states

    def unused_forward(*_args, **_kwargs):
        raise AssertionError("only MTP pre-dispatch should run in this test")

    def fake_build_layer_callables(_layer):
        return (
            [inner_pre_dispatch, unused_forward, unused_forward, unused_forward, None, None],
            {"attn": object()},
        )

    class FakeMTPConfig:
        sequence_parallel = False

    class FakeMTPLayer:
        config = FakeMTPConfig()
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
            mtp_input_mask=None,
        ):
            return input_ids, position_ids, padding_mask, mtp_input_mask, None, hidden_states

        def _concat_embeddings(self, hidden_states, decoder_input):
            return hidden_states

        def _postprocess(self, hidden_states):
            return hidden_states

    monkeypatch.setattr(common_callables, "build_layer_callables", fake_build_layer_callables)
    monkeypatch.setattr(common_callables, "get_layer_moe_metadata", lambda _layer: (True, 1))
    monkeypatch.setattr(
        common_callables, "get_mtp_layer_offset", lambda _config, _vp_stage, pp_rank=None: 0
    )

    model = HybridModel.__new__(HybridModel)
    torch.nn.Module.__init__(model)
    model.decoder = DummyState()
    model.decoder.layers = []
    model.decoder.final_norm = lambda hidden_states: hidden_states + 4.0
    model.embedding = object()
    model.vp_stage = None
    model.pg_collection = DummyState()
    model.pg_collection.pp = DummyState()
    model.pg_collection.pp.rank = lambda: 0

    node = DummyNode()
    node.chunk_state = DummyState()
    node.chunk_state.model = model
    node.chunk_state.context = None
    node.chunk_state.packed_seq_params = None
    node.is_first_layer = True

    hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2).requires_grad_()
    expected = hidden_states + 4.0
    forward_funcs, _ = common_callables.build_mtp_layer_callables(FakeMTPLayer())
    assert len(forward_funcs) == 6 and forward_funcs[5] is None

    output = forward_funcs[0](node, hidden_states)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(node.chunk_state.mtp_hidden_states[0], expected)


def test_common_mtp_callables_keep_six_slots_and_intermediate_multistream(monkeypatch):
    from types import SimpleNamespace

    mhc_post = object()
    monkeypatch.setattr(
        common_callables,
        "build_layer_callables",
        lambda _: ([None, None, None, None, None, mhc_post], {"attn": []}),
    )
    monkeypatch.setattr(common_callables, "get_layer_moe_metadata", lambda _: (True, 1))
    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=False, enable_hyper_connections=True),
        mtp_model_layer=object(),
        e_proj=object(),
        h_proj=object(),
        eh_proj=None,
        _postprocess=lambda hidden: hidden[..., :2],
    )
    forwards, backwards = common_callables.build_mtp_layer_callables(layer)
    assert len(forwards) == 6 and forwards[5] is mhc_post
    assert backwards["attn"] == [layer.e_proj, layer.h_proj]
    hidden = torch.randn(3, 1, 4)
    node = SimpleNamespace(
        is_last_layer=False,
        chunk_state=SimpleNamespace(
            mtp_hidden_states=[torch.zeros(3, 1, 2)], mhc_multistream=hidden
        ),
    )
    assert forwards[4](node, hidden) is hidden
    assert node.chunk_state.mtp_hidden_states[-1].shape == (3, 1, 2)
    node.is_last_layer = True
    assert forwards[4](node, hidden).shape == (9, 1, 2)
    assert node.chunk_state.mtp_hidden_states is None and node.chunk_state.mhc_multistream is None


def test_mhc_single_stream_combine_keeps_shared_expert_output(monkeypatch):
    from types import SimpleNamespace

    layer = gpt_callables.HyperConnectionTransformerLayer.__new__(
        gpt_callables.HyperConnectionTransformerLayer
    )
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        moe_token_dispatcher_type="alltoall", sequence_parallel=False, num_residual_streams=1
    )
    layer.hidden_dropout = 0.0
    layer.mlp_bda = gpt_callables.get_bias_dropout_add
    layer.mlp = gpt_callables.MoELayer.__new__(gpt_callables.MoELayer)
    torch.nn.Module.__init__(layer.mlp)
    layer.mlp.combine = lambda output: output
    layer.mlp.postprocess = lambda output, shared: output if shared is None else output + shared
    layer.init_backward_dw_wrapper = lambda: None
    layer.backward_dw_wrapper = lambda: None
    monkeypatch.setattr(gpt_callables, 'can_use_fused_moe_residual_add', lambda *_: True)
    output = torch.ones(2, 1, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    shared = torch.ones_like(output, requires_grad=True)
    node = SimpleNamespace(
        layer_state=SimpleNamespace(residual=torch.zeros_like(output), shared_expert_output=shared)
    )
    forwards, _ = gpt_callables.build_transformer_layer_callables(layer)
    combined = forwards[3](node, output)
    torch.testing.assert_close(combined, output + shared, rtol=0, atol=0)
    combined.sum().backward()
    torch.testing.assert_close(shared.grad, torch.ones_like(shared), rtol=0, atol=0)
    torch.testing.assert_close(output.grad, torch.ones_like(output), rtol=0, atol=0)


def test_gpt_mtp_schedule_preserves_six_field_embedding_contract(monkeypatch):
    from types import SimpleNamespace

    mask = torch.tensor([[True, False, True]])
    rolled_mask = torch.tensor([[False, True, False]])

    def embeddings(**kwargs):
        assert kwargs["mtp_input_mask"] is mask
        return (
            kwargs["input_ids"],
            kwargs["position_ids"],
            kwargs["padding_mask"],
            rolled_mask,
            torch.zeros_like(kwargs["hidden_states"]),
            kwargs["hidden_states"] + 2,
        )

    monkeypatch.setattr(
        gpt_callables,
        "build_transformer_layer_callables",
        lambda _layer: ([lambda _node, hidden: hidden, None, None, None, None, None], {"attn": []}),
    )
    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=False, enable_hyper_connections=False),
        mtp_model_layer=SimpleNamespace(mlp=gpt_callables.MoELayer.__new__(gpt_callables.MoELayer)),
        eh_proj=object(),
        _get_embeddings=embeddings,
        _concat_embeddings=lambda hidden, decoder_input: hidden + decoder_input,
    )
    state = SimpleNamespace(
        input_ids=torch.arange(3).unsqueeze(0),
        position_ids=torch.arange(3).unsqueeze(0),
        model=SimpleNamespace(embedding=object()),
        packed_seq_params=None,
        padding_mask=None,
        mtp_input_mask=mask,
        context=None,
    )
    node = SimpleNamespace(is_first_layer=False, chunk_state=state)
    hidden = torch.zeros(3, 1, 2)
    forwards, _ = gpt_callables.build_mtp_layer_callables(layer)
    torch.testing.assert_close(forwards[0](node, hidden), hidden + 2)
    assert state.mtp_input_mask is rolled_mask


class TestTransformerLayerSubmoduleCallables:
    """
    Test class for transformer layer submodule callables.

    This class contains tests to verify that the transformer layer submodule callables
    provide the same results as the reference implementation.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        reset_hybrid_ep_buffer()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize(
        "grouped_gemm, dispatcher_type",
        [
            pytest.param(
                grouped_gemm,
                dispatcher_type,
                # All-to-all crashes (#7744) and grouped-GEMM hangs in dev:
                # https://github.com/NVIDIA/Megatron-LM/actions/runs/38128757991/job/114439165065
                marks=pytest.mark.flaky_in_dev if dispatcher_type == "alltoall" else (),
            )
            for grouped_gemm in [True, False]
            for dispatcher_type in get_valid_token_dispatcher_types()
        ],
    )
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
