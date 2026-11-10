"""Contracts and distributed numerical checks for the NCCL EP TE backend."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.megakernel.te import routing_map_to_te_inputs
from megatron.core.transformer.transformer_config import TransformerConfig


def te_config(**overrides):
    """Minimal supported backend configuration without optional execution features."""
    values = dict(
        num_layers=1,
        hidden_size=512,
        num_attention_heads=8,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_ffn_hidden_size=512,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_megakernel_backend="te",
        moe_single_grouped_weight=True,
        gradient_accumulation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "mxfp8_params", "mxfp8_wire"])
def test_te_backend_precision_config(precision):
    options = {}
    if precision != "bf16":
        options.update(fp8="e4m3", fp8_recipe="mxfp8")
    if precision == "mxfp8_params":
        options["fp8_param"] = True
    if precision == "mxfp8_wire":
        options.update(moe_dispatch_fwd_dtype="mxfp8", moe_combine_bwd_dtype="mxfp8")
    config = te_config(**options)
    assert config.moe_megakernel_backend == "te"
    assert not config.use_transformer_engine_op_fuser


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"delay_wgrad_compute": True}, "delay_wgrad"),
        ({"overlap_dispatch_backward_with_experts_wgrad": True}, "overlap_dispatch"),
        ({"moe_ncclep_zero_copy": True}, "zero_copy"),
        ({"moe_shared_expert_overlap": True}, "shared_expert_overlap"),
        ({"expert_model_parallel_size": 1}, "EP>=2"),
        ({"fp8": "e4m3", "fp8_recipe": "blockwise"}, "mxfp8|MXFP8"),
        ({"moe_router_dtype": "fp64"}, "FP32"),
        ({"moe_apply_probs_on_input": True}, "moe_apply_probs_on_input"),
        ({"cuda_graph_impl": "full_iteration"}, "eager"),
        ({"moe_megakernel_backend_config": {"max_tokens_per_rank": 0}}, "positive integer"),
        ({"moe_megakernel_backend_config": {"max_tokens_per_rank": True}}, "positive integer"),
        ({"moe_megakernel_backend_config": {"unknown": 64}}, "only max_tokens_per_rank"),
    ],
)
def test_te_backend_rejects_unsupported_config(overrides, match):
    with pytest.raises((ValueError, AssertionError), match=match):
        te_config(**overrides)


def test_authoritative_routes_and_probability_gradients():
    probs = torch.tensor([[0.0, 90.0, 0.5, 80.0], [7.0, 8.0, 9.0, 10.0]], requires_grad=True)
    routes = torch.tensor([[True, False, True, False], [False, False, False, False]])
    weights, indices = routing_map_to_te_inputs(probs, routes, 2)
    torch.testing.assert_close(indices, torch.tensor([[0, 2], [-1, -1]], dtype=torch.int32))
    torch.testing.assert_close(weights, torch.tensor([[0.0, 0.5], [0.0, 0.0]]))
    weights.backward(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    torch.testing.assert_close(probs.grad, torch.tensor([[2.0, 0.0, 3.0, 0.0], [0.0] * 4]))


def _assert_relative_error(actual, expected, tolerance, name):
    actual, expected = actual.detach().float(), expected.detach().float()
    assert actual.shape == expected.shape, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = torch.linalg.vector_norm(actual - expected)
    scale = torch.linalg.vector_norm(expected).clamp_min(1e-8)
    assert error <= tolerance * scale + 1e-6, f"{name}: relative L2 error {(error / scale).item()}"


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "mxfp8_params", "mxfp8_wire"])
@pytest.mark.parametrize("static_capacity", [False, True])
@pytest.mark.parametrize("accumulate_main_grad", [False, True])
def test_te_sequence_matches_native_moe(
    precision, static_capacity, accumulate_main_grad, monkeypatch
):
    """Compare two in-flight microbatches and all gradients against native all-to-all."""
    from tests.unit_tests.test_utilities import Utils

    if not torch.cuda.is_available() or Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires an even number of CUDA ranks, at least two")
    if precision != "bf16" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell or newer")
    pytest.importorskip("transformer_engine.pytorch.ep")
    from transformer_engine.pytorch import ep, ops

    if not all(hasattr(ops, name) for name in ("MoeDispatch", "MoeCombine")):
        pytest.skip("requires TE with NCCL EP dispatch/combine operations")

    from megatron.core.fp8_utils import get_fp8_context
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules

    options = {"gradient_accumulation_fusion": accumulate_main_grad}
    if precision != "bf16":
        options.update(fp8="e4m3", fp8_recipe="mxfp8")
    if precision == "mxfp8_params":
        options["fp8_param"] = True
    if precision == "mxfp8_wire":
        options.update(moe_dispatch_fwd_dtype="mxfp8", moe_combine_bwd_dtype="mxfp8")
    if static_capacity:
        options["moe_expert_rank_capacity_factor"] = 16.0
    candidate_config = te_config(**options)
    reference_config = replace(
        candidate_config,
        moe_megakernel_backend=None,
        moe_expert_capacity_factor=1.0,
        moe_expert_rank_capacity_factor=None,
        use_transformer_engine_op_fuser=True,
        moe_dispatch_fwd_dtype="bf16",
        moe_combine_bwd_dtype="bf16",
        moe_token_dispatcher_type="alltoall",
    )
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(1234)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=4, moe_grouped_gemm=True
            ).mlp
        )
        with get_fp8_context(candidate_config, is_init=True):
            candidate = MoELayer(candidate_config, submodules, pg_collection=groups).cuda()
            reference = MoELayer(reference_config, submodules, pg_collection=groups).cuda()
        reference.load_state_dict(candidate.state_dict())
        keys = set(candidate.state_dict())
        assert keys == set(reference.state_dict())
        assert candidate.token_dispatcher is None
        assert not list(candidate.megakernel_experts.parameters())
        assert candidate.megakernel_experts.sharded_state_dict() == {}
        if accumulate_main_grad:
            for model in (candidate, reference):
                for parameter in model.experts.parameters():
                    parameter.main_grad = torch.zeros(
                        parameter.shape, device=parameter.device, dtype=torch.float32
                    )
                    parameter.overwrite_main_grad = False
                    parameter.zero_out_wgrad = False
                    parameter.grad_added_to_main_grad = False
        torch.manual_seed(177 + Utils.rank)
        records = []
        buffers = []
        for batch in range(2):
            hidden = torch.randn(32, 1, 512, device="cuda", dtype=torch.bfloat16)
            reference_hidden = hidden.clone().requires_grad_()
            candidate_hidden = hidden.clone().requires_grad_()
            indices = torch.rand(32, 4, device="cuda").topk(2, dim=-1).indices
            if batch == 1:
                indices[:] = torch.tensor([0, 1], device="cuda")
            routes = torch.zeros(32, 4, dtype=torch.bool, device="cuda").scatter_(1, indices, True)
            routes[0] = False
            probs = torch.rand(32, 4, device="cuda") * routes
            reference_probs = probs.clone().requires_grad_()
            candidate_probs = probs.clone().requires_grad_()
            monkeypatch.setattr(
                reference,
                "route",
                lambda *args, weights=reference_probs, mask=routes, **kwargs: (weights, mask),
            )
            monkeypatch.setattr(
                candidate,
                "route",
                lambda *args, weights=candidate_probs, mask=routes, **kwargs: (weights, mask),
            )
            with get_fp8_context(reference_config):
                reference_out, _ = reference(reference_hidden)
            with get_fp8_context(candidate_config):
                candidate_out, _ = candidate(candidate_hidden)
            sequence = candidate.megakernel_experts._sequence[0]
            assert sequence[0].buffer is sequence[-1].buffer
            buffers.append(sequence[0].buffer)
            tolerance = 0.12 if precision == "mxfp8_wire" else 0.02
            _assert_relative_error(candidate_out, reference_out, tolerance, "output")
            records.append(
                (
                    candidate_out,
                    reference_out,
                    candidate_hidden,
                    reference_hidden,
                    candidate_probs,
                    reference_probs,
                )
            )
        assert buffers[0] is not buffers[1]
        for (
            candidate_out,
            reference_out,
            candidate_hidden,
            reference_hidden,
            candidate_probs,
            reference_probs,
        ) in reversed(records):
            upstream = torch.randn_like(reference_out)
            reference_out.backward(upstream)
            candidate_out.backward(upstream)
            _assert_relative_error(
                candidate_hidden.grad, reference_hidden.grad, tolerance, "dinput"
            )
            _assert_relative_error(candidate_probs.grad, reference_probs.grad, tolerance, "dprobs")
        for (name, parameter), (ref_name, reference_parameter) in zip(
            candidate.named_parameters(), reference.named_parameters()
        ):
            assert name == ref_name
            actual_grad = (
                parameter.main_grad
                if accumulate_main_grad and name.startswith("experts.")
                else parameter.grad
            )
            expected_grad = (
                reference_parameter.main_grad
                if accumulate_main_grad and name.startswith("experts.")
                else reference_parameter.grad
            )
            if expected_grad is None:
                assert parameter.grad is None
                continue
            _assert_relative_error(actual_grad, expected_grad, tolerance, name)
        assert set(candidate.state_dict()) == keys
        assert not candidate.megakernel_experts.state_dict()
        candidate.load_state_dict(reference.state_dict())
        with torch.no_grad(), get_fp8_context(candidate_config):
            reloaded, _ = candidate(records[-1][2].detach())
        _assert_relative_error(reloaded, records[-1][0].detach(), tolerance, "checkpoint reload")
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        ep.ep_finalize()
        Utils.destroy_model_parallel()
