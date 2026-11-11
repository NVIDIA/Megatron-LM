# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real MoK entrypoints with MCore DDP and an independent EP1 PyTorch oracle."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F


def _assert_numerics(actual, expected, name):
    """Use the pinned MoK BF16 suite's max-absolute and normalized-L1 limits.

    BF16 intermediates and contribution-wise gradient stores are rounded differently
    from PyTorch. Neither cosine similarity alone nor per-element relative error near
    zero describes that difference. The same bounds cover outputs and gradients.
    """
    actual, expected = actual.detach().float(), expected.detach().float()
    assert actual.shape == expected.shape, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    delta = (actual - expected).abs()
    normalized_l1 = delta.sum() / expected.abs().sum().clamp_min(1e-12)
    assert (
        delta.max() <= 0.5 and normalized_l1 <= 0.01
    ), f"{name}: max_abs={delta.max().item()}, normalized_l1={normalized_l1.item()}"


def _reference(x, probs, weights):
    """Dense EP1 oracle: two routed SwiGLUs plus an optional gated shared expert.

    Weights are FP32 leaves containing BF16 values: FP32 GEMMs followed by BF16
    activation casts retain FP32 wgrad contributions instead of rounding weight.grad
    to BF16. This reference does not import MoK, its adapters, or its test helpers.
    """

    def linear(value, weight):
        return F.linear(value.float(), weight.float()).to(torch.bfloat16)

    def expert(fc1, fc2):
        gate, up = fc1.chunk(2, dim=0)
        return linear(F.silu(linear(x, gate)) * linear(x, up), fc2)

    shared = expert(
        weights["shared_experts.linear_fc1.weight"], weights["shared_experts.linear_fc2.weight"]
    )
    if "shared_experts.gate_weight" in weights:
        gate = torch.sigmoid(linear(x, weights["shared_experts.gate_weight"]))
        # MoK's gated forward combines in FP32; its backward differentiates the
        # BF16 shared*gate product. Express that precision contract with autograd.
        product = shared * gate
        result = product.float() + (shared.float() * gate.float() - product.float()).detach()
    else:
        result = shared.float()
    for index in range(2):
        routed = expert(
            weights[f"experts.linear_fc1.weight{index}"],
            weights[f"experts.linear_fc2.weight{index}"],
        )
        result = result + routed.float() * probs[:, index : index + 1]
    return result.to(torch.bfloat16)


def _reference_backward(x, probs, weights, upstream):
    x = x.detach().clone().requires_grad_()
    probs = probs.detach().clone().requires_grad_()
    weights = {name: value.detach().float().requires_grad_() for name, value in weights.items()}
    output = _reference(x, probs, weights)
    grads = torch.autograd.grad(output, (x, probs, *weights.values()), upstream)
    return output.detach(), grads[0], grads[1], dict(zip(weights, grads[2:]))


def _check_main_grads(parameters, expected):
    for name, parameter in parameters.items():
        _assert_numerics(parameter.main_grad, expected[name], name)
        assert parameter.grad is None, name
        assert parameter.grad_added_to_main_grad, name


class _MoKExperts(torch.nn.Module):
    """Retain native expert ownership; route probabilities are independent inputs."""

    def __init__(self, layer):
        super().__init__()
        self.experts = layer.experts
        self.shared_experts = layer.shared_experts
        self.megakernel_experts = layer.megakernel_experts

    def forward(self, x, probs):
        return self.megakernel_experts(x, probs, torch.ones_like(probs, dtype=torch.bool))


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
def test_real_mok_ddp_numerics(gated, main_grad_dtype):
    """Two microbatches, all weight buffers, no parameter mutation, and SGD refresh."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("MoK requires Blackwell SM100/SM103")
    functional = pytest.importorskip("mok.functional")
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.megakernel.mok import runtime
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_moe_experts=2,
            moe_ffn_hidden_size=256,
            moe_shared_expert_intermediate_size=256,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            gradient_accumulation_fusion=True,
            moe_megakernel_backend="mok",
            moe_shared_expert_gate=gated,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            moe_megakernel_backend_config=dict(
                fwd_num_comm_sms=2,
                bwd_num_comm_sms=2,
                minibatch_size=256,
                macrobatch_size=256,
                schedule_capacity_multiplier=2.0,
            ),
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        assert groups.ep.size() == 1
        submodules = get_submodules(
            get_gpt_layer_with_transformer_engine_submodules(
                num_experts=2, moe_grouped_gemm=True
            ).mlp
        )
        layer = MoELayer(config, submodules, layer_number=1, pg_collection=groups).cuda()
        model = _MoKExperts(layer)
        # Identical per-rank inputs/weights make DDP's average a no-op numerically.
        generator = torch.Generator(device="cuda").manual_seed(9123)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator, device="cuda") / 16
                )
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=main_grad_dtype == torch.float32,
                overlap_grad_reduce=False,
                use_distributed_optimizer=False,
            ),
            model,
            pg_collection=groups,
        )
        parameters = dict(model.named_parameters())
        assert len(parameters) == 6 + int(gated)
        assert len({id(p) for p in parameters.values()}) == len(parameters)
        checkpoint = model.state_dict()
        assert set(parameters) <= set(checkpoint)
        assert all(not name.startswith("megakernel_experts.") for name in checkpoint)
        assert all(not name.startswith("megakernel_experts.") for name in parameters)
        backend = model.megakernel_experts
        if gated:
            assert backend.shared_output_gate_weight is model.shared_experts.gate_weight
        buffers = {name: p.main_grad for name, p in parameters.items()}
        assert all(buf.dtype == main_grad_dtype for buf in buffers.values())
        original = {name: p.detach().clone() for name, p in parameters.items()}
        expected = {name: torch.zeros_like(p.main_grad) for name, p in parameters.items()}
        batches = []
        ddp.zero_grad_buffer()
        for _ in range(2):
            x = torch.randn(512, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
            probs = torch.randn(512, 2, generator=generator, device="cuda").softmax(-1)
            upstream = torch.randn(x.shape, generator=generator, device="cuda", dtype=x.dtype) / 16
            reference, dx, dp, dw = _reference_backward(x, probs, original, upstream)
            x.requires_grad_()
            probs.requires_grad_()
            output = ddp(x, probs)
            _assert_numerics(output, reference, "output")
            output.backward(upstream)
            _assert_numerics(x.grad, dx, "input gradient")
            _assert_numerics(probs.grad, dp, "router probability gradient")
            for name, param in parameters.items():
                # Gate wgrad adds its FP32 contribution before the destination
                # cast; expert kernels round each reduction-store contribution.
                contribution = dw[name]
                if name != "shared_experts.gate_weight":
                    contribution = contribution.to(main_grad_dtype)
                expected[name].add_(contribution)
                assert param.main_grad is buffers[name], name
                torch.testing.assert_close(param, original[name], rtol=0, atol=0)
            _check_main_grads(parameters, expected)
            batches.append((x.detach(), probs.detach(), upstream))
        ddp.finish_grad_sync()
        _check_main_grads(parameters, expected)

        # Use a real optimizer over deduplicated native parameters. This checks
        # payload/cache refresh, not the distributed optimizer or MXFP8 mode.
        reference_weights = {
            name: torch.nn.Parameter(value.clone()) for name, value in original.items()
        }
        optimizer = torch.optim.SGD(parameters.values(), lr=0.03)
        reference_optimizer = torch.optim.SGD(reference_weights.values(), lr=0.03)
        for name, param in parameters.items():
            param.grad = param.main_grad.to(param.dtype)
            reference_weights[name].grad = expected[name].to(param.dtype)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        backend.is_first_microbatch = True
        x, probs, upstream = batches[0]
        with torch.no_grad():
            updated = ddp(x, probs)
            reference_updated = _reference(x, probs, reference_weights)
            _assert_numerics(updated, reference_updated, "post-update output")
            assert not torch.equal(updated, _reference(x, probs, original))

        if gated and main_grad_dtype == torch.float32:
            # Mutation controls use the real backend. They must fail the same
            # oracle, rather than passing just because a mocked hook was called.
            original_arguments = runtime._gate_up_weight_arguments

            def swapped(*args):
                sg, su, rg, ru = original_arguments(*args)
                return su, sg, rg, ru

            with patch.object(runtime, "_gate_up_weight_arguments", swapped), torch.no_grad():
                wrong = ddp(x, probs)
            with pytest.raises(AssertionError, match="swapped gate/up"):
                _assert_numerics(wrong, reference_updated, "swapped gate/up")
            real_backward = functional.backward
            for missing in ("experts.linear_fc1.weight0", "shared_experts.linear_fc2.weight"):
                ddp.zero_grad_buffer()
                _, _, _, expected_dw = _reference_backward(x, probs, reference_weights, upstream)

                def omit_write(*args, **kwargs):
                    result = real_backward(*args, **kwargs)
                    parameters[missing].main_grad.zero_()
                    return result

                with patch.object(functional, "backward", omit_write):
                    ddp(x.requires_grad_(), probs.requires_grad_()).backward(upstream)
                with pytest.raises(AssertionError, match=missing):
                    _assert_numerics(parameters[missing].main_grad, expected_dw[missing], missing)
    finally:
        functional.clear_workspace_cache()
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        Utils.destroy_model_parallel()
