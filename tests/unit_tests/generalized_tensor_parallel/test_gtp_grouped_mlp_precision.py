# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical coverage for BF16 MTP experts inside a global MXFP8 context."""

import math

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import fp8_autocast, fp8_model_init

from megatron.core import parallel_state
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.tensor_parallel.gtp_api import (
    GTP_CONFIG,
    classify_gtp_remat_chains,
    configure_gtp_remat_from_recipe,
    is_gtp_param,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _requires_mxfp8,
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]


def _bf16_mtp_recipe():
    return RecipeConfig.from_config_dict(
        {
            "configs": {
                "bf16": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {},
                }
            },
            "matchers": {
                "mtp": {
                    "type": "glob",
                    "enabled": True,
                    "pattern": "mtp.layers.*",
                    "config": "bf16",
                }
            },
        }
    )


def _reference_mlp(inputs, probs, weights, splits):
    """Independent per-expert GEMMs with BF16 boundaries and FP32 weight gradients.

    Reference weights contain exactly BF16-representable values in FP32 leaves. Explicit
    FP32 accumulation followed by a BF16 cast models the GEMM's output rounding while
    retaining FP32 wgrad, like TE's main_grad. SwiGLU and its probability multiply are
    evaluated together in FP32 before the activation output is rounded to BF16.
    """
    outputs = []
    offset = 0
    for expert, count in enumerate(splits):
        x = inputs[offset : offset + count]
        p = probs[offset : offset + count, None]
        hidden = F.linear(x.float(), weights[f"linear_fc1.weight{expert}"]).to(torch.bfloat16)
        gate, value = hidden.float().chunk(2, dim=-1)
        activated = (F.silu(gate) * value * p).to(torch.bfloat16)
        outputs.append(
            F.linear(activated.float(), weights[f"linear_fc2.weight{expert}"]).to(torch.bfloat16)
        )
        offset += count
    return torch.cat(outputs)


def _assert_bf16_close(actual, expected, label):
    """Bound both aggregate error and outliers relative to the reference signal.

    Different GEMM accumulation orders and BF16 activation-gradient rounding preclude
    bitwise comparison. The 1.2% RMS bound allows a few BF16 roundings, while remaining
    below typical MXFP8 quantization error. Normalized weights keep the signal O(1), so
    small initialized outputs cannot make an absolute tolerance hide the regression.
    """
    actual, expected = actual.detach().float(), expected.detach().float()
    assert torch.isfinite(actual).all(), f"{label}: nonfinite result"
    scale = expected.square().mean().sqrt().clamp_min(1e-12)
    error = actual - expected
    relative_rms = (error.square().mean().sqrt() / scale).item()
    relative_max = (error.abs().max() / scale).item()
    assert relative_rms < 0.012, f"{label}: relative RMS error {relative_rms:.6f} >= 0.012"
    assert relative_max < 0.12, f"{label}: max error / reference RMS {relative_max:.6f} >= 0.12"


def _worker_mtp_bf16_numerics(rank, world_size, port, egtp_size):
    hidden, ffn, num_experts = 256, 256, 2
    # Unequal, pre-padded expert segments exercise the device split contract without a
    # dispatcher or an independent host-padding implementation in the reference.
    splits = (256, 512)
    saved_gtp_config = vars(GTP_CONFIG).copy()
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_gtp_remat_size=egtp_size,
    )
    try:
        configure_gtp_remat_from_recipe(fp8=True, fp8_recipe="mxfp8")
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden,
            num_attention_heads=8,
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=ffn,
            moe_router_topk=2,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_single_grouped_weight=False,
            moe_token_dispatcher_type="alltoall",
            moe_router_padding_for_quantization=True,
            use_transformer_engine_op_fuser=True,
            gradient_accumulation_fusion=True,
            add_bias_linear=False,
            gated_linear_unit=True,
            activation_func=F.silu,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8="e4m3",
            fp8_recipe="mxfp8",
            fp8_param=True,
            quant_recipe=_bf16_mtp_recipe(),
        )
        groups = get_default_pg_collection()
        groups.expt_gtp_remat = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["expt_gtp_remat"]
        ).expt_gtp_remat
        group = groups.expt_gtp_remat
        spec = get_moe_module_spec(use_te=True, num_experts=num_experts, moe_grouped_gemm=True)
        # Exercise the real pre-sharded constructor and precision-name resolution used by
        # MTP. Only its expert computation is invoked, so routing is an independent input.
        with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
            layer = spec(
                config,
                layer_number=1,
                pg_collection=groups,
                name="mtp.layers.0.mtp_model_layer.mlp",
            )
        experts = layer.experts
        assert experts._with_fused_impl, "BF16 override must retain the TE ops fallback"
        assert experts._use_grouped_tensor

        generator = torch.Generator(device="cuda").manual_seed(456)
        reference_weights = {}
        for name, param in experts.named_parameters():
            rows = 2 * ffn if name.startswith("linear_fc1.") else hidden
            cols = hidden if name.startswith("linear_fc1.") else ffn
            full = (
                torch.randn(rows, cols, device="cuda", generator=generator) / math.sqrt(cols)
            ).to(torch.bfloat16)
            assert is_gtp_param(param) == (egtp_size > 1), name
            assert not getattr(param, "_gtp_native_fp8", False), name
            assert param.dtype == torch.bfloat16
            local = full.chunk(egtp_size, dim=0)[group.rank()] if egtp_size > 1 else full
            with torch.no_grad():
                param.copy_(local)
            param.main_grad = torch.zeros(param.shape, device="cuda", dtype=torch.float32)
            param.grad_added_to_main_grad = False
            reference_weights[name] = full.float().requires_grad_()
        classify_gtp_remat_chains(experts)

        split_tensor = torch.tensor(splits, device="cuda", dtype=torch.int64)
        # Accumulate two backwards to cover the reuse of TE ops and GTP prefetch chains.
        for microbatch in range(2):
            generator.manual_seed(1000 + 17 * rank + microbatch)
            inputs = torch.randn(
                sum(splits), hidden, device="cuda", dtype=torch.bfloat16, generator=generator
            ).requires_grad_()
            probs = (
                0.25 + 0.5 * torch.rand(sum(splits), device="cuda", generator=generator)
            ).requires_grad_()
            reference_inputs = inputs.detach().clone().requires_grad_()
            reference_probs = probs.detach().clone().requires_grad_()
            grad_output = torch.randn(
                inputs.shape, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            with fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                actual, bias = experts(inputs, split_tensor, probs)
            assert bias is None
            assert experts._fused_ops is not None, "the numerical check must execute TE ops"
            expected = _reference_mlp(reference_inputs, reference_probs, reference_weights, splits)
            actual.backward(grad_output)
            expected.backward(grad_output)
            wait_for_gtp_grad_reduction_on_current_stream()
            _assert_bf16_close(actual, expected, f"microbatch {microbatch} output")
            _assert_bf16_close(inputs.grad, reference_inputs.grad, "input gradient")
            _assert_bf16_close(probs.grad, reference_probs.grad, "probability gradient")

            for name, param in experts.named_parameters():
                expected_grad = reference_weights[name].grad.detach().clone()
                # GTP's default contract is the MEAN over distinct-token peers, before
                # any separate replica-DP reduction. EGTP=1 remains a local reference.
                if egtp_size > 1:
                    dist.all_reduce(expected_grad, group=group)
                    expected_grad.div_(egtp_size)
                    expected_grad = expected_grad.chunk(egtp_size, dim=0)[group.rank()]
                _assert_bf16_close(param.main_grad, expected_grad, f"{name} accumulated wgrad")
    finally:
        for key, value in saved_gtp_config.items():
            setattr(GTP_CONFIG, key, value)
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()


@pytest.mark.parametrize("egtp_size", [1, 2])
def test_mtp_bf16_override_matches_reference(monkeypatch, egtp_size: int) -> None:
    _requires_multi_gpu(4)
    _requires_mxfp8()
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    _run_distributed(_worker_mtp_bf16_numerics, 4, egtp_size)
