# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical cuDNN grouped MLP through real GTP all-gather and reduce-scatter.

Every rank has distinct tokens. The independent, unsharded TE reference uses the same
precision recipe; its full FP32 wgrad is averaged explicitly before comparison with
GTP's local main_grad. BF16 additionally has an independent torch GEMM reference.
"""

import copy
import functools
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import (
    GTP_CONFIG,
    classify_gtp_remat_chains,
    configure_gtp_remat_from_recipe,
    dequantize_gtp_native_fp8,
    gtp_native_fp8_load_context,
    is_gtp_param,
    wait_for_gtp_grad_reduction_on_current_stream,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _requires_multi_gpu,
    _requires_mxfp8,
    _requires_nvfp4,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)
from tests.unit_tests.generalized_tensor_parallel.test_gtp_grouped_mlp_precision import (
    _assert_bf16_close,
    _reference_mlp,
)

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]


def canonical_recipe(precision):
    if precision == "mxfp8":
        return MXFP8BlockScaling()
    if precision == "nvfp4":
        # Remove RNG ordering as a confounder between sharded and unsharded executions.
        return NVFP4BlockScaling(disable_stochastic_rounding=True)
    return None


def canonical_config(precision, *, egtp_size=1):
    return TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        num_moe_experts=3,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_mlp_glu_interleave_size=None,
        moe_token_dispatcher_type="alltoall",
        moe_router_padding_for_quantization=precision != "bf16",
        use_transformer_engine_op_fuser=True,
        expert_gtp_weight_remat_size=egtp_size,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8="e4m3" if precision == "mxfp8" else None,
        fp8_recipe="mxfp8",
        fp4="e2m1" if precision == "nvfp4" else None,
        fp4_recipe="nvfp4",
    )


def canonical_experts(config, groups):
    return TEGroupedMLP(
        3,
        config,
        GroupedMLPSubmodules(TEColumnParallelGroupedLinear, TERowParallelGroupedLinear),
        pg_collection=groups,
    )


def assert_canonical_fused(experts, precision):
    from transformer_engine.pytorch.ops.fused.grouped_mlp import (
        GroupedMLP_CuTeGEMMGLU,
        GroupedMLP_CuTeGEMMGLUBf16,
    )

    (ops,) = experts._fused_ops
    forward_ops = ops._module_groups[0]._forward_ops
    expected_type = GroupedMLP_CuTeGEMMGLUBf16 if precision == "bf16" else GroupedMLP_CuTeGEMMGLU
    assert len(forward_ops) == 1
    assert isinstance(forward_ops[0][0], expected_type), "canonical fusion silently fell back"
    assert forward_ops[0][0]._gate_then_up


@pytest.fixture
def traced_canonical_glu(monkeypatch):
    """Trace the actual wrappers, without replacing any numerical implementation."""
    import cudnn
    from transformer_engine.pytorch.ops.fused.grouped_mlp import (
        GroupedMLP_CuTeGEMMGLU,
        GroupedMLP_CuTeGEMMGLUBf16,
    )

    calls = []
    for name, kind in (
        ("grouped_gemm_glu_wrapper_sm100", "fwd"),
        ("grouped_gemm_dglu_wrapper_sm100", "bwd"),
    ):
        original = getattr(cudnn, name)

        def wrap(original, kind):
            @functools.wraps(original)
            def traced(*args, **kwargs):
                calls.append((kind, kwargs.get("glu_layout"), kwargs.get("b_tensor") is None))
                return original(*args, **kwargs)

            return traced

        monkeypatch.setattr(cudnn, name, wrap(original, kind))
    for cls in (GroupedMLP_CuTeGEMMGLU, GroupedMLP_CuTeGEMMGLUBf16):
        cls.grouped_gemm_activation_kernel.cache_clear()
        cls.grouped_gemm_dactivation_kernel.cache_clear()
    yield calls
    for cls in (GroupedMLP_CuTeGEMMGLU, GroupedMLP_CuTeGEMMGLUBf16):
        cls.grouped_gemm_activation_kernel.cache_clear()
        cls.grouped_gemm_dactivation_kernel.cache_clear()


@pytest.mark.parametrize(
    "precision,native_params,empty_rank",
    [
        ("bf16", False, False),
        ("mxfp8", False, False),
        ("nvfp4", False, False),
        ("mxfp8", True, False),
        ("nvfp4", False, True),
    ],
    ids=[
        "bf16",
        "mxfp8-bf16-shards",
        "nvfp4-bf16-shards",
        "mxfp8-native-shards",
        "nvfp4-empty-rank",
    ],
)
def test_canonical_grouped_mlp_gtp_numerics(
    monkeypatch, grouped_mlp_fusions, traced_canonical_glu, precision, native_params, empty_rank
):
    """EGTP2 discrete weights: accumulated wgrad, scale-sensitive SGD, and local reload."""
    if not HAVE_GTP:
        pytest.skip("GTP requires TransformerEngine >= 2.19")
    _requires_multi_gpu(4)
    if dist.get_world_size() != 4:
        pytest.skip("requires torchrun --nproc-per-node=4")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("cuDNN canonical grouped GLU requires Blackwell")
    if precision == "mxfp8":
        _requires_mxfp8()
    if precision == "nvfp4":
        _requires_nvfp4()
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    saved_gtp_config = vars(GTP_CONFIG).copy()
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(expert_gtp_remat_size=2)
    try:
        configure_gtp_remat_from_recipe(
            fp4=precision == "nvfp4", fp8=precision == "mxfp8", fp8_recipe="mxfp8"
        )
        model_parallel_cuda_manual_seed(123)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        group = groups.expt_gtp_remat
        recipe = canonical_recipe(precision)
        config = replace(canonical_config(precision, egtp_size=2), fp8_param=native_params)
        with te.quantized_model_init(enabled=native_params, recipe=recipe):
            experts = canonical_experts(config, groups)
        parameters = dict(experts.named_parameters())
        assert len(parameters) == 6
        assert all(is_gtp_param(param) for param in parameters.values())
        assert all(param.dtype == torch.bfloat16 for param in parameters.values())
        assert all(
            bool(getattr(param, "_gtp_native_fp8", False)) == native_params
            for param in parameters.values()
        )
        generator = torch.Generator(device="cuda").manual_seed(456)
        reference_linears = []
        for rows in (512, 256):
            with te.quantized_model_init(enabled=native_params, recipe=recipe):
                linear = te.ops.GroupedLinear(
                    3,
                    256,
                    rows,
                    bias=False,
                    dtype=torch.bfloat16,
                    device="cuda",
                    split_alignment=256,
                    accumulate_into_main_grad=True,
                )
            reference_linears.append(linear)
        reference = te.ops.Sequential(
            reference_linears[0],
            te.ops.ScaledSwiGLU(glu_interleave_size=None),
            reference_linears[1],
        )
        reference_parameters = {}
        full_masters, local_masters = {}, {}
        for name, param in parameters.items():
            fc = 0 if name.startswith("linear_fc1.") else 1
            ref_param = getattr(reference_linears[fc], name.split(".")[1])
            full = (torch.randn(ref_param.shape, device="cuda", generator=generator) / 16).to(
                torch.bfloat16
            )
            with torch.no_grad(), gtp_native_fp8_load_context(experts):
                param.copy_(full.chunk(2, dim=0)[group.rank()])
                ref_param.copy_(full)
            for weight in (param, ref_param):
                weight.main_grad = torch.zeros(weight.shape, device="cuda", dtype=torch.float32)
                weight.grad_added_to_main_grad = False
            reference_parameters[name] = ref_param
            full_masters[name] = torch.nn.Parameter(full.float())
            local_masters[name] = torch.nn.Parameter(
                full.chunk(2, dim=0)[group.rank()].float().clone()
            )
        classify_gtp_remat_chains(experts)
        actual_optimizer = torch.optim.SGD(
            local_masters.values(), lr=0.01, momentum=0.9, foreach=False
        )
        expected_optimizer = torch.optim.SGD(
            full_masters.values(), lr=0.01, momentum=0.9, foreach=False
        )
        splits = (0, 0, 0) if empty_rank and group.rank() == 0 else (256, 0, 512)
        total_tokens = sum(splits)
        split_tensor = torch.tensor(splits, device="cuda", dtype=torch.int64)
        recipe = canonical_recipe(precision)
        for step in range(2):
            expected_accumulated_grads = {
                name: torch.zeros_like(weight.main_grad)
                for name, weight in reference_parameters.items()
            }
            for weight in (*parameters.values(), *reference_parameters.values()):
                weight.main_grad.zero_()
                weight.grad = None
                weight.grad_added_to_main_grad = False
            for microbatch in range(2):
                # GTP computes each microbatch into fresh FP32 scratch, reduces it, then
                # accumulates locally. Match that arithmetic order in the reference;
                # GEMM beta=1 accumulation followed by one final collective rounds
                # differently and can cross BF16 weight boundaries after an SGD step.
                for weight in reference_parameters.values():
                    weight.main_grad.zero_()
                    weight.grad = None
                    weight.grad_added_to_main_grad = False
                generator.manual_seed(1000 + 17 * dist.get_rank() + 2 * step + microbatch)
                inputs = torch.randn(
                    total_tokens, 256, device="cuda", dtype=torch.bfloat16, generator=generator
                ).requires_grad_()
                probs = (
                    0.25 + 0.5 * torch.rand(total_tokens, device="cuda", generator=generator)
                ).requires_grad_()
                ref_inputs = inputs.detach().clone().requires_grad_()
                ref_probs = probs.detach().clone().requires_grad_()
                dy = torch.randn(
                    inputs.shape, device="cuda", dtype=torch.bfloat16, generator=generator
                )
                for name, param in parameters.items():
                    local_weight = (
                        dequantize_gtp_native_fp8(param) if native_params else param.detach()
                    )
                    gathered = torch.empty(
                        reference_parameters[name].shape, device="cuda", dtype=torch.bfloat16
                    )
                    dist.all_gather_into_tensor(gathered, local_weight.contiguous(), group=group)
                    ref_weight = (
                        reference_parameters[name].dequantize()
                        if native_params
                        else reference_parameters[name]
                    )
                    torch.testing.assert_close(
                        gathered,
                        ref_weight,
                        rtol=0,
                        atol=0,
                        msg=f"step={step} microbatch={microbatch} weight={name}",
                    )
                with te.autocast(enabled=recipe is not None, recipe=recipe):
                    output, bias = experts(inputs, split_tensor, probs)
                    expected = reference(ref_inputs, split_tensor, ref_probs, split_tensor)
                assert bias is None
                assert_canonical_fused(experts, precision)
                torch.testing.assert_close(
                    output,
                    expected,
                    rtol=0,
                    atol=0,
                    msg=f"step={step} microbatch={microbatch} forward-before-backward",
                )
                output.backward(dy)
                expected.backward(dy)
                wait_for_gtp_grad_reduction_on_current_stream()
                torch.testing.assert_close(
                    output,
                    expected,
                    rtol=0,
                    atol=0,
                    msg=f"step={step} microbatch={microbatch} forward-after-backward",
                )
                torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=0, atol=0)
                torch.testing.assert_close(probs.grad, ref_probs.grad, rtol=1e-5, atol=1e-4)
                if precision == "bf16" and step == 0 and microbatch == 0:
                    torch_x = inputs.detach().clone().requires_grad_()
                    torch_p = probs.detach().clone().requires_grad_()
                    torch_weights = {
                        name: weight.detach().float().requires_grad_()
                        for name, weight in reference_parameters.items()
                    }
                    torch_output = _reference_mlp(torch_x, torch_p, torch_weights, splits)
                    torch_output.backward(dy)
                    _assert_bf16_close(output, torch_output, "independent output")
                    _assert_bf16_close(inputs.grad, torch_x.grad, "independent input gradient")
                    _assert_bf16_close(probs.grad, torch_p.grad, "independent probability gradient")
                    for name, weight in reference_parameters.items():
                        _assert_bf16_close(
                            weight.main_grad, torch_weights[name].grad, f"independent {name}"
                        )
                for name, weight in reference_parameters.items():
                    microbatch_grad = weight.main_grad.clone()
                    dist.all_reduce(microbatch_grad, group=group)
                    microbatch_grad.div_(2)
                    expected_accumulated_grads[name].add_(microbatch_grad)
            for name, param in parameters.items():
                expected_grad = expected_accumulated_grads[name]
                shard_grad = expected_grad.chunk(2, dim=0)[group.rank()]
                torch.testing.assert_close(param.main_grad, shard_grad, rtol=2e-5, atol=2e-5)
                assert param.grad_added_to_main_grad
                local_masters[name].grad = param.main_grad.clone()
                full_masters[name].grad = expected_grad
            actual_optimizer.step()
            expected_optimizer.step()
            with torch.no_grad(), gtp_native_fp8_load_context(experts):
                for name, param in parameters.items():
                    expected_shard = full_masters[name].chunk(2, dim=0)[group.rank()]
                    torch.testing.assert_close(
                        local_masters[name], expected_shard, rtol=2e-5, atol=2e-6
                    )
                    param.copy_(local_masters[name])
                    reference_parameters[name].copy_(full_masters[name])
            if step == 0:
                # Local state_dict must preserve canonical rows and optimizer momentum.
                snapshots = {
                    name: (
                        dequantize_gtp_native_fp8(param) if native_params else param.detach()
                    ).clone()
                    for name, param in parameters.items()
                }
                state = copy.deepcopy(experts.state_dict())
                optimizer_state = copy.deepcopy(actual_optimizer.state_dict())
                with torch.no_grad(), gtp_native_fp8_load_context(experts):
                    for param in parameters.values():
                        param.zero_()
                experts.load_state_dict(state, strict=True)
                actual_optimizer.load_state_dict(optimizer_state)
                for name, param in parameters.items():
                    restored = dequantize_gtp_native_fp8(param) if native_params else param
                    torch.testing.assert_close(restored, snapshots[name], rtol=0, atol=0)
        assert {call[0] for call in traced_canonical_glu} == {"fwd", "bwd"}
        assert all(
            layout == "gate_then_up" and discrete for _, layout, discrete in traced_canonical_glu
        )
    finally:
        for key, value in saved_gtp_config.items():
            setattr(GTP_CONFIG, key, value)
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()


@pytest.mark.parametrize("kind", ["shared", "dense"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
def test_canonical_singleton_mlp_gtp_numerics(
    monkeypatch, grouped_mlp_fusions, traced_canonical_glu, kind, precision
):
    """Real singleton GTP leaders rematerialize logical weights in shared/dense adapters."""
    _check_canonical_singleton_mlp_gtp(monkeypatch, traced_canonical_glu, kind, precision)


def test_canonical_shared_tp2_gtp2_numerics(monkeypatch, grouped_mlp_fusions, traced_canonical_glu):
    """Logical weight geometry preserves TP-local channels before GTP row all-gather."""
    _check_canonical_singleton_mlp_gtp(
        monkeypatch, traced_canonical_glu, "shared", "bf16", tp_size=2
    )


def _check_canonical_singleton_mlp_gtp(
    monkeypatch, traced_canonical_glu, kind, precision, *, tp_size=1
):
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEFusedMLPWithGroupedLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        reduce_from_tensor_model_parallel_region,
    )
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.shared_experts import FusedSharedExpertMLP

    _requires_multi_gpu(4)
    if dist.get_world_size() != 4:
        pytest.skip("requires torchrun --nproc-per-node=4")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("cuDNN canonical grouped GLU requires Blackwell")
    if precision == "mxfp8":
        _requires_mxfp8()
    if precision == "nvfp4":
        _requires_nvfp4()
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    monkeypatch.delenv("FP4_RECIPE", raising=False)
    saved_gtp_config = vars(GTP_CONFIG).copy()
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size, gtp_remat_size=2)
    try:
        configure_gtp_remat_from_recipe(
            fp4=precision == "nvfp4", fp8=precision == "mxfp8", fp8_recipe="mxfp8"
        )
        model_parallel_cuda_manual_seed(123)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        group = groups.gtp_remat
        config = replace(
            canonical_config(precision),
            num_moe_experts=None,
            moe_ffn_hidden_size=None,
            ffn_hidden_size=256,
            moe_grouped_gemm=False,
            moe_use_grouped_tensor=False,
            use_transformer_engine_op_fuser=False,
            moe_shared_expert_intermediate_size=256 if kind == "shared" else None,
            use_grouped_gemm_for_shared_expert=kind == "shared",
            moe_shared_expert_glu_interleave_size=None,
            moe_shared_expert_overlap=False,
            gtp_weight_remat_size=2,
            tensor_model_parallel_size=tp_size,
            # MCore applies recipe attributes after construction. Set the derived
            # QParams as well: changing only disable_stochastic_rounding does not
            # rerun TE's dataclass __post_init__ that builds backward quantization.
            fp4_recipe_attrs=(
                {
                    "disable_stochastic_rounding": True,
                    "fp4_quant_bwd_grad": canonical_recipe("nvfp4").fp4_quant_bwd_grad,
                }
                if precision == "nvfp4"
                else None
            ),
        )
        if kind == "shared":
            module = FusedSharedExpertMLP(
                config,
                MLPSubmodules(TEColumnParallelLinear, TERowParallelLinear),
                gate=False,
                pg_collection=groups,
            )
        else:
            module = TEFusedMLPWithGroupedLinear(
                config,
                MLPSubmodules(TELayerNormColumnParallelLinear, TERowParallelLinear),
                pg_collection=groups,
            )
        weights = [module.linear_fc1.weight, module.linear_fc2.weight]
        assert all(is_gtp_param(weight) for weight in weights)
        assert all(not weight.is_routed_expert for weight in weights)
        local_ffn = 256 // tp_size
        reference_linears = [
            te.ops.GroupedLinear(
                1,
                columns,
                rows,
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
                split_alignment=256,
                accumulate_into_main_grad=True,
            )
            for columns, rows in ((256, 2 * local_ffn), (local_ffn, 256))
        ]
        activation = te.ops.ScaledSwiGLU(glu_interleave_size=None)
        if kind == "shared":
            activation._grouped_mlp_unit_activation_scale = True
        reference = te.ops.Sequential(reference_linears[0], activation, reference_linears[1])
        reference_norm = None
        if kind == "dense":
            reference_norm = te.ops.LayerNorm(
                256,
                eps=module.linear_fc1.eps,
                device="cuda",
                dtype=torch.bfloat16,
                zero_centered_gamma=module.linear_fc1.zero_centered_gamma,
            )
            with torch.no_grad():
                reference_norm.weight.copy_(module.linear_fc1.layer_norm_weight)
                reference_norm.bias.copy_(module.linear_fc1.layer_norm_bias)
        generator = torch.Generator(device="cuda").manual_seed(654 + groups.tp.rank())
        for weight, linear in zip(weights, reference_linears):
            full = (torch.randn(linear.weight0.shape, device="cuda", generator=generator) / 16).to(
                torch.bfloat16
            )
            with torch.no_grad():
                weight.copy_(full.chunk(2, dim=0)[group.rank()])
                linear.weight0.copy_(full)
            for param in (weight, linear.weight0):
                param.main_grad = torch.zeros(param.shape, device="cuda", dtype=torch.float32)
                param.grad_added_to_main_grad = False
        classify_gtp_remat_chains(module)
        split_tensor = torch.tensor([512], device="cuda", dtype=torch.int64)
        scales = torch.ones(
            1 if kind == "shared" else 512,
            device="cuda",
            dtype=torch.float32 if kind == "shared" else torch.bfloat16,
        )
        recipe = canonical_recipe(precision)
        for step in range(2):
            for weight, linear in zip(weights, reference_linears):
                for param in (weight, linear.weight0):
                    param.main_grad.zero_()
                    param.grad = None
                    param.grad_added_to_main_grad = False
            # TP peers represent the same tokens; GTP peers represent distinct batches.
            generator.manual_seed(800 + 17 * group.rank() + step)
            inputs = torch.randn(
                256, 2, 256, device="cuda", dtype=torch.bfloat16, generator=generator
            ).requires_grad_()
            ref_inputs = inputs.detach().clone().requires_grad_()
            output = module(inputs)
            if kind == "dense":
                output, bias = output
                assert bias is None
            with te.autocast(enabled=recipe is not None, recipe=recipe):
                normed = ref_inputs.reshape(512, 256)
                if reference_norm is not None:
                    normed = reference_norm(normed)
                if tp_size > 1:
                    normed = copy_to_tensor_model_parallel_region(normed, group=groups.tp)
                expected = reference(normed, split_tensor, scales, split_tensor).reshape_as(output)
                if tp_size > 1:
                    expected = reduce_from_tensor_model_parallel_region(expected, group=groups.tp)
            dy = torch.randn(output.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
            output.backward(dy)
            expected.backward(dy)
            wait_for_gtp_grad_reduction_on_current_stream()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(inputs.grad, ref_inputs.grad, rtol=0, atol=0)
            sequence = (
                module._fused_grouped_swiglu_ops[0] if kind == "shared" else module._fused_impl[0]
            )
            forward_ops = sequence._module_groups[0]._forward_ops
            fused_cls = (
                te.ops.fused.GroupedMLP_CuTeGEMMGLUBf16
                if precision == "bf16"
                else te.ops.fused.GroupedMLP_CuTeGEMMGLU
            )
            assert len(forward_ops) == 1 and isinstance(forward_ops[0][0], fused_cls)
            assert forward_ops[0][0]._gate_then_up
            for weight, linear in zip(weights, reference_linears):
                expected_grad = linear.weight0.main_grad.clone()
                dist.all_reduce(expected_grad, group=group)
                expected_grad.div_(2)
                torch.testing.assert_close(
                    weight.main_grad,
                    expected_grad.chunk(2, dim=0)[group.rank()],
                    rtol=2e-5,
                    atol=2e-5,
                )
                with torch.no_grad():
                    full_updated = (linear.weight0.float() - 0.01 * expected_grad).to(
                        torch.bfloat16
                    )
                    weight.copy_(full_updated.chunk(2, dim=0)[group.rank()])
                    linear.weight0.copy_(full_updated)
        assert {call[0] for call in traced_canonical_glu} == {"fwd", "bwd"}
        assert all(layout == "gate_then_up" for _, layout, _ in traced_canonical_glu)
    finally:
        for key, value in saved_gtp_config.items():
            setattr(GTP_CONFIG, key, value)
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()
