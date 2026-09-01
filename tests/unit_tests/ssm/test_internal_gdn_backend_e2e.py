# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GB200 end-to-end correctness and performance coverage for internal GDR."""

import os
from statistics import median

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.launch_on_gb200

_BATCH_SIZE = 2
_SEQUENCE_LENGTH = 8192
_HEADS = 64
_HEAD_DIM = 128
_SAMPLES = 5


def _make_inputs(device):
    torch.manual_seed(1234)
    shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _HEADS, _HEAD_DIM)
    gate_shape = shape[:-1]
    q = (0.1 * torch.randn(shape, device=device, dtype=torch.bfloat16)).requires_grad_()
    k = (0.1 * torch.randn(shape, device=device, dtype=torch.bfloat16)).requires_grad_()
    v = (0.1 * torch.randn(shape, device=device, dtype=torch.bfloat16)).requires_grad_()
    g = F.logsigmoid(torch.randn(gate_shape, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(gate_shape, device=device, dtype=torch.float32))
    inputs = (
        q,
        k,
        v,
        g.to(torch.bfloat16).requires_grad_(),
        beta.to(torch.bfloat16).requires_grad_(),
    )
    return inputs, torch.randn(shape, device=device, dtype=torch.bfloat16)


def _forward_backward(implementation, inputs, grad_output, recompute_h=False, cp_context=None):
    output, _ = implementation.chunk_gated_delta_rule(
        q=inputs[0],
        k=inputs[1],
        v=inputs[2],
        g=inputs[3],
        beta=inputs[4],
        scale=_HEAD_DIM**-0.5,
        recompute_h=recompute_h,
        cp_context=cp_context,
    )
    gradients = torch.autograd.grad(output, inputs, grad_outputs=grad_output)
    return output, gradients


def _median_gpu_ms(implementation, inputs, grad_output, cp_context=None):
    for _ in range(2):
        _forward_backward(implementation, inputs, grad_output, cp_context=cp_context)
    torch.cuda.synchronize()

    samples = []
    for _ in range(_SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _forward_backward(implementation, inputs, grad_output, cp_context=cp_context)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)


@pytest.mark.parametrize("cp_size", [1, 4], ids=["non_cp", "cp4"])
def test_internal_gdr_cute_matches_and_outperforms_fla(monkeypatch, cp_size, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("the fused CuTe DSL kernels require SM100")
    if not torch.version.cuda or int(torch.version.cuda.split(".", 1)[0]) < 13:
        pytest.skip("the fused CuTe DSL kernels require CUDA 13+")
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    pytest.importorskip("fla")

    from fla.ops.cp import build_cp_context

    from megatron.core import parallel_state
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation
    from tests.unit_tests.test_utilities import Utils

    cp_context = None
    if cp_size > 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != cp_size:
            pytest.skip(f"CP{cp_size} coverage requires exactly {cp_size} distributed ranks")
        Utils.initialize_model_parallel(context_parallel_size=cp_size)
        request.addfinalizer(Utils.destroy_model_parallel)
        global_seqlen = _SEQUENCE_LENGTH * cp_size
        cu_seqlens = torch.tensor([0, global_seqlen], device=device, dtype=torch.long)
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens,
            group=parallel_state.get_context_parallel_group(),
            conv1d_kernel_size=4,
        )

    inputs, grad_output = _make_inputs(device)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    reference_output, reference_gradients = _forward_backward(
        implementation, inputs, grad_output, cp_context=cp_context
    )

    calls = {"fwd": 0, "bwd": 0}
    original_forward = implementation._cutedsl_forward
    original_cp_forward = implementation._fla_forward_for_fused_bwd
    original_backward = implementation._call_fused_gdr_bwd_cute

    def tracked_forward(**kwargs):
        calls["fwd"] += 1
        return original_forward(**kwargs)

    def tracked_cp_forward(**kwargs):
        calls["fwd"] += 1
        return original_cp_forward(**kwargs)

    def tracked_backward(**kwargs):
        calls["bwd"] += 1
        return original_backward(**kwargs)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    with monkeypatch.context() as path_guard:
        if cp_context is None:
            path_guard.setattr(
                implementation,
                "_fla_forward_for_fused_bwd",
                lambda **_kwargs: pytest.fail("non-CP cute mode must not run FLA forward"),
            )
            path_guard.setattr(implementation, "_cutedsl_forward", tracked_forward)
        else:
            path_guard.setattr(implementation, "_fla_forward_for_fused_bwd", tracked_cp_forward)
            path_guard.setattr(
                implementation,
                "_cutedsl_forward",
                lambda **_kwargs: pytest.fail("CP mode must use FLA CP forward"),
            )
        path_guard.setattr(implementation, "_call_fused_gdr_bwd_cute", tracked_backward)
        path_guard.setattr(
            implementation,
            "_recompute_fused_bwd_h",
            lambda **_kwargs: pytest.fail("default save-h path must not recompute h"),
        )
        cute_output, cute_gradients = _forward_backward(
            implementation, inputs, grad_output, cp_context=cp_context
        )
    assert calls == {"fwd": 1, "bwd": 1}

    torch.testing.assert_close(cute_output, reference_output, atol=1e-2, rtol=1e-2)
    for name, actual, expected in zip(
        ("q", "k", "v", "g", "beta"), cute_gradients, reference_gradients
    ):
        tolerance = 1e-1 if name in ("g", "beta") else 5e-2
        torch.testing.assert_close(
            actual,
            expected,
            atol=tolerance,
            rtol=tolerance,
            msg=lambda message: f"{name}: {message}",
        )
    auto_forward_calls = []
    original_auto_forward = implementation._fla_forward_for_fused_bwd

    def tracked_auto_forward(**kwargs):
        auto_forward_calls.append(True)
        return original_auto_forward(**kwargs)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    with monkeypatch.context() as path_guard:
        path_guard.setattr(
            implementation,
            "_recompute_fused_bwd_h",
            lambda **_kwargs: pytest.fail("auto save-h path must not recompute h"),
        )
        path_guard.setattr(implementation, "_fla_forward_for_fused_bwd", tracked_auto_forward)
        auto_output, auto_gradients = _forward_backward(
            implementation, inputs, grad_output, cp_context=cp_context
        )
    assert auto_forward_calls == [True]

    torch.testing.assert_close(auto_output, reference_output, atol=1e-2, rtol=1e-2)
    for name, actual, expected in zip(
        ("q", "k", "v", "g", "beta"), auto_gradients, reference_gradients
    ):
        tolerance = 1e-1 if name in ("g", "beta") else 5e-2
        torch.testing.assert_close(
            actual,
            expected,
            atol=tolerance,
            rtol=tolerance,
            msg=lambda message: f"auto {name}: {message}",
        )

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")

    recompute_calls = []
    original_recompute = implementation._recompute_fused_bwd_h

    def tracked_recompute(**kwargs):
        recompute_calls.append(True)
        return original_recompute(**kwargs)

    with monkeypatch.context() as path_guard:
        if cp_context is None:
            path_guard.setattr(
                implementation,
                "_fla_forward_for_fused_bwd",
                lambda **_kwargs: pytest.fail("non-CP cute mode must not run FLA forward"),
            )
        else:
            path_guard.setattr(
                implementation,
                "_cutedsl_forward",
                lambda **_kwargs: pytest.fail("CP mode must use FLA CP forward"),
            )
        path_guard.setattr(implementation, "_recompute_fused_bwd_h", tracked_recompute)
        recompute_output, recompute_gradients = _forward_backward(
            implementation, inputs, grad_output, recompute_h=True, cp_context=cp_context
        )
    assert recompute_calls == ([] if cp_context is not None else [True])

    torch.testing.assert_close(recompute_output, reference_output, atol=1e-2, rtol=1e-2)
    for name, actual, expected in zip(
        ("q", "k", "v", "g", "beta"), recompute_gradients, reference_gradients
    ):
        tolerance = 1e-1 if name in ("g", "beta") else 5e-2
        torch.testing.assert_close(
            actual,
            expected,
            atol=tolerance,
            rtol=tolerance,
            msg=lambda message: f"recompute {name}: {message}",
        )

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    fla_ms = _median_gpu_ms(implementation, inputs, grad_output, cp_context=cp_context)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    cute_ms = _median_gpu_ms(implementation, inputs, grad_output, cp_context=cp_context)
    if cp_context is not None:
        timings = torch.tensor([fla_ms, cute_ms], device=device)
        torch.distributed.all_reduce(
            timings, op=torch.distributed.ReduceOp.MAX, group=cp_context.group
        )
        fla_ms, cute_ms = timings.tolist()

    speedup = fla_ms / cute_ms
    if cp_context is None or torch.distributed.get_rank(group=cp_context.group) == 0:
        global_sequence_length = _SEQUENCE_LENGTH * cp_size
        cute_label = "CuTe fused fwd+bwd" if cp_context is None else "FLA CP fwd + CuTe bwd"
        fla_label = "FLA" if cp_context is None else "FLA CP fwd+bwd"
        print(
            f"internal GDR E2E B={_BATCH_SIZE} local_T={_SEQUENCE_LENGTH} "
            f"global_T={global_sequence_length} CP={cp_size}: "
            f"{cute_label}={cute_ms:.3f} ms, {fla_label}={fla_ms:.3f} ms, "
            f"speedup={speedup:.2f}x"
        )
    assert cute_ms <= fla_ms * 1.10, (
        f"CuTe path regressed: {cute_ms:.3f} ms vs FLA {fla_ms:.3f} ms " f"({speedup:.2f}x)"
    )


@pytest.mark.parametrize("local_seqlen", [8192, 8190], ids=["aligned", "tail"])
def test_gated_delta_net_dense_batch_cp4_matches_fla(monkeypatch, local_seqlen, request):
    """Exercise the real SBHD model entry with dense B=2 and chunkwise CP4."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("dense GatedDeltaNet CP coverage requires exactly four ranks")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("the fused CuTe DSL kernels require SM100")
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    pytest.importorskip("fla")

    from megatron.core import parallel_state
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(context_parallel_size=4)
    request.addfinalizer(Utils.destroy_model_parallel)
    model_parallel_cuda_manual_seed(123)
    cp_group = parallel_state.get_context_parallel_group()
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=cp_group,
        tp_cp=parallel_state.get_tensor_and_context_parallel_group(),
    )
    config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=64,
        linear_num_value_heads=64,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=64,
        num_query_groups=8,
        activation_func=F.silu,
        bf16=True,
        context_parallel_size=4,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        linear_cp_mode="chunkwise",
        gdn_pre_gated_delta_rule_fusion=True,
        gdn_gdr_backend="internal",
        transformer_impl="transformer_engine",
    )
    submodules = get_experimental_attention_variant_module_spec(config=config).submodules
    gdn = GatedDeltaNet(
        config,
        submodules=submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    ).to(device=device, dtype=torch.bfloat16)
    original_rule = gdn.gated_delta_rule

    torch.manual_seed(1234 + local_rank)
    base_input = 0.1 * torch.randn(
        (local_seqlen, _BATCH_SIZE, config.hidden_size), device=device, dtype=torch.bfloat16
    )
    grad_output = torch.randn_like(base_input)

    def execute(*, capture_gdr_inputs):
        gdn.zero_grad(set_to_none=True)
        hidden = base_input.detach().clone().requires_grad_(True)
        captured = []

        def captured_rule(**kwargs):
            tensors = tuple(kwargs[name] for name in ("q", "k", "v", "g", "beta"))
            for tensor in tensors:
                tensor.retain_grad()
            captured.append(tensors)
            return original_rule(**kwargs)

        gdn.gated_delta_rule = captured_rule if capture_gdr_inputs else original_rule
        try:
            output, _ = gdn(hidden, None)
            output.backward(grad_output)
        finally:
            gdn.gated_delta_rule = original_rule

        captured_grads = None
        if capture_gdr_inputs:
            captured_grads = tuple(
                torch.cat([call[index].grad for call in captured], dim=0) for index in range(5)
            )
        return output.detach(), hidden.grad.detach(), captured_grads

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    reference_output, reference_dinput, reference_grads = execute(capture_gdr_inputs=True)

    calls = {"pre": 0, "fwd": 0, "bwd": 0}
    pre_batches = []
    original_pre = gdn._fused_streamed_pre_gated_delta_rule
    original_cp_forward = implementation._fla_forward_for_fused_bwd
    original_backward = implementation._call_fused_gdr_bwd_cute

    def tracked_pre(qkvzba, *args, **kwargs):
        calls["pre"] += 1
        pre_batches.append(qkvzba.shape[1])
        return original_pre(qkvzba, *args, **kwargs)

    def tracked_cp_forward(**kwargs):
        calls["fwd"] += 1
        return original_cp_forward(**kwargs)

    def tracked_backward(**kwargs):
        calls["bwd"] += 1
        return original_backward(**kwargs)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    with monkeypatch.context() as path_guard:
        path_guard.setattr(implementation, "_fla_forward_for_fused_bwd", tracked_cp_forward)
        path_guard.setattr(gdn, "_fused_streamed_pre_gated_delta_rule", tracked_pre)
        path_guard.setattr(implementation, "_call_fused_gdr_bwd_cute", tracked_backward)
        actual_output, actual_dinput, actual_grads = execute(capture_gdr_inputs=True)
    assert calls == {"pre": 1, "fwd": 1, "bwd": 1}
    assert pre_batches == [_BATCH_SIZE]

    def assert_distributed_close(name, actual, expected, *, tolerance):
        close = torch.isclose(actual, expected, atol=tolerance, rtol=tolerance)
        counts = torch.tensor(
            [torch.count_nonzero(~close), close.numel()], device=device, dtype=torch.int64
        )
        max_abs = torch.nan_to_num(
            (actual.float() - expected.float()).abs(),
            nan=float("inf"),
            posinf=float("inf"),
            neginf=float("inf"),
        ).max()
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=cp_group)
        torch.distributed.all_reduce(max_abs, op=torch.distributed.ReduceOp.MAX, group=cp_group)
        bad_count, element_count = counts.tolist()
        mismatch_fraction = bad_count / element_count
        assert mismatch_fraction <= 1e-6 and max_abs.item() <= tolerance * 1.5, (
            f"{name}: {bad_count}/{element_count} elements across CP ranks "
            f"({mismatch_fraction:.3e}) are outside atol=rtol={tolerance}; "
            f"global max abs diff={max_abs.item()}, permitted mismatch fraction=1e-6, "
            f"permitted max abs diff={tolerance * 1.5}"
        )

    assert_distributed_close("output", actual_output, reference_output, tolerance=1e-2)
    assert_distributed_close("input", actual_dinput, reference_dinput, tolerance=1.25e-1)
    for name, actual, expected in zip(("q", "k", "v", "g", "beta"), actual_grads, reference_grads):
        assert_distributed_close(name, actual, expected, tolerance=1e-1)

    def median_model_ms():
        for _ in range(2):
            execute(capture_gdr_inputs=False)
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            execute(capture_gdr_inputs=False)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return median(samples)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    fla_ms = median_model_ms()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    cute_ms = median_model_ms()
    timings = torch.tensor([fla_ms, cute_ms], device=device)
    torch.distributed.all_reduce(timings, op=torch.distributed.ReduceOp.MAX, group=cp_group)
    fla_ms, cute_ms = timings.tolist()

    if torch.distributed.get_rank(group=cp_group) == 0:
        print(
            f"GatedDeltaNet CP4 B=2 local_T={local_seqlen} global_T={local_seqlen * 4}: "
            f"FLA dense-dispatch={fla_ms:.3f} ms, dense fused-bwd={cute_ms:.3f} ms, "
            f"speedup={fla_ms / cute_ms:.2f}x"
        )
    assert cute_ms <= fla_ms * 1.10
