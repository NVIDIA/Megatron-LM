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


def _make_inputs(device, *, packed=False):
    torch.manual_seed(1234)
    shape = (
        (1, _BATCH_SIZE * _SEQUENCE_LENGTH, _HEADS, _HEAD_DIM)
        if packed
        else (_BATCH_SIZE, _SEQUENCE_LENGTH, _HEADS, _HEAD_DIM)
    )
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
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import prepare_cp_context_metadata
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import implementation
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_cp_cute
    from tests.unit_tests.test_utilities import Utils

    cp_context = None
    if cp_size > 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != cp_size:
            pytest.skip(f"CP{cp_size} coverage requires exactly {cp_size} distributed ranks")
        Utils.initialize_model_parallel(context_parallel_size=cp_size)
        request.addfinalizer(Utils.destroy_model_parallel)
        global_seqlen = _SEQUENCE_LENGTH * cp_size
        global_cu_seqlens_cpu = torch.arange(_BATCH_SIZE + 1, dtype=torch.long) * global_seqlen
        cu_seqlens = global_cu_seqlens_cpu.to(device=device)
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens,
            group=parallel_state.get_context_parallel_group(),
            conv1d_kernel_size=4,
        )
        prepare_cp_context_metadata(cp_context, global_cu_seqlens_cpu)

    inputs, grad_output = _make_inputs(device, packed=cp_context is not None)
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    monkeypatch.setenv("MCORE_GDN_CP_CUTEDSL", "0")
    reference_output, reference_gradients = _forward_backward(
        implementation, inputs, grad_output, cp_context=cp_context
    )

    calls = {"fwd": 0, "bwd": 0}
    original_forward = implementation._cutedsl_forward
    original_cp_forward = implementation._fla_forward_for_fused_bwd
    original_backward = implementation._call_fused_gdr_bwd_cute
    cp_forward_before = fused_gdr_cp_cute.get_cutedsl_fused_launch_count()
    cp_backward_before = fused_gdr_cp_cute.get_cutedsl_fused_bwd_launch_count()
    fused_gdr_cp_cute.reset_cutedsl_fallback_reasons()

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
    monkeypatch.setenv("MCORE_GDN_CP_CUTEDSL", "1")
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
    if cp_context is not None:
        assert fused_gdr_cp_cute.get_cutedsl_fused_launch_count() > cp_forward_before
        assert fused_gdr_cp_cute.get_cutedsl_fused_bwd_launch_count() > cp_backward_before
        assert fused_gdr_cp_cute.get_cutedsl_fallback_reasons() == {}

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
