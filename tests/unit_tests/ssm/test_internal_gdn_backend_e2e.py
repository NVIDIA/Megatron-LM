# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GB200 end-to-end correctness and performance coverage for internal GDR."""

import math
import os
from statistics import median

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.launch_on_gb200

_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 8192
_HEADS = 64
_HEAD_DIM = 128
_SAMPLES = 5
# Ragged packing: sequence cuts that are neither 64-aligned nor on CP rank boundaries.
_RAGGED_LOCAL_LENGTHS = (4000, 3240, 65, 1, 886)
_RAGGED_GLOBAL_CUTS = (12345, 20000)


def _ragged_cu_seqlens(total_tokens, device):
    cuts = [c for c in _RAGGED_GLOBAL_CUTS if c < total_tokens]
    return torch.tensor([0, *cuts, total_tokens], device=device, dtype=torch.int32)


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


def _forward_backward(
    implementation, inputs, grad_output, recompute_h=False, cp_context=None, cu_seqlens=None
):
    output, _ = implementation.chunk_gated_delta_rule(
        q=inputs[0],
        k=inputs[1],
        v=inputs[2],
        g=inputs[3],
        beta=inputs[4],
        scale=_HEAD_DIM**-0.5,
        recompute_h=recompute_h,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
    )
    gradients = torch.autograd.grad(output, inputs, grad_outputs=grad_output)
    return output, gradients


def _median_gpu_ms(implementation, inputs, grad_output, cp_context=None, cu_seqlens=None):
    def run():
        _forward_backward(
            implementation, inputs, grad_output, cp_context=cp_context, cu_seqlens=cu_seqlens
        )

    for _ in range(2):
        run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(_SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return median(samples)


@pytest.mark.parametrize("ragged", [False, True], ids=["aligned", "ragged"])
@pytest.mark.parametrize("cp_size", [1, 4], ids=["non_cp", "cp4"])
def test_internal_gdr_cute_matches_and_outperforms_fla(monkeypatch, cp_size, ragged, request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device)[0] != 10:
        pytest.skip("the fused CuTe DSL kernels require a Blackwell data-center GPU (SM10x)")
    if not torch.version.cuda or int(torch.version.cuda.split(".", 1)[0]) < 13:
        pytest.skip("the fused CuTe DSL kernels require CUDA 13+")
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    pytest.importorskip("fla")

    from fla.ops.cp import build_cp_context

    from megatron.core import parallel_state
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import (
        implementation,
        prepare_cp_context_metadata,
    )
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
        if ragged:
            cu_seqlens = _ragged_cu_seqlens(_BATCH_SIZE * global_seqlen, device)
        else:
            cu_seqlens = (
                torch.arange(_BATCH_SIZE + 1, device=device, dtype=torch.int32) * global_seqlen
            )
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens,
            group=parallel_state.get_context_parallel_group(),
            conv1d_kernel_size=4,
        )
        prepare_cp_context_metadata(
            cp_context, global_num_sequences=cu_seqlens.numel() - 1, global_cu_seqlens=cu_seqlens
        )

    # Non-CP ragged coverage packs several sequences with partial final chunks into one
    # THD batch; the CP ragged variant cuts the global sequence inside rank windows.
    packed = cp_context is not None or ragged
    inputs, grad_output = _make_inputs(device, packed=packed)
    cu_seqlens = None
    if cp_context is None and ragged:
        assert sum(_RAGGED_LOCAL_LENGTHS) == _BATCH_SIZE * _SEQUENCE_LENGTH
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(_RAGGED_LOCAL_LENGTHS).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        )
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    monkeypatch.setenv("MCORE_GDN_CP_CUTEDSL", "0")
    reference_output, reference_gradients = _forward_backward(
        implementation, inputs, grad_output, cp_context=cp_context, cu_seqlens=cu_seqlens
    )

    calls = {"fwd": 0, "bwd": 0}
    original_forward = implementation._cutedsl_forward
    original_backward = implementation._call_fused_gdr_bwd_cute
    cp_forward_before = fused_gdr_cp_cute.get_cutedsl_fused_launch_count()
    cp_backward_before = fused_gdr_cp_cute.get_cutedsl_fused_bwd_launch_count()
    fused_gdr_cp_cute.reset_cutedsl_fallback_reasons()

    def tracked_forward(**kwargs):
        calls["fwd"] += 1
        return original_forward(**kwargs)

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
            path_guard.setattr(
                implementation,
                "_fla_forward_for_fused_bwd",
                lambda **_kwargs: pytest.fail("CP cute mode must not run FLA forward"),
            )
            path_guard.setattr(implementation, "_cutedsl_forward", tracked_forward)
        path_guard.setattr(implementation, "_call_fused_gdr_bwd_cute", tracked_backward)
        path_guard.setattr(
            implementation,
            "_recompute_fused_bwd_h",
            lambda **_kwargs: pytest.fail("default save-h path must not recompute h"),
        )
        cute_output, cute_gradients = _forward_backward(
            implementation, inputs, grad_output, cp_context=cp_context, cu_seqlens=cu_seqlens
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
    original_auto_fla_forward = implementation._fla_forward_for_fused_bwd
    original_auto_cute_forward = implementation._cutedsl_forward

    def tracked_auto_fla_forward(**kwargs):
        auto_forward_calls.append("fla")
        return original_auto_fla_forward(**kwargs)

    def tracked_auto_cute_forward(**kwargs):
        auto_forward_calls.append("cute")
        return original_auto_cute_forward(**kwargs)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    with monkeypatch.context() as path_guard:
        path_guard.setattr(
            implementation,
            "_recompute_fused_bwd_h",
            lambda **_kwargs: pytest.fail("auto save-h path must not recompute h"),
        )
        path_guard.setattr(implementation, "_fla_forward_for_fused_bwd", tracked_auto_fla_forward)
        path_guard.setattr(implementation, "_cutedsl_forward", tracked_auto_cute_forward)
        auto_output, auto_gradients = _forward_backward(
            implementation, inputs, grad_output, cp_context=cp_context, cu_seqlens=cu_seqlens
        )
    assert auto_forward_calls == (["fla"] if cp_context is None else ["cute"])

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
                "_fla_forward_for_fused_bwd",
                lambda **_kwargs: pytest.fail("CP cute mode must not run FLA forward"),
            )
        path_guard.setattr(implementation, "_recompute_fused_bwd_h", tracked_recompute)
        recompute_output, recompute_gradients = _forward_backward(
            implementation,
            inputs,
            grad_output,
            recompute_h=True,
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
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
    fla_ms = _median_gpu_ms(
        implementation, inputs, grad_output, cp_context=cp_context, cu_seqlens=cu_seqlens
    )
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    cute_ms = _median_gpu_ms(
        implementation, inputs, grad_output, cp_context=cp_context, cu_seqlens=cu_seqlens
    )
    if cp_context is not None:
        timings = torch.tensor([fla_ms, cute_ms], device=device)
        torch.distributed.all_reduce(
            timings, op=torch.distributed.ReduceOp.MAX, group=cp_context.group
        )
        fla_ms, cute_ms = timings.tolist()

    speedup = fla_ms / cute_ms
    if cp_context is None or torch.distributed.get_rank(group=cp_context.group) == 0:
        global_sequence_length = _SEQUENCE_LENGTH * cp_size
        cute_label = "CuTe fused fwd+bwd" if cp_context is None else "CuTe CP fused fwd+bwd"
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


# --------------------------------------------------------------------------------------------
# KDA: FLA's Triton kernels plus the in-tree CuTe DSL CP pre-processing in its per-key gate mode.
# --------------------------------------------------------------------------------------------

_KDA_HEADS = 8  # the CuTe CP dispatcher needs a multiple of four value heads
_KDA_LOWER_BOUND = -5.0
# Rank-independent: a TP shard holds the same A_log / dt_bias on every CP rank.
_KDA_PARAM_SEED = 4242
_KDA_NAMES = ("out", "dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")
# Relative-L2 limits against FLA's own CP path on the same operands. On GB300 (CP4 and CP8) the
# output and dq are bit-identical and the other gradients sit at 3.7e-3..5.4e-3; the limits
# leave headroom for bf16 accumulation-order noise.
_KDA_REL_L2_LIMITS = {
    "out": 1e-2,
    "dq": 1e-2,
    "dk": 2e-2,
    "dv": 2e-2,
    "dg": 5e-2,
    "dbeta": 2e-2,
    "dA_log": 5e-2,
    "ddt_bias": 5e-2,
}


def _cutedsl_cp_device(cp_size):
    """Skip unless this is a Blackwell CUDA-13 rank of a ``cp_size``-rank distributed run."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(device)[0] != 10:
        pytest.skip("the CuTe DSL CP kernels require a Blackwell data-center GPU (SM10x)")
    if not torch.version.cuda or int(torch.version.cuda.split(".", 1)[0]) < 13:
        pytest.skip("the CuTe DSL CP kernels require CUDA 13+")
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    pytest.importorskip("fla.ops.kda")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != cp_size:
        pytest.skip(f"CP{cp_size} coverage requires exactly {cp_size} distributed ranks")
    return device


def _make_kda_inputs(device, *, head_dim, seed):
    """Operands shaped like KimiDeltaAttention hands them to chunk_kda (B=1, [1, T, H, D])."""
    gen = torch.Generator().manual_seed(seed)
    param_gen = torch.Generator().manual_seed(_KDA_PARAM_SEED)
    shape = (1, _SEQUENCE_LENGTH, _KDA_HEADS, head_dim)

    def rnd(*size):
        return torch.randn(*size, generator=gen).to(device)

    # Megatron L2-normalises q/k before the kernel (use_qk_l2norm_in_kernel=False) and passes
    # the raw gate logits; the kernel applies A_log / dt_bias (use_gate_in_kernel=True).
    q = F.normalize(rnd(*shape), dim=-1).to(torch.bfloat16)
    k = F.normalize(rnd(*shape), dim=-1).to(torch.bfloat16)
    v = (0.1 * rnd(*shape)).to(torch.bfloat16)
    g = rnd(*shape).to(torch.bfloat16)
    beta = torch.sigmoid(rnd(*shape[:-1])).float()
    A_log = torch.log(torch.empty(_KDA_HEADS).uniform_(1.0, 16.0, generator=param_gen)).to(device)
    dt_min, dt_max = 1e-3, 1e-1
    dt = torch.exp(
        torch.rand(_KDA_HEADS * head_dim, generator=param_gen)
        * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=1e-4)
    # Inverse softplus, as KimiDeltaAttention._reset_dt_bias.
    dt_bias = (dt + torch.log(-torch.expm1(-dt))).to(device)
    inputs = tuple(t.requires_grad_() for t in (q, k, v, g, beta, A_log, dt_bias))
    return inputs, rnd(*shape).to(torch.bfloat16)


def _kda_forward_backward(chunk_kda, inputs, grad_output, *, cp_context):
    q, k, v, g, beta, A_log, dt_bias = inputs
    output, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=_KDA_LOWER_BOUND,
        cp_context=cp_context,
    )
    return output, torch.autograd.grad(output, inputs, grad_outputs=grad_output)


def _relative_l2(actual, expected):
    expected = expected.detach().float()
    diff = torch.linalg.vector_norm(actual.detach().float() - expected)
    denom = torch.linalg.vector_norm(expected)
    return (diff / denom).item() if denom > 0 else diff.item()


def _build_kda_cp_context(device, cp_size, *, ragged, request):
    from fla.ops.cp import build_cp_context

    from megatron.core import parallel_state
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import prepare_cp_context_metadata
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    request.addfinalizer(Utils.destroy_model_parallel)
    group = parallel_state.get_context_parallel_group()
    global_tokens = _SEQUENCE_LENGTH * cp_size
    if ragged:
        cu_seqlens = _ragged_cu_seqlens(global_tokens, device)
    else:
        cu_seqlens = torch.tensor([0, global_tokens], device=device, dtype=torch.int32)
    cp_context = build_cp_context(cu_seqlens=cu_seqlens, group=group, conv1d_kernel_size=4)
    prepare_cp_context_metadata(
        cp_context, global_num_sequences=cu_seqlens.numel() - 1, global_cu_seqlens=cu_seqlens
    )
    return cp_context, torch.distributed.get_rank(group=group)


@pytest.mark.parametrize("ragged", [False, True], ids=["aligned", "ragged"])
@pytest.mark.parametrize("cp_size", [2, 4], ids=["cp2", "cp4"])
def test_internal_kda_cute_cp_preprocess_matches_fla(monkeypatch, cp_size, ragged, request):
    """Strict ``cute`` mode at the Kimi head geometry (K == V == 128): the CuTe CP pre-processing
    must launch in both directions, never fall back, and match FLA's own CP path on the output
    and all seven gradients."""
    device = _cutedsl_cp_device(cp_size)
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import chunk_kda
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_cp_cute

    cp_context, rank = _build_kda_cp_context(device, cp_size, ragged=ragged, request=request)
    inputs, grad_output = _make_kda_inputs(device, head_dim=_HEAD_DIM, seed=1234 + rank)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    monkeypatch.setenv("MCORE_GDN_CP_CUTEDSL", "0")
    reference = _kda_forward_backward(chunk_kda, inputs, grad_output, cp_context=cp_context)

    forward_before = fused_gdr_cp_cute.get_cutedsl_fused_launch_count()
    backward_before = fused_gdr_cp_cute.get_cutedsl_fused_bwd_launch_count()
    fused_gdr_cp_cute.reset_cutedsl_fallback_reasons()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    monkeypatch.setenv("MCORE_GDN_CP_CUTEDSL", "1")
    actual = _kda_forward_backward(chunk_kda, inputs, grad_output, cp_context=cp_context)
    # One CuTe boundary launch per direction. Strict mode raises on a fallback; the empty
    # fallback-reason map is the second witness that the dispatcher never declined.
    assert fused_gdr_cp_cute.get_cutedsl_fused_launch_count() == forward_before + 1
    assert fused_gdr_cp_cute.get_cutedsl_fused_bwd_launch_count() == backward_before + 1
    assert fused_gdr_cp_cute.get_cutedsl_fallback_reasons() == {}

    torch.testing.assert_close(actual[0], reference[0], atol=1e-2, rtol=1e-2)
    for name, got, expected in zip(
        _KDA_NAMES, (actual[0], *actual[1]), (reference[0], *reference[1])
    ):
        assert torch.isfinite(got).all(), f"{name} is not finite"
        relative_l2 = _relative_l2(got, expected)
        assert (
            relative_l2 <= _KDA_REL_L2_LIMITS[name]
        ), f"{name}: relative L2 {relative_l2:.3e} > {_KDA_REL_L2_LIMITS[name]:.1e} vs FLA"


@pytest.mark.parametrize("cp_size", [2, 4], ids=["cp2", "cp4"])
def test_internal_kda_head_dim_outside_cute_envelope_is_not_silent(monkeypatch, cp_size, request):
    """Head dims other than 128 cannot pass as CuTe coverage: ``cute`` mode raises with the
    dispatcher's reason, and ``auto`` mode records the fallback while matching FLA's output.

    Only the output is compared: with 64-dim heads FLA's own CP backward is not run-to-run
    reproducible on GB300 (fla-core 0.5.1 and 0.5.2 alike, two FLA runs differ by ~10 % relative
    L2 on dk/dv/dg/dbeta while the output and dq are bit-identical), so gradient closeness says
    nothing about the adapter here; the 128-dim tests above cover the gradients.
    """
    device = _cutedsl_cp_device(cp_size)
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend import chunk_kda
    from megatron.core.ssm.gated_delta_net.internal_gdn_backend.kernels import fused_gdr_cp_cute

    cp_context, rank = _build_kda_cp_context(device, cp_size, ragged=False, request=request)
    inputs, grad_output = _make_kda_inputs(device, head_dim=64, seed=1234 + rank)

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "cute")
    monkeypatch.setenv("MCORE_GDN_CP_CUTEDSL", "1")
    with pytest.raises(RuntimeError, match=r"K=64,V=64 \(need K==V==128\)"):
        _kda_forward_backward(chunk_kda, inputs, grad_output, cp_context=cp_context)

    forward_before = fused_gdr_cp_cute.get_cutedsl_fused_launch_count()
    fused_gdr_cp_cute.reset_cutedsl_fallback_reasons()
    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "auto")
    auto = _kda_forward_backward(chunk_kda, inputs, grad_output, cp_context=cp_context)
    assert fused_gdr_cp_cute.get_cutedsl_fused_launch_count() == forward_before
    assert "K=64,V=64 (need K==V==128)" in fused_gdr_cp_cute.get_cutedsl_fallback_reasons()

    monkeypatch.setenv("MCORE_GDN_INTERNAL_BACKEND", "fla")
    monkeypatch.setenv("MCORE_GDN_CP_CUTEDSL", "0")
    reference = _kda_forward_backward(chunk_kda, inputs, grad_output, cp_context=cp_context)
    torch.testing.assert_close(auto[0], reference[0], atol=1e-2, rtol=1e-2)
    for name, got in zip(_KDA_NAMES[1:], auto[1]):
        assert torch.isfinite(got).all(), f"{name} is not finite"
