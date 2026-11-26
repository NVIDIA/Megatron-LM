# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay coverage for the cuDNN Kimi Delta Attention backend."""

import math
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

from megatron.core import parallel_state
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils


def _has_cudnn_kda() -> bool:
    try:
        return Version(version("nvidia-cudnn-frontend")) >= Version("1.32.0")
    except PackageNotFoundError:
        return False


pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 10
        or not _has_cudnn_kda(),
        reason="cuDNN KDA requires nvidia-cudnn-frontend >= 1.32.0 and SM100+",
    ),
]


@pytest.mark.parametrize(
    ("a_log", "lower_bound", "message"),
    [(None, -5.0, "requires A_log"), ("present", None, "requires lower_bound")],
)
def test_cudnn_kda_rejects_incomplete_safe_gate(a_log, lower_bound, message):
    """Reject incomplete safe-gate inputs before dispatching a cuDNN kernel."""
    from megatron.core.ops.ssm.kda.cudnn_backend import _prepare_inputs

    shape = (1, 2, 1, 64)
    q, k, v, g = (torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    beta = torch.zeros(shape[:3], device="cuda", dtype=torch.bfloat16)
    if a_log is not None:
        a_log = torch.zeros(shape[2], device="cuda")

    with pytest.raises(ValueError, match=message):
        _prepare_inputs(
            q,
            k,
            v,
            g,
            beta,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=lower_bound,
            a_log=a_log,
            dt_bias=None,
        )


def test_cudnn_kda_rejects_unknown_keyword_arguments():
    """Do not silently ignore FLA call options that the cuDNN adapter cannot honor."""
    from megatron.core.ops.ssm.kda.cudnn_backend import cudnn_chunk_kda

    shape = (1, 2, 1, 64)
    q, k, v, g = (torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    beta = torch.zeros(shape[:3], device="cuda", dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="unsupported_option"):
        cudnn_chunk_kda(q, k, v, g, beta, unsupported_option=True)


@pytest.mark.parametrize("gate_mode", ["precomputed", "softplus", "bounded"])
def test_cudnn_kda_gate_precision(gate_mode):
    """Sub-BF16 differences in gate inputs and parameters survive THD preparation."""
    from megatron.core.ops.ssm.kda.cudnn_backend import _prepare_inputs

    shape = (1, 4, 2, 64)
    q = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    g = torch.full(shape, 1 + 2**-9, device=q.device, requires_grad=True)
    beta = torch.full(shape[:3], 0.5 + 2**-10, device=q.device)
    a_log = torch.full((2,), 0.5 + 2**-10, device=q.device, requires_grad=True)
    bias = torch.full((128,), -1 + 2**-9, device=q.device, requires_grad=True)
    tensors, actual_beta, actual_log, actual_bias = _prepare_inputs(
        q,
        q,
        q,
        g,
        beta,
        use_gate_in_kernel=gate_mode != "precomputed",
        safe_gate=gate_mode == "bounded",
        lower_bound=-5.0,
        a_log=a_log,
        dt_bias=bias,
    )
    torch.testing.assert_close(actual_beta, beta.flatten(0, 1), rtol=0, atol=0)
    if gate_mode == "softplus":
        # Independent, higher-precision evaluation of the decay and its VJP.
        rd, ra, rb = (x.detach().double().requires_grad_() for x in (g, a_log, bias))
        ref = -ra.exp().view(1, 1, 2, 1) * torch.logaddexp(rd + rb.view(2, 64), rd.new_zeros(()))
        torch.testing.assert_close(tensors[3], ref.float().flatten(0, 1), rtol=2e-6, atol=0)
        actual_grads = torch.autograd.grad(tensors[3].sum(), (g, a_log, bias))
        ref_grads = torch.autograd.grad(ref.sum(), (rd, ra, rb))
        for actual, expected in zip(actual_grads, ref_grads):
            torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=0)
    else:
        torch.testing.assert_close(tensors[3], g.flatten(0, 1), rtol=0, atol=0)
    if gate_mode == "bounded":
        torch.testing.assert_close(actual_log, a_log, rtol=0, atol=0)
        torch.testing.assert_close(actual_bias, bias.view(2, 64), rtol=0, atol=0)
    else:
        assert actual_log is None and actual_bias is None


@pytest.mark.skipif(Utils.world_size < 4, reason="Requires 4 GPUs")
def test_cudnn_kda_fp32_boundary_exchange():
    """BF16 activations carry FP32 states and adjoints across multiple FROST pieces."""
    from megatron.core.ops.ssm.kda.cudnn_backend import _CPBoundary, _exchange_boundary

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=4
    )
    try:
        group = parallel_state.get_context_parallel_group()
        rank = torch.distributed.get_rank(group)
        value = torch.zeros((128, 2, 64), device="cuda", dtype=torch.bfloat16)
        # This value and its first three multiples distinguish FP32 from a BF16 round trip.
        h = torch.full((1, 2, 64, 64), 1 + 2**-9, device=value.device, dtype=torch.float32)
        matrix = torch.eye(64, device=value.device).expand_as(h).contiguous()
        # A single sequence across CP4, then two sequences each split across two adjacent ranks.
        for pre, post in ((rank, 3 - rank), (rank % 2, 1 - rank % 2)):
            cp = _CPBoundary(group, rank, pre, post, 128, 0)
            for reverse in (False, True):
                sends = pre if reverse else post
                receives = post if reverse else pre
                summary = (h, matrix) if sends else None
                state = _exchange_boundary(summary, value, 64, 2, cp, reverse=reverse)
                expected = torch.zeros((2, 2, 64, 64), device=value.device, dtype=torch.float32)
                expected[-1 if reverse else 0] = receives * h[0]
                torch.testing.assert_close(state, expected, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "cp_size",
    [1, pytest.param(4, marks=pytest.mark.skipif(Utils.world_size < 4, reason="Requires 4 GPUs"))],
)
def test_cudnn_kda_replays_bit_exact(cp_size):
    """Replay local KDA and its distributed FROST boundary-state composition."""
    from megatron.core.ops.ssm.kda.cudnn_backend import cudnn_chunk_kda

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    try:
        seeded()
        device = torch.device("cuda", torch.cuda.current_device())
        tokens, heads, head_dim = 512, 8, 128
        shape = (1, tokens, heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
        beta = torch.rand(
            (1, tokens, heads), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        a_log = torch.empty(heads, device=device).uniform_(0.0, math.log(16.0))
        a_log.requires_grad_(True)
        dt_bias = torch.empty(heads * head_dim, device=device).uniform_(-6.0, -2.0)
        dt_bias.requires_grad_(True)
        cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)

        if cp_size == 1:
            cp_context = None
        else:
            cp_group = parallel_state.get_context_parallel_group()
            cp_rank = torch.distributed.get_rank(group=cp_group)
            cp_context = SimpleNamespace(
                group=cp_group,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens.cpu(),
                pre_num_ranks=cp_rank,
                post_num_ranks=cp_size - cp_rank - 1,
            )

        def run(q, k, v, g, beta, a_log, dt_bias):
            output, _ = cudnn_chunk_kda(
                q,
                k,
                v,
                g,
                beta,
                scale=1.0 / math.sqrt(head_dim),
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
                safe_gate=True,
                lower_bound=-5.0,
                cp_context=cp_context,
                A_log=a_log,
                dt_bias=dt_bias,
            )
            return output

        grad_output = torch.randn(shape, device=device, dtype=torch.bfloat16)
        assert_replays_bit_exact(
            run,
            (q, k, v, g, beta, a_log, dt_bias),
            replays=3,
            grad_outputs={"out": grad_output},
            contention=True,
            what=f"cuDNN KDA CP{cp_size}",
        )
    finally:
        Utils.destroy_model_parallel()
