# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the in-repo SSM Triton kernels (``megatron/core/ssm/ops/``) and the
gated-delta-net code paths.

The Mamba training mixer is covered by ``correctness/test_ssm_conv1d.py``; this module drives
the kernels Megatron ships itself: the Mamba2 varlen chunked scan (``ssd_*`` kernels behind
``mamba_chunk_scan_combined_varlen``), the decode-time ``selective_state_update`` and
``causal_conv1d_update``, the varlen causal conv, the Gated Delta Product varlen chunk scan
(``gdp/*`` kernels) and its decode kernels -- single-token and speculative, where a step
carries several draft tokens and the kernels snapshot the conv and recurrent state after
each of them -- and the torch gated-delta-rule path that ``--deterministic-mode`` selects
instead of FLA.

Autotuning is the mechanism that can break replay here: ``ops/common/determinism.py`` pins a
single config and a zero-initialised, ordered-sum workspace when ``MAMBA_DETERMINISTIC`` (or
``torch.use_deterministic_algorithms``) is on. Each kernel is replayed in that mode.
"""

import pytest
import torch

from megatron.core.ssm.ops.common import determinism as ssm_determinism
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)


@pytest.fixture(autouse=True)
def ssm_deterministic():
    """Pin the SSM autotuner / workspace path like ``--deterministic-mode`` does."""
    ssm_determinism.set_deterministic_mode(True)
    yield
    ssm_determinism.set_deterministic_mode(None)


# --- Mamba2 --------------------------------------------------------------------------------


def test_mamba_chunk_scan_combined_varlen_replays():
    from megatron.core.ssm.ops.mamba2.ssd_combined import mamba_chunk_scan_combined_varlen

    seeded()
    seqlen, nheads, headdim, ngroups, dstate, chunk = 4096, 32, 64, 1, 128, 256
    nchunks = seqlen // chunk
    x = torch.randn(seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    dt = torch.rand(seqlen, nheads, device="cuda") * 0.1
    A = -torch.rand(nheads, device="cuda") - 1.0
    B = torch.randn(seqlen, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
    C = torch.randn(seqlen, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(nheads, device="cuda")
    dt_bias = torch.rand(nheads, device="cuda")
    cu_chunk_seqlens = torch.arange(0, seqlen + 1, chunk, dtype=torch.int32, device="cuda")
    # Two sequences of eight chunks each.
    last_chunk_indices = torch.tensor(
        [nchunks // 2 - 1, nchunks - 1], dtype=torch.int64, device="cuda"
    )
    seq_idx = torch.repeat_interleave(
        torch.arange(2, dtype=torch.int32, device="cuda"), nchunks // 2
    )
    out = torch.empty_like(x)

    def fn(x, dt, A, B, C, D, dt_bias, out):
        states = mamba_chunk_scan_combined_varlen(
            x,
            dt,
            A,
            B,
            C,
            chunk,
            cu_chunk_seqlens,
            last_chunk_indices,
            seq_idx,
            out,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        return out, states

    assert_replays_bit_exact(
        fn,
        (x, dt, A, B, C, D, dt_bias, out),
        replays=4,
        backward=False,
        what="mamba_chunk_scan_combined_varlen",
    )


def test_selective_state_update_replays():
    from megatron.core.ssm.ops.mamba2.mamba_ssm import selective_state_update

    seeded()
    batch, nheads, headdim, dstate, ngroups = 256, 32, 64, 128, 1
    state = torch.randn(batch, nheads, headdim, dstate, device="cuda")
    x = torch.randn(batch, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    dt = torch.rand(batch, nheads, device="cuda").unsqueeze(-1).expand(batch, nheads, headdim)
    A = (
        (-torch.rand(nheads, device="cuda") - 1.0)
        .view(nheads, 1, 1)
        .expand(nheads, headdim, dstate)
    )
    Bm = torch.randn(batch, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
    Cm = torch.randn(batch, ngroups, dstate, device="cuda", dtype=torch.bfloat16)
    D = torch.randn(nheads, device="cuda").unsqueeze(-1).expand(nheads, headdim)
    dt_bias = (torch.rand(nheads, device="cuda") - 4.0).unsqueeze(-1).expand(nheads, headdim)

    def fn(state, x, dt, A, Bm, Cm, D, dt_bias):
        # The kernel picks its TIE_HDIM specialisation from these stride-0 broadcasts, the
        # layout the Mamba mixer dispatches; the harness must hand them over unchanged.
        assert dt.stride(-1) == 0 and A.stride(-1) == 0 and dt_bias.stride(-1) == 0
        out = selective_state_update(
            state, x, dt, A, Bm, Cm, D=D, dt_bias=dt_bias, dt_softplus=True
        )
        return out, state

    assert_replays_bit_exact(
        fn,
        (state, x, dt, A, Bm, Cm, D, dt_bias),
        replays=4,
        backward=False,
        what="selective_state_update",
    )


# --- causal conv (Megatron Triton variants) ---------------------------------------------------


def test_causal_conv1d_update_replays():
    from megatron.core.ssm.ops.common.causal_conv1d_triton import causal_conv1d_update

    seeded()
    batch, dim, width = 256, 4096, 4
    x = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.randn(batch, dim, width - 1, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)

    def fn(x, conv_state, weight, bias):
        out = causal_conv1d_update(x, conv_state, weight, bias, True, None)
        return out, conv_state

    assert_replays_bit_exact(
        fn, (x, conv_state, weight, bias), replays=4, backward=False, what="causal_conv1d_update"
    )


def test_causal_conv1d_update_spec_decode_replays():
    """Speculative decode: several tokens per request, with the rollback snapshots.

    ``intermediate_conv_states`` receives the conv state after each draft token so
    that verification can roll a request back to its accepted prefix. GDP caches
    ``d_conv`` positions per request (``state_len == width``), the branch that
    writes the snapshot from the shifted registers rather than re-reading HBM --
    the single-token test above covers the ``state_len > width`` branch.
    """
    from megatron.core.ssm.ops.common.causal_conv1d_triton import causal_conv1d_update

    seeded()
    batch, seq_len, dim, width = 256, 4, 4096, 4
    slots = batch + 8
    x = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.randn(slots, dim, width, device="cuda", dtype=torch.bfloat16)
    intermediate = torch.zeros(slots, seq_len, dim, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    # Dynamic batching hands out shuffled slots; -1 marks a padding request, whose
    # output is zeroed and whose state and snapshots are left untouched.
    state_indices = torch.randperm(slots, device="cuda")[:batch].to(torch.int32)
    state_indices[::16] = -1

    def fn(x, conv_state, intermediate, weight, bias, state_indices):
        out = causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            "silu",
            state_indices,
            intermediate_conv_states=intermediate,
        )
        return out, conv_state, intermediate

    assert_replays_bit_exact(
        fn,
        (x, conv_state, intermediate, weight, bias, state_indices),
        replays=4,
        backward=False,
        contention=True,
        what="causal_conv1d_update (speculative decode)",
    )


def test_causal_conv1d_varlen_replays():
    from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_fn

    seeded()
    dim, width = 2048, 4
    bounds = [0, 1000, 1500, 2048, 4096, 8192]
    x = torch.randn(bounds[-1], dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    initial_states = torch.randn(
        len(bounds) - 1, dim, width - 1, device="cuda", dtype=torch.bfloat16
    )
    assert_replays_bit_exact(
        lambda x, w, b, s: causal_conv1d_varlen_fn(x, w, b, cu_seqlens, s),
        (x, weight, bias, initial_states),
        replays=4,
        backward=False,
        what="causal_conv1d_varlen_fn",
    )


# --- Gated Delta Product ----------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="GDP varlen chunk scan replays differ on GB300 / Triton 3.7 even with the autotune "
    "pinned to one config and the ordered workspace (measured 2026-09-04: ~0.15% of output "
    "elements, max |diff| 3.6e-2 bf16; final state ~0.13%, 2.9e-4). Recorded for the hybrid "
    "model owners; not gated until root-caused.",
)
def test_chunk_gated_delta_product_varlen_replays():
    from megatron.core.ssm.ops.gdp import chunk_h, chunk_o
    from megatron.core.ssm.ops.gdp.chunk import chunk_gated_delta_product_varlen

    # The deterministic policy must have pinned the autotuners at import; otherwise the
    # replay would measure Triton's timing-based config search, not the kernels.
    for kernel in (
        chunk_h.chunk_gated_delta_product_fwd_kernel_h_blockdim64,
        chunk_o.chunk_fwd_kernel_o,
    ):
        configs = getattr(kernel, "configs", None)
        if configs is not None and len(configs) != 1:
            pytest.skip("GDP autotuners were imported before the deterministic policy was set")

    seeded()
    T, H, K, V, M = 4096, 16, 128, 128, 2
    bounds = [0, 1000, 2500, 4096]
    q = torch.randn(1, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, T * M, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, T * M, H, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device="cuda") * 0.1
    beta = torch.rand(1, T * M, H, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    initial_state = torch.randn(len(bounds) - 1, H, K, V, device="cuda")

    def fn(q, k, v, g, beta, initial_state):
        return chunk_gated_delta_product_varlen(
            q,
            k,
            v,
            g,
            beta,
            M,
            cu_seqlens,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    assert_replays_bit_exact(
        fn,
        (q, k, v, g, beta, initial_state),
        replays=4,
        backward=False,
        what="chunk_gated_delta_product_varlen",
    )


def test_fused_recurrent_gated_delta_rule_update_replays():
    from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

    seeded()
    B, H, K, V = 128, 16, 128, 128
    q = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, 1, H, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(B, 1, H, device="cuda") * 0.1
    beta = torch.rand(B, 1, H, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(B, H, K, V, device="cuda")

    def fn(q, k, v, g, beta, initial_state):
        return fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    assert_replays_bit_exact(
        fn,
        (q, k, v, g, beta, initial_state),
        replays=4,
        backward=False,
        what="fused_recurrent_gated_delta_rule_update",
    )


def test_fused_recurrent_gated_delta_rule_update_spec_decode_replays():
    """Speculative decode: ``S`` draft tokens of ``M`` Householder steps each.

    Exercises the slot-indexed state cache and ``intermediate_states``, which the
    kernel writes on the last step of every draft token so that verification can
    roll the recurrence back to an accepted prefix. The snapshot buffer is an
    output here, so its contents are compared across replays like any other.
    """
    from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

    seeded()
    B, S, M, H, K, V = 64, 4, 2, 8, 128, 128
    T = S * M
    slots = B + 8
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(B, T, H, device="cuda") * 0.1
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(slots, H, K, V, device="cuda")
    intermediate = torch.zeros(slots, S, H, K, V, device="cuda")
    # Dynamic batching hands out shuffled slots; -1 marks a padding request, whose
    # output is zeroed and whose state and snapshots are left untouched.
    state_indices = torch.randperm(slots, device="cuda")[:B].to(torch.int32)
    state_indices[::16] = -1

    def fn(q, k, v, g, beta, state, intermediate, state_indices):
        o, final_state = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            state=state,
            state_indices=state_indices,
            intermediate_states=intermediate,
            steps_per_token=M,
        )
        return o, final_state, intermediate

    assert_replays_bit_exact(
        fn,
        (q, k, v, g, beta, state, intermediate, state_indices),
        replays=4,
        backward=False,
        contention=True,
        what="fused_recurrent_gated_delta_rule_update (speculative decode)",
    )


@pytest.mark.parametrize("S", [1, 4])
def test_gdp_decode_prepare_replays(S):
    """``S`` is one plus the speculative draft length; ``S == 1`` is plain decode."""
    from megatron.core.ssm.ops.gdp.decode_prepare import gdp_decode_prepare

    seeded()
    n, M, H, G, P, N = 256, 2, 16, 4, 128, 128
    x = torch.randn(n, S, M * H * P + M * G * N + G * N, device="cuda", dtype=torch.bfloat16)
    # `ba` reaches the kernel as a slice of the input projection, so only its last
    # dimension is contiguous and the token stride comes from the view, not the width.
    ba = torch.randn(n, S, 2 * (M * H + H), device="cuda", dtype=torch.bfloat16)[..., : M * H + H]
    A_log = torch.randn(H, device="cuda")
    dt_bias = torch.randn(H, device="cuda")
    assert_replays_bit_exact(
        lambda x, ba, A_log, dt_bias: gdp_decode_prepare(x, ba, A_log, dt_bias, M, H, G, P, N),
        (x, ba, A_log, dt_bias),
        replays=4,
        backward=False,
        what=f"gdp_decode_prepare (S={S})",
    )


def test_gdp_l2norm_and_cumsum_replay():
    from megatron.core.ssm.ops.gdp.common import l2norm_fwd
    from megatron.core.ssm.ops.gdp.cumsum import chunk_local_cumsum

    seeded()
    x = torch.randn(65536, 128, device="cuda", dtype=torch.bfloat16)
    assert_replays_bit_exact(l2norm_fwd, (x,), replays=4, backward=False, what="l2norm_fwd")
    g = torch.randn(1, 8192, 16, device="cuda")
    assert_replays_bit_exact(
        lambda g: chunk_local_cumsum(g, 64),
        (g,),
        replays=4,
        backward=False,
        what="chunk_local_cumsum",
    )


# --- Gated Delta Net (deterministic torch path vs FLA) ---------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_gdn_torch_l2norm_matches_fla_and_replays(dtype, head_dim, compiled):
    """Preserve FLA's output/gradient contract while removing autotune-dependent rounding."""
    from fla.modules.l2norm import l2norm

    from megatron.core.ssm.gated_delta_net.common import torch_l2norm

    seeded()
    x = torch.randn(2, 256, head_dim, device="cuda", dtype=dtype)
    # A zero row distinguishes additive epsilon from clamping the norm itself.
    x[0, 0].zero_()
    dy = torch.randn_like(x)
    if dtype == torch.bfloat16:
        # With an exactly parallel upstream gradient, differentiating an
        # unrounded norm would erase FLA's saved-rounded-output contribution.
        x[0, 1].fill_(0.5)
        dy[0, 1].fill_(32.0)
    actual_input = x.clone().requires_grad_(True)
    expected_input = x.clone().requires_grad_(True)
    normalize = torch.compile(torch_l2norm) if compiled else torch_l2norm
    actual = normalize(actual_input)
    expected = l2norm(expected_input)
    actual.backward(dy)
    expected.backward(dy)
    if dtype == torch.bfloat16:
        assert expected_input.grad[0, 1].abs().min().item() > 1e-4
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    assert_replays_bit_exact(
        normalize, (x.requires_grad_(True),), replays=3, contention=True, what="GDN torch L2 norm"
    )


def test_torch_chunk_gated_delta_rule_replays_fwd_bwd():
    """The path ``--deterministic-mode`` selects for GDN (FLA is not deterministic)."""
    from megatron.core.ssm.gated_delta_net.gdn import torch_chunk_gated_delta_rule

    seeded()
    B, T, H, K = 2, 2048, 16, 128
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = (-torch.rand(B, T, H, device="cuda") * 0.1).requires_grad_(True)
    beta = torch.rand(B, T, H, device="cuda").requires_grad_(True)

    def fn(q, k, v, g, beta):
        # use_qk_l2norm_in_kernel routes through FLA's l2norm, whose signature differs across
        # FLA releases; normalise outside so the test only depends on the torch path.
        o, state = torch_chunk_gated_delta_rule(
            torch.nn.functional.normalize(q, dim=-1),
            torch.nn.functional.normalize(k, dim=-1),
            v,
            g,
            beta,
            chunk_size=64,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        return o, state

    assert_replays_bit_exact(fn, (q, k, v, g, beta), replays=3, what="torch_chunk_gated_delta_rule")


@pytest.mark.xfail(
    strict=False, reason="FLA chunk_gated_delta_rule is documented non-deterministic; recorded"
)
def test_fla_chunk_gated_delta_rule_replays_fwd_bwd():
    fla_ops = pytest.importorskip("fla.ops.gated_delta_rule")
    seeded()
    B, T, H, K = 2, 2048, 16, 128
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = (-torch.rand(B, T, H, device="cuda") * 0.1).requires_grad_(True)
    beta = torch.rand(B, T, H, device="cuda").requires_grad_(True)

    def fn(q, k, v, g, beta):
        o, state = fla_ops.chunk_gated_delta_rule(
            q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        return o, state

    assert_replays_bit_exact(fn, (q, k, v, g, beta), replays=4, what="fla chunk_gated_delta_rule")
