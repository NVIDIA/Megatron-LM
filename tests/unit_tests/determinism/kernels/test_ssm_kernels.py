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

Autotuning is the mechanism that can break replay here: ``megatron.core.tuning`` pins a
single config, while ``ops/common/determinism.py`` selects a zero-initialised, ordered-sum
workspace when the explicit deterministic policy (or ``torch.use_deterministic_algorithms``) is on.
Each kernel is replayed in that mode, including with Triton's autotune cache enabled.
"""

import pytest
import torch

from megatron.core.ssm import mamba_mixer as mamba_mixer_module
from megatron.core.ssm.ops.common import determinism as ssm_determinism
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tuning import autotune_configs, choice_log
from megatron.core.tuning import policy as tuning_policy
from tests.unit_tests.determinism.correctness.test_ssm_conv1d import (
    _build_mixer,
    requires_deterministic_conv1d,
)
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)


@pytest.fixture(autouse=True)
def ssm_deterministic(monkeypatch):
    """Pin the SSM autotuner / workspace path like ``--deterministic-mode`` does."""
    monkeypatch.setattr(tuning_policy, "_deterministic_override", True)


def _autotuner(kernel):
    """The ``Autotuner`` under ``kernel``; ``@triton.heuristics`` wraps it in another object."""
    from triton.runtime.autotuner import Autotuner

    while not isinstance(kernel, Autotuner):
        kernel = kernel.fn
    return kernel


def _forbid_benchmarks(monkeypatch, *tuners, cache_results=False):
    """Fail if any tuner times a config, including through Triton's autotune disk cache."""

    def forbid_benchmark(*args, **kwargs):
        raise AssertionError("deterministic SSM execution benchmarked a config")

    for tuner in tuners:
        # Training imports these kernels before determinism is enabled, so they keep every
        # candidate and only the runtime adapter stands between them and timed selection.
        assert len(tuner.configs) > 1, "expected the full candidate list at import"
        monkeypatch.setattr(tuner, "_bench", forbid_benchmark)
        # Triton < 3.4 has no autotune disk cache.
        monkeypatch.setattr(tuner, "cache_results", cache_results, raising=False)


def _assert_pinned(*tuners):
    for tuner in tuners:
        name = tuner.base_fn.__name__
        choices = [value for key, value in choice_log().items() if f".{name}|" in key]
        assert choices and all(value.endswith(";pinned") for value in choices), choices


# --- Mamba2 --------------------------------------------------------------------------------


@pytest.mark.parametrize("mixer_type", ["mamba", "gdp"])
def test_mixer_rmsnorm_gamma_replays(mixer_type):
    """Actual mixer constructors mark gamma without changing gated norm forward/backward."""
    if mamba_mixer_module.mamba_chunk_scan_combined is None:
        pytest.skip("needs mamba_ssm")
    if mixer_type == "gdp":
        from tests.unit_tests.ssm.test_gated_delta_product import HAVE_GDP_DEPS
        from tests.unit_tests.ssm.test_gated_delta_product import _build_mixer as build_gdp
        from tests.unit_tests.ssm.test_gated_delta_product import _make_config

        if not HAVE_GDP_DEPS:
            pytest.skip("needs GDP dependencies")

    seeded()
    try:
        if mixer_type == "mamba":
            mixer = _build_mixer(deterministic_mode=False)
        else:
            Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
            model_parallel_cuda_manual_seed(123)
            mixer = build_gdp(_make_config([]))
        norm = mixer.norm
        assert norm.weight.is_rmsnorm_weight
        x = torch.randn(
            4096, 2, norm.weight.numel(), device="cuda", dtype=norm.weight.dtype, requires_grad=True
        )
        gate = torch.randn_like(x, requires_grad=True)
        assert_module_replays_bit_exact(
            norm, (x, gate), replays=3, contention=True, what=f"{mixer_type} RMSNorm gamma"
        )
    finally:
        Utils.destroy_model_parallel()


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


def test_batch_invariant_decode_step_replays():
    """One buffered batch-invariant decode step (chunk scan in target-rows mode, gated by the
    current token's z) replays bit-exact, including a slot that crosses a chunk boundary."""
    from megatron.core.ssm.ops.mamba2.batch_invariant_decode import (
        BatchInvariantDecodeBuffers,
        batch_invariant_decode_buffered_scan,
    )

    seeded()
    slots, nheads, headdim, ngroups, dstate, chunk = 8, 16, 64, 2, 128, 128
    bufs = BatchInvariantDecodeBuffers.allocate(
        slots, chunk, nheads, headdim, ngroups, dstate, torch.device("cuda"), torch.float32
    )
    rows = (bufs.x, bufs.dt, bufs.B, bufs.C)
    for t in rows:
        t.normal_()
    num_buffered = torch.randint(0, chunk, (slots,), dtype=torch.int32, device="cuda")
    num_buffered[0] = chunk - 1  # this step fills slot 0's chunk
    batch_indices = torch.arange(slots, dtype=torch.int32, device="cuda")
    step = [
        torch.randn(slots, 1, *shape, device="cuda")
        for shape in ((nheads, headdim), (nheads, headdim), (nheads,), (ngroups, dstate))
    ]
    C = torch.randn(slots, 1, ngroups, dstate, device="cuda")
    A = -torch.rand(nheads, device="cuda") - 0.5
    D = torch.randn(nheads, device="cuda")
    dt_bias = torch.rand(nheads, device="cuda")
    ssm_state = torch.randn(slots, nheads, headdim, dstate, device="cuda")

    def fn(x_buf, dt_buf, B_buf, C_buf, num_buffered, x, z, dt, B, C, ssm_state):
        buffers = BatchInvariantDecodeBuffers(
            x=x_buf,
            dt=dt_buf,
            B=B_buf,
            C=C_buf,
            num_buffered=num_buffered,
            out=torch.empty(slots, nheads, headdim, device="cuda"),
            target_rows=torch.empty(slots, dtype=torch.int32, device="cuda"),
            chunk_flags=torch.empty(slots, dtype=torch.int32, device="cuda"),
            chunk_starts=torch.empty(slots, dtype=torch.int32, device="cuda"),
        )
        y = batch_invariant_decode_buffered_scan(
            buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
        return y, ssm_state, num_buffered

    assert_replays_bit_exact(
        fn,
        (*rows, num_buffered, *step, C, ssm_state),
        replays=4,
        backward=False,
        what="batch_invariant_decode_buffered_scan",
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


@pytest.mark.parametrize("cache_autotuning", [False, True], ids=["no_cache", "autotune_cache"])
def test_causal_conv1d_varlen_replays(monkeypatch, cache_autotuning):
    from megatron.core.ssm.ops.common import causal_conv1d_varlen

    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "1" if cache_autotuning else "0")
    assert ssm_determinism.autotune_configs is autotune_configs
    assert ssm_determinism.use_deterministic_mode()
    # Kernels decorated under deterministic mode are reduced to one config at import,
    # whatever the autotune cache setting.
    decorated = autotune_configs(
        [
            triton.Config({"BLOCK_T": 8, "BLOCK_C": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_T": 128, "BLOCK_C": 128}, num_warps=4),
        ]
    )
    assert len(decorated) == 1

    # Replay the real module-level autotuner, which keeps every candidate because it was
    # imported before determinism was enabled, as in training.
    tuner = _autotuner(causal_conv1d_varlen._causal_conv1d_varlen_kernel)
    _forbid_benchmarks(monkeypatch, tuner, cache_results=cache_autotuning)

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
        lambda x, w, b, s: causal_conv1d_varlen.causal_conv1d_varlen_fn(x, w, b, cu_seqlens, s),
        (x, weight, bias, initial_states),
        replays=4,
        backward=False,
        what="causal_conv1d_varlen_fn",
    )
    _assert_pinned(tuner)


# --- Gated Delta Product ----------------------------------------------------------------------


def _gdp_case(M=2):
    from megatron.core.ssm.ops.gdp.chunk import chunk_gated_delta_product_varlen

    seeded()
    T, H, K, V = 4096, 16, 128, 128
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

    return fn, (q, k, v, g, beta, initial_state)


def test_chunk_gated_delta_product_varlen_is_pinned(monkeypatch):
    """Runtime pinning covers the GDP autotuners, which training imports before determinism."""
    from megatron.core.ssm.ops.gdp import chunk_h, chunk_o

    tuners = (
        _autotuner(chunk_h.chunk_gated_delta_product_fwd_kernel_h_blockdim64),
        _autotuner(chunk_o.chunk_fwd_kernel_o),
    )
    _forbid_benchmarks(monkeypatch, *tuners)
    fn, args = _gdp_case()
    fn(*args)
    torch.cuda.synchronize()
    _assert_pinned(*tuners)


@pytest.mark.xfail(
    strict=False,
    raises=AssertionError,
    reason="GDP varlen chunk scan replays differ on GB300 / Triton 3.7 even with the autotune "
    "pinned to one config and the ordered workspace (measured 2026-09-04: ~0.15% of output "
    "elements, max |diff| 3.6e-2 bf16; final state ~0.13%, 2.9e-4). Recorded for the hybrid "
    "model owners; not gated until root-caused.",
)
@pytest.mark.parametrize("M", [1, 2])  # M=1 is GDN's dynamic-inference prefill
def test_chunk_gated_delta_product_varlen_replays(M):
    fn, args = _gdp_case(M)
    assert_replays_bit_exact(
        fn, args, replays=4, backward=False, what=f"chunk_gated_delta_product_varlen (M={M})"
    )


@pytest.mark.parametrize("gate_in_kernel", [False, True])
def test_fused_recurrent_gated_delta_rule_update_replays(gate_in_kernel):
    from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

    seeded()
    B, H, K, V = 128, 16, 128, 128
    q = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, 1, H, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(B, 1, H, device="cuda") * 0.1
    beta = torch.rand(B, 1, H, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(B, H, K, V, device="cuda")
    # The GDN decode form: raw gate and beta logits, with the decay formed in-kernel.
    gate_kwargs = (
        dict(
            A_log=torch.randn(H, device="cuda"),
            dt_bias=torch.randn(H, device="cuda"),
            use_beta_sigmoid_in_kernel=True,
        )
        if gate_in_kernel
        else {}
    )

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
            **gate_kwargs,
        )

    assert_replays_bit_exact(
        fn,
        (q, k, v, g, beta, initial_state),
        replays=4,
        backward=False,
        what="fused_recurrent_gated_delta_rule_update",
    )


@pytest.mark.parametrize("M", [2, 3])
@pytest.mark.parametrize("compact_output", [False, True])
def test_fused_recurrent_gated_delta_rule_update_spec_decode_replays(M, compact_output):
    """Speculative decode: ``S`` draft tokens of ``M`` Householder steps each.

    Exercises the slot-indexed state cache and ``intermediate_states``, which the
    kernel writes on the last step of every draft token so that verification can
    roll the recurrence back to an accepted prefix. The snapshot buffer is an
    output here, so its contents are compared across replays like any other.
    """
    from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

    seeded()
    B, S, H, K, V = 64, 4, 8, 128, 128
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
            compact_output=compact_output,
        )
        return o, final_state, intermediate

    assert_replays_bit_exact(
        fn,
        (q, k, v, g, beta, state, intermediate, state_indices),
        replays=4,
        backward=False,
        contention=True,
        what=f"fused_recurrent_gated_delta_rule_update (speculative decode, compact={compact_output})",
    )


def test_fused_recurrent_gated_delta_rule_update_padding_replays():
    """Single-token decode whose ``-1`` padding rows hold NaN/Inf inputs.

    The kernel zeroes padding rows in its output store, so they must come out as ``+0.0``
    bytes on every replay while real rows and the slot-indexed state cache replay exactly.
    """
    from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

    seeded()
    B, H, K, V = 128, 16, 128, 128
    slots = B + 8
    q = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, 1, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, 1, H, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(B, 1, H, device="cuda") * 0.1
    beta = torch.rand(B, 1, H, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(slots, H, K, V, device="cuda")
    state_indices = torch.randperm(slots, device="cuda")[:B].to(torch.int32)
    state_indices[::16] = -1
    padding = state_indices < 0
    q[padding] = float("nan")
    v[padding] = float("inf")

    def fn(q, k, v, g, beta, state, state_indices):
        return fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            state=state,
            state_indices=state_indices,
        )

    ref_out, _ = assert_replays_bit_exact(
        fn,
        (q, k, v, g, beta, state, state_indices),
        replays=4,
        backward=False,
        contention=True,
        what="fused_recurrent_gated_delta_rule_update (padding rows)",
    )
    padded_out = ref_out["out[0]"][padding]
    assert torch.equal(padded_out, torch.zeros_like(padded_out))
    assert not torch.signbit(padded_out).any()


@pytest.mark.parametrize("S", [1, 4])
@pytest.mark.parametrize("M", [1, 2, 3])
def test_gdp_decode_prepare_replays(S, M):
    """Replay all copy-index paths in plain and speculative decode."""
    from megatron.core.ssm.ops.gdp.prepare import gdp_decode_prepare

    seeded()
    n, H, G, P, N = 256, 16, 4, 128, 128
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
        what=f"gdp_decode_prepare (S={S}, M={M})",
    )


@pytest.mark.parametrize("T", [1, 4096])
def test_gdp_prefill_prepare_replays(T):
    """``T`` is the packed token count of a prefill step."""
    from megatron.core.ssm.ops.gdp.prepare import gdp_prefill_prepare

    seeded()
    M, H, G, P, N = 3, 12, 12, 64, 128
    x = torch.randn(1, T, M * H * P + M * G * N + G * N, device="cuda", dtype=torch.bfloat16)
    # `ba` reaches the kernel as a slice of the input projection, so only its last
    # dimension is contiguous and the token stride comes from the view, not the width.
    ba = torch.randn(1, T, 2 * (M * H + H), device="cuda", dtype=torch.bfloat16)[..., : M * H + H]
    A_log = torch.randn(H, device="cuda")
    dt_bias = torch.randn(H, device="cuda")
    assert_replays_bit_exact(
        lambda x, ba, A_log, dt_bias: gdp_prefill_prepare(x, ba, A_log, dt_bias, M, H, G, P, N),
        (x, ba, A_log, dt_bias),
        replays=4,
        backward=False,
        what=f"gdp_prefill_prepare (T={T})",
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sequence_length", [64, 65])
@pytest.mark.parametrize("with_initial_state", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_torch_chunk_gated_delta_rule_cuda_graph(
    dtype, sequence_length, with_initial_state, output_final_state
):
    """Capture forward/backward and replay changed inputs without a CPU state allocation."""
    from megatron.core.ssm.gated_delta_net.gdn import torch_chunk_gated_delta_rule

    def make_inputs(seed):
        seeded(seed)
        shape = (2, sequence_length, 2, 8)
        q = torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype)
        k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(dtype)
        v = torch.randn(2, sequence_length, 2, 12, device="cuda", dtype=dtype) * 0.1
        g = -torch.rand(2, sequence_length, 2, device="cuda") * 0.1
        beta = torch.rand_like(g).to(dtype)
        values = [q, k, v, g, beta]
        if with_initial_state:
            values.append(torch.randn(2, 2, 8, 12, device="cuda") * 0.1)
        return tuple(value.detach().requires_grad_(True) for value in values)

    def forward_backward(values):
        output, state = torch_chunk_gated_delta_rule(
            *values[:5],
            initial_state=values[5] if with_initial_state else None,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
        )
        outputs = (output,) if state is None else (output, state)
        loss = sum(value.float().sum() for value in outputs)
        return (*outputs, *torch.autograd.grad(loss, values))

    static_inputs = make_inputs(42)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward_backward(static_inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward_backward(static_inputs)

    for seed in (101, 102, 103):
        inputs = make_inputs(seed)
        expected = forward_backward(inputs)
        with torch.no_grad():
            for destination, source in zip(static_inputs, inputs):
                destination.copy_(source)
        graph.replay()
        for actual, reference in zip(captured, expected):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        previous = tuple(value.detach().clone() for value in captured)
        graph.replay()
        for actual, reference in zip(captured, previous):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


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


def test_kda_gate_jit_path_replays_fwd_bwd():
    """KDA's jit-fused gate reshaping and beta activation replay bit-exactly."""
    from types import SimpleNamespace

    from megatron.core.ssm.gated_delta_net.kda import KimiDeltaAttention

    seeded()
    batch, seq_len, heads, head_dim = 2, 256, 4, 32
    dummy = SimpleNamespace(key_head_dim=head_dim)
    A_log = torch.randn(heads, device="cuda")
    dt_bias = torch.randn(heads, device="cuda")
    raw_g = torch.randn(
        batch * seq_len * heads * head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    beta = torch.randn(
        batch * seq_len * heads, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def fn(raw_g, beta):
        raw_g, gates = KimiDeltaAttention._compute_gates(
            dummy, A_log, dt_bias, batch, seq_len, raw_g, beta
        )
        return raw_g, gates["beta"]

    assert_replays_bit_exact(fn, (raw_g, beta), replays=3, what="KDA gate JIT path")


def _record_fla_autotuners(monkeypatch):
    """Collect every multi-candidate FLA autotuner that runs, and fail on any benchmark."""
    from triton.runtime.autotuner import Autotuner

    from megatron.core.tuning import selection

    reached = set()
    run = Autotuner.run

    def record(tuner, *args, **kwargs):
        qualified = f"{selection.kernel_module(tuner)}.{selection.kernel_name(tuner)}"
        if qualified.startswith("fla.") and len(tuner.configs) > 1:
            reached.add(qualified)
        return run(tuner, *args, **kwargs)

    def forbid_benchmark(*args, **kwargs):
        raise AssertionError("deterministic execution benchmarked an FLA config")

    monkeypatch.setattr(Autotuner, "run", record)
    monkeypatch.setattr(Autotuner, "_bench", forbid_benchmark)
    return reached


def _assert_fla_autotuners_pinned(reached, expected):
    """``expected`` kernels ran through Triton's autotuner, and every FLA autotuner was pinned.

    Kernels are matched by function name, which FLA keeps when it moves modules (0.6 moved the
    norms to ``fla.modules.norm``). A kernel that Dynamo traced into a compiled graph never
    reaches the autotuner, since Inductor times its candidates itself, so it is missing here.
    """
    assert expected <= {qualified.rsplit(".", 1)[1] for qualified in reached}, sorted(reached)
    for qualified in sorted(reached):
        choices = [value for key, value in choice_log().items() if f"|{qualified}|" in key]
        assert choices and all(value.endswith(";pinned") for value in choices), qualified


def _record_tilelang_kernel(monkeypatch, package, module, function, backends):
    """Record the calls of one of FLA's TileLang kernels.

    Returns the call list and whether FLA dispatches to this kernel here: its TileLang backend
    (the first of ``backends`` that ``package`` defines) is available, which FLA 0.6 also ties
    to a usable nvcc, and enabled. The kernel module is imported lazily by the backend at each
    call, so patching it takes effect. These kernels never reach Triton's autotuner: their
    tiles, warps and stages follow from the input shapes and the GPU, so a bit-exact replay
    through them is their deterministic-mode coverage.
    """
    import importlib

    tilelang_backends = importlib.import_module(package)
    backend = next(
        getattr(tilelang_backends, name) for name in backends if hasattr(tilelang_backends, name)
    )()
    if not (backend.is_available() and backend.is_enabled()):
        return [], False
    kernels = importlib.import_module(f"{package}.{module}")
    kernel = getattr(kernels, function)
    calls = []

    def record(*args, **kwargs):
        calls.append(function)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(kernels, function, record)
    return calls, True


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    "two_stage_gates, grad",
    [(False, True), (True, True), (True, False)],
    ids=["single_stage", "two_stage", "two_stage_no_grad"],
)
def test_kda_deterministic_mode_pins_fla_autotuners(monkeypatch, two_stage_gates, grad, head_dim):
    """KDA has no non-FLA core kernel, so deterministic mode runs it on FLA's autotuned kernels,
    some of whose candidates give different bits (the L2 norm, the gated output norm and the
    gate cumsum). The policy pins every FLA autotuner the layer reaches, without timing any, and
    the pinned configs launch at every supported head dim.

    The no-grad case covers evaluation and inference: the two-stage gated norm must stay out of
    torch.compile there, where Dynamo would otherwise trace FLA's kernel and Inductor time it.
    FLA's TileLang backward, used when TileLang is installed, must run and replay too."""
    import torch.nn.functional as F

    from megatron.core import parallel_state
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net.kda import HAVE_FLA_KDA, KimiDeltaAttention
    from megatron.core.ssm.kda_layer_config import KDALayerConfig
    from tests.unit_tests.determinism.kernels.harness import deterministic_algorithms

    assert HAVE_FLA_KDA, "KDA needs flash-linear-attention with chunk_kda"
    reached = _record_fla_autotuners(monkeypatch)
    tilelang_calls, tilelang_dispatched = _record_tilelang_kernel(
        monkeypatch,
        "fla.ops.kda.backends.tilelang",
        "chunk_bwd_dqkg",
        "chunk_kda_bwd_wy_dqkg_fused_tilelang",
        backends=("KDATileLangBackend",),
    )
    seeded()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = KDALayerConfig(
            hidden_size=512,
            num_layers=1,
            num_attention_heads=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=head_dim,
            linear_value_head_dim=head_dim,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            kda_two_stage_gates=two_stage_gates,
            kda_safe_gate=True,
            kda_lower_bound=-5.0,
            normalization="RMSNorm",
            activation_func=F.silu,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            experimental_attention_variant="kda",
            is_hybrid_model=True,
            deterministic_mode=True,
        )
        spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
        layer = KimiDeltaAttention(
            config,
            submodules=spec.submodules,
            layer_number=1,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        )
        hidden = torch.randn(
            256, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=grad
        )
        with deterministic_algorithms(True), torch.set_grad_enabled(grad):
            assert_module_replays_bit_exact(
                layer,
                (hidden, None),
                replays=2,
                backward=grad,
                contention=True,
                what=f"deterministic KDA (head dim {head_dim})",
            )
    finally:
        Utils.destroy_model_parallel()

    expected = {"kda_gate_chunk_cumsum_vector_kernel"}
    if two_stage_gates:
        expected |= {"l2norm_fwd_kernel", "layer_norm_gated_fwd_kernel"}
    _assert_fla_autotuners_pinned(reached, expected)
    assert bool(tilelang_calls) == (grad and tilelang_dispatched), tilelang_calls


def test_gdp_deterministic_mode_pins_fla_autotuners(monkeypatch):
    """GDP trains on FLA's chunk_gated_delta_product and L2 norm whatever deterministic_mode says.
    The policy pins their forward and backward autotuners without timing any, FLA's TileLang
    backward (chunk_bwd_dqkwg, which FLA dispatches to on Hopper) runs at fixed tiles, and the
    mixer's forward and backward replay bit-exactly."""
    from tests.unit_tests.ssm.test_gated_delta_product import HAVE_FLA, HAVE_GDP_DEPS
    from tests.unit_tests.ssm.test_gated_delta_product import _build_mixer as build_gdp
    from tests.unit_tests.ssm.test_gated_delta_product import _make_config

    assert HAVE_GDP_DEPS and HAVE_FLA, "GDP needs mamba_ssm, causal_conv1d and FLA"
    reached = _record_fla_autotuners(monkeypatch)
    tilelang_calls, tilelang_dispatched = _record_tilelang_kernel(
        monkeypatch,
        "fla.ops.common.backends.tilelang",
        "chunk_bwd",
        "chunk_bwd_dqkwg_tilelang",
        backends=("CommonTileLangBackend", "TileLangBackend"),
    )
    seeded()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _make_config([])
        config.deterministic_mode = True
        mixer = build_gdp(config)
        hidden = torch.randn(
            256, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        assert_module_replays_bit_exact(
            mixer, (hidden,), replays=2, contention=True, what="deterministic GDP mixer"
        )
    finally:
        Utils.destroy_model_parallel()

    _assert_fla_autotuners_pinned(
        reached,
        {
            "l2norm_fwd_kernel",
            "l2norm_bwd_kernel",
            "chunk_gated_delta_product_fwd_kernel_h_blockdim64",
            "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
        },
    )
    assert bool(tilelang_calls) == tilelang_dispatched, tilelang_calls


def test_kda_fp8_opt_out_scope_is_replay_safe():
    """KDA-only FP8 initialization selection remains stable across the dispatch path."""
    from megatron.core.ssm.gated_delta_net.common import _kda_fp8_disabled
    from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
    from megatron.core.ssm.kda_layer_config import KDALayerConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    config.kda_disable_fp8 = True
    assert _kda_fp8_disabled(KDALayerConfig.from_config(config))
    assert not _kda_fp8_disabled(GDNLayerConfig.from_config(config))


def test_fla_chunk_gated_delta_rule_replays_fwd_bwd():
    """FLA's gated delta rule replays once the default policy pins FLA's autotuners: its chunk
    kernels reduce in a fixed order, with no atomics, so the timed choice of configs was what
    made replays differ."""
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


# --- Mamba mixer: the batch-invariant unfused training path (_static_prefill) ----------------


@pytest.mark.skipif(mamba_mixer_module.mamba_chunk_scan_combined is None, reason="needs mamba_ssm")
@pytest.mark.parametrize("batch_invariant", [True, False], ids=["batch-invariant", "default"])
def test_unfused_mixer_applies_the_gate_exactly_once(monkeypatch, batch_invariant):
    """Batch-invariant mode gates inside the scan (as ssm_prefill does), else in the norm.

    The kernel tests above call the scan with ``z`` supplied, so they cannot see which side of
    the mixer applies it.
    """
    mixer = _build_mixer(deterministic_mode=False)
    try:
        mixer.use_mem_eff_path = False  # dispatch to _static_prefill
        mixer.config.batch_invariant_mode = batch_invariant
        assert mixer.rmsnorm, "the gate placement under test only exists with an RMSNorm"

        scan_gates, norm_gates = [], []
        real_scan = mamba_mixer_module.mamba_chunk_scan_combined
        real_norm_forward = mixer.norm.forward

        def scan(*args, **kwargs):
            scan_gates.append(kwargs.get("z"))
            return real_scan(*args, **kwargs)

        def norm_forward(x, z=None):
            norm_gates.append(z)
            return real_norm_forward(x, z)

        monkeypatch.setattr(mamba_mixer_module, "mamba_chunk_scan_combined", scan)
        monkeypatch.setattr(mixer.norm, "forward", norm_forward)

        with torch.no_grad():
            mixer(torch.randn(256, 2, mixer.config.hidden_size, device="cuda"))

        assert len(scan_gates) == 1 and len(norm_gates) == 1
        gated_in_scan, gated_in_norm = scan_gates[0] is not None, norm_gates[0] is not None
        assert gated_in_scan != gated_in_norm, "the gate must be applied exactly once"
        assert gated_in_scan == batch_invariant, (
            f"batch_invariant_mode={batch_invariant}: the gate went "
            f"{'into the scan' if gated_in_scan else 'into the norm'}"
        )
    finally:
        Utils.destroy_model_parallel()


@requires_deterministic_conv1d
def test_batch_invariant_unfused_mixer_replays_bit_exactly(monkeypatch):
    """Forward and backward through the batch-invariant unfused training path replay bitwise."""
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    mixer = _build_mixer()
    try:
        mixer.use_mem_eff_path = False  # dispatch to _static_prefill
        mixer.config.batch_invariant_mode = True
        hidden_states = torch.randn(1024, 4, mixer.config.hidden_size, device="cuda")
        hidden_states.requires_grad_()
        assert_module_replays_bit_exact(
            mixer,
            (hidden_states,),
            replays=3,
            contention=True,
            what="batch-invariant Mamba mixer (_static_prefill)",
        )
    finally:
        Utils.destroy_model_parallel()
