# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_chunk_scan_fwd_kernel` (megatron/core/ssm/ops/mamba2/ssd_chunk_scan.py:78), the kernel that
writes the Mamba-2 prefill output `y`, through its production entry point `mamba_chunk_scan_combined_varlen`
(ssd_combined.py:158), which also returns the per-sequence final states and the chunk-exit states the
prefix cache stores.

Contract (written from the Mamba-2 recurrence, the launcher's documented API and its caller
`MambaMixer.ssm_prefill`, megatron/core/ssm/mamba_mixer.py:776-1070, before reading the kernel body):

* A pack of prefill requests: x, z (T, nheads, headdim), dt (T, nheads), B, C (T, ngroups, dstate), split
  into pseudo-chunks by `cu_chunk_seqlens` (nchunks + 1): each request is cut every `chunk_size` tokens from
  its own start, a chunk never spans two requests, `seq_idx` (nchunks,) maps a chunk to its request and
  `last_chunk_indices` (nrequests,) gives each request's last chunk. These come from the production rule,
  `MambaMetadata.compute_cpu_metadata` (mamba_metadata.py:675), including padding: zero-length padded
  requests (one empty chunk each), and padded tail tokens that belong to no chunk.
* Request r starts from initial_states[r] (nrequests, nheads, headdim, dstate) (the gathered slot state,
  `ssm_state[batch_indices]`; zeros when None). For its tokens t in order, with g(h) = h // (nheads / ngroups):

      dt'_t   = clamp(softplus(dt[t,h] + dt_bias[h]), dt_limit)          (production dt_limit (0, inf))
      h_t     = exp(dt'_t * A[h]) * h_{t-1} + dt'_t * x[t,h,:] (outer) B[t,g(h),:]
      out[t,h,p] = (sum_n C[t,g(h),n] * h_t[p,n] + D[h(,p)] * x[t,h,p]) * z * sigmoid(z)   (no gate without z)

* Returns final_states[r] = state after request r's last token (h0 for a zero-length request), and, with
  `return_raw_states`, states[c] = state after the last token of chunk c, both in `state_dtype`. `out` is
  written in place at the tokens of every chunk; tokens outside every chunk are left untouched.
* Preconditions (caller): chunk_size a power of two (asserted); chunks never span requests and are at most
  chunk_size long (unchecked); the metadata's chunk size equals the kernels' (unchecked on main; #8884,
  t3-r10f1, rejects a mismatch at the context, not here).
* Composition (chunked prefill and prefix caching): prefilling [0, p) and then [p, L) from the returned
  state, or from a chunk-exit state at a chunk boundary, continues the same recurrence.
* Numerics: fp32 accumulation with operands in the input dtype on tensor cores (bf16, or TF32 for fp32), the
  incoming state rounded to the input dtype for the C * state product; this is the algorithm the training
  scan (pip `mamba_ssm.mamba_chunk_scan_combined`) runs too.
* Tolerance class: floating point, bounded by a trusted implementation: pip `mamba_chunk_scan_combined`
  (the training kernel; same chunked algorithm, separately implemented) run per request with
  `initial_states` and `return_final_states` on the same device; its chunk-exit states come from calls on
  each request's prefix that ends at the chunk boundary. The reference is an fp64 token-by-token recurrence
  (no chunking). Zero-length requests, untouched output rows and read-only inputs: exact.

Variants and dispatch paths (from the kernel, read after the contract): 11 autotune configs, BLOCK_SIZE_M in
{32, 64, 128}, BLOCK_SIZE_N in {32, 64, 128, 256}, BLOCK_SIZE_K in {32, 64} (ssd_chunk_scan.py:15-75), each
pinned in turn by `test_every_chunk_scan_tile`; constexpr HAS_D / D_HAS_HDIM (per-head and per-channel D),
HAS_Z, HAS_INITSTATES (with and without initial states), the dstate <= 128 single-dot branch and the
dstate > 128 K loop (:211-264, dstate 129 and 256); the previous-state source: the request's initial state
when the previous chunk belongs to another request, else the previous chunk's exit state (:183-198). Sizes:
request lengths 1, c-1, c, c+1, 2c, 2c+3 and 3c+5 for chunk sizes 64, 128 (production) and 256, zero-length
padded requests and padded tail tokens; head dim 64 (production) and 48; nheads 1 (#8713's case at the
mixer) and 8; ngroups 1 and 2. The target-rows mode (HAS_TARGET_ROWS, batch-invariant decode fallback) is not
covered here.

Main vs open PRs: none of #8713 (n17), #8884 (t3-r10f1), #8861, #8882, #8883, #8710 changes these kernels or
their launcher; the module passes on main and with each of them.
"""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.attention_context.mamba_metadata import MambaMetadata
from megatron.core.ssm.ops.mamba2 import ssd_chunk_scan
from megatron.core.ssm.ops.mamba2.ssd_combined import mamba_chunk_scan_combined_varlen
from tests.unit_tests.inference.oracles import oracle_utils as ou

DEVICE = "cuda"
PRODUCTION_CHUNK = (
    128  # MambaMixer chunk_size default (mamba_mixer.py:195), MambaInferenceStateConfig
)


# ---------------------------------------------------------------------------------------------
# Production chunk metadata
# ---------------------------------------------------------------------------------------------


def production_metadata(lengths, padded_requests, padded_tail_tokens, chunk_size):
    """Chunk metadata for a prefill pack from `MambaMetadata.compute_cpu_metadata`, the production code.

    lengths: the real requests' token counts; padded_requests zero-length requests and padded_tail_tokens
    tokens are added as a CUDA-graph bucket would. Returns a dict of device tensors and sizes.
    """
    real = len(lengths)
    padded_count = real + padded_requests
    real_tokens = sum(lengths)
    padded_tokens = real_tokens + padded_tail_tokens
    metadata = MambaMetadata(
        max_requests=padded_count,
        max_tokens=padded_tokens,
        max_intermediate_count=1,
        mamba_chunk_size=chunk_size,
    )
    max_chunks = padded_tokens // chunk_size + padded_count
    i32 = dict(dtype=torch.int32)
    bufs = {
        "batch_indices_decode": torch.full((padded_count,), -1, dtype=torch.int64),
        "batch_indices_prefill": torch.full((padded_count,), -1, **i32),
        "seq_idx": torch.full((1, padded_tokens), -1, **i32),
        "cu_seqlens": torch.zeros(padded_count + 1, **i32),
        "cu_chunk_seqlens": torch.zeros(max_chunks + 1, **i32),
        "last_chunk_indices": torch.zeros(padded_count, **i32),
        "seq_idx_for_varlen": torch.zeros(max_chunks, **i32),
        "conv_seq_idx": torch.zeros(padded_tokens, **i32),
        "conv_seq_start": torch.zeros(padded_tokens, **i32),
    }
    metadata.bind_cpu_buffers(bufs)
    cu_query = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), **i32)
    token_to_request = torch.repeat_interleave(
        torch.arange(real, **i32), torch.tensor(lengths, dtype=torch.int64)
    )
    result = metadata.compute_cpu_metadata(
        active_mamba_indices=torch.arange(real, **i32),
        token_to_request_idx=token_to_request,
        cpu_cu_query=cu_query,
        batch_dimensions=InferenceBatchDimensions(
            token_count=real_tokens, prefill_req_count=real, decode_req_count=0
        ),
        padded_batch_dimensions=InferenceBatchDimensions(
            token_count=padded_tokens, prefill_req_count=padded_count, decode_req_count=0
        ),
        enable_chunked_prefill=True,
    )
    nchunks = result["padded_max_chunks"]
    return dict(
        cu_seqlens=bufs["cu_seqlens"][: padded_count + 1].tolist(),
        cu_chunk_seqlens=bufs["cu_chunk_seqlens"][: nchunks + 1].to(DEVICE),
        last_chunk_indices=bufs["last_chunk_indices"][:padded_count].to(DEVICE),
        seq_idx=bufs["seq_idx_for_varlen"][:nchunks].to(DEVICE),
        real_requests=real,
        real_tokens=real_tokens,
        padded_tokens=padded_tokens,
    )


def real_chunks(meta):
    """(chunk index, request, local end) for every chunk of a request (real or padded): the chunk-exit
    states the contract defines. Padded tail chunks (past the last request's last chunk) are left out.
    """
    bounds = meta["cu_chunk_seqlens"].tolist()
    owners = meta["seq_idx"].tolist()
    last = meta["last_chunk_indices"].tolist()
    starts = meta["cu_seqlens"]
    return [(c, owners[c], bounds[c + 1] - starts[owners[c]]) for c in range(last[-1] + 1)]


# ---------------------------------------------------------------------------------------------
# fp64 reference and trusted implementation
# ---------------------------------------------------------------------------------------------


def _scan_reference(x, dt, A, B, C, D, z, dt_bias, h0, starts, lengths, exits):
    """Token-by-token Mamba-2 recurrence per request, written from the contract (no chunking).

    x, z (T, H, P); dt (T, H); A, dt_bias (H,); B, C (T, G, N); D (H,) or (H, P) or None; h0 (R, H, P, N).
    `exits`: (request, local end) pairs. Returns (out at the requests' tokens (T, H, P), final states
    (R, H, P, N), exit states (len(exits), H, P, N)).
    """
    heads, groups = x.shape[1], B.shape[1]
    group_of_head = torch.arange(heads, device=x.device) // (heads // groups)
    if D is not None and D.dim() == 1:
        D = D[:, None]
    out = torch.zeros_like(x)
    finals, exit_states = [], [None] * len(exits)
    for r, (start, length) in enumerate(zip(starts, lengths)):
        h = h0[r]
        for i, (er, end) in enumerate(exits):
            if er == r and end == 0:
                exit_states[i] = h
        for t in range(start, start + length):
            delta = F.softplus(dt[t] + dt_bias).clamp(min=0.0)
            h = (
                torch.exp(delta * A)[:, None, None] * h
                + (delta[:, None] * x[t])[..., None] * B[t, group_of_head][:, None, :]
            )
            y = (h * C[t, group_of_head][:, None, :]).sum(-1)
            if D is not None:
                y = y + D * x[t]
            if z is not None:
                y = y * z[t] * torch.sigmoid(z[t])
            out[t] = y
            for i, (er, end) in enumerate(exits):
                if er == r and end == t - start + 1:
                    exit_states[i] = h
        finals.append(h)
    return out, torch.stack(finals), _stack(exit_states, h0)


def _stack(states, like):
    return torch.stack(states) if states else like.new_zeros((0,) + tuple(like.shape[1:]))


def _pip_scan():
    """pip `mamba_chunk_scan_combined` (the training kernel), or a failure on a GPU without it."""
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except ImportError:
        pytest.fail("mamba_ssm (the trusted implementation) is not installed", pytrace=False)
    return mamba_chunk_scan_combined


def pip_run(x, dt, A, B, C, D, z, dt_bias, h_init, chunk_size):
    """One request through pip `mamba_chunk_scan_combined` from `h_init`: (out, final state in fp32)."""
    y, final = _pip_scan()(
        x[None],
        dt[None],
        A,
        B[None],
        C[None],
        chunk_size,
        D=D,
        z=None if z is None else z[None],
        dt_bias=dt_bias,
        initial_states=h_init[None].float(),
        dt_softplus=True,
        return_final_states=True,
    )
    return y[0], final[0].float()


def trusted_scan(x, dt, A, B, C, D, z, dt_bias, h0, starts, lengths, exits, chunk_size):
    """The trusted implementation: pip `mamba_chunk_scan_combined` per request, from its initial state.

    Same signature and return as `_scan_reference`, in the kernel's dtypes (out in x's dtype; states in
    fp32). Exit states come from runs on the prefix that ends at each chunk boundary.
    """
    out = torch.zeros_like(x)
    finals, exit_states = [], [None] * len(exits)

    def run(r, start, length):
        sl = slice(start, start + length)
        return pip_run(
            x[sl],
            dt[sl],
            A,
            B[sl],
            C[sl],
            D,
            None if z is None else z[sl],
            dt_bias,
            h0[r],
            chunk_size,
        )

    for r, (start, length) in enumerate(zip(starts, lengths)):
        if length == 0:
            finals.append(h0[r].float())
        else:
            y, final = run(r, start, length)
            out[start : start + length] = y
            finals.append(final)
        for i, (er, end) in enumerate(exits):
            if er != r:
                continue
            if end == 0:
                exit_states[i] = h0[r].float()
            elif end == length:
                exit_states[i] = finals[-1]
            else:
                exit_states[i] = run(r, start, end)[1]
    return out, torch.stack(finals), _stack(exit_states, h0.float())


# ---------------------------------------------------------------------------------------------
# One pack through the kernels, checked against the contract
# ---------------------------------------------------------------------------------------------

PACKS = {
    # name: (lengths in terms of chunk size c, padded zero-length requests, padded tail tokens)
    "boundaries": (lambda c: [1, c - 1, c, c + 1, 2 * c, 2 * c + 3], 2, 5),
    "short": (lambda c: [3, 1, c - 1], 1, 0),
    "single": (lambda c: [3 * c + 5], 0, 0),
}


def _scan_inputs(tokens, heads, groups, headdim, dstate, io_dtype, d_kind, with_z, g):
    x = torch.randn(tokens, heads, headdim, device=DEVICE, generator=g).to(io_dtype)
    z = (
        torch.randn(tokens, heads, headdim, device=DEVICE, generator=g).to(io_dtype)
        if with_z
        else None
    )
    dt = torch.randn(tokens, heads, device=DEVICE, generator=g).to(io_dtype)
    B = torch.randn(tokens, groups, dstate, device=DEVICE, generator=g).to(io_dtype)
    C = torch.randn(tokens, groups, dstate, device=DEVICE, generator=g).to(io_dtype)
    # Production ranges: A = -exp(A_log) with A_log ~ log U(1, 16); dt_bias = softplus^-1(U(1e-3, 1e-1)).
    A = -torch.empty(heads, device=DEVICE).uniform_(1.0, 16.0, generator=g)
    dt0 = torch.empty(heads, device=DEVICE).uniform_(1e-3, 1e-1, generator=g)
    dt_bias = dt0 + torch.log(-torch.expm1(-dt0))
    if d_kind == "per_head":
        D = torch.randn(heads, device=DEVICE, generator=g)
    elif d_kind == "per_channel":
        D = torch.randn(heads, headdim, device=DEVICE, generator=g)
    else:
        D = None
    return dict(x=x, dt=dt, A=A, B=B, C=C, D=D, z=z, dt_bias=dt_bias)


def run_scan_case(
    *,
    chunk_size,
    dstate,
    headdim,
    heads_groups,
    pack,
    io_dtype=torch.bfloat16,
    state_dtype=torch.float32,
    init="random",
    d_kind="per_head",
    with_z=False,
    lengths=None,
    initial_states=None,
    inputs=None,
):
    """Run one pack through `mamba_chunk_scan_combined_varlen` and check every output against the contract.

    `lengths`, `initial_states` and `inputs` (as `_scan_inputs` returns, one row per token) replace the
    named pack and random inputs. Returns the kernel's out, final and chunk-exit states and the metadata.
    """
    heads, groups = heads_groups
    g = ou.make_generator(device=DEVICE)
    if lengths is None:
        make_lengths, padded_requests, padded_tail = PACKS[pack]
        lengths = make_lengths(chunk_size)
    else:
        padded_requests, padded_tail = 0, 0
    meta = production_metadata(lengths, padded_requests, padded_tail, chunk_size)
    real, total = meta["real_requests"], meta["padded_tokens"]
    nseq = real + padded_requests
    starts = meta["cu_seqlens"][:-1]
    all_lengths = list(lengths) + [0] * padded_requests

    if inputs is None:
        inputs = _scan_inputs(total, heads, groups, headdim, dstate, io_dtype, d_kind, with_z, g)
    # Padded tail tokens belong to no chunk: NaN there must not reach any output.
    for name in ("x", "dt", "B", "C", "z"):
        if inputs[name] is not None:
            inputs[name][meta["real_tokens"] :] = float("nan")
    if initial_states is None and init != "none":
        initial_states = torch.randn(nseq, heads, headdim, dstate, device=DEVICE, generator=g)
        if init == "zero":
            initial_states.zero_()
        initial_states = initial_states.to(state_dtype)
        # Padded requests gather the last slot in production; poison it.
        initial_states[real:] = float("nan")
    read_only = {k: v.clone() for k, v in inputs.items() if v is not None}
    init_before = None if initial_states is None else initial_states.clone()

    out = ou.guarded_empty((total, heads, headdim), io_dtype, DEVICE)
    final, raw = mamba_chunk_scan_combined_varlen(
        x=inputs["x"],
        dt=inputs["dt"],
        A=inputs["A"],
        B=inputs["B"],
        C=inputs["C"],
        chunk_size=chunk_size,
        cu_chunk_seqlens=meta["cu_chunk_seqlens"],
        last_chunk_indices=meta["last_chunk_indices"],
        seq_idx=meta["seq_idx"],
        out=out.tensor,
        D=inputs["D"],
        z=inputs["z"],
        dt_bias=inputs["dt_bias"],
        initial_states=initial_states,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        return_raw_states=True,
        state_dtype=state_dtype,
    )

    ou.check_guards(out=out)
    for name, before in read_only.items():
        ou.assert_exact(inputs[name], before, what=f"{name} is read-only")
    if init_before is not None:
        ou.assert_exact(initial_states, init_before, what="initial_states is read-only")
    ou.assert_poisoned(out.tensor[meta["real_tokens"] :], what="out rows of padded tail tokens")
    assert final.shape == (nseq, heads, headdim, dstate) and final.dtype == state_dtype
    assert raw.dtype == state_dtype
    if padded_requests:
        # A zero-length request ends where it starts: its (NaN-poisoned) initial state, or zeros.
        if initial_states is None:
            ou.assert_exact(final[real:], torch.zeros_like(final[real:]), what="padded requests")
        else:
            assert torch.isnan(final[real:]).all(), "a padded request's final state is its h0 (NaN)"

    chunks = real_chunks(meta)
    live_chunks = [(c, r, end) for c, r, end in chunks if r < real]
    exits = [(r, end) for _, r, end in live_chunks]
    h0 = (
        torch.zeros(real, heads, headdim, dstate, device=DEVICE, dtype=state_dtype)
        if initial_states is None
        else initial_states[:real]
    )
    real_tokens = meta["real_tokens"]
    args = (
        inputs["x"][:real_tokens],
        inputs["dt"][:real_tokens],
        inputs["A"],
        inputs["B"][:real_tokens],
        inputs["C"][:real_tokens],
        inputs["D"],
        None if inputs["z"] is None else inputs["z"][:real_tokens],
        inputs["dt_bias"],
        h0,
        starts[:real],
        all_lengths[:real],
        exits,
    )
    ref_out, ref_final, ref_exits = ou.fp64_reference(_scan_reference, *args)
    t_out, t_final, t_exits = trusted_scan(*args, chunk_size)
    ou.assert_close_to_reference(out.tensor[:real_tokens], ref_out, t_out, what="out")
    ou.assert_close_to_reference(
        final[:real], ref_final, t_final.to(state_dtype), what="final states"
    )
    ou.assert_close_to_reference(
        raw[[c for c, _, _ in live_chunks]],
        ref_exits,
        t_exits.to(state_dtype),
        what="chunk-exit states",
    )
    return dict(out=out.tensor, final=final, raw=raw, meta=meta, chunks=chunks)


# ---------------------------------------------------------------------------------------------
# Structure-derived cases
# ---------------------------------------------------------------------------------------------


def _scan_cases():
    factors = {
        "pack": ("boundaries", "short", "single"),
        "chunk_size": (PRODUCTION_CHUNK, 64, 256),
        "dstate": (128, 64, 129, 256),
        "heads_groups": ((8, 1), (1, 1), (8, 2)),
        "headdim": (64, 48),
        "io_dtype": (torch.bfloat16, torch.float32),
        "state_dtype": (torch.float32, torch.bfloat16),
        "init": ("random", "zero", "none"),
        "d_kind": ("per_head", "per_channel", None),
        "with_z": (False, True),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                case["pack"],
                f"c{case['chunk_size']}",
                f"n{case['dstate']}",
                "h{}g{}".format(*case["heads_groups"]),
                f"p{case['headdim']}",
                str(case["io_dtype"]).replace("torch.", ""),
                "state_" + str(case["state_dtype"]).replace("torch.", ""),
                f"init_{case['init']}",
                f"D_{case['d_kind']}",
                "z" if case["with_z"] else "noz",
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _scan_cases())
def test_varlen_prefill_matches_fp64(case):
    """Multi-request packs with initial states, chunk boundaries and padding against the fp64 recurrence."""
    run_scan_case(**case)


@ou.requires_sm(ou.HOPPER)
def test_production_prefill():
    """Production prefill: chunk 128, 8 local heads, 1 group, head dim 64, dstate 128, bf16 activations,
    fp32 states, per-head D, z gated by the RMSNorm (z=None), every request with a cached initial state.
    """
    run_scan_case(
        chunk_size=PRODUCTION_CHUNK, dstate=128, headdim=64, heads_groups=(8, 1), pack="boundaries"
    )


@contextlib.contextmanager
def pinned_chunk_scan_config(config):
    """Leave `_chunk_scan_fwd_kernel`'s autotuner one candidate, `config`."""
    tuner = chunk_scan_autotuner()
    saved = tuner.configs, tuner.cache
    try:
        tuner.configs = [config]
        tuner.cache = {}
        yield
    finally:
        tuner.configs, tuner.cache = saved


def chunk_scan_autotuner():
    """The Triton autotuner around `_chunk_scan_fwd_kernel`."""
    from triton.runtime.autotuner import Autotuner

    kernel = ssd_chunk_scan._chunk_scan_fwd_kernel
    while not isinstance(kernel, Autotuner):
        kernel = kernel.fn
    return kernel


def _tile_params():
    configs = chunk_scan_autotuner().configs
    params = []
    for dstate in (128, 256):
        for i, config in enumerate(configs):
            kw = config.kwargs
            tile = "M{}N{}K{}".format(kw["BLOCK_SIZE_M"], kw["BLOCK_SIZE_N"], kw["BLOCK_SIZE_K"])
            params.append(pytest.param(dstate, i, id=f"n{dstate}-cfg{i}-{tile}"))
    return params


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("dstate,config_index", _tile_params())
def test_every_chunk_scan_tile(dstate, config_index):
    """Each autotune candidate of the chunk scan kernel, pinned, on the boundary pack at the production chunk
    size (dstate 128: one dot; dstate 256: the BLOCK_SIZE_K loop over dstate)."""
    config = chunk_scan_autotuner().configs[config_index]
    with pinned_chunk_scan_config(config):
        run_scan_case(
            chunk_size=PRODUCTION_CHUNK,
            dstate=dstate,
            headdim=64,
            heads_groups=(8, 2),
            pack="boundaries",
            d_kind="per_channel",
            with_z=True,
        )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "mode,where",
    [
        pytest.param("prefill", 1, id="prefill-p1"),
        pytest.param("prefill", PRODUCTION_CHUNK - 1, id="prefill-p127[c-1]"),
        pytest.param("prefill", PRODUCTION_CHUNK, id="prefill-p128[c]"),
        pytest.param("prefill", PRODUCTION_CHUNK + 1, id="prefill-p129[c+1]"),
        pytest.param("restore", PRODUCTION_CHUNK, id="restore-p128[1 chunk]"),
        pytest.param("restore", 2 * PRODUCTION_CHUNK, id="restore-p256[2 chunks]"),
    ],
)
def test_chunked_prefill_and_prefix_restore_continue_the_recurrence(mode, where):
    """Chunked prefill: [0, p) and then [p, L) from the returned final state. Prefix-cache hit: [p, L) from
    the chunk-exit state at p of an earlier prefill of [0, L). The continuation's `out` must match the fp64
    recurrence of the whole request from its original initial state, bounded by pip making the same two
    calls. Another request shares each pack, so the handoff is not the pack's first state by accident.
    """
    c, length, p = PRODUCTION_CHUNK, 2 * PRODUCTION_CHUNK + 3, where
    heads, groups, headdim, dstate = 4, 1, 64, 128
    common = dict(chunk_size=c, dstate=dstate, headdim=headdim, heads_groups=(heads, groups))
    g = ou.make_generator(device=DEVICE)
    seq = _scan_inputs(
        length + 12, heads, groups, headdim, dstate, torch.bfloat16, "per_head", True, g
    )
    h0 = torch.randn(3, heads, headdim, dstate, device=DEVICE, generator=g)

    def pack(*spans):
        return {
            k: torch.cat([v[a:b] for a, b in spans]) if k in _TOKEN_KEYS else v
            for k, v in seq.items()
        }

    if mode == "prefill":
        first = run_scan_case(
            lengths=[p, 5],
            inputs=pack((0, p), (length, length + 5)),
            initial_states=h0[:2].clone(),
            pack=None,
            **common,
        )
        handoff = first["final"][0]
    else:
        first = run_scan_case(
            lengths=[length, 5],
            inputs=pack((0, length), (length, length + 5)),
            initial_states=h0[:2].clone(),
            pack=None,
            **common,
        )
        chunk = next(ch for ch, r, end in first["chunks"] if r == 0 and end == p)
        handoff = first["raw"][chunk]
    second = run_scan_case(
        lengths=[length - p, 7],
        inputs=pack((p, length), (length + 5, length + 12)),
        initial_states=torch.stack([handoff, h0[2]]),
        pack=None,
        **common,
    )

    def span(a, b):
        return [seq[k][a:b] if k in _TOKEN_KEYS else seq[k] for k in _ARGS]

    ref_out = ou.fp64_reference(_scan_reference, *span(0, length), h0[:1], [0], [length], [])[0]
    _, pip_final = pip_run(*span(0, p), h0[0], c)
    pip_out, _ = pip_run(*span(p, length), pip_final, c)
    ou.assert_close_to_reference(
        second["out"][: length - p], ref_out[p:], pip_out, what="out after the handoff"
    )


_ARGS = ("x", "dt", "A", "B", "C", "D", "z", "dt_bias")  # `pip_run` / `_scan_reference` order
_TOKEN_KEYS = ("x", "dt", "B", "C", "z")


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: the dstate > 128 K loop, padded requests and tail tokens, z and
    per-channel D."""
    run_scan_case(
        chunk_size=64,
        dstate=129,
        headdim=48,
        heads_groups=(2, 1),
        pack="short",
        d_kind="per_channel",
        with_z=True,
    )


def test_case_list_covers_every_branch():
    """The sweep reaches both dstate branches, every pack, chunk size and init mode (CPU check)."""
    cases = [p.values[0] for p in _scan_cases()]
    assert {c["dstate"] > 128 for c in cases} == {False, True}
    for factor in ("pack", "chunk_size", "init", "d_kind", "with_z", "heads_groups"):
        assert len({str(c[factor]) for c in cases}) > 1
