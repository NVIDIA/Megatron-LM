# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the GDP/GDN chunked prefill, `chunk_gated_delta_product_varlen`
(megatron/core/ssm/ops/gdp/chunk.py), end to end through its seven launched kernels: `l2norm_fwd_kernel`,
`chunk_local_cumsum_scalar_kernel` (twice), `chunk_scaled_dot_kkt_fwd_kernel`,
`merge_16x16_to_64x64_inverse_kernel`, `recompute_w_u_fwd_kernel`,
`chunk_gated_delta_product_fwd_kernel_h_blockdim64` and `chunk_fwd_kernel_o`. Each also has its own module
here; this one checks what the call sites see.

Call sites: `GatedDeltaProductMixer.ssm_prefill` (megatron/core/ssm/gated_delta_product.py: after
`gdp_prefill_prepare` and #8926's optional clamp `_clamp_g`, M = gdp_num_householder, beta in the activation
dtype, `use_qk_l2norm_in_kernel=True`, `return_chunk_states` under prefix caching) and
`GatedDeltaNet.ssm_prefill` (megatron/core/ssm/gated_delta_net/gdn.py: M = 1, q/k repeated to the value
heads, g = -exp(A_log) softplus(alpha + dt_bias) and beta = sigmoid in fp32, `use_qk_l2norm` configurable).
Both pass `initial_state = ssm_state[batch_indices]`, `state = ssm_state`, `state_indices = batch_indices`
(int32, -1 for padding requests) and the padded descriptors of `build_gdp_chunk_descriptors`.

Contract (from the gated delta product math, the launcher's documented arguments and the call sites, before
reading the kernels): a packed prefill step of N sequences (`cu_seqlens` on the token stream; padding
requests are zero-length) and tokens past the last sequence that belong to none. Sequence n runs the GDP
recurrence (`delta_rule.gdp_sequence`: per token, M delta-rule steps on its M key/value/beta copies, the
decay g on the first, the query on the last, q and k L2-normalised when asked, scale K^-0.5) from
initial_state[n], and:
* o[t] (q's dtype) is token t's output; tokens no sequence owns read 0;
* state[state_indices[n]] = its final state in the cache dtype (skipped for -1); other slots unchanged;
* with `return_chunk_states`, row chunk_offsets[n] + c of the returned [NT, H, K, V] (cache dtype) is the
  state entering its chunk c, after its first 64c tokens (c = 0: initial_state[n] itself);
* the inputs are read-only.
Composition (the call sites' contract): chunked prefill, [0, p) then [p, L) from the slot the first call
wrote, and a prefix-cache restore, [64c, L) from chunk state c, continue the recurrence of [0, L).

Tolerance class: floating point against the fp64 recurrence (no chunking), with two trusted
implementations: (1) FLA's `chunk_gated_delta_product`, the training path's kernel, on the same inputs and
device (factor 2): the chunked algorithm keeps w, u, v_new and the per-chunk states feeding o in the
activation dtype, so the fp32 cache state carries bf16-level error by design, which only the same algorithm
can calibrate (an fp32 recurrence is ~10^4x more accurate there, Triton interpreter). It bounds o, the final
states and the chunk states (FLA on each prefix). (2) the recurrence in torch fp32 rounded to q's dtype, an
implementation independent of FLA's lineage, bounds o with `INDEPENDENT_FACTOR` in the production decay
regime (see there). Exact: tokens
no sequence owns (0), untouched slots, chunk state 0, read-only inputs.

Variants (read after the contract): M 1/2/3; H 1 and 4; K 64, 96 (masked), 128; V 32, 64, 128; bf16 and fp16;
beta in the activation dtype (GDP) and fp32 (GDN); L2 norm on and off (off: keys of unit expected norm,
`unit_norm_keys`); fp32 and bf16 caches; chunk states on
and off; random and zero initial states; decay regimes production / #8926's clamp on (`_clamp_g`, 50) /
clamp off (decays to -1e4); packs from `delta_rule.STAGE_PACKS` (token and expanded-stream boundaries,
padding requests, padded descriptors, NaN tail tokens). The sweep runs under deterministic-mode selection
(`delta_rule.pinned_autotuning`: each kernel's cheapest candidate, as `--deterministic-mode` selects);
`test_timed_autotuning` runs one case on the autotuners' timed choice; each kernel's every tile is pinned in
its own module. Production: the GDP 800m layer (32 heads, K 128, V 64, M 3, fp32 cache, chunk states) and
the GDN default layer (32 value heads, K = V = 128, fp32 beta, bf16 cache) on a 4096-token pack of six
sequences (boundaries at chunk offsets 0, 1, 63), two padding requests and padded tail tokens.

Memory: the cache in guard bands with unmapped slots NaN-poisoned (the -1 requests' host gather lands on the
last slot, which is one of them); NaN tail tokens; `sanitizer` on a small padded case. Needs FLA (the CI
image's fla 0.5.1): without it the test fails, not skips.
"""

import types

import pytest
import torch

from megatron.core.ssm.ops.gdp import chunk_gated_delta_product_varlen
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
CHUNK = dr.CHUNK
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

INDEPENDENT_FACTOR = 8.0
"""Bound factor for o against the independent fp32 recurrence. The chunked algorithm rounds (I + A)^-1, w,
u, v_new, the per-chunk states and the score matrix to the activation dtype, each costing about one more
output rounding; measured 1.8-3.4x the fp32 recurrence's rms error at M 1 and 3 in the Triton interpreter
(KDA, with fewer such roundings, measured 1.5-2.2x). 8 leaves room for tensor-core accumulation on the GPU;
wiring errors (decay, beta, scale, L2 norm, sequence boundaries, slots) are O(1), orders of magnitude above.

Applied in the production decay regime only. In the clamp regimes most tokens' decays wipe the state, so
a token's output shrinks to its own q.k-sized term (|o| ~ 1e-3 |v|) while the chunked intermediates keep
the size of v: their input-dtype rounding, not the output rounding, sets the error, at up to 30x the fp32
recurrence's for fp16 (Triton interpreter; FLA identical). There FLA calibrates alone."""


def fla_prefill(q, k, v, g, beta, *, M, initial_state, cu_seqlens, l2norm):
    """FLA's `chunk_gated_delta_product` (the training kernel) on the given real sequences."""
    from fla.ops.gated_delta_product import chunk_gated_delta_product

    with dr.pinned_autotuning(("fla",)):
        o, final = chunk_gated_delta_product(
            q,
            k,
            v,
            g,
            beta,
            num_householder=M,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=l2norm,
            cu_seqlens=cu_seqlens.long(),
        )
    return o, final


def _slice(pack, start, end):
    """Tokens [start, end) of a pack on both streams: (q, k, v, g, beta)."""
    M = pack.M
    return (
        pack.q[:, start:end],
        pack.k[:, start * M : end * M],
        pack.v[:, start * M : end * M],
        pack.g[:, start:end],
        pack.beta[:, start * M : end * M],
    )


def _sequence_reference(pack, n, h0, *, l2norm, scale, fp32=False):
    """(o, final, chunk-entry states) of sequence n from h0, fp64 or (fp32=True) the trusted fp32."""
    start, end = dr.bounds(pack.cu_seqlens)[n]
    q, k, v, g, beta = (x[0] for x in _slice(pack, start, end))

    def fn(q, k, v, g, beta, h0):
        return dr.gdp_sequence(
            q, k, v, g, beta, h0, num_householder=pack.M, scale=scale, l2norm_qk=l2norm
        )

    if fp32:
        return dr.trusted_fp32(fn, q, k, v, g, beta, h0)
    return ou.fp64_reference(fn, q, k, v, g, beta, h0)


def run_prefill(c, *, g0=None, pack=None, autotune="pinned"):
    """Launch one prefill step for case `c` (a dict) and check it against the contract."""
    g0 = ou.make_generator(device=DEVICE) if g0 is None else g0
    dtype = DTYPES[c.get("dtype", "bf16")]
    H, K, V = c["H"], c["K"], c["V"]
    if pack is None:
        lens, M = c.get("lens") or dr.STAGE_PACKS[c["pack"]]
        pack = dr.make_pack(
            lens,
            num_householder=M,
            heads=H,
            head_k=K,
            head_v=V,
            dtype=dtype,
            beta_dtype=torch.float32 if c.get("beta") == "fp32" else None,
            regime=c.get("regime", "production"),
            padded_tokens=sum(lens) + dr.TAIL_TOKENS,
            generator=g0,
            device=DEVICE,
        )
        if not c.get("l2norm", True):
            unit_norm_keys(pack)
    lens, M = pack.lens, pack.M
    N = len(lens)
    cache_dtype = DTYPES[c.get("cache", "fp32")]
    l2norm = c.get("l2norm", True)
    chunk_states = c.get("chunk_states", False)
    desc = dr.descriptors(pack, device=DEVICE)

    # Cache: N + 3 slots, real requests on a permutation, padding requests -1; the unmapped slots
    # (including the last, where a -1 gather lands) NaN-poisoned.
    slot_count = N + 3
    perm = torch.randperm(slot_count - 1, generator=ou.make_generator())[:N].tolist()
    slot_list = [s if n_len > 0 else -1 for s, n_len in zip(perm, lens)]
    init = torch.randn(slot_count, H, K, V, generator=g0, device=DEVICE)
    if c.get("init") == "zero":
        init.zero_()
    init = init.to(cache_dtype)
    unmapped = [s for s in range(slot_count) if s not in slot_list]
    for s in unmapped:
        ou.poison_(init[s])
    cache = ou.guarded_like(init)
    slots = torch.tensor(slot_list, dtype=torch.int32, device=DEVICE)
    initial_state = cache.tensor[slots.long()]  # the call sites' gather (-1 -> last slot)

    inputs = [pack.q, pack.k, pack.v, pack.g, pack.beta, initial_state]
    before = [x.clone() for x in inputs]
    ctx = dr.pinned_autotuning() if autotune == "pinned" else dr.pinned_autotuning(("fla",))
    with ctx:
        result = chunk_gated_delta_product_varlen(
            pack.q,
            pack.k,
            pack.v,
            g=pack.g,
            beta=pack.beta,
            num_householder=M,
            cu_seqlens=pack.cu_seqlens,
            initial_state=initial_state,
            state=cache.tensor,
            state_indices=slots,
            use_qk_l2norm_in_kernel=l2norm,
            return_chunk_states=chunk_states,
            **desc,
        )
    o, final = result[0], result[1]
    assert final is cache.tensor, "with a cache, the final state returned is the cache"
    ou.check_guards(cache=cache)
    for x, b in zip(inputs, before):
        ou.assert_exact(x, b, what="inputs are read-only")
    real = pack.tokens
    ou.assert_exact(
        o[:, real:], torch.zeros_like(o[:, real:]), what="tokens no sequence owns read 0"
    )
    for s in unmapped:
        ou.assert_poisoned(cache.tensor[s], what=f"unmapped slot {s}")

    scale = K**-0.5
    live = [n for n, n_len in enumerate(lens) if n_len > 0]
    h0 = initial_state.to(torch.float32)
    cu_live = torch.tensor(
        [0, *torch.tensor([lens[n] for n in live]).cumsum(0).tolist()], device=DEVICE
    )
    fla_o, fla_final = fla_prefill(
        *_slice(pack, 0, real),
        M=M,
        initial_state=initial_state[live],
        cu_seqlens=cu_live,
        l2norm=l2norm,
    )
    offsets = desc["chunk_offsets"].tolist()
    what = f"prefill {c}"
    for i, n in enumerate(live):
        start, end = dr.bounds(pack.cu_seqlens)[n]
        ref = _sequence_reference(pack, n, h0[n], l2norm=l2norm, scale=scale)
        mine = _sequence_reference(pack, n, h0[n], l2norm=l2norm, scale=scale, fp32=True)
        ou.assert_close_to_reference(
            o[0, start:end], ref[0], fla_o[0, start:end], what=f"o of sequence {n} vs FLA, {what}"
        )
        if c.get("regime", "production") == "production":
            ou.assert_close_to_reference(
                o[0, start:end],
                ref[0],
                mine[0].to(o.dtype),
                factor=INDEPENDENT_FACTOR,
                what=f"o of sequence {n} vs the fp32 recurrence, {what}",
            )
        ou.assert_close_to_reference(
            cache.tensor[slot_list[n]],
            ref[1],
            fla_final[i].to(cache_dtype),
            what=f"final state of sequence {n} (slot {slot_list[n]}), {what}",
        )
        if chunk_states:
            states = result[2]
            assert states.dtype == cache_dtype
            rows = states[offsets[n] : offsets[n + 1]]
            ou.assert_exact(
                rows[0], initial_state[n], what=f"chunk state 0 of sequence {n} = its initial state"
            )
            if rows.shape[0] > 1:
                prefixes = [start + CHUNK * j for j in range(1, rows.shape[0])]
                trusted = _fla_prefix_states(pack, start, prefixes, initial_state[n], l2norm=l2norm)
                ou.assert_close_to_reference(
                    rows[1:],
                    ref[2][1:],
                    trusted.to(cache_dtype),
                    what=f"chunk states 1.. of sequence {n}, {what}",
                )
    return pack, cache.tensor, slot_list


def unit_norm_keys(pack):
    """Without the in-kernel L2 norm, scale q and k to unit expected norm (N(0, 1/K) entries): the delta
    rule is a contraction only for beta |k|^2 <= 2, and the GDN layers that turn the norm off feed keys
    of that size; N(0, 1) keys would make every reference grow like 60^t."""
    scale = pack.q.shape[-1] ** -0.5
    for name in ("q", "k"):
        x = getattr(pack, name)
        setattr(pack, name, (x.float() * scale).to(x.dtype))


def _fla_prefix_states(pack, start, ends, h0, *, l2norm):
    """FLA's final states of the prefixes [start, e) of one sequence, all from h0: [len(ends), H, K, V]."""
    parts = [_slice(pack, start, e) for e in ends]
    cat = [torch.cat([p[i] for p in parts], dim=1) for i in range(5)]
    cu = torch.tensor(
        [0, *torch.tensor([e - start for e in ends]).cumsum(0).tolist()], device=DEVICE
    )
    _, final = fla_prefill(
        *cat,
        M=pack.M,
        initial_state=h0[None].expand(len(ends), *h0.shape).contiguous(),
        cu_seqlens=cu,
        l2norm=l2norm,
    )
    return final


CASES = [
    pytest.param(
        f,
        id=(
            "{pack}-H{H}-K{K}-V{V}-{dtype}-beta_{beta}-{l2}-cache_{cache}-{cs}-init_{init}-{regime}".format(
                l2="l2" if f["l2norm"] else "nol2",
                cs="states" if f["chunk_states"] else "nostates",
                **f,
            )
        ),
    )
    for f in ou.pairwise_cases(
        {
            "pack": list(dr.STAGE_PACKS),
            "H": [1, 4],
            "K": [64, 96, 128],
            "V": [32, 64, 128],
            "dtype": ["bf16", "fp16"],
            "beta": ["act", "fp32"],
            "l2norm": [True, False],
            "cache": ["fp32", "bf16"],
            "chunk_states": [True, False],
            "init": ["random", "zero"],
            "regime": list(dr.G_REGIMES),
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASES)
def test_prefill_matches_token_recurrence(case):
    """Pairwise over packs (M 1/2/3), heads, K, V, dtypes, beta dtype, L2 norm, cache dtype, chunk states,
    initial state and decay regime: o, final states, chunk states, padding and untouched slots."""
    run_prefill(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("clamp", [None, dr.GATE_CLAMP], ids=["clamp_off", "clamp_on"])
def test_mixer_clamp_feeds_the_kernel(clamp):
    """#8926: the GDP mixer's `_clamp_g` (gdp_g_clamp_value None or 50) applied to decays that reach
    -1e4; the kernel must follow the recurrence of whichever decays it is given, and with the clamp
    every decay is >= -50."""
    from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer

    g0 = ou.make_generator(device=DEVICE)
    lens, M = dr.STAGE_PACKS["M3"]
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=2,
        head_k=64,
        head_v=32,
        regime="clamp_off",
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    stub = types.SimpleNamespace(config=types.SimpleNamespace(gdp_g_clamp_value=clamp))
    raw = pack.g
    clamped = GatedDeltaProductMixer._clamp_g(stub, raw)
    real = pack.tokens
    if clamp is None:
        assert clamped is raw
        assert float(raw[:, :real].min()) < -dr.GATE_CLAMP, "the regime must reach the clamp"
    else:
        assert float(clamped[:, :real].min()) == -clamp
        assert bool((clamped[:, :real] >= -clamp).all())
    pack.g = clamped
    regime = "clamp_on" if clamp else "clamp_off"
    run_prefill(
        dict(H=2, K=64, V=32, cache="fp32", chunk_states=True, regime=regime), pack=pack, g0=g0
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("p", [1, CHUNK - 1, CHUNK, CHUNK + 1], ids=lambda p: f"p{p}")
def test_chunked_prefill_continues_the_recurrence(p):
    """[0, p) then [p, L) from the slot the first call wrote (as the dynamic context schedules a long
    prompt), L = 150, M 3: the second call's o and final state against the fp64 recurrence of [0, L);
    trusted = FLA making the same two calls."""
    g0 = ou.make_generator(device=DEVICE)
    L, M, H, K, V = 150, 3, 2, 64, 32
    full = dr.make_pack(
        (L,), num_householder=M, heads=H, head_k=K, head_v=V, generator=g0, device=DEVICE
    )
    cache = torch.randn(4, H, K, V, generator=g0, device=DEVICE)
    slot = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    h_start = cache[2].clone()
    outs, fla_state, fla_outs = [], h_start[None].clone(), []
    for start, end in ((0, p), (p, L)):
        part = dr.Pack(
            *(x.contiguous() for x in _slice(full, start, end)),
            cu_seqlens=torch.tensor([0, end - start], dtype=torch.int32, device=DEVICE),
            lens=[end - start],
            num_householder=M,
            tokens=end - start,
        )
        with dr.pinned_autotuning():
            o, _ = chunk_gated_delta_product_varlen(
                part.q,
                part.k,
                part.v,
                g=part.g,
                beta=part.beta,
                num_householder=M,
                cu_seqlens=part.cu_seqlens,
                initial_state=cache[slot.long()],
                state=cache,
                state_indices=slot,
                use_qk_l2norm_in_kernel=True,
                **dr.descriptors(part, device=DEVICE),
            )
        outs.append(o)
        fo, fla_state = fla_prefill(
            *_slice(full, start, end),
            M=M,
            initial_state=fla_state,
            cu_seqlens=part.cu_seqlens,
            l2norm=True,
        )
        fla_outs.append(fo)
    ref = _sequence_reference(full, 0, h_start, l2norm=True, scale=K**-0.5)
    ou.assert_close_to_reference(outs[1][0], ref[0][p:], fla_outs[1][0], what=f"o of [{p}, {L})")
    ou.assert_close_to_reference(
        cache[2], ref[1], fla_state[0], what="final state after both calls"
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("c", [1, 2])
def test_prefix_cache_restore_continues_the_recurrence(c):
    """A prefix-cache hit: chunk state c of a first run, restored into a slot, then [64c, L): o and
    final state against the fp64 recurrence of [0, L); trusted = FLA on [0, 64c) and then [64c, L).
    """
    g0 = ou.make_generator(device=DEVICE)
    L, M, H, K, V = 3 * CHUNK + 5, 2, 2, 64, 32
    full = dr.make_pack(
        (L,), num_householder=M, heads=H, head_k=K, head_v=V, generator=g0, device=DEVICE
    )
    cache = torch.randn(3, H, K, V, generator=g0, device=DEVICE)
    slot = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    h_start = cache[1].clone()
    with dr.pinned_autotuning():
        _, _, states = chunk_gated_delta_product_varlen(
            full.q,
            full.k,
            full.v,
            g=full.g,
            beta=full.beta,
            num_householder=M,
            cu_seqlens=full.cu_seqlens,
            initial_state=cache[slot.long()],
            state=cache,
            state_indices=slot,
            use_qk_l2norm_in_kernel=True,
            return_chunk_states=True,
            **dr.descriptors(full, device=DEVICE),
        )
    p = CHUNK * c
    cache[0] = states[c]
    suffix = dr.Pack(
        *(x.contiguous() for x in _slice(full, p, L)),
        cu_seqlens=torch.tensor([0, L - p], dtype=torch.int32, device=DEVICE),
        lens=[L - p],
        num_householder=M,
        tokens=L - p,
    )
    restore = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    with dr.pinned_autotuning():
        o, _ = chunk_gated_delta_product_varlen(
            suffix.q,
            suffix.k,
            suffix.v,
            g=suffix.g,
            beta=suffix.beta,
            num_householder=M,
            cu_seqlens=suffix.cu_seqlens,
            initial_state=cache[restore.long()],
            state=cache,
            state_indices=restore,
            use_qk_l2norm_in_kernel=True,
            **dr.descriptors(suffix, device=DEVICE),
        )
    _, fla_state = fla_prefill(
        *_slice(full, 0, p),
        M=M,
        initial_state=h_start[None],
        cu_seqlens=torch.tensor([0, p], device=DEVICE),
        l2norm=True,
    )
    fo, fla_final = fla_prefill(
        *_slice(full, p, L), M=M, initial_state=fla_state, cu_seqlens=suffix.cu_seqlens, l2norm=True
    )
    ref = _sequence_reference(full, 0, h_start, l2norm=True, scale=K**-0.5)
    ou.assert_close_to_reference(o[0], ref[0][p:], fo[0], what=f"o after restoring chunk state {c}")
    ou.assert_close_to_reference(
        cache[0], ref[1], fla_final[0], what="final state after the restore"
    )


@ou.requires_sm(ou.HOPPER)
def test_timed_autotuning():
    """One case on the fork's timed autotuning (the default outside deterministic mode)."""
    run_prefill(
        dict(pack="M3", H=4, K=128, V=64, cache="fp32", chunk_states=True, regime="production"),
        autotune="timed",
    )


PRODUCTION_LENS = (2048, 1024, 769, 191, 63, 1, 0, 0)
"""A 4096-token step of six sequences (boundaries at chunk offsets 0, 1 and 63) and two padding requests."""

PRODUCTION = {
    "gdp_800m": dict(lens=(PRODUCTION_LENS, 3), H=32, K=128, V=64, cache="fp32", chunk_states=True),
    "gdn_default": dict(
        lens=(PRODUCTION_LENS, 1), H=32, K=128, V=128, beta="fp32", cache="bf16", chunk_states=False
    ),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_prefill(name):
    """Production layers on a 4096-token step with padding (see the module docstring)."""
    run_prefill(PRODUCTION[name])


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """A small padded M 3 step with a bf16 cache and chunk states, for the compute-sanitizer pass."""
    run_prefill(
        dict(pack="M3", H=2, K=96, V=32, cache="bf16", chunk_states=True, regime="clamp_off")
    )
