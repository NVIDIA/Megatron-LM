# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `chunk_gated_delta_product_fwd_kernel_h_blockdim64` (megatron/core/ssm/ops/gdp/chunk_h.py),
launched by `chunk_gated_delta_product_fwd_h`: the inter-chunk state recurrence of the GDP/GDN prefill.

Reached in inference only through `chunk_gated_delta_product_varlen` (GDP and GDN prefill) with the
expanded-stream k, w, u (activation dtype), the expanded within-chunk base-2 decays, `initial_state` (the
cache rows gathered at `batch_indices`, always passed), `state` + `state_indices` (the slot-indexed cache,
written in place; `-1` padding), the token-stream descriptors and `chunk_offsets`, and `states_dtype` = the
cache dtype when prefix caching snapshots the chunk states (`return_chunk_states`). The chunk states h and
v_new are internal to the entry point, and the dense final state, the no-decay and fixed-length variants are
not selectable there, so it gets this direct test.

Contract (from the launcher's documented arguments and the chunked delta-rule math, before reading the
kernel): for each sequence n (token-stream bounds from `cu_seqlens // M`) and head h, from h_0 =
initial_state[n, h] (zeros without one), over the sequence's expanded chunks c of 64 rows:
  if c % M == 0: h[chunk_offsets[n] + c / M] = h_c       (the state entering token chunk c / M)
  v_new_c = u_c - w_c h_c
  h_{c+1} = 2^G_last h_c + sum_r k_r (2^(G_last - G_r) v_new_r)^T     (no decay without g)
The final state goes to state[state_indices[n]] in the cache dtype (skipped for -1), or to a dense fp32
final_state [N, H, K, V]. h is in `states_dtype` (default the activation dtype), v_new like u. Rows of h and
v_new outside every sequence are not written; slots no sequence maps to are unchanged; inputs read-only.
Preconditions (asserted): K <= 256; the cache's (K, V) contiguous; T % M == 0; `state_indices` requires
`state`.

Numerics: the state is carried in fp32; it enters w_c h_c and the decayed v_new enters k^T v_new as
tensor-core operands in the activation dtype (the module docstring of chunk.py: the scan accumulates in
fp32 and rounds on store; FLA's design). Tolerance class: floating point against the fp64 recurrence (no
intermediate rounding); trusted = the recurrence in torch fp32 on the same device with those two operands
rounded to the activation dtype (factor 2). An fp32-operand recurrence is ~10^4x more accurate than any
bf16-operand one on an fp32 state (Triton interpreter), so it cannot calibrate. Unwritten rows, untouched
slots and the chunk-0 entry state (an exact copy of h_0 in `states_dtype`) are exact.

Variants and dispatch (read from the kernel after the contract): heuristics USE_G, USE_INITIAL_STATE,
STORE_FINAL_STATE, HAS_STATE_INDICES, SAVE_NEW_VALUE, IS_VARLEN; constexprs K (64-row register blocks:
K <= 64, 128, 192, 256), V, H, BT, num_householder; autotune over BV in {32, 64} x num_warps {2, 4} x
num_stages {2, 3, 4}. Covered: K in {32, 64, 96, 128, 192, 256} (each register-block count, masked K 96),
V in {16, 32, 64, 96, 128} (one and several BV tiles, masked), H 1 and 4, M 1/2/3, states dtype default and
fp32, initial state fp32 / bf16 / none, final state into an fp32 or bf16 slot cache with -1 padding, or
dense, the three decay regimes and no decay, save_new_value off, fixed-length batches, each BV tile pinned in
turn; chunk boundaries from `delta_rule.STAGE_PACKS`. Production: GDP 800m (32 heads, K 128, V 64, M 3) and
GDN (32 heads, K = V = 128, M 1) on a 4096-token pack with fp32 caches and fp32 chunk states.

Memory: a direct launch into guarded NaN-poisoned h, v_new and cache (unmapped slots and padding rows
poisoned); `sanitizer` on a small case. int32: on main the chunk-state offset
(chunk_offsets[n] * H + h) * K * V and the row offsets bos * H * K / V are int32 products, so chunk
states at or past 2^31 elements are addressed wrongly (finding O7b-M2); a prefill step reaches it at chunk
rows x H x K x V >= 2^31, e.g. 262144 tokens at the GDN default shape (H 32, K = V = 128). #9315
(fix/gdp-chunk-int64-offsets) forms them in int64 and carries those cases as its regression tests
(tests/unit_tests/ssm/ops/gdp/test_chunk_offsets.py, `test_chunk_states_past_int32_offsets`);
`test_chunk_state_offsets_past_int32` (`large`) puts a sequence's chunk states just below 2^31 elements.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.chunk_h import (
    chunk_gated_delta_product_fwd_h,
    chunk_gated_delta_product_fwd_kernel_h_blockdim64,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.gdp import delta_rule as dr

DEVICE = "cuda"
CHUNK = dr.CHUNK
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
KERNEL = chunk_gated_delta_product_fwd_kernel_h_blockdim64


def _recurrence(dtype, M):
    def fn(k, w, u, G, h0, cu, *, rounded):
        rnd = dr.round_to(dtype) if rounded else None
        return dr.chunk_recurrence(
            k, w, u, G, h0, cu, num_householder=M, state_operand=rnd, update_operand=rnd
        )

    return fn


def run_chunk_h(c, g0=None, *, direct=False):
    """Launch for case `c` (a dict) and check every output against the contract."""
    g0 = ou.make_generator(device=DEVICE) if g0 is None else g0
    dtype = DTYPES[c.get("dtype", "bf16")]
    lens, M = dr.STAGE_PACKS[c["pack"]]
    H, K, V = c["H"], c["K"], c["V"]
    use_g = c.get("regime", "production") != "no_decay"
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=H,
        head_k=K,
        head_v=V,
        dtype=dtype,
        regime=c.get("regime", "production") if use_g else "production",
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    N = len(lens)
    init_kind = c.get("init", "fp32")
    h0 = (
        None
        if init_kind == "none"
        else torch.randn(N, H, K, V, generator=g0, device=DEVICE).to(DTYPES[init_kind])
    )
    chain = dr.build_chain(pack, use_g=use_g, device=DEVICE)
    cache = c.get("cache", "fp32")
    states_dtype = DTYPES[c["states"]] if c.get("states") else None

    state = slots = None
    unmapped = []
    if cache != "dense":
        slot_count = N + 3
        perm = torch.randperm(slot_count, generator=ou.make_generator())[:N].tolist()
        slot_list = [s if n_len > 0 else -1 for s, n_len in zip(perm, lens)]
        init = torch.randn(slot_count, H, K, V, generator=g0, device=DEVICE).to(DTYPES[cache])
        unmapped = [s for s in range(slot_count) if s not in slot_list]
        for s in unmapped:
            ou.poison_(init[s])
        state = ou.guarded_like(init)
        slots = torch.tensor(slot_list, dtype=torch.int32, device=DEVICE)

    inputs = [x for x in (chain.k, chain.w, chain.u, chain.Gdp, h0) if x is not None]
    before = [x.clone() for x in inputs]
    save = c.get("save_new_value", True)
    with dr.pinned_autotuning():
        h, v_new, final = chunk_gated_delta_product_fwd_h(
            k=chain.k,
            w=chain.w,
            u=chain.u,
            g=chain.Gdp,
            initial_state=h0,
            output_final_state=cache == "dense",
            chunk_size=CHUNK,
            save_new_value=save,
            cu_seqlens=chain.cu_dp,
            num_householder=M,
            chunk_indices=chain.desc["chunk_indices"],
            chunk_offsets=chain.desc["chunk_offsets"],
            state=None if state is None else state.tensor,
            state_indices=slots,
            states_dtype=states_dtype,
        )
    for x, b in zip(inputs, before):
        ou.assert_exact(x, b, what="inputs are read-only")
    assert h.dtype == (states_dtype or dtype)
    assert (v_new is None) == (not save)

    fn = _recurrence(dtype, M)
    args = (chain.k, chain.w, chain.u, chain.Gdp, h0, chain.cu)
    ref = ou.fp64_reference(fn, *args, rounded=False)
    trusted = dr.trusted_fp32(fn, *args, rounded=True)
    n_rows = ref.entries.shape[0]
    what = f"chunk_h {c}"
    ou.assert_close_to_reference(
        h[0, :n_rows], ref.entries, trusted.entries.to(h.dtype), what=f"h {what}"
    )
    first = chain.desc["chunk_offsets"][:-1].tolist()
    real = [n for n, n_len in enumerate(lens) if n_len > 0]
    h0_or_zero = torch.zeros(N, H, K, V, device=DEVICE) if h0 is None else h0
    ou.assert_exact(
        h[0, [first[n] for n in real]],
        h0_or_zero[real].to(h.dtype),
        what="chunk-0 entry state = h0",
    )
    if save:
        rows = dr.owned_rows(chain.cu_dp, chain.u.shape[1]).to(DEVICE)
        ou.assert_close_to_reference(
            v_new[:, rows],
            ref.v_new[:, rows],
            trusted.v_new[:, rows].to(v_new.dtype),
            what=f"v_new {what}",
        )
    if cache == "dense":
        assert final.dtype == torch.float32 and final.shape == (N, H, K, V)
        ou.assert_close_to_reference(final, ref.final, trusted.final, what=f"final state {what}")
    else:
        assert final is state.tensor
        ou.check_guards(state=state)
        live = [n for n in real]
        live_slots = [slot_list[n] for n in live]
        ou.assert_close_to_reference(
            state.tensor[live_slots],
            ref.final[live],
            trusted.final[live].to(state.tensor.dtype),
            what=f"final state in the cache {what}",
        )
        for s in unmapped:
            ou.assert_poisoned(state.tensor[s], what=f"unmapped slot {s}")
    return chain


CASES = [
    pytest.param(
        f,
        id="{pack}-K{K}-V{V}-H{H}-{dtype}-states_{states}-init_{init}-cache_{cache}-{regime}".format(
            **f
        ),
    )
    for f in ou.pairwise_cases(
        {
            "pack": list(dr.STAGE_PACKS),
            "K": [32, 64, 96, 128, 192, 256],
            "V": [16, 32, 64, 96, 128],
            "H": [1, 4],
            "dtype": ["bf16", "fp16"],
            "states": [None, "fp32"],
            "init": ["fp32", "bf16", "none"],
            "cache": ["fp32", "bf16", "dense"],
            "regime": list(dr.G_REGIMES) + ["no_decay"],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", CASES)
def test_chunk_states_match_fp64(case):
    """Pairwise over packs, K register blocks, V tiles, heads, dtypes, states dtype, initial state, final
    state destination and decay regime."""
    run_chunk_h(case)


@ou.requires_sm(ou.HOPPER)
def test_without_saved_values():
    """save_new_value=False (SAVE_NEW_VALUE off): no v_new, the same states."""
    run_chunk_h(dict(pack="M2", K=128, V=64, H=2, save_new_value=False))


@ou.requires_sm(ou.HOPPER)
def test_every_chunk_h_tile():
    """Each BV tile (32, 64) pinned in turn, at K 256 (four register blocks) and V 96 (masked tile)."""
    tiles = dr.distinct_tiles(KERNEL, ["BV"])
    assert {c.kwargs["BV"] for c in tiles} == {32, 64}
    for config in tiles:
        with dr.pinned(KERNEL, config):
            g0 = ou.make_generator(device=DEVICE)
            lens, M = dr.STAGE_PACKS["M3"]
            pack = dr.make_pack(
                lens,
                num_householder=M,
                heads=2,
                head_k=256,
                head_v=96,
                padded_tokens=sum(lens) + dr.TAIL_TOKENS,
                generator=g0,
                device=DEVICE,
            )
            chain = dr.build_chain(pack, device=DEVICE)
            h0 = torch.randn(len(lens), 2, 256, 96, generator=g0, device=DEVICE)
            h, v_new, final = chunk_gated_delta_product_fwd_h(
                k=chain.k,
                w=chain.w,
                u=chain.u,
                g=chain.Gdp,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=chain.cu_dp,
                num_householder=M,
                chunk_indices=chain.desc["chunk_indices"],
                chunk_offsets=chain.desc["chunk_offsets"],
            )
        fn = _recurrence(torch.bfloat16, M)
        args = (chain.k, chain.w, chain.u, chain.Gdp, h0, chain.cu)
        ref = ou.fp64_reference(fn, *args, rounded=False)
        trusted = dr.trusted_fp32(fn, *args, rounded=True)
        n = ref.entries.shape[0]
        cid = dr.config_id(config)
        ou.assert_close_to_reference(
            h[0, :n], ref.entries, trusted.entries.bfloat16(), what=f"h {cid}"
        )
        ou.assert_close_to_reference(final, ref.final, trusted.final, what=f"final {cid}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("M", [1, 3])
def test_fixed_length_batch(M):
    """No `cu_seqlens` (IS_VARLEN off): B = 2 rows of 2 token chunks and a partial one."""
    g0 = ou.make_generator(device=DEVICE)
    T, H, K, V = 2 * CHUNK + 9, 2, 64, 32
    chains = []
    for _ in range(2):
        pack = dr.make_pack(
            (T,), num_householder=M, heads=H, head_k=K, head_v=V, generator=g0, device=DEVICE
        )
        chains.append(dr.build_chain(pack, device=DEVICE))
    k, w, u, G = (torch.cat([getattr(c, n) for c in chains]) for n in ("k", "w", "u", "Gdp"))
    h0 = torch.randn(2, H, K, V, generator=g0, device=DEVICE)
    with dr.pinned_autotuning():
        h, v_new, final = chunk_gated_delta_product_fwd_h(
            k=k, w=w, u=u, g=G, initial_state=h0, output_final_state=True, num_householder=M
        )
    fn = _recurrence(torch.bfloat16, M)
    for b in range(2):
        s = slice(b, b + 1)
        args = (k[s], w[s], u[s], G[s], h0[s], torch.tensor([0, T], device=DEVICE))
        ref = ou.fp64_reference(fn, *args, rounded=False)
        trusted = dr.trusted_fp32(fn, *args, rounded=True)
        ou.assert_close_to_reference(
            h[b], ref.entries, trusted.entries.bfloat16(), what=f"h row {b}"
        )
        ou.assert_close_to_reference(final[s], ref.final, trusted.final, what=f"final row {b}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.sanitizer
def test_chunk_h_writes_only_owned_rows():
    """A direct launch, as the launcher makes it, into guarded NaN-poisoned h and v_new and a guarded cache:
    h rows of the padded descriptors and v_new rows no sequence owns stay poisoned; padding (-1) and
    unmapped slots untouched (`run_chunk_h` checks the cache); guards intact."""
    g0 = ou.make_generator(device=DEVICE)
    lens, M = dr.STAGE_PACKS["M3"]
    H, K, V = 2, 96, 32
    pack = dr.make_pack(
        lens,
        num_householder=M,
        heads=H,
        head_k=K,
        head_v=V,
        regime="clamp_off",
        padded_tokens=sum(lens) + dr.TAIL_TOKENS,
        generator=g0,
        device=DEVICE,
    )
    chain = dr.build_chain(pack, device=DEVICE)
    N = len(lens)
    h0 = torch.randn(N, H, K, V, generator=g0, device=DEVICE)
    slot_list = [2, 0, 4, 5, 1, -1]
    cache = torch.randn(7, H, K, V, generator=g0, device=DEVICE)
    for s in (3, 6):
        ou.poison_(cache[s])
    state = ou.guarded_like(cache)
    NT = chain.desc["chunk_indices"].shape[0]
    T = chain.k.shape[1]
    h = ou.guarded_empty((1, NT, H, K, V), torch.float32, DEVICE)
    v_new = ou.guarded_empty(tuple(chain.u.shape), chain.u.dtype, DEVICE)
    slots = torch.tensor(slot_list, dtype=torch.int32, device=DEVICE)
    with dr.pinned_autotuning():
        KERNEL[lambda meta: (-(-V // meta["BV"]), N * H)](
            k=chain.k,
            v=chain.u,
            w=chain.w,
            v_new=v_new.tensor,
            g=chain.Gdp,
            h=h.tensor,
            h0=h0,
            ht=state.tensor,
            state_indices=slots,
            ht_slot_stride=state.tensor.stride(0),
            ht_head_stride=state.tensor.stride(1),
            cu_seqlens=chain.cu_dp,
            chunk_offsets=chain.desc["chunk_offsets"],
            num_householder=M,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=CHUNK,
        )
    ou.check_guards(h=h, v_new=v_new, state=state)
    fn = _recurrence(torch.bfloat16, M)
    args = (chain.k, chain.w, chain.u, chain.Gdp, h0, chain.cu)
    ref = ou.fp64_reference(fn, *args, rounded=False)
    trusted = dr.trusted_fp32(fn, *args, rounded=True)
    n_rows = ref.entries.shape[0]
    ou.assert_poisoned(h.tensor[0, n_rows:], what="h rows of the padded descriptors")
    ou.assert_close_to_reference(
        h.tensor[0, :n_rows], ref.entries, trusted.entries, what="h (direct launch)"
    )
    rows = dr.owned_rows(chain.cu_dp, T).to(DEVICE)
    ou.assert_poisoned(v_new.tensor[:, ~rows], what="v_new rows no sequence owns")
    for s in (3, 6):
        ou.assert_poisoned(
            state.tensor[s], what=f"slot {s} (unmapped, or the -1 request's gather target)"
        )
    live = [n for n, s in enumerate(slot_list) if s >= 0]
    ou.assert_close_to_reference(
        state.tensor[[slot_list[n] for n in live]],
        ref.final[live],
        trusted.final[live],
        what="final states (direct launch)",
    )


PRODUCTION = {
    # name: (H, K, V, M, beta fp32)
    "gdp_800m": (32, 128, 64, 3),
    "gdn_default": (32, 128, 128, 1),
}
PRODUCTION_LENS = (2048, 1024, 769, 191, 63, 1)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_chunk_h_production(name):
    """A 4096-token prefill at the production shapes, fp32 slot cache and fp32 chunk states (prefix
    caching), timed autotuning."""
    H, K, V, M = PRODUCTION[name]
    g0 = ou.make_generator(device=DEVICE)
    pack = dr.make_pack(
        PRODUCTION_LENS, num_householder=M, heads=H, head_k=K, head_v=V, generator=g0, device=DEVICE
    )
    chain = dr.build_chain(pack, device=DEVICE)
    N = len(PRODUCTION_LENS)
    cache = torch.randn(N + 2, H, K, V, generator=g0, device=DEVICE)
    slot_list = list(range(2, N + 2))
    h0 = cache[slot_list].clone()
    h, v_new, final = chunk_gated_delta_product_fwd_h(
        k=chain.k,
        w=chain.w,
        u=chain.u,
        g=chain.Gdp,
        initial_state=h0,
        cu_seqlens=chain.cu_dp,
        num_householder=M,
        chunk_indices=chain.desc["chunk_indices"],
        chunk_offsets=chain.desc["chunk_offsets"],
        state=cache,
        state_indices=torch.tensor(slot_list, dtype=torch.int32, device=DEVICE),
        states_dtype=torch.float32,
    )
    fn = _recurrence(torch.bfloat16, M)
    args = (chain.k, chain.w, chain.u, chain.Gdp, h0, chain.cu)
    ref = ou.fp64_reference(fn, *args, rounded=False)
    trusted = dr.trusted_fp32(fn, *args, rounded=True)
    n = ref.entries.shape[0]
    ou.assert_close_to_reference(h[0, :n], ref.entries, trusted.entries, what=f"h {name}")
    ou.assert_close_to_reference(cache[slot_list], ref.final, trusted.final, what=f"final {name}")


ROW_ELEMS = 256 * 256
"""Elements of one chunk-state row at H 1, K = V = 256."""


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "rows_before",
    [
        pytest.param(2**31 // ROW_ELEMS - 1, id="base-fits", marks=pytest.mark.large),
        # A base at 2^31 fails on main (O7b-M2): #9315's test_chunk_states_past_int32_offsets[states].
    ],
)
def test_chunk_state_offsets_past_int32(rows_before):
    """A sequence whose chunk states start at row `rows_before` of h (H 1, K = V = 256: 65536 elements a
    row), after a sequence of zeros `rows_before` chunks long: its base offset (chunk_offsets[n] * H + h)
    * K * V is just below 2^31 elements. A base at exactly 2^31 fails on main (O7b-M2: the offset is an
    int32 product) and is #9315's regression case.
    """
    K = V = 256
    L_a, L_b = rows_before * CHUNK, CHUNK + 1
    ou.require_free_memory(
        (L_a + L_b) * (3 * K + 2 * V) * 2 + (rows_before + 2) * K * V * 2 + 2**30
    )
    g0 = ou.make_generator(device=DEVICE)
    small = dr.make_pack(
        (L_b,), num_householder=1, heads=1, head_k=K, head_v=V, generator=g0, device=DEVICE
    )
    chain = dr.build_chain(small, device=DEVICE)
    T = L_a + L_b

    def embed(x, width):
        out = torch.zeros(1, T, 1, width, dtype=x.dtype, device=DEVICE)
        out[:, L_a:] = x
        return out

    k, w, u = embed(chain.k, K), embed(chain.w, K), embed(chain.u, V)
    G = torch.zeros(1, T, 1, dtype=torch.float32, device=DEVICE)
    G[:, L_a:] = chain.Gdp
    h0 = torch.randn(2, 1, K, V, generator=g0, device=DEVICE)
    cu = torch.tensor([0, L_a, T], dtype=torch.int32, device=DEVICE)
    offsets = torch.tensor([0, rows_before, rows_before + 2], dtype=torch.int32, device=DEVICE)
    h, _, final = chunk_gated_delta_product_fwd_h(
        k=k,
        w=w,
        u=u,
        g=G,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=torch.zeros(rows_before + 2, 2, dtype=torch.int32, device=DEVICE),
        chunk_offsets=offsets,
        states_dtype=torch.bfloat16,
    )
    fn = _recurrence(torch.bfloat16, 1)
    args = (chain.k, chain.w, chain.u, chain.Gdp, h0[1:], chain.cu)
    ref = ou.fp64_reference(fn, *args, rounded=False)
    trusted = dr.trusted_fp32(fn, *args, rounded=True)
    ou.assert_close_to_reference(
        h[0, rows_before : rows_before + 2],
        ref.entries,
        trusted.entries.bfloat16(),
        what=f"chunk states at rows {rows_before}, {rows_before + 1} (past 2^31 elements)",
    )
    ou.assert_close_to_reference(final[1:], ref.final, trusted.final, what="final state")
