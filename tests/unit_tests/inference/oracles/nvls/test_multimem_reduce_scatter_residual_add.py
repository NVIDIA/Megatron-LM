# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_multimem_reduce_scatter_residual_add_kernel`
(megatron/core/inference/communication/torch_symm_triton/fused_collectives.py:115), launched by
`fused_multimem_rs_add_norm_ag` (fused_collectives.py:214): the fused tensor-parallel
reduce-scatter + residual add + RMSNorm + all-gather of `--inference-fuse-tp-communication`.

Contract (from the call site, `InferenceRowParallelLinear._matmul_reduce_scatter`,
inference_layers.py:502-521, and the RMSNorm the next layer would otherwise apply):

  Every rank q of a group of ws ranks holds a partial P_q (bf16 [T, H], the row-parallel GEMM
  output) at offset 0 of its symmetric buffer. Rank r holds the residual shard R_r (bf16
  [T/ws, H]: tokens [r*T/ws, (r+1)*T/ws)). The norm weights w (bf16 [H]) and eps are the same
  on every rank. With S = sum_q P_q and S_r its rows of rank r:

  1. residual_out_r = bf16(bf16(S_r) + R_r): the reduce-scatter accumulates in fp32 and rounds
     once to bf16 (the kernel's `multimem.ld_reduce ... .acc::f32 ... .bf16x2`), then the
     residual is added in bf16 with one rounding, as the unfused path (reduce-scatter, then
     torch.add) does. residual_out may be residual_in (production passes one tensor for both).
     How the switch rounds is checked with designed sums in
     test_multimem_reduce_scatter.py::test_bf16_in_switch_sum_rounding (GB200 does not round
     to nearest even: finding O5-N4).
  2. Every rank's buffer bytes [0, T*H*2) hold Y = RMSNorm(concat_r residual_out_r), row by row:
     y = x * rsqrt(mean(x^2) + eps) * w in fp32, rounded once to bf16. Y is bit-identical on
     every rank. An all-zero row gives y = 0.
  3. Nothing else is written: buffer bytes past T*H*2, residual memory outside its T/ws rows, the
     weights, and residual_in when it is not residual_out.
  4. On return every peer has finished reading this rank's buffer and every rank's stores to it
     are visible, so the buffer may be written (the next GEMM `out=`) and read (the next layer's
     QKV/FC1 input) right away; a following all-gather into it still needs barrier_before (its
     stores could land while a peer reads Y).
  Preconditions, which the launcher must check (it launches nothing otherwise): bf16 tensors;
  T % ws == 0 (every rank owns T/ws tokens); H % 8 == 0 (rows of whole 16-byte lanes); w is bf16
  with H elements; the residual tensors hold T/ws * H elements; ws a power of two.

Tolerance classes:
  * residual_out: bit-exact on ``nu.EXACT`` partials and residuals, whose sums are exact in bf16
    under any order and rounding (the reference is NCCL's fp32 reduce-scatter, exact here, plus
    R, rounded once to nearest even). On N(0, 1) data: bounded (``ou.assert_close_to_reference``,
    per row) against the fp64 S_r + R_r, calibrated by the unfused composition (NCCL fp32
    reduce-scatter rounded to bf16, then torch's bf16 add), which is the contract's arithmetic.
  * Y: bounded per row against fp64 RMSNorm of the kernel's own residual_out (gathered with
    NCCL), calibrated by torch's bf16 ``rms_norm`` on the same device; bit-identical across
    ranks (exact); doubling w doubles Y bit for bit, and zero inputs give zero (exact).
  * The launcher's preconditions: it raises and launches nothing (CPU, kernel replaced).

Structure-derived inputs (kernel and launcher read after the contract above): 8 bf16 per 16-byte
lane, so a row is H/8 lanes; blocks of at most 1024 lanes (multi-window rows for H > 8192) and
grids of at most 128 blocks (grid-stride over tokens beyond 128 per rank). On main the block is
sized from the whole shard (T*H/(8*ws) lanes); #8880 (t3-r09f2) sizes it from one row
(next_pow2(H/8), one block per token). The cases cover both geometries:
  * H around the per-token block's powers of two: 1016/1024/1032 (127/128/129 lanes);
  * H around the 1024-lane window: 8184/8192/8200, 12288 (1.5 windows), 16384 (2), 16392 (3,
    the last with one lane);
  * T/ws around the 128-block grid at H=8192: 127/128/129 tokens per rank, and 257 at H=1152;
  * the shard-sized block cap on main: H=2048 with 4/5 tokens per rank (1024/1280 shard lanes);
  * production: H=1152 (the functional-test model), 4096 with 64 tokens per rank;
  * rows narrower than a warp (H <= 128 with one or two tokens per rank);
  * ws in {2, 4, 8} (subgroups of the job), residual in place (production) and separate,
    eps in {1e-5, 1e-6}; combined pairwise. Every third token's residual has two 1024 outliers
    at lane or window edges (first and last lanes, 8191/8192), which dominate the row's sum of
    squares; with N(0, 1) data, rows scaled by 2**-10 (mean(x^2) ~ eps) and an all-zero last
    token. Each case also checks the NCCL-gathered buffers of all ranks for bit identity.
Memory: residual buffers and weights carry NaN guard bands (an out-of-bounds weight or residual
read turns a row NaN; a write changes the guard), a separate residual_out starts NaN-poisoned
(an unwritten element, or a stage-2 reload hoisted above its stage-1 store, shows up), and the
symmetric buffer has a 64 KiB sentinel past T*H*2. Back-to-back calls interleaved with the plain
all-gather and reduce-scatter on one buffer, with changing token counts, fresh data every step
and random per-rank delays, and CUDA-graph replay with the token count changing between replays.
Int32 offsets cannot overflow: the index math is in 16-byte lanes and the symmetric buffer is
512 MiB.

On main, `test_rows_narrower_than_a_warp` fails where the shard has at most 16 lanes
(T*H/(8*ws) <= 16): the launcher passes num_warps = 0, which Triton rejects. #8880 (t3-r09f2)
fixes it. `test_launcher_refuses_inputs_outside_the_contract` fails on main and on every open
PR: the launcher checks none of those preconditions (suspected findings O5-N1, O5-N2).
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.communication.torch_symm_triton import fused_collectives
from megatron.core.inference.communication.torch_symm_triton.collectives import (
    multimem_all_gather,
    multimem_reduce_scatter,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu

BF16 = torch.bfloat16
GUARD_BYTES = 32 * 1024
"""Guard band around residuals and weights: more than one 1024-lane window (16 KiB), the most a
wrong mask can overrun a row by."""

# (H, tokens per rank); see the module docstring for why each is here.
SHAPES = [
    (256, 1),
    (1016, 1),
    (1024, 1),
    (1032, 3),
    (1152, 1),
    (1152, 257),
    (2048, 4),
    (2048, 5),
    (4096, 64),
    (8184, 1),
    (8192, 1),
    (8200, 1),
    (8192, 127),
    (8192, 128),
    (8192, 129),
    (12288, 3),
    (16384, 2),
    (16392, 1),
]
TINY_ROWS = [(8, 1), (64, 1), (128, 1), (128, 2), (8, 129)]


def gpu_test(fn):
    """NVLS needs Hopper or Blackwell; GB200 CI launches the file through launch_on_gb200."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.HOPPER, ou.BLACKWELL)(fn))


def _sweep_params():
    cases = ou.pairwise_cases(
        {
            "shape": SHAPES,
            "ws": nu.GROUP_SIZES,
            "residual": ["inplace", "separate"],
            "data": ["exact", "randn"],
            "eps": [1e-5, 1e-6],
        }
    )
    params = []
    for c in cases:
        hidden, tokens = c["shape"]
        params.append(
            pytest.param(
                c["ws"],
                hidden,
                tokens,
                c["residual"],
                c["data"],
                c["eps"],
                id=f"ws{c['ws']}-H{hidden}-tpr{tokens}-{c['residual']}-{c['data']}-eps{c['eps']:g}",
            )
        )
    return params


# ---------------------------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------------------------


def rms_norm_reference(x, w, eps):
    """RMSNorm from its definition (fp64 under ``ou.fp64_reference``)."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def residual_reference(partials, residual):
    """S_r + R_r: the shard's sum over ranks, plus the residual (fp64)."""
    return partials.sum(0) + residual


OUTLIER = 1024.0
"""2**10, exact in bf16, placed in the residual. Its square dominates its row's sum of squares
(H * E[x^2] is at most about 1.5e5 here), so a lane left out of the sum, or counted twice, moves
rsqrt(mean(x^2)) by far more than one bf16 ulp. It sits in the residual, not in the partials, so
the in-switch sum stays small: its rounding is the subject of test_bf16_in_switch_sum_rounding,
not of this sweep."""


def structural_columns(hidden):
    """Columns at lane and 1024-lane window edges, most telling first (each lane is 8 columns)."""
    cols = [hidden - 1, 8191, 8192, 0, hidden - 8, hidden - 9, 8, 7, hidden // 2]
    cols += [c for k in range(2, hidden // 8192 + 1) for c in (8192 * k - 1, 8192 * k)]
    return list(dict.fromkeys(c for c in cols if 0 <= c < hidden))


def make_inputs(ws, rank, tokens, hidden, data, seed, device="cuda"):
    """Partials (distinct per rank), rank `rank`'s residual shard and the shared norm weights.

    "exact": partials and residuals from ``nu.EXACT``, whose sums are exact in bf16 in any order
    and rounding, so the reduce-scatter's result is the exact sum on every architecture; the
    residual add then rounds once (on the SM, to nearest even). "randn": N(0, 1); the token
    after every outlier token is scaled by 2**-10 (mean(x^2) ~ eps, so a wrong eps shows) and
    the last token is all zero (y = 0 through rsqrt(eps)). In both, every third token's residual
    has an OUTLIER at two of the structural columns (the first such token at the last column
    and at the 1024-lane window edge).
    """
    tokens_per_rank = tokens // ws
    gen = ou.make_generator(nu.rank_seed(seed, rank), device=device)
    shared = ou.make_generator(seed, device=device)
    shape, shard_shape = (tokens, 1, hidden), (tokens_per_rank, 1, hidden)
    if data == "zeros":
        partial = torch.zeros(shape, dtype=BF16, device=device)
        residual = torch.zeros(shard_shape, dtype=BF16, device=device)
    elif data == "exact":
        partial = ou.dyadic_tensor(shape, device=device, generator=gen, **nu.EXACT)
        residual = ou.dyadic_tensor(shard_shape, device=device, generator=gen, **nu.EXACT)
    else:
        partial = torch.randn(shape, device=device, generator=gen).to(BF16)
        residual = torch.randn(shard_shape, device=device, generator=gen).to(BF16)
        small = torch.arange(1, tokens, 3, device=device)
        partial[small] *= 2**-10
        local_small = small[(small // tokens_per_rank) == rank] - rank * tokens_per_rank
        residual[local_small] *= 2**-10
        partial[-1] = 0
        if rank == ws - 1:
            residual[-1] = 0
    if data != "zeros":
        columns = torch.tensor(structural_columns(hidden))
        rows = torch.arange(0, tokens, 3)
        rows = rows[(rows // tokens_per_rank == rank) & (rows != tokens - 1)]
        for k in (0, 1):
            cols = columns[(2 * (rows // 3) + k) % len(columns)]
            residual[(rows - rank * tokens_per_rank).to(device), 0, cols.to(device)] = OUTLIER
    weight = torch.randn(hidden, device=device, generator=shared).to(BF16)
    return partial, residual, weight


def _inputs(g, tokens, hidden, data, seed):
    return make_inputs(g.ws, g.rank, tokens, hidden, data, seed)


class FusedCall:
    """One call of the launcher on guarded buffers, with this rank's partial in its buffer."""

    def __init__(self, g, partial, residual, weight, eps, residual_mode):
        self.residual_mode = residual_mode
        self.residual = residual
        self.weight = weight
        symm = g.buffer()
        self.view = symm.view(partial.shape, BF16)
        self.view.copy_(partial)
        self.sentinel = symm.sentinel_after(partial.numel() * partial.element_size())
        self.weight_buf = ou.guarded_like(weight, guard_bytes=GUARD_BYTES)
        if residual_mode == "inplace":
            self.residual_in = self.residual_out = ou.guarded_like(
                residual, guard_bytes=GUARD_BYTES
            )
        else:
            self.residual_in = ou.guarded_like(residual, guard_bytes=GUARD_BYTES)
            self.residual_out = ou.guarded_empty(residual.shape, BF16, guard_bytes=GUARD_BYTES)
        fused_collectives.fused_multimem_rs_add_norm_ag(
            self.residual_out.tensor,
            self.view,
            symm.hdl,
            self.residual_in.tensor,
            self.weight_buf.tensor,
            eps,
        )

    def check_memory(self):
        guards = dict(residual_out=self.residual_out, weights=self.weight_buf)
        if self.residual_mode == "separate":
            guards["residual_in"] = self.residual_in
            ou.assert_exact(self.residual_in.tensor, self.residual, what="residual_in (read-only)")
        ou.check_guards(**guards)
        ou.assert_exact(self.weight_buf.tensor, self.weight, what="weights (read-only)")
        self.sentinel.check("symmetric buffer past T*H*2")


def check_residual(out, partials_shard, residual, data, nccl_rs):
    """residual_out of this rank against the reference (exact on ``nu.EXACT`` data)."""
    if data in ("exact", "zeros"):
        # The sum is exact in bf16 (nu.EXACT), so NCCL's fp32 sum is it; the residual add is an
        # exact fp32 add of two bf16 values rounded once to nearest even (contract item 1).
        expected = (nccl_rs.to(BF16).float() + residual.float()).to(BF16)
        ou.assert_exact(out, expected, what="residual_out (exact sum, one rounding)")
        return
    hidden = out.shape[-1]
    ou.assert_close_to_reference(
        out.reshape(-1, hidden),
        ou.fp64_reference(residual_reference, partials_shard, residual).reshape(-1, hidden),
        (nccl_rs.to(BF16) + residual).reshape(-1, hidden),
        dim=-1,
        what="residual_out (reduce-scatter + residual add)",
    )


def check_normalized(y, gathered_y, gathered_residual_out, weight, eps):
    """Y on this rank against fp64 RMSNorm of every rank's residual_out, and across ranks."""
    hidden = y.shape[-1]
    x = gathered_residual_out.reshape(-1, hidden)
    ou.assert_close_to_reference(
        y.reshape(-1, hidden),
        ou.fp64_reference(rms_norm_reference, x, weight, eps),
        F.rms_norm(x, (hidden,), weight, eps),
        dim=-1,
        what="normalized all-gathered output",
    )
    for q in range(gathered_y.shape[0]):
        ou.assert_exact(gathered_y[q], y, what=f"rank {q}'s buffer against this rank's")


def run_and_check(g, hidden, tokens_per_rank, residual_mode, data, eps, what):
    """One fused call checked against the NCCL / fp64 references on every rank."""
    tokens = tokens_per_rank * g.ws
    seed = nu.case_seed(what, g.ws, hidden, tokens_per_rank, residual_mode, data, eps)
    partial, residual, weight = _inputs(g, tokens, hidden, data, seed)
    shard = slice(g.rank * tokens_per_rank, (g.rank + 1) * tokens_per_rank)
    partials_shard = g.gather(partial)[:, shard]
    nccl_rs = g.reduce_scatter_fp32(partial)
    call = FusedCall(g, partial, residual, weight, eps, residual_mode)
    y = call.view.clone()
    gathered_residual_out = g.gather(call.residual_out.tensor)
    gathered_y = g.gather(y)

    def check():
        call.check_memory()
        check_residual(call.residual_out.tensor, partials_shard, residual, data, nccl_rs)
        check_normalized(y, gathered_y, gathered_residual_out, weight, eps)

    nu.agree(check, what)
    return call.residual_out.tensor.clone(), y


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@gpu_test
@pytest.mark.parametrize("ws,hidden,tokens_per_rank,residual_mode,data,eps", _sweep_params())
def test_fused_matches_nccl_and_fp64(
    nvls_groups, ws, hidden, tokens_per_rank, residual_mode, data, eps
):
    """residual_out, Y on every rank, and the untouched memory, across the structure sweep."""
    g = nvls_groups.get(ws)
    run_and_check(g, hidden, tokens_per_rank, residual_mode, data, eps, what="fused sweep")


@gpu_test
@pytest.mark.sanitizer
def test_sanitizer_small_case(nvls_groups):
    """Small case for compute-sanitizer, the same on every world size: ws 2, H 8200 (1025 lanes:
    a full 1024-lane window, then a window with one live lane) and 3 tokens per rank (4 blocks
    on main's shard-sized launch, the last with no token), separate residual buffers (input
    read-only, output NaN-poisoned) and N(0, 1) data with the all-zero last token."""
    run_and_check(nvls_groups.get(2), 8200, 3, "separate", "randn", 1e-5, what="sanitizer")


@gpu_test
@pytest.mark.parametrize(
    "hidden,tokens_per_rank", [pytest.param(h, t, id=f"H{h}-tpr{t}") for h, t in TINY_ROWS]
)
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_rows_narrower_than_a_warp(nvls_groups, ws, hidden, tokens_per_rank):
    """Rows of 1 to 16 lanes. Fails on main when the whole shard is at most 16 lanes
    (T*H/(8*ws) <= 16: H=8, 64, 128 with one token per rank): the shard-sized launch passes
    num_warps = block_size // 32 = 0 (fused_collectives.py:259-264), which Triton rejects.
    #8880 (t3-r09f2) launches at least one warp per block; these cases must pass with it.
    (128, 2) and (8, 129) pass on main."""
    g = nvls_groups.get(ws)
    run_and_check(g, hidden, tokens_per_rank, "separate", "exact", 1e-5, what="narrow rows")


@gpu_test
@pytest.mark.parametrize("hidden,tokens_per_rank", [(1152, 3), (8200, 2)])
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_metamorphic_weight_scaling_and_zero_input(nvls_groups, ws, hidden, tokens_per_rank):
    """Doubling w doubles Y bit for bit and leaves residual_out unchanged; zero partials and a
    zero residual give zero everywhere (rrms = rsqrt(eps) is finite)."""
    g = nvls_groups.get(ws)
    tokens = tokens_per_rank * g.ws
    seed = nu.case_seed("metamorphic", ws, hidden, tokens_per_rank)
    partial, residual, weight = _inputs(g, tokens, hidden, "exact", seed)
    eps = 1e-5
    first = FusedCall(g, partial, residual, weight, eps, "separate")
    y_first = first.view.clone()
    doubled = FusedCall(g, partial, residual, weight * 2, eps, "separate")
    y_doubled = doubled.view.clone()
    zero_partial, zero_residual, _ = _inputs(g, tokens, hidden, "zeros", seed)
    zero = FusedCall(g, zero_partial, zero_residual, weight, eps, "inplace")
    y_zero = zero.view.clone()

    def check():
        for call in (first, doubled, zero):
            call.check_memory()
        ou.assert_exact(
            doubled.residual_out.tensor, first.residual_out.tensor, what="residual_out with 2w"
        )
        ou.assert_exact(y_doubled, (y_first.float() * 2).to(BF16), what="Y with 2w == 2 * Y")
        ou.assert_exact(zero.residual_out.tensor, zero_residual, what="residual_out of zeros")
        # 0 * w is -0.0 where w < 0, so compare magnitudes.
        ou.assert_exact(y_zero.abs(), torch.zeros_like(y_zero), what="Y of zeros")

    nu.agree(check, "metamorphic")


@gpu_test
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_back_to_back_calls_reuse_the_buffer(nvls_groups, ws):
    """The decoder's order on one symmetric buffer, step after step: fused (a fused-TP layer) ->
    all-gather with barrier_before (it follows the fused kernel's all-gather) -> reduce-scatter ->
    all-gather of the same per-rank size without a barrier (allowed only after an equal-size
    reduce-scatter). Token counts and data change every step and one random rank is delayed
    before every collective, so a missing barrier would read stale or half-written data. No
    NCCL call runs inside the loop; every step is checked afterwards."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    hidden, eps, steps = 1152, 1e-5, 8
    shared = g.shared_generator(nu.case_seed("back-to-back", ws))
    tokens_per_rank = [int(t) for t in torch.randint(1, 70, (steps,), generator=shared)]
    _, _, weight = _inputs(g, ws, hidden, "exact", nu.case_seed("back-to-back-weight", ws))
    plan = []
    for step, tpr in enumerate(tokens_per_rank):
        seed = nu.case_seed("back-to-back", ws, step)
        partial, residual, _ = _inputs(g, tpr * ws, hidden, "exact", seed)
        gen = g.rank_generator(seed + 1)
        gather_in = ou.dyadic_tensor((tpr, 1, hidden), device="cuda", generator=gen, **nu.EXACT)
        rs_in = ou.dyadic_tensor((tpr * ws, 1, hidden), device="cuda", generator=gen, **nu.EXACT)
        shard = slice(g.rank * tpr, (g.rank + 1) * tpr)
        plan.append(
            SimpleNamespace(
                tpr=tpr,
                partial=partial,
                residual=residual,
                gather_in=gather_in,
                rs_in=rs_in,
                nccl_rs=g.reduce_scatter_fp32(partial),
                gathered_gather_in=g.gather(gather_in),
                rs_reference=g.reduce_scatter_fp32(rs_in).to(BF16),
                partials_shard=g.gather(partial)[:, shard],
            )
        )

    results = []
    for step in plan:
        view = symm.view((step.tpr * ws, 1, hidden), BF16)
        view.copy_(step.partial)
        residual = step.residual.clone()
        g.maybe_sleep(shared)
        fused_collectives.fused_multimem_rs_add_norm_ag(
            residual, view, symm.hdl, residual, weight, eps
        )
        y = view.clone()
        g.maybe_sleep(shared)
        multimem_all_gather(view, step.gather_in, symm.hdl, barrier_before=True)
        gathered = view.clone()
        view.copy_(step.rs_in)
        rs_out = torch.empty_like(step.gather_in)
        g.maybe_sleep(shared)
        multimem_reduce_scatter(rs_out, view, symm.hdl)
        g.maybe_sleep(shared)
        multimem_all_gather(view, rs_out, symm.hdl, barrier_before=False)
        regathered = view.clone()
        results.append(
            SimpleNamespace(
                residual=residual, y=y, gathered=gathered, rs_out=rs_out, regathered=regathered
            )
        )
    for res in results:
        res.gathered_residual = g.gather(res.residual)
        res.gathered_y = g.gather(res.y)
        res.gathered_rs = g.gather(res.rs_out)

    def check():
        for i, (step, res) in enumerate(zip(plan, results)):
            check_residual(res.residual, step.partials_shard, step.residual, "exact", step.nccl_rs)
            check_normalized(res.y, res.gathered_y, res.gathered_residual, weight, eps)
            ou.assert_exact(
                res.gathered,
                step.gathered_gather_in.reshape(res.gathered.shape),
                what=f"step {i}: all-gather after the fused kernel",
            )
            ou.assert_exact(res.rs_out, step.rs_reference, what=f"step {i}: reduce-scatter")
            ou.assert_exact(
                res.regathered,
                res.gathered_rs.reshape(res.regathered.shape),
                what=f"step {i}: equal-size all-gather after the reduce-scatter",
            )

    nu.agree(check, "back-to-back fused / all-gather / reduce-scatter")


@gpu_test
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_cuda_graph_replay_with_changing_token_count(nvls_groups, ws):
    """One CUDA graph per token count (as inference captures per batch size), replayed in a
    changing order with fresh data each time and a random rank delayed before each replay.
    Each replay must give the eager contract: grids, pointers and the barrier's signal slots are
    baked into the graphs, and graphs of different sizes share the buffer and the slots."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    hidden, eps, replays = 1152, 1e-6, 12
    sizes = [1, 3, 64, 129]
    shared = g.shared_generator(nu.case_seed("graph", ws))
    _, _, weight = _inputs(g, ws, hidden, "exact", nu.case_seed("graph-weight", ws))
    graphs = {}
    for tpr in sizes:
        static = SimpleNamespace(
            partial=torch.zeros(tpr * ws, 1, hidden, dtype=BF16, device="cuda"),
            residual_init=torch.zeros(tpr, 1, hidden, dtype=BF16, device="cuda"),
            residual=torch.zeros(tpr, 1, hidden, dtype=BF16, device="cuda"),
            y=torch.zeros(tpr * ws, 1, hidden, dtype=BF16, device="cuda"),
            view=symm.view((tpr * ws, 1, hidden), BF16),
        )

        def body(s=static):
            s.view.copy_(s.partial)
            s.residual.copy_(s.residual_init)
            fused_collectives.fused_multimem_rs_add_norm_ag(
                s.residual, s.view, symm.hdl, s.residual, weight, eps
            )
            s.y.copy_(s.view)

        body()  # compile and warm up eagerly
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        graphs[tpr] = (graph, static)
    nu.sync_world()

    order = [sizes[int(i)] for i in torch.randint(0, len(sizes), (replays,), generator=shared)]
    plan = []
    for i, tpr in enumerate(order):
        partial, residual, _ = _inputs(g, tpr * ws, hidden, "exact", nu.case_seed("graph", ws, i))
        shard = slice(g.rank * tpr, (g.rank + 1) * tpr)
        plan.append(
            SimpleNamespace(
                tpr=tpr,
                partial=partial,
                residual=residual,
                nccl_rs=g.reduce_scatter_fp32(partial),
                partials_shard=g.gather(partial)[:, shard],
            )
        )
    results = []
    for step in plan:
        graph, static = graphs[step.tpr]
        static.partial.copy_(step.partial)
        static.residual_init.copy_(step.residual)
        g.maybe_sleep(shared)
        graph.replay()
        results.append(SimpleNamespace(residual=static.residual.clone(), y=static.y.clone()))
    for res in results:
        res.gathered_residual = g.gather(res.residual)
        res.gathered_y = g.gather(res.y)

    def check():
        for step, res in zip(plan, results):
            check_residual(res.residual, step.partials_shard, step.residual, "exact", step.nccl_rs)
            check_normalized(res.y, res.gathered_y, res.gathered_residual, weight, eps)

    nu.agree(check, f"CUDA-graph replays in the order {order}")


class _LaunchRecorder:
    """Stands in for the Triton kernel: records launches instead of running them."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((grid, args, kwargs))


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "tokens-not-a-multiple-of-ws",
        "hidden-not-a-multiple-of-8",
        "weights-not-bf16",
        "weights-not-hidden-long",
        "residual-not-shard-sized",
    ],
)
def test_launcher_refuses_inputs_outside_the_contract(monkeypatch, case):
    """The launcher must reject inputs the kernel cannot process, before launching; "valid" is
    the control (one launch).

    The other cases fail on main and with every open PR: it checks only dtypes and numel % 8.
      * T % ws != 0 (suspected finding O5-N1): the kernel gives each rank cdiv(T, ws) tokens, so
        the last rank adds and normalizes rows past T and multicasts them past T*H*2 into every
        rank's buffer, and reads and writes its residual past T/ws rows.
      * H % 8 != 0 with T*H % 8 == 0 (O5-N2): rows are cdiv(H, 8) lanes, so every row after the
        first is misaligned and reads the next row's elements.
      * Norm weights that are not bf16, or not H long (O5-N2): the kernel reads them as H bf16
        values, silently misdecoding fp32 weights.
      * Residual tensors that are not T/ws * H elements (O5-N2): read and written out of bounds.
    Runs on CPU: the kernel is replaced by a recorder and the handle by a namespace.
    """
    pytest.importorskip("triton")
    recorder = _LaunchRecorder()
    monkeypatch.setattr(fused_collectives, "_multimem_reduce_scatter_residual_add_kernel", recorder)
    ws, tokens, hidden = 4, 8, 1024
    weight_dtype, weight_len, residual_rows = BF16, None, None
    if case == "valid":
        pass
    elif case == "tokens-not-a-multiple-of-ws":
        tokens = 10
    elif case == "hidden-not-a-multiple-of-8":
        hidden = 12
    elif case == "weights-not-bf16":
        weight_dtype = torch.float32
    elif case == "weights-not-hidden-long":
        weight_len = hidden + 8
    else:
        residual_rows = tokens // ws + 1
    handle = SimpleNamespace(world_size=ws, rank=ws - 1, multicast_ptr=0, signal_pad_ptrs_dev=0)
    symm = torch.zeros(tokens, 1, hidden, dtype=BF16)
    residual = torch.zeros(residual_rows or math.ceil(tokens / ws), 1, hidden, dtype=BF16)
    weight = torch.ones(weight_len or hidden, dtype=weight_dtype)

    def launch():
        fused_collectives.fused_multimem_rs_add_norm_ag(
            residual, symm, handle, residual, weight, 1e-5
        )

    if case == "valid":
        launch()
        assert len(recorder.launches) == 1
        return
    with pytest.raises((AssertionError, ValueError, TypeError)):
        launch()
    assert not recorder.launches, f"{case}: the kernel was launched"
