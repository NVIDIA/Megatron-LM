# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_unpermute_tokens_kernel`` and ``_zero_output_rows_kernel``
(megatron/core/inference/moe/permute.py), through their launcher ``unpermute_tokens``.

``unpermute_tokens`` (non-batch-invariant branch; the batch-invariant branch is another family)
zeroes the live output rows and scatters the weighted expert outputs back to token order, on
every non-BI torch-backend / MXFP8 MoE (paths P1, P3, P6). Under NVLS EP > 1, ``out`` is the
fp32 RSV symmetric buffer, which holds stale rows from earlier layers and steps.

Contract (from the launcher's documented API):

* For every token ``t < valid``::

      out[t, :] = sum over rows r < n_used with permutation_map[r] == t of
                  permuted_probs[r] * float32(expert_output[r, :])

  accumulated in fp32 (``0`` when no row maps to ``t``). Rows with ``permutation_map == -1``
  (alignment padding) and rows ``>= n_used`` (stale data from a larger earlier step) are not
  read. Output rows ``>= valid`` are not written: a caller-provided ``out`` keeps them.
* Inputs: ``expert_output`` ``[output_size, H]`` (bf16 from the grouped GEMM);
  ``permuted_probs`` ``[output_size]`` fp32 (asserted); ``permutation_map`` ``[output_size]``
  int32; ``n_used``, ``valid`` int32 device scalars; ``num_tokens`` the output height; ``out``
  optional ``[num_tokens, H]`` fp32 contiguous. Returns ``out`` (or a new fp32 buffer).
  Precondition: live map entries ``< valid <= num_tokens``.
* Numerics: fp32 products and fp32 sums in unspecified order.
* Tolerance class: **exact** on dyadic inputs (every product and partial sum is exact in fp32,
  bounds in ``ou.dyadic_tensor``), so any order must give the fp64 result; on random bf16 inputs,
  **bounded** by ``torch.index_add_`` in fp32 on the same device against an fp64 reference
  (factor 2: both are fp32 products summed in some order).

Oracle: a per-token fp64 sum written from the formula above (``index_add_`` in fp64).

Structure (read after the contract): zero kernel: ``BLOCK_H = min(next_pow2(H), 1024)``,
``NUM_BLOCKS = min(num_tokens, 512)``, gated by ``valid``; unpermute: same ``BLOCK_H``,
``NUM_BLOCKS = min(output_size, 512)``, gated by ``n_used``, fp32 atomics. Inputs therefore
cover ``H`` {1, 33, 1023, 1024, 1025, 2688}, ``num_tokens`` and ``output_size`` at 511/512/513,
``valid`` {0, 1, num_tokens-1, num_tokens}, up to 1, 2 or 8 contributions per token (three or
more race in the atomics), padding rows, a stale tail past ``n_used`` whose map entries point at
live tokens with large values, ``n_used`` 0, launcher-allocated and caller-provided (poisoned,
guarded) ``out``, pairwise; production shapes; CUDA-graph replay of the fixed-buffer (RSV) form
with ``n_used`` and ``valid`` changing; opt-in (``large``), expert-row and output-row element
offsets at 2^31 (``H = 7168``).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

permute = pytest.importorskip("megatron.core.inference.moe.permute")
pytest.importorskip("triton")

STALE_VALUE = 2.0**12
"""Dyadic value of every padding / stale row: reading one changes the sum exactly and visibly."""


def build_inputs(num_tokens, valid, max_contrib, hidden_dim, *, stale_rows, dyadic, seed=0, dev):
    """Permuted rows for ``valid`` tokens (0..max_contrib rows each, shuffled, with ``-1``
    padding rows interleaved), followed by ``stale_rows`` rows past ``n_used`` whose map entries
    point at live tokens. Indices are drawn on the CPU, values on ``dev``."""
    g = ou.make_generator(ou.DEFAULT_SEED + seed)
    gd = ou.make_generator(ou.DEFAULT_SEED + seed, device=dev)
    per_token = torch.randint(0, max_contrib + 1, (max(valid, 0),), generator=g)
    tokens = torch.repeat_interleave(torch.arange(max(valid, 0)), per_token)
    if tokens.numel():
        tokens = torch.cat([tokens, torch.full((tokens.numel() // 6 + 1,), -1, dtype=torch.int64)])
    rows = tokens[torch.randperm(tokens.numel(), generator=g)]
    n_used = rows.numel()
    stale = torch.randint(0, max(valid, 1), (stale_rows,), generator=g)
    perm_map = torch.cat([rows, stale]).to(dtype=torch.int32, device=dev)
    size = perm_map.numel()
    if dyadic:
        expert = ou.dyadic_tensor(
            (size, hidden_dim), mantissa_bits=4, exponents=(-2, 2), device=dev, generator=gd
        )
        probs = ou.dyadic_tensor(
            (size,),
            mantissa_bits=4,
            exponents=(-4, 0),
            dtype=torch.float32,
            device=dev,
            generator=gd,
        )
    else:
        expert = torch.randn(size, hidden_dim, device=dev, generator=gd).to(torch.bfloat16)
        probs = torch.rand(size, device=dev, generator=gd)
    skipped = perm_map < 0
    skipped[n_used:] = True
    expert[skipped] = STALE_VALUE
    probs[skipped] = 1.0
    return expert, probs, perm_map, n_used


def reference(expert, probs, perm_map, n_used, valid, num_tokens):
    """fp64 ``out[t] = sum_r probs[r] * expert[r]`` over live rows mapped to ``t < valid``."""
    live = perm_map[:n_used].long()
    keep = live >= 0
    weighted = probs[:n_used][keep, None] * expert[:n_used][keep]
    out = torch.zeros(num_tokens, expert.shape[1], dtype=weighted.dtype, device=expert.device)
    return out.index_add_(0, live[keep], weighted)[: max(valid, 0)]


def run_and_check(
    num_tokens, valid, max_contrib, hidden_dim, *, stale_rows, provided_out, dyadic=True, seed=0
):
    dev = rr.device()
    expert, probs, perm_map, n_used = build_inputs(
        num_tokens,
        valid,
        max_contrib,
        hidden_dim,
        stale_rows=stale_rows,
        dyadic=dyadic,
        seed=seed,
        dev=dev,
    )
    before = (expert.clone(), probs.clone(), perm_map.clone())
    buf = ou.guarded_empty((num_tokens, hidden_dim), torch.float32, dev) if provided_out else None
    with rr.poisoned_outputs():
        out = permute.unpermute_tokens(
            expert,
            probs,
            perm_map,
            num_tokens,
            rr.valid_tensor(n_used, dev),
            rr.valid_tensor(valid, dev),
            out=buf.tensor if provided_out else None,
        )
    what = (
        f"T={num_tokens} valid={valid} n_used={n_used} size={perm_map.numel()} H={hidden_dim}"
        f" contrib<={max_contrib} out={'provided' if provided_out else 'launcher'}"
    )
    assert out.dtype == torch.float32 and out.shape == (num_tokens, hidden_dim), what
    if provided_out:
        assert out.data_ptr() == buf.tensor.data_ptr(), what
        ou.check_guards(out=buf)
    ref = ou.fp64_reference(reference, expert, probs, perm_map, n_used, valid, num_tokens)
    if dyadic:
        ou.assert_exact(out[:valid], ref.to(torch.float32), what=f"live rows ({what})")
    else:
        # Trusted: the same formula in fp32 with torch's index_add_ on the same device.
        trusted = reference(expert.float(), probs, perm_map, n_used, valid, num_tokens)
        ou.assert_close_to_reference(out[:valid], ref, trusted, dim=-1, what=f"live rows ({what})")
    rr.assert_untouched(out[valid:], what=f"output rows >= valid ({what})")
    for name, now, then in zip(
        ("expert_output", "probs", "permutation_map"), (expert, probs, perm_map), before
    ):
        ou.assert_exact(now, then, what=f"{name} is read-only")


_VARIANTS = ou.pairwise_cases(
    {
        "hidden_dim": [1, 33, 1023, 1024, 1025, 2688],
        "num_tokens": [1, 511, 512, 513],
        "valid": ["all", "all-1", "0", "1"],
        "max_contrib": [1, 2, 8],
        "stale_rows": [0, 1, 300],
        "out": ["launcher", "provided"],
    }
)


@pytest.mark.parametrize(
    "case",
    _VARIANTS,
    ids=[
        f"H{c['hidden_dim']}-T{c['num_tokens']}-v{c['valid']}-c{c['max_contrib']}"
        f"-stale{c['stale_rows']}-{c['out']}"
        for c in _VARIANTS
    ],
)
def test_variants_pairwise_exact(case):
    """Dyadic inputs: exact for any atomic order, including three or more rows per token."""
    t = case["num_tokens"]
    valid = {"all": t, "all-1": t - 1, "0": 0, "1": 1}[case["valid"]]
    run_and_check(
        t,
        valid,
        case["max_contrib"],
        case["hidden_dim"],
        stale_rows=case["stale_rows"],
        provided_out=case["out"] == "provided",
    )


@pytest.mark.parametrize("target", [511, 512, 513])
def test_output_size_at_grid_cap(target):
    """``output_size`` (rows of the permuted buffer) around the 512-program cap."""
    # One row per token plus padding: pick the stale tail so the buffer has ``target`` rows.
    dev = rr.device()
    _, _, perm_map, n_used = build_inputs(400, 400, 1, 8, stale_rows=0, dyadic=True, dev=dev)
    run_and_check(400, 400, 1, 64, stale_rows=max(target - perm_map.numel(), 0), provided_out=True)


def test_nothing_routed_here():
    """``n_used = 0``: the live rows read back zero, rows past ``valid`` untouched."""
    run_and_check(64, 50, 0, 128, stale_rows=40, provided_out=True)


@pytest.mark.parametrize("hidden_dim", [128, 2688])
@pytest.mark.parametrize("max_contrib", [2, 8])
def test_random_bf16_bounded_by_torch(hidden_dim, max_contrib):
    """Random bf16 expert outputs: fp32 products and sums, within 2x ``index_add_``'s error."""
    run_and_check(
        200, 190, max_contrib, hidden_dim, stale_rows=50, provided_out=False, dyadic=False
    )


@pytest.mark.parametrize(
    "num_tokens,hidden_dim,max_contrib",
    [
        pytest.param(8192, 2688, 6, id="nanov3-prefill"),
        pytest.param(8192, 7168, 8, id="dsv3-prefill"),
        pytest.param(8 * 8192, 2688, 2, id="nanov3-ep8-agv-rows"),
    ],
)
def test_production_sizes(num_tokens, hidden_dim, max_contrib):
    """Production shapes (NVLS ``global_max`` rows for the last), caller-provided buffer."""
    rr.require_cuda()
    run_and_check(
        num_tokens, num_tokens - 123, max_contrib, hidden_dim, stale_rows=4096, provided_out=True
    )


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    run_and_check(40, 37, 3, 96, stale_rows=10, provided_out=True)


def test_cuda_graph_replay_fixed_buffer():
    """The NVLS form: a fixed ``out`` (RSV buffer) captured once, replayed with ``n_used`` and
    ``valid`` changing and the permuted inputs rewritten in place; ``out`` re-poisoned before
    every replay so rows past ``valid`` must stay untouched."""
    dev = rr.require_cuda()
    num_tokens, hidden_dim, size = 300, 256, 4000
    expert = torch.zeros(size, hidden_dim, dtype=torch.bfloat16, device=dev)
    probs = torch.zeros(size, dtype=torch.float32, device=dev)
    perm_map = torch.zeros(size, dtype=torch.int32, device=dev)
    n_used_t, valid_t = rr.valid_tensor(0, dev), rr.valid_tensor(0, dev)
    out = ou.guarded_empty((num_tokens, hidden_dim), torch.float32, dev)
    args = (expert, probs, perm_map, num_tokens, n_used_t, valid_t)
    permute.unpermute_tokens(*args, out=out.tensor)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        permute.unpermute_tokens(*args, out=out.tensor)
    for step, (valid, contrib) in enumerate(
        [(300, 3), (40, 8), (1, 2), (0, 1), (299, 1), (300, 8)]
    ):
        e, p, m, n_used = build_inputs(
            num_tokens, valid, contrib, hidden_dim, stale_rows=0, dyadic=True, seed=step, dev=dev
        )
        assert n_used <= size
        expert[:n_used].copy_(e)
        probs[:n_used].copy_(p)
        perm_map[:n_used].copy_(m)
        # Rows past n_used keep whatever the previous step left (stale, must be ignored).
        n_used_t.fill_(n_used)
        valid_t.fill_(valid)
        ou.poison_(out.tensor)
        graph.replay()
        torch.cuda.synchronize()
        ou.check_guards(out=out)
        ref = ou.fp64_reference(reference, e, p, m, n_used, valid, num_tokens)
        ou.assert_exact(out.tensor[:valid], ref.to(torch.float32), what=f"replay {step}")
        ou.assert_poisoned(out.tensor[valid:], what=f"rows >= valid, replay {step}")


@pytest.mark.parametrize("num_tokens", ou.int32_overflow_params(7168))
def test_int32_row_offsets(num_tokens):
    """Output rows (``t * H``) and expert rows (``r * H``) whose element offsets cross 2^31.

    One row per token in shuffled order plus ``-1`` padding rows. Values are small integers times
    dyadic probabilities, so every product and sum is exact in fp32 and the fp32 ``index_add_``
    reference equals the fp64 one (an fp64 reference would need another 34 GiB). Opt-in
    ``large``; about 24 GiB.
    """
    dev = rr.require_cuda()
    hidden_dim = 7168
    ou.require_free_memory(num_tokens * hidden_dim * (4 + 2 + 4 + 4) + (4 << 30))
    g = ou.make_generator(device=dev)
    rows = torch.randperm(num_tokens, device=dev, generator=g)
    perm_map = torch.cat([rows, torch.full((num_tokens // 7,), -1, device=dev)]).to(torch.int32)
    perm_map = perm_map[torch.randperm(perm_map.numel(), device=dev, generator=g)]
    size = perm_map.numel()
    expert = torch.randint(-15, 16, (size, hidden_dim), dtype=torch.int8, device=dev, generator=g)
    expert = expert.to(torch.bfloat16)
    probs = ou.dyadic_tensor(
        (size,), mantissa_bits=4, exponents=(-4, 0), dtype=torch.float32, device=dev, generator=g
    )
    out = permute.unpermute_tokens(
        expert,
        probs,
        perm_map,
        num_tokens,
        rr.valid_tensor(size, dev),
        rr.valid_tensor(num_tokens, dev),
    )
    expected = torch.zeros(num_tokens, hidden_dim, dtype=torch.float32, device=dev)
    for chunk in torch.arange(size, device=dev).split(1 << 16):
        rows_c = chunk[perm_map[chunk] >= 0]
        weighted = probs[rows_c, None] * expert[rows_c].float()
        expected.index_add_(0, perm_map[rows_c].long(), weighted)
    ou.assert_exact(out, expected, what=f"int32 offsets num_tokens={num_tokens}")
