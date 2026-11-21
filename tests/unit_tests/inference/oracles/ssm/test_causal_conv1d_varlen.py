# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the fork's `_causal_conv1d_varlen_kernel` (megatron/core/ssm/ops/common/causal_conv1d_varlen.py),
the prefill-time depthwise causal conv with per-request initial states (Mamba, GDN and GDP dynamic prefill),
through its launcher `causal_conv1d_varlen_fn`.

Before this module the kernel had the unit tests of its own PR (pip `causal_conv1d_fn` per request, fp32 and
bf16 at loose tolerances, derived metadata, bias, W = 4) and bitwise parity for one sequence without
initial states; the precomputed (CUDA-graph) metadata, `bias=None`, padded tail tokens, W != 4, the
production initial-state layout and int32 offsets were not checked.

Contract (written from the launcher's docstring and its callers, `MambaMixer.ssm_prefill`,
`GatedDeltaProductMixer._prefill_conv` and `GatedDeltaNet.ssm_prefill`, before reading the kernel body):

* Inputs: x (T, C) packed tokens of the step's prefill requests (contiguous); weight (C, W); bias (C,) or
  None; cu_seqlens (N + 1,) int32; initial_states (N, C, W - 1) or None — production passes
  `conv_state[batch_indices][:, :, 1:]`, a view whose rows are W apart (last stride 1); the activation
  "silu"; optionally precomputed per-token metadata `seq_idx` (request) and `seq_start` (its first token),
  at least T long (CUDA graphs; `MambaMetadata` writes 0 / 0 for padded tail tokens), else derived from
  cu_seqlens.
* For a token t of request r (tokens [s_r, e_r)), with the request's history h_r = [initial_states[r]
  (W - 1 columns, oldest first) or zeros, x[s_r], ..., x[t]]:

      out[t, c] = silu(bias[c] + sum_{j < W} weight[c, j] * h_r[c, last W columns][j])

  so taps before s_r come from the request's own initial state, never from another request. out has
  x's dtype and shape. Tokens past the real ones (padded tail tokens) get an unspecified output; real
  tokens never read them. Inputs are read-only.
* Numerics: fp32 accumulation of W products and the bias, SiLU in fp32, one rounding to x's dtype; x and
  the initial states may differ in dtype (GDP: bf16 x, fp32 states).
* Tolerance class: floating point (reference: fp64; trusted: the same math in torch fp32 on the device);
  exact for the read-only checks and the guard bands; a NaN in one request's token reaches exactly that
  request's next W outputs.

Variants and dispatch (from the kernel, read after the contract): the two autotune configs (BLOCK_T 8 x
BLOCK_C 256, BLOCK_T 128 x BLOCK_C 128), each pinned; constexpr WIDTH 2, 3, 4 (production 4),
HAS_INITIAL_STATES, HAS_BIAS; precomputed vs derived metadata; C around both BLOCK_C (127..129,
255..257) and production 10240 (Nemotron-H 8B at TP 1); T around both BLOCK_T (7..9, 127..129); request
lengths 1, W - 2, W - 1, W, W + 1 (initial-state taps) and long ones; the production initial-state view
and a contiguous one; bf16, fp32, bf16 x with fp32 states.

Finding O7-M1 (`large` only): on main the kernel forms its token offsets as int32 products
(`src_safe * conv_dim` and `t_off * conv_dim`, causal_conv1d_varlen.py), so a prefill step with
T * C >= 2^31 elements (209716 tokens at C = 10240, 104858 at Nemotron-H 56B's C = 20480) reads and writes
at wrapped offsets. The default `max_tokens` (16384) stays far below; a larger configured prefill budget
does not. #9310 (fix/conv1d-varlen-int64-offsets) forms them in int64 and carries the step whose last
token's row starts past 2^31 as its regression test
(tests/unit_tests/ssm/ops/common/test_causal_conv1d_varlen.py, `TestInt32TokenOffsets`);
`test_token_offsets_past_int32` stops at the size whose last row crosses 2^31.

Main vs open PRs: #9310 (above), and #9316 (fix/gdp-grid-axis), which launches the kernel on grid axis 0;
the cases here call the launcher.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.ops.common import causal_conv1d_varlen as varlen_module
from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_fn
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.ssm import ssd_utils as su

DEVICE = "cuda"
KERNEL = varlen_module._causal_conv1d_varlen_kernel
PRODUCTION_DIM = 4096 * 2 + 2 * 8 * 128  # Nemotron-H 8B at TP 1


def varlen_conv_reference(x, weight, bias, initial_states, starts, lengths):
    """The contract for the real tokens, (sum(lengths), C), in the inputs' dtype."""
    width = weight.shape[1]
    outs = []
    for r, (s, n) in enumerate(zip(starts, lengths)):
        if n == 0:
            continue
        if initial_states is None:
            history = x.new_zeros(width - 1, x.shape[1])
        else:
            history = initial_states[r].to(x.dtype).T  # (W - 1, C), oldest first
        full = torch.cat([history, x[s : s + n]])  # (W - 1 + n, C)
        acc = x.new_zeros(n, x.shape[1]) if bias is None else bias.expand(n, -1).clone()
        for j in range(width):  # tap j of token t is full[t + j]
            acc = acc + full[j : j + n] * weight[:, j]
        outs.append(F.silu(acc))
    return torch.cat(outs) if outs else x.new_zeros(0, x.shape[1])


def run_varlen(
    lengths,
    dim,
    width,
    *,
    x_dtype=torch.bfloat16,
    state_dtype=None,
    bias=True,
    init="production",
    metadata="precomputed",
    padded_tail=0,
    nan_token=None,
    g=None,
):
    """One launch on random inputs, checked against the contract. `init`: "production" (the gathered
    slot states sliced [:, :, 1:]), "contiguous", or None. `metadata`: "precomputed" (MambaMetadata's
    conv_seq_idx / conv_seq_start, with `padded_tail` padded tokens) or "derived"."""
    g = g or ou.make_generator(device=DEVICE)
    state_dtype = state_dtype or x_dtype
    real = sum(lengths)
    total = real + padded_tail
    x = ou.guarded_like(torch.randn(total, dim, device=DEVICE, generator=g).to(x_dtype))
    if padded_tail:
        x.tensor[real:] = float("nan")  # padded tail tokens: never read by a real token
    if nan_token is not None:
        x.tensor[nan_token] = float("nan")
    weight = torch.randn(dim, width, device=DEVICE, generator=g).to(x_dtype) * 0.5
    bias_t = torch.randn(dim, device=DEVICE, generator=g).to(x_dtype) if bias else None
    initial = None
    if init == "production":
        cache = torch.randn(len(lengths) + 2, dim, width, device=DEVICE, generator=g).to(
            state_dtype
        )
        gather = torch.randperm(len(lengths) + 2, generator=torch.Generator().manual_seed(1))
        initial = cache[gather[: len(lengths)].to(DEVICE)][:, :, 1:]
    elif init == "contiguous":
        initial = torch.randn(len(lengths), dim, width - 1, device=DEVICE, generator=g).to(
            state_dtype
        )
    meta = su.production_metadata(lengths, 0, padded_tail, 64, DEVICE)
    cu_seqlens = meta["cu_seqlens"]
    kw = {}
    if metadata == "precomputed":
        # Longer than T, as the context's buffers are: the kernel must read only the first T.
        pad = torch.full((5,), 1 << 20, dtype=torch.int32, device=DEVICE)
        kw = dict(
            precomputed_seq_idx=torch.cat([meta["conv_seq_idx"], pad]),
            precomputed_seq_start=torch.cat([meta["conv_seq_start"], pad]),
        )
    kept = {"x": x.tensor.clone(), "weight": weight.clone()}
    if initial is not None:
        kept["initial"] = initial.clone()
    # The output in a guard band of a BLOCK_T tile of rows (the largest overrun) on each side.
    out_guard = 128 * dim * x_dtype.itemsize
    with su.poisoned_allocations(varlen_module, guard_bytes=out_guard) as buffers:
        out = causal_conv1d_varlen_fn(
            x.tensor, weight, bias_t, cu_seqlens, initial_states=initial, activation="silu", **kw
        )
    assert out.shape == (total, dim) and out.dtype == x_dtype
    su.check_all(buffers, "launcher allocations")
    ou.check_guards(x=x)
    ou.assert_exact(x.tensor, kept["x"], what="x is read-only")
    ou.assert_exact(weight, kept["weight"], what="weight is read-only")
    if initial is not None:
        ou.assert_exact(initial, kept["initial"], what="initial_states is read-only")
    starts = cu_seqlens[:-1].tolist()
    args = (x.tensor[:real], weight, bias_t, initial, starts, lengths)
    ref = ou.fp64_reference(varlen_conv_reference, *args)
    f32 = [None if a is None else a.float() if torch.is_tensor(a) else a for a in args]
    trusted = varlen_conv_reference(*f32).to(x_dtype)
    ou.assert_close_to_reference(out[:real], ref, trusted, what="conv output")
    return out


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------


def _packs(width):
    return {
        "taps": [1, max(width - 2, 1), width - 1, width, width + 1, 3],
        "block_t_8": [7, 1, 9, 8],
        "block_t_128": [127, 129, 128],
        "long": [300, 5],
    }


def _cases():
    factors = {
        "pack": ("taps", "block_t_8", "block_t_128", "long"),
        "width": (4, 2, 3),
        "dim": (128, 127, 129, 255, 256, 257),
        "dtypes": ("bf16", "fp32", "bf16_x_fp32_state"),
        "bias": (True, False),
        "init": ("production", "contiguous", None),
        "metadata": ("precomputed", "derived"),
        "config": (0, 1),
    }
    params = []
    for case in ou.pairwise_cases(factors):
        case_id = "-".join(
            [
                case["pack"],
                f"W{case['width']}",
                f"C{case['dim']}",
                case["dtypes"],
                "bias" if case["bias"] else "nobias",
                f"init_{case['init']}",
                case["metadata"],
                su.tile_id(su.kernel_configs(KERNEL)[case["config"]]),
            ]
        )
        params.append(pytest.param(case, id=case_id))
    return params


DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp32": (torch.float32, torch.float32),
    "bf16_x_fp32_state": (torch.bfloat16, torch.float32),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", _cases())
def test_varlen_conv_matches_fp64(case):
    """Pairwise over both pinned configs, the widths, sizes around both tiles, dtypes, bias, initial-state
    layouts and the two metadata sources (precomputed with padded tail tokens)."""
    x_dtype, state_dtype = DTYPES[case["dtypes"]]
    lengths = _packs(case["width"])[case["pack"]]
    with su.pinned(KERNEL, su.kernel_configs(KERNEL)[case["config"]]):
        run_varlen(
            lengths,
            case["dim"],
            case["width"],
            x_dtype=x_dtype,
            state_dtype=state_dtype,
            bias=case["bias"],
            init=case["init"],
            metadata=case["metadata"],
            padded_tail=5 if case["metadata"] == "precomputed" else 0,
        )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("config", [0, 1])
def test_nan_token_stays_in_its_request(config):
    """A NaN in one request's token reaches exactly that request's next W outputs (the reference's NaN
    pattern), never the neighbouring requests' (whose taps before their start come from their own initial
    states)."""
    lengths = [6, 3, 10, 2, 9]
    nan_token = (
        6 + 3 + 8
    )  # request 2's ninth token: its outputs 8 and 9 are NaN, request 3's are not
    with su.pinned(KERNEL, su.kernel_configs(KERNEL)[config]):
        run_varlen(lengths, 257, 4, nan_token=nan_token)
        run_varlen(lengths, 257, 4, nan_token=6 + 2)  # request 1's last token, before request 2


@ou.requires_sm(ou.HOPPER)
def test_timed_autotuning_choice():
    """The config timed autotuning picks from a cold cache (production, non-deterministic mode) is
    checked too."""
    tuner = su.autotuner(KERNEL)
    saved = tuner.cache
    tuner.cache = {}
    try:
        run_varlen([1, 3, 4, 5, 200], 1025, 4, padded_tail=7)
    finally:
        tuner.cache = saved


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("config", [0, 1])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_production_prefill_step(state_dtype, config):
    """Production: the Nemotron-H 8B conv (C 10240, W 4) over a 16384-token prefill step (the default
    max_tokens) of mixed requests with cached conv states (chunked-prefill continuations and prefix-cache
    hits), padded to a CUDA-graph size with tail tokens; x and the states in the conv-state dtype.
    """
    lengths = [8192, 4096, 2048, 1000, 1024, 3, 1, 12]
    with su.pinned(KERNEL, su.kernel_configs(KERNEL)[config]):
        run_varlen(
            lengths, PRODUCTION_DIM, 4, x_dtype=state_dtype, padded_tail=16384 - sum(lengths)
        )


@pytest.mark.parametrize(
    "tokens",
    [
        p
        for p in ou.int32_overflow_params(PRODUCTION_DIM)
        if (p.values[0] - 1) * PRODUCTION_DIM < 2**31
    ],
)
@ou.requires_sm(ou.HOPPER)
def test_token_offsets_past_int32(tokens):
    """A prefill step whose T * C reaches 2^31 elements (C = 10240): the last request's tokens, whose
    offsets cross it, must still be convolved from their own inputs and state. A last row that starts
    past 2^31 fails on main (finding O7-M1) and is #9310's regression case."""
    ou.require_free_memory(24 * 2**30)
    g = ou.make_generator(device=DEVICE)
    dim, width, tail = PRODUCTION_DIM, 4, 300
    head = tokens - tail
    x = torch.randn(tokens, dim, device=DEVICE, generator=g).bfloat16()
    weight = torch.randn(dim, width, device=DEVICE, generator=g).bfloat16() * 0.5
    bias = torch.randn(dim, device=DEVICE, generator=g).bfloat16()
    initial = torch.randn(2, dim, width - 1, device=DEVICE, generator=g).bfloat16()
    cu = torch.tensor([0, head, tokens], dtype=torch.int32, device=DEVICE)
    seq_idx = torch.cat([torch.zeros(head), torch.ones(tail)]).to(device=DEVICE, dtype=torch.int32)
    seq_start = torch.cat([torch.zeros(head), torch.full((tail,), head)]).to(
        device=DEVICE, dtype=torch.int32
    )
    out = causal_conv1d_varlen_fn(
        x,
        weight,
        bias,
        cu,
        initial_states=initial,
        activation="silu",
        precomputed_seq_idx=seq_idx,
        precomputed_seq_start=seq_start,
    )
    for r, (s, n) in enumerate([(0, 64), (head, tail)]):
        args = (x[s : s + n], weight, bias, initial[r : r + 1], [0], [n])
        ref = ou.fp64_reference(varlen_conv_reference, *args)
        trusted = varlen_conv_reference(*[a.float() if torch.is_tensor(a) else a for a in args])
        ou.assert_close_to_reference(
            out[s : s + n], ref, trusted.bfloat16(), what=f"request {r} (tokens {s}..{s + n})"
        )


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer_small_case():
    """Small case for compute-sanitizer: C 129 (partial channel tile), the initial-state taps, the
    production state layout, precomputed metadata with padded tail tokens."""
    run_varlen([1, 2, 3, 4, 5], 129, 4, padded_tail=3)


def test_case_list_covers_every_variant():
    """Both configs, every width, size, dtype pair, init layout and metadata source occur (CPU check)."""
    cases = [p.values[0] for p in _cases()]
    for factor in ("pack", "width", "dim", "dtypes", "bias", "init", "metadata", "config"):
        assert len({str(c[factor]) for c in cases}) > 1
