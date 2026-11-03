# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``flashinfer.rope.apply_rope_with_cos_sin_cache`` at Megatron's call site.

Call site: ``DynamicInferenceContext.apply_fused_qk_rotary_emb``
(``megatron/core/inference/contexts/dynamic_context.py:2192-2241``), called by
``Attention._adjust_key_value_for_inference`` (``transformer/attention.py:726-729``) when
``use_flashinfer_fused_rope`` is on. The cache is ``GPTModel``'s
``torch.cat(RotaryEmbedding.get_cos_sin(L), -1)``: fp32 ``[L, R]``, the cosines of the ``R/2``
rotary frequencies at every position, then their sines.

Contract (from the call site, the RoPE definition that Megatron's unfused path implements
(``rope_utils._apply_rotary_pos_emb_bshd``) and FlashInfer's documented API, "compatible with
the SGL/vLLM implementation"; written before reading the kernel):

* Inputs: ``query [>= n, 1, Hq, D]`` and ``key [>= n, 1, Hk, D]`` in bf16 or fp16, possibly views
  into a fused QKV buffer (last dimension contiguous); ``cos_sin_emb`` fp32 ``[L, R]``, R even,
  ``R <= D``; positions ``gpu_view.token_to_pos_ids[:n]`` (int64, each in ``[0, L)``);
  ``config.rotary_interleaved``; ``n = padded_active_token_count``.
* Computes, for each of the first n tokens t (position p), each head, with
  ``c_i = cache[p, i]`` and ``s_i = cache[p, R/2 + i]`` for ``i < R/2``:
  NeoX (``rotary_interleaved=False``): ``out[i] = x[i] c_i - x[i + R/2] s_i`` and
  ``out[i + R/2] = x[i + R/2] c_i + x[i] s_i``; interleaved (GPT-J):
  ``out[2i] = x[2i] c_i - x[2i+1] s_i`` and ``out[2i+1] = x[2i+1] c_i + x[2i] s_i``;
  ``out[R:] = x[R:]`` bit for bit. New tensors ``[n, 1, Hq, D]`` and ``[n, 1, Hk, D]`` in the input
  dtype; the inputs are not modified.
* Every head shape Megatron can configure is rotated correctly (D = ``kv_channels``,
  ``R = int(D * rotary_percent)``); Megatron uses FlashInfer's kernel for the shapes it rotates
  exactly.
* Numerics: fp32 rotation of the exact inputs with the fp32 cache values, one rounding to the
  output dtype. Deterministic: a second call returns the same bits; a CUDA-graph replay with new
  inputs returns what the eager call returns.
* Tolerance class: floating point, bounded. The rotated span against the fp64 rotation of the
  exact inputs with ``ou.assert_close_to_reference`` per (token, head) row (factor 2, one-ulp
  floor); the trusted implementation is plain torch on the same GPU, fp32 rotation rounded once.
  The pass-through span exact. A second, independent check (``test_matches_the_rope_definition``)
  recomputes the frequencies in fp64 from the RoPE definition (base, rotary_percent, position
  interpolation, llama-3 scaling) and bounds the fused path by Megatron's unfused
  ``_apply_rotary_pos_emb_bshd`` in the same dtype, fed those angles rounded to fp32: neither
  side shares ``RotaryEmbedding``'s frequencies with the cache under test.

Structure (read from FlashInfer 0.7.2 after the contract, ``include/flashinfer/pos_enc.cuh``):
D in {64, 128, 256, 512} runs a dedicated kernel (:1502-1575) with ``vec = max(8, D / 32)``
lanes per thread, ``bdx = D / vec``, 128 threads, ``bdy = 128 / bdx`` tokens per block, a
head-parallel launch while ``ceil(nnz / bdy)`` is below the resident CTAs and a token-parallel
one (head-split or not) above; any other D runs ``RopeQuantize`` (:1577-1592, :1240-1330) with
``vec = 16``, ``bdx = ceil(R / 16)``, ``bdy = 128 / bdx``, the pass-through span copied in
chunks of R. A thread rotates its vec lanes with one partner offset and one cos/sin index, so
FlashInfer is exact only when the rotated span (R/2 for NeoX, R interleaved) is a multiple of
vec (finding R08-F1). Inputs: D in {16, 32, 40, 48, 64, 72, 80, 96, 112, 128, 160, 192, 256,
512} with R in {D, D/2, D/4} plus 128/40, 96/24 and 512/48; NeoX and interleaved; bf16 and fp16;
nnz in {1, bdy-1, bdy, bdy+1, 4096, 65536}; MHA, GQA 32:8 and one K head; Q/K contiguous or
strided views of a fused QKV buffer; positions 0, 1, L-2, L-1 and random, L up to 131072.

Main vs open PR #8862 (``fix/flashinfer-fused-rope-shapes``): main rotates NeoX D16 itself in
fp32, at any rotary dim (``test_fused_rope_rotates_neox_head_dim_16``, which passes on both), and
sends every other shape to FlashInfer, which mis-rotates NeoX shapes whose R/2 is not a multiple
of its vec (e.g. D 40, 48, 72, 80, 112; 128/40, 96/48, 96/24, 512/48; most D/4 rotary dims) and
overlaps its stores past the rotary span of interleaved shapes whose R is not a multiple of 16
(e.g. 40/40, 72/72, 80/40, 96/24). ``test_fused_rope_rotates_shapes_flashinfer_cannot`` holds
exactly those shapes: it fails on main and passes with #8862, which rotates them in fp32. Every
other test takes shapes main rotates correctly and passes on both.

Memory: Q/K, the positions and the cache inside NaN guard bands (an out-of-range read turns an
output NaN), positions past n poisoned, the in-place kernel on guard-banded buffers (an
out-of-range write changes a guard), ``sanitizer`` cases on both kernels and on two shapes main
mis-rotates; rows past ``2**31`` elements (``large``, opt-in).

Hardware: H100 (sm_90).
"""

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from tests.unit_tests.inference.oracles import oracle_utils as ou

try:
    import flashinfer
    import flashinfer.rope
except ImportError:  # the package conftest fails or skips every test without FlashInfer
    flashinfer = None

pytestmark = ou.requires_sm(ou.HOPPER)

DEVICE = "cuda"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
HEAD_DIMS = (16, 32, 40, 48, 64, 72, 80, 96, 112, 128, 160, 192, 256, 512)
EXTRA_SHAPES = ((128, 40), (96, 24), (512, 48))
SHAPES = sorted(
    {(d, r) for d in HEAD_DIMS for r in (d, d // 2, d // 4) if r >= 2 and r % 2 == 0}
    | set(EXTRA_SHAPES)
)
DEDICATED_HEAD_DIMS = (64, 128, 256, 512)  # pos_enc.cuh:1502
THREADS_PER_BLOCK = 128  # pos_enc.cuh:1512 and :1254 (max(128, bdx); bdx <= 32 here)


def vector_lanes(head_dim):
    """Lanes per FlashInfer thread for 2-byte dtypes (pos_enc.cuh:1509 and :1251)."""
    return max(8, head_dim // 32) if head_dim in DEDICATED_HEAD_DIMS else 16


def block_tokens(head_dim, rotary_dim):
    """Tokens per block (``bdy``) of the kernel FlashInfer launches for this head shape."""
    vec = vector_lanes(head_dim)
    bdx = head_dim // vec if head_dim in DEDICATED_HEAD_DIMS else -(-rotary_dim // vec)
    return max(1, THREADS_PER_BLOCK // bdx)


def flashinfer_rotates_exactly(head_dim, rotary_dim, interleaved):
    """FlashInfer's vectorization rotates this shape exactly (structure-derived, R08-F1)."""
    span = rotary_dim if interleaved else rotary_dim // 2
    return head_dim % 8 == 0 and span % vector_lanes(head_dim) == 0


EXACT_SHAPES = [
    (d, r, interleaved)
    for d, r in SHAPES
    for interleaved in (False, True)
    if flashinfer_rotates_exactly(d, r, interleaved)
]
INEXACT_SHAPES = [
    (d, r, interleaved)
    for d, r in SHAPES
    for interleaved in (False, True)
    if not flashinfer_rotates_exactly(d, r, interleaved)
]
# NeoX D16: FlashInfer's 16-lane vector crosses the half-head boundary, so Megatron's call site
# rotates these shapes itself (on main for D16 NeoX at any rotary dim; with #8862 every inexact
# shape). The rest of INEXACT_SHAPES main sends to FlashInfer.
NEOX_D16_SHAPES = [s for s in INEXACT_SHAPES if s[0] == 16 and not s[2]]
MISROTATED_ON_MAIN = [s for s in INEXACT_SHAPES if s not in NEOX_D16_SHAPES]


def shape_id(d, r, interleaved=None):
    layout = "" if interleaved is None else ("-gptj" if interleaved else "-neox")
    return f"d{d}r{r}{layout}"


# ---------------------------------------------------------------------------------------------
# Reference and trusted implementation (from the definition, not from the kernel)
# ---------------------------------------------------------------------------------------------


def rotate(x, cos, sin, interleaved):
    """RoPE of ``x[..., :R]`` (``R = 2 * cos.shape[-1]``) in x's dtype; the rest passes through.

    ``x``: ``[n, 1, H, D]``; ``cos``, ``sin``: ``[n, R/2]``.
    """
    half = cos.shape[-1]
    rotary = 2 * half
    c, s = cos[:, None, None, :], sin[:, None, None, :]
    out = x.clone()
    if interleaved:
        a, b = x[..., 0:rotary:2], x[..., 1:rotary:2]
        out[..., 0:rotary:2] = a * c - b * s
        out[..., 1:rotary:2] = b * c + a * s
    else:
        a, b = x[..., :half], x[..., half:rotary]
        out[..., :half] = a * c - b * s
        out[..., half:rotary] = b * c + a * s
    return out


def assert_rope(actual, x, cos, sin, interleaved, what):
    """``actual`` is the rotation of ``x``: bounded on the rotary span, exact past it."""
    n = actual.shape[0]
    rotary = 2 * cos.shape[-1]
    assert actual.dtype == x.dtype and tuple(actual.shape) == (
        n,
        *x.shape[1:],
    ), f"{what}: got {actual.dtype} {tuple(actual.shape)}, want {x.dtype} {(n, *x.shape[1:])}"
    x = x[:n]
    reference = ou.fp64_reference(rotate, x, cos, sin, interleaved)
    trusted = rotate(x.float(), cos.float(), sin.float(), interleaved).to(x.dtype)
    ou.assert_close_to_reference(
        actual[..., :rotary],
        reference[..., :rotary],
        trusted[..., :rotary],
        dim=-1,
        what=f"{what}: rotary span",
    )
    ou.assert_exact(actual[..., rotary:], x[..., rotary:], what=f"{what}: pass-through span")


def megatron_cache(head_dim, rotary_dim, max_len, interleaved, **kwargs):
    """The cos/sin cache GPTModel builds: cat(RotaryEmbedding.get_cos_sin(L), -1), fp32 [L, R]."""
    percent = rotary_dim / head_dim
    if int(head_dim * percent) != rotary_dim:
        raise ValueError(f"rotary_percent {percent} does not give R={rotary_dim} at D={head_dim}")
    embedding = RotaryEmbedding(
        kv_channels=head_dim,
        rotary_percent=percent if rotary_dim < head_dim else 1.0,
        rotary_interleaved=interleaved,
        use_cpu_initialization=True,
        **kwargs,
    )
    cache = torch.cat(embedding.get_cos_sin(max_len), -1)
    assert cache.dtype == torch.float32 and tuple(cache.shape) == (max_len, rotary_dim)
    return cache, embedding


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def positions_for(n, max_len, g):
    """``n`` positions: 0, 1, L-2, L-1 first (as many as fit), the rest random in [0, L)."""
    pos = torch.randint(0, max_len, (n,), generator=g, device=DEVICE)
    edges = torch.tensor([0, 1, max_len - 2, max_len - 1], device=DEVICE)[:n]
    pos[: len(edges)] = edges
    return pos


class Case:
    """Inputs of one ``apply_fused_qk_rotary_emb`` call, in guard-banded buffers."""

    def __init__(self, d, r, interleaved, dtype, n, hq, hk, *, fused_qkv, max_len=4096, extra=3):
        self.d, self.r, self.interleaved, self.n = d, r, interleaved, n
        g = ou.make_generator(device=DEVICE)
        rows = n + extra  # rows past n are NaN: they must not reach the outputs
        if fused_qkv:
            qkv = torch.randn((rows, 1, (hq + 2 * hk) * d), generator=g, device=DEVICE)
            self.storage = ou.guarded_like(qkv.to(dtype))
            buf = self.storage.tensor
            self.query = buf[..., : hq * d].view(rows, 1, hq, d)
            self.key = buf[..., hq * d : (hq + hk) * d].view(rows, 1, hk, d)
            self.query[n:] = float("nan")
            self.key[n:] = float("nan")
        else:
            q = torch.randn((rows, 1, hq, d), generator=g, device=DEVICE).to(dtype)
            k = torch.randn((rows, 1, hk, d), generator=g, device=DEVICE).to(dtype)
            q[n:] = float("nan")
            k[n:] = float("nan")
            self.q_buf, self.k_buf = ou.guarded_like(q), ou.guarded_like(k)
            self.query, self.key = self.q_buf.tensor, self.k_buf.tensor
        cache, _ = megatron_cache(d, r, max_len, interleaved)
        self.cache_buf = ou.guarded_like(cache)
        self.cache = self.cache_buf.tensor
        pos = ou.poisoned_empty((n + 16,), torch.int64)  # entries past n index far out of range
        pos[:n] = positions_for(n, max_len, g)
        self.pos_buf = ou.guarded_like(pos)
        self.context = SimpleNamespace(
            use_flashinfer_fused_rope=True,
            padded_active_token_count=n,
            gpu_view=SimpleNamespace(token_to_pos_ids=self.pos_buf.tensor),
        )
        self.config = SimpleNamespace(rotary_interleaved=interleaved)

    @property
    def guards(self):
        named = dict(cache=self.cache_buf, positions=self.pos_buf)
        if hasattr(self, "storage"):
            named["qkv"] = self.storage
        else:
            named.update(query=self.q_buf, key=self.k_buf)
        return named

    def cos_sin(self):
        pos = self.pos_buf.tensor[: self.n]
        half = self.r // 2
        return self.cache[pos, :half], self.cache[pos, half:]

    def apply(self):
        return DynamicInferenceContext.apply_fused_qk_rotary_emb(
            self.context, self.query, self.key, self.cache, self.config
        )

    def check(self, outputs, what):
        cos, sin = self.cos_sin()
        query_out, key_out = outputs
        assert_rope(query_out, self.query, cos, sin, self.interleaved, f"{what} query")
        assert_rope(key_out, self.key, cos, sin, self.interleaved, f"{what} key")


class RopeSpy:
    """Counts Megatron's calls of FlashInfer's cos/sin-cache RoPE."""

    def __init__(self, monkeypatch):
        self.calls = 0
        original = flashinfer.rope.apply_rope_with_cos_sin_cache

        def spied(*args, **kwargs):
            self.calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(flashinfer.rope, "apply_rope_with_cos_sin_cache", spied)


def run_case(case, monkeypatch, what):
    """Apply Megatron's fused RoPE twice; check values, bits, read-only inputs, guards, dispatch."""
    before = {name: buf.tensor.clone() for name, buf in case.guards.items()}
    spy = RopeSpy(monkeypatch)
    first = case.apply()
    second = case.apply()
    ou.check_guards(**case.guards)
    for name, buf in case.guards.items():
        ou.assert_exact(buf.tensor, before[name], what=f"{what}: {name} is read-only")
    case.check(first, what)
    ou.assert_exact(second[0], first[0], what=f"{what}: query, second call")
    ou.assert_exact(second[1], first[1], what=f"{what}: key, second call")
    if flashinfer_rotates_exactly(case.d, case.r, case.interleaved):
        assert spy.calls == 2, f"{what}: FlashInfer rotates this shape exactly but ran {spy.calls}x"


# ---------------------------------------------------------------------------------------------
# Megatron's call site: every head shape
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(DTYPES.values()), ids=list(DTYPES))
@pytest.mark.parametrize("d,r,interleaved", EXACT_SHAPES, ids=[shape_id(*s) for s in EXACT_SHAPES])
def test_fused_rope_matches_the_rotation(d, r, interleaved, dtype, monkeypatch):
    """Head shapes FlashInfer rotates exactly: GQA heads, a strided fused-QKV view, one block of
    tokens plus one."""
    n = block_tokens(d, r) + 1
    case = Case(d, r, interleaved, dtype, n, hq=6, hk=2, fused_qkv=True)
    run_case(case, monkeypatch, shape_id(d, r, interleaved))


@pytest.mark.parametrize("fused_qkv", [True, False], ids=["fused", "contig"])
@pytest.mark.parametrize("dtype", list(DTYPES.values()), ids=list(DTYPES))
@pytest.mark.parametrize(
    "d,r,interleaved", NEOX_D16_SHAPES, ids=[shape_id(*s) for s in NEOX_D16_SHAPES]
)
def test_fused_rope_rotates_neox_head_dim_16(d, r, interleaved, dtype, fused_qkv, monkeypatch):
    """NeoX D16 at full, half and quarter rotary, which the call site rotates without FlashInfer
    (on main, and in fp32 with #8862): the first R lanes rotated by the cache's R/2 frequencies,
    the rest passed through bit for bit."""
    n = block_tokens(d, r) + 1
    case = Case(d, r, interleaved, dtype, n, hq=6, hk=2, fused_qkv=fused_qkv)
    run_case(case, monkeypatch, shape_id(d, r, interleaved))


def _inexact_cases():
    cases = [
        pytest.param(
            d,
            r,
            interleaved,
            dtype,
            block_tokens(d, r) + 1,
            4096,
            id=f"{shape_id(d, r, interleaved)}-{dtype}",
        )
        for d, r, interleaved in MISROTATED_ON_MAIN
        for dtype in DTYPES
    ]
    for d, r, interleaved in ((80, 80, False), (72, 72, True)):  # the R08-F1 examples
        bdy = block_tokens(d, r)
        for n in (1, bdy - 1, bdy, 4096, 65536):
            cases.append(
                pytest.param(
                    d, r, interleaved, "bf16", n, 4096, id=f"{shape_id(d, r, interleaved)}-n{n}"
                )
            )
        cases.append(
            pytest.param(
                d, r, interleaved, "bf16", 64, 131072, id=f"{shape_id(d, r, interleaved)}-L131072"
            )
        )
        cases.append(
            pytest.param(
                d,
                r,
                interleaved,
                "bf16",
                9,
                4096,
                id=f"{shape_id(d, r, interleaved)}-sanitizer",
                marks=pytest.mark.sanitizer,
            )
        )
    return cases


@pytest.mark.parametrize("d,r,interleaved,dtype,n,max_len", _inexact_cases())
def test_fused_rope_rotates_shapes_flashinfer_cannot(
    d, r, interleaved, dtype, n, max_len, monkeypatch
):
    """Head shapes FlashInfer's vectorization cannot rotate exactly (R08-F1) are still rotated.

    Fails on main, which sends all of them to FlashInfer (wrong lanes, neighbor heads read,
    overlapping stores; D72 interleaved stores past the output); passes with PR #8862, which
    rotates them in fp32. NeoX D16, which main rotates itself, is
    ``test_fused_rope_rotates_neox_head_dim_16``. The sanitizer cases show the out-of-bounds
    stores under memcheck on main.
    """
    hq, hk = (2, 1) if n >= 4096 else (6, 2)
    case = Case(d, r, interleaved, DTYPES[dtype], n, hq, hk, fused_qkv=n < 4096, max_len=max_len)
    run_case(case, monkeypatch, shape_id(d, r, interleaved))


# A few shapes per kernel path, each crossed with the launch-shape factors.
LAUNCH_SHAPES = [
    (128, 128, False),  # dedicated, vec 8, bdy 8
    (64, 64, True),  # dedicated, vec 8, bdy 16
    (256, 64, False),  # dedicated, partial rotary
    (512, 512, True),  # dedicated, vec 16, bdy 4
    (96, 96, False),  # RopeQuantize, bdx 6, bdy 21
    (160, 160, True),  # RopeQuantize, bdx 10, bdy 12
    (80, 32, False),  # RopeQuantize, partial rotary, nope in chunks of R
    (48, 48, True),  # RopeQuantize, bdx 3, bdy 42
]


def _launch_cases():
    cases = []
    for shape in LAUNCH_SHAPES:
        bdy = block_tokens(shape[0], shape[1])
        sizes = sorted({1, max(1, bdy - 1), bdy, bdy + 1, 4096, 65536})
        for case in ou.pairwise_cases(
            {
                "n": sizes,
                "heads": [(8, 8), (32, 8), (4, 1)],
                "fused_qkv": [True, False],
                "dtype": list(DTYPES),
            }
        ):
            cases.append(
                pytest.param(
                    shape,
                    case,
                    id=f"{shape_id(*shape)}-n{case['n']}-h{case['heads'][0]}x{case['heads'][1]}"
                    f"-{'fused' if case['fused_qkv'] else 'contig'}-{case['dtype']}",
                )
            )
    return cases


@pytest.mark.parametrize("shape,case", _launch_cases())
def test_launch_shapes(shape, case, monkeypatch):
    """nnz around one block of tokens and up to 65536 (head-parallel and token-parallel launches),
    MHA / GQA / one K head, contiguous and strided Q/K, combined pairwise."""
    d, r, interleaved = shape
    hq, hk = case["heads"]
    n = case["n"]
    if n * (hq + hk) * d > 2**25:  # keep the fp64 reference under a few GB
        hq, hk = 2, 1
    inputs = Case(d, r, interleaved, DTYPES[case["dtype"]], n, hq, hk, fused_qkv=case["fused_qkv"])
    run_case(inputs, monkeypatch, shape_id(d, r, interleaved))


@pytest.mark.parametrize("d,r,interleaved", [(128, 128, False), (128, 64, True), (96, 96, False)])
def test_positions_near_a_long_context(d, r, interleaved, monkeypatch):
    """L = 131072: the last positions of the cache, behind a NaN guard band, are read correctly."""
    case = Case(d, r, interleaved, torch.bfloat16, 64, 8, 2, fused_qkv=False, max_len=131072)
    run_case(case, monkeypatch, f"{shape_id(d, r, interleaved)} L=131072")


# ---------------------------------------------------------------------------------------------
# The RoPE definition, independently of the cache (frequencies recomputed in fp64)
# ---------------------------------------------------------------------------------------------


def _inv_freq_fp64(rotary_dim, base, rope_scaling_factor=None):
    """Inverse frequencies 1 / base^(2i/R), and llama-3 frequency scaling (factor, low 1, high 4,
    original context 8192) as published with Llama 3.1, in fp64."""
    inv = 1.0 / base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    if rope_scaling_factor is None:
        return inv
    low, high, original = 1.0, 4.0, 8192.0
    wavelen = 2 * math.pi / inv
    smooth = (original / wavelen - low) / (high - low)
    scaled = (1 - smooth) * inv / rope_scaling_factor + smooth * inv
    out = torch.where(wavelen > original / low, inv / rope_scaling_factor, inv)
    middle = (wavelen <= original / low) & (wavelen >= original / high)
    return torch.where(middle, scaled, out)


DEFINITION_CASES = {
    "d128-neox": dict(d=128, r=128, interleaved=False, kwargs={}),
    "d128-half-neox-base5e5-llama3": dict(
        d=128,
        r=64,
        interleaved=False,
        kwargs=dict(rotary_base=500000, rope_scaling=True, rope_scaling_factor=8.0),
    ),
    "d64-gptj-interp4": dict(
        d=64, r=64, interleaved=True, kwargs=dict(seq_len_interpolation_factor=4.0)
    ),
    "d256-quarter-neox": dict(d=256, r=64, interleaved=False, kwargs={}),
    "d96-neox": dict(d=96, r=96, interleaved=False, kwargs={}),
}


@pytest.mark.parametrize("name", list(DEFINITION_CASES))
def test_matches_the_rope_definition(name, monkeypatch):
    """Fused RoPE with GPTModel's cache against RoPE from the definition in fp64, bounded by
    Megatron's unfused ``_apply_rotary_pos_emb_bshd`` in bf16 (an independent implementation).

    The trusted path takes the definition's angles rounded to fp32, laid out as ``get_emb`` lays
    out its own (the R/2 angles twice: concatenated for NeoX, each repeated for GPT-J), not
    ``RotaryEmbedding``'s frequencies: a cache built from wrong frequencies is then outside the
    bound instead of shared by both sides.
    """
    spec = DEFINITION_CASES[name]
    d, r, interleaved, kwargs = spec["d"], spec["r"], spec["interleaved"], spec["kwargs"]
    max_len, n = 8192, 512
    case = Case(d, r, interleaved, torch.bfloat16, n, 8, 2, fused_qkv=True, max_len=max_len)
    cache, _ = megatron_cache(d, r, max_len, interleaved, **kwargs)
    case.cache.copy_(cache)
    query_out, key_out = case.apply()
    pos = case.pos_buf.tensor[:n].cpu()
    inv = _inv_freq_fp64(r, kwargs.get("rotary_base", 10000), kwargs.get("rope_scaling_factor"))
    angles = pos.to(torch.float64)[:, None] / kwargs.get("seq_len_interpolation_factor", 1.0) * inv
    freqs = angles.float()
    emb = freqs.repeat_interleave(2, dim=-1) if interleaved else torch.cat((freqs, freqs), dim=-1)
    emb = emb[:, None, None, :].to(DEVICE)  # [n, 1, 1, R] fp32, Megatron's layout
    for actual, x in ((query_out, case.query[:n]), (key_out, case.key[:n])):
        reference = ou.fp64_reference(
            rotate, x, angles.cos().to(DEVICE), angles.sin().to(DEVICE), interleaved
        )
        trusted = _apply_rotary_pos_emb_bshd(x, emb, rotary_interleaved=interleaved)
        ou.assert_close_to_reference(
            actual[..., :r],
            reference[..., :r],
            trusted[..., :r],
            dim=-1,
            what=f"{name} rotary span",
        )
        ou.assert_exact(actual[..., r:], x[..., r:], what=f"{name} pass-through span")


# ---------------------------------------------------------------------------------------------
# CUDA graphs: the fused RoPE runs inside the captured decoder forward
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d,r,interleaved", [(128, 128, False), (96, 96, True), (16, 16, False), (192, 192, False)]
)
def test_cuda_graph_replay_matches_eager(d, r, interleaved):
    """Captured once, replayed with new Q/K/positions: the same bits as an eager call."""
    case = Case(d, r, interleaved, torch.bfloat16, 40, 8, 2, fused_qkv=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        case.apply()  # warm up FlashInfer's JIT module and the allocator outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = case.apply()
    g = ou.make_generator(seed=ou.DEFAULT_SEED + 1, device=DEVICE)
    live = case.storage.tensor[: case.n]
    live.copy_(torch.randn(live.shape, generator=g, device=DEVICE).to(live.dtype))
    case.pos_buf.tensor[: case.n].copy_(positions_for(case.n, 4096, g).flip(0))
    graph.replay()
    eager = case.apply()
    ou.assert_exact(captured[0], eager[0], what="query: graph replay vs eager")
    ou.assert_exact(captured[1], eager[1], what="key: graph replay vs eager")
    case.check(eager, f"{shape_id(d, r, interleaved)} after replay")


# ---------------------------------------------------------------------------------------------
# FlashInfer's kernel directly, on the shapes it rotates exactly
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d,r,interleaved", EXACT_SHAPES, ids=[shape_id(*s) for s in EXACT_SHAPES])
def test_flashinfer_kernel_in_place_stays_in_bounds(d, r, interleaved):
    """The in-place variant (same kernel, outputs aliasing inputs) on guard-banded Q/K: no write
    leaves the tensors, and int32 positions rotate as int64 ones do."""
    case = Case(d, r, interleaved, torch.bfloat16, block_tokens(d, r) + 1, 4, 2, fused_qkv=False)
    n = case.n
    cos, sin = case.cos_sin()
    q_in, k_in = case.query[:n].clone(), case.key[:n].clone()
    for pos_dtype in (torch.int64, torch.int32):
        q = ou.guarded_like(q_in.reshape(n, -1))
        k = ou.guarded_like(k_in.reshape(n, -1))
        positions = case.pos_buf.tensor[:n].to(pos_dtype)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            positions, q.tensor, k.tensor, d, case.cache, is_neox=not interleaved
        )
        ou.check_guards(query=q, key=k, cache=case.cache_buf)
        what = f"{shape_id(d, r, interleaved)} in place, {pos_dtype}"
        assert_rope(q.tensor.view_as(q_in), q_in, cos, sin, interleaved, f"{what} query")
        assert_rope(k.tensor.view_as(k_in), k_in, cos, sin, interleaved, f"{what} key")


@pytest.mark.sanitizer
@pytest.mark.parametrize("d,r,interleaved", [(128, 128, False), (96, 96, True)])
def test_sanitizer_megatron_fused_rope(d, r, interleaved, monkeypatch):
    """Small cases for compute-sanitizer memcheck: the dedicated kernel and RopeQuantize (the
    shapes FlashInfer cannot rotate have their own sanitizer cases above)."""
    case = Case(d, r, interleaved, torch.bfloat16, 9, 4, 1, fused_qkv=False)
    run_case(case, monkeypatch, f"sanitizer {shape_id(d, r, interleaved)}")


@pytest.mark.parametrize("n", ou.int32_overflow_params(32 * 128))
def test_int32_token_offsets(n, monkeypatch):
    """Q offsets token * Hq * D crossing 2**31: the first and last tokens are rotated correctly."""
    d, r, hq, hk = 128, 128, 32, 1
    ou.require_free_memory(n * (hq + hk) * d * 2 * 3 + 2**30)
    g = ou.make_generator(device=DEVICE)
    query = torch.randn((n, 1, hq, d), generator=g, device=DEVICE, dtype=torch.bfloat16)
    key = torch.randn((n, 1, hk, d), generator=g, device=DEVICE, dtype=torch.bfloat16)
    cache, _ = megatron_cache(d, r, 4096, False)
    pos = positions_for(n, 4096, g)
    context = SimpleNamespace(
        use_flashinfer_fused_rope=True,
        padded_active_token_count=n,
        gpu_view=SimpleNamespace(token_to_pos_ids=pos),
    )
    config = SimpleNamespace(rotary_interleaved=False)
    query_out, key_out = DynamicInferenceContext.apply_fused_qk_rotary_emb(
        context, query, key, cache, config
    )
    for rows in (slice(0, 64), slice(n - 64, n)):
        cos, sin = cache[pos[rows], : r // 2], cache[pos[rows], r // 2 :]
        assert_rope(query_out[rows], query[rows], cos, sin, False, f"query rows {rows}")
        assert_rope(key_out[rows], key[rows], cos, sin, False, f"key rows {rows}")
