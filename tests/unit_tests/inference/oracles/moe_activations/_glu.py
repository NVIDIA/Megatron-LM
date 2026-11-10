# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The gated-activation (GLU) contract shared by the four GLU kernel oracles (not a test module).

Model math (training, ``megatron/core/transformer/moe/experts.py``, the unfused ``glu`` of
``TEGroupedMLP``): FC1 produces ``[gate | linear]`` halves of width ``N`` along the last dim
(``torch.chunk(x, 2, -1)``), and the activation is

    act(min(gate, c)) * (clamp(linear, -c, c) + offset)        act = SiLU (SwiGLU) or relu^2

where the clamp ``c`` is ``activation_func_clamp_value`` (absent by default), ``offset`` is
``glu_linear_offset`` (0 by default), and with ``moe_mlp_glu_interleave_size = I`` FC1 rows hold
alternating ``I``-wide gate and linear blocks that training de-interleaves first. The routing
probability, where a kernel applies it, multiplies the activation output.

On ``main`` the inference GLU kernels implement only the default (SwiGLU, no clamp, offset 0, no
interleaving); the references here take the options as well. Open PR #8566 adds them to the
kernels; ``test_glu_options.py``, on a branch stacked on that PR, checks them.
"""

from typing import Any, Optional

import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common


def split_halves(x: torch.Tensor, interleave: Optional[int] = None) -> tuple:
    """The gate and linear inputs of each output column, in the FC1 layout.

    ``[gate | linear]`` halves, or with ``interleave = I`` the blocks ``gate_0, linear_0,
    gate_1, linear_1, ...`` of width ``I`` (training's ``_remove_glu_interleaving``).
    """
    n = x.shape[-1] // 2
    if interleave is None:
        return x[..., :n], x[..., n:]
    blocks = x.reshape(*x.shape[:-1], n // interleave, 2, interleave)
    return blocks[..., 0, :].reshape(*x.shape[:-1], n), blocks[..., 1, :].reshape(*x.shape[:-1], n)


def fp32_bound(value: float) -> float:
    """A Python float as the fp32 kernel argument (or fp32 tensor op) sees it."""
    return float(torch.tensor(value, dtype=torch.float32))


def swiglu_exact(
    x: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    clamp: Optional[float] = None,
    offset: float = 0.0,
    interleave: Optional[int] = None,
) -> torch.Tensor:
    """fp64 SwiGLU of the model math: ``silu(min(g, c)) * (clamp(u, -c, c) + offset) [* p]``.

    For ``fp64_reference``; the clamp bound and the offset are the fp32 values the fp32
    computation uses. Non-finite inputs follow IEEE evaluation of ``g / (1 + exp(-g)) * u``.
    """
    gate, linear = split_halves(x, interleave)
    if clamp is not None:
        c = fp32_bound(clamp)
        gate = torch.clamp(gate, max=c)  # torch.clamp keeps NaN
        linear = torch.clamp(linear, min=-c, max=c)
    if offset:
        linear = linear + fp32_bound(offset)
    out = gate / (1.0 + torch.exp(-gate)) * linear
    if probs is not None:
        out = out * probs.reshape(-1, 1)
    return out


def swiglu_trusted(
    x: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    clamp: Optional[float] = None,
    offset: float = 0.0,
    interleave: Optional[int] = None,
) -> torch.Tensor:
    """The trusted implementation: torch's own fp32 SiLU on the same device, rounded once to
    the output dtype (bf16). Independent of the kernels (no Triton)."""
    gate, linear = (h.float() for h in split_halves(x, interleave))
    if clamp is not None:
        gate = gate.clamp(max=clamp)
        linear = linear.clamp(min=-clamp, max=clamp)
    out = torch.nn.functional.silu(gate) * (linear + offset)
    if probs is not None:
        out = out * probs.float().reshape(-1, 1)
    return out.to(x.dtype)


def glu_inputs(
    shape: tuple, generator: torch.Generator, device: Any, scale: float = 2.0
) -> torch.Tensor:
    """FC1-like bf16 inputs: standard normal times ``scale`` (gates well inside the region
    where fp32 SiLU has no subnormal intermediates)."""
    return (torch.randn(shape, generator=generator, device=device) * scale).to(torch.bfloat16)


SIGMOID_SUBNORMAL_GATE = -87.33654
"""Below this gate, sigmoid(gate) < 2**-126: the fp32 SiLU passes through a subnormal (or exp
overflows), which GPU arithmetic may flush to zero."""


def silu_may_flush(x: torch.Tensor, exact: torch.Tensor, interleave: Optional[int] = None):
    """Elements whose fp32 SwiGLU evaluation passes through a subnormal (see
    ``_common.assert_rounded_once``): a subnormal gate or linear input, a gate whose
    sigmoid is subnormal, or a subnormal result."""
    gate, linear = (h.to(torch.float64) for h in split_halves(x, interleave))
    tiny = _common.FP32_TINY
    return (
        (gate < SIGMOID_SUBNORMAL_GATE)
        | ((gate.abs() < tiny) & (gate != 0))
        | ((linear.abs() < tiny) & (linear != 0))
        | (exact.to(gate.device).abs() < tiny)
    )


def assert_swiglu_values(
    out: torch.Tensor,
    x: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    what: str,
    cols: Optional[int] = None,
    **options: Any,
) -> None:
    """The live rows ``out`` of a SwiGLU kernel against the contract, for inputs ``x``
    (``out`` may hold only the first ``cols`` output columns of each row).

    Two assertions: every element is the single bf16 rounding of an fp32 evaluation
    (``_common.assert_rounded_once``: the correctly rounded fp64 value, or a neighbour within
    the fp32 evaluation window of a tie), and the harness policy, per row, against torch's fp32
    SiLU rounded once (``swiglu_trusted``).
    """
    if out.numel() == 0:
        return
    exact = ou.fp64_reference(swiglu_exact, x, probs, **options)[..., :cols]
    trusted = swiglu_trusted(x, probs, **options)[..., :cols]
    _common.assert_rounded_once(out, exact, what=what)
    ou.assert_close_to_reference(out, exact, trusted, dim=-1, what=what)


SPECIAL_GATES = (
    0.0,
    -0.0,
    1e-30,
    -1e-30,
    1e-39,  # bf16 / fp32 subnormal
    -1e-39,
    0.5,
    -3.0,
    20.0,
    -20.0,
    80.0,
    -80.0,
    -87.0,
    -88.0,  # sigmoid subnormal in fp32
    -88.5,
    -89.0,  # exp(89) overflows fp32
    88.5,
    -100.0,
    100.0,
    1e4,
    -1e4,
    3e38,  # silu(g) * u overflows bf16 for |u| > 1.13
    -3e38,
    float("inf"),
    float("-inf"),
    float("nan"),
)


def special_value_inputs(device: Any) -> list:
    """Gate x linear grids of special values (one row per gate, one column per linear value).

    (a) every gate against finite linears; (b) gates whose fp32 SiLU does not underflow against
    infinite and NaN linears; (c) finite gates against subnormal linears (not under the Triton
    interpreter, which converts bf16 subnormals wrongly). Underflowing SiLU
    times an infinite linear, and an infinite gate times a subnormal linear, are left out: GPU
    arithmetic may flush the subnormal to zero there, which turns an infinity into NaN.
    """

    def grid(gates, linears):
        g = torch.tensor(gates, dtype=torch.float64)
        u = torch.tensor(linears, dtype=torch.float64)
        x = torch.cat([g[:, None].expand(-1, u.numel()), u[None, :].expand(g.numel(), -1)], 1)
        return x.to(torch.bfloat16).to(device).contiguous()

    gates = SPECIAL_GATES
    if _common.INTERPRET:
        # The Triton 3.8 interpreter converts bf16 subnormals to fp32 wrongly (1.01e-39 reads
        # as 4.41e-39, 2**-133 as 0); compiled kernels convert them exactly.
        gates = [g for g in gates if not 0 < abs(g) < _common.FP32_TINY]
    finite = [g for g in gates if g == g and abs(g) != float("inf")]
    no_underflow = [g for g in gates if not (g == g and g < SIGMOID_SUBNORMAL_GATE)]
    inf, nan = float("inf"), float("nan")
    grids = [grid(gates, (0.0, -0.0, 1.0, -2.5, 1e30, -1e30)), grid(no_underflow, (inf, -inf, nan))]
    if not _common.INTERPRET:
        grids.append(grid(finite, (1e-39, -1e-39)))
    return grids


def assert_swiglu_special_values(
    out: torch.Tensor,
    x: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    *,
    what: str,
    **options: Any,
) -> None:
    """Special-value rows of a SwiGLU kernel: the same NaN and infinities as the correctly
    rounded fp64 value (overflow past the bf16 range included), and every finite element the
    single rounding of an fp32 evaluation, or ±0 where that evaluation may flush a subnormal."""
    exact = ou.fp64_reference(swiglu_exact, x, probs, **options)
    expected = _common.rne_bf16(exact)
    _common.assert_same_nonfinite(out, expected, what=what)
    finite = torch.isfinite(expected)
    may_flush = silu_may_flush(x, exact, options.get("interleave"))
    _common.assert_rounded_once(out[finite], exact[finite], what=what, may_flush=may_flush[finite])


# ---------------------------------------------------------------------------------------------
# Live rows and padding rows of the expert-grouped buffer
# ---------------------------------------------------------------------------------------------

PADDING_PATTERNS = ("none", "runs", "program-stride", "edges", "random", "all")
"""Where the alignment-padding rows (``permutation_map == -1``) sit: nowhere; runs of five; at
the rows the persistent grid's programs start and wrap at (multiples of 512, plus or minus 1);
the first and last rows; 10% at random; every row."""


ROUTES_PER_TOKEN = 2
"""Top-k of the expert-grouped maps below: every token appears in this many expert blocks."""
EXPERT_BLOCKS = 4
"""Experts (blocks) of the expert-grouped maps below."""


def expert_grouped_token_indices(rows: int) -> torch.Tensor:
    """The live-row values of ``permutation_map`` as ``permute_tokens`` fills it: the original
    token index of each row of the expert-grouped buffer.

    ``ceil(rows / ROUTES_PER_TOKEN)`` tokens, each routed to ``ROUTES_PER_TOKEN`` distinct
    experts out of ``EXPERT_BLOCKS``, listed expert by expert (tokens in increasing order inside
    a block); the first ``rows`` routes fill the buffer. Row ``r`` holds a token index that is in
    general not ``r`` and repeats across experts, so a kernel that indexes per-row data
    (``x``, ``probs``) through the map instead of by the row reads another row's data. Drawn
    from a CPU generator of its own, seeded by ``rows``, so the caller's generator stream (the
    inputs drawn after the map) is unchanged.
    """
    g = torch.Generator().manual_seed(ou.DEFAULT_SEED + rows)
    tokens = -(-rows // ROUTES_PER_TOKEN)
    keys = torch.rand(tokens, EXPERT_BLOCKS, generator=g)
    experts = keys.argsort(-1)[:, :ROUTES_PER_TOKEN].reshape(-1)
    token_of_route = torch.arange(tokens).repeat_interleave(ROUTES_PER_TOKEN)
    order = torch.argsort(experts * tokens + token_of_route)
    return token_of_route[order][:rows].clone()


def permutation_map(
    rows: int, pattern: str, dtype: torch.dtype, generator: torch.Generator, device: Any
) -> torch.Tensor:
    """A permutation map of ``rows`` rows: the original token index of each live row in
    expert-grouped order (``expert_grouped_token_indices``, not the identity), or -1 for an
    alignment-padding row placed by ``pattern`` (``PADDING_PATTERNS``)."""
    perm = expert_grouped_token_indices(rows)
    if pattern == "runs":
        for start in range(3, rows, 37):
            perm[start : start + 5] = -1
    elif pattern == "program-stride":
        for start in range(0, rows + 512, 512):
            for r in (start - 1, start, start + 1):
                if 0 <= r < rows:
                    perm[r] = -1
    elif pattern == "edges":
        perm[0] = -1
        perm[-1] = -1
    elif pattern == "random":
        draw = torch.rand(rows, generator=generator, device=generator.device).cpu()
        perm[draw < 0.1] = -1
    elif pattern == "all":
        perm[:] = -1
    elif pattern != "none":
        raise ValueError(pattern)
    return perm.to(device=device, dtype=dtype)


def live_count(kind: str, rows: int, programs: int) -> int:
    """The live-row count for a symbolic ``kind`` around the grid size ``programs`` and the
    buffer height ``rows``; ``rows+1`` is a count past the buffer, which must not overrun it."""
    values = {
        "0": 0,
        "1": 1,
        "B-1": programs - 1,
        "B": programs,
        "B+1": programs + 1,
        "M-1": rows - 1,
        "M": rows,
        "M+1": rows + 1,
    }
    return max(0, min(values[kind], rows + 1))


LIVE_COUNTS = ("0", "1", "B-1", "B", "B+1", "M-1", "M", "M+1")
