# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""fp64 references and trusted low-precision yardsticks for the attention oracles.

Written from the attention and RoPE definitions, not from any kernel:

* ``paged_attention_reference``: for each request, the keys and values it owns are gathered from a
  paged pool through its block-table row and length only (``gather_request_kv``), then attention
  is computed directly from the definition: bottom-right causal mask, optional sliding window
  ``(W, 0)``, optional softcap ``c * tanh(s / c)``, optional per-head sink as an extra
  ``exp(sink_h)`` in the softmax denominator, GQA head ``h`` reading KV head ``h // (H / Hk)``.
* ``attention_lowp``: the same math in the kernel's dtype, in the order torch's SDPA math backend
  uses (scores and probabilities rounded to the dtype). It is the trusted yardstick of the
  tolerance policy: an attention kernel's error against fp64 must stay within a small factor of it.
* ``rope_reference`` / ``rope_lowp``: rotation of channel pairs by given cos/sin (NeoX half-split
  pairs ``(i, i + d/2)`` or interleaved pairs ``(2i, 2i + 1)``), channels past ``d_rot`` passed
  through.

Shapes follow the attention call sites: tokens first, then heads, then the head dimension.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

# ---------------------------------------------------------------------------------------------
# Paged KV pools
# ---------------------------------------------------------------------------------------------


@dataclass
class PagedBatch:
    """A batch of requests over a paged KV pool, laid out as the dynamic context lays it out.

    ``q_lens[r]`` new query tokens and ``kv_lens[r]`` cached keys (the new tokens included) per
    request; padding requests (``kv_lens[r] == 0``) come last. Pages of the pool not owned by a
    live request, the rows of each request's last page past its length, and the dummy page that
    unused block-table entries point at are filled with ``tail_key``/``tail_value``/``unused``
    poison. Guard pages before and after the live pool (never referenced) hold NaN and must stay
    untouched.
    """

    q_lens: List[int]
    kv_lens: List[int]
    page_size: int
    k_pool: torch.Tensor  # [guard + pages + guard, page, Hk, D]
    v_pool: torch.Tensor  # [guard + pages + guard, page, Hk, Dv]
    block_table: torch.Tensor  # [R, max_pages] int32, indices into the pool without guards
    guard_pages: int
    dummy_page: int
    k_pool_before: torch.Tensor = field(repr=False, default=None)
    v_pool_before: torch.Tensor = field(repr=False, default=None)

    @property
    def k(self) -> torch.Tensor:
        """The pool view the kernel reads (guard pages excluded)."""
        return self.k_pool[self.guard_pages : self.k_pool.shape[0] - self.guard_pages]

    @property
    def v(self) -> torch.Tensor:
        return self.v_pool[self.guard_pages : self.v_pool.shape[0] - self.guard_pages]

    def int_tensor(self, values: Sequence[int]) -> torch.Tensor:
        return torch.tensor(list(values), dtype=torch.int32, device=self.k_pool.device)

    def cu(self, lens: Sequence[int]) -> torch.Tensor:
        out = [0]
        for n in lens:
            out.append(out[-1] + n)
        return self.int_tensor(out)

    def check_pool_unchanged(self) -> None:
        """The kernel only reads the pool: every byte, guard pages included, is unchanged."""
        ou.assert_exact(self.k_pool, self.k_pool_before, what="K pool (read-only, guards included)")
        ou.assert_exact(self.v_pool, self.v_pool_before, what="V pool (read-only, guards included)")


def make_paged_batch(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    *,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    value_dim: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    device="cuda",
    generator: Optional[torch.Generator] = None,
    spare_pages: int = 2,
    extra_table_entries: int = 2,
    tail_key: float = float("nan"),
    tail_value: float = 0.0,
    unused: float = float("nan"),
    shuffle_pages: bool = True,
    guard_pages: int = 1,
    value_scale: float = 1.0,
    live_page_offset: int = 0,
) -> PagedBatch:
    """A paged pool holding random K/V for each live request, with poisoned unused memory.

    Each live request owns ``ceil(kv_len / page_size)`` pages, in a shuffled order so that the
    block table, not the pool order, defines the sequence. Block-table entries past a request's
    pages, and every entry of a padding request, point at the dummy page (the dynamic context's
    convention, ``dynamic_context.py`` MHA metadata). ``spare_pages`` free pages are poisoned like
    the dummy page. ``tail_value`` defaults to zero: the context zeroes value pages when it
    acquires them (``_initialize_mha_value_pages``), so a live page's unwritten value rows are zero,
    while key rows are never initialized (NaN here). ``live_page_offset`` unused pages come first,
    so the live pages sit at large pool offsets (int32 overflow cases).
    """
    value_dim = head_dim if value_dim is None else value_dim
    request_pages = [math.ceil(n / page_size) for n in kv_lens]
    live_pages = sum(request_pages)
    num_pages = live_page_offset + live_pages + spare_pages + 1  # + dummy page, last
    dummy = num_pages - 1
    total = num_pages + 2 * guard_pages
    k_pool = torch.full(
        (total, page_size, kv_heads, head_dim), float("nan"), dtype=dtype, device=device
    )
    v_pool = torch.full(
        (total, page_size, kv_heads, value_dim), float("nan"), dtype=dtype, device=device
    )
    inner = slice(guard_pages, guard_pages + num_pages)
    k_pool[inner] = unused
    v_pool[inner] = unused
    order = torch.arange(live_pages)
    if shuffle_pages and live_pages > 1:
        order = torch.randperm(live_pages, generator=torch.Generator().manual_seed(live_pages))
    order = order + live_page_offset
    width = max(request_pages + [1]) + extra_table_entries
    block_table = torch.full((len(kv_lens), width), dummy, dtype=torch.int32)
    first = 0
    for r, (length, pages) in enumerate(zip(kv_lens, request_pages)):
        ids = order[first : first + pages]
        block_table[r, :pages] = ids.to(torch.int32)
        for p, page in enumerate(ids.tolist()):
            rows = min(page_size, length - p * page_size)
            kp = k_pool[guard_pages + page]
            vp = v_pool[guard_pages + page]
            kp[:rows] = torch.randn(
                rows, kv_heads, head_dim, generator=generator, device=device
            ).to(dtype)
            vp[:rows] = (
                torch.randn(rows, kv_heads, value_dim, generator=generator, device=device)
                * value_scale
            ).to(dtype)
            kp[rows:] = tail_key
            vp[rows:] = tail_value
        first += pages
    batch = PagedBatch(
        q_lens=list(q_lens),
        kv_lens=list(kv_lens),
        page_size=page_size,
        k_pool=k_pool,
        v_pool=v_pool,
        block_table=block_table.to(device),
        guard_pages=guard_pages,
        dummy_page=dummy,
    )
    batch.k_pool_before = k_pool.clone()
    batch.v_pool_before = v_pool.clone()
    return batch


def gather_request_kv(
    pool: torch.Tensor, block_table: torch.Tensor, request: int, length: int
) -> torch.Tensor:
    """Rows ``[0, length)`` of a request's sequence: row ``j`` is ``pool[table[j // P], j % P]``."""
    page = pool.shape[1]
    rows = torch.arange(length, device=pool.device)
    pages = block_table[request].to(pool.device).long()[rows // page]
    return pool[pages, rows % page]


# ---------------------------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------------------------


def allowed_keys(q_len: int, kv_len: int, window: Optional[int], device) -> torch.Tensor:
    """[q_len, kv_len] bool: query ``i`` sits at position ``kv_len - q_len + i`` (bottom-right
    causal) and sees keys ``j <= pos``, and with a window ``(W, 0)`` also ``j >= pos - W``."""
    pos = (kv_len - q_len) + torch.arange(q_len, device=device)[:, None]
    key = torch.arange(kv_len, device=device)[None, :]
    allowed = key <= pos
    if window is not None and window >= 0:
        allowed &= key >= pos - window
    return allowed


def _expand_gqa(x: torch.Tensor, heads: int) -> torch.Tensor:
    """[L, Hk, D] -> [L, H, D]: query head ``h`` reads KV head ``h // (H / Hk)``."""
    group = heads // x.shape[1]
    assert group * x.shape[1] == heads, f"{heads} query heads over {x.shape[1]} KV heads"
    return x.repeat_interleave(group, dim=1)


def attention_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
    *,
    scale: float,
    softcap: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp64 attention of q [S, H, D] over k [L, Hk, D], v [L, Hk, Dv] with mask [S, L].

    Returns (out [S, H, Dv], lse [S, H]); ``lse`` is the natural-log sum of the exponentiated
    (capped, masked) scores without the sink. With ``sink`` [H], ``exp(sink_h)`` joins the
    softmax denominator only.
    """
    for t in (q, k, v):
        assert t.dtype == torch.float64, "attention_fp64 takes fp64 inputs"
    heads = q.shape[1]
    k = _expand_gqa(k, heads)
    v = _expand_gqa(v, heads)
    scores = torch.einsum("shd,lhd->hsl", q, k) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [H, S]
    denominator = lse
    if sink is not None:
        sink = sink.to(torch.float64).view(heads, 1)
        denominator = torch.logaddexp(lse, sink.expand_as(lse))
    probs = torch.exp(scores - denominator[..., None])
    out = torch.einsum("hsl,lhd->shd", probs, v)
    return out, lse.transpose(0, 1)


def attention_lowp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
    *,
    scale: float,
    softcap: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The trusted yardstick: the same attention in q's dtype, as SDPA's math backend computes it
    (scores ``q @ k^T`` rounded to the dtype and scaled, softmax, probabilities rounded to the
    dtype, then ``p @ v``), extended with softcap and sink. Returns [S, H, Dv] in q's dtype."""
    dtype = q.dtype
    heads = q.shape[1]
    k = _expand_gqa(k, heads)
    v = _expand_gqa(v, heads)
    scores = torch.einsum("shd,lhd->hsl", q, k) * scale
    if softcap is not None:
        scores = (softcap * torch.tanh(scores / softcap)).to(dtype)
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    if sink is not None:
        sink_col = sink.to(dtype).view(heads, 1, 1).expand(heads, scores.shape[1], 1)
        probs = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hsl,lhd->shd", probs.to(dtype), v)


def paged_attention_reference(
    q: torch.Tensor,
    batch: PagedBatch,
    *,
    scale: float,
    window: Optional[int] = None,
    softcap: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    lowp: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attention of every live query row of ``q`` [T, H, D] over its request's paged KV.

    Rows are the requests' new tokens in request order (``q_lens``); padding rows after them are
    ignored. Returns (out [T_live, H, Dv], lse [T_live, H]) in fp64 (``q`` and the pool are
    upcast exactly), or with ``lowp`` the trusted yardstick in q's dtype and no lse.
    """
    outs, lses, start = [], [], 0
    for r, (q_len, kv_len) in enumerate(zip(batch.q_lens, batch.kv_lens)):
        if q_len == 0:
            continue
        assert 0 < q_len <= kv_len, f"request {r}: {q_len} queries over {kv_len} keys"
        k = gather_request_kv(batch.k, batch.block_table, r, kv_len)
        v = gather_request_kv(batch.v, batch.block_table, r, kv_len)
        q_r = q[start : start + q_len]
        allowed = allowed_keys(q_len, kv_len, window, q.device)
        if lowp:
            outs.append(attention_lowp(q_r, k, v, allowed, scale=scale, softcap=softcap, sink=sink))
        else:
            out, lse = attention_fp64(
                q_r.double(),
                k.double(),
                v.double(),
                allowed,
                scale=scale,
                softcap=softcap,
                sink=None if sink is None else sink.double(),
            )
            outs.append(out)
            lses.append(lse)
        start += q_len
    out = torch.cat(outs)
    return out, (None if lowp else torch.cat(lses))


# ---------------------------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------------------------


def rope_angles(positions: torch.Tensor, rot_dim: int, base: float) -> torch.Tensor:
    """fp64 angles [N, rot_dim / 2]: ``theta[n, i] = pos[n] * base ** (-2 i / rot_dim)``."""
    i = torch.arange(rot_dim // 2, dtype=torch.float64, device=positions.device)
    inv_freq = base ** (-2.0 * i / rot_dim)
    return positions.to(torch.float64)[:, None] * inv_freq[None, :]


def _pairs(x: torch.Tensor, rot_dim: int, interleaved: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if interleaved:
        return x[..., 0:rot_dim:2], x[..., 1:rot_dim:2]
    half = rot_dim // 2
    return x[..., :half], x[..., half:rot_dim]


def _unpairs(a: torch.Tensor, b: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if interleaved:
        return torch.stack([a, b], dim=-1).flatten(-2)
    return torch.cat([a, b], dim=-1)


def rope_reference(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, interleaved: bool
) -> torch.Tensor:
    """Rotate each channel pair of ``x`` [..., D] by the angle whose cos/sin are given.

    ``cos``/``sin`` broadcast against ``x[..., :d_rot/2]`` (``d_rot = 2 * cos.shape[-1]``):
    ``(a, b) -> (a cos - b sin, a sin + b cos)``; channels ``>= d_rot`` pass through. Exact in
    whatever dtype the inputs have (call it through ``ou.fp64_reference``).
    """
    rot_dim = 2 * cos.shape[-1]
    a, b = _pairs(x, rot_dim, interleaved)
    rotated = _unpairs(a * cos - b * sin, a * sin + b * cos, interleaved)
    return torch.cat([rotated, x[..., rot_dim:]], dim=-1)


def rope_lowp(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, interleaved: bool
) -> torch.Tensor:
    """The trusted yardstick: the same rotation with torch's elementwise ops in x's dtype (each
    product and sum rounded to the dtype)."""
    return rope_reference(x, cos.to(x.dtype), sin.to(x.dtype), interleaved=interleaved)
