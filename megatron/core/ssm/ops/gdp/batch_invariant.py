# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical GDP block factorization and incremental GPU execution.

Inputs q/k are prepared (including any L2 normalization), BF16. Log decay and
beta are FP32. All internal update rows of a token are processed before its readout.
Blocks are measured in *updates*, not tokens. State orientation is [K,V].
The FP32 state is committed only at fixed block boundaries. Cache right-hand
sides and inverse-factor rows are BF16; all mode switches preserve this contract.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _right_hand_side(value, decay, projection, beta):
    # A compiler layout change can otherwise change the rounding of this
    # expression despite enable_fp_fusion=False. Pin all three FP32 operations.
    return tl.inline_asm_elementwise(
        "{ .reg .f32 p, e; mul.rn.f32 p, $2, $3; sub.rn.f32 e, $1, p; mul.rn.f32 $0, e, $4; }",
        constraints="=f,f,f,f,f",
        args=[value, decay, projection, beta],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _prefix(x, C: tl.constexpr):
    rows = tl.arange(0, C)
    out = tl.full((C,), 0.0, tl.float32)
    running = tl.full((), 0.0, tl.float32)
    for i in range(C):
        value = tl.sum(tl.gather(x, tl.full((1,), i, tl.int32), 0), 0)
        running = running + value
        out = tl.where(rows == i, running, out)
    return out


@triton.jit
def _pair_rows(x, PREPARED: tl.constexpr):
    M: tl.constexpr = x.shape[0]
    N: tl.constexpr = x.shape[1]
    if N == 16:
        return tl.sum(tl.reshape(x, (M // 2, 2, N)), 1)
    if not PREPARED:
        # The same adjacent-pair tree with a lower-latency decode layout.
        even = tl.broadcast_to((2 * tl.arange(0, M // 2))[:, None], (M // 2, N))
        return tl.gather(x, even, 0) + tl.gather(x, even + 1, 0)
    # Explicit splits reduce register traffic in the parallel preparation grid.
    left, right = tl.split(tl.permute(tl.reshape(x, (M // 2, 2, N)), (0, 2, 1)))
    return left + right


@triton.jit
def _ordered_row_sum(x, PREPARED: tl.constexpr):
    # tl.sum over the complete row axis can use a different addition tree in
    # factor preparation and incremental inference. Pair adjacent logical rows
    # explicitly, independent of their distribution across lanes and warps.
    M: tl.constexpr = x.shape[0]
    if M >= 64:
        x = _pair_rows(x, PREPARED)
    if M >= 32:
        x = _pair_rows(x, PREPARED)
    if M >= 16:
        x = _pair_rows(x, PREPARED)
    if M >= 8:
        x = _pair_rows(x, PREPARED)
    if M >= 4:
        x = _pair_rows(x, PREPARED)
    if M >= 2:
        x = _pair_rows(x, PREPARED)
    return tl.sum(x, 0)


@triton.jit
def _append_factor(a, factor, start, end, C: tl.constexpr, PREPARED: tl.constexpr = False):
    cs = tl.arange(0, C)
    for i in range(start, end):
        ai = tl.sum(tl.gather(a.to(tl.float32), tl.full((1, C), i, tl.int32), 0), 0)
        ti = _ordered_row_sum(ai[:, None] * factor.to(tl.float32), PREPARED)
        ti = (ti + (cs == i).to(tl.float32)).to(tl.bfloat16)
        factor = tl.where(cs[:, None] == i, ti[None, :], factor).to(tl.bfloat16)
    return factor


@triton.jit(do_not_specialize=["N"])
def _prepare(
    K, G, Beta, Trans, Prefix, N, H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, C: tl.constexpr
):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    cs = tl.arange(0, C)
    ds = tl.arange(0, D)
    row = block * C + cs
    valid = row < N * R
    token = row // R
    rid = row % R
    base = ((batch * N + token) * R + rid) * H + head
    keys = tl.load(K + base[:, None] * D + ds[None, :], valid[:, None], other=0)
    beta = tl.load(Beta + base, valid, other=0)
    gates = tl.load(G + (batch * N + token) * H + head, valid & (rid == 0), other=0)
    p = _prefix(gates, C)
    ratio = tl.exp(tl.where(cs[:, None] >= cs[None, :], p[:, None] - p[None, :], 0.0))
    gram = tl.dot(keys, tl.trans(keys))
    a = tl.where(cs[:, None] > cs[None, :], -beta[:, None] * ratio * gram, 0.0).to(tl.bfloat16)
    factor = tl.full((C, C), 0.0, tl.bfloat16)
    factor = _append_factor(a, factor, 0, tl.minimum(C, N * R - block * C), C, True)
    ti = (batch * tl.cdiv(N * R, C) + block) * H + head
    tl.store(Prefix + ti * C + cs, p)
    tl.store(Trans + ti * C * C + cs[:, None] * C + cs[None, :], factor)


@triton.jit(do_not_specialize=["N"])
def _forward(
    Q,
    K,
    V,
    G,
    Beta,
    Slots,
    State,
    Keys,
    Gates,
    Factors,
    Rhs,
    Cursor,
    O,
    TapeS,
    TapeT,
    TapeH,
    TapeU,
    Prefix,
    Cu,
    SS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    FS: tl.constexpr,
    HS: tl.constexpr,
    CS: tl.constexpr,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    TAPE: tl.constexpr,
    PREPARED: tl.constexpr,
    OUTPUTS: tl.constexpr,
    VARLEN: tl.constexpr,
    BV: tl.constexpr,
):
    bh = tl.program_id(0)
    batch, head = bh // H, bh % H
    if VARLEN:
        input_start = tl.load(Cu + batch)
        count = tl.load(Cu + batch + 1) - input_start
    else:
        input_start = batch * N
        count = N
    slot = tl.load(Slots + batch)
    if slot < 0:
        for token in range(count):
            tl.store(O + ((input_start + token) * H + head) * W + tl.arange(0, W), 0.0)
        return
    sh = head
    cs = tl.arange(0, C)
    ds = tl.arange(0, D)
    vs = tl.program_id(1) * BV + tl.arange(0, BV)
    so = slot * SS + sh * D * W + ds[:, None] * W + vs[None, :]
    state = tl.load(State + so)
    if PREPARED:
        # Prepared scans start at a fresh block boundary. Value tiles own
        # disjoint state/RHS columns; their shared cache fields have one writer.
        pos = 0
        keys = tl.full((C, D), 0.0, tl.bfloat16)
        gates = tl.full((C,), 0.0, tl.float32)
        factor = tl.full((C, C), 0.0, tl.bfloat16)
        rhs = tl.full((C, BV), 0.0, tl.bfloat16)
    else:
        pos = tl.load(Cursor + slot * CS + sh)
        keys = tl.load(Keys + slot * KS + sh * C * D + cs[:, None] * D + ds[None, :])
        gates = tl.load(Gates + slot * GS + sh * C + cs)
        factor = tl.load(Factors + slot * FS + sh * C * C + cs[:, None] * C + cs[None, :])
        rhs = tl.load(Rhs + slot * HS + sh * C * W + cs[:, None] * W + vs[None, :])
    offset = 0
    block = 0
    while offset < count * R:
        take = tl.minimum(C - pos, count * R - offset)
        end = pos + take
        new = (cs >= pos) & (cs < end)
        old = cs < pos
        update = offset + cs - pos
        token = update // R
        r = update % R
        base = ((input_start + token) * R + r) * H + head
        key = tl.load(K + base[:, None] * D + ds[None, :], new[:, None], other=0)
        value = tl.load(V + base[:, None] * W + vs[None, :], new[:, None], other=0)
        beta = tl.load(Beta + base, new, other=0)
        gate = tl.load(G + (input_start + token) * H + head, new & (r == 0), other=0)
        query = tl.load(
            Q + ((input_start + token) * H + head)[:, None] * D + ds[None, :],
            (new & (r == R - 1))[:, None],
            other=0,
        )
        keys = tl.where(new[:, None], key, tl.where(old[:, None], keys, 0.0)).to(tl.bfloat16)
        gates = tl.where(new, gate, tl.where(old, gates, 0.0))
        factor = tl.where(old[:, None], factor, 0.0).to(tl.bfloat16)
        rhs = tl.where(old[:, None], rhs, 0.0).to(tl.bfloat16)
        if PREPARED:
            ti = (batch * tl.cdiv(N * R, C) + block) * H + head
            p = tl.load(Prefix + ti * C + cs)
        else:
            p = _prefix(gates, C)
        decay = tl.exp(p)
        ratio = tl.exp(tl.where(cs[:, None] >= cs[None, :], p[:, None] - p[None, :], 0.0))
        if PREPARED:
            ti = (batch * tl.cdiv(N * R, C) + block) * H + head
            factor = tl.load(TapeT + ti * C * C + cs[:, None] * C + cs[None, :])
        else:
            gram = tl.dot(keys, tl.trans(keys))
            a = tl.where(cs[:, None] > cs[None, :], -beta[:, None] * ratio * gram, 0.0).to(
                tl.bfloat16
            )
            factor = _append_factor(a, factor, pos, end, C)
        sb = state.to(tl.bfloat16)
        projection = tl.dot(keys, sb)
        fresh_rhs = _right_hand_side(
            value.to(tl.float32), decay[:, None], projection, beta[:, None]
        ).to(tl.bfloat16)
        rhs = tl.where(new[:, None], fresh_rhs, rhs).to(tl.bfloat16)
        u = tl.dot(factor, rhs).to(tl.bfloat16)
        if OUTPUTS:
            qstate = tl.dot(query, sb)
            qk = tl.dot(query, tl.trans(keys))
            weights = tl.where(cs[:, None] >= cs[None, :], ratio * qk, 0.0).to(tl.bfloat16)
            output = ((decay[:, None] * qstate + tl.dot(weights, u)) * SCALE).to(tl.bfloat16)
            tl.store(
                O + ((input_start + token) * H + head)[:, None] * W + vs[None, :],
                output,
                (new & (r == R - 1))[:, None],
            )
        if TAPE:
            nb = tl.cdiv(N * R, C)
            ti = (batch * nb + block) * H + head
            tl.store(TapeS + ti * D * W + ds[:, None] * W + vs[None, :], state)
            if not PREPARED:
                tl.store(TapeT + ti * C * C + cs[:, None] * C + cs[None, :], factor)
            tl.store(TapeH + ti * C * W + cs[:, None] * W + vs[None, :], rhs)
            tl.store(TapeU + ti * C * W + cs[:, None] * W + vs[None, :], u)
        if end == C:
            last = tl.sum(tl.where(cs == C - 1, p, 0.0), 0)
            f = (keys.to(tl.float32) * tl.exp(last - p)[:, None]).to(tl.bfloat16)
            state = state * tl.exp(last) + tl.dot(tl.trans(f), u)
            pos = 0
        else:
            pos = end
        offset += take
        block += 1
    tl.store(State + so, state)
    # Clear invalid tails so slot reuse and exact cache comparison are unambiguous.
    valid = cs < pos
    if tl.program_id(1) == 0:
        tl.store(
            Keys + slot * KS + sh * C * D + cs[:, None] * D + ds[None, :],
            tl.where(valid[:, None], keys, 0.0),
        )
        tl.store(Gates + slot * GS + sh * C + cs, tl.where(valid, gates, 0.0))
        tl.store(
            Factors + slot * FS + sh * C * C + cs[:, None] * C + cs[None, :],
            tl.where(valid[:, None], factor, 0.0),
        )
    tl.store(
        Rhs + slot * HS + sh * C * W + cs[:, None] * W + vs[None, :],
        tl.where(valid[:, None], rhs, 0.0),
    )
    if tl.program_id(1) == 0:
        tl.store(Cursor + slot * CS + sh, pos)


@triton.jit(do_not_specialize=["N"])
def _outputs(
    Q,
    K,
    Prefix,
    TapeS,
    TapeU,
    O,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    cs, ds, vs = tl.arange(0, C), tl.arange(0, D), tl.arange(0, W)
    row = block * C + cs
    valid = row < N * R
    token, rid = row // R, row % R
    base = ((batch * N + token) * R + rid) * H + head
    key = tl.load(K + base[:, None] * D + ds[None, :], valid[:, None], other=0)
    QC: tl.constexpr = 16 if C == 32 and R >= 2 else C
    if QC != C:
        qcs = R - 1 - (block * C) % R + tl.arange(0, QC) * R
    else:
        qcs = cs
    qrow = block * C + qcs
    qtoken = qrow // R
    qvalid = (qcs < C) & (qrow < N * R) & (qrow % R == R - 1)
    q = tl.load(
        Q + ((batch * N + qtoken) * H + head)[:, None] * D + ds[None, :], qvalid[:, None], other=0
    )
    ti = (batch * tl.cdiv(N * R, C) + block) * H + head
    state = tl.load(TapeS + ti * D * W + ds[:, None] * W + vs[None, :]).to(tl.bfloat16)
    u = tl.load(TapeU + ti * C * W + cs[:, None] * W + vs[None, :])
    p = tl.load(Prefix + ti * C + cs)
    qp = tl.gather(p, tl.minimum(qcs, C - 1), 0)
    decay = tl.exp(qp)
    ratio = tl.exp(tl.where(qcs[:, None] >= cs[None, :], qp[:, None] - p[None, :], 0.0))
    qstate = tl.dot(q, state)
    qk = tl.dot(q, tl.trans(key))
    weight = tl.where(qcs[:, None] >= cs[None, :], ratio * qk, 0.0).to(tl.bfloat16)
    output = ((decay[:, None] * qstate + tl.dot(weight, u)) * SCALE).to(tl.bfloat16)
    tl.store(
        O + ((batch * N + qtoken) * H + head)[:, None] * W + vs[None, :], output, qvalid[:, None]
    )


@dataclass
class GDPCache:
    """Complete logical cache. Move/reset all fields together, never state alone."""

    state: torch.Tensor
    keys: torch.Tensor
    gates: torch.Tensor
    factors: torch.Tensor
    rhs: torch.Tensor
    cursor: torch.Tensor

    @classmethod
    def allocate(
        cls,
        slots: int,
        heads: int,
        key_dim: int,
        value_dim: int,
        *,
        block_size: int = 16,
        device: str | torch.device = "cuda",
    ) -> "GDPCache":
        """Allocate zeros for independent sequences, with no implicit slot sharing."""
        if (
            block_size not in (16, 32)
            or key_dim not in (16, 32, 64, 128)
            or value_dim not in (16, 32, 64, 128)
        ):
            raise ValueError("Supported C=16/32 and K,V=16/32/64/128")

        def zeros(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        return cls(
            zeros((slots, heads, key_dim, value_dim), torch.float32),
            zeros((slots, heads, block_size, key_dim), torch.bfloat16),
            zeros((slots, heads, block_size), torch.float32),
            zeros((slots, heads, block_size, block_size), torch.bfloat16),
            zeros((slots, heads, block_size, value_dim), torch.bfloat16),
            zeros((slots, heads), torch.int32),
        )

    @staticmethod
    def storage_size(heads: int, key_dim: int, value_dim: int, block_size: int = 16) -> int:
        """Return FP32 storage elements for one complete request cache."""
        c = block_size
        sizes = (
            heads * key_dim * value_dim * 4,
            heads * c * key_dim * 2,
            heads * c * 4,
            heads * c * c * 2,
            heads * c * value_dim * 2,
            heads * 4,
        )
        if any(n % 4 for n in sizes):
            raise ValueError("Cache fields must be aligned to four bytes")
        return sum(sizes) // 4

    @classmethod
    def from_storage(
        cls, storage: torch.Tensor, heads: int, key_dim: int, value_dim: int, block_size: int = 16
    ) -> "GDPCache":
        """View one FP32 request slab without copies; allocator owns its lifecycle."""
        c = block_size
        size = cls.storage_size(heads, key_dim, value_dim, c)
        if (
            storage.ndim != 2
            or storage.shape[1] != size
            or storage.dtype != torch.float32
            or not storage.is_contiguous()
        ):
            raise ValueError("Expected contiguous FP32 [slots,complete_cache_size]")
        specs = (
            ((heads, key_dim, value_dim), torch.float32),
            ((heads, c, key_dim), torch.bfloat16),
            ((heads, c), torch.float32),
            ((heads, c, c), torch.bfloat16),
            ((heads, c, value_dim), torch.bfloat16),
            ((heads,), torch.int32),
        )
        fields = []
        offset_bytes = 0
        for shape, dtype in specs:
            typed = storage.view(dtype)
            itemsize = typed.element_size()
            strides: list[int] = []
            elements = 1
            for dim in reversed(shape):
                strides.insert(0, elements)
                elements *= dim
            field = typed.as_strided(
                (storage.shape[0], *shape),
                (size * 4 // itemsize, *strides),
                storage_offset=typed.storage_offset() + offset_bytes // itemsize,
            )
            fields.append(field)
            offset_bytes += elements * itemsize
        return cls(*fields)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return every logical cache field in its kernel argument order."""
        return self.state, self.keys, self.gates, self.factors, self.rhs, self.cursor

    def reset(self, slots: torch.Tensor) -> None:
        """Release/reset selected slots before assigning them a new sequence."""
        for tensor in self.tensors():
            tensor.index_fill_(0, slots, 0)

    def snapshot(self, slots: torch.Tensor) -> "GDPCache":
        """Copy selected complete request states, including partially built blocks."""
        return GDPCache(*(x.index_select(0, slots) for x in self.tensors()))

    def restore(self, slots: torch.Tensor, snapshot: "GDPCache") -> None:
        """Restore complete snapshots for prefix reuse or speculative rollback."""
        for target, source in zip(self.tensors(), snapshot.tensors()):
            if (
                source.shape != (slots.numel(), *target.shape[1:])
                or source.dtype != target.dtype
                or source.device != target.device
            ):
                raise ValueError("Snapshot must match cache format and selected slots")
        for target, source in zip(self.tensors(), snapshot.tensors()):
            target.index_copy_(0, slots, source)


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cache: GDPCache,
    slots: torch.Tensor,
    requests: int | None = None,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError("q shape must be [B,T,H,K]")
    b, t, h, d = q.shape
    if k.ndim != 5 or v.ndim != 5:
        raise ValueError("k/v shapes must be [B,T,R,H,D/V]")
    r, w = k.shape[2], v.shape[-1]
    if (
        min(b, t, h, r) < 1
        or k.shape != (b, t, r, h, d)
        or v.shape != (b, t, r, h, w)
        or g.shape != (b, t, h)
        or beta.shape != (b, t, r, h)
        or cache.state.shape[1:] != (h, d, w)
        or slots.shape != (b if requests is None else requests,)
    ):
        raise ValueError("Inconsistent GDP/cache shapes")
    for x, dtype in (
        (q, torch.bfloat16),
        (k, torch.bfloat16),
        (v, torch.bfloat16),
        (g, torch.float32),
        (beta, torch.float32),
        (slots, torch.int32),
    ):
        if not x.is_cuda or x.device != q.device or x.dtype != dtype or not x.is_contiguous():
            raise ValueError(
                "Inputs must be contiguous on one CUDA device: q/k/v BF16, g/beta FP32, slots int32"
            )
    count = cache.state.shape[0]
    c = cache.keys.shape[2]
    if c not in (16, 32) or d not in (16, 32, 64, 128) or w not in (16, 32, 64, 128):
        raise ValueError("Supported C=16/32 and K,V=16/32/64/128")
    specs = (
        ((count, h, d, w), torch.float32),
        ((count, h, c, d), torch.bfloat16),
        ((count, h, c), torch.float32),
        ((count, h, c, c), torch.bfloat16),
        ((count, h, c, w), torch.bfloat16),
        ((count, h), torch.int32),
    )
    for x, (shape, dtype) in zip(cache.tensors(), specs):
        if (
            x.shape != shape
            or x.dtype != dtype
            or x.device != q.device
            or count < 1
            or not x[0].is_contiguous()
        ):
            raise ValueError("Invalid cache shape/dtype/device/strides")
    return b, t, h, d, w


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cache: GDPCache,
    slots: torch.Tensor,
    *,
    scale: float,
    tape: bool = False,
    prepared: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    requests = slots.numel() if cu_seqlens is not None else q.shape[0]
    b, t, h, d, w = _validate(q, k, v, g, beta, cache, slots, requests)
    if cu_seqlens is not None and (prepared or tape):
        raise ValueError("Packed inference uses the incremental path")
    c = cache.keys.shape[2]
    nb = triton.cdiv(t * k.shape[2], c)
    if tape or prepared:
        ts = torch.empty((b, nb, h, d, w), device=q.device, dtype=torch.float32)
        tt = torch.empty((b, nb, h, c, c), device=q.device, dtype=torch.bfloat16)
        th = torch.empty((b, nb, h, c, w), device=q.device, dtype=torch.bfloat16)
        tu = torch.empty_like(th)
    else:
        ts = tt = th = tu = q
    prefix = torch.empty((b, nb, h, c), device=q.device, dtype=torch.float32) if prepared else q
    if prepared:
        _prepare[(nb, b * h)](
            k, g, beta, tt, prefix, t, h, d, k.shape[2], c, num_warps=1, enable_fp_fusion=False
        )
    out = torch.empty((b, t, h, w), device=q.device, dtype=q.dtype)
    forward_tile = min(w, 16) if prepared and b * h < 128 else w
    _forward[(requests * h, triton.cdiv(w, forward_tile))](
        q,
        k,
        v,
        g,
        beta,
        slots,
        *cache.tensors(),
        out,
        ts,
        tt,
        th,
        tu,
        prefix,
        q if cu_seqlens is None else cu_seqlens,
        *(x.stride(0) for x in cache.tensors()),
        t,
        h,
        d,
        w,
        k.shape[2],
        c,
        scale,
        tape or prepared,
        prepared,
        not prepared,
        cu_seqlens is not None,
        forward_tile,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if prepared:
        _outputs[(nb, b * h)](
            q,
            k,
            prefix,
            ts,
            tu,
            out,
            t,
            h,
            d,
            w,
            k.shape[2],
            c,
            scale,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, (ts, tt, th, tu, prefix)


def append(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cache: GDPCache,
    *,
    slots: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Append whole real tokens to independent slots, including partial blocks.

    Active slots must be unique and in range; -1 lanes are inert and output zero.
    The caller owns slot scheduling, allowing CUDA graph replay without CPU sync.
    Use train() for autograd; this operation mutates the supplied cache.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, g, beta)):
        raise ValueError("append mutates inference cache; use train for gradients")
    if torch.is_grad_enabled() and any(x.requires_grad for x in cache.tensors()):
        raise ValueError("Inference cache must not require gradients")
    if slots is None:
        if q.shape[0] > cache.state.shape[0]:
            raise ValueError("Default slots exceed cache capacity")
        slots = torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
    return _run(
        q, k, v, g, beta, cache, slots, scale=q.shape[-1] ** -0.5 if scale is None else scale
    )[0]


def append_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cache: GDPCache,
    cu_seqlens: torch.Tensor,
    slots: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Append variable-length slices with device metadata, safe for CUDA graphs.

    Input batch is one and the token axis concatenates request slices. Caller
    supplies int32 offsets [requests+1] partitioning [0,total_tokens] in order,
    including zero-length slices, and unique active slot IDs (-1 is inactive).
    Slot IDs and offset *values* are a scheduler contract; they are not copied to
    the CPU for validation. Every slice resumes its own partial-block cursor.
    """
    if (
        q.shape[0] != 1
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() != slots.numel() + 1
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != q.device
        or not cu_seqlens.is_contiguous()
        or slots.numel() == 0
    ):
        raise ValueError("Packed input requires B=1, int32 CUDA offsets and matching slots")
    if torch.is_grad_enabled() and any(
        x.requires_grad for x in (q, k, v, g, beta, *cache.tensors())
    ):
        raise ValueError("Packed append mutates inference cache; use train for gradients")
    return _run(
        q,
        k,
        v,
        g,
        beta,
        cache,
        slots,
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        cu_seqlens=cu_seqlens,
    )[0]


@triton.jit
def _grad_dot(a, b):
    # Preserve FP32 operand residuals with native BF16 tensor-core products.
    # Most gradient products have one exact BF16 operand: omit its zero residual
    # explicitly so the compiler does not allocate registers or issue an MMA.
    ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
    result = tl.dot(ah, bh)
    if a.dtype != tl.bfloat16:
        al = (a.to(tl.float32) - ah.to(tl.float32)).to(tl.bfloat16)
        result += tl.dot(al, bh)
    if b.dtype != tl.bfloat16:
        bl = (b.to(tl.float32) - bh.to(tl.float32)).to(tl.bfloat16)
        result += tl.dot(ah, bl)
    return result


@triton.jit(do_not_specialize=["N"])
def _local_backward(
    Q,
    K,
    V,
    Prefix,
    Beta,
    TapeS,
    TapeT,
    TapeH,
    TapeU,
    TapeDS,
    DO,
    DU,
    DQ,
    DK,
    DV,
    DBeta,
    DT,
    DP,
    DR,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    BV: tl.constexpr,
):
    block, bh, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh // H, bh % H
    cs, ds, vs = tl.arange(0, C), tl.arange(0, D), part * BV + tl.arange(0, BV)
    nb = tl.cdiv(N * R, C)
    span = tl.num_programs(1)
    ti = (batch * nb + block) * H + head
    DQ += part * span * N * D
    DK += part * span * N * R * D
    DR += part * span * nb * C * C
    DBeta += part * span * N * R
    DT += part * span * nb * C * C
    DP += part * span * nb * C
    row = block * C + cs
    valid = row < N * R
    token, rid = row // R, row % R
    base = ((batch * N + token) * R + rid) * H + head
    key = tl.load(K + base[:, None] * D + ds[None, :], valid[:, None], other=0)
    QC: tl.constexpr = 16 if C == 32 and R >= 2 else C
    if QC != C:
        qcs = R - 1 - (block * C) % R + tl.arange(0, QC) * R
    else:
        qcs = cs
    qrow = block * C + qcs
    qtoken = qrow // R
    qvalid = (qcs < C) & (qrow < N * R) & (qrow % R == R - 1)
    q = tl.load(
        Q + ((batch * N + qtoken) * H + head)[:, None] * D + ds[None, :], qvalid[:, None], other=0
    )
    do = (
        tl.load(
            DO + ((batch * N + qtoken) * H + head)[:, None] * W + vs[None, :],
            qvalid[:, None],
            other=0,
        ).to(tl.float32)
        * SCALE
    )
    state = tl.load(TapeS + ti * D * W + ds[:, None] * W + vs[None, :])
    sb = state.to(tl.bfloat16)
    u = tl.load(TapeU + ti * C * W + cs[:, None] * W + vs[None, :])
    p = tl.load(Prefix + ti * C + cs)
    decay = tl.exp(p)
    qp = tl.gather(p, tl.minimum(qcs, C - 1), 0)
    qdecay = tl.exp(qp)
    ratio = tl.exp(tl.where(qcs[:, None] >= cs[None, :], qp[:, None] - p[None, :], 0.0))
    qstate = tl.dot(q, sb)
    qk = tl.dot(q, tl.trans(key))
    causal = qcs[:, None] >= cs[None, :]
    dw = _grad_dot(do, tl.trans(u))
    dw = tl.where(causal, dw, 0.0)
    dqs = do * qdecay[:, None]
    dq = _grad_dot(dqs, tl.trans(sb))
    dqk = dw * ratio
    dq += _grad_dot(dqk, key)
    dk = _grad_dot(tl.trans(dqk), q)
    dratio = dw * qk
    dpq = tl.sum(do * qstate, 1) * qdecay
    if QC != C:
        qi = tl.maximum(cs - (R - 1 - (block * C) % R), 0) // R
        dratio = tl.where(
            (rid == R - 1)[:, None], tl.gather(dratio, tl.broadcast_to(qi[:, None], (C, C)), 0), 0.0
        )
        dp = tl.where(rid == R - 1, tl.gather(dpq, qi, 0), 0.0)
    else:
        dp = dpq
    tl.store(DQ + ((batch * N + qtoken) * H + head)[:, None] * D + ds[None, :], dq, qvalid[:, None])
    if (block + 1) * C <= N * R:
        dstate = tl.load(TapeDS + ti * D * W + ds[:, None] * W + vs[None, :])
        last = tl.sum(tl.where(cs == C - 1, p, 0.0), 0)
        e = tl.exp(last - p)
        df = _grad_dot(u, tl.trans(dstate))
        dk += df * e[:, None]
        de = tl.sum(df * key.to(tl.float32), 1) * e
        dp -= de
        dlast = tl.sum(de, 0) + tl.sum(tl.sum(dstate * state, 1), 0) * tl.exp(last)
        dp += tl.where(cs == C - 1, dlast, 0.0)
    tl.store(DR + ti * C * C + cs[:, None] * C + cs[None, :], dratio)
    trans = tl.load(TapeT + ti * C * C + cs[:, None] * C + cs[None, :])
    rhs = tl.load(TapeH + ti * C * W + cs[:, None] * W + vs[None, :])
    du = tl.load(DU + ti * C * W + cs[:, None] * W + vs[None, :])
    dt = _grad_dot(du, tl.trans(rhs))
    tl.store(DT + ti * C * C + cs[:, None] * C + cs[None, :], dt)
    dh = _grad_dot(tl.trans(trans), du)
    val = tl.load(V + base[:, None] * W + vs[None, :], valid[:, None], other=0)
    beta = tl.load(Beta + base, valid, other=0)
    projection = tl.dot(key, sb)
    dbeta = tl.sum(dh * (val.to(tl.float32) - decay[:, None] * projection), 1)
    dv = dh * beta[:, None]
    dproj = -dv * decay[:, None]
    dp += tl.sum(-dv * projection, 1) * decay
    dk += _grad_dot(dproj, tl.trans(sb))
    tl.store(DP + ti * C + cs, dp)
    tl.store(DK + base[:, None] * D + ds[None, :], dk, valid[:, None])
    tl.store(DV + base[:, None] * W + vs[None, :], dv, valid[:, None])
    tl.store(DBeta + base, dbeta, valid)


@triton.jit(do_not_specialize=["N"])
def _factor_backward(
    K,
    Prefix,
    Beta,
    TapeT,
    DT,
    DP,
    DR,
    DKPartial,
    DK,
    DBPartial,
    DBeta,
    DG,
    DQPartial,
    DQ,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    PARTS: tl.constexpr,
):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    cs, ds = tl.arange(0, C), tl.arange(0, D)
    row = block * C + cs
    valid = row < N * R
    token, rid = row // R, row % R
    base = ((batch * N + token) * R + rid) * H + head
    ti = (batch * tl.cdiv(N * R, C) + block) * H + head
    key = tl.load(K + base[:, None] * D + ds[None, :], valid[:, None], other=0)
    beta = tl.load(Beta + base, valid, other=0)
    trans = tl.load(TapeT + ti * C * C + cs[:, None] * C + cs[None, :])
    dt = tl.load(DT + ti * C * C + cs[:, None] * C + cs[None, :])
    dratio = tl.load(DR + ti * C * C + cs[:, None] * C + cs[None, :])
    dp = tl.load(DP + ti * C + cs)
    dk = tl.load(DKPartial + base[:, None] * D + ds[None, :], valid[:, None], other=0)
    dbeta = tl.load(DBPartial + base, valid, other=0)
    qbase = ((batch * N + token) * H + head)[:, None] * D + ds[None, :]
    qvalid = (valid & (rid == R - 1))[:, None]
    dq = tl.full((C, D), 0.0, tl.float32)
    if PARTS > 1:
        dq = tl.load(DQPartial + qbase, qvalid, other=0)
    span = tl.num_programs(1)
    nb = tl.cdiv(N * R, C)
    for part in range(1, PARTS):
        dt += tl.load(DT + part * span * nb * C * C + ti * C * C + cs[:, None] * C + cs[None, :])
        dratio += tl.load(
            DR + part * span * nb * C * C + ti * C * C + cs[:, None] * C + cs[None, :]
        )
        dp += tl.load(DP + part * span * nb * C + ti * C + cs)
        dk += tl.load(
            DKPartial + part * span * N * R * D + base[:, None] * D + ds[None, :],
            valid[:, None],
            other=0,
        )
        dbeta += tl.load(DBPartial + part * span * N * R + base, valid, other=0)
        dq += tl.load(DQPartial + part * span * N * D + qbase, qvalid, other=0)
    if PARTS > 1:
        tl.store(DQ + qbase, dq, qvalid)
    p = tl.load(Prefix + ti * C + cs)
    ratio = tl.exp(tl.where(cs[:, None] >= cs[None, :], p[:, None] - p[None, :], 0.0))
    gram = tl.dot(key, tl.trans(key))
    strict = (cs[:, None] > cs[None, :]) & valid[:, None]
    a = tl.where(strict, -beta[:, None] * ratio * gram, 0.0).to(tl.bfloat16)
    # Merge upper-triangular inverse blocks. Each stage solves one new
    # off-diagonal block without forming high powers of the full matrix.
    upper = tl.trans(a)
    inverse = (cs[:, None] == cs[None, :]).to(tl.float32)
    for level in tl.static_range(0, 5 if C == 32 else 4):
        half = 1 << level
        cross = (
            (cs[:, None] // (2 * half) == cs[None, :] // (2 * half))
            & (cs[:, None] % (2 * half) < half)
            & (cs[None, :] % (2 * half) >= half)
        )
        off = tl.where(cross, upper, 0.0).to(tl.bfloat16)
        inverse = inverse + _grad_dot(inverse, _grad_dot(off, inverse))
    up = _grad_dot(inverse, dt)
    da = _grad_dot(up, tl.trans(trans))
    da = tl.where(strict, da, 0.0)
    dbeta += tl.sum(-da * ratio * gram, 1)
    dgram = -da * beta[:, None] * ratio
    dratio += -da * beta[:, None] * gram
    dk += _grad_dot(dgram + tl.trans(dgram), key)
    drp = dratio * ratio
    dp += tl.sum(drp, 1) - tl.sum(drp, 0)
    dg = tl.cumsum(dp, 0, reverse=True)
    tl.store(DK + base[:, None] * D + ds[None, :], dk, valid[:, None])
    tl.store(DBeta + base, dbeta, valid)
    tl.store(DG + (batch * N + token) * H + head, dg, valid & (rid == 0))


@triton.jit(do_not_specialize=["N"])
def _state_backward(
    Q,
    K,
    Prefix,
    Beta,
    TapeT,
    DO,
    DFinal,
    TapeDS,
    DU,
    DInitial,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    BV: tl.constexpr,
):
    bh = tl.program_id(0)
    batch, head = bh // H, bh % H
    cs, ds, vs = (tl.arange(0, C), tl.arange(0, D), tl.program_id(1) * BV + tl.arange(0, BV))
    dstate = tl.load(DFinal + bh * D * W + ds[:, None] * W + vs[None, :])
    nb = tl.cdiv(N * R, C)
    for back in tl.range(nb, num_stages=2 if C == 32 else 3):
        block = nb - back - 1
        row = block * C + cs
        valid = row < N * R
        token, rid = row // R, row % R
        base = ((batch * N + token) * R + rid) * H + head
        key = tl.load(K + base[:, None] * D + ds[None, :], valid[:, None], other=0)
        QC: tl.constexpr = 16 if C == 32 and R >= 2 else C
        if QC != C:
            qcs = R - 1 - (block * C) % R + tl.arange(0, QC) * R
        else:
            qcs = cs
        qrow = block * C + qcs
        qtoken = qrow // R
        qvalid = (qcs < C) & (qrow < N * R) & (qrow % R == R - 1)
        q = tl.load(
            Q + ((batch * N + qtoken) * H + head)[:, None] * D + ds[None, :],
            qvalid[:, None],
            other=0,
        )
        beta = tl.load(Beta + base, valid, other=0)
        do = (
            tl.load(
                DO + ((batch * N + qtoken) * H + head)[:, None] * W + vs[None, :],
                qvalid[:, None],
                other=0,
            ).to(tl.float32)
            * SCALE
        )
        ti = (batch * nb + block) * H + head
        tl.store(TapeDS + ti * D * W + ds[:, None] * W + vs[None, :], dstate)
        trans = tl.load(TapeT + ti * C * C + cs[:, None] * C + cs[None, :])
        p = tl.load(Prefix + ti * C + cs)
        decay = tl.exp(p)
        qp = tl.gather(p, tl.minimum(qcs, C - 1), 0)
        qdecay = tl.exp(qp)
        ratio = tl.exp(tl.where(qcs[:, None] >= cs[None, :], qp[:, None] - p[None, :], 0.0))
        qk = tl.dot(q, tl.trans(key))
        weight = tl.where(qcs[:, None] >= cs[None, :], ratio * qk, 0.0).to(tl.bfloat16)
        du = _grad_dot(tl.trans(weight), do)
        if (block + 1) * C <= N * R:
            last = tl.sum(tl.where(cs == C - 1, p, 0.0), 0)
            f = (key.to(tl.float32) * tl.exp(last - p)[:, None]).to(tl.bfloat16)
            du += _grad_dot(f, dstate)
            dstate = dstate * tl.exp(last)
        tl.store(DU + ti * C * W + cs[:, None] * W + vs[None, :], du)
        dh = _grad_dot(tl.trans(trans), du)
        dproj = -(dh * beta[:, None]) * decay[:, None]
        dsb = _grad_dot(tl.trans(q), do * qdecay[:, None])
        dsb += _grad_dot(tl.trans(key), dproj)
        dstate += dsb

    tl.store(DInitial + bh * D * W + ds[:, None] * W + vs[None, :], dstate)


class _Train(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial, block_size, scale):
        b, t, h, d = q.shape
        cache = GDPCache.allocate(b, h, d, v.shape[-1], block_size=block_size, device=q.device)
        cache.state.copy_(initial)
        slots = torch.arange(b, device=q.device, dtype=torch.int32)
        out, tape = _run(q, k, v, g, beta, cache, slots, scale=scale, tape=True, prepared=True)
        ctx.save_for_backward(q, k, v, g, beta, *tape)
        ctx.block_size, ctx.scale = block_size, scale
        return out, cache.state

    @staticmethod
    @once_differentiable
    def backward(ctx, do, dfinal):
        q, k, v, g, beta, *tape = ctx.saved_tensors
        b, t, h, d = q.shape
        w = v.shape[-1]
        do = do.contiguous()
        dfinal = dfinal.contiguous()
        grads = tuple(torch.empty_like(x) for x in (q, k, v, g, beta))
        dinitial = torch.empty((b, h, d, w), device=q.device, dtype=torch.float32)
        nb = triton.cdiv(t * k.shape[2], ctx.block_size)
        dstate_tape = torch.empty_like(tape[0])
        du = torch.empty_like(tape[3], dtype=torch.float32)
        large_tile = b * h >= 128 and w >= 64
        state_tile = min(w, 64 if large_tile else (16 if b * h < 128 else 32))
        _state_backward[(b * h, triton.cdiv(w, state_tile))](
            q,
            k,
            tape[4],
            beta,
            tape[1],
            do,
            dfinal,
            dstate_tape,
            du,
            dinitial,
            t,
            h,
            d,
            w,
            k.shape[2],
            ctx.block_size,
            ctx.scale,
            state_tile,
            num_warps=4 if b * h < 128 or large_tile else 2,
            enable_fp_fusion=False,
        )
        parts = triton.cdiv(w, 64)
        dt = torch.empty(
            (parts, b, nb, h, ctx.block_size, ctx.block_size), device=q.device, dtype=torch.float32
        )
        dr = torch.empty_like(dt)
        dp = torch.empty((parts, b, nb, h, ctx.block_size), device=q.device, dtype=torch.float32)
        dk_partial = torch.empty((parts, *k.shape), device=q.device, dtype=torch.float32)
        # A single value tile owns the final query gradient; avoid a full
        # FP32 intermediate and its readback in the factor kernel.
        dq_partial = (
            grads[0].unsqueeze(0)
            if parts == 1
            else torch.empty((parts, *q.shape), device=q.device, dtype=torch.float32)
        )
        db_partial = torch.empty((parts, *beta.shape), device=q.device, dtype=torch.float32)
        _local_backward[(nb, b * h, parts)](
            q,
            k,
            v,
            tape[4],
            beta,
            tape[0],
            tape[1],
            tape[2],
            tape[3],
            dstate_tape,
            do,
            du,
            dq_partial,
            dk_partial,
            grads[2],
            db_partial,
            dt,
            dp,
            dr,
            t,
            h,
            d,
            w,
            k.shape[2],
            ctx.block_size,
            ctx.scale,
            min(w, 64),
            num_warps=4,
            enable_fp_fusion=False,
        )
        _factor_backward[(nb, b * h)](
            k,
            tape[4],
            beta,
            tape[1],
            dt,
            dp,
            dr,
            dk_partial,
            grads[1],
            db_partial,
            grads[4],
            grads[3],
            dq_partial,
            grads[0],
            t,
            h,
            d,
            k.shape[2],
            ctx.block_size,
            parts,
            num_warps=2,
            enable_fp_fusion=False,
        )
        return *grads, dinitial, None, None


def train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    block_size: int = 16,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable canonical GDP forward and FP32 *boundary* state.

    The returned state is at the last complete update block, not a materialized
    state after a partial tail. It is useful for loss/gradient diagnostics; use
    the full GDPCache to resume inference. Backward differentiates the rounded
    factorization with straight-through cast derivatives (first-order only).
    """
    b, _, h, d = q.shape
    if initial_state is None:
        initial_state = torch.zeros((b, h, d, v.shape[-1]), device=q.device, dtype=torch.float32)
    if (
        initial_state.shape != (b, h, d, v.shape[-1])
        or initial_state.dtype != torch.float32
        or initial_state.device != q.device
        or not initial_state.is_contiguous()
    ):
        raise ValueError("initial_state must be contiguous FP32 [B,H,K,V] on input device")
    return _Train.apply(
        q, k, v, g, beta, initial_state, block_size, d**-0.5 if scale is None else scale
    )


def prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    block_size: int = 16,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, GDPCache]:
    """Prefill fresh sequences with parallel factor preparation and return full cache."""
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, g, beta)):
        raise ValueError("Use train for gradients")
    cache = GDPCache.allocate(
        q.shape[0], q.shape[2], q.shape[3], v.shape[-1], block_size=block_size, device=q.device
    )
    if initial_state is not None:
        if (
            initial_state.shape != cache.state.shape
            or initial_state.dtype != torch.float32
            or initial_state.device != q.device
            or initial_state.requires_grad
        ):
            raise ValueError("initial_state must be FP32 [B,H,K,V]")
        cache.state.copy_(initial_state)
    slots = torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
    output, _ = _run(
        q,
        k,
        v,
        g,
        beta,
        cache,
        slots,
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        prepared=True,
    )
    return output, cache
