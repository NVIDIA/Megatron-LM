# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU stand-ins for DeepGEMM's MQA logits and DeepGEMM's packed UE8M0 MXFP8 quantization.

The fakes reproduce the contracts the DSA DeepGEMM scorer relies on rather than the kernels'
arithmetic order:

* :func:`fake_quantize_packed_ue8m0` mirrors TE MXFP8 seen through
  ``quantize_indexer_mxfp8_packed_ue8m0``: E4M3 data and one int32 per 128-element row whose
  byte ``g`` (little endian) is the E8M0 exponent of 32-element group ``g`` (ceil power of
  two of ``amax / 448``; 0xFF for a group holding NaN or Inf, whose data is NaN).
* :class:`FakeDeepGEMM` mirrors ``deep_gemm`` 2.8.2: keyword-named ``get_mqa_logits_metadata``
  and ``fp8_fp4_mqa_logits``, the argument checks of the real binding (including the
  scheduler's shared-memory row limit, :func:`metadata_smem_bytes`), BF16 "compressed"
  logits where column ``t`` of row ``r`` is key ``ks[r] + t`` for ``t < ke[r] - ks[r]``, a row
  stride padded to ``align(align(max_seqlen_k, 384), 512)`` elements (1024-byte rows) in an
  allocation of whole padded rows, and every unwritten element poisoned (NaN, +inf, 1e30 in
  turn) so a reader past a row's length is caught. A NaN logit is written with one chosen
  sign (what the real kernel writes for NaN inputs is recorded by the GPU test).
"""

from __future__ import annotations

import torch

# Bound before any test patches ``Tensor.tolist`` to detect host syncs in MCore code.
_TOLIST = torch.Tensor.tolist

E4M3_MAX = 448.0
_SPLIT_KV = 384
_ROW_ALIGN_BF16 = 512
SM100_SMEM_CAPACITY = 232448


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def metadata_smem_bytes(num_rows: int, num_heads: int) -> int:
    """Shared memory of DeepGEMM 2.8.2's ``sm100_mqa_logits_metadata`` (256 threads): two int32
    words per query block of ``128 // num_heads`` rows plus one per warp."""
    return (2 * -(-num_rows // (128 // num_heads)) + 256 // 32) * 4


def fake_quantize_packed_ue8m0(x: torch.Tensor, calls: list | None = None):
    """E4M3 data and packed UE8M0 int32 scales of BF16 ``x`` ``(..., 128)``."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[-1] == 128
    if calls is not None:
        calls.append({"shape": tuple(x.shape), "x": x})
    groups = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = groups.abs().amax(dim=-1)
    finite = torch.isfinite(amax)
    exponent = torch.where(
        finite & (amax > 0), torch.ceil(torch.log2(amax / E4M3_MAX)), torch.zeros_like(amax)
    ).clamp(min=-127, max=127)
    scaled = groups / torch.exp2(exponent).unsqueeze(-1)
    scaled = torch.where(finite.unsqueeze(-1), scaled, torch.full_like(scaled, float("nan")))
    data = scaled.reshape(x.shape).to(torch.float8_e4m3fn)
    scale_bytes = torch.where(finite, exponent + 127, torch.full_like(exponent, 255)).long()
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int64)
    packed = (scale_bytes << shifts).sum(dim=-1)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    return data, packed


def dequantize_packed_ue8m0(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 values of E4M3 ``data`` ``(..., 128)`` with packed UE8M0 int32 ``scales``."""
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32)
    scale_bytes = (scales.unsqueeze(-1) >> shifts) & 0xFF
    factor = torch.exp2(scale_bytes.float() - 127)
    factor = torch.where(scale_bytes == 255, torch.full_like(factor, float("nan")), factor)
    groups = data.float().reshape(*data.shape[:-1], 4, 32) * factor.unsqueeze(-1)
    return groups.reshape(data.shape)


class FakeDeepGEMM:
    """CPU model of the ``deep_gemm`` 2.8.2 module surface the DSA scorer uses."""

    __version__ = "2.8.2"
    __file__ = "fake_deep_gemm/__init__.py"

    def __init__(self, num_sms: int = 148, nan_bits: int = 0x7FC0):
        self.num_sms = num_sms
        # BF16 bits written for a NaN logit: 0x7FC0 (positive) or e.g. -64 (0xFFC0, negative).
        self.nan_bits = nan_bits
        self.calls = []
        self.metadata_calls = []

    def get_num_sms(self) -> int:
        return self.num_sms

    def set_num_sms(self, num_sms: int) -> None:
        self.num_sms = num_sms

    def get_mqa_logits_metadata(
        self, cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, num_heads
    ) -> torch.Tensor:
        ks, ke = cu_seq_len_k_start, cu_seq_len_k_end
        assert ks.dtype == torch.int32 and ke.dtype == torch.int32
        assert ks.is_contiguous() and ke.is_contiguous() and ks.shape == ke.shape
        assert ks.numel() > 0 and num_kv_tokens > 0 and 128 % num_heads == 0
        if metadata_smem_bytes(ks.numel(), num_heads) > SM100_SMEM_CAPACITY:
            # DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity) surfaces as this.
            raise RuntimeError("Assertion error: smem_size <= SM100ArchSpec::smem_capacity")
        meta = torch.zeros((2 * self.num_sms + 2,), dtype=torch.int32)
        self.metadata_calls.append(
            dict(ks=ks, ke=ke, num_kv_tokens=num_kv_tokens, num_heads=num_heads, meta=meta)
        )
        return meta

    def fp8_fp4_mqa_logits(
        self, q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k, schedule_meta=None
    ) -> torch.Tensor:
        q_fp8, q_scale = q
        kv_fp8, kv_scale = kv
        ks, ke = cu_seq_len_k_start, cu_seq_len_k_end
        rows, num_heads, head_dim = q_fp8.shape
        num_keys = kv_fp8.size(0)
        assert q_fp8.dtype == torch.float8_e4m3fn and kv_fp8.dtype == torch.float8_e4m3fn
        assert q_scale.dtype == torch.int32 and tuple(q_scale.shape) == (rows, num_heads)
        assert kv_scale.dtype == torch.int32 and tuple(kv_scale.shape) == (num_keys,)
        assert kv_fp8.is_contiguous() and kv_scale.is_contiguous()
        assert kv_fp8.size(1) == head_dim
        assert weights.dtype == torch.bfloat16 and tuple(weights.shape) == (rows, num_heads)
        assert weights.stride(1) == 1 and (weights.stride(0) * 2) % 16 == 0
        assert ks.dtype == torch.int32 and ke.dtype == torch.int32
        assert ks.is_contiguous() and ke.is_contiguous() and ks.numel() == rows == ke.numel()
        assert isinstance(max_seqlen_k, int) and max_seqlen_k > 0
        lengths = _TOLIST(ke - ks)
        starts = _TOLIST(ks)
        assert all(0 <= n <= max_seqlen_k for n in lengths), "ke - ks must be in [0, max]"
        assert all(0 <= s <= num_keys for s in starts), "ks must index the K rows"
        q_values = dequantize_packed_ue8m0(q_fp8, q_scale)
        k_values = dequantize_packed_ue8m0(kv_fp8, kv_scale)
        dense = torch.einsum("thd,kd->thk", q_values, k_values).relu()
        dense = (dense * weights.float().unsqueeze(-1)).sum(dim=1)
        block_q = 128 // num_heads
        row_stride = _align(_align(max_seqlen_k, _SPLIT_KV), _ROW_ALIGN_BF16)
        padded_rows = _align(rows, block_q)
        poison = torch.tensor([float("nan"), float("inf"), 1.0e30])
        storage = poison[torch.arange(padded_rows * row_stride) % 3].to(torch.bfloat16)
        storage = storage.view(padded_rows, row_stride)
        columns = torch.arange(max_seqlen_k).view(1, -1)
        keys = ks.long().view(-1, 1) + columns
        written = columns < (ke - ks).long().view(-1, 1)
        gathered = torch.gather(dense, 1, keys.clamp(min=0, max=num_keys - 1))
        logits = storage[:rows, :max_seqlen_k]
        # One NaN bit pattern (``nan_bits``), whatever sign the arithmetic left on it.
        nan = torch.tensor(self.nan_bits, dtype=torch.int16).view(torch.bfloat16)
        values = torch.where(torch.isnan(gathered), nan, gathered.to(torch.bfloat16))
        logits.copy_(torch.where(written, values, logits))
        self.calls.append(
            dict(
                q=q_fp8,
                q_scale=q_scale,
                kv=kv_fp8,
                kv_scale=kv_scale,
                weights=weights,
                ks=ks,
                ke=ke,
                max_seqlen_k=max_seqlen_k,
                schedule_meta=schedule_meta,
                logits=logits,
            )
        )
        return logits


def reference_logits(q_bshd, k_bsd, w_bsh) -> torch.Tensor:
    """BF16 ``[sq, sk]`` logits the fakes produce for the same BF16 inputs (no ranges)."""
    q_fp8, q_scale = fake_quantize_packed_ue8m0(q_bshd[0].contiguous())
    k_fp8, k_scale = fake_quantize_packed_ue8m0(k_bsd[0].contiguous())
    dense = torch.einsum(
        "thd,kd->thk",
        dequantize_packed_ue8m0(q_fp8, q_scale),
        dequantize_packed_ue8m0(k_fp8, k_scale),
    ).relu()
    return (dense * w_bsh[0].float().unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)
