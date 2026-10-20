# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""N-gram hashing and gated row-sharded lookup with caller-supplied policies."""
import weakref
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.tensor_parallel.mappings import (
    async_gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
)


class NgramHash(nn.Module):
    """Hash a full local batch; packed boundaries restart every lookback."""

    def __init__(self, token_map, pad_id, multipliers, primes):
        super().__init__()
        mapping = torch.as_tensor(token_map, dtype=torch.int64)
        if multipliers.ndim != 2 or primes.ndim != 3:
            raise ValueError(
                "Expected multipliers [layers,order] and primes [layers,order-1,heads]"
            )
        if primes.shape[:2] != (multipliers.shape[0], multipliers.shape[1] - 1):
            raise ValueError("Hash layout and multiplier shape mismatch")
        self.pad_id = int(mapping[pad_id])
        self.register_buffer("token_map", mapping, persistent=False)
        self.register_buffer("multipliers", multipliers.to(torch.int64), persistent=False)
        self.register_buffer("primes", primes.to(torch.int64), persistent=False)
        flat = primes.flatten(1)
        self.register_buffer("offsets", flat.cumsum(-1) - flat, persistent=False)

    def forward(self, input_ids, token_mask=None, *, cu_seqlens=None):
        if input_ids.ndim != 2:
            raise ValueError("Expected input IDs [B,S]")
        b, length = input_ids.shape
        if token_mask is not None and (
            token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool
        ):
            raise ValueError("Expected boolean token mask [B,S]")
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask, -1)
        positions = torch.arange(length, device=input_ids.device).expand(b, -1)
        starts = torch.zeros_like(positions)
        if cu_seqlens is not None:
            if b != 1:
                raise ValueError("THD packing requires B=1")
            bounds = cu_seqlens.tolist()
            if bounds[0] != 0 or bounds[-1] != length:
                raise ValueError("cu_seqlens must cover the packed sequence")
            for begin, end in zip(bounds[:-1], bounds[1:]):
                starts[:, begin:end] = begin
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        history = []
        for shift in range(self.multipliers.shape[1]):
            source = compressed.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions - shift < starts) | (source == -1)
            history.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(history, -1).unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.multipliers.shape[1]):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, -1) + self.offsets


class Engram(nn.Module):
    def __init__(self, hidden_size, copies, embedding, projection, *, eps=1e-6):
        super().__init__()
        self.dim = hidden_size
        self.copies = copies
        self.eps = eps
        self.embed = embedding
        self.wkv = projection
        self.q_weight = nn.Parameter(torch.ones(copies, hidden_size))
        self.k_weight = nn.Parameter(torch.ones(copies, hidden_size))

    def forward(self, hidden, hash_ids, token_mask=None):
        if hidden.ndim != 4 or hidden.shape[-2:] != (self.copies, self.dim):
            raise ValueError("Expected residual stream [B,S,HC,D]")
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.copies * self.dim, self.dim], -1)
        key = key.float().unflatten(-1, (self.copies, self.dim))
        h = hidden.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (
            (h * key * self.q_weight.float() * self.k_weight.float()).sum(-1)
            * rstd
            * self.dim**-0.5
        )
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)


class ShardedFrozenTable(nn.Module):
    """Frozen row-sharded FP8 table (E4M3 rows, E8M0 scale per 32) in pinned host memory."""

    def __init__(self, rows, width, group, *, block=32, initialize=True):
        super().__init__()
        if width % block:
            raise ValueError("FP8 table width must be a multiple of its scale block")
        self.group, self.rows, self.block = group, rows, block
        shard = -(-rows // group.size())
        self.embedding = nn.Module()  # holds the persistent byte buffers and row metadata
        bounds = tuple(i * shard for i in range(group.size() + 1))
        self.embedding.lookup = SimpleNamespace(group=group, boundaries=bounds)
        self.embedding.register_buffer("weight", _pinned_empty((shard, width), torch.float8_e4m3fn))
        scale = _pinned_empty((shard, width // block), torch.float8_e8m0fnu)
        self.embedding.register_buffer("scale", scale)
        if initialize:  # random rows, unit scales; a release load skips this and fills the bytes
            self.embedding.weight.copy_(torch.randn(shard, width) * 0.02)
            self.embedding.scale.view(torch.uint8).fill_(127)  # 2**0

    def _apply(self, fn, recurse=True):
        return self  # the frozen FP8 bytes stay on the host

    def forward(self, ids):
        rows, size = ids.shape[0], self.group.size()
        if size > 1:  # every rank's IDs, padded to a common row count
            sizes = async_gather_from_sequence_parallel_region(
                ids.new_tensor([rows]), group=self.group
            )
            maximum = int(sizes.wait().max())
            padded = F.pad(ids, (0, 0) * (ids.ndim - 1) + (0, maximum - rows))
            ids = async_gather_from_sequence_parallel_region(padded, group=self.group).wait()
        begin, end = self.embedding.lookup.boundaries[self.group.rank() : self.group.rank() + 2]
        local = ids - begin
        owned = (local >= 0) & (local < end - begin)
        picked = local[owned].cpu()
        values = self.embedding.weight[picked].to(ids.device)
        scales = self.embedding.scale[picked].to(ids.device)
        decoded = values.float().unflatten(-1, (-1, self.block)) * scales.float().unsqueeze(-1)
        out = torch.zeros(*ids.shape, values.shape[-1], dtype=torch.bfloat16, device=ids.device)
        out[owned] = decoded.flatten(-2).to(torch.bfloat16)
        if size > 1:
            out = reduce_from_tensor_model_parallel_region(out, self.group)
            out = out[self.group.rank() * maximum : self.group.rank() * maximum + rows]
        return out


def _pinned_empty(shape, dtype):
    """Exact-size uninitialized host buffer, page-locked in place when CUDA is available"""
    nbytes = shape[0] * shape[1] * dtype.itemsize
    owner = np.empty(max(nbytes, 1), dtype=np.uint8)
    out = torch.from_numpy(owner)[:nbytes].view(dtype).view(shape)
    if torch.cuda.is_available() and nbytes:
        pointer, cudart = owner.ctypes.data, torch.cuda.cudart()
        if int(cudart.cudaHostRegister(pointer, nbytes, 0)) != 0:
            raise RuntimeError("cudaHostRegister failed for an Engram table")
        weakref.finalize(owner, cudart.cudaHostUnregister, pointer)
    return out
