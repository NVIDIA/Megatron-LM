# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF ↔ Megatron Lite weight conversion.

Everything needed for HuggingFace safetensors ↔ Megatron Lite model conversion:
- HFWeights protocol (model implements this)
- SafeTensorReader / save_safetensors (file I/O)
- Tensor utilities (split_dim, allgather_concat, remap_layer_index, ...)
- Generic load_hf_weights / export_hf_weights / save_hf_weights (orchestration)
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Generator, Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file as _safe_save

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - older torch without DTensor
    DTensor = None  # type: ignore[assignment]

from megatron.lite.primitive.ckpt.weight_sync_probe import get_weight_sync_probe


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu":
        return tensor
    with get_weight_sync_probe().measure("d2h", nbytes=_tensor_nbytes(tensor), device=tensor.device):
        return tensor.cpu()


def _copy_to_cpu(dst: torch.Tensor, src: torch.Tensor) -> None:
    if src.device.type == "cpu":
        dst.copy_(src)
        return
    with get_weight_sync_probe().measure("d2h", nbytes=_tensor_nbytes(src), device=src.device):
        dst.copy_(src)


def _ep_all_gather(outputs: list[torch.Tensor], tensor: torch.Tensor, group) -> None:
    with get_weight_sync_probe().measure(
        "ep_gather",
        nbytes=sum(_tensor_nbytes(output) for output in outputs),
        device=tensor.device,
    ):
        dist.all_gather(outputs, tensor, group=group)


def _native_to_hf(spec: HFWeights, name: str, tensor: torch.Tensor):
    with get_weight_sync_probe().measure("mapping") as sample:
        mapped = spec.native_to_hf(name, tensor)
        sample.nbytes = sum(_tensor_nbytes(value) for _, value in mapped)
        return mapped


DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES = 2 * 1024**3


def _materialize_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reconstruct a plain local tensor from an FSDP2 ``DTensor`` parameter.

    FSDP2 (``fully_shard``) stores parameters as ``DTensor`` sharded over the
    data-parallel mesh, while manual TP/EP sharding leaves each rank holding its
    own local shard as a regular tensor. The HF export gather (``_gather_dense`` /
    ``_gather_expert``) and the downstream rollout weight sender both assume plain
    ``torch.Tensor`` inputs; handing them a ``DTensor`` raises
    ``aten.copy_.default: got mixed torch.Tensor and DTensor``. ``full_tensor()``
    gathers the FSDP shards back into the full (TP/EP-local) tensor; non-DTensor
    params (dist_opt backend) pass through untouched.
    """
    if DTensor is not None and isinstance(tensor, DTensor):
        with get_weight_sync_probe().measure("fsdp_gather", device=tensor.device) as sample:
            result = tensor.full_tensor()
            sample.nbytes = _tensor_nbytes(result)
            return result
    return tensor


@runtime_checkable
class HFWeights(Protocol):
    """Protocol for HF ↔ Megatron Lite weight conversion.

    Model-specific implementations only do tensor math, never distributed comm.
    """

    def weight_map(self) -> dict[str, list[str]]:
        """Megatron Lite param name → [HF param names]. Multiple = concat (QKV, gate+up)."""
        ...

    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        """Convert HF tensors → single Megatron Lite tensor (e.g. merge QKV)."""
        ...

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert Megatron Lite tensor → [(hf_name, hf_tensor)] (e.g. split QKV back)."""
        ...

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        """TP sharding: ``(split_dim, 0=TP|1=ETP)``, or None if replicated."""
        ...

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        """If native_name is a fused QKV weight, return (num_q_heads, num_kv_heads, head_dim).

        Needed for correct GQA TP sharding — Q/K/V must be split independently.
        Return None for non-QKV parameters.
        """
        return None

    @property
    def num_experts(self) -> int:
        """Total number of experts (needed for EP gather index math)."""
        ...

    def is_expert(self, native_name: str) -> bool:
        """Whether this param belongs to an expert (for EP sharding)."""
        ...

    def expert_global_id(self, native_name: str) -> int | None:
        """Global expert ID from synthetic name. None if not expert."""
        ...

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        """Synthetic expert name → actual model param name."""
        ...


# ======================================================================
# SafeTensors I/O
# ======================================================================


class SafeTensorReader:
    """Read tensors from an HF safetensors directory.

    As a context manager, keep one lazy mmap handle per referenced shard and
    close every handle on exit. The one-shot API retains its historical
    open/read/close behavior.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.index = self._load_index()
        self._stack: ExitStack | None = None
        self._handles: dict[Path, Any] = {}

    def __enter__(self) -> SafeTensorReader:
        if self._stack is not None:
            raise RuntimeError("SafeTensorReader context cannot be entered twice")
        self._stack = ExitStack()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        stack, self._stack = self._stack, None
        self._handles.clear()
        if stack is not None:
            stack.close()

    def _load_index(self) -> dict[str, str]:
        idx_file = self.path / "model.safetensors.index.json"
        if idx_file.exists():
            with open(idx_file) as f:
                return json.load(f)["weight_map"]
        return {}

    def get_tensor(self, name: str) -> torch.Tensor:
        if self.index:
            filepath = self.path / self.index[name]
        else:
            filepath = self.path / "model.safetensors"
        if self._stack is not None:
            handle = self._handles.get(filepath)
            if handle is None:
                handle = self._stack.enter_context(
                    safe_open(str(filepath), framework="pt", device="cpu")
                )
                self._handles[filepath] = handle
            return handle.get_tensor(name)
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip nested wrapper modules like DDP -> model."""
    base_model = model
    seen: set[int] = set()
    while hasattr(base_model, "module"):
        ident = id(base_model)
        if ident in seen:
            break
        seen.add(ident)
        next_model = base_model.module
        if not isinstance(next_model, nn.Module) or next_model is base_model:
            break
        base_model = next_model
    return base_model


def save_safetensors(
    tensors: dict[str, torch.Tensor], path: str, filename: str = "model.safetensors"
) -> None:
    os.makedirs(path, exist_ok=True)
    _safe_save(tensors, os.path.join(path, filename))


def _resolve_export_dtype(export_dtype: str | torch.dtype | None) -> torch.dtype | None:
    if export_dtype is None:
        return None
    if isinstance(export_dtype, torch.dtype):
        return export_dtype
    normalized = str(export_dtype).lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported export_dtype={export_dtype!r}")
    return aliases[normalized]


def _cast_export_tensor(tensor: torch.Tensor, export_dtype: torch.dtype | None) -> torch.Tensor:
    if export_dtype is None or not tensor.is_floating_point():
        return tensor
    return tensor.to(dtype=export_dtype)


# ======================================================================
# Tensor utilities
# ======================================================================


def split_dim(tensor: torch.Tensor, rank: int, world: int, dim: int = 0) -> torch.Tensor:
    if world <= 1:
        return tensor
    return tensor.chunk(world, dim=dim)[rank].contiguous()


def split_qkv(
    tensor: torch.Tensor, rank: int, world: int, num_q_heads: int, num_kv_heads: int, head_dim: int
) -> torch.Tensor:
    """TP-shard a fused [Q, K, V] weight, splitting Q/K/V heads independently.

    Naive ``split_dim`` would slice across the Q/K/V boundary incorrectly
    when num_q_heads != num_kv_heads (GQA).
    """
    if world <= 1:
        return tensor
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q = tensor[:q_size]
    k = tensor[q_size : q_size + kv_size]
    v = tensor[q_size + kv_size :]
    q_shard = q.chunk(world, dim=0)[rank]
    k_shard = k.chunk(world, dim=0)[rank]
    v_shard = v.chunk(world, dim=0)[rank]
    return torch.cat([q_shard, k_shard, v_shard], dim=0).contiguous()


def split_gate_up(tensor: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if world <= 1:
        return tensor
    ffn = tensor.shape[0] // 2
    gate = tensor[:ffn].chunk(world, dim=0)[rank]
    up = tensor[ffn:].chunk(world, dim=0)[rank]
    return torch.cat([gate, up], dim=0).contiguous()


def allgather_concat(
    tensor: torch.Tensor, world_size: int, group: dist.ProcessGroup | None, dim: int
) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def bucketed_all_gather_into_tensor(
    bucket: list[tuple[str, torch.Tensor]],
    *,
    group: dist.ProcessGroup | None,
    group_size: int,
    buffer_max_size_bytes: int = DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES,
) -> list[tuple[str, torch.Tensor, list[torch.Tensor]]]:
    """Gather a same-dtype tensor bucket through bounded flat GPU buffers."""
    if not bucket:
        return []
    if group_size <= 1:
        return [(name, tensor, [tensor]) for name, tensor in bucket]

    dtype = bucket[0][1].dtype
    device = bucket[0][1].device
    if any(tensor.dtype != dtype for _, tensor in bucket):
        raise ValueError("bucket tensors must share the same dtype")
    if buffer_max_size_bytes <= 0:
        raise ValueError("buffer_max_size_bytes must be positive")

    element_size = bucket[0][1].element_size()
    per_rank_buffer_bytes = max(buffer_max_size_bytes // group_size, element_size)
    max_chunk_numel = max(1, per_rank_buffer_bytes // element_size)
    flat_shards = [tensor.reshape(-1) for _, tensor in bucket]
    numel_per_tensor = [tensor.numel() for tensor in flat_shards]
    total_numel = sum(numel_per_tensor)

    # Normal export buckets are capped before entering this function, so they
    # fit in one collective. Keep the rank-major receive buffer alive through
    # returned views instead of allocating and copying ``num_tensors * world``
    # temporary shards. Oversized single tensors use the bounded chunked path
    # below.
    if total_numel <= max_chunk_numel:
        send_buffer = torch.empty(total_numel, dtype=dtype, device=device)
        send_views = list(send_buffer.split(numel_per_tensor))
        torch._foreach_copy_(send_views, flat_shards)
        recv_buffer = torch.empty(group_size * total_numel, dtype=dtype, device=device)
        dist.all_gather_into_tensor(recv_buffer, send_buffer, group=group)

        offsets = []
        offset = 0
        for numel in numel_per_tensor:
            offsets.append(offset)
            offset += numel
        return [
            (
                name,
                tensor,
                [
                    recv_buffer[
                        rank * total_numel + offsets[idx] : rank * total_numel
                        + offsets[idx]
                        + numel_per_tensor[idx]
                    ].view_as(tensor)
                    for rank in range(group_size)
                ],
            )
            for idx, (name, tensor) in enumerate(bucket)
        ]

    gathered_shards_by_rank = [
        [torch.empty_like(tensor) for _, tensor in bucket] for _ in range(group_size)
    ]
    gathered_flat_views = [
        [tensor.view(-1) for tensor in rank_shards] for rank_shards in gathered_shards_by_rank
    ]

    send_buffer = torch.empty(max_chunk_numel, dtype=dtype, device=device)
    recv_buffer = torch.empty(group_size * max_chunk_numel, dtype=dtype, device=device)
    tensor_idx = 0
    tensor_offset = 0
    while tensor_idx < len(bucket):
        chunk_segments = []
        chunk_numel = 0
        while tensor_idx < len(bucket) and chunk_numel < max_chunk_numel:
            available = numel_per_tensor[tensor_idx] - tensor_offset
            take_numel = min(available, max_chunk_numel - chunk_numel)
            send_buffer[chunk_numel : chunk_numel + take_numel].copy_(
                flat_shards[tensor_idx][tensor_offset : tensor_offset + take_numel]
            )
            chunk_segments.append((tensor_idx, tensor_offset, chunk_numel, take_numel))
            chunk_numel += take_numel
            tensor_offset += take_numel
            if tensor_offset == numel_per_tensor[tensor_idx]:
                tensor_idx += 1
                tensor_offset = 0

        recv_view = recv_buffer[: group_size * chunk_numel]
        dist.all_gather_into_tensor(recv_view, send_buffer[:chunk_numel], group=group)
        for rank in range(group_size):
            rank_recv_view = recv_view[rank * chunk_numel : (rank + 1) * chunk_numel]
            for tensor_i, output_offset, chunk_offset, segment_numel in chunk_segments:
                gathered_flat_views[rank][tensor_i][
                    output_offset : output_offset + segment_numel
                ].copy_(rank_recv_view[chunk_offset : chunk_offset + segment_numel])

    return [
        (
            name,
            tensor,
            [gathered_shards_by_rank[rank][idx] for rank in range(group_size)],
        )
        for idx, (name, tensor) in enumerate(bucket)
    ]


def _iter_bucketed_materialized_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    buffer_max_size_bytes: int = DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Materialize sharded DTensors through bounded, layout-aware flat gathers.

    Replicated DTensors and plain tensors are already complete on every DP rank,
    so they bypass collectives.  They remain in the pending stream until the
    current sharded bucket is flushed, which preserves parameter order without
    breaking a flat bucket at every replicated parameter.
    """
    bucket: list[tuple[str, torch.Tensor]] = []
    metadata: list[tuple[int, tuple[int, ...]]] = []
    pending: list[tuple[str, torch.Tensor | None, int | None]] = []
    bucket_bytes = 0
    bucket_group = None
    bucket_group_ranks: tuple[int, ...] | None = None
    bucket_group_size = 1

    def _flush_bucket():
        nonlocal bucket, metadata, pending, bucket_bytes
        nonlocal bucket_group, bucket_group_ranks, bucket_group_size
        if not pending:
            return
        materialized: list[torch.Tensor] = []
        if bucket:
            with get_weight_sync_probe().measure(
                "fsdp_gather", device=bucket[0][1].device
            ) as sample:
                gathered = bucketed_all_gather_into_tensor(
                    bucket,
                    group=bucket_group,
                    group_size=bucket_group_size,
                    buffer_max_size_bytes=buffer_max_size_bytes,
                )
                total_global_numel = sum(
                    local_tensor.numel() * bucket_group_size
                    for _, local_tensor in bucket
                )
                materialized_buffer = torch.empty(
                    total_global_numel,
                    dtype=bucket[0][1].dtype,
                    device=bucket[0][1].device,
                )
                source_views: list[torch.Tensor] = []
                destination_views: list[torch.Tensor] = []
                materialized_offset = 0
                for (_, _, shards), (shard_dim, global_shape) in zip(
                    gathered, metadata, strict=True
                ):
                    global_numel = 1
                    for size in global_shape:
                        global_numel *= size
                    full = materialized_buffer[
                        materialized_offset : materialized_offset + global_numel
                    ].view(global_shape)
                    local_extent = shards[0].shape[shard_dim]
                    for rank, shard in enumerate(shards):
                        source_views.append(shard)
                        destination_views.append(
                            full.narrow(
                                shard_dim,
                                rank * local_extent,
                                local_extent,
                            )
                        )
                    materialized.append(full)
                    materialized_offset += global_numel
                torch._foreach_copy_(destination_views, source_views)
                sample.nbytes = _tensor_nbytes(materialized_buffer)

        outputs = []
        for name, local_tensor, bucket_idx in pending:
            if bucket_idx is None:
                assert local_tensor is not None
                outputs.append((name, local_tensor))
            else:
                outputs.append((name, materialized[bucket_idx]))
        bucket = []
        metadata = []
        pending = []
        bucket_bytes = 0
        bucket_group = None
        bucket_group_ranks = None
        bucket_group_size = 1
        yield from outputs

    for name, tensor in named_tensors:
        if DTensor is None or not isinstance(tensor, DTensor):
            if bucket:
                pending.append((name, tensor, None))
            else:
                yield name, tensor
            continue

        placements = tuple(tensor.placements)
        placement = placements[0] if len(placements) == 1 else None
        shard_dim = getattr(placement, "dim", None)
        if placement is not None and type(placement).__name__ == "Replicate":
            local = tensor.to_local()
            if bucket:
                pending.append((name, local, None))
            else:
                yield name, local
            continue
        if (
            placement is None
            or type(placement).__name__ != "Shard"
            or shard_dim is None
        ):
            yield from _flush_bucket()
            yield name, _materialize_dtensor(tensor)
            continue

        local = tensor.to_local()
        global_shape = tuple(int(size) for size in tensor.shape)
        shard_dim = int(shard_dim) % len(global_shape) if global_shape else -1
        group = tensor.device_mesh.get_group(0)
        group_size = dist.get_world_size(group)
        if group_size <= 1:
            if bucket:
                pending.append((name, local, None))
            else:
                yield name, local
            continue
        group_ranks = tuple(dist.get_process_group_ranks(group))
        evenly_sharded = (
            shard_dim >= 0
            and global_shape[shard_dim] % group_size == 0
            and local.shape[shard_dim] * group_size == global_shape[shard_dim]
        )
        if not evenly_sharded:
            yield from _flush_bucket()
            yield name, _materialize_dtensor(tensor)
            continue

        local_bytes = local.numel() * local.element_size()
        per_rank_limit = max(buffer_max_size_bytes // group_size, 1)
        if local_bytes > per_rank_limit:
            yield from _flush_bucket()
            local_extent = local.shape[shard_dim]
            bytes_per_extent = local_bytes // local_extent
            chunk_extent = max(per_rank_limit // bytes_per_extent, 1)
            full = torch.empty(global_shape, dtype=local.dtype, device=local.device)
            with get_weight_sync_probe().measure(
                "fsdp_gather", device=local.device
            ) as sample:
                for offset in range(0, local_extent, chunk_extent):
                    extent = min(chunk_extent, local_extent - offset)
                    local_chunk = local.narrow(shard_dim, offset, extent)
                    _, _, gathered_chunks = bucketed_all_gather_into_tensor(
                        [(name, local_chunk)],
                        group=group,
                        group_size=group_size,
                        buffer_max_size_bytes=buffer_max_size_bytes,
                    )[0]
                    for rank, gathered_chunk in enumerate(gathered_chunks):
                        full.narrow(
                            shard_dim,
                            rank * local_extent + offset,
                            extent,
                        ).copy_(gathered_chunk)
                sample.nbytes = _tensor_nbytes(full)
            yield name, full
            continue

        incompatible = bool(bucket) and (
            local.dtype != bucket[0][1].dtype
            or local.device != bucket[0][1].device
            or group_ranks != bucket_group_ranks
            or group_size != bucket_group_size
            or bucket_bytes + local_bytes > per_rank_limit
        )
        if incompatible:
            yield from _flush_bucket()

        bucket.append((name, local))
        metadata.append((shard_dim, global_shape))
        pending.append((name, None, len(bucket) - 1))
        bucket_bytes += local_bytes
        if bucket_group is None:
            bucket_group = group
            bucket_group_ranks = group_ranks
            bucket_group_size = group_size
        if bucket_bytes >= per_rank_limit:
            yield from _flush_bucket()

    yield from _flush_bucket()


def _maybe_cpu(tensor: torch.Tensor, *, cpu: bool) -> torch.Tensor:
    return _to_cpu(tensor) if cpu else tensor


def remap_layer_index(name: str, global_to_local: dict[int, int]) -> str | None:
    if not global_to_local:
        return name
    m = re.match(r"(layers\.)(\d+)(\..*)", name)
    if not m:
        return name
    gidx = int(m.group(2))
    if gidx not in global_to_local:
        return None
    return f"{m.group(1)}{global_to_local[gidx]}{m.group(3)}"


def extract_layer_idx(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else 0


def parse_expert_idx(name: str) -> int:
    m = re.search(r"weight(\d+)$", name)
    return int(m.group(1)) if m else 0


def set_expert_idx(name: str, idx: int) -> str:
    return re.sub(r"weight\d+$", f"weight{idx}", name)


def to_global_layer_name(name: str, layer_map: dict[int, int]) -> str:
    if not layer_map:
        return name

    def _replace(m: re.Match) -> str:
        return f"layers.{layer_map.get(int(m.group(1)), int(m.group(1)))}."

    return re.sub(r"layers\.(\d+)\.", _replace, name)


def gather_gate_up(tensor: torch.Tensor, world_size: int, group: dist.ProcessGroup) -> torch.Tensor:
    ffn_local = tensor.shape[0] // 2
    gate_full = allgather_concat(tensor[:ffn_local], world_size, group, dim=0)
    up_full = allgather_concat(tensor[ffn_local:], world_size, group, dim=0)
    return torch.cat([gate_full, up_full], dim=0)


# ======================================================================
# Generic load / export / save using HFWeights
# ======================================================================


def load_hf_weights(
    model: nn.Module, hf_path: str, spec: HFWeights, ps, *, vocab_size: int | None = None
) -> None:
    """Load HF safetensors into a Megatron Lite model using HFWeights.

    Handles PP layer filtering, TP split, EP shard assignment.
    ``ps`` is a ParallelState (lazy import to avoid GPU dep at module level).
    """
    from megatron.lite.primitive.parallel import pad_vocab_for_tp
    from megatron.lite.primitive.utils import log_rank0

    base_model = unwrap_model(model)
    reader = SafeTensorReader(hf_path)
    wmap = spec.weight_map()

    global_to_local: dict[int, int] = (
        {gi: li for li, gi in enumerate(base_model.layer_indices)}
        if hasattr(base_model, "layer_indices")
        else {}
    )

    state = base_model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    num_experts_total = getattr(spec, "num_experts", None)
    expert_shard = None
    if num_experts_total is None:
        expert_ids = [spec.expert_global_id(name) for name in wmap]
        expert_ids = [expert_id for expert_id in expert_ids if expert_id is not None]
        if expert_ids:
            num_experts_total = max(expert_ids) + 1
    if num_experts_total:
        from megatron.lite.primitive.utils import ensure_divisible

        experts_per_rank = ensure_divisible(num_experts_total, ps.ep_size)
        local_start = ps.ep_rank * experts_per_rank
        expert_shard = (experts_per_rank, local_start)

    for native_name, hf_names in wmap.items():
        mapped = remap_layer_index(native_name, global_to_local)
        if mapped is None:
            continue

        expert_gid = spec.expert_global_id(mapped)
        if expert_gid is not None:
            _load_expert_weight(
                mapped, hf_names, reader, spec, ps, loaded, expert_gid, expert_shard
            )
            continue

        hf_tensors = [reader.get_tensor(n) for n in hf_names]
        tensor = spec.hf_to_native(mapped, hf_tensors)

        tp_info = spec.tp_spec(mapped)
        if tp_info is not None:
            split_d, tp_or_etp = tp_info
            if tp_or_etp == 0:
                if vocab_size is not None and ("embed" in mapped or "head" in mapped):
                    padded = pad_vocab_for_tp(vocab_size, ps.tp_size)
                    if tensor.size(0) < padded:
                        pad = torch.zeros(
                            padded - tensor.size(0), *tensor.shape[1:], dtype=tensor.dtype
                        )
                        tensor = torch.cat([tensor, pad], dim=0)
                qkv = spec.qkv_spec(mapped) if hasattr(spec, "qkv_spec") else None
                if qkv is not None:
                    tensor = split_qkv(tensor, ps.tp_rank, ps.tp_size, *qkv)
                else:
                    tensor = split_dim(tensor, ps.tp_rank, ps.tp_size, dim=split_d)
            else:
                tensor = split_dim(tensor, ps.etp_rank, ps.etp_size, dim=split_d)

        actual = _resolve_param_name(mapped, state)
        if actual:
            loaded[actual] = tensor.to(dtype=torch.bfloat16)

    for name, param in base_model.named_parameters():
        if name in loaded:
            param.data.copy_(loaded[name])
        elif "lora" in name.lower() or "adapter" in name.lower():
            continue
        else:
            log_rank0(f"WARNING: {name} not loaded from checkpoint")


def _load_expert_weight(native_name, hf_names, reader, spec, ps, loaded, expert_gid, expert_shard):
    if expert_shard is None:
        raise RuntimeError("Expert weight encountered but expert shard metadata is unavailable.")
    experts_per_rank, local_start = expert_shard
    if expert_gid < local_start or expert_gid >= local_start + experts_per_rank:
        return

    hf_tensors = [reader.get_tensor(n) for n in hf_names]
    tensor = spec.hf_to_native(native_name, hf_tensors)

    if ps.etp_size > 1:
        tp_info = spec.tp_spec(native_name)
        if tp_info is not None:
            split_d, _ = tp_info
            if "fc1" in native_name:
                tensor = split_gate_up(tensor, ps.etp_rank, ps.etp_size)
            else:
                tensor = split_dim(tensor, ps.etp_rank, ps.etp_size, dim=split_d)

    loaded[spec.expert_local_name(native_name, expert_gid - local_start)] = tensor.to(
        dtype=torch.bfloat16
    )


def _resolve_param_name(name: str, state_dict: dict) -> str | None:
    if name in state_dict:
        return name
    for key in state_dict:
        if name in key:
            return key
    return None


def _merge_dense_shards(
    name: str,
    tensor: torch.Tensor,
    shards: list[torch.Tensor],
    spec: HFWeights,
) -> torch.Tensor:
    custom_merge = getattr(spec, "merge_dense_shards", None)
    if callable(custom_merge):
        merged = custom_merge(name, shards)
        if merged is not None:
            return merged

    tp_info = spec.tp_spec(name)
    if tp_info is None:
        return tensor
    split_dim, tp_or_etp = tp_info
    if tp_or_etp != 0:
        return tensor
    return torch.cat(shards, dim=split_dim)


def export_hf_weights(
    model: nn.Module | list[nn.Module],
    spec: HFWeights,
    ps,
    *,
    vocab_size: int | None = None,
    limit: int | None = None,
    rank0_only: bool = False,
    export_dtype: str | torch.dtype | None = None,
    cpu: bool = False,
    buffer_max_size_bytes: int = DEFAULT_EXPORT_BUFFER_MAX_SIZE_BYTES,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Export model weights as HF-format (name, tensor) pairs.

    Gathers across TP/ETP/EP/PP so the output is the full unsharded HF state on
    every participating rank. RL weight sync needs every colocated rollout rank
    to receive weights; save paths can pass ``rank0_only=True`` to avoid
    materializing duplicate writers.
    """
    if isinstance(model, nn.ModuleList):
        chunks: list[nn.Module] = list(model)
    elif isinstance(model, list):
        chunks = model
    else:
        chunks = [model]

    rank = dist.get_rank() if dist.is_initialized() else 0
    resolved_export_dtype = _resolve_export_dtype(export_dtype)

    if ps.pp_size <= 1:
        exported_params = 0
        expert_bucket: list[tuple[str, torch.Tensor]] = []
        expert_bucket_bytes = 0
        packed_expert_buffers: dict[str, dict[int, torch.Tensor]] = {}
        expert_bucket_limit_bytes = (
            buffer_max_size_bytes
            if ps.ep_size <= 1
            else max(buffer_max_size_bytes // ps.ep_size, 1)
        )
        dense_bucket: list[tuple[str, torch.Tensor]] = []
        dense_bucket_bytes = 0
        dense_bucket_limit_bytes = (
            buffer_max_size_bytes
            if ps.tp_size <= 1
            else max(buffer_max_size_bytes // ps.tp_size, 1)
        )

        def _iter_mapped(gathered_tensors: dict[str, torch.Tensor]):
            if rank0_only and rank != 0:
                return
            for native_name, gathered_tensor in gathered_tensors.items():
                if vocab_size is not None and ("embed" in native_name or "head" in native_name):
                    gathered_tensor = gathered_tensor[:vocab_size]
                for hf_name, hf_tensor in _native_to_hf(spec, native_name, gathered_tensor):
                    yield hf_name, _cast_export_tensor(hf_tensor, resolved_export_dtype)

        def _flush_dense_bucket():
            nonlocal dense_bucket, dense_bucket_bytes
            if not dense_bucket:
                return
            gathered_bucket = bucketed_all_gather_into_tensor(
                dense_bucket,
                group=ps.tp_group,
                group_size=ps.tp_size,
                buffer_max_size_bytes=buffer_max_size_bytes,
            )
            dense_bucket = []
            dense_bucket_bytes = 0
            for native_name, local_tensor, shards in gathered_bucket:
                merged = _merge_dense_shards(native_name, local_tensor, shards, spec)
                yield from _iter_mapped({native_name: _maybe_cpu(merged, cpu=cpu)})

        def _flush_expert_bucket():
            nonlocal expert_bucket, expert_bucket_bytes
            if not expert_bucket:
                return
            gathered_bucket = bucketed_all_gather_into_tensor(
                expert_bucket,
                group=ps.ep_group,
                group_size=ps.ep_size,
                buffer_max_size_bytes=buffer_max_size_bytes,
            )
            expert_bucket = []
            expert_bucket_bytes = 0
            experts_per_rank = spec.num_experts // ps.ep_size
            for native_name, _, shards in gathered_bucket:
                local_idx = parse_expert_idx(native_name)
                packed_group_name = getattr(spec, "packed_expert_group_name", None)
                packed_name = (
                    packed_group_name(native_name)
                    if callable(packed_group_name)
                    else None
                )
                for ep_rank, shard in enumerate(shards):
                    global_idx = ep_rank * experts_per_rank + local_idx
                    global_name = set_expert_idx(native_name, global_idx)
                    export_shard = _maybe_cpu(shard, cpu=cpu)
                    if packed_name is None:
                        yield from _iter_mapped({global_name: export_shard})
                        continue
                    packed_expert_buffers.setdefault(packed_name, {})[
                        global_idx
                    ] = export_shard
                if packed_name is not None:
                    packed = packed_expert_buffers[packed_name]
                    if len(packed) == spec.num_experts:
                        packed_tensor = torch.stack(
                            [packed[idx] for idx in range(spec.num_experts)], dim=0
                        )
                        del packed_expert_buffers[packed_name]
                        yield from _iter_mapped({packed_name: packed_tensor})

        def _iter_native_params():
            for chunk in chunks:
                base_chunk = unwrap_model(chunk)
                layer_map = (
                    {
                        i: base_chunk.layer_indices[i]
                        for i in range(len(base_chunk.layer_indices))
                    }
                    if hasattr(base_chunk, "layer_indices")
                    else {}
                )
                for name, param in base_chunk.named_parameters():
                    tensor = _cast_export_tensor(
                        param.data.detach(), resolved_export_dtype
                    )
                    yield to_global_layer_name(name, layer_map), tensor

        native_params = _iter_native_params()
        materialized_params = (
            _iter_bucketed_materialized_tensors(
                native_params, buffer_max_size_bytes=buffer_max_size_bytes
            )
            if not cpu and limit is None
            else (
                (name, _materialize_dtensor(tensor)) for name, tensor in native_params
            )
        )
        for gname, tensor in materialized_params:
            gathered_one: dict[str, torch.Tensor] = {}
            if spec.is_expert(gname):
                yield from _flush_dense_bucket()
                if limit is None:
                    tensor = _gather_expert_etp(gname, tensor, spec, ps)
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    should_flush = bool(expert_bucket) and (
                        tensor.dtype != expert_bucket[0][1].dtype
                        or expert_bucket_bytes + tensor_bytes > expert_bucket_limit_bytes
                    )
                    if should_flush:
                        yield from _flush_expert_bucket()
                    expert_bucket.append((gname, tensor))
                    expert_bucket_bytes += tensor_bytes
                    exported_params += 1
                    if (
                        len(expert_bucket) >= 4
                        or expert_bucket_bytes >= expert_bucket_limit_bytes
                    ):
                        yield from _flush_expert_bucket()
                    continue
                _gather_expert(gname, tensor, spec, ps, gathered_one, cpu=cpu)
            else:
                yield from _flush_expert_bucket()
                tp_info = spec.tp_spec(gname)
                is_tp = tp_info is not None and tp_info[1] == 0 and ps.tp_size > 1
                if is_tp:
                    tensor_bytes = tensor.numel() * tensor.element_size()
                    should_flush = bool(dense_bucket) and (
                        tensor.dtype != dense_bucket[0][1].dtype
                        or dense_bucket_bytes + tensor_bytes > dense_bucket_limit_bytes
                    )
                    if should_flush:
                        yield from _flush_dense_bucket()
                    dense_bucket.append((gname, tensor))
                    dense_bucket_bytes += tensor_bytes
                    exported_params += 1
                    if dense_bucket_bytes >= dense_bucket_limit_bytes:
                        yield from _flush_dense_bucket()
                    if limit is not None and exported_params >= limit:
                        yield from _flush_dense_bucket()
                        return
                    continue

                yield from _flush_dense_bucket()
                gathered_one[gname] = _maybe_cpu(tensor, cpu=cpu)

            exported_params += 1
            yield from _iter_mapped(gathered_one)

            if limit is not None and exported_params >= limit:
                return

        yield from _flush_dense_bucket()
        yield from _flush_expert_bucket()
        return

    # Streaming pp>1 export. The legacy path materialized every PP stage's
    # gathered dict on every rank (all_gather_object) before the first yield.
    # Instead: each stage in turn lazily gathers just enough params to fill one
    # bounded bucket, broadcasts a (name, shape, dtype) header then the tensors
    # one at a time over the NCCL pp_group; params are yielded and released on
    # arrival. Peak residency = one bucket + one in-flight param. pp=1 untouched.
    gather_cpu = cpu
    emit = not (rank0_only and rank != 0)
    bcast_device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    def _iter_own_gathered() -> Generator[tuple[str, torch.Tensor], None, None]:
        """Lazily yield this PP stage's fully TP/ETP/EP-collapsed params.

        Advanced only during this rank's source turn, so at most one bucket of
        gathered params is ever resident — the whole stage is never built.
        """
        for chunk in chunks:
            base_chunk = unwrap_model(chunk)
            layer_map = (
                {i: base_chunk.layer_indices[i] for i in range(len(base_chunk.layer_indices))}
                if hasattr(base_chunk, "layer_indices")
                else {}
            )
            for name, param in base_chunk.named_parameters():
                gname = to_global_layer_name(name, layer_map)
                t = _materialize_dtensor(param.data.detach())
                if spec.is_expert(gname):
                    one: dict[str, torch.Tensor] = {}
                    _gather_expert(gname, t, spec, ps, one, cpu=False)
                    for gathered_name, gathered_tensor in one.items():
                        yield gathered_name, gathered_tensor
                else:
                    yield gname, _gather_dense(gname, t, spec, ps, cpu=False)

    def _emit_param(native_name: str, tensor: torch.Tensor):
        tensor = _maybe_cpu(tensor, cpu=gather_cpu)
        if vocab_size is not None and ("embed" in native_name or "head" in native_name):
            tensor = tensor[:vocab_size]
        for hf_name, hf_tensor in _native_to_hf(spec, native_name, tensor):
            yield hf_name, _cast_export_tensor(hf_tensor, resolved_export_dtype)

    own = _iter_own_gathered()
    for src_pp in range(ps.pp_size):
        src_global = ps.pp_global_ranks[src_pp]
        is_source = src_pp == ps.pp_rank
        while True:
            bucket: list[tuple[str, torch.Tensor]] = []
            if is_source:
                bucket_bytes = 0
                for gathered_name, gathered_tensor in own:
                    gathered_tensor = gathered_tensor.to(bcast_device).contiguous()
                    bucket.append((gathered_name, gathered_tensor))
                    bucket_bytes += _tensor_nbytes(gathered_tensor)
                    if bucket_bytes >= buffer_max_size_bytes:
                        break
                header: list[Any] = [
                    [(name, tuple(tensor.shape), tensor.dtype) for name, tensor in bucket]
                ]
            else:
                header = [None]
            dist.broadcast_object_list(
                header, src=src_global, group=ps.pp_group, device=bcast_device
            )
            entries = header[0]
            if not entries:
                break  # empty header = end of this stage's stream
            for idx, (native_name, shape, dtype) in enumerate(entries):
                if is_source:
                    tensor = bucket[idx][1]
                else:
                    tensor = torch.empty(shape, dtype=dtype, device=bcast_device)
                dist.broadcast(tensor, src=src_global, group=ps.pp_group)
                if emit:
                    yield from _emit_param(native_name, tensor)
                del tensor
            del bucket
    return


def _gather_dense(
    name: str, tensor: torch.Tensor, spec: HFWeights, ps, *, cpu: bool = True
) -> torch.Tensor:
    """Gather a dense (non-expert) param across TP."""
    custom_gather = getattr(spec, "gather_dense", None)
    if callable(custom_gather):
        gathered = custom_gather(name, tensor, ps)
        if gathered is not None:
            return _maybe_cpu(gathered, cpu=cpu)

    tp_info = spec.tp_spec(name)
    if tp_info is not None and ps.tp_size > 1:
        split_d, tp_or_etp = tp_info
        if tp_or_etp == 0:
            tensor = allgather_concat(tensor, ps.tp_size, ps.tp_group, dim=split_d)
    return _maybe_cpu(tensor, cpu=cpu)


def _gather_expert(
    name: str,
    tensor: torch.Tensor,
    spec: HFWeights,
    ps,
    out: dict[str, torch.Tensor],
    *,
    cpu: bool = True,
) -> None:
    """Gather an expert param across ETP + EP."""
    tensor = _gather_expert_etp(name, tensor, spec, ps)

    # EP gather: global_id = ep_rank * n_local + local_id.
    local_idx = parse_expert_idx(name)
    if ps.ep_size > 1 and ps.ep_group is not None:
        n_local = spec.num_experts // ps.ep_size
        ep_gathered = [torch.empty_like(tensor) for _ in range(ps.ep_size)]
        _ep_all_gather(ep_gathered, tensor.contiguous(), ps.ep_group)
        for ep_rank, ep_tensor in enumerate(ep_gathered):
            global_idx = ep_rank * n_local + local_idx
            out[set_expert_idx(name, global_idx)] = _maybe_cpu(ep_tensor, cpu=cpu)
    else:
        out[name] = _maybe_cpu(tensor, cpu=cpu)


def _gather_expert_etp(name: str, tensor: torch.Tensor, spec: HFWeights, ps) -> torch.Tensor:
    # ETP gather
    if ps.etp_size > 1 and ps.etp_group is not None:
        tp_info = spec.tp_spec(name)
        if tp_info is not None:
            split_d, _ = tp_info
            if "fc1" in name:
                return gather_gate_up(tensor, ps.etp_size, ps.etp_group)
            return allgather_concat(tensor, ps.etp_size, ps.etp_group, dim=split_d)
    return tensor


def save_hf_weights(
    model: nn.Module | list[nn.Module],
    hf_path: str,
    spec: HFWeights,
    ps,
    *,
    vocab_size: int | None = None,
) -> None:
    """Export + write to safetensors."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    out = dict(
        export_hf_weights(
            model, spec, ps, vocab_size=vocab_size, rank0_only=True, cpu=True
        )
    )
    if rank == 0 and out:
        save_safetensors(out, hf_path)
    if dist.is_initialized():
        dist.barrier()
