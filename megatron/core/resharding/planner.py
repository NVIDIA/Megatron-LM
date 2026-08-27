# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .shard_planner import plan_sharded_transfer
from .utils import (
    ParameterMetadata,
    ReshardPlan,
    ShardingDescriptor,
    TensorReshardSpec,
    TransferOp,
    _build_layer_module_prefix_map,
    _get_rank_in_group,
    extract_param_metadata,
    named_refit_tensors,
    select_src_metadata_balanced,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _NativeParameterPart:
    global_shape: tuple[int, ...]
    src_local_shape: tuple[int, ...]
    dst_local_shape: tuple[int, ...]
    src_slice: tuple[slice, ...] | None
    dst_slice: tuple[slice, ...] | None


def _find_source_metadata(
    src_param_metadata: dict[str, list[ParameterMetadata]], resolved_name: str
) -> list[ParameterMetadata] | None:
    """Find source metadata, including the tied-output embedding alias."""
    src_meta_list = src_param_metadata.get(resolved_name)
    if not src_meta_list and resolved_name.endswith("output_layer.weight"):
        for embedding_name in ("embedding.word_embeddings.weight", "word_embeddings.weight"):
            src_meta_list = src_param_metadata.get(embedding_name)
            if src_meta_list:
                break
    return src_meta_list


def _sort_ops_by_dst_offset(ops, dim):
    """Sort transfer ops by destination offset on the sharded dimension."""
    ops.sort(key=lambda op: op[2][dim].start if isinstance(op[2][dim], slice) else 0)


def _build_descriptors_for_param(
    src_metadata: ParameterMetadata, dst_metadata: ParameterMetadata
) -> list[ShardingDescriptor]:
    """Construct sharding descriptors (currently TP) for this parameter based on actual layout.
    Guard TP descriptor with size conservation so we don't mis-classify replicated tensors.
    """
    descriptors: list[ShardingDescriptor] = []

    # TP descriptor: allow when either side participates in TP
    if src_metadata.is_tp or dst_metadata.is_tp:
        # Prefer destination partition_dim, else source
        tp_dim = dst_metadata.partition_dim if dst_metadata.is_tp else src_metadata.partition_dim
        src_tp_ranks = src_metadata.tensor_parallel_group_ranks
        dst_tp_ranks = dst_metadata.tensor_parallel_group_ranks
        if src_tp_ranks is None or dst_tp_ranks is None:
            # Not enough context to build TP descriptor
            return descriptors
        src_stride = src_metadata.partition_stride if src_metadata.is_tp else 1
        dst_stride = dst_metadata.partition_stride if dst_metadata.is_tp else 1

        # Size conservation check on partition dim
        src_world = len(src_tp_ranks)
        dst_world = len(dst_tp_ranks)
        src_local = src_metadata.shape[tp_dim]
        dst_local = dst_metadata.shape[tp_dim]
        if src_world * src_local != dst_world * dst_local:
            raise RuntimeError(
                f"Cannot build TP descriptor for {dst_metadata.name} dim{tp_dim}: "
                f"src_world*src_local={src_world}*{src_local} != {dst_world}*{dst_local}. "
                "This usually means the param is marked TP but is effectively replicated on that "
                "dim or partition_dim/metadata is inconsistent between source and destination."
            )

        descriptors.append(
            ShardingDescriptor(
                name="tp",
                dim=tp_dim,
                src_stride=src_stride,
                dst_stride=dst_stride,
                src_dim_ranks=src_tp_ranks,
                dst_dim_ranks=dst_tp_ranks,
            )
        )
    return descriptors


def _emit_lcm_block_ops(
    *,
    param_name: str,
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
    dim: int,
    src_world: int,
    dst_world: int,
    src_stride: int,
    dst_stride: int,
    full_block_len: int,
    dst_local_rank: int,
    src_dim_ranks: list[int],
    src_block_offset: int,
    dst_block_offset: int,
    block_label: str,
    ops: list,
) -> None:
    """Emit (src_rank, src_slice, dst_slice) ops for one LCM-tiled block.

    Used both by the single-block stride-aware TP planner and by the
    per-block loop of the block-interleaved planner.
    """
    Ns = src_world * max(1, src_stride)
    Nd = dst_world * max(1, dst_stride)
    L = math.lcm(Ns, Nd)
    if full_block_len % L != 0:
        raise RuntimeError(
            f"{param_name}: {block_label} length {full_block_len} not divisible by LCM {L} "
            f"(Ns={Ns}, Nd={Nd})"
        )
    unit = full_block_len // L
    cps = L // Ns
    cpd = L // Nd
    seg_src = cps * unit
    seg_dst = cpd * unit

    for k in range(max(1, dst_stride)):
        g_dst_seg = dst_local_rank + k * dst_world
        for off in range(cpd):
            g_micro = g_dst_seg * cpd + off
            s_idx = g_micro // cps
            in_seg = g_micro % cps
            src_global_rank = src_dim_ranks[s_idx % src_world]
            src_local_seg_idx = s_idx // src_world
            src_start = src_block_offset + src_local_seg_idx * seg_src + in_seg * unit
            dst_start = dst_block_offset + k * seg_dst + off * unit
            src_slice = [slice(None)] * len(src_shape)
            dst_slice = [slice(None)] * len(dst_shape)
            src_slice[dim] = slice(src_start, src_start + unit)
            dst_slice[dim] = slice(dst_start, dst_start + unit)
            ops.append((src_global_rank, tuple(src_slice), tuple(dst_slice)))


def _tp_block_layout(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    descriptor: ShardingDescriptor,
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
) -> list[tuple[int, int, int, int, int, str]]:
    """Compute the per-block layout for a TP transfer.

    Returns a list of ``(src_offset, dst_offset, full_block_len, src_stride,
    dst_stride, label)`` tuples that the LCM micro-tiler iterates.

    - Plain TP (no ``partition_sizes``): single block covering the full
      partition dim with the descriptor's strides.
    - Block-interleaved TP (``partition_sizes`` present, e.g. Mamba ``in_proj``):
      one block per packed component, each independently sharded with stride=1.
    """
    d = descriptor
    dim = d.dim
    src_world = len(d.src_dim_ranks)
    dst_world = len(d.dst_dim_ranks)
    src_sizes = src_metadata.partition_sizes
    dst_sizes = dst_metadata.partition_sizes

    if src_sizes is None and dst_sizes is None:
        src_local = src_shape[dim]
        dst_local = dst_shape[dim]
        if src_world * src_local != dst_world * dst_local:
            raise RuntimeError(
                f"{param_name}: size mismatch on TP dim{dim} "
                f"(src_world={src_world}, src_local={src_local}, "
                f"dst_world={dst_world}, dst_local={dst_local})"
            )
        return [(0, 0, dst_local * dst_world, d.src_stride, d.dst_stride, f"TP dim{dim}")]

    if src_sizes is not None:
        num_blocks = len(src_sizes)
        full_sizes = [s * src_world for s in src_sizes]
    else:
        num_blocks = len(dst_sizes)
        full_sizes = [s * dst_world for s in dst_sizes]
    if src_sizes is None:
        src_sizes = [f // src_world for f in full_sizes]
    if dst_sizes is None:
        dst_sizes = [f // dst_world for f in full_sizes]

    blocks: list[tuple[int, int, int, int, int, str]] = []
    src_off = 0
    dst_off = 0
    for i in range(num_blocks):
        if src_sizes[i] * src_world != dst_sizes[i] * dst_world:
            raise RuntimeError(
                f"{param_name}: block {i} size mismatch: "
                f"src_sizes[{i}]={src_sizes[i]}*{src_world} != "
                f"dst_sizes[{i}]={dst_sizes[i]}*{dst_world}"
            )
        blocks.append((src_off, dst_off, full_sizes[i], 1, 1, f"block {i}"))
        src_off += src_sizes[i]
        dst_off += dst_sizes[i]
    return blocks


def _plan_tp(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    descriptors: list[ShardingDescriptor],
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Plan TP transfers via LCM tiling, supporting both plain and block-interleaved TP.

    The block layout is derived once by ``_tp_block_layout`` — the inner
    LCM micro-tile math (``_emit_lcm_block_ops``) is identical for both cases,
    so the single-block plain-TP path is just a special case of the
    multi-block partitioned path.
    """
    if not descriptors:
        return []
    if len(descriptors) != 1 or descriptors[0].name != "tp":
        raise NotImplementedError(f"{param_name}: _plan_tp supports TP-only (one descriptor)")
    d = descriptors[0]
    if my_global_rank not in d.dst_dim_ranks:
        return []

    src_shape = tuple(src_metadata.shape)
    dst_shape = tuple(dst_metadata.shape)
    src_world = len(d.src_dim_ranks)
    dst_world = len(d.dst_dim_ranks)
    dst_local_rank = _get_rank_in_group(my_global_rank, d.dst_dim_ranks)

    blocks = _tp_block_layout(param_name, src_metadata, dst_metadata, d, src_shape, dst_shape)

    ops: list[tuple[int, tuple[slice, ...], tuple[slice, ...]]] = []
    for src_off, dst_off, full_len, src_stride, dst_stride, label in blocks:
        _emit_lcm_block_ops(
            param_name=param_name,
            src_shape=src_shape,
            dst_shape=dst_shape,
            dim=d.dim,
            src_world=src_world,
            dst_world=dst_world,
            src_stride=src_stride,
            dst_stride=dst_stride,
            full_block_len=full_len,
            dst_local_rank=dst_local_rank,
            src_dim_ranks=d.src_dim_ranks,
            src_block_offset=src_off,
            dst_block_offset=dst_off,
            block_label=label,
            ops=ops,
        )
    _sort_ops_by_dst_offset(ops, d.dim)
    return ops


def _finalize_dp_transfers(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Return receiver-side transfer for a parameter that is not TP-sharded.

    This is reached when we cannot build a TP sharding descriptor for the parameter
    (i.e., it is effectively replicated with respect to sharding).  We use this when the
    destination and source mode have no TP or the parameter is replicted on all ranks
    such as layernorm. If the source and destination DP groups match, we return a local
    full-tensor copy; otherwise we pick a source rank from the source DP group in a
    deterministic round-robin manner based on the receiver's global rank for better load
    distribution.
    """
    dst_dp_ranks = dst_metadata.data_parallel_group_ranks
    src_dp_ranks = src_metadata.data_parallel_group_ranks
    if my_global_rank not in dst_dp_ranks:
        return []

    dst_shape = dst_metadata.shape

    # Same DP layout - local copy (only if this rank has the source parameter)
    if src_dp_ranks == dst_dp_ranks and my_global_rank in src_dp_ranks:
        full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
        return [(my_global_rank, full_slice, full_slice)]

    # Use the owner of the metadata picked by select_src_metadata_balanced.
    # That selection already handles DP round-robin and non-collocated cases
    # (where some src DP ranks don't actually own the source model).
    full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
    return [(src_metadata.owner_rank, full_slice, full_slice)]


def _determine_source_ranks_for_dst_param(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Route to dimension-specific planner based on parameter sharding type."""

    # Regular TP/DP planning with EP-resolved metadata.  _plan_tp handles both
    # plain TP and block-interleaved TP (partition_sizes-driven) layouts.
    descriptors = _build_descriptors_for_param(src_metadata=src_metadata, dst_metadata=dst_metadata)
    if descriptors:
        return _plan_tp(
            param_name=param_name,
            src_metadata=src_metadata,
            dst_metadata=dst_metadata,
            descriptors=descriptors,
            my_global_rank=my_global_rank,
        )
    # DP / replicated fallback
    return _finalize_dp_transfers(param_name, src_metadata, dst_metadata, my_global_rank)


def _iter_global_transfer_ops(
    dst_param_metadata_by_rank: dict[int, dict[str, ParameterMetadata]],
    src_param_metadata: dict[str, list[ParameterMetadata]],
):
    """Yield the whole reshard schedule in a deterministic order.

    The iteration order (dst rank ascending, then that rank's dst params in
    gathered order, then per-source sub-ops) depends only on the rosters, so
    replaying this on any rank produces the same sequence and assigns the same
    task_id to the same transfer. That's what lets each rank build its own
    send/recv ops while sender and receiver still agree on task_id.

    Ranks are taken from the roster keys rather than range(world_size), so a
    sparse or growing rank set (nodes added later) rebuilds identically.

    Yields (task_id, dst_rank, src_rank, src_slice, dst_slice, src_metadata,
    dst_metadata). PP is handled implicitly: each rank contributes metadata only
    for the params it owns, and any source holding the same resolved_name can
    serve as sender (with DP balancing).
    """
    # Shared between a send and its recv; NVSHMEM builds a schedule from it and
    # local copies match on it.
    next_task_id = 0
    for dst_rank in sorted(dst_param_metadata_by_rank):
        dst_rank_params = dst_param_metadata_by_rank[dst_rank]
        for resolved_name, dst_metadata in dst_rank_params.items():
            src_meta_list = _find_source_metadata(src_param_metadata, resolved_name)
            if not src_meta_list:
                raise RuntimeError(
                    f"Destination parameter '{resolved_name}' on rank {dst_rank} "
                    "not found in source model."
                )
            # Choose a representative source metadata with DP round-robin balancing.
            src_metadata = select_src_metadata_balanced(src_meta_list, dst_metadata, dst_rank)
            if dst_metadata.is_gtp or any(metadata.is_gtp for metadata in src_meta_list):
                # A GTP shard is an additional dim-0 partition layered on top
                # of the TP layout. Plan in logical global coordinates so TP,
                # packed parameters, GTP padding, and their combinations compose.
                sources = plan_sharded_transfer(
                    resolved_name, src_meta_list, src_metadata, dst_metadata
                )
            else:
                # Preserve the established TP/DP lowering for non-GTP models.
                sources = _determine_source_ranks_for_dst_param(
                    resolved_name, src_metadata, dst_metadata, dst_rank
                )
            for src_rank, src_slice, dst_slice in sources:
                task_id = next_task_id
                next_task_id += 1
                yield task_id, dst_rank, src_rank, src_slice, dst_slice, src_metadata, dst_metadata


def _tensor_mesh(metadata: ParameterMetadata) -> tuple[int, ...]:
    """Return the TP mesh that owns a local parameter shard or replica."""
    ranks = metadata.tensor_parallel_group_ranks
    return tuple(ranks) if ranks else (metadata.owner_rank,)


def _global_shape_for_mesh(metadata: ParameterMetadata, mesh: tuple[int, ...]) -> tuple[int, ...]:
    shape = list(metadata.shape)
    if metadata.is_tp:
        shape[metadata.partition_dim] *= len(mesh)
    return tuple(shape)


def _validate_native_mesh_metadata(
    resolved_name: str,
    side: str,
    mesh: tuple[int, ...],
    metadata_by_rank: dict[int, ParameterMetadata],
) -> str | None:
    """Return why a parameter mesh cannot use whole-tensor native resharding."""
    if not mesh:
        return f"{resolved_name}: {side} TP mesh is empty"
    if set(metadata_by_rank) != set(mesh):
        missing = sorted(set(mesh) - set(metadata_by_rank))
        extra = sorted(set(metadata_by_rank) - set(mesh))
        return (
            f"{resolved_name}: {side} TP metadata does not cover its mesh "
            f"(missing={missing}, extra={extra})"
        )

    first = metadata_by_rank[mesh[0]]
    for rank in mesh:
        metadata = metadata_by_rank[rank]
        if tuple(metadata.shape) != tuple(first.shape):
            return f"{resolved_name}: {side} TP ranks have different local shapes"
        if metadata.dtype != first.dtype:
            return f"{resolved_name}: {side} TP ranks have different dtypes"
        if metadata.is_tp != first.is_tp:
            return f"{resolved_name}: {side} TP ranks disagree on sharding"
        if metadata.is_tp and metadata.partition_dim != first.partition_dim:
            return f"{resolved_name}: {side} TP ranks disagree on the shard dimension"
        if metadata.partition_sizes != first.partition_sizes:
            return f"{resolved_name}: {side} TP ranks disagree on packed component sizes"
        if metadata.partition_stride != 1:
            return f"{resolved_name}: partition_stride={metadata.partition_stride} is unsupported"
    return None


def _native_parameter_parts(
    resolved_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    src_mesh: tuple[int, ...],
    dst_mesh: tuple[int, ...],
) -> tuple[list[_NativeParameterPart] | None, str | None]:
    """Split a packed TP parameter into regular tensors for native resharding.

    Mamba-style projections concatenate several components that are each
    independently TP-sharded. The concatenation itself is not a regular
    sharded tensor, but each component is. Plain parameters remain a single
    full-tensor part.
    """
    src_global_shape = _global_shape_for_mesh(src_metadata, src_mesh)
    dst_global_shape = _global_shape_for_mesh(dst_metadata, dst_mesh)
    src_sizes = src_metadata.partition_sizes
    dst_sizes = dst_metadata.partition_sizes
    if src_sizes is None and dst_sizes is None:
        if src_global_shape != dst_global_shape:
            return None, (
                f"{resolved_name}: source global shape {src_global_shape} does not match "
                f"destination global shape {dst_global_shape}"
            )
        return [
            _NativeParameterPart(
                global_shape=src_global_shape,
                src_local_shape=tuple(src_metadata.shape),
                dst_local_shape=tuple(dst_metadata.shape),
                src_slice=None,
                dst_slice=None,
            )
        ], None

    partition_dim = (
        src_metadata.partition_dim
        if src_sizes is not None or src_metadata.is_tp
        else dst_metadata.partition_dim
    )
    if (src_sizes is not None or src_metadata.is_tp) and (
        dst_sizes is not None or dst_metadata.is_tp
    ):
        if src_metadata.partition_dim != dst_metadata.partition_dim:
            return None, (
                f"{resolved_name}: packed source and destination use different partition "
                "dimensions"
            )

    src_shards = len(src_mesh) if src_metadata.is_tp else 1
    dst_shards = len(dst_mesh) if dst_metadata.is_tp else 1
    if src_sizes is not None:
        full_sizes = [size * src_shards for size in src_sizes]
    else:
        full_sizes = [size * dst_shards for size in dst_sizes]
    if src_sizes is None:
        if any(size % src_shards for size in full_sizes):
            return None, f"{resolved_name}: packed source component is not TP-divisible"
        src_sizes = [size // src_shards for size in full_sizes]
    if dst_sizes is None:
        if any(size % dst_shards for size in full_sizes):
            return None, f"{resolved_name}: packed destination component is not TP-divisible"
        dst_sizes = [size // dst_shards for size in full_sizes]
    if len(src_sizes) != len(dst_sizes):
        return None, f"{resolved_name}: source and destination packed component counts differ"
    for index, (src_size, dst_size, full_size) in enumerate(zip(src_sizes, dst_sizes, full_sizes)):
        if src_size * src_shards != full_size or dst_size * dst_shards != full_size:
            return None, f"{resolved_name}: packed component {index} has inconsistent TP sizes"
    if sum(src_sizes) != src_metadata.shape[partition_dim]:
        return None, f"{resolved_name}: source packed component sizes do not match its shape"
    if sum(dst_sizes) != dst_metadata.shape[partition_dim]:
        return None, f"{resolved_name}: destination packed component sizes do not match its shape"

    src_other_shape = list(src_global_shape)
    dst_other_shape = list(dst_global_shape)
    src_other_shape[partition_dim] = dst_other_shape[partition_dim] = 0
    if src_other_shape != dst_other_shape or sum(full_sizes) != src_global_shape[partition_dim]:
        return None, f"{resolved_name}: source and destination packed shapes do not match"
    if sum(full_sizes) != dst_global_shape[partition_dim]:
        return None, f"{resolved_name}: source and destination packed shapes do not match"

    parts = []
    src_offset = 0
    dst_offset = 0
    for src_size, dst_size, full_size in zip(src_sizes, dst_sizes, full_sizes):
        global_shape = list(src_global_shape)
        src_local_shape = list(src_metadata.shape)
        dst_local_shape = list(dst_metadata.shape)
        global_shape[partition_dim] = full_size
        src_local_shape[partition_dim] = src_size
        dst_local_shape[partition_dim] = dst_size
        src_slice = [slice(None)] * len(src_metadata.shape)
        dst_slice = [slice(None)] * len(dst_metadata.shape)
        src_slice[partition_dim] = slice(src_offset, src_offset + src_size)
        dst_slice[partition_dim] = slice(dst_offset, dst_offset + dst_size)
        parts.append(
            _NativeParameterPart(
                global_shape=tuple(global_shape),
                src_local_shape=tuple(src_local_shape),
                dst_local_shape=tuple(dst_local_shape),
                src_slice=tuple(src_slice),
                dst_slice=tuple(dst_slice),
            )
        )
        src_offset += src_size
        dst_offset += dst_size
    return parts, None


def _build_tensor_reshard_specs(
    dst_param_metadata_by_rank: dict[int, dict[str, ParameterMetadata]],
    src_param_metadata: dict[str, list[ParameterMetadata]],
    my_global_rank: int,
) -> tuple[list[TensorReshardSpec] | None, str | None]:
    """Build native mesh specs without affecting the generic plan."""
    dst_groups: dict[tuple[str, tuple[int, ...]], dict[int, ParameterMetadata]] = {}
    for dst_rank in sorted(dst_param_metadata_by_rank):
        for resolved_name, metadata in dst_param_metadata_by_rank[dst_rank].items():
            mesh = _tensor_mesh(metadata)
            dst_groups.setdefault((resolved_name, mesh), {})[metadata.owner_rank] = metadata

    specs: list[TensorReshardSpec] = []
    for (resolved_name, dst_mesh), dst_by_rank in dst_groups.items():
        src_meta_list = _find_source_metadata(src_param_metadata, resolved_name)
        if not src_meta_list:
            # The generic schedule raises the authoritative missing-weight
            # error. Keep this helper side-effect free for callers that invoke
            # it independently in tests.
            return None, f"{resolved_name}: source parameter metadata is missing"
        if any(metadata.is_gtp for metadata in src_meta_list) or any(
            metadata.is_gtp for metadata in dst_by_rank.values()
        ):
            return None, f"{resolved_name}: GTP shards are unsupported by native resharding"

        src_groups: dict[tuple[int, ...], dict[int, ParameterMetadata]] = {}
        for metadata in src_meta_list:
            mesh = _tensor_mesh(metadata)
            src_groups.setdefault(mesh, {})[metadata.owner_rank] = metadata
        dst_metadata = dst_by_rank[dst_mesh[0]]
        selected_src = select_src_metadata_balanced(src_meta_list, dst_metadata, dst_mesh[0])
        src_mesh = _tensor_mesh(selected_src)
        src_by_rank = src_groups[src_mesh]

        error = _validate_native_mesh_metadata(
            resolved_name, "source", src_mesh, src_by_rank
        ) or _validate_native_mesh_metadata(resolved_name, "destination", dst_mesh, dst_by_rank)
        if error is not None:
            return None, error
        if set(src_mesh) & set(dst_mesh):
            return None, f"{resolved_name}: source and destination meshes overlap"

        src_metadata = src_by_rank[src_mesh[0]]
        if src_metadata.dtype != dst_metadata.dtype:
            return None, (
                f"{resolved_name}: source dtype {src_metadata.dtype} does not match "
                f"destination dtype {dst_metadata.dtype}"
            )
        if (src_metadata.is_ep and src_metadata.global_expert_index is None) or (
            dst_metadata.is_ep and dst_metadata.global_expert_index is None
        ):
            return None, (
                f"{resolved_name}: fused expert tensors are not supported by native resharding"
            )

        parts, error = _native_parameter_parts(
            resolved_name, src_metadata, dst_metadata, src_mesh, dst_mesh
        )
        if error is not None:
            return None, error
        assert parts is not None

        local_src = src_by_rank.get(my_global_rank)
        local_dst = dst_by_rank.get(my_global_rank)
        for part_index, part in enumerate(parts):
            specs.append(
                TensorReshardSpec(
                    resolved_name=resolved_name,
                    src_ranks=src_mesh,
                    dst_ranks=dst_mesh,
                    global_shape=part.global_shape,
                    src_local_shape=part.src_local_shape,
                    dst_local_shape=part.dst_local_shape,
                    dtype=src_metadata.dtype,
                    src_shard_dim=src_metadata.partition_dim if src_metadata.is_tp else None,
                    dst_shard_dim=dst_metadata.partition_dim if dst_metadata.is_tp else None,
                    src_param_name=local_src.name if local_src is not None else None,
                    dst_param_name=local_dst.name if local_dst is not None else None,
                    src_param_shape=tuple(src_metadata.shape),
                    dst_param_shape=tuple(dst_metadata.shape),
                    src_slice=part.src_slice,
                    dst_slice=part.dst_slice,
                    part_index=part_index,
                    part_count=len(parts),
                )
            )

    specs.sort(
        key=lambda spec: (spec.src_ranks, spec.dst_ranks, spec.resolved_name, spec.part_index)
    )
    if not specs:
        return None, "the reshard plan contains no parameters"
    return specs, None


def _extract_module_metadata(
    module, owner_rank, num_experts, rank_offset, rank_list_cache
) -> list[ParameterMetadata]:
    """Metadata for a module's params and persistent buffers, or [] if None.

    Persistent buffers travel too so training state (e.g. MoE router expert_bias)
    refits with the weights.
    """
    if module is None:
        return []
    pg = getattr(module, "pg_collection", None)
    if pg is None:
        raise ValueError("Module must have pg_collection")
    layer_prefix_map = _build_layer_module_prefix_map(module)
    return [
        extract_param_metadata(
            p,
            name,
            owner_rank,
            pg,
            num_experts=num_experts,
            layer_module_prefix_map=layer_prefix_map,
            rank_offset=rank_offset,
            _rank_list_cache=rank_list_cache,
        )
        for name, p in named_refit_tensors(module)
    ]


def index_metadata_rosters(gathered_pairs: list):
    """Turn a rank-ordered list of ``(src_meta, dst_meta)`` (index == rank) into the
    two rosters the plan builder consumes: dst params keyed by rank, and src params
    keyed by resolved_name. The list may come from the all-gather, or be reassembled
    in rank order as nodes are added, before calling build_plan_from_rosters.
    """
    dst_param_metadata_by_rank: dict[int, dict[str, ParameterMetadata]] = {}
    src_param_metadata: dict[str, list[ParameterMetadata]] = {}
    for rank_id, (src_meta_list, dst_meta_list) in enumerate(gathered_pairs):
        dst_param_metadata_by_rank[rank_id] = {m.resolved_name: m for m in dst_meta_list}
        for metadata in src_meta_list:
            src_param_metadata.setdefault(metadata.resolved_name, []).append(metadata)
    return dst_param_metadata_by_rank, src_param_metadata


def build_plan_from_rosters(
    dst_param_metadata_by_rank: dict[int, dict[str, ParameterMetadata]],
    src_param_metadata: dict[str, list[ParameterMetadata]],
    my_global_rank: int,
) -> ReshardPlan:
    """Replay the deterministic global schedule and keep only this rank's ops.

    Pure and collective-free, so it can be tested or reused with preassembled
    rosters without touching the process group. Live membership orchestration
    is intentionally outside this module.
    """
    my_plan = ReshardPlan([], [])
    for (
        task_id,
        dst_rank,
        src_rank,
        src_slice,
        dst_slice,
        src_metadata,
        dst_metadata,
    ) in _iter_global_transfer_ops(dst_param_metadata_by_rank, src_param_metadata):
        if dst_rank == my_global_rank:
            my_plan.recv_ops.append(
                TransferOp(
                    param_name=dst_metadata.name,
                    peer_rank=src_rank,
                    is_send=False,
                    my_slice=dst_slice,
                    peer_slice=src_slice,
                    task_id=task_id,
                )
            )
        if src_rank == my_global_rank:
            my_plan.send_ops.append(
                TransferOp(
                    param_name=src_metadata.name,
                    peer_rank=dst_rank,
                    is_send=True,
                    my_slice=src_slice,
                    peer_slice=dst_slice,
                    task_id=task_id,
                )
            )

    logger.info(
        f"Rank {my_global_rank}: Built plan locally - {len(my_plan.recv_ops)} recvs, "
        f"{len(my_plan.send_ops)} sends"
    )
    my_plan.tensor_reshard_specs, my_plan.tensor_reshard_error = _build_tensor_reshard_specs(
        dst_param_metadata_by_rank, src_param_metadata, my_global_rank
    )
    return my_plan


def build_local_reshard_plan(
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    num_experts: int | None = None,
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
) -> ReshardPlan:
    """
    Build this rank's reshard plan locally: all-gather the parameter metadata,
    replay the global schedule (see _iter_global_transfer_ops), and keep only the
    ops where this rank is the sender or receiver. No rank-0 bottleneck and no
    scatter, since sender and receiver derive matching task_ids from the same
    metadata.

    The metadata gather (the one collective) and the plan build are split into
    index_metadata_rosters + build_plan_from_rosters, so the deterministic build
    can also be tested against preassembled rosters without a process group.

    src_module/dst_module may be None for non-collocated ranks (destination-only,
    source-only, or idle). Each rank contributes metadata only for the models it
    owns, including its parallel-group membership.
    """
    # group.rank()/size() (not dist.get_rank(group)) support cross-cluster PGs
    # whose members have independent default PGs.
    my_global_rank = group.rank() if group is not None else dist.get_rank()
    world_size = group.size() if group is not None else dist.get_world_size()

    # Dedup rank lists so params sharing a group reuse one list object; shrinks
    # the pickled all-gather ~75%.
    rank_list_cache: dict = {}
    my_src_metadata = _extract_module_metadata(
        src_module, my_global_rank, num_experts, src_rank_offset, rank_list_cache
    )
    my_dst_metadata = _extract_module_metadata(
        dst_module, my_global_rank, num_experts, dst_rank_offset, rank_list_cache
    )

    # One all-gather gives every rank the full (src, dst) picture, replacing the
    # gather-to-0 + scatter.
    gathered_pairs = [None] * world_size
    dist.all_gather_object(gathered_pairs, (my_src_metadata, my_dst_metadata), group=group)
    del my_src_metadata, my_dst_metadata

    dst_param_metadata_by_rank, src_param_metadata = index_metadata_rosters(gathered_pairs)
    del gathered_pairs
    return build_plan_from_rosters(dst_param_metadata_by_rank, src_param_metadata, my_global_rank)


def build_centralized_reshard_plan(
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    num_experts: int | None = None,
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
) -> ReshardPlan:
    """Deprecated compatibility wrapper for :func:`build_local_reshard_plan`."""
    warnings.warn(
        "build_centralized_reshard_plan is deprecated; use build_local_reshard_plan instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_local_reshard_plan(
        src_module,
        dst_module,
        num_experts=num_experts,
        group=group,
        src_rank_offset=src_rank_offset,
        dst_rank_offset=dst_rank_offset,
    )
