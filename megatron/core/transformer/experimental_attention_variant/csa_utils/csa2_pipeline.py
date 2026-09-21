# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 source and field declarations at existing Hybrid pipeline cuts.

The Hybrid host owns placement, payloads and transport. These declarations use
its global sublayer IDs, including FFN layers, to describe native state liveness.
"""

from dataclasses import dataclass, replace

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.hybrid_stack_adapter import (
    HybridPipelineBoundary,
    HybridPipelineChunk,
    build_hybrid_state_pipeline_plan,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    csa2_dependency_edges,
    csa2_field_factory,
    csa2_state_dependencies,
    csa2_state_key,
)
from megatron.core.transformer.state_boundary import (
    StateBoundary,
    StatePlacement,
    TensorField,
    prepare_boundaries,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass(frozen=True, kw_only=True)
class CSA2PipelineBoundary(HybridPipelineBoundary):
    """Live fields before ``layer_offset``; source IDs are zero-based Hybrid IDs.

    Shared K uses canonical [compressed_rows, B, dim], with B=1 for THD. Top-k uses [B, S, K]
    for SBHD and [T, K] for THD. Candidate block IDs/counts use the same leading
    dimensions as top-k, but retain sequence-local block coordinates. THD prefixes
    preserve logical lengths and physical starts separately. Derived flat K and
    fused attention indices never appear in the schema. With CP, S/T and selection
    rows are local, while compressed K rows and packed prefixes remain global.
    """

    last_attention_layer: int | None
    kv_source_layer: int | None
    index_source_layer: int | None
    candidate_source_layer: int | None
    needs_indexer_k: bool
    compress_ratio: int | None
    candidate_block_size: int
    placement: tuple[StatePlacement, ...] = ()

    def field_key(self, name: str) -> str:
        """Keep the original producer's version through every relay stage."""
        source = (
            self.kv_source_layer
            if name in ("global_kv", "indexer_k")
            else (
                self.index_source_layer
                if name == "global_indices"
                else self.candidate_source_layer if name.startswith("candidate_") else None
            )
        )
        return csa2_state_key(name, source)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Native field order, including packed prefixes at relay-only boundaries."""
        names = []
        if self.kv_source_layer is not None:
            names.append("global_kv")
        if self.needs_indexer_k:
            names.append("indexer_k")
        if self.index_source_layer is not None:
            names.append("global_indices")
        if self.candidate_source_layer is not None:
            names.extend(("candidate_indices", "candidate_lengths"))
        if self.qkv_format == "thd":
            names.extend(("cu_seqlens", "cu_seqlens_padded"))
        return tuple(names)

    def tensor_fields(
        self,
        config: MLATransformerConfig,
        seq_length: int,
        micro_batch_size: int,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        requires_grad: bool = True,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> tuple[TensorField, ...]:
        """Derive exact wire capacities from host metadata and the static boundary.

        Prefix contents remain on device. Their vector lengths/dtypes, the host
        maximum sequence length, and physical token capacity determine all shapes,
        including empty compressed buffers and sequence-local candidate blocks.
        """
        if (
            type(seq_length) is not int
            or seq_length < 0
            or type(micro_batch_size) is not int
            or micro_batch_size < 1
        ):
            raise ValueError("CSA2 pipeline requires valid host token/batch capacities")
        packed = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if packed and packed_seq_params.cp_group is not None:
            cp_group = packed_seq_params.cp_group
        if config.context_parallel_size > 1 and cp_group is None:
            raise ValueError("CSA2 CP pipeline planning requires an explicit CP group")
        cp_size = cp_group.size() if cp_group is not None else 1
        if cp_size > 1 and (not packed or packed_seq_params.cp_partition_mode != "contiguous"):
            raise ValueError("CSA2 CP pipeline requires contiguous THD metadata")
        if packed and packed_seq_params.local_cp_size not in (None, cp_size):
            raise ValueError("CSA2 pipeline local_cp_size must match its CP group")
        if packed != (self.qkv_format == "thd"):
            raise ValueError("CSA2 pipeline batch layout disagrees with its boundary")
        prefixes = {}
        max_seqlen = -1
        if packed:
            if micro_batch_size != 1:
                raise ValueError("CSA2 THD pipeline requires a flattened batch of size one")
            max_seqlen = packed_seq_params.max_seqlen_q
            if type(max_seqlen) is not int or max_seqlen < 0:
                raise ValueError("CSA2 pipeline max_seqlen must be prepared as a host integer")
            logical = packed_seq_params.cu_seqlens_q
            physical = packed_seq_params.cu_seqlens_q_padded
            physical = logical if physical is None else physical
            prefixes = {"cu_seqlens": logical, "cu_seqlens_padded": physical}
            for prefix in prefixes.values():
                if (
                    prefix is None
                    or prefix.ndim != 1
                    or prefix.numel() < 1
                    or prefix.dtype not in (torch.int32, torch.int64)
                ):
                    raise ValueError("CSA2 pipeline requires integer prefix vectors")
            if logical.shape != physical.shape:
                raise ValueError("CSA2 pipeline logical/physical prefix capacities must match")

        fields = {}
        factory = csa2_field_factory(
            config,
            seq_length,
            micro_batch_size,
            qkv_format=self.qkv_format,
            max_seqlen=max_seqlen if packed else None,
            num_sequences=logical.numel() - 1 if packed else None,
            cp_size=cp_size,
        )
        for name in self.field_names:
            if name in prefixes:
                continue
            source = int(self.field_key(name).rsplit(":L", 1)[1])
            field = factory(name, source)
            fields[name] = replace(field, differentiable=requires_grad and field.differentiable)
        # The generic dependency checker validates ordinary PP. For VPP, source
        # liveness is evaluated at each cut in the host's logical chunk order.
        ranks = [p.pp_rank for p in self.placement]
        if self.placement and len(set(ranks)) == len(ranks):
            dependencies = csa2_state_dependencies(config, factory, self.placement)
            (schema,) = prepare_boundaries(
                dependencies,
                (
                    StateBoundary(
                        f"csa2.decoder/pp:{self.layer_offset}",
                        "pipeline",
                        2 * self.layer_offset,
                        2 * self.layer_offset,
                    ),
                ),
                self.placement,
            )
            expected = {self.field_key(name) for name in self.field_names if name not in prefixes}
            if {f.key for f in schema.inputs} != expected:
                raise ValueError("CSA2 pipeline boundary disagrees with its state dependencies")
        for name, prefix in prefixes.items():
            fields[name] = TensorField(
                self.field_key(name), tuple(prefix.shape), prefix.dtype, self.qkv_format, False
            )
        return tuple(fields[name] for name in self.field_names)


def build_csa2_pipeline_plan(
    config: MLATransformerConfig, layer_pattern: str, *, pp_size: int = 1, qkv_format: str = "sbhd"
) -> tuple[HybridPipelineChunk, ...]:
    """Bind CSA2 source facts to the common host's existing chunk placement."""
    if config.experimental_attention_variant != "dsv4_hybrid" or config.dsv4_version != "v4.1":
        raise ValueError("CSA2 pipeline planning requires the DSv4 V4.1 configuration")
    chunks = build_hybrid_state_pipeline_plan(
        config, layer_pattern, pp_size=pp_size, qkv_format=qkv_format
    )
    pattern = layer_pattern.replace(Symbols.PIPE, "")
    if set(pattern) - {Symbols.DS_ATTENTION, Symbols.WINDOW, Symbols.MOE, Symbols.MLP}:
        raise ValueError("CSA2 pipeline requires D/W/E/- sublayers without MTP")
    if any(
        ratio and symbol != Symbols.DS_ATTENTION
        for symbol, ratio in zip(pattern, config.csa_compress_ratios)
    ):
        raise ValueError("CSA2 compressed layers must use the D Hybrid symbol")
    placement = tuple(
        StatePlacement(
            2 * chunk.layer_offset,
            2 * (chunk.layer_offset + len(chunk.layer_pattern)),
            chunk.pp_rank,
            chunk.vp_stage,
        )
        for chunk in chunks
    )
    boundaries = {
        chunk.outgoing.layer_offset: csa2_pipeline_boundary(
            config, pattern, chunk.outgoing.layer_offset, qkv_format, placement
        )
        for chunk in chunks
        if chunk.outgoing is not None
    }
    return tuple(
        replace(
            chunk,
            incoming=None if chunk.incoming is None else boundaries[chunk.incoming.layer_offset],
            outgoing=None if chunk.outgoing is None else boundaries[chunk.outgoing.layer_offset],
        )
        for chunk in chunks
    )


def csa2_pipeline_boundary(
    config: MLATransformerConfig,
    pattern: str,
    offset: int,
    qkv_format: str,
    placement: tuple[StatePlacement, ...] = (),
) -> CSA2PipelineBoundary:
    """Bind native source facts to an existing host cut; never choose placement."""
    sources = {}
    for name, dependencies in csa2_dependency_edges(config).items():
        live = {source for source, consumer in dependencies if source < offset <= consumer}
        if len(live) > 1:
            raise ValueError(f"CSA2 boundary has multiple live {name} sources")
        sources[name] = next(iter(live), None)
    kv = sources["global_kv"]
    return CSA2PipelineBoundary(
        layer_offset=offset,
        last_attention_layer=next(
            (
                i
                for i in range(offset - 1, -1, -1)
                if pattern[i] in (Symbols.DS_ATTENTION, Symbols.WINDOW)
            ),
            None,
        ),
        kv_source_layer=kv,
        index_source_layer=sources["global_indices"],
        candidate_source_layer=sources["candidates"],
        needs_indexer_k=sources["indexer_k"] is not None,
        compress_ratio=None if kv is None else config.csa_compress_ratios[kv],
        qkv_format=qkv_format,
        candidate_block_size=config.csa2_candidate_block_size,
        placement=placement,
    )
