# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 logical chunk boundaries and forward-local payloads.

The plan uses global Hybrid sublayer IDs, including FFN layers. It describes activation
dependencies, independently of the process groups or schedule used to execute the chunks.
Payload tensors are immutable by convention and retain their autograd edges. Transport,
joint backward, and send-buffer lifetime management belong to the pipeline schedule.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayload,
    PipelinePayloadSpec,
    PipelineTensorSpec,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
    get_thd_compressed_capacity,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCState
from megatron.core.transformer.transformer_config import MLATransformerConfig

if TYPE_CHECKING:
    from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State


# Preserve the PP-1 descriptor name while sharing the transport's generic type.
CSA2PipelineTensorSpec = PipelineTensorSpec


@dataclass(frozen=True)
class CSA2PipelineBoundary:
    """Live fields before ``layer_offset``; source IDs are zero-based Hybrid IDs.

    Hidden states use [S, B, streams*C], and pre_mix uses [S, B, streams]. Shared K
    uses canonical [compressed_rows, B, dim], with B=1 for THD. Top-k uses [B, S, K]
    for SBHD and [T, K] for THD. Candidate block IDs/counts use the same leading
    dimensions as top-k, but retain sequence-local block coordinates. THD prefixes
    preserve logical lengths and physical starts separately. Derived flat K and
    fused attention indices never appear in the schema.
    """

    layer_offset: int
    last_attention_layer: int | None
    kv_source_layer: int | None
    index_source_layer: int | None
    candidate_source_layer: int | None
    needs_indexer_k: bool
    compress_ratio: int | None
    qkv_format: str
    hidden_size: int
    activation_dtype: torch.dtype
    num_streams: int
    single_pass_mhc: bool
    kv_dim: int
    indexer_dim: int
    indexer_k_requires_grad: bool
    candidate_block_size: int

    @property
    def field_names(self) -> tuple[str, ...]:
        """Static wire order, including metadata even at an mHC-only THD boundary."""
        names = ["hidden_states"]
        if self.single_pass_mhc:
            names.append("pre_mix")
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

    def payload_spec(
        self,
        config: MLATransformerConfig,
        seq_length: int,
        micro_batch_size: int,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        requires_grad: bool = True,
    ) -> PipelinePayloadSpec:
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

        def compressed_capacity(ratio):
            if packed:
                return get_thd_compressed_capacity(
                    seq_length, max_seqlen, logical.numel() - 1, ratio
                )
            return seq_length // ratio

        leading = (seq_length,) if packed else (micro_batch_size, seq_length)
        hidden_dtype = (
            torch.float32
            if config.fp32_residual_connection and not config.enable_hyper_connections
            else self.activation_dtype
        )
        fields = {
            "hidden_states": (
                (seq_length, micro_batch_size, self.num_streams * self.hidden_size),
                hidden_dtype,
                requires_grad,
            ),
            "pre_mix": (
                (seq_length, micro_batch_size, self.num_streams),
                self.activation_dtype if config.use_fused_mhc else torch.float32,
                requires_grad,
            ),
        }
        if self.kv_source_layer is not None:
            capacity = compressed_capacity(self.compress_ratio)
            fields["global_kv"] = (
                (capacity, micro_batch_size, self.kv_dim),
                self.activation_dtype,
                requires_grad,
            )
            fields["indexer_k"] = (
                (capacity, micro_batch_size, self.indexer_dim),
                self.activation_dtype,
                requires_grad and self.indexer_k_requires_grad,
            )
        if self.index_source_layer is not None:
            capacity = compressed_capacity(config.csa_compress_ratios[self.index_source_layer])
            fields["global_indices"] = (
                (*leading, min(config.dsa_indexer_topk, capacity)),
                torch.int32,
                False,
            )
        if self.candidate_source_layer is not None:
            ratio = config.csa_compress_ratios[self.candidate_source_layer]
            # THD candidates use local block IDs, whereas token top-k addresses
            # the full physical compressed capacity across all packed sequences.
            keys = max_seqlen // ratio if packed else seq_length // ratio
            blocks = (keys + self.candidate_block_size - 1) // self.candidate_block_size
            width = min(config.csa2_candidate_topk_blocks, blocks)
            fields["candidate_indices"] = ((*leading, width), torch.int32, False)
            fields["candidate_lengths"] = (leading, torch.int32, False)
        for name, prefix in prefixes.items():
            fields[name] = (tuple(prefix.shape), prefix.dtype, False)
        return PipelinePayloadSpec(
            tuple(PipelineTensorSpec(name, *fields[name]) for name in self.field_names),
            (self.layer_offset, max_seqlen),
        )

    def export_payload(
        self,
        hidden_states: torch.Tensor,
        csa2_state: "CSA2State",
        mhc_state: SinglePassMHCState | None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> "CSA2PipelinePayload":
        """Snapshot live tensor references without copying or detaching floating tensors."""
        if csa2_state.last_layer != self.last_attention_layer:
            raise ValueError("CSA2 pipeline state has not reached this chunk boundary")
        values = {"hidden_states": hidden_states}
        if self.single_pass_mhc:
            values["pre_mix"] = None if mhc_state is None else mhc_state.pre_mix
        for name, source in (
            ("kv_source_layer", self.kv_source_layer),
            ("index_source_layer", self.index_source_layer),
            ("candidate_source_layer", self.candidate_source_layer),
        ):
            if source is not None and getattr(csa2_state, name) != source:
                raise ValueError(
                    f"CSA2 pipeline boundary {self.layer_offset} requires {name}={source}"
                )
        for name in ("global_kv", "indexer_k", "global_indices"):
            values[name] = getattr(csa2_state, name)
        if self.candidate_source_layer is not None:
            candidates = csa2_state.candidates
            if candidates is None or candidates.block_size != self.candidate_block_size:
                raise ValueError("CSA2 pipeline candidate block size does not match the plan")
            values["candidate_indices"] = candidates.indices
            values["candidate_lengths"] = candidates.lengths
        max_seqlen = None
        if self.qkv_format == "thd":
            layout = csa2_state.thd_layout
            if layout is None:
                if packed_seq_params is None:
                    raise ValueError("CSA2 THD pipeline payload requires packed sequence metadata")
                layout = build_csa2_thd_layout(packed_seq_params, hidden_states.shape[0])
            elif packed_seq_params is not None:
                layout.validate_compatible(packed_seq_params, hidden_states.shape[0])
            # These prefixes are already independent snapshots of the batch input buffers.
            values["cu_seqlens"] = layout.cu_seqlens
            values["cu_seqlens_padded"] = layout.cu_seqlens_padded
            max_seqlen = layout.max_seqlen
        elif csa2_state.thd_layout is not None or (
            packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        ):
            raise ValueError("CSA2 pipeline plan expects SBHD, not THD")
        payload = CSA2PipelinePayload(
            self, tuple(values.get(n) for n in self.field_names), max_seqlen
        )
        payload.validate()
        return payload


@dataclass(frozen=True)
class CSA2PipelineChunk:
    """One nonempty logical segment, ordered by vp_stage * pp_size + pp_rank."""

    layer_offset: int
    layer_pattern: str
    pp_rank: int
    vp_stage: int
    incoming: CSA2PipelineBoundary | None
    outgoing: CSA2PipelineBoundary | None


def build_csa2_pipeline_plan(
    config: MLATransformerConfig, layer_pattern: str, *, pp_size: int = 1, qkv_format: str = "sbhd"
) -> tuple[CSA2PipelineChunk, ...]:
    """Plan explicit ``|`` segments without constructing distributed process groups.

    ``pp_size`` describes the intended placement, independently of config's current PP
    guard. Every dependency crossing a boundary stays live, including through chunks
    with no local consumer. Empty segments and MTP are outside this initial interface.
    """
    if config.experimental_attention_variant != "dsv4_hybrid" or config.dsv4_version != "v4.1":
        raise ValueError("CSA2 pipeline planning requires the DSv4 V4.1 configuration")
    if config.enable_hyper_connections and not config.mhc_single_pass:
        raise ValueError("CSA2 pipeline planning requires single-pass mHC when mHC is enabled")
    if qkv_format not in ("sbhd", "thd"):
        raise ValueError("CSA2 pipeline qkv_format must be sbhd or thd")
    segments = layer_pattern.split(Symbols.PIPE)
    pattern = "".join(segments)
    if not all(segments) or set(pattern) - {
        Symbols.DS_ATTENTION,
        Symbols.WINDOW,
        Symbols.MOE,
        Symbols.MLP,
    }:
        raise ValueError("CSA2 pipeline requires nonempty D/W/E/- segments without MTP")
    if len(pattern) != config.num_layers:
        raise ValueError("CSA2 pipeline pattern must contain exactly num_layers sublayers")
    if type(pp_size) is not int or pp_size < 1 or len(segments) % pp_size:
        raise ValueError("CSA2 pipeline segment count must be divisible by positive pp_size")
    ratios = config.csa_compress_ratios
    if any(ratio and symbol != Symbols.DS_ATTENTION for symbol, ratio in zip(pattern, ratios)):
        raise ValueError("CSA2 compressed layers must use the D Hybrid symbol")

    # Edges are (producer, consumer). Full replaces all CSA2 state; Reindex replaces
    # only top-k. Non-attention sublayers neither consume nor reset CSA2 state.
    edges = {name: [] for name in ("kv", "indexer_k", "indices", "candidates")}
    for layer, ratio in enumerate(ratios):
        if not ratio:
            continue
        owner = max(source for source in config.csa2_kv_source_layers if source <= layer)
        index = max(source for source in config.csa2_index_source_layers if source <= layer)
        if layer != owner:
            edges["kv"].append((owner, layer))
        if layer in config.csa2_index_source_layers:
            if layer != owner:
                edges["indexer_k"].append((owner, layer))
            candidate = config.csa2_candidate_source_layer
            if candidate is not None and layer > candidate:
                edges["candidates"].append((candidate, layer))
        else:
            edges["indices"].append((index, layer))

    boundaries = [None]
    offset = 0
    for segment in segments[:-1]:
        offset += len(segment)
        sources = {}
        for name, dependencies in edges.items():
            live = {source for source, consumer in dependencies if source < offset <= consumer}
            if len(live) > 1:
                raise ValueError(f"CSA2 boundary has multiple live {name} sources")
            sources[name] = next(iter(live), None)
        boundaries.append(
            CSA2PipelineBoundary(
                layer_offset=offset,
                last_attention_layer=next(
                    (
                        i
                        for i in range(offset - 1, -1, -1)
                        if pattern[i] in (Symbols.DS_ATTENTION, Symbols.WINDOW)
                    ),
                    None,
                ),
                kv_source_layer=sources["kv"],
                index_source_layer=sources["indices"],
                candidate_source_layer=sources["candidates"],
                needs_indexer_k=sources["indexer_k"] is not None,
                compress_ratio=None if sources["kv"] is None else ratios[sources["kv"]],
                qkv_format=qkv_format,
                hidden_size=config.hidden_size,
                activation_dtype=config.params_dtype,
                num_streams=config.num_residual_streams if config.enable_hyper_connections else 1,
                single_pass_mhc=config.mhc_single_pass,
                kv_dim=config.v_head_dim,
                indexer_dim=config.dsa_indexer_head_dim,
                indexer_k_requires_grad=(config.dsa_indexer_loss_coeff or 0) > 0,
                candidate_block_size=config.csa2_candidate_block_size,
            )
        )
    boundaries.append(None)
    chunks = []
    offset = 0
    for i, segment in enumerate(segments):
        chunks.append(
            CSA2PipelineChunk(
                offset, segment, i % pp_size, i // pp_size, boundaries[i], boundaries[i + 1]
            )
        )
        offset += len(segment)
    return tuple(chunks)


@dataclass(frozen=True)
class CSA2PipelinePayload(PipelinePayload):
    """Stable references to one chunk's live outputs, never a mutable CSA2State.

    ``max_seqlen`` is host metadata, not a device reduction. Prefix values and tensor
    shapes vary by microbatch. The transport sends this metadata together with the
    typed tensor list.
    """

    boundary: CSA2PipelineBoundary
    tensors: tuple[torch.Tensor, ...]
    max_seqlen: int | None = None

    @property
    def metadata(self) -> tuple[int, int]:
        """Identify the cut and packed layout; all other boundary fields are static."""
        return self.boundary.layer_offset, -1 if self.max_seqlen is None else self.max_seqlen

    def validate(self) -> None:
        """Check field count, shapes, types and device without device-to-host reads."""
        names = self.boundary.field_names
        if len(self.tensors) != len(names) or any(
            not isinstance(t, torch.Tensor) for t in self.tensors
        ):
            raise ValueError("CSA2 pipeline payload is missing required tensor fields")
        values = dict(zip(names, self.tensors))
        hidden = values["hidden_states"]
        b = self.boundary
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != b.hidden_size * b.num_streams
            or not hidden.is_floating_point()
        ):
            raise ValueError("CSA2 pipeline hidden_states must have shape [S, B, streams*C]")
        s, batch = hidden.shape[:2]
        if b.qkv_format == "thd" and batch != 1:
            raise ValueError("CSA2 THD pipeline hidden_states must have batch size 1")
        for name, tensor in values.items():
            if tensor.device != hidden.device:
                raise ValueError(f"CSA2 pipeline {name} must be on the hidden_states device")
            if name == "pre_mix":
                if tensor.shape != (s, batch, b.num_streams) or tensor.dtype not in (
                    b.activation_dtype,
                    torch.float32,
                ):
                    raise ValueError("CSA2 pipeline pre_mix has incompatible shape or dtype")
            elif name in ("global_kv", "indexer_k"):
                dim = b.kv_dim if name == "global_kv" else b.indexer_dim
                if (
                    tensor.ndim != 3
                    or tensor.shape[1:] != (batch, dim)
                    or tensor.dtype != b.activation_dtype
                ):
                    raise ValueError(f"CSA2 pipeline {name} has incompatible shape or dtype")
            elif name in ("global_indices", "candidate_indices", "candidate_lengths"):
                leading = (s,) if b.qkv_format == "thd" else (batch, s)
                shape = tensor.shape if name == "candidate_lengths" else tensor.shape[:-1]
                if shape != leading or tensor.dtype != torch.int32:
                    raise ValueError(f"CSA2 pipeline {name} has incompatible shape or dtype")
            elif name in ("cu_seqlens", "cu_seqlens_padded"):
                if (
                    tensor.ndim != 1
                    or tensor.numel() == 0
                    or tensor.dtype not in (torch.int32, torch.int64)
                ):
                    raise ValueError(
                        "CSA2 pipeline cumulative lengths must be nonempty integer vectors"
                    )
        if b.qkv_format == "thd":
            if type(self.max_seqlen) is not int or self.max_seqlen < 0:
                raise ValueError("CSA2 THD pipeline max_seqlen must be a nonnegative host integer")
        elif self.max_seqlen is not None:
            raise ValueError("CSA2 SBHD pipeline payload cannot contain THD max_seqlen")
        if "indexer_k" in values and values["indexer_k"].shape[:2] != values["global_kv"].shape[:2]:
            raise ValueError("CSA2 pipeline shared K capacities must match")
        if (
            "global_kv" in values
            and b.qkv_format == "sbhd"
            and values["global_kv"].shape[0] != s // b.compress_ratio
        ):
            raise ValueError("CSA2 pipeline shared K capacity does not match the compression ratio")

    @property
    def tensor_specs(self) -> tuple[CSA2PipelineTensorSpec, ...]:
        """Describe actual dtypes, including native FP32 versus fused BF16 pre_mix."""
        differentiable = {"hidden_states", "pre_mix", "global_kv"}
        if self.boundary.indexer_k_requires_grad:
            differentiable.add("indexer_k")
        return tuple(
            CSA2PipelineTensorSpec(
                name, tuple(t.shape), t.dtype, name in differentiable and t.requires_grad
            )
            for name, t in zip(self.boundary.field_names, self.tensors)
        )

    def restore(
        self, *, use_fused_kernels: bool = False
    ) -> tuple[torch.Tensor, "CSA2State", SinglePassMHCState | None, PackedSeqParams | None]:
        """Create fresh working states and rebuild physical THD/flat-K caches locally.

        Repeated restoration is safe for replay: execution cursors and derived caches
        are independent, while shared activations keep their original autograd graph.
        """
        from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State

        self.validate()
        values = dict(zip(self.boundary.field_names, self.tensors))
        hidden = values["hidden_states"]
        b = self.boundary
        params = layout = compressed = None
        if b.qkv_format == "thd":
            params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=values["cu_seqlens"],
                cu_seqlens_kv=values["cu_seqlens"],
                cu_seqlens_q_padded=values["cu_seqlens_padded"],
                cu_seqlens_kv_padded=values["cu_seqlens_padded"],
                max_seqlen_q=self.max_seqlen,
                max_seqlen_kv=self.max_seqlen,
                pad_between_seqs=True,
            )
            layout = build_csa2_thd_layout(params, hidden.shape[0])
            if b.compress_ratio is not None:
                compressed = layout.for_compression(b.compress_ratio)
                if values["global_kv"].shape[0] != compressed.capacity:
                    raise ValueError(
                        "CSA2 pipeline shared K capacity does not match the THD layout"
                    )
        candidates = None
        if b.candidate_source_layer is not None:
            candidates = CSA2CandidateBlocks(
                values["candidate_indices"], values["candidate_lengths"], b.candidate_block_size
            )
        state = CSA2State(
            global_kv=values.get("global_kv"),
            indexer_k=values.get("indexer_k"),
            global_indices=values.get("global_indices"),
            candidates=candidates,
            kv_source_layer=b.kv_source_layer,
            index_source_layer=b.index_source_layer,
            candidate_source_layer=b.candidate_source_layer,
            sequence_length=hidden.shape[0],
            batch_size=hidden.shape[1],
            device=hidden.device,
            # The residual stream can be FP32 while attention computes in BF16.
            dtype=b.activation_dtype,
            last_layer=b.last_attention_layer,
            thd_layout=layout,
            compressed_layout=compressed,
        )
        if use_fused_kernels:
            state.prepare_fused_kv()
        mhc_state = SinglePassMHCState(values["pre_mix"]) if b.single_pass_mhc else None
        return hidden, state, mhc_state, params
