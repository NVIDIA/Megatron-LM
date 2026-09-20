# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Common state boundaries for the Hybrid layer loop."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, ContextManager, Protocol

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Module, ModuleList

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayload,
    PipelinePayloadFactory,
    PipelinePayloadSpec,
    PipelineTensorSpec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.state_boundary import (
    BoundarySchema,
    CheckpointBoundaryPolicy,
    StateGraphAdapter,
    StatePlacement,
    StateRegion,
    TensorField,
    TensorMappingCodec,
    TensorSchema,
    compose_state_regions,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class HybridStackForwardContext(SimpleNamespace):
    """Working state for one forward; never retained on the stack or its adapter.

    Declarations supply the attributes and native state objects. Their codecs
    own tensor export and restoration at each boundary.
    """


class HybridStateDeclaration(Protocol):
    """A module's native state contract for the existing Hybrid host.

    The model wiring selects declarations; the executor never dispatches by
    module name. Fields, source identities and cache semantics belong here.
    Implementations retain configuration only, never a forward's tensors.
    """

    context_attribute: str

    def initial_state(self, hidden: Tensor, packed_seq_params: PackedSeqParams | None) -> Any:
        """Create independent working state at the model entry."""
        ...

    def pipeline_region(
        self,
        boundary: HybridPipelineBoundary,
        hidden: Tensor,
        packed_seq_params: PackedSeqParams | None,
        *,
        requires_grad: bool = True,
    ) -> StateRegion:
        """Declare live tensors and immutable native metadata at a host cut."""
        ...

    def checkpoint_region(self, start: int, end: int, hidden: Tensor, state: Any) -> StateRegion:
        """Declare inputs, outputs and retained fields for a host-selected region."""
        ...

    def graph_state(self, layer: Module, symbol: str) -> Any:
        """Bind native tensor conversion/publication hooks for StateGraphAdapter."""
        ...


def checkpointed_hybrid_forward(
    config: TransformerConfig,
    layers: ModuleList,
    hidden_states: Tensor,
    context: HybridStackForwardContext,
    *,
    tp_group: ProcessGroup,
    quantization_context: Callable[[TransformerConfig, int], ContextManager],
    layer_forward: Callable[[Module, Tensor, HybridStackForwardContext], Tensor],
    layer_kwargs: dict[str, Any] | None = None,
    boundary_factory: Callable[[int, int, Tensor, HybridStackForwardContext], StateRegion],
) -> Tensor:
    """Run full-recompute groups using the declarations' explicit tensor boundaries.

    ``layer_forward`` must use the supplied working context; it must not capture
    the caller's mutable context. Dynamic masks, positions and routing tensors
    travel in explicit arguments. Only returned tensor edges and independent
    restored state reach the next group or pipeline export.
    """

    use_te = bool(config.fp8 or config.fp4 or config.quant_recipe is not None)
    if config.distribute_saved_activations and not use_te:
        raise ValueError("State boundary checkpoint requires distribute_saved_activations=False")
    prepared, restore_arguments = _explicit_layer_arguments(layer_kwargs or {})
    prepared_count = len(prepared)

    def region_forward(start, end, region):
        def forward(hidden, *inputs):
            kwargs = restore_arguments(inputs[:prepared_count])
            working = region.codec.restore(
                region.schema.inputs, inputs[prepared_count:], region.input_metadata
            )
            for index in range(start, end):
                layer = layers[index]
                with quantization_context(config, layer.layer_number - 1):
                    hidden = layer_forward(layer, hidden, working, **kwargs)
            side = region.codec.export(working, region.schema.outputs)
            return (hidden, *side)

        return forward

    uniform = config.recompute_method == "uniform"
    remaining = config.recompute_num_layers
    group_size = remaining if uniform else 1
    for start in range(0, len(layers), group_size):
        end = min(start + group_size, len(layers))
        region = boundary_factory(start, end, hidden_states, context)
        # Retained values never enter the checkpoint body or its closure. Only
        # formal checkpoint outputs can replace state produced by this region.
        retained = region.codec.export(context, region.retained_fields)
        side = region.codec.export(context, region.schema.inputs)
        output_fields = [
            TensorField(
                "host/hidden", tuple(hidden_states.shape), hidden_states.dtype, "sbhd", True
            )
        ]
        output_fields.extend(region.schema.outputs)
        policy = CheckpointBoundaryPolicy(TensorSchema(tuple(output_fields)))
        forward = region_forward(start, end, region)
        args = (hidden_states, *prepared, *side)
        # Reentrant backends need a differentiable input. Frozen groups run
        # eagerly; block mode counts only groups with a live input edge.
        if (uniform or remaining > 0) and any(tensor.requires_grad for tensor in args):
            if use_te:
                outputs = te_checkpoint(
                    forward,
                    config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    tp_group,
                    *args,
                )
            else:
                outputs = tensor_parallel.checkpoint(forward, False, *args, boundary_policy=policy)
            remaining -= 1
        else:
            outputs = forward(*args)
        hidden_states = outputs[0]
        restored = region.codec.restore(
            (*region.retained_fields, *region.schema.outputs),
            (*retained, *outputs[1:]),
            region.output_metadata,
        )
        vars(context).update(vars(restored))
    return hidden_states


def _explicit_layer_arguments(arguments):
    """Separate original masks/positions/packed prefixes from immutable replay structure."""
    tensors = []

    def split(value):
        if isinstance(value, Tensor):
            index = len(tensors)
            tensors.append(value)
            return ("tensor", index)
        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                (
                    type(value),
                    tuple((f.name, split(getattr(value, f.name))) for f in fields(value) if f.init),
                ),
            )
        if isinstance(value, dict):
            return ("dict", tuple((key, split(item)) for key, item in value.items()))
        if isinstance(value, (tuple, list)):
            return ("tuple" if isinstance(value, tuple) else "list", tuple(map(split, value)))
        if value is None or isinstance(
            value, (bool, int, float, str, torch.dtype, torch.device, ProcessGroup)
        ):
            return ("constant", value)
        raise TypeError(f"Unsupported mutable checkpoint argument: {type(value).__name__}")

    structure = split(arguments)

    def restore(values):
        def unpack(spec):
            kind, value = spec
            if kind == "tensor":
                return values[value]
            if kind == "dataclass":
                cls, members = value
                return cls(**{key: unpack(item) for key, item in members})
            if kind == "dict":
                result = {key: unpack(item) for key, item in value}
                return result
            if kind in ("tuple", "list"):
                result = tuple(map(unpack, value))
                return list(result) if kind == "list" else result
            return value

        return unpack(structure)

    return tuple(tensors), restore


@dataclass(frozen=True)
class HybridPipelineBoundary:
    """A host-only cut used when no attention feature needs an additional payload."""

    layer_offset: int
    qkv_format: str = "sbhd"


@dataclass(frozen=True)
class HybridPipelineChunk:
    """Existing physical/virtual placement, independent of attention architecture."""

    layer_offset: int
    layer_pattern: str
    pp_rank: int
    vp_stage: int
    incoming: HybridPipelineBoundary | None
    outgoing: HybridPipelineBoundary | None


def build_hybrid_state_pipeline_plan(
    config: TransformerConfig, pattern: str, *, pp_size: int = 1, qkv_format: str = "sbhd"
) -> tuple[HybridPipelineChunk, ...]:
    """Describe existing nonempty chunks; do not choose a new layer placement."""
    parts = pattern.split(Symbols.PIPE)
    if (
        not all(parts)
        or len("".join(parts)) != config.num_layers
        or set("".join(parts)) - Symbols.VALID_LAYERS
        or type(pp_size) is not int
        or pp_size < 1
        or len(parts) % pp_size
        or qkv_format not in ("sbhd", "thd")
    ):
        raise ValueError("Invalid Hybrid state pipeline placement")
    offsets = [0]
    for part in parts:
        offsets.append(offsets[-1] + len(part))
    boundaries = [None, *(HybridPipelineBoundary(o, qkv_format) for o in offsets[1:-1]), None]
    return tuple(
        HybridPipelineChunk(
            offsets[i], part, i % pp_size, i // pp_size, boundaries[i], boundaries[i + 1]
        )
        for i, part in enumerate(parts)
    )


@dataclass(frozen=True)
class _HiddenPayload(PipelinePayload):
    """Ordinary hidden and packed prefixes; no attention-specific activation state."""

    boundary: HybridPipelineBoundary
    tensors: tuple[Tensor, ...]
    max_seqlen: int | None = None
    cp_size: int = 1
    cp_rank: int = 0
    cp_partition_mode: str | None = None

    @property
    def metadata(self) -> tuple[int, ...]:
        return (self.boundary.layer_offset, -1 if self.max_seqlen is None else self.max_seqlen) + (
            (self.cp_size, self.cp_rank) if self.cp_size > 1 else ()
        )

    @property
    def boundary_id(self) -> str:
        return f"hybrid.decoder/pp:{self.boundary.layer_offset}"

    @property
    def tensor_specs(self) -> tuple[PipelineTensorSpec, ...]:
        names = (
            ("hidden_states",)
            if self.max_seqlen is None
            else ("hidden_states", "cu_seqlens", "cu_seqlens_padded")
        )
        return tuple(
            PipelineTensorSpec(
                name,
                tuple(t.shape),
                t.dtype,
                name == "hidden_states" and t.requires_grad,
                self.boundary.qkv_format,
                key=(
                    f"host/{name}:L{self.boundary.layer_offset - 1}"
                    if name == "hidden_states"
                    else f"host/{name}:batch"
                ),
            )
            for name, t in zip(names, self.tensors)
        )

    def validate(self) -> None:
        if (
            type(self.cp_size) is not int
            or self.cp_size < 1
            or type(self.cp_rank) is not int
            or not 0 <= self.cp_rank < self.cp_size
        ):
            raise ValueError("Hybrid pipeline payload has invalid CP coordinates")
        if (self.boundary.qkv_format == "thd") != (self.max_seqlen is not None):
            raise ValueError("Hybrid pipeline packed metadata does not match its boundary")
        if len(self.tensors) != (1 if self.max_seqlen is None else 3):
            raise ValueError("Hybrid pipeline is missing hidden or packed prefix tensors")
        hidden = self.tensors[0]
        if hidden.ndim != 3 or not hidden.is_floating_point():
            raise ValueError("Hybrid pipeline hidden must have shape [S, B, C]")
        if self.max_seqlen is not None:
            if hidden.shape[1] != 1 or type(self.max_seqlen) is not int or self.max_seqlen < 0:
                raise ValueError("Invalid Hybrid THD activation shape or max_seqlen")
            if (
                any(
                    t.ndim != 1 or t.numel() < 1 or t.dtype not in (torch.int32, torch.int64)
                    for t in self.tensors[1:]
                )
                or self.tensors[1].shape != self.tensors[2].shape
            ):
                raise ValueError("Hybrid THD requires matching integer prefix vectors")

    def restore(
        self, *, cp_group: ProcessGroup | None = None
    ) -> tuple[Tensor, PackedSeqParams | None]:
        self.validate()
        coordinates = (1, 0) if cp_group is None else (cp_group.size(), cp_group.rank())
        if coordinates != (self.cp_size, self.cp_rank):
            raise ValueError("Hybrid pipeline payload has different CP coordinates")
        params = None
        if self.max_seqlen is not None:
            logical, physical = self.tensors[1:]
            params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=logical,
                cu_seqlens_kv=logical,
                cu_seqlens_q_padded=physical,
                cu_seqlens_kv_padded=physical,
                max_seqlen_q=self.max_seqlen,
                max_seqlen_kv=self.max_seqlen,
                cp_group=cp_group,
                local_cp_size=self.cp_size,
                cp_partition_mode=self.cp_partition_mode,
                pad_between_seqs=True,
            )
        return self.tensors[0], params


@dataclass(frozen=True)
class HybridStatePayload(PipelinePayload):
    """One flat snapshot for any set of declared module states."""

    tensors: tuple[Tensor, ...]
    spec: PipelinePayloadSpec
    region: StateRegion
    boundary: HybridPipelineBoundary
    cp_partition_mode: str | None = None

    @property
    def tensor_specs(self) -> tuple[PipelineTensorSpec, ...]:
        return self.spec.tensor_specs

    @property
    def metadata(self) -> tuple[int, ...]:
        return self.spec.metadata

    @property
    def boundary_id(self) -> str:
        return self.spec.boundary_id

    @property
    def max_seqlen(self) -> int | None:
        return None if self.metadata[1] == -1 else self.metadata[1]

    def _host_payload(self):
        values = dict(zip((s.field.key for s in self.tensor_specs if s.present), self.tensors))
        tensors = (values[f"host/hidden_states:L{self.boundary.layer_offset - 1}"],)
        if self.max_seqlen is not None:
            tensors += (values["host/cu_seqlens:batch"], values["host/cu_seqlens_padded:batch"])
        return _HiddenPayload(
            self.boundary,
            tensors,
            self.max_seqlen,
            *(self.metadata[2:] if len(self.metadata) == 4 else (1, 0)),
            self.cp_partition_mode,
        )

    def validate(self) -> None:
        """Check the declared wire contract without reading any tensor values."""
        self.spec.schema.validate(self.tensors)
        if any(t.device != self.tensors[0].device for t in self.tensors):
            raise ValueError("Hybrid pipeline tensors must share a device")
        self._host_payload().validate()

    def restore(
        self, *, cp_group: ProcessGroup | None = None
    ) -> tuple[Tensor, HybridStackForwardContext, PackedSeqParams | None]:
        """Restore fresh native states through the common codec dispatcher."""
        self.validate()
        hidden, params = self._host_payload().restore(cp_group=cp_group)
        context = self.region.codec.restore(
            self.region.schema.inputs, self.tensors, self.region.input_metadata
        )
        del context._boundary_inputs
        return hidden, context, params


class HybridStateAdapter:
    """Execute PP, checkpoint and graph boundaries for peer module declarations.

    Components declare fields/codecs and native state semantics. This host owns
    packing and execution; it never branches on a component's identity.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        components: tuple[HybridStateDeclaration, ...],
        hidden_size: int,
        hidden_dtype: torch.dtype,
        layer_type_list: list[str],
        pp_layer_offset: int,
        pre_process: bool,
        post_process: bool,
        is_mtp_layer: bool,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        if is_mtp_layer:
            raise ValueError("Hybrid state boundaries do not yet support MTP")
        self.config, self.components = config, tuple(components)
        self.hidden_size, self.hidden_dtype = hidden_size, hidden_dtype
        self.layer_offset, self.layer_pattern = pp_layer_offset, "".join(layer_type_list)
        self.pre_process, self.post_process = pre_process, post_process
        self.cp_group = pg_collection.cp
        self._pipeline_chunk, self._pipeline_chunks = None, {}
        names = [component.context_attribute for component in self.components]
        if len(set(names)) != len(names) or "_boundary_inputs" in names:
            raise ValueError("Hybrid state declarations must own distinct context attributes")

    def configure_cuda_graphs(self, layers: ModuleList) -> None:
        """Attach one graph boundary with all module declarations at the same level."""
        if self.config.cuda_graph_impl == "transformer_engine":
            for layer, symbol in zip(layers, self.layer_pattern):
                layer._te_cuda_graph_adapter = StateGraphAdapter(
                    tuple(component.graph_state(layer, symbol) for component in self.components)
                )

    def configure_pipeline(self, chunk: HybridPipelineChunk) -> None:
        """Bind an existing logical cut for local split execution."""
        if (
            chunk.layer_offset != self.layer_offset
            or chunk.layer_pattern != self.layer_pattern
            or self.pre_process != (chunk.incoming is None)
            or self.post_process != (chunk.outgoing is None)
        ):
            raise ValueError("Hybrid pipeline chunk does not match this stack")
        if self._pipeline_chunk is not None and self._pipeline_chunk != chunk:
            raise ValueError("Hybrid pipeline chunk already configured with a different plan")
        self._pipeline_chunk = chunk
        boundary = chunk.outgoing or chunk.incoming
        if boundary is not None:
            self._pipeline_chunks[boundary.qkv_format] = chunk

    def configure_distributed_pipeline(
        self, pattern: str, pp_group: ProcessGroup, vp_stage: int | None = None
    ) -> PipelinePayloadFactory | None:
        """Use the host's physical/virtual placement once for every state module."""
        if pp_group.size() == 1:
            return None
        vp_size = self.config.virtual_pipeline_model_parallel_size or 1
        if (vp_size > 1 and vp_stage is None) or not 0 <= (vp_stage or 0) < vp_size:
            raise ValueError("Hybrid pipeline requires the current virtual stage")
        if len(pattern.split("|")) != pp_group.size() * vp_size:
            raise ValueError("Hybrid pipeline segment count must equal PP size times VPP size")
        index = (vp_stage or 0) * pp_group.size() + pp_group.rank()
        plans = {
            layout: build_hybrid_state_pipeline_plan(
                self.config, pattern, pp_size=pp_group.size(), qkv_format=layout
            )
            for layout in ("sbhd", "thd")
        }
        self.configure_pipeline(plans["sbhd"][index])
        self._pipeline_chunks = {layout: chunks[index] for layout, chunks in plans.items()}
        placement = tuple(
            StatePlacement(
                2 * c.layer_offset,
                2 * (c.layer_offset + len(c.layer_pattern)),
                c.pp_rank,
                c.vp_stage,
            )
            for c in plans["sbhd"]
        )
        for component in self.components:
            bind = getattr(component, "bind_placement", None)
            if bind is not None:
                bind(pattern.replace("|", ""), placement)
        return self.make_pipeline_payload

    @property
    def consumes_input_tensor(self) -> bool:
        return self._pipeline_chunk is not None and self._pipeline_chunk.incoming is not None

    def _host_payload(self, boundary, hidden, params):
        packed = params is not None and params.qkv_format == "thd"
        maximum, prefixes = (params.max_seqlen_q, ()) if packed else (None, ())
        if packed:
            logical, physical = params.cu_seqlens_q, params.cu_seqlens_q_padded
            prefixes = (logical, logical if physical is None else physical)
        cp = (1, 0) if self.cp_group is None else (self.cp_group.size(), self.cp_group.rank())
        return _HiddenPayload(
            boundary, (hidden, *prefixes), maximum, *cp, self.config.cp_partition_mode
        )

    def _pipeline_region(self, boundary, hidden, params, requires_grad):
        host = self._host_payload(boundary, hidden, params)
        host.validate()
        host_fields = tuple(s.field for s in host.tensor_specs)
        host_fields = (replace(host_fields[0], differentiable=requires_grad), *host_fields[1:])
        regions = [
            (
                component.context_attribute,
                component.pipeline_region(boundary, hidden, params, requires_grad=requires_grad),
            )
            for component in self.components
        ]
        regions.append(
            (
                "_boundary_inputs",
                StateRegion(
                    BoundarySchema(host.boundary_id, host_fields, host_fields), TensorMappingCodec()
                ),
            )
        )
        region = compose_state_regions(host.boundary_id, regions, HybridStackForwardContext)
        # Preserve host-first wire order; shared prepared prefixes are sent only once.
        side = tuple(f for f in region.schema.inputs if f.key not in {h.key for h in host_fields})
        fields = (host_fields[0], *side, *host_fields[1:])
        region = replace(region, schema=BoundarySchema(host.boundary_id, fields, fields))
        specs = tuple(
            PipelineTensorSpec(
                f.key.split("/", 1)[1].split(":", 1)[0],
                f.shape,
                f.dtype,
                f.differentiable,
                f.layout,
                f.present,
                f.key,
            )
            for f in fields
        )
        return host, region, PipelinePayloadSpec(specs, host.metadata, host.boundary_id)

    def _snapshot_payload(self, tensors, spec, region, boundary):
        # Presence and gradient qualification use distinct maps. Absent entries
        # remain in the schema but never consume a position in the wire tuple.
        schema = TensorSchema(region.schema.inputs)
        schema.validate(tensors)
        active = {
            field.key: tensor.requires_grad for field, tensor in zip(schema.packed_fields, tensors)
        }
        fields = tuple(
            (
                replace(field, differentiable=field.differentiable and active[field.key])
                if field.present
                else field
            )
            for field in schema.fields
        )
        region = replace(region, schema=BoundarySchema(region.schema.boundary_id, fields, fields))
        spec = replace(
            spec,
            tensor_specs=tuple(
                replace(item, requires_grad=field.differentiable)
                for item, field in zip(spec.tensor_specs, fields)
            ),
        )
        payload = HybridStatePayload(tensors, spec, region, boundary, self.config.cp_partition_mode)
        payload.validate()
        return payload

    def make_pipeline_payload(
        self, tensors: tuple[Tensor, ...], metadata: tuple[int, ...]
    ) -> HybridStatePayload:
        """Reconstruct receive descriptors from tensor capacities and immutable metadata."""
        if len(metadata) not in (2, 4) or any(type(v) is not int for v in metadata):
            raise ValueError("Invalid Hybrid pipeline metadata")
        offset, maximum = metadata[:2]
        layout = "sbhd" if maximum == -1 else "thd"
        chunk = self._pipeline_chunks.get(layout)
        if chunk is None or chunk.incoming is None or chunk.incoming.layer_offset != offset:
            raise ValueError("Hybrid pipeline payload does not match the incoming boundary")
        if not tensors or tensors[0].ndim != 3 or tensors[0].shape[-1] != self.hidden_size:
            raise ValueError("Hybrid pipeline hidden width does not match the host declaration")
        host = _HiddenPayload(
            chunk.incoming,
            (tensors[0], *tensors[-2:]) if maximum != -1 else (tensors[0],),
            None if maximum == -1 else maximum,
            *(metadata[2:] if len(metadata) == 4 else (1, 0)),
            self.config.cp_partition_mode,
        )
        hidden, params = host.restore(cp_group=self.cp_group)
        _, region, spec = self._pipeline_region(chunk.incoming, hidden, params, True)
        return self._snapshot_payload(tensors, spec, region, chunk.incoming)

    def validate_input(self, payload: Any) -> None:
        """Validate the pending payload against the current receiving cut."""
        if isinstance(payload, HybridStatePayload):
            if self._pipeline_chunk is None:
                raise ValueError("Hybrid payload requires adapter.configure_pipeline() first")
            if self._pipeline_chunk.incoming is None:
                raise ValueError("The first Hybrid pipeline chunk cannot receive a payload")
            payload.validate()
            self.make_pipeline_payload(payload.tensors, payload.metadata)
        elif self._pipeline_chunk is not None and payload is not None:
            raise ValueError("Configured Hybrid chunks require a typed pipeline payload")

    @staticmethod
    def _validate_packed_params(actual, expected):
        if actual is None:
            return
        if (expected is None) != (actual.qkv_format != "thd"):
            raise ValueError("Hybrid pipeline packed layout does not match this batch")
        if expected is None:
            return
        if (
            actual.max_seqlen_q != expected.max_seqlen_q
            or actual.max_seqlen_kv != expected.max_seqlen_kv
            or actual.cp_partition_mode != expected.cp_partition_mode
        ):
            raise ValueError("Hybrid pipeline packed metadata does not match this batch")
        for suffix in ("q", "kv", "q_padded", "kv_padded"):
            value = getattr(actual, "cu_seqlens_" + suffix)
            if value is None and suffix.endswith("_padded"):
                value = getattr(actual, "cu_seqlens_" + suffix.removesuffix("_padded"))
            reference = getattr(expected, "cu_seqlens_" + suffix)
            message = "Hybrid pipeline packed prefixes do not match this batch"
            if value is None or (value.shape, value.dtype, value.device) != (
                reference.shape,
                reference.dtype,
                reference.device,
            ):
                raise ValueError(message)
            if value.is_cuda:
                torch._assert_async((value == reference).all(), message)
            elif not torch.equal(value, reference):
                raise ValueError(message)
        if actual.local_cp_size not in (None, expected.local_cp_size):
            raise ValueError("Hybrid pipeline packed CP size does not match this batch")
        if actual.cp_group is not None:
            coordinates = (actual.cp_group.size(), actual.cp_group.rank())
            expected_coordinates = (
                (1, 0)
                if expected.cp_group is None
                else (expected.cp_group.size(), expected.cp_group.rank())
            )
            if coordinates != expected_coordinates:
                raise ValueError("Hybrid pipeline packed CP coordinates do not match this batch")

    def prepare_forward(
        self,
        hidden_states: Any,
        packed_seq_params: PackedSeqParams | None,
        inference_context: BaseInferenceContext | None,
    ) -> tuple[Tensor, PackedSeqParams | None, HybridStackForwardContext]:
        """Create or restore each module's forward-local state through the same path."""
        if self._pipeline_chunk is not None and inference_context is not None:
            raise ValueError("Hybrid state pipeline supports training forwards only")
        if self.consumes_input_tensor:
            if hidden_states is None:
                raise ValueError(
                    "Hybrid receiving chunk requires a payload from set_input_tensor()"
                )
            self.validate_input(hidden_states)
            # Bind receiving declarations, even for direct same-process pipeline tests.
            payload = self.make_pipeline_payload(hidden_states.tensors, hidden_states.metadata)
            hidden_states, context, restored = payload.restore(cp_group=self.cp_group)
            self._validate_packed_params(packed_seq_params, restored)
            return hidden_states, restored, context
        context = HybridStackForwardContext()
        for component in self.components:
            setattr(
                context,
                component.context_attribute,
                component.initial_state(hidden_states, packed_seq_params),
            )
        return hidden_states, packed_seq_params, context

    def finalize_forward(
        self,
        output: Tensor | tuple[Tensor, Tensor],
        packed_seq_params: PackedSeqParams | None,
        context: HybridStackForwardContext,
    ) -> Any:
        """Snapshot every live component through one schema and one flat tensor tuple."""
        layout = (
            "thd"
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            else "sbhd"
        )
        chunk = self._pipeline_chunks.get(layout, self._pipeline_chunk)
        if chunk is None or chunk.outgoing is None:
            return output
        host, region, spec = self._pipeline_region(chunk.outgoing, output, packed_seq_params, True)
        _, restored = host.restore(cp_group=self.cp_group)
        self._validate_packed_params(packed_seq_params, restored)
        # Prefix snapshots must survive dataloader buffer reuse while PP is outstanding.
        values = tuple(
            t.clone() if not f.requires_grad else t for f, t in zip(host.tensor_specs, host.tensors)
        )
        context._boundary_inputs = dict(zip((s.field.key for s in host.tensor_specs), values))
        try:
            tensors = region.codec.export(context, region.schema.outputs)
        finally:
            del context._boundary_inputs
        return self._snapshot_payload(tensors, spec, region, chunk.outgoing)

    def pipeline_payload_spec(
        self,
        seq_length: int,
        micro_batch_size: int,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        requires_grad: bool = True,
    ) -> tuple[PipelinePayloadSpec | None, PipelinePayloadSpec | None]:
        """Prepare exact receive capacities without reading prefix values on device."""
        if self._pipeline_chunk is None:
            return None, None
        layout = (
            "thd"
            if packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            else "sbhd"
        )
        chunk = self._pipeline_chunks[layout]
        hidden = torch.empty(
            (seq_length, micro_batch_size, self.hidden_size), device="meta", dtype=self.hidden_dtype
        )
        return tuple(
            (
                None
                if b is None
                else self._pipeline_region(b, hidden, packed_seq_params, requires_grad)[2]
            )
            for b in (chunk.incoming, chunk.outgoing)
        )

    def checkpoint_region(
        self, start: int, end: int, hidden: Tensor, context: HybridStackForwardContext
    ) -> StateRegion:
        """Compose native checkpoint declarations without inspecting their fields."""
        return compose_state_regions(
            f"hybrid/checkpoint:{self.layer_offset+start}:{self.layer_offset+end}",
            tuple(
                (
                    component.context_attribute,
                    component.checkpoint_region(
                        start, end, hidden, getattr(context, component.context_attribute)
                    ),
                )
                for component in self.components
            ),
            HybridStackForwardContext,
        )
