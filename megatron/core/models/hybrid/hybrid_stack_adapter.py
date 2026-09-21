# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Common state boundaries for the Hybrid layer loop."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, ContextManager, Mapping

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Module, ModuleList

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import te_checkpoint
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_state import (
    HybridGraphState,
    HybridPipelineBoundary,
    HybridPipelineChunk,
    HybridRuntimeContext,
    HybridStateDeclaration,
    build_hybrid_state_pipeline_plan,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayload,
    PipelinePayloadFactory,
    PipelinePayloadSpec,
    PipelineTensorSpec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import CudaGraphCaptureRegion, GraphableMegatronModule
from megatron.core.transformer.state_boundary import (
    BoundarySchema,
    CheckpointBoundaryPolicy,
    StatePlacement,
    StateRegion,
    StaticTensorSchema,
    TensorField,
    TensorMappingCodec,
    TensorSchema,
    compose_state_regions,
    validate_shared_input,
)
from megatron.core.transformer.transformer_config import TransformerConfig

# Host-prepared inputs may be consumed by multiple components at any composed
# boundary. Model specs extend this policy for PP, checkpoint and graph; produced state still
# requires a unique owner, as enforced by compose_state_regions.
_SHARED_PREPARED_INPUTS = ("host/cu_seqlens:batch", "host/cu_seqlens_padded:batch")


class HybridStackForwardContext(SimpleNamespace):
    """Working state for one forward; never retained on the stack or its adapter.

    Declarations supply the attributes and native state objects. Their codecs
    own tensor export and restoration at each boundary.
    """


class _HybridLayerState:
    """Merged native arguments for one layer, with forward-local recompute edges."""

    def __init__(self, bindings, components):
        self.bindings, self.components = bindings, components

    def branch_kwargs(self, branch: str) -> dict[str, Any]:
        return self.bindings.get(branch, {})

    def attention_kwargs(self) -> dict[str, Any]:
        return self.branch_kwargs("attention")

    def mlp_kwargs(self) -> dict[str, Any]:
        return self.branch_kwargs("mlp")

    def recompute_boundary_tensors(self) -> tuple[Tensor, ...]:
        tensors = {}
        for component, state in self.components:
            for tensor in component.recompute_boundary_tensors(state):
                tensors[id(tensor)] = tensor
        return tuple(tensors.values())


class HybridStateGraphAdapter:
    """Hybrid/TE capture and replay for a set of native state declarations.

    A declaration supplies native input restoration, output export and publication
    (including feature-owned side effects). This class alone calls the graph,
    packs/unpacks side outputs, validates static inputs and orders publication
    before the caller resumes its eager continuation. Packed arguments, default
    masks and TE invocation conventions belong to this host, not the common schema.
    """

    def __init__(
        self,
        components: tuple[HybridGraphState | None, ...],
        bind_states: Callable[[tuple[Any, ...]], dict[str, Any]],
        *,
        restore_packed: Callable | None = None,
        decompose_packed: Callable | None = None,
        cp_group: ProcessGroup | None = None,
        max_seqlen: int | None = None,
        shared_inputs: tuple[str, ...] = _SHARED_PREPARED_INPUTS,
    ) -> None:
        self.components = tuple(components)
        self.bind_states = bind_states
        self._static_names = ()
        self._static_schema = None
        self._single_output = None
        self._output_schemas = None
        self._component_inputs = ()
        self._state_contracts = None
        self.shared_inputs = tuple(shared_inputs)
        self.restore_packed = restore_packed
        self.decompose_packed = decompose_packed
        self.cp_group = cp_group
        self.max_seqlen = max_seqlen

    def _call_context(self, kwargs):
        context = dict(kwargs)
        reconstructing = context.get("packed_seq_params") is None and "cu_seqlens_q" in context
        if reconstructing and self.restore_packed is not None:
            self.restore_packed(context)
            params = context.get("packed_seq_params")
            if params is not None and self.cp_group is not None and self.cp_group.size() > 1:
                context["packed_seq_params"] = replace(
                    params, cp_group=self.cp_group, local_cp_size=self.cp_group.size()
                )
        return MappingProxyType(context)

    def _merge_inputs(self, target, owned, *, samples=False, declared_fields=()):
        conflicts = (target.keys() & owned.keys()) - set(self.shared_inputs)
        if conflicts:
            raise ValueError(f"Graph components bind duplicate tensor inputs: {sorted(conflicts)}")
        if any(not isinstance(value, Tensor) for value in owned.values()):
            raise TypeError("Graph components must return tensor-only inputs")
        fields = (
            {}
            if self._static_schema is None
            else {field.key: field for field in self._static_schema.schema.fields}
        )
        fields.update((field.key, field) for field in declared_fields)
        for name, tensor in owned.items():
            if name not in target:
                target[name] = tensor
                continue
            previous = target[name]
            field = fields.get(name) or TensorField(
                name, tuple(previous.shape), previous.dtype, "strided", previous.requires_grad
            )
            if samples:
                if (
                    previous.requires_grad != tensor.requires_grad
                    or previous.stride() != tensor.stride()
                ):
                    raise ValueError(f"Shared graph input {name} has conflicting sample signatures")
                # Samples are placeholders, not producer edges. Capture binds
                # both consumers to the one canonical sample retained in target.
                validate_shared_input(
                    replace(field, differentiable=False), previous.detach(), tensor.detach()
                )
            else:
                validate_shared_input(field, previous, tensor)

    def _compose_regions(self, regions, *, capture=False):
        # Metadata is immutable and tensor-free. repr also distinguishes scalar
        # types such as True/1 which compare equal but can change native control flow.
        contract = tuple(
            (r.schema, repr(r.input_metadata), repr(r.output_metadata), r.retained_fields)
            for r in regions
        )
        if self._state_contracts is not None and contract != self._state_contracts:
            raise ValueError("Hybrid graph state contract/metadata differs from capture")
        region = compose_state_regions(
            "hybrid/graph",
            tuple((f"component_{i}", r) for i, r in enumerate(regions)),
            HybridStackForwardContext,
            shared_inputs=self.shared_inputs,
        )
        if capture:
            self._state_contracts = contract
        return region

    def get_static_inputs(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Add component-owned inputs before the host finishes building TE samples."""
        context = self._call_context(inputs)
        inputs = dict(inputs)
        names = []
        for component in self.components:
            owned = {} if component is None else component.get_static_inputs(context)
            self._merge_inputs(inputs, owned, samples=True)
            names.append(tuple(owned))
        self._component_inputs = tuple(names)
        return inputs

    def finalize_sample_inputs(
        self, args: tuple[Tensor, ...], kwargs: Mapping[str, Tensor]
    ) -> None:
        """Bind the complete TE sample signature after all host input transformations."""
        if len(args) != 1 or "hidden_states" in kwargs:
            raise ValueError(
                "Hybrid graph samples require one positional hidden input and TE >= 1.10"
            )
        inputs = dict(hidden_states=args[0], **kwargs)
        missing = {name for names in self._component_inputs for name in names} - inputs.keys()
        if missing:
            raise ValueError(f"Final TE samples omit component inputs: {sorted(missing)}")
        fields = tuple(
            TensorField(name, tuple(t.shape), t.dtype, "strided", t.requires_grad)
            for name, t in inputs.items()
        )
        schema = StaticTensorSchema.from_tensors(fields, tuple(inputs.values()))
        if self._static_schema is not None and schema != self._static_schema:
            raise ValueError("Hybrid graph sample signatures must agree across microbatch slots")
        self._static_names, self._static_schema = tuple(inputs), schema

    def capture(self, function: Callable, *args, **kwargs) -> tuple[Tensor, ...]:
        """Restore all native states, execute the original region once, then export."""
        hidden = args[0] if args else kwargs["hidden_states"]
        context = self._call_context(kwargs)
        regions, states = [], []
        for component in self.components:
            if component is None:
                states.append(None)
                continue
            region, state = component.restore_inputs(hidden, context)
            regions.append(region)
            states.append(state)
        region = self._compose_regions(regions, capture=True)
        working = HybridStackForwardContext(
            **{
                f"component_{i}": state
                for i, state in enumerate(
                    state
                    for component, state in zip(self.components, states)
                    if component is not None
                )
            }
        )
        # Apply the same ownership and shared-input checks as PP/checkpoint.
        region.codec.export(working, region.schema.inputs)
        owned_names = {name for names in self._component_inputs for name in names}
        kwargs = {name: value for name, value in context.items() if name not in owned_names}
        bindings = self.bind_states(tuple(states))
        if kwargs.keys() & bindings.keys():
            raise ValueError("Hybrid graph state bindings cannot replace host arguments")
        kwargs.update(bindings)
        result = function(*args, **kwargs)
        single_output = isinstance(result, Tensor)
        if self._single_output is not None and self._single_output != single_output:
            raise ValueError("Hybrid graph capture changed its Tensor/tuple return structure")
        self._single_output = single_output
        outputs = (result,) if single_output else tuple(result)
        fields = tuple(
            TensorField(
                f"host/graph_output:{i}",
                tuple(t.shape),
                t.dtype,
                "strided",
                t.is_floating_point() or t.is_complex(),
            )
            for i, t in enumerate(outputs)
        )
        schemas = (fields, region.schema.outputs)
        if self._output_schemas is not None and self._output_schemas != schemas:
            raise ValueError("Hybrid graph capture changed its output schemas")
        self._output_schemas = schemas
        outputs += region.codec.export(working, region.schema.outputs)
        fields += region.schema.outputs
        return CheckpointBoundaryPolicy(TensorSchema(fields)).apply(outputs)

    def replay(self, function: Callable, *args, **kwargs) -> Any:
        """Wrap the actual graph call, after its eager prefix and before its eager tail."""
        if self._static_schema is None:
            raise RuntimeError("Hybrid graph replay requires finalized TE sample inputs")
        hidden = args[0] if args else kwargs["hidden_states"]
        context = self._call_context(kwargs)
        params = context.get("packed_seq_params")
        if (
            params is not None
            and self.max_seqlen is not None
            and (
                params.qkv_format != "thd"
                or params.max_seqlen_q != self.max_seqlen
                or params.max_seqlen_kv != self.max_seqlen
            )
        ):
            raise ValueError("Hybrid CUDA Graph requires the configured static max_seqlen")
        publications, component_inputs = [], []
        for component in self.components:
            if component is None:
                continue
            region, owned, publish = component.prepare_replay(hidden, context)
            component_inputs.append(owned)
            publications.append((region, publish))
        region = self._compose_regions(tuple(r for r, _ in publications))
        owned_inputs = {}
        for owned in component_inputs:
            self._merge_inputs(owned_inputs, owned, declared_fields=region.schema.inputs)
        # Packed conversion is host-owned and runs on a separate mapping only
        # after every component has seen the original, read-only call context.
        graph_kwargs = dict(kwargs)
        if self.decompose_packed is not None:
            self.decompose_packed(graph_kwargs)
        if params is not None:
            for suffix in ("q", "kv", "q_padded", "kv_padded"):
                name = "cu_seqlens_" + suffix
                if name in self._static_names:
                    value = getattr(params, name)
                    graph_kwargs[name] = (
                        getattr(params, name.removesuffix("_padded")) if value is None else value
                    )
        graph_kwargs = {
            name: value
            for name, value in graph_kwargs.items()
            if name in self._static_names and (value is None or isinstance(value, Tensor))
        }
        self._merge_inputs(graph_kwargs, owned_inputs, declared_fields=region.schema.inputs)
        # The default helper names its inputs by canonical field key. Enforce
        # their eligibility after checking the original shared producer edges.
        fields = tuple(f for f in region.schema.inputs if f.present and f.key in owned_inputs)
        tensors = TensorSchema(fields).detach_ineligible(tuple(graph_kwargs[f.key] for f in fields))
        graph_kwargs.update((f.key, tensor) for f, tensor in zip(fields, tensors))
        if self._static_schema is not None:
            values = dict(graph_kwargs, hidden_states=hidden)
            for name, field in zip(self._static_names, self._static_schema.schema.fields):
                if name in ("padding_mask", "attention_mask") and values.get(name) is None:
                    values[name] = graph_kwargs[name] = torch.zeros(
                        field.shape, dtype=field.dtype, device=hidden.device
                    )
            self._static_schema.validate(tuple(values.get(name) for name in self._static_names))

        def publish(outputs):
            outputs = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
            if self._output_schemas is None or (region.schema.outputs != self._output_schemas[1]):
                raise ValueError("Hybrid graph replay output schema differs from capture")
            schema = TensorSchema(
                tuple(field for fields in self._output_schemas for field in fields)
            )
            outputs = CheckpointBoundaryPolicy(schema).apply(outputs)
            position = len(TensorSchema(self._output_schemas[0]).packed_fields)
            result = outputs[:position]
            native = region.codec.restore(
                region.schema.outputs, outputs[position:], region.output_metadata
            )
            for i, (_, update) in enumerate(publications):
                result = update(result, getattr(native, f"component_{i}"))
            return result[0] if self._single_output else result

        return publish(function(*args, **graph_kwargs))


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
    if config.distribute_saved_activations:
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
        retained = TensorSchema(region.retained_fields).detach_ineligible(
            region.codec.export(context, region.retained_fields)
        )
        side = TensorSchema(region.schema.inputs).detach_ineligible(
            region.codec.export(context, region.schema.inputs)
        )
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
                    boundary_policy=policy,
                )
            else:
                outputs = tensor_parallel.checkpoint(forward, False, *args, boundary_policy=policy)
            remaining -= 1
        else:
            outputs = policy.apply(forward(*args))
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
class _HiddenPayload(PipelinePayload):
    """Ordinary hidden and packed prefixes; no attention-specific activation state."""

    boundary: HybridPipelineBoundary
    tensors: tuple[Tensor, ...]
    max_seqlen: int | None = None
    cp_size: int = 1
    cp_rank: int = 0
    cp_partition_mode: str | None = None
    tokens_per_sample: int | None = None

    @property
    def metadata(self) -> tuple[int, ...]:
        # Optional CP coordinates occupy two slots; packed sample grouping adds
        # one trailing slot. Keep descriptors without sample grouping unchanged.
        return (
            (self.boundary.layer_offset, -1 if self.max_seqlen is None else self.max_seqlen)
            + ((self.cp_size, self.cp_rank) if self.cp_size > 1 else ())
            + (() if self.tokens_per_sample is None else (self.tokens_per_sample,))
        )

    @classmethod
    def from_metadata(cls, boundary, tensors, metadata, *, cp_partition_mode):
        """Decode the same host contract for receive factories and native restore."""
        if len(metadata) not in (2, 3, 4, 5) or any(type(v) is not int for v in metadata):
            raise ValueError("Invalid Hybrid pipeline metadata")
        offset, maximum = metadata[:2]
        if offset != boundary.layer_offset:
            raise ValueError("Hybrid pipeline metadata does not match its boundary")
        return cls(
            boundary,
            tensors,
            None if maximum == -1 else maximum,
            *(metadata[2:4] if len(metadata) >= 4 else (1, 0)),
            cp_partition_mode,
            metadata[-1] if len(metadata) in (3, 5) else None,
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
                TensorField(
                    (
                        f"host/{name}:L{self.boundary.layer_offset - 1}"
                        if name == "hidden_states"
                        else f"host/{name}:batch"
                    ),
                    tuple(t.shape),
                    t.dtype,
                    self.boundary.qkv_format,
                    name == "hidden_states",
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
        if self.tokens_per_sample is not None and (
            self.max_seqlen is None
            or type(self.tokens_per_sample) is not int
            or self.tokens_per_sample <= 0
        ):
            raise ValueError("Invalid Hybrid pipeline packed tokens_per_sample")
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
                tokens_per_sample=self.tokens_per_sample,
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
        return _HiddenPayload.from_metadata(
            self.boundary, tensors, self.metadata, cp_partition_mode=self.cp_partition_mode
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
        shared_inputs: tuple[str, ...] = (),
    ) -> None:
        self.config, self.components = config, tuple(components)
        self.is_mtp_layer = is_mtp_layer
        self.shared_inputs = tuple(dict.fromkeys((*_SHARED_PREPARED_INPUTS, *shared_inputs)))
        self.hidden_size, self.hidden_dtype = hidden_size, hidden_dtype
        self.layer_offset, self.layer_pattern = pp_layer_offset, "".join(layer_type_list)
        self.pre_process, self.post_process = pre_process, post_process
        self.cp_group = pg_collection.cp
        self._pipeline_chunk, self._pipeline_chunks = None, {}
        self._placement_binding = None
        self._graphs_configured = False
        self._validate_runtime()
        names = [component.context_attribute for component in self.components]
        if len(set(names)) != len(names) or "_boundary_inputs" in names:
            raise ValueError("Hybrid state declarations must own distinct context attributes")

    def layer_kwargs(self, layer: Module, context: HybridStackForwardContext) -> dict[str, Any]:
        """Resolve component bindings identically in eager execution and replay."""
        return self._bind_layer_states(
            layer, tuple(getattr(context, c.context_attribute) for c in self.components)
        )

    def _bind_layer_states(self, layer, states):
        bindings, live = {}, []
        for component, state in zip(self.components, states):
            if state is None:
                continue
            live.append((component, state))
            for branch, arguments in component.layer_kwargs(layer, state).items():
                target = bindings.setdefault(branch, {})
                conflicts = target.keys() & arguments.keys()
                if conflicts:
                    raise ValueError(
                        f"Hybrid state components bind duplicate {branch} arguments: {sorted(conflicts)}"
                    )
                target.update(arguments)
        kwargs = bindings.pop("layer", {})
        if bindings:
            if "cross_layer_state" in kwargs:
                raise ValueError("Direct cross_layer_state conflicts with native branch bindings")
            kwargs["cross_layer_state"] = _HybridLayerState(bindings, tuple(live))
        return kwargs

    def recompute_boundary_tensors(self, context: HybridStackForwardContext) -> tuple[Tensor, ...]:
        """Combine live edges by tensor identity without inspecting native state."""
        tensors = {}
        for component in self.components:
            state = getattr(context, component.context_attribute)
            for tensor in component.recompute_boundary_tensors(state):
                tensors[id(tensor)] = tensor
        return tuple(tensors.values())

    def _validate_runtime(self, graph_region: CudaGraphCaptureRegion | None = None) -> None:
        """Check host limits, then native requirements, before querying declarations."""
        runtime = HybridRuntimeContext(
            placement=() if self._placement_binding is None else self._placement_binding[1],
            tensor_parallel_size=self.config.tensor_model_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
            sequence_parallel=self.config.sequence_parallel,
            recompute_granularity=self.config.recompute_granularity,
            graph_backend=self.config.cuda_graph_impl,
            is_mtp_layer=self.is_mtp_layer,
            graph_region=graph_region,
        )
        if runtime.is_mtp_layer:
            raise ValueError("Hybrid host: state boundaries do not yet support MTP")
        if runtime.graph_backend not in ("none", "transformer_engine"):
            raise ValueError(
                "Hybrid host: state boundaries support Transformer Engine CUDA Graphs only"
            )
        if runtime.graph_backend != "none" and runtime.recompute_granularity == "full":
            raise ValueError("Hybrid TE backend: state boundaries do not support full recompute")
        # Component requirements may depend on physical/virtual placement. Do not
        # invoke them with a placeholder plan while the stack is being constructed.
        if not runtime.placement:
            return
        for component in self.components:
            component.validate_runtime(runtime)

    def configure_cuda_graphs(self, layers: ModuleList) -> None:
        """Attach one graph boundary with all module declarations at the same level."""
        if self.config.cuda_graph_impl == "transformer_engine":
            if self._graphs_configured:
                return
            if self._placement_binding is None:
                raise ValueError("Bind Hybrid state placement before configuring CUDA graphs")
            for layer, symbol in zip(layers, self.layer_pattern):
                if not isinstance(layer, GraphableMegatronModule):
                    continue
                region = layer.get_te_cuda_graph_capture_region()
                self._validate_runtime(region)
                graphs = []
                for component in self.components:
                    graphs.append(component.graph_state(layer, symbol, region))
                layer._te_cuda_graph_adapter = HybridStateGraphAdapter(
                    tuple(graphs),
                    lambda states, branch=getattr(
                        layer, "inner_layer", layer
                    ): self._bind_layer_states(branch, states),
                    restore_packed=getattr(
                        layer, "_reconstruct_packed_seq_params_from_kwargs", None
                    ),
                    decompose_packed=getattr(layer, "_decompose_packed_seq_params_to_kwargs", None),
                    cp_group=self.cp_group,
                    max_seqlen=(
                        self.config.max_seqlen_per_dp_cp_rank * self.config.context_parallel_size
                        if self.config.max_seqlen_per_dp_cp_rank is not None
                        else None
                    ),
                    shared_inputs=self.shared_inputs,
                )
            self._graphs_configured = True

    def bind_placement(self, pattern: str, placement: tuple[StatePlacement, ...]) -> None:
        """Bind complete physical/virtual placement once before querying components."""
        binding = (pattern, tuple(placement))
        if self._placement_binding is not None:
            if binding != self._placement_binding:
                raise ValueError("Hybrid state placement is already bound to a different plan")
            return
        for component in self.components:
            component.bind_placement(*binding)
        self._placement_binding = binding
        self._validate_runtime()

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
        placement = tuple(
            StatePlacement(
                2 * c.layer_offset,
                2 * (c.layer_offset + len(c.layer_pattern)),
                c.pp_rank,
                c.vp_stage,
            )
            for c in plans["sbhd"]
        )
        self.bind_placement(pattern.replace("|", ""), placement)
        if pp_group.size() == 1:
            return None
        self.configure_pipeline(plans["sbhd"][index])
        self._pipeline_chunks = {layout: chunks[index] for layout, chunks in plans.items()}
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
            boundary,
            (hidden, *prefixes),
            maximum,
            *cp,
            self.config.cp_partition_mode,
            tokens_per_sample=params.tokens_per_sample if packed else None,
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
        region = compose_state_regions(
            host.boundary_id, regions, HybridStackForwardContext, shared_inputs=self.shared_inputs
        )
        # Preserve host-first wire order; shared prepared prefixes are sent only once.
        side = tuple(f for f in region.schema.inputs if f.key not in {h.key for h in host_fields})
        fields = (host_fields[0], *side, *host_fields[1:])
        region = replace(region, schema=BoundarySchema(host.boundary_id, fields, fields))
        specs = tuple(
            PipelineTensorSpec(f.key.rsplit("/", 1)[-1].split(":", 1)[0], f) for f in fields
        )
        return host, region, PipelinePayloadSpec(specs, host.metadata, host.boundary_id)

    def _snapshot_payload(self, tensors, spec, region, boundary):
        # Gradient eligibility belongs to the static communication contract.
        # Frozen producers may have no local autograd edge for an eligible field.
        payload = HybridStatePayload(tensors, spec, region, boundary, self.config.cp_partition_mode)
        payload.validate()
        return payload

    def make_pipeline_payload(
        self, tensors: tuple[Tensor, ...], descriptor: PipelinePayloadSpec
    ) -> HybridStatePayload:
        """Restore a receive boundary using its declared gradient eligibility."""
        metadata = descriptor.metadata
        if len(metadata) not in (2, 3, 4, 5) or any(type(v) is not int for v in metadata):
            raise ValueError("Invalid Hybrid pipeline metadata")
        offset, maximum = metadata[:2]
        layout = "sbhd" if maximum == -1 else "thd"
        chunk = self._pipeline_chunks.get(layout)
        if chunk is None or chunk.incoming is None or chunk.incoming.layer_offset != offset:
            raise ValueError("Hybrid pipeline payload does not match the incoming boundary")
        if not tensors or tensors[0].ndim != 3 or tensors[0].shape[-1] != self.hidden_size:
            raise ValueError("Hybrid pipeline hidden width does not match the host declaration")
        host = _HiddenPayload.from_metadata(
            chunk.incoming,
            (tensors[0], *tensors[-2:]) if maximum != -1 else (tensors[0],),
            metadata,
            cp_partition_mode=self.config.cp_partition_mode,
        )
        hidden, params = host.restore(cp_group=self.cp_group)
        requires_grad = any(field.requires_grad for field in descriptor.tensor_specs)
        _, region, spec = self._pipeline_region(chunk.incoming, hidden, params, requires_grad)
        if spec != descriptor:
            raise ValueError("Received Hybrid schema disagrees with its state declarations")
        return self._snapshot_payload(tensors, spec, region, chunk.incoming)

    def validate_input(self, payload: Any) -> None:
        """Validate the pending payload against the current receiving cut."""
        if isinstance(payload, HybridStatePayload):
            if self._pipeline_chunk is None:
                raise ValueError("Hybrid payload requires adapter.configure_pipeline() first")
            if self._pipeline_chunk.incoming is None:
                raise ValueError("The first Hybrid pipeline chunk cannot receive a payload")
            payload.validate()
            self.make_pipeline_payload(payload.tensors, payload.descriptor)
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
            actual.tokens_per_sample is not None
            and actual.tokens_per_sample != expected.tokens_per_sample
        ):
            raise ValueError("Hybrid pipeline packed tokens_per_sample does not match this batch")
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
        *,
        runtime_inputs: Mapping[str, Any] | None = None,
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
            payload = self.make_pipeline_payload(hidden_states.tensors, hidden_states.descriptor)
            hidden_states, context, restored = payload.restore(cp_group=self.cp_group)
            self._validate_packed_params(packed_seq_params, restored)
            packed_seq_params = restored
        else:
            context = HybridStackForwardContext()
            for component in self.components:
                setattr(
                    context,
                    component.context_attribute,
                    component.initial_state(hidden_states, packed_seq_params),
                )
        inputs = MappingProxyType(dict(runtime_inputs or {}, packed_seq_params=packed_seq_params))
        for component in self.components:
            state = getattr(context, component.context_attribute)
            setattr(
                context,
                component.context_attribute,
                component.prepare_local_state(hidden_states, state, inputs),
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
        # Eligibility follows the execution mode used to prepare PP plans.
        # Frozen producers still declare gradient slots during training; only
        # no-grad evaluation removes them, independently of tensor.requires_grad.
        host, region, spec = self._pipeline_region(
            chunk.outgoing, output, packed_seq_params, torch.is_grad_enabled()
        )
        _, restored = host.restore(cp_group=self.cp_group)
        self._validate_packed_params(packed_seq_params, restored)
        context._boundary_inputs = dict(zip((s.field.key for s in host.tensor_specs), host.tensors))
        try:
            tensors = region.codec.export(context, region.schema.outputs)
        finally:
            del context._boundary_inputs
        # Validate shared input identity before taking one snapshot per prefix.
        # Snapshots must survive dataloader reuse while PP is outstanding.
        prefix_keys = {s.field.key for s in host.tensor_specs[1:]}
        tensors = tuple(
            t.clone() if f.key in prefix_keys else t
            for f, t in zip(TensorSchema(region.schema.outputs).packed_fields, tensors)
        )
        return self._snapshot_payload(tensors, spec, region, chunk.outgoing)

    def pipeline_payload_spec(
        self,
        seq_length: int,
        micro_batch_size: int,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        requires_grad: bool = True,
    ) -> tuple[PipelinePayloadSpec | None, PipelinePayloadSpec | None]:
        """Prepare receive capacities without reading prefix values on device.

        ``requires_grad`` must match the forward's grad mode (False for no-grad
        evaluation), regardless of whether individual producers are frozen.
        """
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
            shared_inputs=self.shared_inputs,
        )
