# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid state contracts and ordinary tensor adapters, independent of execution backends.

Checkpoint, graph invocation and pipeline transport live in hybrid_stack_adapter.
Components can import these declarations without importing those executors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

import torch
from torch import Tensor
from torch.nn import Module

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.state_boundary import (
    StateBoundary,
    StateDependency,
    StatePlacement,
    StateRegion,
    TensorMappingCodec,
    TensorSchema,
    prepare_state_region,
)

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.module import CudaGraphCaptureRegion
    from megatron.core.transformer.transformer_config import TransformerConfig


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


class HybridGraphState(Protocol):
    """Native state conversion for one Hybrid layer's captured region.

    Hooks retain configuration and schemas only. Capture restores fresh native
    state from explicit tensors; replay publishes returned tensors into that
    invocation's state before the Hybrid host resumes its eager continuation.
    """

    def get_static_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Tensor]:
        """Return owned sample tensors, without modifying the shared call context."""
        ...

    def restore_inputs(self, hidden: Tensor, kwargs: Mapping[str, Any]) -> tuple[StateRegion, Any]:
        """Read explicit tensors and return a region plus fresh native state.

        The host binds the native state through the declaration's layer_kwargs,
        and exports produced fields through the region's codec. Unrelated
        host/component kwargs are read-only.
        """
        ...

    def prepare_replay(
        self, hidden: Tensor, kwargs: Mapping[str, Any]
    ) -> tuple[StateRegion, dict[str, Tensor], Callable]:
        """Return a region, owned tensor inputs and a native state publisher.

        The host derives output count/order/validation from the schema and decodes
        outputs with the codec. The publisher receives (ordinary outputs, decoded
        state), updates forward-local state and returns ordinary outputs, possibly
        attaching feature-owned autograd effects. All components observe the same
        read-only context, including packed metadata and native layer bindings.
        """
        ...


@dataclass(frozen=True)
class HybridTensorGraphState:
    """Default graph packing for states described entirely by a region and codec.

    Field keys also name the tensor inputs. The region factory receives the
    concrete hidden tensor and read-only call context. Native code supplies only
    state lookup and, for non-mapping states, publication. Mapping states publish
    with update(), preserving values retained outside the captured region.
    Features with special warmup values, derived caches or auxiliary autograd
    effects can implement HybridGraphState directly.
    """

    region_factory: Callable[[Tensor, Mapping[str, Any]], StateRegion]
    state_getter: Callable[[Mapping[str, Any]], Any]
    publish_state: Callable[[Any, Any], None] | None = None

    def get_static_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Tensor]:
        """Allocate one sample tensor for each present schema input."""
        hidden = inputs["hidden_states"]
        region = self.region_factory(hidden, inputs)
        return {
            f.key: torch.zeros(
                f.shape, dtype=f.dtype, device=hidden.device, requires_grad=f.differentiable
            )
            for f in region.schema.inputs
            if f.present
        }

    def restore_inputs(self, hidden: Tensor, kwargs: Mapping[str, Any]) -> tuple[StateRegion, Any]:
        """Restore fresh native capture state using schema field order."""
        region = self.region_factory(hidden, kwargs)
        tensors = tuple(kwargs[f.key] for f in region.schema.inputs if f.present)
        tensors = TensorSchema(region.schema.inputs).detach_ineligible(tensors)
        return region, region.codec.restore(region.schema.inputs, tensors, region.input_metadata)

    def prepare_replay(
        self, hidden: Tensor, kwargs: Mapping[str, Any]
    ) -> tuple[StateRegion, dict[str, Tensor], Callable]:
        """Export native input edges and publish only this region's outputs."""
        region, state = self.region_factory(hidden, kwargs), self.state_getter(kwargs)
        inputs = region.codec.export(state, region.schema.inputs)
        TensorSchema(region.schema.inputs).validate(inputs)
        owned = dict(zip((f.key for f in region.schema.inputs if f.present), inputs))

        def publish(result, restored):
            if self.publish_state is None:
                state.update(restored)
            else:
                self.publish_state(state, restored)
            return result

        return region, owned, publish


class HybridStateDeclaration:
    """A module's native state contract for the existing Hybrid host.

    The model wiring selects declarations; the executor never dispatches by
    module name. Fields, source identities and cache semantics belong here.
    Implementations retain configuration and model-owned module references,
    never a forward's working tensors.
    Subclass this declaration and override the native state methods below, or
    use HybridTensorState for mapping states. Optional setup and preparation
    hooks have defaults; graph support must be explicit when capture is enabled.
    """

    context_attribute: str

    def layer_kwargs(self, layer: Module, state: Any) -> dict[str, dict[str, Any]]:
        """Bind native arguments by consumer: attention, mlp, or the whole layer.

        The Hybrid host supplies the inner branch when a residual wrapper is
        present. For example, {"attention": {"memory_state": state}} targets the
        attention module without requiring a custom TransformerLayer. "layer"
        binds direct arguments of a custom layer or its residual wrapper.
        Additional named branches can read ``cross_layer_state.branch_kwargs``.
        Return {} for a non-consumer; duplicate native names within a consumer
        are errors even when different components use the same state argument.
        """
        raise NotImplementedError

    def recompute_boundary_tensors(self, state: Any) -> tuple[Tensor, ...]:
        """Snapshot live edges that may trigger selective recompute before hidden."""
        raise NotImplementedError

    def initial_state(self, hidden: Tensor, packed_seq_params: PackedSeqParams | None) -> Any:
        """Create independent working state at the model entry."""
        raise NotImplementedError

    def pipeline_region(
        self,
        boundary: HybridPipelineBoundary,
        hidden: Tensor,
        packed_seq_params: PackedSeqParams | None,
        *,
        requires_grad: bool = True,
    ) -> StateRegion:
        """Declare live tensors and immutable native metadata at a host cut."""
        raise NotImplementedError

    def checkpoint_region(self, start: int, end: int, hidden: Tensor, state: Any) -> StateRegion:
        """Declare inputs, outputs and retained fields for a host-selected region."""
        raise NotImplementedError

    def bind_placement(self, pattern: str, placement: tuple[StatePlacement, ...]) -> None:
        """Optionally bind the pipe-free pattern and ordered physical/virtual chunks.

        Position 2*i is layer i's entry and 2*i+1 its produced state. The host
        calls this once after building the layers, including PP=1, before
        region/schema queries. The default is placement independent.
        """

    def graph_state(
        self, layer: Module, symbol: str, region: CudaGraphCaptureRegion
    ) -> HybridGraphState | None:
        """Bind graph state, or return None when this component stays outside the region."""
        raise ValueError(
            f"{type(self).__name__} must declare graph_state for TE capture; "
            "return None when its consumers stay outside the captured region"
        )

    def prepare_local_state(self, hidden: Tensor, state: Any, inputs: Mapping[str, Any]) -> Any:
        """Merge local prepared tensors into native state, retaining received edges.

        The default returns state unchanged. Called once per forward on every
        chunk, outside checkpoint replay. Model-owned parameters may produce
        differentiable tensors from local input_ids;
        checkpoint_region must declare those tensors as explicit prepared inputs.
        """
        return state

    def validate_runtime(self, runtime: HybridRuntimeContext) -> None:
        """Optionally reject unsupported modes before schema queries or communication.

        The host checks its own limits first. Components check only their native
        requirements and identify the unsupported capability in the error.
        This hook may run again for each actual graph region.
        """


@dataclass(frozen=True)
class HybridRuntimeContext:
    """Setup facts supplied by the host, including the backend's guarantees.

    Placement is complete, including PP=1 and virtual chunks. A None graph_region
    denotes stack setup; later checks receive each layer's actual capture region.
    graph_output_activity describes independent None/zero output gradients, not
    tensor differentiability. The current TE host does not provide that guarantee.
    """

    placement: tuple[StatePlacement, ...]
    tensor_parallel_size: int
    context_parallel_size: int
    sequence_parallel: bool
    recompute_granularity: str | None
    graph_backend: str
    is_mtp_layer: bool
    graph_region: CudaGraphCaptureRegion | None = None
    graph_output_activity: bool = False


class HybridTensorState(HybridStateDeclaration):
    """Default declaration for ordinary mappings of field keys to tensors.

    Subclasses provide context_attribute, dependencies(hidden) and
    state_bindings(layer). Dependencies use the bound global placement and
    concrete hidden shape/dtype; they must not inspect activation values. PP,
    checkpoint, graph, retained fields and selective-recompute edges then share
    those declarations and TensorMappingCodec. Local prepared inputs can be
    supplied with prepare_local_state or initial_state.

    Stateful codecs, evolving metadata, special graph sample values or autograd
    publication should use HybridStateDeclaration and the smaller region/graph
    helpers directly. This convenience path adds no new execution lifecycle.
    """

    context_attribute: str

    def __init__(
        self, config: TransformerConfig, *, pp_layer_offset: int = 0, **kwargs: Any
    ) -> None:
        self.config, self.layer_offset = config, pp_layer_offset
        self.pattern, self.placement = "", ()
        self.codec = TensorMappingCodec()
        self._regions: dict[tuple, StateRegion] = {}

    def bind_placement(self, pattern: str, placement: tuple[StatePlacement, ...]) -> None:
        """Use the actual host placement for every boundary kind."""
        if self.placement and (pattern, placement) != (self.pattern, self.placement):
            raise ValueError(f"{self.context_attribute} placement is already bound")
        self.pattern, self.placement = pattern, placement

    def dependencies(self, hidden: Tensor) -> tuple[StateDependency, ...]:
        """Declare fields, producers and readers once in global event coordinates."""
        raise NotImplementedError

    def state_bindings(self, layer: Module) -> Mapping[str, str]:
        """Map each consuming branch to its native state argument, or return {}."""
        raise NotImplementedError

    def initial_state(
        self, hidden: Tensor, packed_seq_params: PackedSeqParams | None
    ) -> dict[str, Tensor]:
        """Create one empty mapping per invocation; native producers populate it."""
        return {}

    def layer_kwargs(self, layer: Module, state: Any) -> dict[str, dict[str, Any]]:
        """Apply the declared native argument names in eager and replay execution."""
        return {branch: {name: state} for branch, name in self.state_bindings(layer).items()}

    def recompute_boundary_tensors(self, state: Mapping[str, Tensor | None]) -> tuple[Tensor, ...]:
        """Expose the invocation's live tensor edges to selective recompute."""
        return tuple(value for value in state.values() if value is not None)

    def region(
        self, boundary: StateBoundary, hidden: Tensor, *, requires_grad: bool = True
    ) -> StateRegion:
        """Derive live inputs, outputs and retained fields from one dependency list."""
        if not self.placement:
            raise ValueError(f"Bind {self.context_attribute} placement before querying regions")
        # Only this static mapping helper caches regions. Arbitrary graph region
        # factories still run so their invocation-dependent codec metadata can be
        # checked. No hidden tensor or forward-local mapping enters this cache.
        profile = (boundary, tuple(hidden.shape), hidden.dtype, requires_grad)
        if profile in self._regions:
            return self._regions[profile]
        dependencies = self.dependencies(hidden)
        if not requires_grad:
            dependencies = tuple(
                replace(dep, field=replace(dep.field, differentiable=False)) for dep in dependencies
            )
        region = prepare_state_region(dependencies, boundary, self.placement, self.codec)
        self._regions[profile] = region
        return region

    def pipeline_region(
        self,
        boundary: HybridPipelineBoundary,
        hidden: Tensor,
        packed_seq_params: PackedSeqParams | None,
        *,
        requires_grad: bool = True,
    ) -> StateRegion:
        """Select live fields at an existing PP/VPP cut."""
        cut = 2 * boundary.layer_offset
        return self.region(
            StateBoundary(f"{self.context_attribute}/pp:{cut}", "pipeline", cut, cut),
            hidden,
            requires_grad=requires_grad,
        )

    def checkpoint_region(
        self, start: int, end: int, hidden: Tensor, state: Mapping[str, Tensor | None]
    ) -> StateRegion:
        """Select one host-owned checkpoint group without redeclaring its fields."""
        start, end = 2 * (self.layer_offset + start), 2 * (self.layer_offset + end)
        return self.region(
            StateBoundary(
                f"{self.context_attribute}/checkpoint:{start}:{end}", "checkpoint", start, end
            ),
            hidden,
        )

    def graph_state(
        self, layer: Module, symbol: str, region: CudaGraphCaptureRegion
    ) -> HybridTensorGraphState | None:
        """Derive graph packing and native state lookup from the same bindings."""
        bindings = self.state_bindings(getattr(layer, "inner_layer", layer))
        captured = bindings.keys() & region.branches
        if not captured:
            return None
        if captured != bindings.keys() or ("mlp" in captured and region.partial_mlp):
            raise ValueError(
                f"{self.context_attribute}: default tensor graph state requires complete consumer "
                "branches; implement graph_state for a split stateful region"
            )
        start = 2 * (layer.layer_number - 1)
        boundary = StateBoundary(
            f"{self.context_attribute}/capture:{start}", "capture", start, start + 2
        )
        branch, name = next(iter(bindings.items()))

        def state_getter(context):
            if branch == "layer":
                return context[name]
            return context["cross_layer_state"].branch_kwargs(branch)[name]

        return HybridTensorGraphState(lambda hidden, _: self.region(boundary, hidden), state_getter)
