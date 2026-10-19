# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Auxiliary invocation inputs and outputs for hybrid layers.

Legacy forwards remain supported during migration. The private adapter dispatches
only on the input container, never on the kind of layer or core.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch import Tensor
from torch.utils._pytree import register_pytree_node

from megatron.core.utils import experimental_api

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from contextlib import AbstractContextManager

    from megatron.core.context_parallel.layout import ContextParallelLayoutState
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
    from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
    from megatron.core.transformer.transformer_config import TransformerConfig


@experimental_api
@dataclass(frozen=True)
class LayerExecutionContext:
    """Per-invocation scheduling state for layers that execute several physical layers."""

    quant_context_factory: Callable[[TransformerConfig, int], AbstractContextManager]
    cp_layout_state: ContextParallelLayoutState | None = None
    residual_stream_recompute_plan: Sequence[ResidualStreamRecomputeContext | None] = ()


@experimental_api
@dataclass(frozen=True)
class LayerAuxInputs:
    """Inputs accompanying hidden states; cores consume only the fields they need.

    Residual recompute and mHC managers are transitional skeleton-only fields.
    The CP layout fields are used by an MTP depth containing a hybrid stack.
    """

    attention_mask: Tensor | None = None
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None
    inference_context: BaseInferenceContext | None = None
    packed_seq_params: PackedSeqParams | None = None
    padding_mask: Tensor | None = None
    input_ids: Tensor | None = None
    sequence_len_offset: Tensor | int | None = None
    packed_sequence_cp_metadata: PackedSequenceCPMetadata | None = None
    rotary_pos_cos: Tensor | None = None
    rotary_pos_sin: Tensor | None = None
    rotary_pos_cos_sin: Tensor | None = None
    attention_bias: Tensor | None = None
    context: Tensor | None = None
    context_mask: Tensor | None = None
    packed_seq_params_by_layout: Any = None
    cp_layout_plan: Any = None
    residual_stream_recompute_context: ResidualStreamRecomputeContext | None = None
    mhc_recompute_manager: Any = None
    execution_context: LayerExecutionContext | None = None


@experimental_api
@dataclass(frozen=True)
class LayerAuxOutputs:
    """Named outputs accompanying hidden states.

    Only ``bias`` is an additive contribution. ``context`` carries cross-attention
    state; ``mhc_multistream`` preserves the decoder stream consumed by MTP.
    """

    bias: Tensor | None = None
    context: Tensor | None = None
    mhc_multistream: Tensor | None = None


def _flatten_layer_aux_outputs(aux_outputs):
    """Expose only present tensors to graph and autograd tree utilities."""
    names = tuple(
        field.name for field in fields(aux_outputs) if getattr(aux_outputs, field.name) is not None
    )
    return [getattr(aux_outputs, name) for name in names], names


register_pytree_node(
    LayerAuxOutputs,
    _flatten_layer_aux_outputs,
    lambda values, names: LayerAuxOutputs(**dict(zip(names, values))),
)


LayerOutput = tuple[Tensor, LayerAuxOutputs]


def _normalize_layer_output(result, *, output_field="bias") -> LayerOutput:
    """Name a raw auxiliary output, preserving already-bound outputs."""
    if isinstance(result, Tensor):
        return result, LayerAuxOutputs()
    hidden_states, aux_outputs = result
    if not isinstance(aux_outputs, LayerAuxOutputs):
        return hidden_states, LayerAuxOutputs(**{output_field: aux_outputs})
    return result


@experimental_api
def bind_layer_outputs(forward=None, *, output_field="bias"):
    """Normalize a completed split-stage layer's natural return value.

    Unlike ``bind_layer_interface``, this decorator preserves the method's arguments
    and always binds its output. Use ``output_field="context"`` for a stage that returns
    hidden states and cross-attention context. Tensor-only stages need no options.
    """

    if output_field not in {field.name for field in fields(LayerAuxOutputs)}:
        raise ValueError(f"Unknown layer output field: {output_field}")
    if forward is None:
        return lambda method: bind_layer_outputs(method, output_field=output_field)

    @wraps(forward)
    def bound(*args, **kwargs):
        return _normalize_layer_output(forward(*args, **kwargs), output_field=output_field)

    return bound


@experimental_api
@runtime_checkable
class LayerInterface(Protocol):
    """A layer returns hidden states and named auxiliary outputs.

    Compute cores leave output bias separate for dropout/residual fusion. Complete
    residual layers return their updated hidden states with ``aux_outputs.bias=None``.
    Shared expert contributions are combined inside MoE. Attention context and
    the mHC stream have their own fields and are never treated as additive bias.

    Implement ``forward(x, aux_inputs) -> LayerOutput`` directly, or decorate an
    explicit-argument forward with ``bind_layer_interface``. Decorated forwards
    may return a tensor when no auxiliary outputs remain; the helper supplies an empty container.
    Legacy ``(hidden_states, bias)`` core outputs are adapted only for shared calls.
    """

    def forward(self, x: Tensor, aux_inputs: LayerAuxInputs) -> LayerOutput:
        """Return ``(hidden_states, aux_outputs)`` without applying deferred bias."""
        ...


def _parse_layer_aux_inputs(args, kwargs):
    """Extract a shared invocation, or return None for an ordinary forward call."""
    positional_inputs = len(args) >= 2 and isinstance(args[1], LayerAuxInputs)
    if not positional_inputs and "inputs" not in kwargs:
        return None
    if positional_inputs:
        if len(args) != 2 or kwargs:
            raise TypeError("LayerAuxInputs cannot be mixed with legacy arguments")
        return args
    aux_inputs = kwargs["inputs"]
    if not isinstance(aux_inputs, LayerAuxInputs):
        raise TypeError("inputs must be a LayerAuxInputs instance")
    if len(args) == 1 and len(kwargs) == 1:
        return args[0], aux_inputs
    if not args and len(kwargs) == 2 and ("x" in kwargs or "hidden_states" in kwargs):
        return kwargs["x"] if "x" in kwargs else kwargs["hidden_states"], aux_inputs
    raise TypeError("Use layer(hidden_states, aux_inputs) without legacy arguments")


@experimental_api
def bind_layer_interface(legacy_forward=None, *, aliases=None, input_transform=None):
    """Bind shared inputs and normalize a complete layer's output.

    Accept ``(hidden_states, LayerAuxInputs)`` through an ordinary forward signature
    and return ``(hidden_states, LayerAuxOutputs)`` for shared calls.

    Binding is prepared at decoration time. Only named parameters matching input
    fields are selected; ``**kwargs`` never opts a layer into every field. Use
    ``copy_signature`` beneath this decorator for forwarding-only methods.
    A parameter named ``aux_inputs`` receives the original container unchanged,
    allowing orchestration layers to pass it on without unpacking and rebuilding it.

    ``aliases`` maps forward parameter names to input field names, including
    explicitly forwarded kwargs. Layers that omit arguments customize
    ``layer_aux_inputs_forward``.
    ``input_transform(module, hidden_states)`` expresses shared-call-only normalization.
    Implementations may return a tensor or ``(hidden_states, LayerAuxOutputs)``.
    Tensor results receive an empty auxiliary-output container. Existing core ``(hidden_states, bias)``
    results receive ``LayerAuxOutputs(bias=bias)``. Named auxiliary outputs pass through unchanged.

    MegatronModule routes shared calls through its overridable
    ``layer_aux_inputs_forward`` method, which applies the binder before hooks and
    graph dispatch and normalizes the result after execution. Direct ``forward``
    calls use the same override but bypass hooks and graph dispatch. Legacy calls
    bypass shared-call policies. The compatibility keyword ``inputs=`` is reserved
    for the container and cannot be mixed with legacy arguments.
    """

    if legacy_forward is None:
        return lambda forward: bind_layer_interface(
            forward, aliases=aliases, input_transform=input_transform
        )

    bind = _make_layer_aux_input_binding(
        legacy_forward, aliases=aliases, input_transform=input_transform
    )

    def invoke(module, call, hidden_states, aux_inputs, *, hidden_as_keyword=False):
        return _normalize_layer_output(
            bind(module, call, hidden_states, aux_inputs, hidden_as_keyword=hidden_as_keyword)
        )

    @wraps(legacy_forward)
    def forward(self, *args, **kwargs):
        shared_inputs = _parse_layer_aux_inputs(args, kwargs)
        if shared_inputs is None:
            return legacy_forward(self, *args, **kwargs)
        return self.layer_aux_inputs_forward(
            *shared_inputs, _call=lambda *a, **kw: legacy_forward(self, *a, **kw)
        )

    forward._layer_aux_input_binding = invoke
    return forward


def _make_layer_aux_input_binding(legacy_forward, *, aliases=None, input_transform=None):
    """Prepare argument selection once for a full forward or an opaque split stage."""
    parameters = list(inspect.signature(legacy_forward).parameters.values())[1:]
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("A LayerAuxInputs forward needs an explicit hidden-state parameter")
    hidden_parameter = parameters[0]
    field_names = {field.name for field in fields(LayerAuxInputs)}
    aliases = dict(aliases or {})
    named = {
        parameter.name: parameter
        for parameter in parameters[1:]
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
    if "inputs" in named:
        raise TypeError("A decorated forward must use explicit fields, not an inputs parameter")
    for name, source in aliases.items():
        if (
            name in {hidden_parameter.name, "inputs", "aux_inputs"}
            or source not in field_names
            or (name not in named and not has_kwargs)
        ):
            raise ValueError(f"Invalid LayerAuxInputs alias: {name} <- {source}")
    mapping = {
        name: aliases.get(name, name)
        for name in dict.fromkeys((*named, *aliases))
        if name == "aux_inputs" or aliases.get(name, name) in field_names
    }
    for name, parameter in named.items():
        if name in mapping and parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError("LayerAuxInputs fields must be keyword-capable parameters")
        if parameter.default is inspect.Parameter.empty and name not in mapping:
            raise ValueError(f"Required parameter {name} has no LayerAuxInputs binding")
    binding = tuple(mapping.items())

    def invoke(module, call, hidden_states, aux_inputs, *, hidden_as_keyword=False):
        bound = {}
        for name, source in binding:
            value = aux_inputs if source == "aux_inputs" else getattr(aux_inputs, source)
            bound[name] = value
        if input_transform is not None:
            hidden_states = input_transform(module, hidden_states)
        if hidden_as_keyword and hidden_parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
            bound[hidden_parameter.name] = hidden_states
            result = call(**bound)
        else:
            result = call(hidden_states, **bound)
        return result

    return invoke


@experimental_api
def bind_layer_inputs(stage=None):
    """Bind shared inputs to a pre-core stage while preserving its opaque return state."""
    if stage is None:
        return lambda method: bind_layer_inputs(method)
    bind = _make_layer_aux_input_binding(stage)

    @wraps(stage)
    def forward(self, *args, **kwargs):
        shared_inputs = _parse_layer_aux_inputs(args, kwargs)
        if shared_inputs is None:
            return stage(self, *args, **kwargs)
        return self.layer_aux_inputs_forward_pre_attn_and_core_attn(*shared_inputs)

    forward._layer_aux_input_binding = bind
    return forward
