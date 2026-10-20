# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Auxiliary invocation inputs and outputs for hybrid layers.

Computational cores keep their natural forward arguments; composition layers carry
the input container. Binding dispatches on that container, never on the kind of layer.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields, replace
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
    position_ids: Tensor | None = None
    mtp_input_mask: Tensor | None = None
    decoder_input: Tensor | None = None
    embedding: Callable | None = None
    mhc_multistream: Tensor | None = None
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

    def update(self, **updates) -> LayerAuxInputs:
        """Reuse this snapshot unless a field changes; never mutate earlier inputs."""
        if all(getattr(self, name) is value for name, value in updates.items()):
            return self
        return replace(self, **updates)


@experimental_api
@dataclass(frozen=True)
class LayerAuxOutputs:
    """Named outputs accompanying hidden states.

    ``bias`` is an additive contribution; ``mhc_multistream`` preserves the
    decoder stream consumed by MTP. Layers do not return or advance their invocation inputs.
    """

    bias: Tensor | None = None
    mhc_multistream: Tensor | None = None


def _flatten_layer_aux(aux_outputs):
    """Expose present fields to graph and autograd tree utilities."""
    names = tuple(
        field.name for field in fields(aux_outputs) if getattr(aux_outputs, field.name) is not None
    )
    return [getattr(aux_outputs, name) for name in names], names


register_pytree_node(
    LayerAuxOutputs,
    _flatten_layer_aux,
    lambda values, names: LayerAuxOutputs(**dict(zip(names, values))),
)
register_pytree_node(
    LayerAuxInputs,
    _flatten_layer_aux,
    lambda values, names: LayerAuxInputs(**dict(zip(names, values))),
)


LayerOutput = tuple[Tensor, LayerAuxOutputs]


def _normalize_layer_output(result) -> LayerOutput:
    """Bind natural tensors or bias pairs to hidden states and named outputs."""
    if isinstance(result, Tensor):
        return result, LayerAuxOutputs()
    hidden_states, aux_outputs = result
    if not isinstance(aux_outputs, LayerAuxOutputs):
        if aux_outputs is not None and not isinstance(aux_outputs, Tensor):
            raise TypeError(
                "Layer outputs must contain auxiliary outputs or a tensor bias, not inputs"
            )
        aux_outputs = LayerAuxOutputs(bias=aux_outputs)
    return hidden_states, aux_outputs


@experimental_api
def bind_layer_output_stage(stage):
    """Apply the layer's output hook to a completed split stage."""

    @wraps(stage)
    def forward(self, *args, **kwargs):
        return self.bind_layer_outputs(stage(self, *args, **kwargs))

    return forward


@experimental_api
@runtime_checkable
class LayerInterface(Protocol):
    """A layer accepts auxiliary inputs and returns hidden states and named outputs.

    Implement ``forward(x, aux_inputs) -> LayerOutput`` directly, or use
    ``@bind_layer_interface`` on an explicit-argument forward. Customize binding
    only through ``bind_layer_inputs`` and ``bind_layer_outputs``. Inputs are immutable;
    layers process them without returning a replacement for the next layer.
    """

    def bind_layer_inputs(self, hidden_states, aux_inputs: LayerAuxInputs, *, forward=None):
        """Return ``(args, kwargs)`` for execution, without running the layer.

        Override for validation, normalization, or argument adaptation. ``forward``
        selects the decorated method for direct calls and split stages; forward it
        to super when delegating. Normal module calls use ``self.forward``.
        """
        method = self.forward if forward is None else forward
        hidden_name, field_names = method._layer_input_schema
        kwargs = {
            name: aux_inputs if name == "aux_inputs" else getattr(aux_inputs, name)
            for name in field_names
        }
        if hidden_name is not None and getattr(self, "_layer_aux_input_hidden_as_keyword", False):
            kwargs[hidden_name] = hidden_states
            return (), kwargs
        return (hidden_states,), kwargs

    def bind_layer_outputs(self, result) -> LayerOutput:
        """Normalize a natural result to hidden states and named auxiliary outputs."""
        return _normalize_layer_output(result)

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


def _call_layer_interface(
    module, call, hidden_states, aux_inputs, *, forward=None, bind_outputs=True
):
    """Bind around execution; input-only stages leave their intermediate result intact."""
    args, kwargs = module.bind_layer_inputs(hidden_states, aux_inputs, forward=forward)
    result = call(*args, **kwargs)
    return module.bind_layer_outputs(result) if bind_outputs else result


@experimental_api
def bind_layer_interface(forward):
    """Bind inputs and outputs through the layer's two hooks.

    Module calls bind around PyTorch hooks and graph dispatch; direct forward calls
    bind around forward execution. Explicit-argument calls bypass both hooks.
    Use ``copy_signature`` beneath this decorator for forwarding-only methods.
    """
    return _bind_layer_method(forward, bind_outputs=True)


@experimental_api
def bind_layer_input_stage(stage):
    """Bind a split stage's inputs, preserving its opaque intermediate output."""
    return _bind_layer_method(stage, bind_outputs=False)


def _bind_layer_method(method, *, bind_outputs):
    """Cache signature fields and dispatch shared calls through the binding hooks."""
    parameters = list(inspect.signature(method).parameters.values())[1:]
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("A LayerAuxInputs forward needs an explicit hidden-state parameter")
    hidden_parameter = parameters[0]
    field_names = {field.name for field in fields(LayerAuxInputs)} | {"aux_inputs"}
    binding = []
    for parameter in parameters[1:]:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        name = parameter.name
        if name == "inputs":
            raise TypeError("A decorated forward must use explicit fields, not an inputs parameter")
        if name in field_names:
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError("LayerAuxInputs fields must be keyword-capable parameters")
            binding.append(name)
        elif parameter.default is inspect.Parameter.empty:
            raise ValueError(f"Required parameter {name} has no LayerAuxInputs binding")

    @wraps(method)
    def forward(self, *args, **kwargs):
        shared_inputs = _parse_layer_aux_inputs(args, kwargs)
        if shared_inputs is None:
            return method(self, *args, **kwargs)
        return _call_layer_interface(
            self, method.__get__(self), *shared_inputs, forward=forward, bind_outputs=bind_outputs
        )

    hidden_name = (
        None
        if hidden_parameter.kind == inspect.Parameter.POSITIONAL_ONLY
        else hidden_parameter.name
    )
    forward._layer_input_schema = hidden_name, tuple(binding)
    return forward
