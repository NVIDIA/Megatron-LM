# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common invocation inputs and output/bias pairs for hybrid layers.

Legacy forwards remain supported during migration. The private adapter dispatches
only on the input container, never on the kind of layer or core.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch import Tensor

from megatron.core.utils import experimental_api, internal_api

if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
    from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext


@experimental_api
@dataclass(frozen=True)
class LayerInputs:
    """Invocation inputs; cores consume only the fields they need.

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
    cp_metadata: PackedSequenceCPMetadata | None = None
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


LayerOutput = tuple[Tensor, Tensor | None]


@experimental_api
@runtime_checkable
class LayerInterface(Protocol):
    """A layer returns hidden states and an optional deferred output-projection bias.

    Compute cores leave output bias separate for dropout/residual fusion. Complete
    residual layers return their updated hidden states with ``bias=None``. Shared
    expert contributions are combined inside MoE; auxiliary losses, attention
    context, and recurrent state are never carried in the bias slot.

    Implement ``forward(x, inputs) -> LayerOutput`` directly, or decorate an
    explicit-argument forward with ``layer_inputs_forward``. Legacy output
    formats are adapted only for shared calls.
    """

    def forward(self, x: Tensor, inputs: LayerInputs) -> LayerOutput:
        """Return exactly ``(hidden_states, bias)`` without applying deferred bias."""
        ...


@internal_api
def wants_input_ids(module) -> bool:
    """Whether a layer's MLP uses token IDs for hash routing."""
    router = getattr(getattr(module, "mlp", None), "router", None)
    return bool(getattr(router, "is_hash_layer", False))


@experimental_api
def layer_inputs_forward(
    legacy_forward=None,
    *,
    aliases=None,
    output=None,
    omit_none=(),
    exclude=(),
    include_if=None,
    input_transform=None,
    reject=None,
):
    """Accept ``(hidden_states, LayerInputs)`` through an ordinary forward signature.

    Binding is prepared at decoration time. Only named parameters matching input
    fields are selected; ``**kwargs`` never opts a layer into every field. Use
    ``copy_signature`` beneath this decorator for forwarding-only methods.

    ``aliases`` maps forward parameter names to input field names, including
    explicitly forwarded kwargs. ``exclude`` skips unused parameters, while
    ``omit_none`` preserves legacy graph argument sets. ``include_if`` maps
    parameter names to predicates on the module.
    ``input_transform(module, hidden_states)`` expresses shared-call-only normalization.
    ``reject`` is an exception for all shared calls, or a field-to-exception map
    rejecting non-None values. ``output`` converts legacy results explicitly.

    MegatronModule applies the attached binder before hooks and graph dispatch,
    then converts the result after execution. The wrapper also supports direct
    ``forward`` calls. Legacy calls bypass all shared-call policies. ``inputs``
    is reserved for the container and cannot be mixed with legacy arguments.
    """
    if legacy_forward is None:
        return lambda forward: layer_inputs_forward(
            forward,
            aliases=aliases,
            output=output,
            omit_none=omit_none,
            exclude=exclude,
            include_if=include_if,
            input_transform=input_transform,
            reject=reject,
        )

    parameters = list(inspect.signature(legacy_forward).parameters.values())[1:]
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("A LayerInputs forward needs an explicit hidden-state parameter")
    hidden_parameter = parameters[0]
    field_names = {field.name for field in fields(LayerInputs)}
    aliases = dict(aliases or {})
    include_if = dict(include_if or {})
    omit_none, exclude = frozenset(omit_none), frozenset(exclude)
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
            name in {hidden_parameter.name, "inputs"}
            or source not in field_names
            or (name not in named and not has_kwargs)
        ):
            raise ValueError(f"Invalid LayerInputs alias: {name} <- {source}")
    mapping = {
        name: aliases.get(name, name)
        for name in dict.fromkeys((*named, *aliases))
        if name not in exclude and aliases.get(name, name) in field_names
    }
    if not (set(omit_none) | set(include_if)) <= mapping.keys():
        raise ValueError("LayerInputs policies must name bound forward parameters")
    if not exclude <= named.keys():
        raise ValueError("LayerInputs exclusions must name forward parameters")
    for name, parameter in named.items():
        if name in mapping and parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError("LayerInputs fields must be keyword-capable parameters")
        if parameter.default is inspect.Parameter.empty and name not in mapping:
            raise ValueError(f"Required parameter {name} has no LayerInputs binding")
    if isinstance(reject, dict):
        if not reject.keys() <= field_names:
            raise ValueError("Rejected inputs must name LayerInputs fields")
        reject = tuple(reject.items())
    binding = tuple(
        (name, source, name in omit_none, include_if.get(name)) for name, source in mapping.items()
    )

    def adapter(module, call, args, kwargs, *, hidden_as_keyword=False):
        positional_inputs = len(args) >= 2 and isinstance(args[1], LayerInputs)
        if not positional_inputs and "inputs" not in kwargs:
            return call(*args, **kwargs)
        if positional_inputs:
            if len(args) != 2 or kwargs:
                raise TypeError("LayerInputs cannot be mixed with legacy arguments")
            hidden_states, inputs = args
        else:
            inputs = kwargs["inputs"]
            if not isinstance(inputs, LayerInputs):
                raise TypeError("inputs must be a LayerInputs instance")
            if len(args) == 1 and len(kwargs) == 1:
                hidden_states = args[0]
            elif not args and len(kwargs) == 2 and ("x" in kwargs or "hidden_states" in kwargs):
                hidden_states = kwargs["x"] if "x" in kwargs else kwargs["hidden_states"]
            else:
                raise TypeError("Use layer(hidden_states, inputs) without legacy arguments")
        if isinstance(reject, Exception):
            raise type(reject)(*reject.args)
        for source, error in reject or ():
            if getattr(inputs, source) is not None:
                raise type(error)(*error.args)
        bound = {}
        for name, source, skip_none, predicate in binding:
            value = getattr(inputs, source)
            if (skip_none and value is None) or (predicate is not None and not predicate(module)):
                continue
            bound[name] = value
        if input_transform is not None:
            hidden_states = input_transform(module, hidden_states)
        if hidden_as_keyword and hidden_parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
            bound[hidden_parameter.name] = hidden_states
            result = call(**bound)
        else:
            result = call(hidden_states, **bound)
        return output(result) if output is not None else result

    @wraps(legacy_forward)
    def forward(self, *args, **kwargs):
        return adapter(self, lambda *a, **kw: legacy_forward(self, *a, **kw), args, kwargs)

    forward._layer_input_adapter = adapter
    return forward
