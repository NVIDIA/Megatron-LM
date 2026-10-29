# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared argument packing and the interface for interchangeable layers."""

from __future__ import annotations

from dataclasses import MISSING, FrozenInstanceError, dataclass, field, make_dataclass
from functools import wraps
from inspect import Parameter, Signature, signature
from typing import Any, Protocol, runtime_checkable

from torch import Tensor
from torch.utils._pytree import register_pytree_node

from megatron.core.utils import deprecate_inference_params, experimental_api


@experimental_api
class AuxInputs:
    """Read-only named arguments shared across calls.

    Missing attributes read as None. Missing inputs still use the forward's
    defaults or raise an error if required.
    """

    def __init__(self, **values) -> None:
        self.__dict__.update(values)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return None

    def __setattr__(self, name, value):
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name):
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __repr__(self):
        values = ", ".join(f"{name}={value!r}" for name, value in vars(self).items())
        return f"{type(self).__name__}({values})"

    def update(self, **updates) -> AuxInputs:
        """Return inputs with added fields; reject changes to existing fields."""
        _check_layer_input_collisions(self, updates)
        return AuxInputs(**vars(self), **updates) if updates else self


@experimental_api
@dataclass(frozen=True)
class LayerAuxOutputs:
    """Extra layer outputs, including any bias the caller must apply."""

    bias: Tensor | None = None


def _register_layer_aux(cls):
    """Let PyTorch find and rebuild the tensors inside these objects."""
    register_pytree_node(
        cls,
        lambda value: (list(vars(value).values()), tuple(vars(value))),
        lambda values, names: cls(**dict(zip(names, values))),
    )


_register_layer_aux(AuxInputs)
_register_layer_aux(LayerAuxOutputs)


LayerOutput = tuple[Tensor, LayerAuxOutputs]


@experimental_api
class AuxInputBinding(Protocol):
    """Turn shared inputs into forward arguments, leaving the return type alone."""

    def bind_layer_inputs(self, hidden_states, aux_inputs: AuxInputs, *, forward=None):
        """Select forward arguments and pass leftover fields through ``aux_inputs``.

        Override to validate or change inputs. ``forward`` selects a split stage
        or another method; pass it to super when delegating.
        """
        method = self.forward if forward is None else forward
        parameters = method._layer_input_signature.parameters
        values = vars(aux_inputs)
        kwargs = {
            name: values.get(name, parameters[name].default)
            for name in method.Inputs.__dataclass_fields__
            if name in values or parameters[name].default is not Parameter.empty
        }
        if "aux_inputs" in parameters:
            kwargs["aux_inputs"] = (
                AuxInputs(**{name: value for name, value in values.items() if name not in kwargs})
                if values.keys() & kwargs.keys()
                else aux_inputs
            )
        return (hidden_states,), kwargs


@experimental_api
@runtime_checkable
class LayerInterface(AuxInputBinding, Protocol):
    """Give interchangeable layers the same inputs and output pair.

    Inheritance binds ``forward`` automatically, including overrides. No decorator
    is needed. Use ``@bind_aux_inputs`` alone when only input packing is needed.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        forward = cls.forward
        if forward is not LayerInterface.forward and not hasattr(forward, "_layer_input_signature"):
            cls.forward = bind_aux_inputs(forward)

    def bind_layer_outputs(self, result) -> LayerOutput:
        """Return hidden states and LayerAuxOutputs from a tensor or a bias pair."""
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

    def forward(self, x: Tensor, aux_inputs: AuxInputs) -> LayerOutput:
        """Return ``(hidden_states, aux_outputs)`` with any bias left for the caller."""
        ...


def _check_layer_input_collisions(aux_inputs, explicit_inputs):
    duplicates = vars(aux_inputs).keys() & explicit_inputs.keys()
    if duplicates:
        raise TypeError(f"Fields already present in AuxInputs: {', '.join(sorted(duplicates))}")


def _parse_layer_aux_inputs(args, kwargs, *, forward, allow_aux_keyword=False):
    """Find hidden states and shared inputs, or return None for a regular call."""
    if len(args) >= 2 and isinstance(args[1], AuxInputs):
        hidden_name = next(iter(forward._layer_input_signature.parameters))
        reserved = {"inputs", "aux_inputs", "hidden_states", "x", hidden_name}
        if len(args) != 2 or kwargs.keys() & reserved:
            raise TypeError("Pass hidden states and AuxInputs once, followed by keyword arguments")
        return args[0], args[1].update(**kwargs) if kwargs else args[1]

    input_key = "inputs" if "inputs" in kwargs else "aux_inputs"
    aux_inputs = kwargs.get(input_key)
    if input_key == "aux_inputs":
        if not isinstance(aux_inputs, AuxInputs):
            return None
        if not allow_aux_keyword or len(args) > 1:
            # Direct forward calls keep their ordinary arguments. Module calls
            # unpack aux_inputs before hooks and graph execution.
            signature = forward._layer_input_signature
            explicit = signature.bind_partial(
                *args, **{name: value for name, value in kwargs.items() if name != input_key}
            ).arguments
            for parameter in signature.parameters.values():
                if parameter.kind == Parameter.VAR_KEYWORD:
                    explicit.update(explicit.pop(parameter.name, {}))
            _check_layer_input_collisions(aux_inputs, explicit)
            if not allow_aux_keyword:
                return None
            hidden_states = explicit.pop(next(iter(signature.parameters)))
            return hidden_states, aux_inputs.update(**explicit)
    elif not isinstance(aux_inputs, AuxInputs):
        raise TypeError("inputs must be an AuxInputs instance")

    overrides = kwargs.copy()
    overrides.pop(input_key)
    hidden_name = next(iter(forward._layer_input_signature.parameters))
    hidden_names = overrides.keys() & {"hidden_states", "x", hidden_name}
    if len(args) == 1 and not hidden_names:
        hidden_states = args[0]
    elif not args and len(hidden_names) == 1:
        hidden_states = overrides.pop(hidden_names.pop())
    else:
        raise TypeError("Pass one hidden-state tensor and AuxInputs, followed by keyword arguments")
    if overrides.keys() & {"aux_inputs", "inputs"}:
        raise TypeError("Pass AuxInputs only once")
    return hidden_states, aux_inputs.update(**overrides) if overrides else aux_inputs


def _call_with_aux_inputs(module, call, hidden_states, aux_inputs, *, forward=None):
    """Unpack shared inputs, run the layer, and bind its output if requested."""
    forward = module.forward if forward is None else forward
    args, kwargs = module.bind_layer_inputs(hidden_states, aux_inputs, forward=forward)
    result = call(*args, **kwargs)
    return module.bind_layer_outputs(result) if _uses_layer_outputs(module, forward) else result


def _uses_layer_outputs(module, forward):
    """Only LayerInterface subclasses bind outputs; input-only stages do not."""
    return LayerInterface in type(module).__mro__ and not forward._layer_input_only


@experimental_api
def bind_aux_inputs(forward):
    """Pack shared arguments for a method and generate its ``Inputs`` type.

    Works for layers and non-layer code. Return values stay unchanged unless the
    class inherits LayerInterface.
    Explicit arguments cannot repeat fields in the shared inputs. For forwarding
    methods with ``*args, **kwargs``, apply ``copy_signature`` underneath this.
    """
    return _bind_layer_method(forward, input_only=False)


@experimental_api
def bind_layer_input_stage(stage):
    """Unpack a LayerInterface stage's inputs and leave its output unchanged."""
    return _bind_layer_method(stage, input_only=True)


@experimental_api
def bind_layer_output_stage(stage):
    """Bind a LayerInterface final stage's output when called with ``aux_inputs``."""

    @wraps(stage)
    def forward(self, *args, **kwargs):
        if LayerInterface not in type(self).__mro__:
            raise TypeError("@bind_layer_output_stage requires LayerInterface inheritance")
        aux_inputs = kwargs.pop("aux_inputs", None)
        if aux_inputs is not None and not isinstance(aux_inputs, AuxInputs):
            raise TypeError("aux_inputs must be an AuxInputs instance")
        result = stage(self, *args, **kwargs)
        return self.bind_layer_outputs(result) if aux_inputs is not None else result

    return forward


def _make_layer_inputs(method, parameters):
    """Build the Inputs dataclass from the forward's named arguments."""
    input_fields = []
    for parameter in parameters:
        if parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        if parameter.name == "inputs":
            raise TypeError("A decorated forward must use explicit fields, not an inputs parameter")
        if parameter.kind == Parameter.POSITIONAL_ONLY:
            raise TypeError("AuxInputs fields must be keyword-capable parameters")
        if parameter.name == "aux_inputs":
            continue
        default = MISSING if parameter.default is Parameter.empty else parameter.default
        # Preserve even a mutable forward default's identity; dataclasses require a factory.
        spec = (
            field(default_factory=lambda value=default: value)
            if default.__class__.__hash__ is None
            else field(default=default)
        )
        annotation = Any if parameter.annotation is Parameter.empty else parameter.annotation
        input_fields.append((parameter.name, annotation, spec))

    input_type = make_dataclass(
        f"{method.__name__}Inputs",
        input_fields,
        bases=(AuxInputs,),
        namespace={
            "__module__": method.__module__,
            "__qualname__": f"{method.__qualname__}.Inputs",
        },
        frozen=True,
        kw_only=True,
        eq=False,
    )
    _register_layer_aux(input_type)
    return input_type


def _bind_legacy_layer_inputs(module, forward, args, kwargs):
    """Pack legacy keyword arguments for a forward that requires ``aux_inputs``."""
    hidden_name = next(iter(forward._layer_input_signature.parameters))
    if len(args) > 1 or "aux_inputs" in kwargs or not (args or hidden_name in kwargs):
        return args, kwargs
    if args and hidden_name in kwargs:
        raise TypeError(f"Multiple values for hidden-state argument {hidden_name!r}")
    values = kwargs.copy()
    hidden_states = args[0] if args else values.pop(hidden_name)
    if "inference_params" in values:
        values["inference_context"] = deprecate_inference_params(
            values.get("inference_context"), values.pop("inference_params")
        )
    return module.bind_layer_inputs(hidden_states, AuxInputs(**values), forward=forward)


def _bind_layer_method(method, *, input_only):
    """Save the method's signature and add support for shared inputs."""
    parameters = list(signature(method).parameters.values())[1:]
    if not parameters or parameters[0].kind not in (
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("An AuxInputs forward needs an explicit hidden-state parameter")
    required_aux_inputs = any(
        p.name == "aux_inputs" and p.default is Parameter.empty for p in parameters[1:]
    )

    @wraps(method)
    def forward(self, *args, **kwargs):
        if input_only and LayerInterface not in type(self).__mro__:
            raise TypeError("@bind_layer_input_stage requires LayerInterface inheritance")
        shared_inputs = _parse_layer_aux_inputs(args, kwargs, forward=forward)
        if shared_inputs is not None:
            return _call_with_aux_inputs(
                self, method.__get__(self), *shared_inputs, forward=forward
            )
        if required_aux_inputs:
            args, kwargs = _bind_legacy_layer_inputs(self, forward, args, kwargs)
        return method(self, *args, **kwargs)

    forward._layer_input_only = input_only
    forward.Inputs = _make_layer_inputs(method, parameters[1:])
    forward._layer_input_signature = Signature(parameters)
    return forward
