# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common compute inputs and additive outputs for hybrid layer cores.

Legacy forwards remain supported during migration. The private adapter dispatches
only on the input container, never on the kind of layer or core.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
    from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext


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


Addends = tuple[Tensor, ...]


def as_addends(output: tuple[Tensor, Tensor | None]) -> Addends:
    """Preserve output/bias order without performing any arithmetic."""
    value, bias = output
    return (value,) if bias is None else (value, bias)


@runtime_checkable
class LayerInterface(Protocol):
    """A core consumes hidden-width input and returns additive tensor updates."""

    def forward(self, x: Tensor, inputs: LayerInputs) -> Addends:
        """Run compute without the residual connection or output dropout."""
        ...


def layer_inputs_forward(legacy_forward):
    """Accept the new call shape while preserving legacy positional and keyword calls."""

    @wraps(legacy_forward)
    def forward(self, *args, **kwargs):
        inputs = kwargs.pop("inputs", None)
        if inputs is None and len(args) == 2 and isinstance(args[1], LayerInputs):
            args, inputs = args[:1], args[1]
        if inputs is not None:
            if not isinstance(inputs, LayerInputs):
                raise TypeError("inputs must be a LayerInputs instance")
            if args:
                if len(args) != 1 or kwargs:
                    raise TypeError("Use layer(hidden_states, inputs) without legacy arguments")
                hidden_states = args[0]
            else:
                hidden_states = kwargs.pop("x") if "x" in kwargs else kwargs.pop("hidden_states")
                if kwargs:
                    raise TypeError("LayerInputs cannot be mixed with legacy arguments")
            return self.forward_layer(hidden_states, inputs)
        return legacy_forward(self, *args, **kwargs)

    return forward
