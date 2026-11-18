# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 consumer regions on the common tensor-state graph contract."""

import torch
from torch import nn

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.transformer.stateful_module import StatefulModule


class AttentionRegion(nn.Module):
    """Capture attention and its shared producers after eager mHC aggregation."""

    def __init__(self, layer, codec, kwargs, loss, outputs):
        super().__init__()
        self.layer, self.codec, self.kwargs, self.loss = layer, codec, kwargs, loss
        self.outputs = TensorSchema(outputs)
        self.training = layer.training

    def forward(self, hidden, values):
        """Restore only this invocation's tensor state, then publish live outputs."""
        boundary = self.layer.layer_number - 1
        csa, mhc, packed = self.codec.restore(values, boundary)
        with get_fp8_context(self.layer.config, boundary):
            output, norm_manager, residual = self.layer._forward_self_attention_output_with_bias(
                hidden_states=hidden, packed_seq_params=packed, cross_layer_state=csa, **self.kwargs
            )
            output = self.layer._group_offload_output_with_bias(
                output, norm_manager, forced_released_tensors=[residual]
            )
        if output[1] is not None:
            raise ValueError('V4.1 attention graph requires bias-free output projections')
        # mHC coefficients and tokens remain in the caller's mapping, outside capture.
        state = dict(values, mhc=mhc.pre_mix)
        return output[0], self.codec.snapshot(
            csa, mhc, state, boundary + 1, loss=self.loss, schema=self.outputs
        )


def attention_region(layer, codec, kwargs, loss):
    """Declare shared CSA2 fields without capturing eager mHC or Engram owners."""
    boundary = layer.layer_number - 1
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ('padding_mask', 'input_ids', 'packed_sequence_cp_metadata')
    }

    def fields(schema):
        return tuple(f for f in schema.fields if f.key not in ('mhc', 'tokens'))

    outputs = fields(codec.schema(boundary + 1, loss=loss))
    return StatefulModule(
        AttentionRegion(layer, codec, kwargs, loss, outputs),
        fields(codec.schema(boundary)),
        outputs,
    )


class RouterRegion(nn.Module):
    """Capture routing only; normalization, dispatch and experts stay eager."""

    def __init__(self, moe, codec):
        super().__init__()
        self.router = moe.router
        # The bound operation accesses dispatcher metadata but owns no expert parameters.
        self.compute_routing = moe.compute_routing
        self.codec = codec
        self.training = moe.training

    def forward(self, hidden, values):
        """Keep the routing map and its differentiable probabilities explicit."""
        probs, routes = self.compute_routing(
            hidden,
            packed_seq_params=self.codec.packed(values),
            padding_mask=values.get('padding_mask'),
            input_ids=values.get('input_ids'),
        )
        return hidden, dict(probs=probs, routes=routes)


def router_region(moe, codec, hidden, values, *, padding_mask=None, input_ids=None):
    """Infer the configured router's dense/compact result format without updating counts."""
    fields = tuple(f for f in codec.schema(0).fields if f.key in ('cu', 'cu_padded'))
    values = {f.key: values[f.key] for f in fields}
    for key, tensor in (('padding_mask', padding_mask), ('input_ids', input_ids)):
        if tensor is not None:
            values[key] = tensor
            fields += (TensorField(key, tuple(tensor.shape), tensor.dtype, 'contiguous', False),)
    module = RouterRegion(moe, codec)
    with torch.no_grad():
        _, sample = module(hidden, values)
    outputs = tuple(
        TensorField(name, tuple(t.shape), t.dtype, 'contiguous', name == 'probs')
        for name, t in sample.items()
    )
    return StatefulModule(module, fields, outputs), values
