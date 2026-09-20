# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the cross-layer contract with shared memory independent of CSA2.

CPU modules use the production Hybrid/Transformer layers and checkpoint engine.
These tests cover autograd and replay contracts, not distributed or CUDA kernels.
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid import hybrid_stack_adapter as boundary_runtime
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateAdapter
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.state_boundary import (
    BoundarySchema,
    StateRegion,
    TensorField,
    TensorSchema,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules


@dataclass
class _MemoryState:
    memory: torch.Tensor | None = None
    last_layer: int = 0

    def attention_kwargs(self):
        return {"memory_state": self}

    def recompute_boundary_tensors(self):
        return () if self.memory is None else (self.memory,)


class _MemoryDeclaration:
    """Native memory fields and codec using the same contract as CSA2 and mHC."""

    context_attribute = "cross_layer_state"

    def __init__(self, config, *, layer_type_list, pp_layer_offset, **kwargs):
        self.config = config
        self.pattern, self.offset = layer_type_list, pp_layer_offset

    def initial_state(self, hidden, packed_seq_params):
        return _MemoryState()

    @staticmethod
    def _key(last_layer):
        source = 4 if last_layer >= 4 else int(last_layer > 0)
        return f"memory/shared:L{source}"

    def _field(self, hidden, last_layer):
        return TensorField(
            self._key(last_layer),
            (*hidden.shape[:2], self.config.hidden_size),
            self.config.params_dtype,
            "sbhd",
            True,
            present=last_layer > 0,
        )

    def checkpoint_region(self, start, end, hidden, state):
        readers = [self.offset + i + 1 for i in range(start, end) if self.pattern[i] != "-"]
        last_layer = readers[-1] if readers else state.last_layer
        return StateRegion(
            BoundarySchema(
                f"memory/checkpoint:{start}:{end}",
                (self._field(hidden, state.last_layer),),
                (self._field(hidden, last_layer),),
            ),
            self,
            (state.last_layer,),
            (last_layer,),
        )

    def export(self, state, fields):
        assert len(fields) == 1 and fields[0].key == self._key(state.last_layer)
        assert fields[0].present == (state.memory is not None)
        tensors = (state.memory,) if fields[0].present else ()
        TensorSchema(fields).validate(tensors)
        return tensors

    def restore(self, fields, tensors, metadata):
        TensorSchema(fields).validate(tensors)
        state = _MemoryState(tensors[0] if tensors else None, metadata[0])
        self.export(state, fields)
        return state


class _Norm(nn.LayerNorm):
    def __init__(self, config, hidden_size, eps, **kwargs):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


class _MemoryAttention(nn.Module):
    def __init__(self, config, layer_number, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, *, memory_state, **kwargs):
        assert memory_state.last_layer < self.layer_number
        if self.layer_number == 1:
            assert memory_state.memory is None
        projected = self.projection(hidden_states)
        if self.layer_number in (1, 4):
            memory_state.memory = projected.sin()
        memory_state.last_layer = self.layer_number
        return projected.tanh() + 0.2 * memory_state.memory.square(), None


class _MemoryLayer(_MemoryAttention):
    supports_cross_layer_state = True

    def forward(
        self,
        hidden_states,
        attention_mask,
        inference_context,
        packed_seq_params,
        *,
        cross_layer_state,
    ):
        # Deliberately narrow: full replay must use the same kwargs as eager.
        delta, _ = super().forward(hidden_states, memory_state=cross_layer_state)
        return hidden_states + delta


class _PlainLayer(nn.Module):
    def __init__(self, config, layer_number, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask, inference_context, packed_seq_params):
        # Unrelated layers must retain their existing interface.
        return hidden_states * 0.9


def _config(kind):
    return TransformerConfig(
        num_layers=5,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.float32,
        enable_hyper_connections=kind.startswith("mhc_"),
        mhc_single_pass=kind.startswith("mhc_single_"),
        num_residual_streams=2,
        use_fused_mhc=False,
        bias_dropout_fusion=False,
    )


def _stack(config, kind):
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    submodules = HybridStackSubmodules(
        forward_adapter=_MemoryDeclaration,
        mamba_layer=_MemoryLayer,
        mlp_layer=_PlainLayer,
        attention_layer=ModuleSpec(
            TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=_Norm,
                self_attention=_MemoryAttention,
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
    )
    # A layer that does not consume memory separates its producer and consumers.
    stack = HybridStack(
        config,
        submodules,
        layer_type_list=list(
            "*****"
            if kind.startswith("mhc_single_")
            else "*-***" if kind.endswith("attention") else "M-MMM"
        ),
        post_layer_norm=False,
        pg_collection=groups,
    )
    finalize = stack.forward_adapter.finalize_forward

    def observe_memory(output, packed_seq_params, context):
        # Expose a native side output only to test independent backward roots.
        # Model state initialization and checkpoint replay use the shared executor.
        return finalize(output, packed_seq_params, context), context.cross_layer_state.memory

    stack.forward_adapter.finalize_forward = observe_memory
    return stack


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(checkpoint_runtime, "_get_cuda_rng_state", lambda **kwargs: None)
        monkeypatch.setattr(checkpoint_runtime, "_set_cuda_rng_state", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda message: None)
        monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)


def _assert_grad(actual, expected):
    if expected is None:
        assert actual is None or torch.count_nonzero(actual) == 0
    else:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("backend", ["mcore", "te"])
@pytest.mark.parametrize("objective", ["memory", "both"])
def test_checkpoint_codec_with_arbitrary_context_attribute(monkeypatch, backend, objective):
    """Both checkpoint backends carry state with no built-in context name or snapshot API."""
    torch.manual_seed(852)
    config = replace(
        _config("attention"),
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=2,
    )
    declaration = _MemoryDeclaration(config, layer_type_list=["*"] * 5, pp_layer_offset=0)
    declaration.context_attribute = "memory_owner"
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    adapter = HybridStateAdapter(
        config,
        components=(declaration,),
        hidden_size=config.hidden_size,
        hidden_dtype=config.params_dtype,
        layer_type_list=["*"] * 5,
        pp_layer_offset=0,
        pre_process=True,
        post_process=True,
        is_mtp_layer=False,
        pg_collection=groups,
    )
    actual = nn.ModuleList(_MemoryAttention(config, i + 1) for i in range(5))
    reference = nn.ModuleList(_MemoryAttention(config, i + 1) for i in range(5))
    reference.load_state_dict(actual.state_dict())
    calls = []

    def te_checkpoint(function, distribute, rng_tracker, tp_group, *inputs):
        assert tp_group is groups.tp
        calls.append(len(inputs))
        return checkpoint_runtime.checkpoint(function, distribute, *inputs)

    if backend == "te":
        # Exercise TE dispatch with the real MCore checkpoint engine on CPU.
        config.quant_recipe = object()
        monkeypatch.setattr(boundary_runtime, "te_checkpoint", te_checkpoint)

    def forward(layer, hidden, working):
        return layer(hidden, memory_state=working.memory_owner)[0]

    forwards = []
    for length in (3, 5):
        x = torch.randn(length, 2, config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        _, _, context = adapter.prepare_forward(x, None, None)
        _, _, ref_context = adapter.prepare_forward(ref_x, None, None)
        assert set(vars(context)) == {"memory_owner"}
        hidden = boundary_runtime.checkpointed_hybrid_forward(
            config,
            actual,
            x,
            context,
            tp_group=groups.tp,
            quantization_context=lambda *_: nullcontext(),
            layer_forward=forward,
            boundary_factory=adapter.checkpoint_region,
        )
        ref_hidden = ref_x
        for layer in reference:
            ref_hidden = forward(layer, ref_hidden, ref_context)
        output = (hidden, context.memory_owner.memory)
        expected = (ref_hidden, ref_context.memory_owner.memory)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        forwards.append((x, ref_x, output, expected))
    for x, ref_x, output, expected in reversed(forwards):
        for values in (output, expected):
            selected = values if objective == "both" else values[1:]
            sum(t.square().sum() for t in selected).backward()
        _assert_grad(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_grad(parameter.grad, ref_parameter.grad)
    assert len(calls) == (6 if backend == "te" else 0)


@pytest.mark.parametrize("kind", ["attention", "custom", "mhc_custom", "mhc_single_attention"])
@pytest.mark.parametrize("method,count", [("uniform", 1), ("uniform", 2), ("block", 2)])
@pytest.mark.parametrize("objective", ["hidden", "memory", "both"])
def test_memory_full_recompute_and_outstanding_microbatches(kind, method, count, objective):
    """Owner replacement and reverse-order backward preserve every consumer's gradient."""
    torch.manual_seed(419)
    config = _config(kind)
    reference = _stack(config, kind)
    actual = _stack(
        replace(
            config,
            recompute_granularity="full",
            recompute_method=method,
            recompute_num_layers=count,
        ),
        kind,
    )
    actual.load_state_dict(reference.state_dict())
    forwards = []
    for length in (3, 5):
        x = torch.randn(length, 2, config.hidden_size, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        output, ref_output = actual(x, None), reference(ref_x, None)
        torch.testing.assert_close(output, ref_output, atol=0, rtol=0)
        forwards.append((x, ref_x, output, ref_output))
    for x, ref_x, output, ref_output in reversed(forwards):
        for values in (output, ref_output):
            selected = values if objective == "both" else (values[objective == "memory"],)
            sum(value.square().sum() for value in selected).backward()
        _assert_grad(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_grad(parameter.grad, ref_parameter.grad)


@pytest.mark.parametrize("kind", ["attention", "mhc_attention"])
def test_memory_only_backward_restores_norm_output(kind):
    """A shared-memory loss must restore the producer norm without a local output loss."""
    torch.manual_seed(527)
    config = _config(kind)
    reference = _stack(config, kind)
    actual = _stack(
        replace(config, recompute_granularity="selective", recompute_modules=["layernorm"]), kind
    )
    actual.load_state_dict(reference.state_dict())
    x = torch.randn(3, 2, config.hidden_size, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    memory, ref_memory = actual(x, None)[1], reference(ref_x, None)[1]
    torch.testing.assert_close(memory, ref_memory, atol=0, rtol=0)
    memory.square().sum().backward()
    ref_memory.square().sum().backward()
    _assert_grad(x.grad, ref_x.grad)
    for parameter, ref_parameter in zip(actual.parameters(), reference.parameters()):
        _assert_grad(parameter.grad, ref_parameter.grad)
