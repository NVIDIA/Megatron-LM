# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the cross-layer contract with shared memory independent of CSA2.

CPU modules use the production Hybrid/Transformer layers and checkpoint engine.
These tests cover autograd and replay contracts, not distributed or CUDA kernels.
"""

from dataclasses import dataclass, replace

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import random as checkpoint_runtime
from megatron.core.transformer.spec_utils import ModuleSpec
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

    def save_for_recompute(self):
        # Capture only immutable metadata, never this mutable state instance.
        last_layer = self.last_layer

        def restore(tensors):
            return _MemoryState(tensors[0], last_layer)

        return (self.memory,), restore


class _MemoryAdapter:
    consumes_input_tensor = False

    def __init__(self, config, **kwargs):
        pass

    def configure_cuda_graphs(self, layers):
        pass

    def configure_distributed_pipeline(self, pattern, pp_group, vp_stage=None):
        assert pp_group.size() == 1

    def pipeline_payload_spec(self, *args, **kwargs):
        return None, None

    def validate_input(self, input_tensor):
        pass

    def prepare_forward(self, hidden_states, packed_seq_params, inference_context):
        return hidden_states, packed_seq_params, HybridStackForwardContext(_MemoryState())

    def finalize_forward(self, output, packed_seq_params, context):
        return output, context.cross_layer_state.memory


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
        num_residual_streams=2,
        use_fused_mhc=False,
        bias_dropout_fusion=False,
    )


def _stack(config, kind):
    groups = ProcessGroupCollection()
    groups.tp = groups.pp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    submodules = HybridStackSubmodules(
        forward_adapter=_MemoryAdapter,
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
    return HybridStack(
        config,
        submodules,
        layer_type_list=list("*-***" if kind.endswith("attention") else "M-MMM"),
        post_layer_norm=False,
        pg_collection=groups,
    )


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


@pytest.mark.parametrize("kind", ["attention", "custom", "mhc_custom"])
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
