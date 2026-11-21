# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""mHC integration with inherited scheduling; attention/MLP and offload are inert fixtures."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.hyper_connection_transformer_layer import (
    HyperConnectionTransformerLayer,
)
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.test_utilities import Utils


class _NormContext(nullcontext):
    def group_offload(self, value, **kwargs):
        return value


def _base_init(self, config, submodules, layer_number=1, **kwargs):
    nn.Module.__init__(self)
    self.config, self.layer_number = config, layer_number
    self.hidden_dropout, self.is_moe_layer = config.hidden_dropout, False
    self.input_layernorm = nn.LayerNorm(config.hidden_size)
    self.pre_mlp_layernorm = nn.LayerNorm(config.hidden_size)
    self.recompute_input_layernorm = self.recompute_pre_mlp_layernorm = False
    self.mhc_checkpoint_input_layernorm = self.mhc_checkpoint_pre_mlp_layernorm = False
    self._input_layernorm_returns_residual = self._pre_mlp_layernorm_returns_residual = False
    self.offload_attn_norm = self.offload_mlp_norm = False
    self.off_interface = lambda enabled, value, tag: _NormContext(value)
    self.bias_dropout_add_exec_handler = torch.enable_grad
    self._legacy_forward_post_mlp = object()  # The dedicated constructor must clear this shim.


def _make_layer():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=3,
        use_fused_mhc=False,
        hidden_dropout=0.25,
        bias_dropout_fusion=False,
        cuda_graph_impl="none",
    )
    submodules = TransformerLayerSubmodules(
        self_attention_hyper_connection=HyperConnectionModule,
        mlp_hyper_connection=HyperConnectionModule,
    )
    with patch.object(TransformerLayer, "__init__", _base_init):
        layer = HyperConnectionTransformerLayer(config, submodules)
    return layer.to("cuda" if torch.cuda.is_available() else "cpu")


def test_dedicated_inherits_dispatch_and_preserves_owner_and_manager_boundaries():
    layer = _make_layer()
    for name in (
        "_get_residual_connection_for_self_attention",
        "_get_residual_connection_for_mlp",
        "_apply_self_attn_bda_step",
        "_apply_mlp_bda_step",
    ):
        assert getattr(type(layer), name) is getattr(TransformerLayer, name)
    assert layer._legacy_forward_post_mlp is None
    assert {name.split(".")[0] for name in layer.state_dict()} == {
        "input_layernorm",
        "pre_mlp_layernorm",
        "self_attention_hyper_connection",
        "mlp_hyper_connection",
    }
    manager = CheckpointWithoutOutputManager()
    with patch.object(TransformerLayer, "__call__", return_value="base") as call:
        assert layer("input", mhc_recompute_manager=manager, marker=1) == "base"
        assert layer._mhc_recompute_manager is manager
        assert call.call_args.kwargs == {"marker": 1}
        layer("input")
        assert layer._mhc_recompute_manager is None
        context = ResidualStreamRecomputeContext(manager, is_block_end=False)
        with pytest.raises(ValueError, match="context"):
            layer("input", residual_stream_recompute_context=context)
        assert call.call_count == 2


@pytest.fixture
def gpu_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize(
    "terminal,norm_policy,no_grad",
    [
        (None, "none", False),
        (False, "none", False),
        (True, "mhc", False),
        (True, "selective", False),
        (None, "none", True),
        (False, "none", True),
        (True, "none", True),
    ],
)
@torch.compiler.set_stance("force_eager")
def test_dedicated_shared_schedule_preserves_mhc_policy(
    terminal, norm_policy, no_grad, gpu_model_parallel
):
    layer = _make_layer()
    layer.mhc_checkpoint_input_layernorm = layer.mhc_checkpoint_pre_mlp_layernorm = (
        norm_policy == "mhc"
    )
    layer.recompute_input_layernorm = layer.recompute_pre_mlp_layernorm = norm_policy == "selective"
    manager = None if terminal is None else CheckpointWithoutOutputManager()
    if manager is not None:
        manager.is_last_layer_in_recompute_block = terminal
    layer._mhc_recompute_manager = manager
    # Sharing adapters must not confuse stages, and owners remain replaceable at call time.
    device = next(layer.parameters()).device
    layer.self_attention_hyper_connection = HyperConnectionModule(layer.config, 2).to(device)
    layer.residual_read_mlp = layer.residual_read_self_attn
    layer.residual_write_mlp = layer.residual_write_self_attn
    layer.config.params_dtype = torch.bfloat16  # Preserve the owner's natural FP32 read dtype.
    owners = [layer.self_attention_hyper_connection, layer.mlp_hyper_connection]
    owner_outputs, calls = [], []

    def owner_hook(module, args, output):
        assert len(output) == 3
        selected = (output[0] + 0.125, output[1], output[2] * 0.75)
        owner_outputs.append(selected)
        return selected

    for owner in owners:
        owner.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
        owner.register_forward_hook(owner_hook)
    for adapter in (layer.residual_read_self_attn, layer.residual_write_self_attn):
        adapter.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(
                (kwargs["execution_context"].owner, kwargs["recompute_context"])
            ),
            with_kwargs=True,
        )
    initial = torch.randn(2, 1, 32, device=device, requires_grad=True)
    biases = [torch.randn(8, device=device, requires_grad=True) for _ in owners]
    value = initial
    with torch.no_grad() if no_grad else nullcontext():
        for index, (read, write) in enumerate(
            (
                (layer._run_input_layernorm, layer._apply_self_attn_bda_step),
                (layer._pre_mlp_layernorm_and_residual, layer._apply_mlp_bda_step),
            )
        ):
            branch, residual, state = read(value)
            assert residual is value and state[0] is value and branch.dtype == value.dtype
            assert state[1] is owner_outputs[-1][1] and state[2] is owner_outputs[-1][2]
            value = write((branch.sin(), biases[index]), residual, state)
    assert [owner for owner, _ in calls] == [owners[0], owners[0], owners[1], owners[1]]
    if manager is not None:
        assert all(context.manager is manager for _, context in calls)
        assert [context.is_block_end for _, context in calls] == [False, False, terminal, terminal]
        expected = (2 if no_grad else 4 + 2 * (norm_policy != "none")) - int(terminal)
        assert len(manager.checkpoints) == expected
    else:
        assert all(context is None for _, context in calls)
    boundary = value.square().sum()
    if manager is not None:
        manager.discard_all_outputs_and_register_unified_recompute(boundary)
        assert bool(value.untyped_storage().nbytes()) == terminal
    boundary.backward()
    assert initial.grad is not None and all(bias.grad is not None for bias in biases)
    assert all((parameter.grad is None) == no_grad for parameter in layer.parameters())
    assert len(calls) == 4 and len(owner_outputs) == 2
