# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run main's HybridStack with real CSA2/mHC and CPU projection substitutes."""

import copy
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
from megatron.core.models.deepseek_v41.hybrid_adapter import (
    V41ForwardContext,
    deepseek_v41_stack_spec,
)
from megatron.core.models.deepseek_v41.state import BatchShape, V41StateCodec
from megatron.core.models.hybrid import hybrid_block
from megatron.core.models.hybrid.layers.utils import create_layer_config
from megatron.core.pipeline_parallel.pipeline_payload import (
    TensorStatePayload,
    backward_pipeline_payload,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as attention,
)
from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
    get_csa2_module_spec_for_backend,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import (
    CpuRoPE,
    Group,
    Linear,
    Norm,
    _setup,
)


class Backend:
    def linear(self):
        return Linear

    column_parallel_linear = linear
    row_parallel_linear = linear

    def layer_norm(self, **kwargs):
        return Norm


class CpuYarn(CpuRoPE):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_rotary_seq_len(self, inference, transformer, hidden, config, packed=None):
        return hidden.shape[0] if packed is None else packed.max_seqlen_q

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs), 1.0


class Dense(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, **kwargs):
        return self.proj(x).tanh(), None


def _config():
    cfg, _ = _setup()
    values = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init}
    values.update(
        num_layers=6,
        csa_compress_ratios=[2, 0, 2, 0, 2, 0],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 2],
        enable_mhc_connections=True,
        mhc_single_pass=True,
        mhc_num_residual_streams=2,
        output_projection_groups=1,
        output_projection_lora_rank=4,
    )
    return DeepSeekV41Config(**values)


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(attention, 'YarnRotaryEmbedding', CpuYarn)
    monkeypatch.setattr(hybrid_block, 'TENorm', Norm)
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, 'save_loss_to_tracker', lambda **kw: None)
    monkeypatch.setattr(DSAIndexerLossAutoScaler, 'main_loss_backward_scale', torch.tensor(1.0))
    attn = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=get_csa2_module_spec_for_backend(Backend()),
            self_attn_bda=get_bias_dropout_add,
        ),
    )
    dense = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=Norm, mlp=Dense, mlp_bda=get_bias_dropout_add
        ),
    )
    spec = replace(deepseek_v41_stack_spec.submodules, csa2_layer=attn, moe_layer=dense)
    g = Group()
    groups = ProcessGroupCollection(tp=g, cp=g, pp=g, dp=g, dp_cp=g, ep=g, expt_dp=g, tp_cp=g)
    return _config(), spec, groups


def _stack(cfg, spec, groups, start=0, end=6):
    return hybrid_block.HybridStack(
        cfg,
        spec,
        pre_process=start == 0,
        post_process=end == 6,
        pp_layer_offset=start,
        layer_config_list=[create_layer_config(cfg, s) for s in ('VE' * 3)[start:end]],
        pg_collection=groups,
    )


def _run(stack, x, cfg, groups, values=None):
    codec = V41StateCodec(cfg, BatchShape(x.shape[0], x.shape[1]), groups.cp)
    context = V41ForwardContext(codec=codec, values={} if values is None else values)
    _, context.mhc_state, _ = codec.restore(context.values, stack.forward_adapter.start)
    if not stack.pre_process:
        stack.set_input_tensor(x)
    out = stack(x, None, forward_context=context)
    return out, context


def test_hybrid_stack_checkpoint_and_pipeline_owner_gradients(setup):
    cfg, spec, groups = setup
    full = _stack(cfg, spec, groups)
    checkpointed = copy.deepcopy(full)
    checkpointed.config.recompute_granularity = 'full'
    # Split between Full and its future Reindex/Reuse consumers, including the mHC edge.
    stages = [_stack(cfg, spec, groups, 0, 2), _stack(cfg, spec, groups, 2, 6)]
    for stage, offset in zip(stages, (0, 2)):
        for i, layer in enumerate(stage.layers):
            layer.load_state_dict(full.layers[offset + i].state_dict())
    stages[-1].final_norm.load_state_dict(full.final_norm.state_dict())
    x = torch.randn(8, 2, cfg.hidden_size)
    expected, _ = _run(full, x.clone().requires_grad_(), cfg, groups)
    actual, _ = _run(checkpointed, x.clone().requires_grad_(), checkpointed.config, groups)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for reference, recomputed in zip(full.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(recomputed.grad, reference.grad)
    hidden, context = _run(stages[0], x.clone().requires_grad_(), cfg, groups)
    sent = context.codec.payload(hidden, context.values, 2)
    assert sent.spec == context.codec.descriptor(2)
    received = TensorStatePayload(
        tuple(
            t.detach().clone().requires_grad_(f.differentiable)
            for t, f in zip(sent.tensors, sent.spec.schema.packed_fields)
        ),
        sent.spec,
    )
    hidden, values = received.restore()
    actual, _ = _run(stages[1], hidden, cfg, groups, values)
    torch.testing.assert_close(actual, expected)
    grads = backward_pipeline_payload(received, actual.square().mean(), None)
    backward_pipeline_payload(None, sent, grads)
    for stage, offset in zip(stages, (0, 2)):
        for i, layer in enumerate(stage.layers):
            for ref, split in zip(full.layers[offset + i].parameters(), layer.parameters()):
                torch.testing.assert_close(split.grad, ref.grad)
