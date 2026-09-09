# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise Engram with real Mamba, attention, MoE, and MTP layers together."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_SSM
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.engram import Engram
from megatron.core.transformer.engram.hybrid_adapter import EngramHybridProvider
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _model(enabled, backend, family):
    pattern = family + '*-*E/*E'
    config = TransformerConfig(
        num_layers=5,
        hidden_size=128,
        num_attention_heads=2,
        kv_channels=64,
        ffn_hidden_size=256,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        normalization='RMSNorm',
        activation_func=F.silu,
        gated_linear_unit=True,
        mamba_num_groups=1,
        mamba_state_dim=16,
        mamba_head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type='alltoall',
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.3,
        engram_layer_ids=[0] if enabled else None,
        engram_target_layer_indices=[1] if enabled else None,
        engram_hash_table_min_sizes=[101, 103],
        engram_embedding_dim_per_ngram=16,
        engram_num_hash_heads_per_ngram=2,
        engram_table_backend=backend,
    )
    provider = None
    if enabled:
        provider = ModuleSpec(
            module=EngramHybridProvider,
            params=dict(tokenizer_lookup=torch.arange(256), pad_id=0, hybrid_layer_pattern=pattern),
        )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        hybrid_layer_pattern=pattern,
        vocab_size=256,
        max_sequence_length=128,
        position_embedding_type='rope',
        token_context_provider_spec=provider,
    ).cuda()
    return Float16Module(config, model)


@pytest.mark.parametrize('family', ['M', 'G'])
@pytest.mark.parametrize('backend', ['local', 'row_a2a'])
def test_engram_composes_with_recurrent_attention_moe_and_mtp(monkeypatch, backend, family):
    if family == 'M' and not HAVE_MAMBA_SSM:
        pytest.skip('Mamba SSM extension is unavailable')
    if family == 'G' and not HAVE_FLA:
        pytest.skip('Flash Linear Attention extension is unavailable')
    Utils.initialize_model_parallel()
    monkeypatch.setenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')
    try:
        torch.manual_seed(2026)
        model_parallel_cuda_manual_seed(2026)
        reference = _model(False, backend, family)
        enabled = _model(True, backend, family)
        result = enabled.module.load_state_dict(reference.module.state_dict(), strict=False)
        assert not result.unexpected_keys
        assert result.missing_keys and all('.engram.' in key for key in result.missing_keys)
        assert not any(isinstance(module, Engram) for module in reference.modules())
        assert not any(isinstance(module, Engram) for module in enabled.module.mtp.modules())
        consumers = [
            index
            for index, layer in enumerate(enabled.module.decoder.layers)
            if hasattr(layer, 'engram')
        ]
        assert consumers == [1]
        memory = enabled.module.decoder.layers[1].engram.layers['0']
        # A zero memory residual gives a direct feature-off/on control through
        # actual heterogeneous kernels; memory still receives a trainable gradient.
        with torch.no_grad():
            memory.value_proj.weight.zero_()
            memory.value_proj.bias.zero_()
        calls = []
        handles = []
        for index, layer in enumerate(enabled.module.decoder.layers):

            def before(module, args, kwargs, index=index):
                calls.append((index, 'token_context' in kwargs))

            handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        tokens = ((torch.arange(128, device='cuda') * 37 + 7) % 255 + 1).view(1, -1)
        positions = torch.arange(128, device='cuda').view(1, -1)
        labels = (tokens + 17) % 256
        mask = torch.ones_like(tokens, dtype=torch.float32)
        expected = reference(tokens, positions, None, labels=labels, loss_mask=mask)
        expected.mean().backward()
        actual = enabled(tokens, positions, None, labels=labels, loss_mask=mask)
        actual.mean().backward()
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert sorted(calls) == [(0, False), (1, True), (2, False), (3, False), (4, False)]
        for model in (reference, enabled):
            grads = {
                name: parameter.grad.to_dense() if parameter.grad.is_sparse else parameter.grad
                for name, parameter in model.module.named_parameters()
                if parameter.grad is not None
            }
            assert grads and all(torch.isfinite(gradient).all() for gradient in grads.values())
            for prefix in ('decoder.layers.0.', 'decoder.layers.1.', 'decoder.layers.4.', 'mtp.'):
                assert any(
                    name.startswith(prefix) and grad.norm() > 0 for name, grad in grads.items()
                ), prefix
        assert memory.value_proj.weight.grad.norm() > 0
    finally:
        Utils.destroy_model_parallel()
