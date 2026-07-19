# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Manual GPU smoke for an MDA layer built through HybridModel."""

import inspect
import json

import fla.ops.mda as mda_ops
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_decay_attention import (
    MultiDecayAttention,
    MultiDecaySelfAttention,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def main() -> None:
    Utils.initialize_model_parallel(1, 1)
    original_fused = mda_ops.fused_parallel_mda_attn_integrated
    fused_calls = 0

    def tracked_fused(*args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return original_fused(*args, **kwargs)

    mda_ops.fused_parallel_mda_attn_integrated = tracked_fused
    try:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=64,
            ffn_hidden_size=128,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            multi_decay_num_channels=4,
            multi_decay_decay_generation='scaled_basis',
            multi_decay_aggregate_mode='query_mix',
            multi_decay_training_kernel='auto',
            multi_decay_qkv_bias=False,
            multi_decay_qk_norm=False,
            multi_decay_window_size=None,
            multi_decay_decay_bias=True,
            multi_decay_use_output_gate=False,
            multi_decay_use_nope=True,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=64,
            max_sequence_length=128,
            hybrid_layer_pattern='#',
            position_embedding_type='none',
        ).cuda()
        model.train()

        mda_wrappers = [
            module for module in model.modules() if isinstance(module, MultiDecaySelfAttention)
        ]
        assert len(mda_wrappers) == 1, f'expected exactly one MDA wrapper, got {len(mda_wrappers)}'
        wrapper = mda_wrappers[0]
        core = wrapper.core_attention
        assert isinstance(core, MultiDecayAttention), type(core)
        source_file = inspect.getfile(original_fused)
        assert core.num_decay_channels == 4
        assert core.decay_generation == 'scaled_basis'
        assert core.aggregate_mode == 'query_mix'
        assert core.training_kernel == 'auto'
        assert core.use_nope is True
        decay_scales = wrapper._decay_scales().detach()
        assert torch.count_nonzero(decay_scales[:, 0]).item() == 0
        assert torch.all(decay_scales[:, 1:] > 0)

        input_ids = torch.randint(0, 64, (2, 128), device='cuda')
        position_ids = torch.arange(128, device='cuda').unsqueeze(0).expand(2, -1)
        logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        loss = logits.float().square().mean()
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        tracked_weight = wrapper.f_proj.weight
        before = tracked_weight.detach().float().clone()
        loss.backward()
        assert tracked_weight.grad is not None
        assert torch.isfinite(tracked_weight.grad).all()
        assert tracked_weight.grad.float().abs().sum().item() > 0
        optimizer.step()
        changed = not torch.equal(before, tracked_weight.detach().float())
        assert changed
        assert fused_calls > 0, 'auto did not invoke the fused Multi-Decay backend'

        print(
            json.dumps(
                {
                    'status': 'pass',
                    'device': torch.cuda.get_device_name(),
                    'pattern': '#',
                    'mda_wrapper': f'{type(wrapper).__module__}.{type(wrapper).__name__}',
                    'mda_core': f'{type(core).__module__}.{type(core).__name__}',
                    'mda_source': source_file,
                    'num_decay_channels': core.num_decay_channels,
                    'use_nope': core.use_nope,
                    'first_decay_scale_max_abs': decay_scales[:, 0].abs().max().item(),
                    'fused_backend_calls': fused_calls,
                    'loss': loss.item(),
                    'mda_grad_l1': tracked_weight.grad.float().abs().sum().item(),
                    'optimizer_changed_mda_weight': changed,
                },
                sort_keys=True,
            )
        )
    finally:
        mda_ops.fused_parallel_mda_attn_integrated = original_fused
        Utils.destroy_model_parallel()


if __name__ == '__main__':
    main()
