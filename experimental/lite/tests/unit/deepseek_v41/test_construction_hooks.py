# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Construction hooks keep native parameters and explicit process groups."""
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model


def test_head_factory_uses_model_group_and_does_not_register_early(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        model = DeepseekV41Model.__new__(DeepseekV41Model)
        torch.nn.Module.__init__(model)
        model.ps = SimpleNamespace(tp_group=dist.group.WORLD)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            normalization="RMSNorm",
        )
        release = SimpleNamespace(hidden_size=32, vocab_size=128, rms_norm_eps=1e-6)
        norm, head = model._build_head(config, release)
        assert not list(model.named_parameters())
        assert head.tp_group is model.ps.tp_group
        assert head.weight.shape == (128, 32)
        assert norm.weight.shape == (32,)
        assert head.weight.is_embedding_or_output_parameter
        model.norm, model.head = norm, head
        assert [name for name, _ in model.named_parameters()] == ["norm.weight", "head.weight"]
    finally:
        dist.destroy_process_group()
