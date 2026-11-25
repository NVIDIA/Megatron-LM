# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small numerical and config checks for native LoRA."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.peft.lora import LoRALinearSplitQKV
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.peft import LoRA
from megatron.training.peft.checkpointing import checkpoint_kwargs
from megatron.training.peft.transform import apply_lora
from tests.unit_tests.test_utilities import Utils


def test_validation_does_not_modify_recipe():
    config = LoRA(canonical_qkv_projections=True)
    config.target_modules = ["linear_q", "linear_v"]
    before = deepcopy(config)
    config.validate()
    config.validate()
    assert config == before
    config.dropout = 1.0
    before = deepcopy(config)
    with pytest.raises(ValueError, match="dropout"):
        config.validate()
    assert config == before


def test_lora_freezes_base_and_updates_only_adapter():
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(8, 4))
    x = torch.randn(3, 8)
    expected = model(x).detach()
    base = model[0].weight.detach().clone()
    apply_lora(model, LoRA(target_modules=["0"], dim=2), pg_collection=None)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    model(x).square().sum().backward()
    assert model[0].to_wrap.weight.grad is None
    assert model[0].adapter.linear_out.weight.grad.abs().sum() > 0
    optimizer.step()
    assert torch.equal(model[0].to_wrap.weight, base)
    assert not torch.equal(model(x), expected)
    model[0].disable_adapter_layers()
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_target_mutation_and_exclusion_are_applied_at_build_time():
    model = nn.ModuleDict({"first": nn.Linear(4, 4), "second": nn.Linear(4, 4)})
    config = LoRA(target_modules=["not_present"])
    config.target_modules = ["*"]
    config.exclude_modules = ["second"]
    apply_lora(model, config, pg_collection=None)
    assert hasattr(model["first"], "adapter")
    assert isinstance(model["second"], nn.Linear)
    assert not model["second"].weight.requires_grad


def test_float32_adapters_survive_bfloat16_wrapping():
    model = nn.Sequential(nn.Linear(4, 4))
    transformer = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1, bf16=True)
    custom_finalizer = lambda *args: None
    transformer.finalize_model_grads_func = custom_finalizer
    model.config = transformer
    apply_lora(
        model, LoRA(target_modules=["0"], dim=2, lora_dtype=torch.float32), pg_collection=None
    )
    wrapped = Float16Module(transformer, model)
    assert transformer.finalize_model_grads_func is custom_finalizer
    assert wrapped.module[0].to_wrap.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in wrapped.module[0].adapter.parameters())
    output = wrapped.module(torch.randn(2, 4, dtype=torch.bfloat16))
    output.float().sum().backward()
    assert output.dtype == torch.bfloat16
    assert wrapped.module[0].adapter.linear_out.weight.grad is not None


def test_peft_config_roundtrip_and_checkpoint_selection(run_config):
    run_config.peft = LoRA(target_modules=["0"], dim=2, canonical_qkv_projections=True)
    restored = type(run_config).from_dict(run_config.to_dict())
    assert restored.peft == run_config.peft
    model = nn.Sequential(nn.Linear(4, 4))
    apply_lora(model, run_config.peft, pg_collection=None)
    select = checkpoint_kwargs(model, run_config)["adapter_key_filter"]
    selected = {name for name in model.state_dict() if select(name)}
    assert selected == {"0.adapter.linear_in.weight", "0.adapter.linear_out.weight"}


@pytest.mark.parametrize("canonical", [False, True, "query_only"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_parallel_qkv_adapter_matches_explicit_delta(canonical, tp_size):
    if Utils.world_size < tp_size:
        pytest.skip("Requires two ranks")
    Utils.initialize_model_parallel(tp_size, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        transformer = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            num_query_groups=2,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
        )
        model = nn.Module()
        model.linear_qkv = ColumnParallelLinear(
            8,
            16,
            config=transformer,
            init_method=transformer.init_method,
            bias=False,
            gather_output=False,
            tp_group=groups.tp,
        )
        x = torch.randn(2, 1, 8, device="cuda")
        model.cuda()
        baseline = model.linear_qkv(x)[0].detach()
        apply_lora(
            model,
            LoRA(
                target_modules=["linear_q" if canonical == "query_only" else "linear_qkv"],
                dim=2,
                alpha=2,
                canonical_qkv_projections=bool(canonical),
            ),
            pg_collection=groups,
        )
        model.cuda()
        wrapper = model.linear_qkv
        adapters = (
            [a for a in wrapper.adapter.values() if a is not None]
            if canonical
            else [wrapper.adapter]
        )
        with torch.no_grad():
            for index, adapter in enumerate(adapters):
                adapter.linear_in.weight.fill_(0.125)
                adapter.linear_out.weight.fill_(index + 1)
        output = wrapper(x)[0]
        if canonical:
            assert isinstance(wrapper, LoRALinearSplitQKV)
            # Two GQA groups, each ordered Q0,Q1,K,V (two channels per head).
            group_scale = (
                [1, 1, 1, 1, 0, 0, 0, 0] if canonical == "query_only" else [1, 1, 1, 1, 2, 2, 3, 3]
            )
            scale = x.new_tensor(group_scale * (2 // tp_size))
        else:
            scale = x.new_ones(16 // tp_size)
        if tp_size > 1:
            from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

            x = gather_from_sequence_parallel_region(x, group=groups.tp)
        delta = x.sum(dim=-1, keepdim=True) * 0.25 * scale
        torch.testing.assert_close(output, baseline + delta)
        output.square().mean().backward()
        assert all(adapter.linear_out.weight.grad is not None for adapter in adapters)
    finally:
        Utils.destroy_model_parallel()
