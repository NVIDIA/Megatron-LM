# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit LR overrides retain tensor state without a new checkpoint format."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import match_param_groups
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _group(lr, **kwargs):
    return {
        "params": [],
        "lr": lr,
        "max_lr": lr,
        "min_lr": lr / 10,
        "wd_mult": 1.0,
        "default_config": False,
        "step": 7,
        **kwargs,
    }


def test_ordinary_resume_requires_exact_identifiers_and_retains_legacy_duplicates():
    saved = [_group(0.01, step=6), _group(0.01, step=7)]
    assert match_param_groups([_group(0.01)], saved)[0]["step"] == 7
    with pytest.raises(ValueError, match="Could not find parameter group"):
        match_param_groups([_group(0.02)], saved)


@pytest.mark.parametrize("rates", [(0.02, 0.01), (0.03, 0.03), (0.04,)])
def test_lr_override_uses_equivalent_retained_metadata_even_when_rates_cross(rates):
    saved = [_group(0.01), _group(0.02)]
    current = [_group(lr) for lr in rates]
    original = deepcopy(saved)
    result = match_param_groups(current, saved, override_lr=True)
    assert [group["max_lr"] for group in result] == list(rates)
    assert [group["step"] for group in result] == [7] * len(rates)
    assert saved == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("betas", (0.8, 0.95)),
        ("eps", 1e-6),
        ("step", 8),
        ("backend_extra", {"counter": torch.tensor([1, 2])}),
    ],
)
def test_conflicting_retained_metadata_is_rejected_even_with_an_exact_old_lr(field, value):
    saved = [_group(0.01), _group(0.02, **{field: value})]
    with pytest.raises(ValueError, match="different retained optimizer metadata"):
        match_param_groups([_group(0.02), _group(0.03)], saved, override_lr=True)


def test_aliases_counters_and_empty_group_classification():
    saved = [_group(0.01), _group(0.02)]
    for group in saved:
        group["step"] = torch.tensor(7)
        group["backend_extra"] = {"counter": torch.tensor([1, 2])}
    saved[0]["pre_max_lr"] = saved[0].pop("max_lr")
    saved[0]["pre_min_lr"] = saved[0].pop("min_lr")
    saved[1]["default_config"] = True
    current = [_group(0.03, default_config=True), _group(0.03, default_config=False)]
    result = match_param_groups(current, saved, override_lr=True)
    assert [group["default_config"] for group in result] == [True, False]
    assert all(group["max_lr"] == 0.03 and "pre_max_lr" not in group for group in result)
    assert all(group["step"].item() == 7 for group in result)


def test_structural_changes_are_not_lr_changes_and_pre_dsa_still_works():
    saved = [_group(0.01, is_dsa_indexer=False)]
    with pytest.raises(ValueError, match="Could not find parameter group"):
        match_param_groups([_group(0.02, wd_mult=0.0)], saved, override_lr=True)
    current = [_group(0.02), _group(0.02, is_dsa_indexer=True)]
    result = match_param_groups(current, saved, override_lr=True)
    assert [group["is_dsa_indexer"] for group in result] == [False, True]
    assert all(group["step"] == 7 for group in result)


def test_lr_override_ignores_conflicts_in_unused_saved_groups():
    saved = [_group(0.01), _group(0.02, wd_mult=0.0), _group(0.03, wd_mult=0.0, step=8)]
    result = match_param_groups([_group(0.04)], saved, override_lr=True)
    assert result[0]["max_lr"] == 0.04
    assert result[0]["step"] == 7


def test_lr_override_does_not_treat_an_explicit_indexer_checkpoint_as_pre_split():
    saved = [_group(0.01), _group(0.02, is_dsa_indexer=True, wd_mult=0.0)]
    with pytest.raises(ValueError, match="Could not find parameter group"):
        match_param_groups([_group(0.03, is_dsa_indexer=True)], saved, override_lr=True)


def test_lr_override_checks_conflicts_in_pre_split_source_group():
    saved = [_group(0.01), _group(0.02, step=8)]
    with pytest.raises(ValueError, match="different retained optimizer metadata"):
        match_param_groups([_group(0.03, is_dsa_indexer=True)], saved, override_lr=True)


@pytest.mark.parametrize("single", [False, True])
def test_chained_optimizer_forwards_explicit_override_without_changing_normal_calls(single):
    children = [Mock()] if single else [Mock(), Mock()]
    optimizer = ChainedOptimizer.__new__(ChainedOptimizer)
    optimizer.chained_optimizers = children
    optimizer._synchronize_steps = Mock()
    state = {} if single else {0: {}, 1: {}}
    optimizer.load_state_dict(state)
    for child in children:
        child.load_state_dict.assert_called_once_with({})
        child.load_state_dict.reset_mock()
    optimizer.load_state_dict(state, override_lr=True)
    for child in children:
        child.load_state_dict.assert_called_once_with({}, override_lr=True)


class TwoWeights(torch.nn.Module):
    """Different gradients make a parameter-state mix-up observable."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(128, 128, device="cuda", dtype=torch.bfloat16))
        self.b = torch.nn.Parameter(torch.full_like(self.a, 2.0))
        self.config = TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=1, bf16=True
        )

    def sharded_state_dict(self):
        return {
            name: ShardedTensor.from_rank_offsets(name, param, replica_id=(0, 0, Utils.rank))
            for name, param in self.named_parameters()
        }


def _build(rates):
    raw = TwoWeights()
    model = DistributedDataParallel(
        raw.config, DistributedDataParallelConfig(use_distributed_optimizer=True), raw
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            lr=0.001,
            min_lr=0.0,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=0.0,
        ),
        [model],
        config_overrides={
            ParamKey(name=(name, "*." + name)): {"max_lr": lr, "min_lr": lr / 10}
            for name, lr in zip(("a", "b"), rates)
        },
    )
    while isinstance(optimizer, ChainedOptimizer):
        assert len(optimizer.chained_optimizers) == 1
        optimizer = optimizer.chained_optimizers[0]
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=0.001,
        min_lr=0.0,
        lr_warmup_steps=2,
        lr_decay_steps=20,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=20,
        wd_incr_style="constant",
        override_opt_param_scheduler=True,
        use_checkpoint_opt_param_scheduler=False,
    )
    return raw, model, optimizer, scheduler


def _step(model, optimizer, scheduler):
    scheduler.step(increment=1)
    for i, param in enumerate(model.parameters()):
        param.main_grad.fill_(0.25 + i * 0.5)
    assert optimizer.step()[0]


@pytest.mark.parametrize("format_name", ["dp_reshardable", "fully_reshardable"])
@pytest.mark.parametrize(
    "old_rates,new_rates",
    [
        ((0.001, 0.002), (0.003, 0.001)),
        ((0.001, 0.002), (0.004, 0.004)),
        ((0.001, 0.001), (0.002, 0.003)),
    ],
)
def test_disk_resume_preserves_parameter_state_and_next_updates(format_name, old_rates, new_rates):
    Utils.initialize_model_parallel(1, 1)
    try:
        raw, model, source, scheduler = _build(old_rates)
        for _ in range(3):
            _step(model, source, scheduler)
        # Use the existing checkpoint key version; no new group schema is written.
        from megatron.core.optimizer.distrib_optimizer import _FQN_SAFE_DTYPE_KEY_VERSION

        metadata = {
            "distrib_optim_sharding_type": format_name,
            "checkpoint_version": _FQN_SAFE_DTYPE_KEY_VERSION,
        }
        with TempNamedDir("lr_override_resume", sync=True) as directory:
            model_sd = raw.sharded_state_dict()
            dist_checkpointing.save(
                {
                    "model": model_sd,
                    "optimizer": source.sharded_state_dict(model_sd, metadata=metadata),
                    "scheduler": scheduler.state_dict(),
                },
                directory,
            )
            new_raw, new_model, resumed, new_scheduler = _build(new_rates)
            new_model_sd = new_raw.sharded_state_dict()
            loaded = dist_checkpointing.load(
                {
                    "model": new_model_sd,
                    "optimizer": resumed.sharded_state_dict(
                        new_model_sd, is_loading=True, metadata=metadata
                    ),
                },
                directory,
            )
            new_raw.load_state_dict(loaded["model"])
            resumed.load_state_dict(loaded["optimizer"], override_lr=True)
            new_scheduler.load_state_dict(loaded["scheduler"])

        # Independent reference: transfer tensors by model parameter, with no group matcher.
        ref_raw, ref_model, reference, ref_scheduler = _build(new_rates)
        ref_raw.load_state_dict(raw.state_dict())
        reference.load_state_dict(reference.state_dict())
        for group in reference.param_groups:
            group["step"] = 3
        for name, param in raw.named_parameters():
            if param not in source.model_param_group_index_map:
                continue
            expected = source._get_main_param_and_optimizer_states(param)
            actual = resumed._get_main_param_and_optimizer_states(getattr(new_raw, name))
            for key in ("param", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            reference._set_main_param_and_optimizer_states(getattr(ref_raw, name), expected)
        ref_scheduler.step(increment=3)
        assert new_scheduler.num_steps == 3
        assert sorted({group["max_lr"] for group in resumed.param_groups}) == sorted(set(new_rates))
        for _ in range(2):
            _step(new_model, resumed, new_scheduler)
            _step(ref_model, reference, ref_scheduler)
            for name, param in ref_raw.named_parameters():
                torch.testing.assert_close(getattr(new_raw, name), param, rtol=0, atol=0)
                if param in reference.model_param_group_index_map:
                    expected = reference._get_main_param_and_optimizer_states(param)
                    actual = resumed._get_main_param_and_optimizer_states(getattr(new_raw, name))
                    for key in ("param", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


def test_mimo_wrapper_forwards_lr_override():
    from megatron.core.models.mimo.optimizer import MimoOptimizer

    leaf = Mock()
    wrapper = MimoOptimizer.__new__(MimoOptimizer)
    wrapper.module_infos = {"model": SimpleNamespace(is_active=True, optimizer=leaf)}
    state = {"model": {"optimizer": {"param_groups": []}}}
    wrapper.load_state_dict(state, override_lr=True)
    leaf.load_state_dict.assert_called_once_with(state["model"], override_lr=True)


@pytest.mark.parametrize("kind", ["fp32", "mixed", "layer_wise", "fsdp"])
def test_unsupported_lr_override_rejects_before_loading(kind):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer

    cls = {
        "fp32": FP32Optimizer,
        "mixed": Float16OptimizerWithFloat16Params,
        "layer_wise": LayerWiseDistributedOptimizer,
        "fsdp": DistributedOptimizer,
    }[kind]
    optimizer = cls.__new__(cls)
    if kind == "fsdp":
        optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=True)
    with pytest.raises(NotImplementedError, match="LR reconciliation"):
        optimizer.load_state_dict({}, override_lr=True)
