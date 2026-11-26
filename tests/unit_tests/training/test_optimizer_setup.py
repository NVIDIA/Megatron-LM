# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The public setup entry point composes finalized inputs without training globals."""

from functools import partial
from unittest.mock import Mock

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.training import global_vars, training
from megatron.training.config.optimizer_config import build_optimizer_config_overrides
from megatron.training.config.training_config import SchedulerConfig
from megatron.training.optim import build_optimizer_param_scheduler, setup_optimizer
from tests.unit_tests.test_utilities import Utils


def _scheduler_config():
    config = SchedulerConfig(lr_decay_style="linear", start_weight_decay=0.1, end_weight_decay=0.1)
    config.lr_warmup_steps = 4
    config.lr_decay_steps = config.wd_incr_steps = 20
    return config


class SetupModel(MegatronModule):
    def __init__(self):
        super().__init__(
            TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=1, bf16=True)
        )
        self.weight = torch.nn.Parameter(torch.ones(32, 32, device="cuda", dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.zeros(32, device="cuda", dtype=torch.bfloat16))


@pytest.mark.parametrize("custom_factory", [False, True])
def test_setup_constructs_distributed_adam_without_training_globals(monkeypatch, custom_factory):
    Utils.initialize_model_parallel(1, 1)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        unavailable = Mock(side_effect=AssertionError("training globals must not be read"))
        monkeypatch.setattr(global_vars, "get_args", unavailable)
        monkeypatch.setattr(global_vars, "get_run_config", unavailable)
        monkeypatch.setattr(training, "get_args", unavailable)
        monkeypatch.setattr(training, "get_run_config", unavailable)
        raw = SetupModel()
        model = [
            DistributedDataParallel(
                raw.config,
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                raw,
                pg_collection=groups,
            )
        ]
        config = OptimizerConfig(
            lr=0.01, min_lr=0.001, bf16=True, use_distributed_optimizer=True, clip_grad=0
        )
        policy = build_optimizer_config_overrides(
            config,
            model,
            {ParamKey(name=("weight", "*.weight")): {"max_lr": 0.02, "min_lr": 0.002}},
        )
        factory = (
            Mock(wraps=partial(get_megatron_optimizer, config=config)) if custom_factory else None
        )
        # An explicit domain must not silently materialize a global replacement.
        monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", unavailable)
        optimizer, scheduler = setup_optimizer(
            config,
            _scheduler_config(),
            model,
            config_overrides=policy,
            pg_collection=groups,
            optimizer_factory=factory,
        )
        if factory is not None:
            assert factory.call_args.kwargs["model_chunks"] is model
            assert factory.call_args.kwargs["config_overrides"] is policy
            assert factory.call_args.kwargs["pg_collection"] is groups
            assert factory.call_count == 1
        scheduler.step(increment=2)
        assert {group["lr"] for group in optimizer.param_groups} == {0.005, 0.01}
        for param in raw.parameters():
            param.main_grad.fill_(0.25)
        before = raw.weight.detach().clone()
        assert optimizer.step()[0]
        assert not torch.equal(before, raw.weight)
        assert scheduler.num_steps == 2
        assert optimizer.state_dict()
        unavailable.assert_not_called()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("style", ["linear", "WSD"])
def test_shared_scheduler_matches_existing_constructor_and_checkpoint_state(style):
    config = OptimizerConfig(lr=0.01, min_lr=0.001)
    schedule = _scheduler_config()
    schedule.lr_decay_style = style
    schedule.wsd_decay_steps = 8 if style == "WSD" else None
    schedule.lr_wsd_decay_style = "linear"
    schedule.step_lr_rescale_schedule = "0:0.5,8:1.0"
    schedule.step_lr_rescale_warmup_samples = 2
    actual_opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    expected_opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    actual = build_optimizer_param_scheduler(config, schedule, actual_opt)
    expected = OptimizerParamScheduler(
        expected_opt,
        0.0,
        0.01,
        0.001,
        4,
        20,
        style,
        0.1,
        0.1,
        20,
        "constant",
        use_checkpoint_opt_param_scheduler=False,
        wsd_decay_steps=schedule.wsd_decay_steps,
        lr_wsd_decay_style="linear",
        step_lr_rescale_schedule="0:0.5,8:1.0",
        step_lr_rescale_warmup_samples=2,
    )
    for increment in (2, 4, 3, 3, 8):
        actual.step(increment)
        expected.step(increment)
        assert actual_opt.param_groups[0]["lr"] == expected_opt.param_groups[0]["lr"]
        assert (
            actual_opt.param_groups[0]["weight_decay"]
            == expected_opt.param_groups[0]["weight_decay"]
        )
        assert actual.state_dict() == expected.state_dict()


def test_existing_scheduler_entry_point_remains_available(monkeypatch, run_config):
    run_config.train.train_iters = 20
    run_config.optimizer = OptimizerConfig(lr=0.01, min_lr=0.001)
    run_config.scheduler = _scheduler_config()
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    monkeypatch.setattr(training, "get_args", Mock(side_effect=AssertionError("legacy args")))
    scheduler = training.get_optimizer_param_scheduler(optimizer)
    scheduler.step(increment=2)
    assert optimizer.param_groups[0]["lr"] == 0.005


def test_resume_legacy_decoupled_groups_with_equal_lr(tmp_path_dist_ckpt):
    """Restore Adam moments written with the pre-recipe standard group policy."""
    from megatron.core import dist_checkpointing
    from megatron.core.optimizer import get_standard_config_overrides
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    Utils.initialize_model_parallel(1, 1)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = OptimizerConfig(
            lr=0.01, min_lr=0.001, decoupled_lr=0.01, bf16=True, use_distributed_optimizer=True
        )
        metadata = {"distrib_optim_sharding_type": "dp_reshardable"}

        def build(legacy):
            raw = SetupModel()
            raw.embedding = torch.nn.Parameter(torch.ones_like(raw.weight))
            raw.embedding.is_embedding_or_output_parameter = True
            ddp = DistributedDataParallel(
                raw.config,
                DistributedDataParallelConfig(use_distributed_optimizer=True),
                raw,
                pg_collection=groups,
            )
            policy = (
                get_standard_config_overrides(config)
                if legacy
                else build_optimizer_config_overrides(config, [ddp])
            )
            opt, _ = setup_optimizer(
                config, _scheduler_config(), [ddp], config_overrides=policy, pg_collection=groups
            )
            return raw, opt

        def update(raw, opt, sign):
            for group in opt.param_groups:
                group["lr"] = config.lr
            for i, param in enumerate(raw.parameters(), 1):
                param.main_grad.fill_(sign * 0.1 * i)
            assert opt.step()[0]

        original, old = build(True)
        update(original, old, 1)
        with TempNamedDir(tmp_path_dist_ckpt / "legacy-group-policy") as path:
            state = original.sharded_state_dict()
            dist_checkpointing.save(
                {"model": state, "optimizer": old.sharded_state_dict(state, metadata=metadata)},
                path,
            )
            resumed, new = build(False)
            assert len(old.param_groups) == len(new.param_groups) + 1
            state = resumed.sharded_state_dict()
            restored = dist_checkpointing.load(
                {
                    "model": state,
                    "optimizer": new.sharded_state_dict(state, metadata=metadata, is_loading=True),
                },
                path,
            )
            resumed.load_state_dict(restored["model"])
            new.load_state_dict(restored["optimizer"])
            update(original, old, -1)
            update(resumed, new, -1)
            for a, b in zip(original.parameters(), resumed.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
