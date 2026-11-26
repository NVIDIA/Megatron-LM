# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import checkpointing
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_checkpointing import (
    create_args,
    create_ckpt_load_args,
    init_model_parallel,
)
from tests.unit_tests.test_utilities import Utils, sync_checkpoint_config


@pytest.mark.parametrize("ckpt_format", ["torch_dist", "fsdp_dtensor"])
@pytest.mark.parametrize(
    "save_optim,optim_kwargs,include",
    [
        (False, None, False),
        (False, {"is_loading": False}, False),
        (False, {"is_loading": True}, True),
        (True, None, True),
    ],
)
def test_load_template_is_independent_of_future_saves(
    init_model_parallel,
    create_ckpt_load_args,
    run_config,
    ckpt_format,
    save_optim,
    optim_kwargs,
    include,
):
    run_config.checkpoint.save_optim = save_optim
    model, optimizer, scheduler = Mock(), Mock(is_stub_optimizer=False), Mock()
    result = checkpointing.generate_state_dict(
        create_ckpt_load_args,
        [model],
        optimizer,
        scheduler,
        None,
        ckpt_format=ckpt_format,
        optim_sd_kwargs=optim_kwargs,
    )
    assert ("optimizer" in result) is include
    assert ("opt_param_scheduler" in result) is include
    assert optimizer.sharded_state_dict.call_count == int(include)
    assert scheduler.state_dict.call_count == int(include)
    assert run_config.checkpoint.save_optim is save_optim


class ResumeModel(MegatronModule):
    """Small replicated model with real distributed Adam shards."""

    def __init__(self) -> None:
        super().__init__(
            TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=1, bf16=True)
        )
        self.weight = torch.nn.Parameter(torch.ones(32, 32, device="cuda", dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.zeros(32, device="cuda", dtype=torch.bfloat16))

    def sharded_state_dict(self, **kwargs):
        return {
            name: ShardedTensor.from_rank_offsets(name, param, replica_id=(0, 0, Utils.rank))
            for name, param in self.named_parameters()
        }


def _build(config):
    raw = ResumeModel()
    model = DistributedDataParallel(
        raw.config, DistributedDataParallelConfig(use_distributed_optimizer=True), raw
    )
    optimizer = get_megatron_optimizer(config, [model])
    while isinstance(optimizer, ChainedOptimizer):
        assert len(optimizer.chained_optimizers) == 1
        optimizer = optimizer.chained_optimizers[0]
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=0.01,
        min_lr=0.001,
        lr_warmup_steps=0,
        lr_decay_steps=20,
        lr_decay_style="linear",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=20,
        wd_incr_style="constant",
    )
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler):
    for index, parameter in enumerate(model.parameters()):
        parameter.main_grad.fill_(0.25 * (index + 1))
    assert optimizer.step()[0]
    scheduler.step(increment=1)


def _assert_restored(reference, resumed):
    model, optimizer, scheduler = reference
    restored_model, restored_optimizer, restored_scheduler = resumed
    torch.testing.assert_close(model.state_dict(), restored_model.state_dict(), rtol=0, atol=0)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
    assert [g["lr"] for g in optimizer.param_groups] == [
        g["lr"] for g in restored_optimizer.param_groups
    ]
    for group, restored_group in zip(optimizer.param_groups, restored_optimizer.param_groups):
        assert group["step"] == restored_group["step"]
    for parameter, restored_parameter in zip(model.parameters(), restored_model.parameters()):
        if parameter in optimizer.model_param_group_index_map:
            torch.testing.assert_close(
                optimizer._get_main_param_and_optimizer_states(parameter),
                restored_optimizer._get_main_param_and_optimizer_states(restored_parameter),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "ckpt_format,fully_reshardable", [("torch", False), ("torch_dist", False), ("torch_dist", True)]
)
@pytest.mark.parametrize("load_policy", ["resume", "no_load_optim", "finetune", "model_only"])
def test_resume_adam_without_saving_future_optimizer_state(
    init_model_parallel,
    create_ckpt_load_args,
    run_config,
    tmp_path_dist_ckpt,
    ckpt_format,
    fully_reshardable,
    load_policy,
):
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_dist_ckpt = ckpt_format == "torch_dist"
    args.use_distributed_optimizer = True
    args.fp16, args.bf16 = False, True
    args.no_save_rng = args.no_load_rng = True
    args.save_tokenizer_assets = False
    args.dist_ckpt_optim_fully_reshardable = fully_reshardable
    run_config.optimizer = OptimizerConfig(
        lr=0.01,
        min_lr=0.001,
        weight_decay=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        clip_grad=0.0,
    )
    reference = _build(run_config.optimizer)
    run_config.model.transformer = reference[0].module.config
    for _ in range(3):
        _step(*reference)

    with TempNamedDir(
        tmp_path_dist_ckpt / "resume_without_optimizer_saves", sync=True
    ) as directory:
        args.save = args.load = str(directory)
        set_args(args)
        sync_checkpoint_config(args)
        checkpointing.save_checkpoint(3, [reference[0]], reference[1], reference[2], 99)

        resumed = _build(run_config.optimizer)
        args.no_save_optim = True
        sync_checkpoint_config(args)
        with (
            patch.object(
                resumed[1], "load_state_dict", wraps=resumed[1].load_state_dict
            ) as load_optimizer,
            patch.object(
                resumed[2], "load_state_dict", wraps=resumed[2].load_state_dict
            ) as load_scheduler,
        ):
            iteration, flops = checkpointing.load_checkpoint(
                [resumed[0]],
                resumed[1],
                resumed[2],
                restore_training_state=load_policy != "model_only",
                load_optim=load_policy != "no_load_optim",
                finetune=load_policy == "finetune",
            )
        if load_policy != "resume":
            load_optimizer.assert_not_called()
            load_scheduler.assert_not_called()
            assert resumed[2].num_steps == 0
            assert iteration == (0 if load_policy == "finetune" else 3)
            torch.testing.assert_close(
                reference[0].state_dict(), resumed[0].state_dict(), rtol=0, atol=0
            )
            return
        assert (iteration, flops) == (3, 99)
        _assert_restored(reference, resumed)
        for _ in range(2):
            _step(*reference)
            _step(*resumed)
            _assert_restored(reference, resumed)

        checkpointing.save_checkpoint(5, [resumed[0]], resumed[1], resumed[2], 100)
        output = directory / "iter_0000005"
        if ckpt_format == "torch_dist":
            saved = dist_checkpointing.load_common_state_dict(output)
            assert not any(
                "optimizer" in key for key in dist_checkpointing.load_tensors_metadata(output)
            )
        else:
            saved = torch.load(output / "mp_rank_00" / "model_optim_rng.pt", weights_only=False)
            assert not list(output.rglob("distrib_optim.pt"))
        assert "optimizer" not in saved
        assert "opt_param_scheduler" not in saved
        assert run_config.checkpoint.save_optim is False
