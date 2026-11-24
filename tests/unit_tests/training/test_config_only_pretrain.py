# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the recipe training path without a legacy argument namespace."""

import math

import pytest
import torch

from megatron.core import num_microbatches_calculator, rerun_state_machine
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training import global_vars
from megatron.training.api import pretrain
from megatron.training.callbacks import CallbackManager
from megatron.training.config import DataConfig, DataloaderConfig, TokenizerConfig
from megatron.training.forward_steps.llm_step import forward_step
from tests.unit_tests.test_utilities import Utils


def test_pretrain_updates_and_evaluates_without_args(run_config, monkeypatch):
    """A new mandatory args read must fail a real config-only training workflow.

    Keep get_args() real and its backing global None: patching only one imported
    alias would miss readers in other modules. Optional getattr(None, ..., default)
    fallbacks remain supported during migration. This covers the ordinary GPT
    path, not every optional training feature.
    """
    # Isolate training services from neighboring tests; monkeypatch restores them.
    for name in vars(global_vars):
        if name.startswith("_GLOBAL_"):
            monkeypatch.setattr(global_vars, name, None)
    monkeypatch.setattr(num_microbatches_calculator, "_GLOBAL_NUM_MICROBATCHES_CALCULATOR", None)
    monkeypatch.setattr(rerun_state_machine, "_GLOBAL_RERUN_STATE_MACHINE", None)

    def reject_args(*args, **kwargs):
        pytest.fail("Config-only training must not register a legacy args namespace")

    monkeypatch.setattr(global_vars, "set_args", reject_args)
    Utils.initialize_distributed()

    cfg = run_config
    cfg.model.transformer.max_seq_length_per_sample = 16
    cfg.model.transformer.cross_entropy_loss_fusion = False
    cfg.train.train_iters = 2
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = torch.distributed.get_world_size()
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.min_lr = 0.0
    cfg.scheduler.lr_decay_iters = 2
    cfg.validation.eval_iters = 1
    cfg.validation.eval_interval = 2
    cfg.logger.log_interval = 1
    cfg.logger.enable_one_logger = False
    cfg.logger.log_throughput = False
    cfg.logger.log_progress = False
    cfg.data = DataConfig(
        dataset=GPTDatasetConfig(
            random_seed=1234,
            sequence_length=16,
            split="80,10,10",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        ),
        dataset_type=MockGPTDataset,
        dataloader=DataloaderConfig(num_workers=0),
    )
    cfg.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=128)

    callbacks = CallbackManager()
    initial_weights = []
    train_losses = []
    eval_losses = []

    def before_training(context):
        assert global_vars.get_args() is None
        initial_weights.extend(p.detach().clone() for p in context.model[0].parameters())

    def after_step(context):
        assert global_vars.get_args() is None
        assert not context.skipped_iter
        assert math.isfinite(context.grad_norm) and context.grad_norm > 0
        train_losses.append(context.loss_dict["lm loss"].item())

    def after_eval(context):
        assert global_vars.get_args() is None
        eval_losses.append(context.total_loss_dict["lm loss"].item())

    callbacks.register("on_train_start", before_training)
    callbacks.register("on_train_step_end", after_step)
    callbacks.register("on_eval_end", after_eval)
    try:
        pretrain(cfg, forward_step, callbacks=callbacks)
        assert global_vars.get_args() is None
        assert len(train_losses) == 2
        assert eval_losses
        assert all(math.isfinite(loss) and loss > 0 for loss in train_losses + eval_losses)
        state = global_vars.get_train_state()
        assert state.iteration == 2
        assert state.consumed_train_samples == 2 * cfg.train.global_batch_size
        assert any(
            not torch.equal(before, after)
            for before, after in zip(
                initial_weights, callbacks.callback_context.model[0].parameters(), strict=True
            )
        )
    finally:
        Utils.destroy_model_parallel()
