# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""overlap_optimizer_step_with_forward: the update the next forward issues matches the eager step."""

import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _build(overlap_optimizer_step_with_forward: bool):
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        *[torch.nn.Linear(64, 64, dtype=torch.bfloat16, device='cuda') for _ in range(6)]
    )
    # Small buckets: several bucket groups, so the prefetch and the per-bucket partials matter.
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        bucket_size=4096,
    )
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-2,
        weight_decay=0.1,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
        overlap_param_gather=True,
        overlap_optimizer_step_with_forward=overlap_optimizer_step_with_forward,
    )
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    return model, optimizer


def _train_step(model, optimizer, iteration):
    # Rank-dependent inputs so that the reduce-scatter changes the gradients.
    torch.manual_seed(1000 * iteration + torch.distributed.get_rank())
    inputs = torch.randn(8, 64, dtype=torch.bfloat16, device='cuda')
    model.zero_grad_buffer()
    optimizer.zero_grad()
    loss = model(inputs).float().square().mean()
    loss.backward()
    model.finish_grad_sync()
    _, grad_norm, _ = optimizer.step()
    return loss.detach(), grad_norm


def _sync_params(model, optimizer):
    """What the training loops do before evaluation and checkpointing."""
    optimizer.prepare_model_params_for_param_sync()
    model.start_param_sync(force_sync=True)


def _main_params(optimizer):
    return [
        shard
        for chained in optimizer.chained_optimizers
        for group in chained.shard_fp32_from_float16_groups
        for shard in group
    ]


def _assert_same_state(reference_model, reference_optimizer, model, optimizer):
    for expected, actual in zip(reference_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    for expected, actual in zip(_main_params(reference_optimizer), _main_params(optimizer)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for expected, actual in zip(reference_optimizer.param_groups, optimizer.param_groups):
        assert actual['step'] == expected['step']


class TestOverlappedOptimizerStep:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_matches_eager_step(self):
        """Same losses and gradient norms over several iterations, same params and optimizer
        state once the scheduled update is applied."""
        reference_model, reference_optimizer = _build(overlap_optimizer_step_with_forward=False)
        model, optimizer = _build(overlap_optimizer_step_with_forward=True)
        for iteration in range(4):
            expected_loss, expected_grad_norm = _train_step(
                reference_model, reference_optimizer, iteration
            )
            # The forward of this iteration applies the previous iteration's update.
            loss, grad_norm = _train_step(model, optimizer, iteration)
            torch.testing.assert_close(loss, expected_loss, rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(grad_norm, expected_grad_norm, rtol=1e-4, atol=1e-6)
        _sync_params(reference_model, reference_optimizer)
        _sync_params(model, optimizer)
        _assert_same_state(reference_model, reference_optimizer, model, optimizer)

    def test_discard_scheduled_update(self):
        """A discarded update leaves params, optimizer state and step count untouched."""
        reference_model, reference_optimizer = _build(overlap_optimizer_step_with_forward=False)
        model, optimizer = _build(overlap_optimizer_step_with_forward=True)
        _train_step(reference_model, reference_optimizer, 0)
        _train_step(model, optimizer, 0)
        _train_step(model, optimizer, 1)
        optimizer.discard_scheduled_update()
        _sync_params(reference_model, reference_optimizer)
        _sync_params(model, optimizer)
        _assert_same_state(reference_model, reference_optimizer, model, optimizer)
        # Training continues from the kept state.
        _train_step(reference_model, reference_optimizer, 2)
        _train_step(model, optimizer, 2)
        _sync_params(reference_model, reference_optimizer)
        _sync_params(model, optimizer)
        _assert_same_state(reference_model, reference_optimizer, model, optimizer)
