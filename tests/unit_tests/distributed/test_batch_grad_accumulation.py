# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""batch_grad_accumulation: the gradients accumulated per bucket group in one launch equal the
per-parameter ones."""

import pytest
import torch

import megatron.core.distributed.param_and_grad_buffer as param_and_grad_buffer
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 64


def _build(batch_grad_accumulation, overlap_grad_reduce, use_distributed_optimizer):
    torch.manual_seed(1234)
    module = torch.nn.Sequential(
        *[torch.nn.Linear(HIDDEN, HIDDEN, dtype=torch.bfloat16, device='cuda') for _ in range(6)]
    )
    ddp_config = DistributedDataParallelConfig(
        batch_grad_accumulation=batch_grad_accumulation,
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=use_distributed_optimizer,
        bucket_size=8192,  # Several bucket groups.
    )
    return DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, module
    )


def _bucket_groups(model):
    return model.bucket_groups + model.expert_parallel_bucket_groups


def _train_batches(model, num_batches, num_microbatches):
    """The main_grads after each batch; the first batch records the grad-ready counts, from the
    second one on the reduction is dispatched from the backward hook."""
    grads = []
    for batch in range(num_batches):
        model.zero_grad_buffer()
        for microbatch in range(num_microbatches):
            # Rank-dependent inputs so that the reduction changes the gradients.
            torch.manual_seed(100 * batch + microbatch + torch.distributed.get_rank())
            inputs = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device='cuda')
            model(inputs).float().square().mean().backward()
        model.finish_grad_sync()
        torch.cuda.synchronize()
        grads.append([p.main_grad.clone() for p in model.parameters()])
    return grads


class TestBatchGradAccumulation:

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    @pytest.mark.parametrize("use_distributed_optimizer", [False, True])
    @pytest.mark.parametrize("num_microbatches", [1, 3])
    def test_gradients_match_per_parameter_accumulation(
        self, overlap_grad_reduce, use_distributed_optimizer, num_microbatches
    ):
        ref_model = _build(False, overlap_grad_reduce, use_distributed_optimizer)
        model = _build(True, overlap_grad_reduce, use_distributed_optimizer)

        ref_grads = _train_batches(ref_model, 2, num_microbatches)
        grads = _train_batches(model, 2, num_microbatches)

        for batch_ref_grads, batch_grads in zip(ref_grads, grads):
            for ref_grad, grad in zip(batch_ref_grads, batch_grads):
                torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
        for bucket_group in _bucket_groups(model):
            assert not bucket_group.deferred_grads
            assert not bucket_group.deferred_grad_params

    def test_gradients_are_deferred_until_the_reduction(self):
        model = _build(True, overlap_grad_reduce=False, use_distributed_optimizer=True)
        model.zero_grad_buffer()
        inputs = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device='cuda')
        model(inputs).float().square().mean().backward()
        torch.cuda.synchronize()

        for bucket_group in _bucket_groups(model):
            assert bucket_group.deferred_grad_params == bucket_group.params
            for param in bucket_group.params:
                assert torch.count_nonzero(param.main_grad) == 0

        model.finish_grad_sync()
        torch.cuda.synchronize()
        for bucket_group in _bucket_groups(model):
            assert not bucket_group.deferred_grad_params
            for param in bucket_group.params:
                assert torch.count_nonzero(param.main_grad) > 0

    def test_large_gradients_are_accumulated_right_away(self, monkeypatch):
        # Weights (HIDDEN * HIDDEN elements) above the threshold, biases (HIDDEN) below.
        monkeypatch.setattr(param_and_grad_buffer, "BATCH_GRAD_ACCUMULATION_MAX_NUMEL", HIDDEN + 1)
        ref_model = _build(False, overlap_grad_reduce=True, use_distributed_optimizer=True)
        model = _build(True, overlap_grad_reduce=True, use_distributed_optimizer=True)

        model.zero_grad_buffer()
        inputs = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device='cuda')
        model(inputs).float().square().mean().backward()
        for bucket_group in _bucket_groups(model):
            assert all(param.dim() == 1 for param in bucket_group.deferred_grad_params)
        model.finish_grad_sync()

        (ref_grads,) = _train_batches(ref_model, 1, 1)
        (grads,) = _train_batches(model, 1, 1)
        for ref_grad, grad in zip(ref_grads, grads):
            torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
