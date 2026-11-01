# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests that the quickstart training loop in examples/run_simple_mcore_train_loop.py trains."""

from typing import Dict, Iterator
from unittest import mock

import pytest
import torch

from examples import run_simple_mcore_train_loop as example
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


def _synthetic_batches(batch_size: int = 8) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield the same CPU batches on every rank, shaped like the example's MockGPTDataset."""
    seq_length = example._SEQUENCE_LENGTH
    generator = torch.Generator().manual_seed(0)
    attention_mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0) < 0.5
    while True:
        yield {
            "tokens": torch.randint(0, 64, (batch_size, seq_length), generator=generator),
            "labels": torch.randint(0, 64, (batch_size, seq_length), generator=generator),
            "attention_mask": attention_mask.expand(batch_size, 1, seq_length, seq_length),
            "position_ids": torch.arange(seq_length).expand(batch_size, seq_length),
            "loss_mask": torch.ones(batch_size, seq_length),
        }


def test_train_step_clears_grads_before_backward_and_steps_after_finalize(monkeypatch):
    """train_step zeroes DDP's grad buffer and the optimizer's grads before the backward pass,
    and steps the optimizer only after the gradients are finalized."""
    calls = mock.Mock()
    calls.forward_backward.return_value = {"lm loss": 1.0}
    monkeypatch.setattr(example, "finalize_model_grads", calls.finalize_model_grads)

    losses = example.train_step(calls.model, calls.optim, calls.forward_backward, iter(()))

    assert [name for name, _, _ in calls.mock_calls] == [
        "model.zero_grad_buffer",
        "optim.zero_grad",
        "forward_backward",
        "finalize_model_grads",
        "optim.step",
    ]
    assert calls.forward_backward.call_args.kwargs["model"] is calls.model
    assert calls.forward_backward.call_args.kwargs["forward_only"] is False
    calls.finalize_model_grads.assert_called_once_with([calls.model])
    assert losses == {"lm loss": 1.0}


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Megatron DDP and the FP32 optimizer need CUDA."
)
@pytest.mark.skipif(Utils.world_size % 2 != 0, reason="The example runs with TP=2.")
class TestTrainStepUpdatesWeights:
    """Two iterations of the example's loop on its TP=2 grid, data parallel over the rest."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_two_iterations_update_weights_from_fresh_grads(self, monkeypatch):
        # forward_step_func reads the module-level `device` that the example's __main__ sets.
        monkeypatch.setattr(example, "device", torch.device("cuda"), raising=False)

        gpt_model = example.model_provider().cuda()
        gpt_model = DistributedDataParallel(
            config=gpt_model.config,
            ddp_config=DistributedDataParallelConfig(
                grad_reduce_in_fp32=False,
                overlap_grad_reduce=False,
                use_distributed_optimizer=False,
            ),
            module=gpt_model,
        )
        optim = example.get_optimizer(gpt_model)
        params = dict(gpt_model.named_parameters())

        schedule = get_forward_backward_func()
        main_grad_max_at_backward_start = []

        def forward_backward_spy(**kwargs):
            main_grad_max_at_backward_start.append(
                max(param.main_grad.abs().max().item() for param in params.values())
            )
            return schedule(**kwargs)

        data_iterator = _synthetic_batches()
        for _ in range(2):
            before = {name: param.detach().clone() for name, param in params.items()}
            example.train_step(gpt_model, optim, forward_backward_spy, data_iterator)

            with_grad = [name for name, param in params.items() if param.main_grad.any()]
            assert with_grad, "the backward pass produced no gradients"
            unchanged = [name for name in with_grad if torch.equal(before[name], params[name])]
            assert unchanged == [], f"the optimizer step left these weights unchanged: {unchanged}"

        # The second iteration starts from a zeroed buffer, so it steps on its own gradients
        # instead of the sum of both iterations'.
        assert main_grad_max_at_backward_start == [0.0, 0.0]
