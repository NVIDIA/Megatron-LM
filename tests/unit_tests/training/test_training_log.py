# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import re
import sys

import pytest
import torch

from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import destroy_global_vars, get_timers, set_global_variables
from megatron.training.training import training_log
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def logging_args(monkeypatch):
    """Initialize real logging dependencies using Megatron's defaults."""
    if not torch.cuda.is_available():
        pytest.skip("training_log uses CUDA loss accumulators")

    try:
        Utils.initialize_model_parallel()
        with monkeypatch.context() as patch:
            patch.setattr(
                sys,
                "argv",
                (
                    "test_training_log "
                    "--num-layers 1 --hidden-size 16 --num-attention-heads 1 "
                    "--seq-length 8 --max-position-embeddings 8 "
                    "--micro-batch-size 1 --train-iters 6 --lr 0.25 "
                    "--timing-log-level 0 --no-one-logger"
                ).split(),
            )
            args = parse_args()

        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        validate_args(args)

        # FLOPs reporting needs this; the scalar model needs no tokenizer.
        args.padded_vocab_size = 128
        set_global_variables(args, build_tokenizer=False)
        yield args
    finally:
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("start_iteration", [0, 6], ids=["from-zero", "offset-numbering"])
@pytest.mark.parametrize(
    "log_interval, logged_steps, expected_losses",
    [
        (1, [1, 2, 3, 4, 5, 6], [4.0, 1.0, 0.25, 0.0625, 0.015625, 0.00390625]),
        (2, [1, 2, 4, 6], [4.0, 2.5, 0.15625, 0.009765625]),
        (3, [1, 3, 6], [4.0, 1.75, 0.02734375]),
    ],
)
def test_training_log_reports_window_losses(
    logging_args, capsys, log_interval, logged_steps, expected_losses, start_iteration
):
    args = logging_args
    args.log_interval = log_interval
    args.train_iters = start_iteration + 6
    args.consumed_train_samples = start_iteration * args.global_batch_size

    # For L(w) = w^2, SGD with lr=1/4 halves w at each step.
    # Starting at w=2 gives losses: 4, 1, 1/4, 1/16, 1/64, 1/256.
    weight = torch.nn.Parameter(torch.tensor([2.0], device="cuda", dtype=torch.float32))
    optimizer = torch.optim.SGD([weight], lr=0.25)
    total_loss_dict = {}
    timer = get_timers()("interval-time", log_level=0)

    capsys.readouterr()
    timer.start()
    try:
        for step in range(1, 7):
            optimizer.zero_grad()
            loss = weight.square().sum()
            loss.backward()
            optimizer.step()
            args.consumed_train_samples += args.global_batch_size

            training_log(
                loss_dict={"lm loss": loss.detach()},
                total_loss_dict=total_loss_dict,
                learning_rate=0.25,
                iteration=start_iteration + step,
                loss_scale=1.0,
                report_memory_flag=False,
                skipped_iter=0,
                grad_norm=None,
                params_norm=None,
                num_zeros_in_grad=None,
                max_attention_logit=None,
                is_first_iteration=(step == 1),
            )
    finally:
        timer.stop()

    output = capsys.readouterr().out
    records = re.findall(r"\biteration\s+(\d+)\s*/\s*\d+[^\n]*?\blm loss:\s*([^\s|]+)", output)

    if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
        assert [int(step) for step, _ in records] == [
            start_iteration + step for step in logged_steps
        ], output
        assert [float(loss) for _, loss in records] == pytest.approx(
            expected_losses, rel=1e-6, abs=1e-8
        ), output
    else:
        assert records == [], output
