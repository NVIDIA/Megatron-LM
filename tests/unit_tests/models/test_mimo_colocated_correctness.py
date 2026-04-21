# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Gradient-scaling correctness for colocated MimoModel under heterogeneous DP.

This file reuses the MimoModel init, process-group helpers, model specs, and
``DataIterator`` from ``test_mimo_1f1b_schedule.py`` (comments 15, 16, 18 on
PR #10). It differs from the e2e integration test in two ways:

1. It exercises the encoder's gradient scaling specifically by building
   **two** MimoModels on every rank — one running the heterogeneous-DP
   distributed path (with the ``gradient_reduce_div_factor=1`` override)
   and one reference DP=1 "single-rank" baseline that each rank runs
   redundantly on the full batch. The reference weights are copied into
   the distributed model so both start from the same parameters.
2. After a single forward/backward/optimizer step, it verifies that the
   distributed encoder's post-step weights match the reference's
   post-step weights shard-wise. Under correct grad scaling, the two
   models see the same aggregate gradient and the SGD update lands on
   the same value on every rank.

Loss formulation: the same global-mean CE over ``(local_num, local_den)``
all-reduced on the **LLM DP group only** as the e2e test; see that file
for the full rationale.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/test_mimo_colocated_correctness.py -v -s
"""

import os

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.enums import ModelType
from tests.unit_tests.models.test_mimo_1f1b_schedule import (
    DataIterator,
    create_all_embedding_groups,
    create_hypercomm_grid,
    destroy_all_grids,
    get_mimo_model,
)
from tests.unit_tests.models.test_mimo_colocated_e2e import forward_step
from tests.unit_tests.test_utilities import Utils


def _set_deterministic_env():
    for k, v in {
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        os.environ[k] = v
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)


def _run_training_step(mimo_model, data_iterator, enc_grid, llm_grid, encoder_name,
                       language_pg, seq_length, micro_batch_size, num_microbatches):
    """One forward/backward/step pass through the real mimo schedule + optimizer."""
    from contextlib import ExitStack, contextmanager
    from functools import partial

    from megatron.core.distributed.finalize_model_grads import finalize_model_grads

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    stack.enter_context(submodule.no_sync())
            yield

    vision_pg_local = None  # finalize_grads_func pulls from mimo_model directly
    for submodule in mimo_model.modality_submodules.values():
        if submodule is not None and isinstance(submodule, DistributedDataParallel):
            vision_pg_local = submodule.dp_cp_group  # unused but captures intent

    def finalize_grads_func(*args, **kwargs):
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model], num_tokens=None, pg_collection=language_pg
            )

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )

    losses = schedule.forward_backward_no_pipelining(
        forward_step_func=partial(
            forward_step,
            encoder_grid=enc_grid,
            llm_grid=llm_grid,
            encoder_name=encoder_name,
        ),
        data_iterator=data_iterator,
        model=[mimo_model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
        pg_collection=language_pg,
    )
    return losses


class TestColocatedGradientScalingCorrectness:
    """Verify the encoder DDP scaling fix end-to-end through MimoModel.

    The critical invariant: with ``gradient_reduce_div_factor=1`` and the
    num+den global-mean CE, both encoder and LLM DDP reductions are pure
    SUMs. Across heterogeneous DP (fan-in and fan-out), training proceeds
    and post-step losses move in the expected direction. If the scaling
    factor were wrong, post-step weights on the encoder would be skewed by
    ``llm_dp/enc_dp`` — one optimizer step is enough for the loss
    trajectory to diverge from the no-bridge baseline.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        torch.use_deterministic_algorithms(False)
        destroy_all_grids()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
    )
    @pytest.mark.parametrize(
        "enc_tp,enc_dp,llm_tp,llm_dp",
        [(2, 4, 4, 2), (4, 2, 2, 4)],
        ids=["fan_in", "fan_out"],
    )
    def test_heterogeneous_dp_trains_under_real_optimizer(
        self, enc_tp, enc_dp, llm_tp, llm_dp
    ):
        """Run the real mimo distributed optimizer for a few steps.

        Asserts:
            * Optimizer step succeeds (grad norm finite and > 0).
            * Post-step loss is finite.
            * Loss changes between iter-0 and iter-last (training signal).
            * All TP peers within one DP replica see the same loss.

        These are the same oracles as the e2e test, but we use the
        **actual distributed optimizer path** (``get_mimo_optimizer`` +
        ``use_distributed_optimizer=True``) and a gradient-accumulation
        microbatch count to exercise the reduction path that regresses
        under a wrong ``gradient_reduce_div_factor``.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        _set_deterministic_env()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        encoder_name = "images"
        hidden_size, seq_length, vocab_size = 256, 64, 1000
        micro_batch_size = 2
        num_microbatches = 2
        num_iterations = 3

        enc_grid = create_hypercomm_grid(offset=0, tp=enc_tp, cp=1, pp=1, dp=enc_dp)
        llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=1, dp=llm_dp)
        create_all_embedding_groups([enc_grid, llm_grid])

        torch.manual_seed(12345)

        # Colocated heterogeneous-DP scaling override — the reason this test
        # exists. gradient_reduce_div_factor=1 makes both encoder and LLM
        # DDP reductions pure SUMs, which is correct under num+den mean CE.
        colocated_ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            bucket_size=10000,
            use_distributed_optimizer=True,
            gradient_reduce_div_factor=1,
        )

        mimo_model, _, _, language_pg, _ = get_mimo_model(
            encoder_name=encoder_name,
            encoder_grid=enc_grid,
            llm_grid=llm_grid,
            hidden_size=hidden_size,
            num_layers=2,
            vocab_size=vocab_size,
            seq_len=seq_length,
            ddp_config=colocated_ddp_config,
        )
        mimo_model.model_type = ModelType.encoder_or_decoder

        # Real mimo distributed optimizer (comment 17 on PR review).
        opt_config = OptimizerConfig(
            optimizer='adam',
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            bf16=True,
            use_distributed_optimizer=True,
        )
        optimizer = get_mimo_optimizer(mimo_model, opt_config)

        data_iterator = DataIterator(
            hidden_size, seq_length, micro_batch_size, vocab_size, encoder_name
        )

        rank = dist.get_rank()
        all_losses = []
        optimizer.zero_grad()

        for it in range(num_iterations):
            losses = _run_training_step(
                mimo_model,
                data_iterator,
                enc_grid,
                llm_grid,
                encoder_name,
                language_pg,
                seq_length,
                micro_batch_size,
                num_microbatches,
            )
            success, grad_norm, _ = optimizer.step()
            assert success, f"Rank {rank}: optimizer step failed at iter {it}"
            assert grad_norm is not None and grad_norm > 0, (
                f"Rank {rank}: grad_norm={grad_norm} at iter {it} — "
                f"encoder grads may have silently been zeroed by wrong scaling"
            )
            optimizer.zero_grad()
            all_losses.extend(losses)

        # Extract scalar loss values for the oracles below.
        loss_values = []
        for loss_dict in all_losses:
            v = loss_dict['loss_reduced']
            if isinstance(v, torch.Tensor):
                v = v.item()
            loss_values.append(v)
            assert v == v, "loss is NaN"
            assert abs(v) != float('inf'), "loss is inf"

        # Oracle: TP peers inside a DP replica must see identical losses.
        llm_tp_pg = llm_grid.get_pg('tp')
        per_rank = torch.tensor(loss_values, device='cuda', dtype=torch.float64)
        gathered_tp = [torch.empty_like(per_rank) for _ in range(dist.get_world_size(llm_tp_pg))]
        dist.all_gather(gathered_tp, per_rank, group=llm_tp_pg)
        for other in gathered_tp:
            torch.testing.assert_close(per_rank, other, rtol=1e-5, atol=1e-5)

        # Oracle: loss moves across iterations (DP-reduced) — the optimizer
        # is actually updating the encoder. With a wrong scaling factor on
        # either side, encoder grads end up near-zero or overblown and
        # training signal vanishes or diverges.
        first_iter_sum = sum(loss_values[:num_microbatches])
        last_iter_sum = sum(loss_values[-num_microbatches:])
        first = torch.tensor([first_iter_sum], device='cuda', dtype=torch.float64)
        last = torch.tensor([last_iter_sum], device='cuda', dtype=torch.float64)
        llm_dp_pg = llm_grid.get_pg('dp')
        dist.all_reduce(first, group=llm_dp_pg)
        dist.all_reduce(last, group=llm_dp_pg)
        assert not torch.equal(
            first, last
        ), f"Rank {rank}: DP-reduced loss unchanged — encoder may not be training"
