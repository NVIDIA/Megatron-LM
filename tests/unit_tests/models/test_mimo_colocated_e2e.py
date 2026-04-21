# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""End-to-end integration test for MIMO model with colocated modules (no pipeline parallelism).

Both encoder and LLM share the same ranks (offset=0) but use different TP/DP
configurations. Communication between heterogeneous TP/DP layouts is handled by
``ColocatedBridgeCommunicator``.

This file intentionally reuses the helpers, model builders, and data
iterator from ``test_mimo_1f1b_schedule.py`` so the colocated and
non-colocated tests go through the same MimoModel init path. The only
differences from the 1F1B tests are:

1. Grids are colocated (same ``rank_offset`` and ``size``), so
   ``MimoModel`` builds per-edge ``ColocatedBridgeCommunicator`` instances
   instead of pipeline bridges.
2. The DDP config sets ``gradient_reduce_div_factor=1`` so the DP
   reduction is a pure SUM on both encoder and LLM sides. Under the
   global-mean CE formulation used in ``loss_func`` below, each rank's
   per-token grad scalar is already ``1/global_den`` and any further
   division by ``dp_size`` would drop a factor.

``loss_func`` computes the exact distributed equivalent of full-batch
``F.cross_entropy(..., reduction='mean')`` by all-reducing
``(local_num, local_den)`` over the **LLM DP group only** (not dp*tp),
then dividing. Exact shard-wise gradient and weight comparisons against
a single-GPU reference live in ``test_mimo_colocated_correctness.py``.

Run with::

    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        -m pytest tests/unit_tests/models/test_mimo_colocated_e2e.py -v
"""

import logging
import os
from contextlib import ExitStack, contextmanager
from functools import partial

import pytest
import torch
import torch.distributed as dist
from packaging import version

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
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
from tests.unit_tests.test_utilities import Utils

logger = logging.getLogger(__name__)


def loss_func(loss_mask, llm_dp_pg, output_tensor):
    """Global-mean CE across the LLM DP group via all-reduced ``(num, den)``.

    ``output_tensor`` is the per-token CE from
    ``GPTModel.compute_language_model_loss`` with shape ``[b, s]`` (the
    "actual Megatron cross entropy"). Masking ignored tokens, all-reducing
    numerator and valid-token count across the LLM DP group, and dividing
    is the exact distributed equivalent of full-batch
    ``F.cross_entropy(..., reduction='mean')``.

    The all-reduce is scoped to the **LLM DP group only** — never the
    full dp*tp group. All TP ranks within a DP replica already hold the
    identical per-token loss (since the LLM's output is TP-replicated),
    so summing over TP peers would double-count.
    """
    if output_tensor is None:
        return torch.tensor(0.0, device='cuda', requires_grad=True), {'loss_reduced': 0.0}

    masked = output_tensor.float() * loss_mask.float()
    local_num = masked.sum()
    local_den = loss_mask.float().sum()
    dist.all_reduce(local_num, group=llm_dp_pg)
    dist.all_reduce(local_den, group=llm_dp_pg)
    # clamp_min(1.0) guards against the pathological "all tokens masked"
    # batch. In normal training local_den > 0 on every rank, but a CE loss
    # that divides by zero crashes the entire step; better to return 0.
    loss = local_num / local_den.clamp_min(1.0)
    return loss, {'loss_reduced': loss.detach().item()}


def forward_step(data_iterator, model, encoder_grid, llm_grid, encoder_name):
    """Forward step with data slicing for heterogeneous DP."""
    batch = next(data_iterator) if data_iterator is not None else {'input_ids': None}
    llm_dp_pg = llm_grid.get_pg("dp")

    if batch.get('input_ids') is None:
        output_tensor, loss_mask = model(**batch)
        return output_tensor, partial(loss_func, loss_mask, llm_dp_pg)

    encoder_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()

    if encoder_dp > llm_dp:
        # Fan-in: data loaded with LLM DP (larger batch per rank). Slice
        # modality_inputs for the encoder's smaller per-rank batch.
        scale = encoder_dp // llm_dp
        encoder_dp_idx = encoder_grid.get_pg("dp").rank()
        slot = encoder_dp_idx % scale

        if 'modality_inputs' in batch and batch['modality_inputs'] is not None:
            for mod_name, mod_data in batch['modality_inputs'].items():
                for enc_name, enc_data in mod_data.items():
                    for key, tensor in enc_data.items():
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            batch_size = tensor.shape[1]  # [seq, batch, hidden]
                            slice_size = batch_size // scale
                            start = slot * slice_size
                            enc_data[key] = tensor[:, start : start + slice_size, :].contiguous()

    elif llm_dp > encoder_dp:
        # Fan-out: slice LLM-side inputs for the LLM's smaller per-rank batch.
        scale = llm_dp // encoder_dp
        llm_dp_idx = llm_grid.get_pg("dp").rank()
        slot = llm_dp_idx % scale

        batch_size = batch['input_ids'].shape[0]
        slice_size = batch_size // scale
        start = slot * slice_size

        for key in ['input_ids', 'labels', 'loss_mask', 'position_ids']:
            if key in batch and batch[key] is not None:
                batch[key] = batch[key][start : start + slice_size].contiguous()

    output_tensor, loss_mask = model(**batch)
    return output_tensor, partial(loss_func, loss_mask, llm_dp_pg)


def run_colocated_test(
    encoder_tp,
    encoder_dp,
    llm_tp,
    llm_dp,
    hidden_size=256,
    num_layers=2,
    vocab_size=1000,
    seq_length=64,
    micro_batch_size=2,
    num_microbatches=2,
):
    """Run MIMO model through forward_backward_no_pipelining with colocated modules."""
    # GPTModel asserts these are unset or match the attention backend.
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)

    encoder_name = "images"

    # Both grids at offset=0 (colocated on same ranks)
    encoder_grid = create_hypercomm_grid(offset=0, tp=encoder_tp, cp=1, pp=1, dp=encoder_dp)
    llm_grid = create_hypercomm_grid(offset=0, tp=llm_tp, cp=1, pp=1, dp=llm_dp)

    # dist.new_group is a collective — create all embedding PGs up front.
    create_all_embedding_groups([encoder_grid, llm_grid])

    torch.manual_seed(12345)

    # Colocated heterogeneous-DP scaling: under the num+den global-mean CE
    # in loss_func above, the per-token grad already carries 1/global_den
    # on every rank. The DP reduction must therefore be a pure SUM on both
    # encoder and LLM sides — gradient_reduce_div_factor=1 achieves that.
    colocated_ddp_config = DistributedDataParallelConfig(
        overlap_grad_reduce=True,
        bucket_size=10000,
        use_distributed_optimizer=True,
        gradient_reduce_div_factor=1,
    )

    mimo_model, _, _, language_pg, vision_pg = get_mimo_model(
        encoder_name=encoder_name,
        encoder_grid=encoder_grid,
        llm_grid=llm_grid,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        seq_len=seq_length,
        ddp_config=colocated_ddp_config,
    )
    # forward_backward_no_pipelining keys off ``model.model_type``.
    mimo_model.model_type = ModelType.encoder_or_decoder

    # Real mimo distributed optimizer (handles per-module DP groups + global
    # grad norm). Comment 17 on the PR review: exercise this path, not a
    # hand-rolled SGD, since it's what production training uses.
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-4,
        weight_decay=0.01,
        clip_grad=1.0,
        bf16=True,
        use_distributed_optimizer=True,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for submodule in mimo_model.modality_submodules.values():
                if submodule is not None:
                    stack.enter_context(submodule.no_sync())
            yield

    def finalize_grads_func(*args, **kwargs):
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model], num_tokens=None, pg_collection=language_pg
            )
        for submodule in mimo_model.modality_submodules.values():
            if submodule is not None:
                finalize_model_grads([submodule], num_tokens=None, pg_collection=vision_pg)

    mimo_model.config.no_sync_func = no_sync_func
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    mimo_model.config.grad_scale_func = lambda loss: (
        torch.tensor(loss, dtype=torch.float32, device='cuda', requires_grad=True)
        if isinstance(loss, (int, float))
        else loss
    )

    data_iterator = DataIterator(
        hidden_size, seq_length, micro_batch_size, vocab_size, encoder_name
    )

    all_losses = []
    num_iterations = 3
    rank = dist.get_rank()
    optimizer.zero_grad()

    for iteration in range(num_iterations):
        losses = schedule.forward_backward_no_pipelining(
            forward_step_func=partial(
                forward_step,
                encoder_grid=encoder_grid,
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

        success, grad_norm, num_zeros = optimizer.step()
        assert success, f"Rank {rank}: Optimizer step failed at iteration {iteration}"
        optimizer.zero_grad()

        all_losses.extend(losses)
        logger.info(f"Rank {rank}: iter {iteration}: {len(losses)} microbatches")

    assert len(all_losses) > 0, f"Rank {rank}: Expected non-empty losses list"

    loss_values = []
    for i, loss_dict in enumerate(all_losses):
        assert 'loss_reduced' in loss_dict, f"Rank {rank}: Missing 'loss_reduced' at mb {i}"
        loss_val = loss_dict['loss_reduced']
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()
        assert loss_val == loss_val, f"Rank {rank}: Loss is NaN at mb {i}"
        assert abs(loss_val) != float('inf'), f"Rank {rank}: Loss is inf at mb {i}"
        loss_values.append(loss_val)

    assert any(v != 0.0 for v in loss_values), f"Rank {rank}: All losses are zero"

    expected_total = num_iterations * num_microbatches
    assert len(all_losses) == expected_total

    # Oracle 1: cross-rank loss consistency within the LLM DP group. All TP
    # ranks in a DP replica should see identical losses (same batch, same
    # post-sync weights). A silently wrong bridge route breaks this.
    llm_dp_pg = llm_grid.get_pg('dp')
    llm_tp_pg = llm_grid.get_pg('tp')
    per_rank = torch.tensor(loss_values, device='cuda', dtype=torch.float64)
    gathered_tp = [torch.empty_like(per_rank) for _ in range(dist.get_world_size(llm_tp_pg))]
    dist.all_gather(gathered_tp, per_rank, group=llm_tp_pg)
    for other in gathered_tp:
        torch.testing.assert_close(per_rank, other, rtol=1e-5, atol=1e-5)

    # Oracle 2: training signal — sum of losses across the full DP group
    # should change between iter-0 and iter-(N-1). If the optimizer step is
    # silently a no-op (e.g. grads landing on the wrong params), it stays flat.
    first_iter_sum = sum(loss_values[:num_microbatches])
    last_iter_sum = sum(loss_values[-num_microbatches:])
    first_iter_mean = torch.tensor([first_iter_sum], device='cuda', dtype=torch.float64)
    last_iter_mean = torch.tensor([last_iter_sum], device='cuda', dtype=torch.float64)
    dist.all_reduce(first_iter_mean, group=llm_dp_pg)
    dist.all_reduce(last_iter_mean, group=llm_dp_pg)
    assert not torch.equal(
        first_iter_mean, last_iter_mean
    ), f"Rank {rank}: DP-reduced loss unchanged across iterations"

    return all_losses


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh requires PyTorch 2.3+",
)
class TestMimoColocatedE2E:
    """MimoModel + forward_backward_no_pipelining on colocated heterogeneous DP."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_colocated_fan_in_8gpu(self):
        """Encoder TP2/DP4, LLM TP4/DP2 — fan-in case."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_colocated_test(
            encoder_tp=2,
            encoder_dp=4,
            llm_tp=4,
            llm_dp=2,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=2,
        )

    def test_colocated_fan_out_8gpu(self):
        """Encoder TP4/DP2, LLM TP2/DP4 — fan-out case."""
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_colocated_test(
            encoder_tp=4,
            encoder_dp=2,
            llm_tp=2,
            llm_dp=4,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=2,
        )

    def test_colocated_fan_in_grad_accumulation_8gpu(self):
        """Fan-in with ``num_microbatches=4`` — exercises gradient accumulation.

        Per AXIOM review: the DDP path sees ``num_microbatches`` forward/
        backward pairs before the optimizer step; if accumulation is wrong
        (e.g. grads overwritten instead of added) the training-signal oracle
        would still trigger, so running the full schedule with >1 mbs is
        specifically how we catch accumulation regressions.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")
        run_colocated_test(
            encoder_tp=2,
            encoder_dp=4,
            llm_tp=4,
            llm_dp=2,
            hidden_size=256,
            num_layers=2,
            vocab_size=1000,
            seq_length=64,
            micro_batch_size=2,
            num_microbatches=4,
        )
