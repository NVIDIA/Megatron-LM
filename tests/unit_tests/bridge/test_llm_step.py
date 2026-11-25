# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The packaged text step preserves MLM's batch partitioning and masked loss."""

from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.config import DataConfig
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDatasetConfig
from megatron.training.forward_steps import llm_step
from megatron.training.models.hybrid import HybridModelConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("tp,pp,cp", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
@pytest.mark.parametrize("packed", [False, True])
def test_batch_matches_existing_mlm_partitioning(run_config, tp, pp, cp, packed):
    from pretrain_hybrid import get_batch as existing_get_batch

    Utils.initialize_model_parallel(tp, pp, context_parallel_size=cp)
    try:
        transformer = run_config.model.transformer
        transformer.tensor_model_parallel_size = tp
        transformer.pipeline_model_parallel_size = pp
        transformer.context_parallel_size = cp
        transformer.max_seq_length_per_sample = 16
        run_config.train.micro_batch_size = 1
        dataset = (SFTDatasetConfig if packed else GPTDatasetConfig)(
            random_seed=1234,
            sequence_length=16,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            create_attention_mask=False,
        )
        if packed:
            dataset.packed_sequence_length = 16
        run_config.data = DataConfig(
            dataset=dataset, dataset_type=MockSFTDataset if packed else MockGPTDataset
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()

        def iterator():
            if groups.tp.rank() != 0:
                return None  # Non-source TP ranks have no loader in MLM.
            batch = {
                "tokens": torch.arange(16).unsqueeze(0),
                "labels": torch.arange(1, 17).unsqueeze(0),
                "position_ids": torch.arange(16).unsqueeze(0),
                "loss_mask": torch.ones(1, 16),
                "attention_mask": None,
            }
            if packed:
                batch.update(
                    cu_seqlens=torch.tensor([[0, 8, 16]], dtype=torch.int32),
                    cu_seqlens_padded=torch.tensor([[0, 8, 16]], dtype=torch.int32),
                    max_seqlen=torch.tensor([8], dtype=torch.int32),
                )
            return iter([batch])

        expected = existing_get_batch(iterator(), vp_stage=None, pg_collection=groups)
        actual = llm_step.get_batch(iterator(), vp_stage=None, pg_collection=groups)
        for name, value in expected.get_batch().items():
            if value is None:
                assert actual.get_batch().get(name) is None
            else:
                torch.testing.assert_close(actual.get_batch()[name], value, rtol=0, atol=0)
        expected_packed = expected.get_packed_seq_params()
        actual_packed = actual.get_packed_seq_params()
        if packed:
            for name in ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded"):
                torch.testing.assert_close(
                    getattr(actual_packed, name), getattr(expected_packed, name), rtol=0, atol=0
                )
        else:
            assert actual_packed is None
    finally:
        Utils.destroy_model_parallel()


def test_unpacked_mrope_keeps_full_positions_with_context_parallelism(run_config):
    Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
    try:
        transformer = run_config.model.transformer
        transformer.context_parallel_size = 2
        transformer.max_seq_length_per_sample = 16
        run_config.model.position_embedding_type = "mrope"
        run_config.train.micro_batch_size = 1
        run_config.data = DataConfig(
            dataset=GPTDatasetConfig(
                random_seed=1234,
                sequence_length=16,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                create_attention_mask=False,
            ),
            dataset_type=MockGPTDataset,
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        positions = torch.arange(48, device="cuda").reshape(3, 1, 16)
        tokens = torch.arange(16, device="cuda").reshape(1, 16)
        labels = tokens + 1
        loss_mask = (tokens % 3 != 0).float()
        batch = llm_step.get_batch(
            iter([dict(tokens=tokens, labels=labels, position_ids=positions, loss_mask=loss_mask)]),
            vp_stage=None,
            pg_collection=groups,
        )
        rank = groups.cp.rank()
        indices = torch.cat(
            [torch.arange(rank * 4, (rank + 1) * 4), torch.arange((3 - rank) * 4, (4 - rank) * 4)]
        ).cuda()
        for view in batch.batches_by_layout.values():
            torch.testing.assert_close(view["position_ids"], positions, rtol=0, atol=0)
            for name, full in (("tokens", tokens), ("labels", labels), ("loss_mask", loss_mask)):
                torch.testing.assert_close(view[name], full[:, indices], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


def test_masked_loss_preserves_token_count_and_gradients(run_config):
    run_config.rerun_state_machine.check_for_nan_in_loss = False
    run_config.rerun_state_machine.check_for_spiky_loss = False
    losses = torch.tensor([[1.0, 2.0, 3.0]], device="cuda", requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 1.0]], device="cuda")
    loss, count, report = llm_step.masked_next_token_loss(mask, losses)
    assert loss.item() == 4
    assert count.item() == 2
    torch.testing.assert_close(report["lm loss"], torch.tensor([4.0, 2.0], device="cuda"))
    loss.backward()
    torch.testing.assert_close(losses.grad, mask)


@pytest.mark.parametrize("pre_process,hybrid", [(True, False), (False, False), (False, True)])
def test_forward_step_scatter_is_only_for_later_gpt_stages(
    run_config, monkeypatch, pre_process, hybrid
):
    Utils.initialize_model_parallel(2, 1)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        transformer = run_config.model.transformer
        transformer.sequence_parallel = True
        if hybrid:
            run_config.model = HybridModelConfig(transformer=transformer, vocab_size=128)
        padding = (torch.arange(16, device="cuda") >= 12).reshape(1, 16)
        cp_batch = ContextParallelBatch.from_single_layout(
            transformer.linear_cp_layout, {"padding_mask": padding}, None
        )
        monkeypatch.setattr(llm_step, "get_batch", lambda *a, **k: cp_batch)
        monkeypatch.setattr(llm_step, "get_timers", MagicMock())
        monkeypatch.setattr(llm_step, "stimer", MagicMock())
        model = MagicMock(pg_collection=groups, vp_stage=None, pre_process=pre_process)
        for schedule_plan in (False, True):
            transformer.overlap_moe_expert_parallel_comm = schedule_plan
            llm_step.forward_step(None, model, return_schedule_plan=schedule_plan)
            call = model.build_schedule_plan.call_args if schedule_plan else model.call_args
            expected = padding
            if not pre_process and not hybrid:
                expected = padding[:, groups.tp.rank() * 8 : (groups.tp.rank() + 1) * 8]
            torch.testing.assert_close(call.kwargs["padding_mask"], expected)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("scheduled", [False, True])
def test_packed_flops_count_once_across_virtual_chunks(run_config, monkeypatch, scheduled):
    from megatron.training import training

    transformer = run_config.model.transformer
    transformer.sequence_packing_scheduler = "test_scheduler" if scheduled else None
    cp_batch = ContextParallelBatch.from_single_layout(
        transformer.linear_cp_layout,
        {"cu_seqlens": torch.tensor([[0, 8, 16]], device="cuda", dtype=torch.int32)},
        None,
    )
    monkeypatch.setattr(llm_step, "get_batch", lambda *a, **k: cp_batch)
    monkeypatch.setattr(llm_step, "get_timers", MagicMock())
    monkeypatch.setattr(llm_step, "stimer", MagicMock())
    monkeypatch.setattr(training, "_seqlen_stats_in_iteration", None)
    monkeypatch.setattr(training, "_seqlen_stats_active", False)
    monkeypatch.setattr(training, "_seqlen_stats_are_global", False)
    if scheduled:
        training.set_seqlen_stats_in_iteration(32, 256)
    for vp_stage in (0, 1):
        llm_step.forward_step(None, MagicMock(pg_collection=object(), vp_stage=vp_stage))
    expected = [32, 256] if scheduled else [16, 128]
    assert training._seqlen_stats_in_iteration.tolist() == expected
    assert training._seqlen_stats_are_global is scheduled
