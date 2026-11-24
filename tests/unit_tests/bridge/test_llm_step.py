# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The packaged text step preserves MLM's batch partitioning and masked loss."""

import pytest
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.config import DataConfig
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDatasetConfig
from megatron.training.forward_steps import llm_step
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
