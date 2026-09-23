# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Public training schedule and complete holdout evaluation contracts."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from megatron.training.datasets.data_samplers import PaddedEvaluationDataset
from megatron.training.training import _update_mtp_loss_scaling_factor


def mtp_weight(completed, start=33079):
    config = SimpleNamespace(mtp_loss_scaling_factor=0.3)
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(config=config)])
    args = SimpleNamespace(
        mtp_loss_scaling_factor=0.3,
        mtp_loss_scaling_factor_decay=0.15,
        mtp_loss_scaling_factor_decay_start=start,
    )
    _update_mtp_loss_scaling_factor([model], args, completed)
    return config.mtp_loss_scaling_factor


class TestEngramRecipe(unittest.TestCase):
    """Exercise observable scheduling and evaluation boundaries."""

    def test_disabled_mtp_schedule_preserves_native_config(self) -> None:
        config = SimpleNamespace(mtp_loss_scaling_factor=0.7)
        model = SimpleNamespace(modules=lambda: [SimpleNamespace(config=config)])
        _update_mtp_loss_scaling_factor([model], SimpleNamespace(), None)
        self.assertEqual(config.mtp_loss_scaling_factor, 0.7)

    def test_mtp_boundary_and_resume(self) -> None:
        for completed, expected in (
            (999, 0.3),
            (1000, 0.3),
            (33078, 0.3),
            (33079, 0.15),
            (33080, 0.15),
        ):
            self.assertEqual(mtp_weight(completed), expected)
        self.assertEqual(mtp_weight(29, start=30), 0.3)
        self.assertEqual(mtp_weight(30, start=30), 0.15)
        self.assertEqual(mtp_weight(49, start=30), 0.15)

    def test_each_actual_module_configuration_is_updated(self) -> None:
        configs = [SimpleNamespace(mtp_loss_scaling_factor=0.3) for _ in range(3)]
        model = SimpleNamespace(
            modules=lambda: [SimpleNamespace(config=config) for config in configs]
        )
        args = SimpleNamespace(
            mtp_loss_scaling_factor=0.3,
            mtp_loss_scaling_factor_decay=0.15,
            mtp_loss_scaling_factor_decay_start=30,
        )
        _update_mtp_loss_scaling_factor([model], args, 30)
        self.assertEqual([config.mtp_loss_scaling_factor for config in configs], [0.15] * 3)

    def test_native_update_lr_and_checkpoint_boundary(self) -> None:
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        def make_scheduler() -> OptimizerParamScheduler:
            optimizer = SimpleNamespace(param_groups=[{"default_config": True}])
            return OptimizerParamScheduler(
                optimizer=optimizer,
                init_lr=0.0,
                max_lr=8e-4,
                min_lr=8e-5,
                lr_warmup_steps=1000 * 64,
                lr_decay_steps=36754 * 64,
                lr_decay_style="WSD",
                start_wd=0.1,
                end_wd=0.1,
                wd_incr_steps=36754 * 64,
                wd_incr_style="constant",
                wsd_decay_steps=3675 * 64,
                lr_wsd_decay_style="cosine",
            )

        continuous = make_scheduler()
        continuous.step(33078 * 64)
        for completed in (33078, 33079, 33080):
            resumed = make_scheduler()
            resumed.load_state_dict(continuous.state_dict())
            used_lr = continuous.optimizer.param_groups[0]["lr"]
            self.assertEqual(used_lr, resumed.optimizer.param_groups[0]["lr"])
            self.assertEqual(mtp_weight(completed), 0.3 if completed == 33078 else 0.15)
            if completed <= 33079:
                self.assertEqual(used_lr, 8e-4)
            else:
                self.assertLess(used_lr, 8e-4)
            continuous.step(64)
            resumed.step(64)
            self.assertEqual(
                continuous.optimizer.param_groups[0]["lr"], resumed.optimizer.param_groups[0]["lr"]
            )

    def test_full_holdout_padding_masks_only_added_samples(self) -> None:
        class Dataset:
            index_split = "valid"

            def __len__(self) -> int:
                return 5

            def __getitem__(self, index):
                import torch

                return {"tokens": index, "loss_mask": torch.ones(3)}

        padded = PaddedEvaluationDataset(Dataset(), 4)
        self.assertEqual(len(padded), 8)
        self.assertEqual(
            [int(padded[index]["loss_mask"].sum()) for index in range(8)], [3, 3, 3, 3, 3, 0, 0, 0]
        )
        self.assertEqual(int(padded[0]["loss_mask"].sum()), 3)

    def test_partial_first_sample_does_not_poison_full_eval_mask(self) -> None:
        import torch

        from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
        from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
        from megatron.core.datasets.utils import Split

        tokenizer = SimpleNamespace(
            unique_identifiers={"kind": "test"}, pad=None, eod=99, eos=99, vocab_size=128
        )
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "data")
            builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.uint16)
            builder.add_document(torch.arange(47), [47])
            builder.finalize(prefix + ".idx")
            config = GPTDatasetConfig(
                random_seed=2026,
                sequence_length=8,
                tokenizer=tokenizer,
                reset_position_ids=False,
                reset_attention_mask=False,
                eod_mask_loss=False,
                drop_last_partial_validation_sequence=False,
                full_validation=True,
                path_to_cache=directory,
            )
            # Each test process owns its temporary dataset and cache. Build it
            # as a standalone CPU dataset even when an earlier test initialized
            # distributed training; no collective runs inside this constructor.
            with patch("torch.distributed.is_initialized", return_value=False):
                dataset = GPTDataset(
                    IndexedDataset(prefix),
                    prefix,
                    np.array([0], dtype=np.int32),
                    None,
                    Split.valid,
                    config,
                )
            tail = int(np.flatnonzero(dataset.shuffle_index == len(dataset) - 1)[0])
            self.assertEqual(int(dataset[tail]["loss_mask"].sum()), 6)
            full = int(np.flatnonzero(dataset.shuffle_index == 0)[0])
            self.assertEqual(int(dataset[full]["loss_mask"].sum()), 8)
            self.assertEqual(
                sum(int(dataset[i]["loss_mask"].sum()) for i in range(len(dataset))), 46
            )
            self.assertIsNone(tokenizer.pad)
