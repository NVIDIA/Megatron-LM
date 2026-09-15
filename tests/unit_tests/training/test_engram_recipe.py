# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the example's schedule, budget and no-repeat data proof."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from examples.engram.audit_data import audit_training_indices
from examples.engram.check_checkpoint import is_optimizer_key
from examples.engram.check_logs import compare_records, tensorboard_records, wandb_records
from examples.engram.prepare_data import split_documents
from examples.engram.recipe import (
    PaddedEvaluationDataset,
    Schedule,
    archive_invocation_manifest,
    expected_parameters,
    recipe_manifest,
    set_mtp_weight,
)


class TestEngramRecipe(unittest.TestCase):
    """Validate experiment boundaries independently of the training implementation."""

    def test_final_metric_does_not_rewind_wandb_history(self) -> None:
        from examples.engram.train import write_final_metric

        tensorboard, wandb = Mock(), Mock()
        with (
            patch("examples.engram.train.get_tensorboard_writer", return_value=tensorboard),
            patch("examples.engram.train.get_wandb_writer", return_value=wandb),
        ):
            write_final_metric("final/valid_ce", 2.5, 36754)
        tensorboard.add_scalar.assert_called_once_with("final/valid_ce", 2.5, 36754)
        wandb.define_metric.assert_any_call("final/*", step_metric="final/checkpoint_step")
        # No step=36754 is sent to a resumed SDK whose internal step may be larger.
        wandb.log.assert_called_once_with({"final/checkpoint_step": 36754, "final/valid_ce": 2.5})

    def test_final_wandb_axis_does_not_change_training_coordinates(self) -> None:
        from wandb.proto import wandb_internal_pb2

        messages = []
        for values in (
            {"_step": 36754, "lm loss": 2.6},
            {"_step": 36755, "final/checkpoint_step": 36754, "final/valid_ce": 2.5},
            {"_step": 36756, "final/checkpoint_step": 36754, "final/test_ce": 2.4},
        ):
            record = wandb_internal_pb2.Record()
            for key, value in values.items():
                record.history.item.add(key=key, value_json=json.dumps(value))
            messages.append(record.SerializeToString())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.wandb").touch()
            with patch("wandb.sdk.internal.datastore.DataStore") as store:
                store.return_value.scan_data.side_effect = [*messages, None]
                records, metadata = wandb_records(root)
        self.assertEqual(records[("final/valid_ce", 36754)], [2.5])
        self.assertEqual(records[("final/test_ce", 36754)], [2.4])
        self.assertEqual(records[("lm loss", 36754)], [2.6])
        self.assertNotIn(("final/valid_ce", 36755), records)
        self.assertEqual(metadata["segments"][0]["last_step"], 36756)
        self.assertEqual(metadata["segments"][0]["first_training_step"], 36754)

    def test_training_provider_excludes_native_short_test_metric_collision(self) -> None:
        from examples.engram.train import Recipe

        recipe = Recipe.__new__(Recipe)
        recipe.full_eval_label = None
        recipe.audit_data = False
        train, valid, test = object(), object(), object()
        with patch(
            "pretrain_hybrid.train_valid_test_datasets_provider", return_value=(train, valid, test)
        ) as provider:
            self.assertEqual(recipe.datasets([10, 2, 2]), (train, valid, None))
        provider.assert_called_once_with([10, 2, 2])

    def test_evaluation_and_resume_preserve_original_training_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"execution": {"full_eval_label": None, "exit_interval": 130}}
            archive_invocation_manifest(root, original)
            archive_invocation_manifest(root, {"execution": {"full_eval_label": "valid"}})
            archive_invocation_manifest(root, {"execution": {"exit_interval": 131}})
            self.assertEqual(json.loads((root / "recipe.json").read_text()), original)
            self.assertEqual(len(list((root / "sessions").glob("*.json"))), 3)

    def test_tensorboard_restart_does_not_hide_duplicate_points(self) -> None:
        from tensorboard.compat.proto.event_pb2 import Event, SessionLog
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        with tempfile.TemporaryDirectory() as directory:
            writer = EventFileWriter(directory)
            summary = Summary(value=[Summary.Value(tag="lm loss", simple_value=3.0)])
            writer.add_event(Event(wall_time=1, step=2, summary=summary))
            writer.add_event(
                Event(wall_time=2, step=2, session_log=SessionLog(status=SessionLog.START))
            )
            writer.add_event(Event(wall_time=3, step=2, summary=summary))
            writer.close()
            records, _ = tensorboard_records(Path(directory))
            self.assertEqual(records[("lm loss", 2)], [3.0, 3.0])

    def test_checkpoint_comparison_excludes_chained_optimizer_tensors(self) -> None:
        for key in ("optimizer.exp_avg.weight", "chained_1.optimizer.engram_sparse.layer_1"):
            self.assertTrue(is_optimizer_key(key))
        self.assertFalse(is_optimizer_key("decoder.layers.2.engram.table.weight"))

    def test_log_audit_requires_real_validation_points_and_rejects_duplicates(self) -> None:
        values = {
            "lm loss": 3.0,
            "learning-rate": 8e-7,
            "grad-norm": 1.0,
            "recipe/update_lr": 0.0,
            "recipe/mtp_weight": 0.3,
            "recipe/phase": 0,
            "recipe/completed_samples_before_update": 0,
            "recipe/completed_tokens_before_update": 0,
        }
        records = {(key, 1): [value] for key, value in values.items()}
        self.assertTrue(compare_records(records, records, 1, 1)["passed"])
        self.assertFalse(
            compare_records(records, records, 1, 1, require_health_metrics=True)["passed"]
        )
        for metric, value in (
            ("optimizer-skipped-iterations", 0),
            ("nan-iterations", 0),
            ("tokens-per-second", 1000),
        ):
            records[(metric, 1)] = [value]
        self.assertTrue(
            compare_records(records, records, 1, 1, require_health_metrics=True)["passed"]
        )
        self.assertFalse(compare_records(records, records, 1, 1, eval_interval=1)["passed"])
        records[("lm loss validation", 1)] = [2.8]
        self.assertTrue(compare_records(records, records, 1, 1, eval_interval=1)["passed"])
        records[("lm loss", 1)].append(3.0)
        self.assertFalse(compare_records(records, records, 1, 1, eval_interval=1)["passed"])

    def test_mtp_boundary_and_resume(self) -> None:
        for completed, expected in (
            (999, 0.3),
            (1000, 0.3),
            (33078, 0.3),
            (33079, 0.15),
            (33080, 0.15),
        ):
            self.assertEqual(Schedule().mtp_weight(completed), expected)
        self.assertEqual(Schedule().phase(1000), "stable")
        with self.assertRaises(ValueError):
            Schedule().phase(-1)

    def test_each_actual_module_configuration_is_updated(self) -> None:
        configs = [SimpleNamespace(mtp_loss_scaling_factor=0.3) for _ in range(3)]
        model = SimpleNamespace(
            modules=lambda: [SimpleNamespace(config=config) for config in configs]
        )
        set_mtp_weight(model, 0.15)
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
            self.assertEqual(Schedule().mtp_weight(completed), 0.3 if completed == 33078 else 0.15)
            if completed <= 33079:
                self.assertEqual(used_lr, 8e-4)
            else:
                self.assertLess(used_lr, 8e-4)
            continuous.step(64)
            resumed.step(64)
            self.assertEqual(
                continuous.optimizer.param_groups[0]["lr"], resumed.optimizer.param_groups[0]["lr"]
            )

    def test_budgets_and_parameter_difference(self) -> None:
        baseline, engram = expected_parameters(False), expected_parameters(True)
        self.assertEqual(baseline["backbone_total"], 730309120)
        self.assertEqual(engram["backbone_total"], 730417360)
        self.assertEqual(engram["table_rows"], 3293522)
        self.assertEqual(engram["activated"] - baseline["activated"], 488576)
        self.assertEqual(recipe_manifest(False)["tokens"], recipe_manifest(True)["tokens"])
        self.assertEqual(recipe_manifest(True)["tokens"], 9634840576)

    def test_full_holdout_padding_masks_only_added_samples(self) -> None:
        class Dataset:
            index_split = "valid"

            def __len__(self) -> int:
                return 5

            def __getitem__(self, index: int | None) -> int | None:
                return index

        padded = PaddedEvaluationDataset(Dataset(), 4)
        self.assertEqual(len(padded), 8)
        self.assertEqual([padded[index] for index in range(8)], [0, 1, 2, 3, 4, None, None, None])

    def test_holdout_split_is_deterministic_disjoint_and_balanced(self) -> None:
        lengths = np.array([3, 8, 12, 7, 11, 5])
        first, second = split_documents(lengths)
        self.assertTrue(np.array_equal(first, split_documents(lengths)[0]))
        self.assertTrue(np.array_equal(np.sort(np.concatenate([first, second])), np.arange(6)))
        self.assertLessEqual(abs(lengths[first].sum() - lengths[second].sum()), lengths.max())

    def test_partial_first_sample_does_not_poison_full_eval_mask(self) -> None:
        import torch

        from examples.engram.train import FullEvaluationGPTDataset
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
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
                path_to_cache=directory,
            )
            # Each test process owns its temporary dataset and cache. Build it
            # as a standalone CPU dataset even when an earlier test initialized
            # distributed training; no collective runs inside this constructor.
            with patch("torch.distributed.is_initialized", return_value=False):
                dataset = FullEvaluationGPTDataset(
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

    def _dataset(self) -> SimpleNamespace:
        child = SimpleNamespace(
            config=SimpleNamespace(
                eod_mask_loss=False, add_extra_token_to_sequence=True, sequence_length=4
            ),
            _pad_token_id=-1,
            num_samples=None,
            document_index=np.array([1, 0]),
            indices=np.array([0, 1]),
            shuffle_index=np.arange(10)[::-1],
            sample_index=np.array(
                [[0, offset] for offset in range(0, 24, 4)]
                + [[1, offset] for offset in range(3, 20, 4)]
            ),
            dataset=SimpleNamespace(sequence_lengths=np.array([20, 21]), path_prefix="source"),
        )

        class Blend(SimpleNamespace):
            def __len__(self) -> int:
                return len(self.dataset_index)

        return Blend(
            datasets=[child],
            dataset_index=np.zeros(10, dtype=int),
            dataset_sample_index=np.arange(10),
        )

    def test_audit_proves_unique_shifted_labels(self) -> None:
        result = audit_training_indices(self._dataset(), 10)
        self.assertEqual(result["unique_valid_main_labels"], 40)
        self.assertEqual(result["coverage"], 40 / 41)

    def test_audit_rejects_repeat_and_mask_mismatch(self) -> None:
        dataset = self._dataset()
        dataset.dataset_sample_index[-1] = 0
        with self.assertRaisesRegex(ValueError, "repeat"):
            audit_training_indices(dataset, 10)
        dataset = self._dataset()
        dataset.datasets[0].config.eod_mask_loss = True
        with self.assertRaisesRegex(ValueError, "unmasked EOD"):
            audit_training_indices(dataset, 10)


if __name__ == "__main__":
    unittest.main()
