# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU checkpoint-audit tests using native Torch DCP tensors and serialized objects."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from examples.engram.check_training_state import StateComparison, compare_checkpoints


class TestCheckpointStateAudit(unittest.TestCase):
    """Prevent model-only comparisons from hiding optimizer, RNG or progress corruption."""

    def _write(
        self, directory: Path, mutation: str | None = None, current_format: bool = False
    ) -> None:
        from torch.distributed.checkpoint import FileSystemWriter, save
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner

        from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config

        directory.mkdir()
        moment = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        bucket = {
            "coordinates": torch.tensor([[1, 0, 3]]),
            "exp_avg": torch.full((1, 4), 0.25),
            "exp_avg_sq": torch.full((1, 4), 0.125),
            "master_param": torch.full((1, 4), 0.5),
            "step": torch.tensor([128]),
        }
        rng: dict[str, Any] = {
            "torch_rng_state": torch.tensor([1, 2, 3], dtype=torch.uint8),
            "np_rng_state": ("MT19937", np.arange(4, dtype=np.uint32), 1, 0, 0.0),
        }
        common: dict[str, Any] = {
            "iteration": 128,
            "args": SimpleNamespace(
                consumed_train_samples=8192,
                consumed_valid_samples=256,
                seed=2026,
                save=str(directory),
            ),
            "optimizer": {"groups": [{"lr": 8e-4, "step": 128}]},
            "opt_param_scheduler": {"num_steps": 8192, "max_lr": 8e-4},
        }
        if mutation == "moment":
            moment[0, 0] = 1.0
        elif mutation == "sparse_step":
            bucket["step"] += 1
        elif mutation == "sparse_float":
            bucket["exp_avg"][0, 0] += 1e-5
        elif mutation == "rng":
            rng["torch_rng_state"][0] = 4
        elif mutation == "samples":
            common["args"].consumed_train_samples += 64
        elif mutation == "scheduler":
            common["opt_param_scheduler"]["num_steps"] += 64
        state: dict[str, Any] = {
            "decoder.weight": torch.ones((2, 4), dtype=torch.bfloat16),
            "optimizer.state.momentum_buffer.decoder.weight": moment,
            "optimizer.state.fp32_param.decoder.weight": torch.ones((2, 4)),
            "optimizer.engram_sparse.layer_1.buckets/shard_0_256": [bucket],
            "rng_state/shard_0.0.0_1.1.1": [rng],
        }
        if current_format:
            state["common_state/shard_0_1"] = [common]
        save(
            state,
            storage_writer=FileSystemWriter(str(directory)),
            planner=DefaultSavePlanner(flatten_state_dict=False),
            no_dist=True,
        )
        if not current_format:
            torch.save(common, directory / "common.pt")
        save_config(CheckpointingConfig("torch_dist"), str(directory))

    def test_current_common_storage_compares_progress_without_run_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root / "left", current_format=True)
            self._write(root / "right", current_format=True)
            result = compare_checkpoints(root / "left", root / "right")
            self.assertTrue(result["passed"], result)

    def test_complete_state_and_individual_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root / "reference")
            for mutation in (None, "moment", "sparse_step", "rng", "samples", "scheduler"):
                path = root / str(mutation)
                self._write(path, mutation)
                result = compare_checkpoints(root / "reference", path)
                self.assertEqual(result["passed"], mutation is None, result)
                self.assertEqual(result["sparse_rows_compared"], 1)
                self.assertEqual(result["storage_entries_compared"]["rng"], 1)

    def test_tolerance_never_hides_sparse_step_or_rng_differences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root / "reference")
            for mutation in ("sparse_step", "sparse_float", "rng"):
                path = root / mutation
                self._write(path, mutation)
                result = compare_checkpoints(root / "reference", path, atol=0.01)
                self.assertEqual(result["passed"], mutation == "sparse_float", result)
                self.assertFalse(result["exact_equal"])

    def test_non_finite_values_fail_even_when_both_sides_match(self) -> None:
        comparison = StateComparison(atol=1.0)
        comparison.compare(torch.tensor([float("inf")]), torch.tensor([float("inf")]), "state")
        self.assertFalse(comparison.result()["passed"])


if __name__ == "__main__":
    unittest.main()
