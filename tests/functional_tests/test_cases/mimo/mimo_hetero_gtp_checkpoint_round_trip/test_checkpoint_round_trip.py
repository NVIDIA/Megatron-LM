# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exact checkpoint save/load round-trip for the heterogeneous MIMO 20L launcher."""

import argparse
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    DefaultLoadPlanner,
    FileSystemReader,
    TensorStorageMetadata,
)

from megatron.core.dist_checkpointing import load_common_state_dict
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

_REPO_ROOT = Path(__file__).parents[5]
_LAUNCHER = _REPO_ROOT / "examples/mimo/scripts/run_hetero_nemotron_20l_mock_train.sh"
_MODULE = (
    "tests.functional_tests.test_cases.mimo."
    "mimo_hetero_gtp_checkpoint_round_trip.test_checkpoint_round_trip"
)

# Run-local arguments vary; their authoritative top-level values are compared.
_ROUND_TRIP_ARGS = {
    "curr_iteration",
    "do_test",
    "do_train",
    "do_valid",
    "exit_on_missing_checkpoint",
    "iteration",
    "load",
    "num_floating_point_operations_so_far",
    "save",
}


def _run_launcher(
    scratch: Path, name: str, *args: str, resave_after_load: bool = False
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MIMO_CHECKPOINT_PADDING_MANIFEST_DIR": str(scratch / "padding-manifest"),
        "MIMO_CHECKPOINT_TEST_PRETRAIN": "1",
        "MIMO_PRETRAIN_MODULE": _MODULE,
        "TRAIN_ITERS": "2",
        "TORCHRUN_LOG_DIR": str(scratch / f"torchrun-{name}"),
    }
    if resave_after_load:
        env["MIMO_CHECKPOINT_TEST_RESAVE"] = "1"
    env.pop("NVTE_FLASH_ATTN", None)
    env.pop("NVTE_FUSED_ATTN", None)
    command = [
        "bash",
        str(_LAUNCHER),
        "--num-experts",
        "8",
        "--no-save-tokenizer-assets",
        "--ckpt-format",
        "torch_dist",
        *args,
    ]
    return subprocess.run(
        command, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800
    )


def _tail(result: subprocess.CompletedProcess) -> str:
    return f"--- stdout ---\n{result.stdout[-6000:]}\n--- stderr ---\n{result.stderr[-3000:]}"


def _checkpoint_iteration(root: Path) -> Path:
    iteration = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    return root / f"iter_{iteration:07d}"


def _load_entry(path: Path, key: str, metadata):
    if isinstance(metadata, TensorStorageMetadata):
        value = torch.empty(tuple(metadata.size), dtype=metadata.properties.dtype)
    elif isinstance(metadata, BytesStorageMetadata):
        value = io.BytesIO()
    else:
        raise TypeError(f"unsupported checkpoint entry {key}: {type(metadata)}")
    state = {key: value}
    dcp.load(
        state, storage_reader=FileSystemReader(path), planner=DefaultLoadPlanner(), no_dist=True
    )
    value = state[key]
    if isinstance(value, io.BytesIO):
        value.seek(0)
        return torch.load(value, map_location="cpu", weights_only=False)
    return value


def _install_padding_manifest_hook() -> None:
    """Record only optimizer ranges explicitly marked as padding."""
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedTensor
    from megatron.core.models.mimo.optimizer import MimoOptimizer

    manifest_dir = Path(os.environ["MIMO_CHECKPOINT_PADDING_MANIFEST_DIR"])
    original = MimoOptimizer.sharded_state_dict

    def sharded_state_dict(self, *args, **kwargs):
        state = original(self, *args, **kwargs)
        records = set()

        def visit(value):
            if isinstance(value, dict):
                padding = value.get("padding")
                if isinstance(padding, LocalNonpersistentObject) and padding.unwrap():
                    for tensor in value.values():
                        if isinstance(tensor, ShardedTensor):
                            assert len(tensor.global_shape) == len(tensor.global_offset) == 1
                            records.add(
                                (
                                    tensor.key,
                                    tensor.global_shape[0],
                                    tensor.global_offset[0],
                                    tensor.local_shape[0],
                                )
                            )
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(state)
        if records:
            manifest_dir.mkdir(parents=True, exist_ok=True)
            rank = dist.get_rank()
            with (manifest_dir / f"rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
                for key, global_numel, offset, numel in sorted(records):
                    stream.write(
                        json.dumps(
                            {
                                "global_numel": global_numel,
                                "numel": numel,
                                "offset": offset,
                                "tensor_key": key,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        return state

    MimoOptimizer.sharded_state_dict = sharded_state_dict


def _install_checkpoint_resave_hook() -> None:
    """Save the normally loaded state before the training loop starts."""
    if not os.environ.get("MIMO_CHECKPOINT_TEST_RESAVE"):
        return

    from megatron.core.utils import unwrap_model
    from megatron.training import training
    from megatron.training.checkpointing import save_checkpoint

    original = training.setup_model_and_optimizer

    def setup_model_and_optimizer(
        model_type, model_provider_func=None, checkpointing_context=None, **kwargs
    ):
        model, optimizer, opt_param_scheduler = original(
            model_type,
            model_provider_func=model_provider_func,
            checkpointing_context=checkpointing_context,
            **kwargs,
        )
        args = training.get_args()
        unwrapped_model = unwrap_model(model)
        pg_collection = getattr(unwrapped_model[0], "pg_collection", None)
        save_checkpoint(
            args.iteration,
            model,
            optimizer,
            opt_param_scheduler,
            args.num_floating_point_operations_so_far,
            checkpointing_context=checkpointing_context,
            preprocess_common_state_dict_fn=training.preprocess_common_state_dict,
            tp_group=pg_collection.tp if pg_collection is not None else None,
            pp_group=pg_collection.pp if pg_collection is not None else None,
            dp_cp_group=getattr(pg_collection, "dp_cp_gtp_remat", None),
            dp_group=pg_collection.dp if pg_collection is not None else None,
            expt_dp_group=pg_collection.expt_dp if pg_collection is not None else None,
            rng_state_key_prefix=getattr(unwrapped_model[0], "rng_state_key_prefix", ""),
        )
        dist.barrier()
        raise SystemExit(0)

    training.setup_model_and_optimizer = setup_model_and_optimizer


def _load_padding_ranges(path: Path, entries) -> dict[str, list[tuple[int, int]]]:
    records = set()
    for manifest in sorted(path.glob("rank-*.jsonl")):
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                records.add(
                    (
                        record["tensor_key"],
                        record["global_numel"],
                        record["offset"],
                        record["numel"],
                    )
                )
    assert records, f"no optimizer padding was recorded in {path}"

    ranges = {}
    for key, global_numel, offset, numel in records:
        metadata = entries[key]
        assert isinstance(metadata, TensorStorageMetadata)
        assert math.prod(metadata.size) == global_numel
        assert 0 <= offset < offset + numel <= global_numel
        ranges.setdefault(key, set()).add((offset, offset + numel))
    result = {key: sorted(key_ranges) for key, key_ranges in ranges.items()}
    for key, key_ranges in result.items():
        assert all(left[1] <= right[0] for left, right in zip(key_ranges, key_ranges[1:])), key
    return result


def _assert_tensor_exact(left, right, path, padding_ranges=()) -> int:
    assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), path
    assert left.dtype == right.dtype and left.shape == right.shape, path
    element_size = left.element_size()
    left = left.contiguous().view(torch.uint8).flatten()
    right = right.contiguous().view(torch.uint8).flatten()
    cursor = 0
    ignored = 0
    for start, end in padding_ranges:
        start *= element_size
        end *= element_size
        assert torch.equal(left[cursor:start], right[cursor:start]), path
        ignored += int((left[start:end] != right[start:end]).sum())
        cursor = end
    assert torch.equal(left[cursor:], right[cursor:]), path
    return ignored


def _assert_exact(left, right, path="checkpoint"):
    if isinstance(left, argparse.Namespace) or isinstance(right, argparse.Namespace):
        assert isinstance(left, argparse.Namespace) and isinstance(right, argparse.Namespace), path
        return _assert_exact(vars(left), vars(right), path)
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert left.tobytes() == right.tobytes(), path
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        if path.endswith(".args"):
            left = {key: value for key, value in left.items() if key not in _ROUND_TRIP_ARGS}
            right = {key: value for key, value in right.items() if key not in _ROUND_TRIP_ARGS}
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_exact(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            _assert_exact(left_value, right_value, f"{path}.{index}")
        return
    assert type(left) is type(right), path
    if isinstance(left, float):
        assert (math.isnan(left) and math.isnan(right)) or struct.pack("!d", left) == struct.pack(
            "!d", right
        ), path
    else:
        assert left == right, path


def _compare_checkpoints(source: Path, round_trip: Path, padding_manifest_dir: Path) -> None:
    dist.init_process_group("gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    source_metadata = FileSystemReader(source).read_metadata()
    round_trip_metadata = FileSystemReader(round_trip).read_metadata()
    source_entries = source_metadata.state_dict_metadata
    round_trip_entries = round_trip_metadata.state_dict_metadata
    padding_ranges = _load_padding_ranges(padding_manifest_dir, source_entries)
    errors = []
    stats = [0, 0, 0]

    def compare(left, right, path):
        try:
            _assert_exact(left, right, path)
        except Exception as error:
            errors.append(str(error))

    if rank == 0:
        source_common = load_common_state_dict(source)
        round_trip_common = load_common_state_dict(round_trip)
        compare(source_common, round_trip_common, "common_state")
        compare(source_metadata.planner_data, round_trip_metadata.planner_data, "planner_data")
        compare(
            getattr(source_metadata, "mcore_data", None),
            getattr(round_trip_metadata, "mcore_data", None),
            "mcore_data",
        )
        compare(set(source_entries), set(round_trip_entries), "checkpoint_keys")
        for key in (
            "iteration",
            "checkpoint_version",
            "num_floating_point_operations_so_far",
            "opt_param_scheduler",
        ):
            if key not in source_common:
                errors.append(f"common_state.{key}: required state is missing")
        if source_common.get("opt_param_scheduler") is None:
            errors.append("common_state.opt_param_scheduler: required state is empty")
        shared_keys = set(source_entries) & set(round_trip_entries)
        for category, predicate in {
            "language_model": lambda key: key.startswith("language_model."),
            "modality_model": lambda key: key.startswith("modality_submodules."),
            "language_optimizer": lambda key: key.startswith("mimo.language."),
            "encoder_optimizer": lambda key: key.startswith("mimo.radio_encoder."),
            "optimizer_param": lambda key: key.startswith("mimo.") and key.endswith(".param"),
            "optimizer_exp_avg": lambda key: key.startswith("mimo.") and key.endswith(".exp_avg"),
            "optimizer_exp_avg_sq": lambda key: key.startswith("mimo.")
            and key.endswith(".exp_avg_sq"),
            "language_rng": lambda key: "language.rng_state" in key,
            "encoder_rng": lambda key: "encoder.rng_state" in key,
        }.items():
            if not any(predicate(key) for key in shared_keys):
                errors.append(f"checkpoint.{category}: required state is empty")

    for index, key in enumerate(sorted(set(source_entries) & set(round_trip_entries))):
        if index % world_size != rank:
            continue
        source_value = _load_entry(source, key, source_entries[key])
        round_trip_value = _load_entry(round_trip, key, round_trip_entries[key])
        if isinstance(source_value, torch.Tensor):
            try:
                stats[2] += _assert_tensor_exact(
                    source_value,
                    round_trip_value,
                    f"sharded_state.{key}",
                    padding_ranges.get(key, ()),
                )
            except Exception as error:
                errors.append(str(error))
        else:
            compare(source_value, round_trip_value, f"sharded_state.{key}")
        stats[0] += 1
        if isinstance(source_value, torch.Tensor):
            stats[1] += source_value.numel() * source_value.element_size()

    totals = torch.tensor(stats, dtype=torch.int64)
    dist.all_reduce(totals)
    all_errors = [None] * world_size
    dist.all_gather_object(all_errors, errors)
    errors = [error for rank_errors in all_errors for error in rank_errors]
    if rank == 0:
        if errors:
            for error in errors[:100]:
                print(f"MISMATCH {error}")
            print(f"Checkpoint comparison failed: {len(errors)} mismatches")
        else:
            print(
                f"Exact checkpoint match: {totals[0].item()} entries / {totals[1].item()} bytes; "
                f"ignored {totals[2].item()} differing padding bytes"
            )
    failed = torch.tensor([bool(errors)], dtype=torch.int32)
    dist.broadcast(failed, src=0)
    dist.destroy_process_group()
    if failed.item():
        raise SystemExit(1)


def _run_comparator(
    source: Path, round_trip: Path, padding_manifest_dir: Path
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "8",
        "--module",
        _MODULE,
        "--compare-checkpoints",
        str(source),
        str(round_trip),
        "--padding-manifest-dir",
        str(padding_manifest_dir),
    ]
    return subprocess.run(command, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=1800)


@pytest.mark.skipif(not HAVE_GTP, reason="GTP requires a supported Transformer Engine version")
def test_hetero_mimo_20l_checkpoint_round_trip_is_exact():
    assert torch.cuda.device_count() >= 8, "requires 8 GPUs"

    scratch = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix="mimo_e2e_", dir=_REPO_ROOT))
        source_root = scratch / "source"
        round_trip_root = scratch / "round-trip"
        padding_manifest_dir = scratch / "padding-manifest"
        save = _run_launcher(
            scratch,
            "save",
            "--save",
            str(source_root),
            "--save-interval",
            "1",
            "--exit-interval",
            "1",
        )
        assert save.returncode == 0, f"checkpoint save failed:\n{_tail(save)}"
        source_checkpoint = _checkpoint_iteration(source_root)

        load_save = _run_launcher(
            scratch,
            "load-save",
            "--save",
            str(round_trip_root),
            "--save-interval",
            "1",
            "--load",
            str(source_root),
            resave_after_load=True,
        )
        assert load_save.returncode == 0, f"checkpoint load/save failed:\n{_tail(load_save)}"
        round_trip_checkpoint = _checkpoint_iteration(round_trip_root)

        comparison = _run_comparator(source_checkpoint, round_trip_checkpoint, padding_manifest_dir)
        assert comparison.returncode == 0, f"checkpoint state changed:\n{_tail(comparison)}"
        assert "Exact checkpoint match:" in comparison.stdout
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    if os.environ.get("MIMO_CHECKPOINT_TEST_PRETRAIN"):
        _install_padding_manifest_hook()
        _install_checkpoint_resave_hook()
        from examples.mimo.pretrain_mimo import main

        main()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--compare-checkpoints", nargs=2, type=Path, required=True)
        parser.add_argument("--padding-manifest-dir", type=Path, required=True)
        parsed = parser.parse_args()
        _compare_checkpoints(*parsed.compare_checkpoints, parsed.padding_manifest_dir)
