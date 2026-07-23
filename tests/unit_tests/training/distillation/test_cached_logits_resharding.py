# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.training.distillation import cached_logits_loss
from megatron.training.distillation.cached_logits_loss import make_teacher_tar_dataset
from megatron.training.distillation.utils import (
    LOGPROBS_TAR_MEMBER_SUFFIX,
    META_TAR_MEMBER,
    LogprobsReshardPlan,
    v2_pack_indices,
)

zstandard = pytest.importorskip("zstandard")


def _torch_save_bytes(payload) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _write_tar(path: Path, metadata: dict, members: dict[str, bytes]) -> None:
    compressor = zstandard.ZstdCompressor(level=1)
    with tarfile.open(path, "w") as tar:
        meta_bytes = json.dumps(metadata).encode("utf-8")
        info = tarfile.TarInfo(META_TAR_MEMBER)
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))

        for name, payload in members.items():
            compressed = compressor.compress(payload)
            info = tarfile.TarInfo(name)
            info.size = len(compressed)
            tar.addfile(info, io.BytesIO(compressed))


def _saved_indices(iteration: int, dp_rank: int, *, mbs: int, dp: int, gbs: int) -> torch.Tensor:
    num_mb = gbs // (mbs * dp)
    sample_ids = [
        iteration * gbs + (microbatch * dp + dp_rank) * mbs + sample
        for microbatch in range(num_mb)
        for sample in range(mbs)
    ]
    return torch.tensor(sample_ids, dtype=torch.long).view(1, -1, 1)


def _write_v2_cache(root: Path, *, mbs: int, dp: int, gbs: int, iterations: int) -> None:
    metadata = {
        "saver": {"format_version": 2, "mbs_save": mbs, "dp_size_save": dp, "gbs_save": gbs}
    }
    for dp_rank in range(dp):
        members = {}
        for iteration in range(iterations):
            indices = _saved_indices(iteration, dp_rank, mbs=mbs, dp=dp, gbs=gbs)
            indices_low, bit_17 = v2_pack_indices(indices)
            members[
                f"{iteration * gbs}-{(iteration + 1) * gbs}" f"{LOGPROBS_TAR_MEMBER_SUFFIX}"
            ] = _torch_save_bytes(
                {
                    "values": torch.zeros_like(indices, dtype=torch.float32),
                    "indices_low": indices_low,
                    "bit_17": bit_17,
                    "format_version": 2,
                }
            )
        _write_tar(root / f"dp{dp_rank}__0-{iterations * gbs}.tar", metadata, members)


def _write_v1_cache(root: Path, *, mbs: int, dp: int, gbs: int, iterations: int) -> None:
    metadata = {
        "saver": {"format_version": 1, "mbs_save": mbs, "dp_size_save": dp, "gbs_save": gbs}
    }
    num_mb = gbs // (mbs * dp)
    for dp_rank in range(dp):
        members = {}
        for iteration in range(iterations):
            indices = _saved_indices(iteration, dp_rank, mbs=mbs, dp=dp, gbs=gbs)
            indices_list = list(indices.split(mbs, dim=1))
            members[f"{iteration}{LOGPROBS_TAR_MEMBER_SUFFIX}"] = _torch_save_bytes(
                {
                    "values": [
                        torch.zeros_like(tensor, dtype=torch.float32) for tensor in indices_list
                    ],
                    "indices_low": [(tensor & 0xFFFF).to(torch.uint16) for tensor in indices_list],
                    "bit_17": [(tensor >> 16).bool() for tensor in indices_list],
                }
            )
        assert len(indices_list) == num_mb
        _write_tar(root / f"cp0_dp{dp_rank}__{iterations - 1}.tar", metadata, members)


def _assert_dataset_indices(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mbs_load: int,
    dp_load: int,
    gbs_load: int,
    expected_iterations: int,
) -> None:
    args = SimpleNamespace(micro_batch_size=mbs_load, global_batch_size=gbs_load)
    monkeypatch.setattr(cached_logits_loss, "get_args", lambda: args)

    num_mb_load = gbs_load // (mbs_load * dp_load)
    for dp_rank in range(dp_load):
        dataset = make_teacher_tar_dataset(
            str(root), cp_rank=0, cp_size=1, dp_rank=dp_rank, dp_size=dp_load, ignore_hash=True
        )
        loaded = list(dataset)
        assert len(loaded) == expected_iterations
        for iteration, values_list, indices_list in loaded:
            assert len(values_list) == num_mb_load
            assert len(indices_list) == num_mb_load
            for microbatch, indices in enumerate(indices_list):
                start = iteration * gbs_load + (microbatch * dp_load + dp_rank) * mbs_load
                expected = list(range(start, start + mbs_load))
                assert indices.reshape(-1).tolist() == expected


@pytest.mark.parametrize(
    ("dp_save", "dp_load", "gbs"),
    [(4, 4, 24), (4, 8, 24), (8, 4, 24), (4, 6, 24), (6, 4, 24), (3, 5, 30), (5, 3, 30)],
)
@pytest.mark.parametrize("format_version", [1, 2])
def test_dataset_dp_resharding_preserves_global_indices(
    tmp_path, monkeypatch, dp_save, dp_load, gbs, format_version
):
    iterations = 2
    if format_version == 1:
        _write_v1_cache(tmp_path, mbs=1, dp=dp_save, gbs=gbs, iterations=iterations)
    else:
        _write_v2_cache(tmp_path, mbs=1, dp=dp_save, gbs=gbs, iterations=iterations)
    _assert_dataset_indices(
        tmp_path,
        monkeypatch,
        mbs_load=1,
        dp_load=dp_load,
        gbs_load=gbs,
        expected_iterations=iterations,
    )


@pytest.mark.parametrize(
    ("dp_save", "gbs_save", "save_iterations", "dp_load", "gbs_load"),
    [(6, 24, 2, 4, 8), (4, 8, 3, 6, 24)],
)
def test_v2_dp_resharding_composes_with_gbs_resharding(
    tmp_path, monkeypatch, dp_save, gbs_save, save_iterations, dp_load, gbs_load
):
    _write_v2_cache(tmp_path, mbs=1, dp=dp_save, gbs=gbs_save, iterations=save_iterations)
    total_samples = save_iterations * gbs_save
    _assert_dataset_indices(
        tmp_path,
        monkeypatch,
        mbs_load=1,
        dp_load=dp_load,
        gbs_load=gbs_load,
        expected_iterations=total_samples // gbs_load,
    )


@pytest.mark.parametrize(
    ("dp_save", "dp_load", "gbs"), [(4, 6, 24), (6, 4, 24), (3, 5, 30), (5, 3, 30)]
)
def test_reshard_plan_sources_reconstruct_global_indices(dp_save, dp_load, gbs):
    plan = LogprobsReshardPlan(
        mbs_save=1, dp_save=dp_save, gbs_save=gbs, mbs_load=1, dp_load=dp_load, gbs_load=gbs
    )
    for dp_rank in range(dp_load):
        used_sources = set()
        for microbatch in range(plan.num_mb_load):
            sources = list(plan.sources_for_microbatch(0, microbatch, dp_rank))
            used_sources.update(source.d_save for source in sources)
            reconstructed = []
            for source in sources:
                for row in range(source.row_start, source.row_end):
                    saved_mb, sample = divmod(row, plan.mbs_save)
                    reconstructed.append(
                        source.iter_save * plan.gbs_save
                        + (saved_mb * plan.dp_save + source.d_save) * plan.mbs_save
                        + sample
                    )
            expected_start = (microbatch * dp_load + dp_rank) * plan.mbs_load
            assert reconstructed == list(range(expected_start, expected_start + plan.mbs_load))
        assert set(plan.needed_d_saves(dp_rank)) == used_sources


def test_needed_d_saves_covers_all_smaller_gbs_phases():
    plan = LogprobsReshardPlan(
        mbs_save=1, dp_save=6, gbs_save=24, mbs_load=1, dp_load=4, gbs_load=8
    )
    assert plan.needed_d_saves(0) == [0, 2, 4]


def test_invalid_global_batch_divisibility_still_fails():
    with pytest.raises(ValueError, match="gbs_load"):
        LogprobsReshardPlan(mbs_save=1, dp_save=4, gbs_save=24, mbs_load=1, dp_load=5, gbs_load=24)
