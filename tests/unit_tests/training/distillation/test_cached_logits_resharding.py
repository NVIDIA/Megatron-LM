# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.training.distillation import cached_logits_loss
from megatron.training.distillation.cached_logits_loss import (
    CachedLogitsKDLoss,
    LossFuncCallable,
    make_teacher_tar_dataset,
)
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


# ---------------------------------------------------------------------------
# LogprobsReshardPlan: MBS mismatch (existing tests above only exercise mbs=1)
# ---------------------------------------------------------------------------


def test_reshard_plan_mbs_mismatch_fails():
    with pytest.raises(ValueError, match="mbs"):
        LogprobsReshardPlan(mbs_save=3, dp_save=4, gbs_save=24, mbs_load=2, dp_load=4, gbs_load=24)


@pytest.mark.parametrize(
    ("mbs_save", "mbs_load", "dp_save", "dp_load", "gbs"), [(2, 2, 4, 6, 48), (4, 2, 6, 4, 48)]
)
def test_reshard_plan_sources_reconstruct_global_indices_with_mbs_gt_1(
    mbs_save, mbs_load, dp_save, dp_load, gbs
):
    plan = LogprobsReshardPlan(
        mbs_save=mbs_save,
        dp_save=dp_save,
        gbs_save=gbs,
        mbs_load=mbs_load,
        dp_load=dp_load,
        gbs_load=gbs,
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


# ---------------------------------------------------------------------------
# CachedLogitsKDLoss: StopIteration / skip-ahead / misalignment / trimming
# ---------------------------------------------------------------------------


def _bare_kd_loss(dataloader_items):
    """Construct a CachedLogitsKDLoss without running __init__ (which touches
    parallel_state), pre-loaded with a plain-iterator "DataLoader" that yields
    ``(loaded_iteration, values_list, indices_list)`` tuples.
    """
    loss = CachedLogitsKDLoss.__new__(CachedLogitsKDLoss)
    loss.logprobs_dir = "unused"
    loss.tp_rank = 0
    loss.tp_size = 1
    loss.tp_group = None
    loss._dataloader_iter = iter(dataloader_items)
    loss._current_iteration = None
    loss._loaded_iteration = None
    loss._microbatch_counter = 0
    loss._current_values = None
    loss._current_indices = None
    return loss


def test_advance_iteration_skips_behind_shards(monkeypatch, caplog):
    values0 = [torch.randn(2, 1, 2)]
    indices0 = [torch.randint(0, 2, (2, 1, 2))]
    values1 = [torch.randn(2, 1, 2)]
    indices1 = [torch.randint(0, 2, (2, 1, 2))]
    loss = _bare_kd_loss([(3, values0, indices0), (5, values1, indices1)])

    calls = []
    monkeypatch.setattr(
        cached_logits_loss,
        "topk_kl_div",
        lambda *args, **kwargs: calls.append(args) or torch.zeros(1, 2),
    )

    student_logits = torch.randn(2, 1, 4)
    with caplog.at_level("WARNING"):
        loss(student_logits, iteration=5)

    assert loss._current_iteration == 5
    assert loss._loaded_iteration == 5
    assert any("behind training" in message for message in caplog.messages)
    # Both shards (behind + aligned) were consumed from the dataloader.
    with pytest.raises(StopIteration):
        next(loss._dataloader_iter)


def test_call_raises_runtime_error_on_gap():
    values = [torch.randn(2, 1, 2)]
    indices = [torch.randint(0, 2, (2, 1, 2))]
    loss = _bare_kd_loss([(7, values, indices)])  # ahead of requested iteration=5

    student_logits = torch.randn(2, 1, 4)
    with pytest.raises(RuntimeError, match="misaligned"):
        loss(student_logits, iteration=5)


def test_advance_iteration_raises_stopiteration_on_exhaustion():
    loss = _bare_kd_loss([])

    with pytest.raises(StopIteration, match="exhausted"):
        loss._advance_iteration()


def test_lossfunccallable_wraps_stopiteration_as_runtimeerror(monkeypatch):
    callable_ = LossFuncCallable(logprobs_dir="unused")
    callable_.kd_func = lambda logits: (_ for _ in ()).throw(StopIteration("boom"))
    monkeypatch.setattr(
        cached_logits_loss,
        "get_student_logits_capture",
        lambda: SimpleNamespace(pop=lambda: torch.randn(2, 1, 4)),
    )

    model = SimpleNamespace(training=True)
    output_tensor = torch.randn(2, 1)
    loss_mask = torch.ones(2, 1)

    with pytest.raises(RuntimeError, match="exhausted"):
        callable_(loss_mask, output_tensor, model)


def test_teacher_logits_trimmed_to_student_seq_len(monkeypatch):
    loss = _bare_kd_loss([])
    loss._current_iteration = 5
    loss._current_values = [torch.randn(4, 1, 2)]
    loss._current_indices = [torch.randint(0, 2, (4, 1, 2))]

    captured = {}

    def _fake_topk_kl_div(student_logits, teacher_values, teacher_indices, *args, **kwargs):
        captured["teacher_values"] = teacher_values
        captured["teacher_indices"] = teacher_indices
        return torch.zeros(1, student_logits.size(0))

    monkeypatch.setattr(cached_logits_loss, "topk_kl_div", _fake_topk_kl_div)

    student_logits = torch.randn(3, 1, 4)  # shorter than the teacher's seq_len=4
    with pytest.warns(UserWarning, match="trimming"):
        loss(student_logits, iteration=5)

    assert captured["teacher_values"].size(0) == 3
    assert captured["teacher_indices"].size(0) == 3


def test_teacher_logits_equal_len_no_trim_no_warning(monkeypatch, recwarn):
    loss = _bare_kd_loss([])
    loss._current_iteration = 5
    loss._current_values = [torch.randn(3, 1, 2)]
    loss._current_indices = [torch.randint(0, 2, (3, 1, 2))]

    captured = {}

    def _fake_topk_kl_div(student_logits, teacher_values, teacher_indices, *args, **kwargs):
        captured["teacher_values"] = teacher_values
        return torch.zeros(1, student_logits.size(0))

    monkeypatch.setattr(cached_logits_loss, "topk_kl_div", _fake_topk_kl_div)

    student_logits = torch.randn(3, 1, 4)
    loss(student_logits, iteration=5)

    assert captured["teacher_values"].size(0) == 3
    assert len(recwarn) == 0


def test_teacher_shorter_than_student_is_not_trimmed_known_gap(monkeypatch):
    """Documents current (unhandled) behavior: only the teacher-longer-than-
    student direction is trimmed; teacher-shorter-than-student tensors pass
    through unchanged, which would surface as a shape mismatch downstream in
    topk_kl_div if this test's stub weren't absorbing the mismatch."""
    loss = _bare_kd_loss([])
    loss._current_iteration = 5
    loss._current_values = [torch.randn(2, 1, 2)]  # shorter than student
    loss._current_indices = [torch.randint(0, 2, (2, 1, 2))]

    captured = {}

    def _fake_topk_kl_div(student_logits, teacher_values, teacher_indices, *args, **kwargs):
        captured["teacher_values"] = teacher_values
        return torch.zeros(1, student_logits.size(0))

    monkeypatch.setattr(cached_logits_loss, "topk_kl_div", _fake_topk_kl_div)

    student_logits = torch.randn(4, 1, 4)
    loss(student_logits, iteration=5)

    # Unchanged: still the teacher's original (shorter) sequence length.
    assert captured["teacher_values"].size(0) == 2
