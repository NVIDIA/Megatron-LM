# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import types
import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.training import checkpointing
from megatron.training import utils


def test_get_ltor_masks_and_position_ids_masks_eod_and_padding_tokens():
    data = torch.tensor([[1, 2, 0, 3], [4, 5, 6, 0]])

    attention_mask, loss_mask, position_ids = utils.get_ltor_masks_and_position_ids(
        data,
        eod_token=2,
        pad_token=0,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=True,
        pad_mask_loss=True,
    )

    assert attention_mask.shape == (1, 1, 4, 4)
    assert attention_mask.dtype == torch.bool
    assert loss_mask.tolist() == [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]
    assert position_ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]


def test_get_ltor_masks_and_position_ids_can_reset_per_batch_attention():
    data = torch.tensor([[1, 2, 0, 3]])

    attention_mask, _, position_ids = utils.get_ltor_masks_and_position_ids(
        data,
        eod_token=2,
        pad_token=0,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=False,
        pad_mask_loss=False,
    )

    assert attention_mask.shape == (1, 1, 4, 4)
    assert position_ids.shape == data.shape


def test_get_blend_and_blend_per_split_from_data_path(monkeypatch):
    monkeypatch.setattr(utils, "get_blend_from_list", lambda values: ("blend", tuple(values)))
    args = SimpleNamespace(
        data_path=["0.7", "train", "0.3", "valid"],
        data_args_path=None,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=None,
    )

    blend, blend_per_split = utils.get_blend_and_blend_per_split(args)

    assert blend == ("blend", ("0.7", "train", "0.3", "valid"))
    assert blend_per_split is None


def test_get_blend_and_blend_per_split_from_data_args_file(monkeypatch, tmp_path):
    data_args = tmp_path / "data_args.txt"
    data_args.write_text("0.5 train 0.5 valid", encoding="utf-8")
    monkeypatch.setattr(utils, "get_blend_from_list", lambda values: ("blend", tuple(values)))
    args = SimpleNamespace(
        data_path=None,
        data_args_path=str(data_args),
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=None,
    )

    blend, blend_per_split = utils.get_blend_and_blend_per_split(args)

    assert blend == ("blend", ("0.5", "train", "0.5", "valid"))
    assert blend_per_split is None


def test_get_blend_and_blend_per_split_from_per_split_json(monkeypatch, tmp_path):
    per_split = tmp_path / "per_split.json"
    per_split.write_text(
        json.dumps(
            {
                "train": "0.8 train-a 0.2 train-b",
                "valid": ["valid-a"],
                "test": ["test-a", "test-b"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(utils, "get_blend_from_list", lambda values: tuple(values))
    args = SimpleNamespace(
        data_path=None,
        data_args_path=None,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=str(per_split),
    )

    blend, blend_per_split = utils.get_blend_and_blend_per_split(args)

    assert blend is None
    assert blend_per_split == [
        ("0.8", "train-a", "0.2", "train-b"),
        ("valid-a",),
        ("test-a", "test-b"),
    ]


def test_get_blend_and_blend_per_split_without_data():
    args = SimpleNamespace(
        data_path=None,
        data_args_path=None,
        train_data_path=None,
        valid_data_path=None,
        test_data_path=None,
        per_split_data_args_path=None,
    )

    assert utils.get_blend_and_blend_per_split(args) == (None, None)


def test_update_use_dist_ckpt_tracks_checkpoint_format():
    args = SimpleNamespace(ckpt_format="torch")
    utils.update_use_dist_ckpt(args)
    assert args.use_dist_ckpt is False

    args.ckpt_format = "torch_dist"
    utils.update_use_dist_ckpt(args)
    assert args.use_dist_ckpt is True


def test_to_empty_if_meta_device_materializes_only_meta_tensors():
    module = torch.nn.Linear(2, 2, device="meta")

    materialized = utils.to_empty_if_meta_device(module, device=torch.device("cpu"))

    assert materialized is module
    assert module.weight.device.type == "cpu"
    assert module.bias.device.type == "cpu"


def test_rank_helpers_use_explicit_rank(capsys):
    utils.print_rank_0("visible", rank=0)
    utils.print_rank_0("hidden", rank=1)

    captured = capsys.readouterr()

    assert "visible" in captured.out
    assert "hidden" not in captured.out


def test_warn_rank_0_uses_explicit_rank():
    with pytest.warns(UserWarning, match="visible warning"):
        utils.warn_rank_0("visible warning", rank=0)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        utils.warn_rank_0("hidden warning", rank=1)

    assert captured == []


def test_print_rank_last_handles_uninitialized_and_last_rank(monkeypatch, capsys):
    monkeypatch.setattr(utils.torch.distributed, "is_initialized", lambda: False)
    utils.print_rank_last("visible without distributed")
    assert "visible without distributed" in capsys.readouterr().out

    monkeypatch.setattr(utils.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(utils.torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(utils, "_safe_get_rank", lambda: 3)
    utils.print_rank_last("visible on last rank")
    assert "visible on last rank" in capsys.readouterr().out

    monkeypatch.setattr(utils, "_safe_get_rank", lambda: 1)
    utils.print_rank_last("hidden on middle rank")
    assert "hidden on middle rank" not in capsys.readouterr().out


def test_append_to_progress_log_writes_only_rank_zero(monkeypatch, tmp_path):
    args = SimpleNamespace(save=str(tmp_path), world_size=8)
    barriers = []
    monkeypatch.setenv("SLURM_JOB_ID", "job-123")
    monkeypatch.setattr(utils, "get_args", lambda: args)
    monkeypatch.setattr(utils.torch.distributed, "barrier", lambda: barriers.append("barrier"))
    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 0)

    utils.append_to_progress_log("Starting job")

    content = (tmp_path / "progress.txt").read_text(encoding="utf-8")
    assert barriers == ["barrier"]
    assert "Job ID: job-123" in content
    assert "# GPUs: 8" in content
    assert "Starting job" in content

    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 1)
    utils.append_to_progress_log("hidden", barrier=False)
    assert "hidden" not in (tmp_path / "progress.txt").read_text(encoding="utf-8")


def test_report_memory_uses_cuda_stats_and_data_parallel_rank(monkeypatch, capsys):
    args = SimpleNamespace(log_device_memory_used=True)
    monkeypatch.setattr(utils, "get_args", lambda: args)
    monkeypatch.setattr(utils.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 7)
    monkeypatch.setattr(utils.torch.cuda, "memory_allocated", lambda: 1024 * 1024)
    monkeypatch.setattr(utils.torch.cuda, "max_memory_allocated", lambda: 2 * 1024 * 1024)
    monkeypatch.setattr(utils.torch.cuda, "memory_reserved", lambda: 3 * 1024 * 1024)
    monkeypatch.setattr(utils.torch.cuda, "max_memory_reserved", lambda: 4 * 1024 * 1024)
    monkeypatch.setattr(utils.torch.cuda, "device_memory_used", lambda: 5 * 1024 * 1024)

    utils.report_memory("after step")

    output = capsys.readouterr().out
    assert "[Rank 7]" in output
    assert "allocated: 1.00" in output
    assert "total device memory used: 5.00" in output


def test_calc_params_l2_norm_handles_dense_moe_and_sharded_params(monkeypatch):
    calls = []
    real_tensor = torch.tensor
    real_zeros = torch.zeros
    real_zeros_like = torch.zeros_like

    def cpu_tensor(*items, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*items, **kwargs)

    def cpu_zeros(*items, **kwargs):
        kwargs.pop("device", None)
        return real_zeros(*items, **kwargs)

    class FakeModel:
        def parameters(self):
            dense = SimpleNamespace(
                data=torch.tensor([1.0, 2.0]),
                allreduce=True,
                main_param=torch.tensor([1.0, 2.0]),
                main_param_sharded=False,
            )
            moe = SimpleNamespace(
                data=torch.tensor([3.0]),
                allreduce=False,
                main_param=torch.tensor([3.0]),
                main_param_sharded=False,
            )
            sharded = SimpleNamespace(
                data=torch.tensor([4.0]),
                allreduce=True,
                main_param=torch.tensor([4.0]),
                main_param_sharded=True,
            )
            return [dense, moe, sharded]

    monkeypatch.setattr(utils, "get_args", lambda: SimpleNamespace(use_megatron_fsdp=False, bf16=True))
    monkeypatch.setattr(utils, "param_is_not_tensor_parallel_duplicate", lambda param: True)
    monkeypatch.setattr(utils, "param_is_not_shared", lambda param: True)
    monkeypatch.setattr(utils, "to_local_if_dtensor", lambda param: param)
    monkeypatch.setattr(utils, "get_data_parallel_group_if_dtensor", lambda param, group: "dtensor-dp")
    monkeypatch.setattr(utils.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(utils.torch, "zeros", cpu_zeros)
    monkeypatch.setattr(utils.torch, "zeros_like", lambda tensor: real_zeros_like(tensor))
    monkeypatch.setattr(
        utils,
        "multi_tensor_applier",
        lambda func, overflow, tensors, per_param: calls.append(("norm", len(tensors[0]))) or (torch.tensor([2.0]), None),
    )
    monkeypatch.setattr(utils.mpu, "get_data_parallel_group", lambda with_context_parallel=False: "dp-cp")
    monkeypatch.setattr(utils.mpu, "get_model_parallel_group", lambda: "dense")
    monkeypatch.setattr(utils.mpu, "get_expert_tensor_model_pipeline_parallel_group", lambda: "expert")
    monkeypatch.setattr(utils.torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(
        utils.torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append(("all-reduce", group)),
    )

    norm = utils.calc_params_l2_norm(FakeModel())

    assert norm == pytest.approx(12.0 ** 0.5)
    assert calls.count(("norm", 1)) == 3
    assert ("all-reduce", "dtensor-dp") in calls
    assert ("all-reduce", "dp-cp") in calls
    assert ("all-reduce", "dense") in calls


def test_calc_params_l2_norm_requires_dtensors_for_megatron_fsdp(monkeypatch):
    class FakeFSDPModel:
        def stop_communication(self):
            pass

        def named_parameters(self):
            return [("not_dtensor", SimpleNamespace())]

    monkeypatch.setattr(utils, "get_args", lambda: SimpleNamespace(use_megatron_fsdp=True))

    with pytest.raises(RuntimeError, match="not a DTensor"):
        utils.calc_params_l2_norm(FakeFSDPModel())


def test_loss_and_stat_reductions_use_expected_distributed_groups(monkeypatch):
    calls = []

    class FakeGroup:
        def size(self):
            return 2

    fake_group = FakeGroup()
    monkeypatch.setattr(utils.mpu, "get_data_parallel_group", lambda: fake_group)
    monkeypatch.setattr(utils.mpu, "get_model_parallel_group", lambda: "mp-group")
    monkeypatch.setattr(utils.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(
        utils.torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: calls.append((tensor.clone(), op, group)),
    )

    averaged = utils.average_losses_across_data_parallel_group(
        [torch.tensor(2.0), torch.tensor(4.0)]
    )
    reduced = utils.reduce_max_stat_across_model_parallel_group(7.5)
    missing = utils.reduce_max_stat_across_model_parallel_group(None)
    logical_true = utils.logical_and_across_model_parallel_group(True)
    logical_false = utils.logical_and_across_model_parallel_group(False)

    assert averaged.tolist() == [1.0, 2.0]
    assert reduced == 7.5
    assert missing is None
    assert logical_true is True
    assert logical_false is False
    assert any(call[2] is fake_group for call in calls)
    assert any(call[2] == "mp-group" for call in calls)


def test_print_params_min_max_norm_emits_parameter_statistics(monkeypatch, capsys):
    param = SimpleNamespace(data=torch.tensor([1.0, -2.0, 3.0]), tensor_model_parallel=True)
    optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"params": [param]}])
    )
    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 5)

    utils.print_params_min_max_norm(optimizer, iteration=9)

    output = capsys.readouterr().out
    assert "iteration, rank, index" in output
    assert "      9,    5,    1" in output
    assert "-2.000000E+00" in output


def test_check_adlr_autoresume_termination_saves_and_exits(monkeypatch):
    calls = []
    autoresume = SimpleNamespace(
        termination_requested=lambda: True,
        request_resume=lambda: calls.append("resume"),
    )
    monkeypatch.setattr(utils, "get_args", lambda: SimpleNamespace(save="/tmp/checkpoints"))
    monkeypatch.setattr(utils, "get_adlr_autoresume", lambda: autoresume)
    monkeypatch.setattr(utils.torch.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(utils.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(utils, "print_rank_0", lambda message: calls.append(("print", message)))
    monkeypatch.setattr(
        checkpointing,
        "save_checkpoint",
        lambda iteration, model, optimizer, scheduler: calls.append(("save", iteration)),
    )

    with pytest.raises(SystemExit):
        utils.check_adlr_autoresume_termination(11, "model", "optimizer", "scheduler")

    assert calls[0] == "barrier"
    assert ("save", 11) in calls
    assert "resume" in calls
    assert any(item == ("print", ">>> training terminated. Returning") for item in calls)


def test_check_adlr_autoresume_termination_noops_without_signal(monkeypatch):
    calls = []
    autoresume = SimpleNamespace(termination_requested=lambda: False)
    monkeypatch.setattr(utils, "get_args", lambda: SimpleNamespace(save=None))
    monkeypatch.setattr(utils, "get_adlr_autoresume", lambda: autoresume)
    monkeypatch.setattr(utils.torch.distributed, "barrier", lambda: calls.append("barrier"))

    utils.check_adlr_autoresume_termination(1, None, None, None)

    assert calls == ["barrier"]


def test_pipeline_stage_helper_uses_rank_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(
        utils.mpu,
        "is_pipeline_first_stage",
        lambda ignore_virtual=False, vp_stage=None: seen.append(("first", ignore_virtual, vp_stage)) or False,
    )
    monkeypatch.setattr(
        utils.mpu,
        "is_pipeline_last_stage",
        lambda ignore_virtual=False, vp_stage=None: seen.append(("last", ignore_virtual, vp_stage)) or True,
    )

    assert utils.is_first_or_last_pipeline_stage(vp_stage=None) is True
    assert ("first", True, None) in seen

    monkeypatch.setattr(
        utils.mpu,
        "is_pipeline_last_stage",
        lambda ignore_virtual=False, vp_stage=None: False,
    )
    assert utils.is_first_or_last_pipeline_stage(vp_stage=2) is False


def test_get_nvtx_range_uses_nvtx_and_optional_timers(monkeypatch):
    calls = []

    class FakeTimer:
        def start(self):
            calls.append("timer-start")

        def stop(self):
            calls.append("timer-stop")

    fake_nvtx = types.SimpleNamespace(
        range_push=lambda msg: calls.append(("push", msg)),
        range_pop=lambda: calls.append("pop"),
    )
    monkeypatch.setattr(utils.torch.cuda, "nvtx", fake_nvtx, raising=False)
    monkeypatch.setattr(utils, "get_timers", lambda: lambda *args, **kwargs: FakeTimer())

    with utils.get_nvtx_range()("section", time=True):
        calls.append("body")

    assert calls == ["timer-start", ("push", "section"), "body", "pop", "timer-stop"]
