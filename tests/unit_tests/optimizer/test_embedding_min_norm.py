# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import _get_megatron_optimizer_based_on_param_groups
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer, copy_optimizer_param_metadata
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.training.arguments import _add_regularization_args


def _fake_distributed_optimizer(
    embedding,
    *,
    min_norm,
    start=0,
    end=None,
    owns_shard=True,
    data_parallel_rank=0,
    optimizer_instance_id=0,
):
    if end is None:
        end = embedding.numel()
    param_range = SimpleNamespace(start=start, end=end)
    return SimpleNamespace(
        config=SimpleNamespace(embedding_min_norm=min_norm),
        _word_embedding_params=[embedding],
        _embedding_min_norm_stats=None,
        model_param_gbuf_map={embedding: object()} if owns_shard else {},
        data_parallel_group=SimpleNamespace(rank=lambda: data_parallel_rank),
        distributed_optimizer_instance_id=optimizer_instance_id,
        _get_model_param_range_map=lambda _: {"param": param_range},
    )


def _initialize_embedding_min_norm(optimizer):
    optimizer._embedding_min_norm_states = DistributedOptimizer._build_embedding_min_norm_states(
        optimizer
    )


def test_embedding_min_norm_projects_split_master_shard(monkeypatch):
    """A flat DP shard may begin and end in the middle of embedding rows."""
    full_master = torch.tensor(
        [[0.3, 0.4, 0.1, 0.0], [0.2, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    start, end = 2, 9
    embedding = torch.nn.Parameter(torch.zeros_like(full_master, dtype=torch.bfloat16))
    embedding.main_param = full_master.flatten()[start:end].clone()
    optimizer = _fake_distributed_optimizer(embedding, min_norm=1.0, start=start, end=end)

    local_indices = torch.arange(start, end).div(embedding.shape[1], rounding_mode="floor")
    local_sums = torch.zeros(embedding.shape[0], dtype=torch.float32)
    local_sums.scatter_add_(0, local_indices, embedding.main_param.square())
    remote_sums = full_master.square().sum(dim=1) - local_sums
    observed_groups = []

    def fake_all_reduce(row_sums, *, group):
        observed_groups.append(group)
        row_sums.add_(remote_sums)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_optimizer.tensor_parallel.gtp_local_pad_zero_count",
        lambda *_: 0,
    )
    _initialize_embedding_min_norm(optimizer)

    DistributedOptimizer._post_optimizer_step(optimizer)

    expected = full_master.clone()
    norms = expected.norm(dim=1, keepdim=True)
    expected.mul_(torch.where((norms > 0) & (norms < 1.0), 2.0 / norms, 1.0))
    torch.testing.assert_close(embedding.main_param, expected.flatten()[start:end])
    assert observed_groups == [optimizer.data_parallel_group]
    # The transform owns the FP32 master; ordinary optimizer staging updates this BF16 copy later.
    assert torch.count_nonzero(embedding) == 0


def test_embedding_min_norm_leaves_zero_at_and_above_threshold_unchanged(monkeypatch):
    full_master = torch.tensor(
        [[0.0, 0.0], [0.6, 0.8], [0.3, 0.4], [1.5, 0.0]], dtype=torch.float32
    )
    embedding = torch.nn.Parameter(torch.zeros_like(full_master, dtype=torch.bfloat16))
    embedding.main_param = full_master.flatten().clone()
    optimizer = _fake_distributed_optimizer(embedding, min_norm=1.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_optimizer.tensor_parallel.gtp_local_pad_zero_count",
        lambda *_: 0,
    )
    _initialize_embedding_min_norm(optimizer)

    DistributedOptimizer._post_optimizer_step(optimizer)
    DistributedOptimizer._post_optimizer_step(optimizer)

    expected = torch.tensor([[0.0, 0.0], [0.6, 0.8], [1.2, 1.6], [1.5, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(embedding.main_param.view_as(expected), expected)
    stats = DistributedOptimizer._consume_embedding_min_norm_stats(optimizer)
    torch.testing.assert_close(stats, torch.tensor([1.0, 8.0, 2.25, 2.0], dtype=torch.float64))
    assert DistributedOptimizer._consume_embedding_min_norm_stats(optimizer) is None


def test_embedding_min_norm_nonowner_still_enters_collective(monkeypatch):
    embedding = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.bfloat16))
    embedding.main_param = None
    optimizer = _fake_distributed_optimizer(embedding, min_norm=1.0, owns_shard=False)
    calls = []
    padding_calls = 0

    def fake_all_reduce(row_sums, *, group):
        calls.append((row_sums, row_sums.clone(), group))
        row_sums.fill_(4.0)

    def fake_padding(*_args):
        nonlocal padding_calls
        padding_calls += 1
        return 0

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_optimizer.tensor_parallel.gtp_local_pad_zero_count",
        fake_padding,
    )
    _initialize_embedding_min_norm(optimizer)

    DistributedOptimizer._post_optimizer_step(optimizer)
    DistributedOptimizer._post_optimizer_step(optimizer)

    assert padding_calls == 1
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0] is optimizer._embedding_min_norm_states[0].row_sums
    torch.testing.assert_close(calls[0][1], torch.zeros(3))
    torch.testing.assert_close(calls[1][1], torch.zeros(3))
    assert all(call[2] is optimizer.data_parallel_group for call in calls)


@pytest.mark.parametrize(("data_parallel_rank", "optimizer_instance_id"), [(1, 0), (0, 1)])
def test_embedding_min_norm_stats_recorded_once_per_model_replica(
    monkeypatch, data_parallel_rank, optimizer_instance_id
):
    embedding = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.bfloat16))
    embedding.main_param = torch.tensor([0.25, 0.0])
    optimizer = _fake_distributed_optimizer(
        embedding,
        min_norm=1.0,
        data_parallel_rank=data_parallel_rank,
        optimizer_instance_id=optimizer_instance_id,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_optimizer.tensor_parallel.gtp_local_pad_zero_count",
        lambda *_: 0,
    )
    _initialize_embedding_min_norm(optimizer)

    DistributedOptimizer._post_optimizer_step(optimizer)

    torch.testing.assert_close(embedding.main_param, torch.tensor([2.0, 0.0]))
    assert DistributedOptimizer._consume_embedding_min_norm_stats(optimizer) is None


def test_embedding_min_norm_excludes_complete_gtp_padding_rows(monkeypatch):
    full_master = torch.tensor([[0.25, 0.0], [2.0, 0.0], [0.1, 0.2]], dtype=torch.float32)
    embedding = torch.nn.Parameter(torch.zeros_like(full_master, dtype=torch.bfloat16))
    embedding.main_param = full_master.flatten().clone()
    optimizer = _fake_distributed_optimizer(embedding, min_norm=1.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_optimizer.tensor_parallel.gtp_local_pad_zero_count",
        lambda *_: embedding.shape[1],
    )
    _initialize_embedding_min_norm(optimizer)

    DistributedOptimizer._post_optimizer_step(optimizer)

    expected = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.1, 0.2]], dtype=torch.float32)
    torch.testing.assert_close(embedding.main_param.view_as(expected), expected)


def test_embedding_min_norm_disabled_is_noop(monkeypatch):
    embedding = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.bfloat16))
    embedding.main_param = torch.tensor([0.25, 0.0])
    optimizer = _fake_distributed_optimizer(embedding, min_norm=None)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("disabled norm floor communicated"),
    )

    DistributedOptimizer._post_optimizer_step(optimizer)

    torch.testing.assert_close(embedding.main_param, torch.tensor([0.25, 0.0]))


def test_post_optimizer_transform_runs_before_model_copy():
    events = []
    optimizer = SimpleNamespace(
        config=SimpleNamespace(timers=None, reuse_grad_buf_for_mxfp8_param_ag=False),
        is_stub_optimizer=False,
        optimizer=SimpleNamespace(step=lambda: events.append("inner-step")),
        _post_optimizer_step=lambda: events.append("post-step"),
        _copy_main_params_to_model_params=lambda: events.append("model-copy"),
    )

    assert MixedPrecisionOptimizer.step_with_ready_grads(optimizer)
    assert events == ["inner-step", "post-step", "model-copy"]


@pytest.mark.parametrize(
    ("overlap_param_gather", "expected"),
    [
        (False, ["inner-step", "post-step", "param-buffer-copy"]),
        (True, ["inner-step", "post-step"]),
    ],
)
def test_post_optimizer_transform_precedes_mxfp8_staging(overlap_param_gather, expected):
    """Overlap defers staging, while the synchronous MXFP8 path stages the floored master."""
    events = []
    optimizer = SimpleNamespace(
        config=SimpleNamespace(
            timers=None,
            reuse_grad_buf_for_mxfp8_param_ag=True,
            overlap_param_gather=overlap_param_gather,
        ),
        is_stub_optimizer=False,
        optimizer=SimpleNamespace(step=lambda: events.append("inner-step")),
        _post_optimizer_step=lambda: events.append("post-step"),
        _copy_main_params_to_param_buffer=lambda: events.append("param-buffer-copy"),
    )

    assert MixedPrecisionOptimizer.step_with_ready_grads(optimizer)
    assert events == expected


def test_stub_optimizer_does_not_run_post_optimizer_transform():
    events = []
    optimizer = SimpleNamespace(
        config=SimpleNamespace(timers=None, reuse_grad_buf_for_mxfp8_param_ag=False),
        is_stub_optimizer=True,
        optimizer=SimpleNamespace(step=lambda: events.append("inner-step")),
        _post_optimizer_step=lambda: events.append("post-step"),
        _copy_main_params_to_model_params=lambda: events.append("model-copy"),
    )

    assert MixedPrecisionOptimizer.step_with_ready_grads(optimizer)
    assert events == []


def test_overflow_skips_optimizer_step_and_post_transform():
    events = []
    optimizer = SimpleNamespace(
        config=SimpleNamespace(timers=None),
        prepare_grads=lambda: True,
        step_with_ready_grads=lambda: events.append("step-with-ready-grads"),
    )

    assert MixedPrecisionOptimizer.step(optimizer) == (False, None, None)
    assert events == []


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_embedding_min_norm_rejects_nonpositive_or_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite positive"):
        OptimizerConfig(embedding_min_norm=value)


def test_embedding_min_norm_rejects_optimizer_cuda_graph():
    with pytest.raises(ValueError, match="optimizer_cuda_graph"):
        OptimizerConfig(embedding_min_norm=1.0, optimizer_cuda_graph=True)


def test_embedding_min_norm_rejects_non_distributed_embedding_optimizer():
    embedding = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    embedding.is_word_embedding_parameter = True

    with pytest.raises(ValueError, match="distributed optimizer"):
        _get_megatron_optimizer_based_on_param_groups(
            OptimizerConfig(embedding_min_norm=1.0, bf16=True),
            model_chunks=[],
            param_groups=[{"params": [embedding]}],
        )


def test_embedding_min_norm_cli_flag():
    parser = argparse.ArgumentParser()
    _add_regularization_args(parser)

    assert parser.parse_args([]).embedding_min_norm is None
    assert parser.parse_args(["--embedding-min-norm", "0.75"]).embedding_min_norm == 0.75


def test_copy_optimizer_metadata_preserves_word_embedding_marker():
    source = torch.empty(1)
    source.is_word_embedding_parameter = True
    destination = torch.empty_like(source)

    copy_optimizer_param_metadata(destination, source)

    assert destination.is_word_embedding_parameter is True
