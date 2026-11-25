# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import (
    _get_megatron_optimizer_based_on_param_groups,
    get_megatron_optimizer,
)
from megatron.core.optimizer.apply_param_norm_constraints import ParamNormSetup
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer, copy_optimizer_param_metadata
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.param_norm_constraints import EmbeddingMinNorm, EmbeddingMinNormRecipe
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import _add_regularization_args
from tests.unit_tests.test_utilities import Utils


def _process_group(name, rank=0):
    return Mock(
        spec=torch.distributed.ProcessGroup, group_name=name, rank=lambda: rank, size=lambda: 2
    )


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
    embedding.is_word_embedding_parameter = True
    parameters = [embedding]
    group = _process_group("dp", data_parallel_rank)
    optimizer = SimpleNamespace(
        config=OptimizerConfig(embedding_min_norm=min_norm, use_distributed_optimizer=True),
        distributed_optimizer_instance_id=optimizer_instance_id,
        optimizer=SimpleNamespace(param_groups=[{"params": parameters}]),
        buffers=[SimpleNamespace(params=parameters)],
        _local_param_range=lambda _: (start, end) if owns_shard else (0, 0),
        data_parallel_group=group,
        _param_norm_plan=None,
        _embedding_min_norm_stats=None,
    )
    if min_norm is not None:
        setup = ParamNormSetup(
            [torch.nn.ParameterList(parameters)], EmbeddingMinNormRecipe(min_norm)
        )
        setup.bind_to_distrib_optimizer(optimizer, data_parallel_group=group, pg_collection=None)
        assert setup.error is None
    return optimizer


def test_embedding_recipe_selects_trainable_input_and_tied_copy():
    model = torch.nn.ModuleList([torch.nn.Embedding(2, 3) for _ in range(4)])
    for table in model[:3]:
        table.weight.is_word_embedding_parameter = True
    model[1].weight.shared = True
    model[2].weight.requires_grad_(False)
    recipe = EmbeddingMinNormRecipe(0.75)
    selected = recipe(model)
    assert {c.parameter for c in selected} == {model[0].weight, model[1].weight}


@pytest.fixture
def ddp_embedding(request):
    if not torch.cuda.is_available():
        pytest.skip("requires the CUDA distributed optimizer")
    pp_size = getattr(request, "param", 1)
    if Utils.world_size % pp_size:
        pytest.skip(f"requires a world size divisible by {pp_size} pipeline stages")
    # Exercise DP at two ranks and both TP and DP on larger even-sized launches.
    tp_size = 2 if pp_size == 1 and Utils.world_size >= 4 and Utils.world_size % 2 == 0 else 1
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size
    )
    try:
        config = TransformerConfig(
            num_layers=pp_size,
            hidden_size=64,
            num_attention_heads=2,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            params_dtype=torch.bfloat16,
            perform_initialization=False,
        )
        model = VocabParallelEmbedding(256, 64, init_method=torch.nn.init.zeros_, config=config)
        torch.nn.init.constant_(model.weight, 1 / 32)
        model.weight.is_word_embedding_parameter = True
        yield DistributedDataParallel(
            config, DistributedDataParallelConfig(use_distributed_optimizer=True), model
        )
    finally:
        Utils.destroy_model_parallel()


def test_direct_optimizer_warns_and_skips_embedding_floor(ddp_embedding):
    weight = ddp_embedding.module.weight
    config = OptimizerConfig(
        bf16=True, use_distributed_optimizer=True, embedding_min_norm=1.0, clip_grad=0.0
    )
    with pytest.warns(UserWarning, match=r"embedding_min_norm.*get_megatron_optimizer\(\)"):
        optimizer = DistributedOptimizer(
            torch.optim.AdamW([weight], lr=0.0),
            config,
            grad_scaler=None,
            init_state_fn=None,
            model_chunks=[ddp_embedding],
            per_model_buffers={0: ddp_embedding.buffers},
            data_parallel_group=ddp_embedding.intra_dp_cp_group,
            data_parallel_group_gloo=None,
            data_parallel_group_idx=0,
            distributed_optimizer_instance_id=0,
        )
    assert optimizer._param_norm_plan is None
    weight.main_grad.zero_()
    assert optimizer.step()[0]
    # With lr=0, each row should keep its initial norm of 1/4.
    norms = weight.float().norm(dim=1)
    torch.testing.assert_close(norms, torch.full_like(norms, 0.25))
    assert optimizer._consume_embedding_min_norm_stats() is None


@pytest.mark.filterwarnings("error:.*embedding_min_norm.*:UserWarning")
@pytest.mark.parametrize("threshold", [None, 1.0], ids=["disabled", "enabled"])
def test_embedding_floor_through_optimizer_builder(threshold, ddp_embedding, monkeypatch, tmp_path):
    """An ordinary optimizer step publishes the projected FP32 masters to the BF16 model."""
    model = ddp_embedding.module

    if threshold is None:
        monkeypatch.setattr(
            "megatron.core.optimizer.ParamNormSetup",
            lambda *args: pytest.fail("disabled factory created a norm plan"),
        )
    lr = 1 / 64
    config = OptimizerConfig(
        bf16=True,
        use_distributed_optimizer=True,
        lr=lr,
        adam_beta1=0.0,
        adam_beta2=0.0,
        adam_eps=0.0,
        weight_decay=0.0,
        clip_grad=0.0,
        embedding_min_norm=threshold,
        config_logger_dir=str(tmp_path),
    )
    optimizer = get_megatron_optimizer(config, [ddp_embedding])
    log_path = tmp_path / f"DistributedOptimizer.rank_{parallel_state.get_all_ranks()}.iter0.json"
    logged = json.loads(log_path.read_text())
    assert logged["pg_collection"]["dp_cp"] == ddp_embedding.dp_cp_group.group_name

    # With zero moments, zero epsilon and gradients of ±1, Adam subtracts exactly lr * grad.
    # Identical inputs on every rank, with distinct rows in each TP and DP shard.
    signs = torch.randint(2, (256, 64), device="cpu", generator=torch.Generator().manual_seed(17))
    grad = signs.to(device=model.weight.device, dtype=torch.float32) * 2 - 1
    vocab = slice(model.vocab_start_index, model.vocab_end_index)
    model.weight.main_grad.copy_(grad[vocab])
    assert optimizer.step()[0]

    expected = 1 / 32 - lr * grad
    if threshold is not None:  # Every row is below the threshold after Adam.
        expected = 2 * threshold * torch.nn.functional.normalize(expected, dim=1)
    local = expected[vocab]
    start, end = optimizer.chained_optimizers[0]._local_param_range(model.weight)
    if start < end:
        [master] = optimizer.get_parameters()
        torch.testing.assert_close(master, local.flatten()[start:end])
    torch.testing.assert_close(model.weight, local.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("ddp_embedding", [2], indirect=True, ids=["pp2"])
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_embedding_floor_enabled_only_on_input_stage(ddp_embedding, tied):
    input_stage = parallel_state.is_pipeline_first_stage()
    weight = ddp_embedding.module.weight
    weight.is_word_embedding_parameter = input_stage or tied
    weight.shared_embedding = tied
    config = OptimizerConfig(
        bf16=True,
        use_distributed_optimizer=True,
        lr=0.0,
        clip_grad=0.0,
        embedding_min_norm=1.0 if input_stage else None,
    )

    # Bound the wait if one stage skips the optimizer setup collective.
    peers = torch.distributed.new_group(backend="gloo", timeout=timedelta(seconds=30))
    try:
        if tied:
            with pytest.raises(ValueError, match=r"shared word embedding .* mismatch: rank"):
                get_megatron_optimizer(config, [ddp_embedding], param_group_process_group=peers)
            return
        optimizer = get_megatron_optimizer(config, [ddp_embedding], param_group_process_group=peers)
        # Both stages must finish setup before either enters optimizer-step collectives.
        torch.distributed.barrier(group=peers)
    finally:
        torch.distributed.destroy_process_group(peers)

    weight.main_grad.zero_()
    assert optimizer.step()[0]
    norms = weight.float().norm(dim=1)
    torch.testing.assert_close(norms, torch.full_like(norms, 2.0 if input_stage else 0.25))


@pytest.mark.skipif(Utils.world_size < 4, reason="requires two multi-rank construction groups")
@pytest.mark.parametrize("ddp_embedding", [Utils.world_size], indirect=True)
def test_embedding_floor_validation_stays_in_construction_group(ddp_embedding, mocker):
    # One pipeline stage per rank keeps each weight's DP group inside its construction group.
    rank = torch.distributed.get_rank()
    size = torch.distributed.get_world_size()
    groups = [list(range(i, size, 2)) for i in (0, 1)]
    members = groups[rank % 2]
    peers, _ = torch.distributed.new_subgroups_by_enumeration(
        groups, backend="gloo", timeout=timedelta(seconds=30)
    )
    try:
        ddp_embedding.module.weight.is_word_embedding_parameter = rank == members[0]
        config = OptimizerConfig(
            bf16=True,
            use_distributed_optimizer=True,
            lr=0.0,
            embedding_min_norm=1.0 if rank == 0 else None,
        )
        gather = torch.distributed.all_gather_object

        def gather_within_group(object_list, obj, group=None):
            # Catch a WORLD fallback before the active ranks wait for the all-off group.
            assert group is peers
            return gather(object_list, obj, group=group)

        mocker.patch.object(torch.distributed, "all_gather_object", new=gather_within_group)
        validate = mocker.spy(ParamNormSetup, "validate")
        get_megatron_optimizer(config, [ddp_embedding], param_group_process_group=peers)
        assert validate.call_count == (1 if 0 in members else 0)
    finally:
        torch.distributed.destroy_process_group(peers)


@pytest.mark.parametrize(
    "matched, error", [(False, "No optimizer owns.*weight"), (True, "weight.*muon")]
)
def test_trainable_embedding_requires_a_supported_owner(matched, error):
    model = torch.nn.Embedding(2, 3)
    model.weight.is_word_embedding_parameter = True
    setup = ParamNormSetup([model], EmbeddingMinNormRecipe(1.0))
    params = [model.weight] if matched else []
    setup.ownership([{"params": params}], "layer-wise muon", supported=False)
    with pytest.raises(ValueError, match=error):
        setup.validate(None)


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
        "megatron.core.optimizer.apply_param_norm_constraints.tensor_parallel.gtp_local_pad_zero_count",
        fake_padding,
    )
    optimizer = _fake_distributed_optimizer(embedding, min_norm=1.0, owns_shard=False)

    DistributedOptimizer._post_optimizer_step(optimizer)
    DistributedOptimizer._post_optimizer_step(optimizer)

    assert padding_calls == 1
    assert len(calls) == 2
    assert calls[0][0].data_ptr() == calls[1][0].data_ptr()
    torch.testing.assert_close(calls[0][1], torch.zeros(3))
    torch.testing.assert_close(calls[1][1], torch.zeros(3))
    assert all(call[2] is optimizer.data_parallel_group for call in calls)


@pytest.mark.parametrize(
    "data_parallel_rank,optimizer_instance_id,tied",
    [(1, 0, False), (0, 1, False), (0, 0, True)],
    ids=["dp-replica", "optimizer-replica", "tied-pipeline-copy"],
)
def test_embedding_min_norm_projects_nonreporting_copies(
    monkeypatch, data_parallel_rank, optimizer_instance_id, tied
):
    embedding = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.bfloat16))
    embedding.main_param = torch.tensor([0.25, 0.0])
    embedding.shared = tied
    optimizer = _fake_distributed_optimizer(
        embedding,
        min_norm=1.0,
        data_parallel_rank=data_parallel_rank,
        optimizer_instance_id=optimizer_instance_id,
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    DistributedOptimizer._post_optimizer_step(optimizer)
    torch.testing.assert_close(embedding.main_param, torch.tensor([2.0, 0.0]))
    assert DistributedOptimizer._consume_embedding_min_norm_stats(optimizer) is None


def test_embedding_min_norm_excludes_complete_gtp_padding_rows(monkeypatch):
    full_master = torch.tensor([[0.25, 0.0], [2.0, 0.0], [0.1, 0.2]], dtype=torch.float32)
    embedding = torch.nn.Parameter(torch.zeros_like(full_master, dtype=torch.bfloat16))
    embedding.main_param = full_master.flatten().clone()

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)
    embedding.group = _process_group("gtp", rank=1)
    embedding.is_gtp_weight_remat = True
    embedding.pad_length = 1
    optimizer = _fake_distributed_optimizer(embedding, min_norm=1.0)

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
        EmbeddingMinNorm(value)
    with pytest.raises(ValueError, match="finite positive"):
        OptimizerConfig(embedding_min_norm=value).validate()


def test_embedding_min_norm_rejects_optimizer_cuda_graph():
    with pytest.raises(ValueError, match="optimizer_cuda_graph"):
        OptimizerConfig(embedding_min_norm=1.0, optimizer_cuda_graph=True).validate()


@pytest.mark.parametrize("offload_fraction", [0.0, 0.5, 1.0])
def test_embedding_min_norm_rejects_optimizer_cpu_offload(offload_fraction):
    with pytest.raises(ValueError, match="CPU offload"):
        OptimizerConfig(
            embedding_min_norm=1.0,
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=offload_fraction,
            use_precision_aware_optimizer=True,
        ).validate()


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
