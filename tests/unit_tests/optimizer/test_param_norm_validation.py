# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise embedding setup and row projection on two or more Gloo ranks."""

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.optimizer.apply_param_norm_constraints import ParamNormSetup
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.param_norm_constraints import EmbeddingMinNormRecipe


@pytest.fixture(scope="module")
def groups():
    world_size = (
        dist.get_world_size() if dist.is_initialized() else int(os.getenv("WORLD_SIZE", "1"))
    )
    if not dist.is_gloo_available() or world_size < 2:
        pytest.skip("requires at least two distributed ranks and Gloo")
    own_default = not dist.is_initialized()
    timeout = timedelta(seconds=30)
    if own_default:
        dist.init_process_group("gloo", timeout=timeout)
    peers = dist.new_group(backend="gloo", timeout=timeout)
    local = dist.new_group(
        [dist.get_rank()], backend="gloo", use_local_synchronization=True, timeout=timeout
    )
    try:
        yield peers, local
    finally:
        dist.destroy_process_group(local)
        dist.destroy_process_group(peers)
        if own_default:
            dist.destroy_process_group()


def _bind(tables, *, dp, peers, threshold=1.0, local_range=None, embd=None):
    model = torch.nn.Module()
    for name, weight in tables:
        model.register_parameter(name, weight)
    parameters = list(model.parameters())
    pg_collection = SimpleNamespace(embd=embd)
    optimizer = SimpleNamespace(
        config=OptimizerConfig(use_distributed_optimizer=True, embedding_min_norm=threshold),
        optimizer=SimpleNamespace(param_groups=[{"params": parameters}]),
        buffers=[SimpleNamespace(params=parameters)],
        _local_param_range=local_range or (lambda p: (0, p.numel())),
        _param_norm_plan=None,
        distributed_optimizer_instance_id=0,
    )
    setup = ParamNormSetup([model], EmbeddingMinNormRecipe(threshold))
    setup.bind_to_distrib_optimizer(optimizer, data_parallel_group=dp, pg_collection=pg_collection)
    setup.validate(pg_collection, process_group=peers)
    return optimizer


def test_dp_table_order_with_split_and_empty_shards(groups):
    peers, _ = groups
    a = torch.nn.Parameter(torch.tensor([[0.3, 0.4], [0.0, 0.25]]))
    b = torch.nn.Parameter(torch.tensor([[0.0, 0.25], [3.0, 4.0]]))
    a.is_word_embedding_parameter = b.is_word_embedding_parameter = True
    tables = [("a", a), ("b", b)]
    if peers.rank() != 0:
        tables.reverse()

    def owned(p):
        if p is b:
            return (0, 4) if peers.rank() == 0 else (0, 0)
        return {0: (0, 1), 1: (1, 4)}.get(peers.rank(), (0, 0))

    optimizer = _bind(tables, dp=peers, peers=peers, local_range=owned)
    expected = {a: a.detach().clone(), b: b.detach().clone()}
    DistributedOptimizer._post_optimizer_step(optimizer)
    for p, projected in [(a, [1.2, 1.6, 0, 2]), (b, [0, 2, 3.0, 4.0])]:
        start, end = owned(p)
        expected[p].flatten()[start:end] = torch.tensor(projected)[start:end]
        torch.testing.assert_close(p, expected[p])


@pytest.mark.parametrize("mismatch", ["row count", "threshold", "frozen", "split row"])
def test_dp_disagreement_is_rejected_at_setup(groups, mismatch):
    peers, _ = groups
    other = peers.rank() != 0
    rows = 2 if other and mismatch == "row count" else 1
    weight = torch.nn.Parameter(
        torch.ones(rows, 2), requires_grad=not (other and mismatch == "frozen")
    )
    weight.is_word_embedding_parameter = True
    if other and mismatch == "split row":
        weight.tensor_model_parallel = True
        weight.partition_dim = 1
    threshold = 2.0 if other and mismatch == "threshold" else 1.0
    error = "weight.*complete rows" if mismatch == "split row" else "weight"
    with pytest.raises(ValueError, match=error):
        _bind([("weight", weight)], dp=peers, peers=peers, threshold=threshold)


@pytest.mark.parametrize("mismatch", [False, True])
def test_gtp_padding_only_tile_participates_in_setup(groups, mismatch):
    peers, local = groups
    padded = peers.rank() != 0
    weight = torch.nn.Parameter(torch.tensor([[0.25, 0.0]] if padded else [[0.3, 0.4]]))
    weight.is_word_embedding_parameter = weight.is_gtp_weight_remat = True
    weight.group, weight.pad_length = peers, peers.size() - 1
    if mismatch:
        with pytest.raises(ValueError, match="weight"):
            _bind([("weight", weight)], dp=local, peers=peers, threshold=2.0 if padded else 1.0)
        return
    optimizer = _bind([("weight", weight)], dp=local, peers=peers)
    DistributedOptimizer._post_optimizer_step(optimizer)
    torch.testing.assert_close(weight, torch.tensor([[0.25, 0.0]] if padded else [[1.2, 1.6]]))


@pytest.mark.parametrize("tied,select_both", [(True, True), (True, False), (False, False)])
def test_pipeline_embedding_aliases(groups, tied, select_both):
    peers, local = groups
    first = peers.rank() == 0
    table = torch.nn.Parameter(torch.tensor([[0.3, 0.4]]))
    table.shared_embedding = tied
    table.is_word_embedding_parameter = first or select_both
    tables = [("input_table" if first else "output_head", table)]
    if tied and not select_both:
        with pytest.raises(ValueError, match="shared word embedding"):
            _bind(tables, dp=local, peers=peers, embd=peers)
        return
    optimizer = _bind(tables, dp=local, peers=peers, embd=peers)
    DistributedOptimizer._post_optimizer_step(optimizer)
    torch.testing.assert_close(
        table, torch.tensor([[1.2, 1.6]] if first or select_both else [[0.3, 0.4]])
    )
