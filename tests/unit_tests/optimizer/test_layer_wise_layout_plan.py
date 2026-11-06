# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for how the layer-wise distributed optimizer plan is resolved.

Covers the gradient collective (reduce-scatter -> padded layout, all-reduce -> compact
layout), the optimizer intent, the DDP wrapping decision and the rank-independent
whole-parameter owners of the compact layout.
"""

import random
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    configure_ddp_for_layer_wise_optimizer,
)
from megatron.core.optimizer.optimizer_config import (
    OptimizerConfig,
    resolve_layer_wise_distributed_optimizer,
)


def _param(shape, **attrs):
    param = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    for name, value in attrs.items():
        setattr(param, name, value)
    return param


def _model():
    """Muon matrices, an embedding, a bias and an excluded matrix."""
    model = torch.nn.Module()
    model.embedding = _param((32, 8), is_embedding_or_output_parameter=True)
    model.fc1 = _param((16, 8))
    model.fc1_bias = _param((16,))
    model.fc2 = _param((8, 16))
    model.router = _param((4, 8), use_muon=False)
    return model


# ---------------------------------------------------------------------------
# Layout and optimizer-intent resolution.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("optimizer", "use_distopt", "collective", "error"),
    [
        ("muon", True, "reduce-scatter", None),
        ("muon", True, "all-reduce", None),
        ("muon", True, "ragged", "must be"),
        # all-reduce only changes the parameters an emerging optimizer owns whole; anywhere
        # else it would silently do nothing, so it is rejected.
        ("adam", True, "all-reduce", "applies to emerging optimizers"),
        ("muon", False, "all-reduce", "applies to emerging optimizers"),
    ],
)
def test_emerging_optimizer_grad_collective_validation(optimizer, use_distopt, collective, error):
    with nullcontext() if error is None else pytest.raises(ValueError, match=error):
        config = OptimizerConfig(
            optimizer=optimizer,
            use_distributed_optimizer=use_distopt,
            emerging_optimizer_grad_collective=collective,
        )
        # Depending on the branch, OptimizerConfig checks at construction or in validate().
        if hasattr(config, "validate"):
            config.validate()


@pytest.mark.parametrize(
    ("optimizer", "use_distopt", "use_layer_wise", "expected"),
    [
        ("adam", True, False, ("adam", False)),
        ("muon", False, False, ("muon", False)),
        ("muon", True, False, ("muon", True)),
        ("muon", False, True, ("muon", True)),
        ("dist_muon", False, False, ("muon", True)),
        ("lion", True, False, ("lion", True)),
    ],
)
def test_resolve_layer_wise_distributed_optimizer(optimizer, use_distopt, use_layer_wise, expected):
    assert (
        resolve_layer_wise_distributed_optimizer(optimizer, use_distopt, use_layer_wise) == expected
    )


# ---------------------------------------------------------------------------
# DDP wrapping decision shared by get_model() and the ModelBuilder path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("collective", "layout"),
    [(None, "padded"), ("reduce-scatter", "padded"), ("all-reduce", "compact")],
)
def test_configure_ddp_for_layer_wise_optimizer(collective, layout):
    model = _model()
    # The container clears use_distributed_optimizer for the layer-wise optimizer.
    ddp_config = DistributedDataParallelConfig()
    kwargs = {} if collective is None else {"emerging_optimizer_grad_collective": collective}
    fn = configure_ddp_for_layer_wise_optimizer(
        [model], ddp_config, use_layer_wise_distributed_optimizer=True, **kwargs
    )
    assert fn.func == LayerWiseDistributedOptimizer.compute_full_param_layout
    assert fn.keywords == {"param_layout": layout}
    assert ddp_config.use_distributed_optimizer
    assert model.fc1.is_managed_by_layer_wise_optimizer
    assert model.fc2.is_managed_by_layer_wise_optimizer
    for name in ("embedding", "fc1_bias", "router"):
        assert not getattr(model, name).is_managed_by_layer_wise_optimizer

    with pytest.raises(ValueError, match="must be one of"):
        configure_ddp_for_layer_wise_optimizer(
            [model],
            ddp_config,
            use_layer_wise_distributed_optimizer=True,
            emerging_optimizer_grad_collective="ragged",
        )


def _full_layout(sharded):
    """A full layout whose only LayerWise buffer is sharded (padded) or not (compact)."""
    key = type("BufferKey", (), {"is_managed_by_layer_wise_optimizer": True})()
    return SimpleNamespace(layouts={key: SimpleNamespace(use_distributed_optimizer=sharded)})


def test_resolve_param_layout_follows_the_grad_collective():
    resolve = LayerWiseDistributedOptimizer._resolve_param_layout
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    padded = SimpleNamespace(emerging_optimizer_grad_collective="reduce-scatter")
    compact = SimpleNamespace(emerging_optimizer_grad_collective="all-reduce")
    assert resolve(padded, ddp_config, [_full_layout(True)]) == "padded"
    assert resolve(compact, ddp_config, [_full_layout(False)]) == "compact"
    # DDP buffers built for one collective cannot be driven with the other.
    with pytest.raises(ValueError, match="does not match"):
        resolve(compact, ddp_config, [_full_layout(True)])
    # Chunks not wrapped for the layer-wise optimizer (e.g. the removed legacy layout).
    with pytest.raises(ValueError, match="wrapped for it"):
        resolve(padded, DistributedDataParallelConfig(use_distributed_optimizer=False), None)
    # Direct construction without DDP-wrapped chunks owns and gathers whole params.
    assert resolve(padded, None, None) == "compact"


@pytest.mark.parametrize(
    ("kind", "reuse_grad_buf", "error"),
    [
        ("bf16", False, None),
        ("mxfp8", True, None),
        # Recipes the all-reduce collective can publish point there ...
        ("mxfp8", False, "all-reduce"),
        ("blockwise", True, "all-reduce"),
        # ... and the others, which no LayerWise gather publishes, to another recipe.
        ("tensorwise", True, "mxfp8 or blockwise recipe"),
    ],
)
def test_padded_layout_rejects_quantized_params_it_cannot_publish(kind, reuse_grad_buf, error):
    # The padded gather moves raw bytes; only MXFP8 staged through the grad buffer is
    # requantized on every rank, so other FP8 params would keep stale scales off-owner.
    optimizer = SimpleNamespace(param_groups=[{"params": [_param((8, 8))]}])
    module = "megatron.core.optimizer.layer_wise_optimizer"
    with (
        mock.patch(f"{module}._param_uses_quantized_storage", return_value=kind != "bf16"),
        mock.patch(f"{module}.is_mxfp8tensor", return_value=kind == "mxfp8"),
        mock.patch(f"{module}.is_blockwise_float8tensor", return_value=kind == "blockwise"),
        nullcontext() if error is None else pytest.raises(ValueError, match=error),
    ):
        LayerWiseDistributedOptimizer._check_padded_param_gather_storage(
            [optimizer], SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buf)
        )


# ---------------------------------------------------------------------------
# Rank-independent whole-parameter owners of the compact layout.
# ---------------------------------------------------------------------------


def _ddp_chunk(params):
    """Exposes the DDP buffer metadata ``_build_param_sort_keys`` reads."""
    index_map, start = {}, 0
    for param in reversed(params):
        index_map[param] = (start, start + param.numel(), 0)
        start += param.numel()
    return SimpleNamespace(
        buffers=[SimpleNamespace(param_index_map=index_map)], expert_parallel_buffers=[]
    )


def _owners(params_in_group_order, sort_keys, dp_size, rank):
    """Run the ping-pong assignment for ``rank`` with mocked process groups."""
    groups = [{"params": [p], "is_expert_parallel": False} for p in params_in_group_order]
    optimizer = SimpleNamespace(param_groups=groups)
    lwo = LayerWiseDistributedOptimizer.__new__(LayerWiseDistributedOptimizer)
    lwo.dp_cp = "dp_cp"
    lwo.expt_dp = "expt_dp"
    with (
        mock.patch("megatron.core.optimizer.layer_wise_optimizer.get_pg_rank", return_value=rank),
        mock.patch(
            "megatron.core.optimizer.layer_wise_optimizer.get_pg_size", return_value=dp_size
        ),
    ):
        lwo.shard_params([optimizer], param_sort_keys=sort_keys)
    local = [p for group in groups for p in group["params"]]
    return lwo.dp_cp_params_list, local


def test_compact_owners_do_not_depend_on_param_group_order():
    # Equal-numel params are where a numel-only stable sort could disagree across ranks.
    params = [_param((8, 8)) for _ in range(6)] + [_param((4, 8)) for _ in range(3)]
    sort_keys = LayerWiseDistributedOptimizer._build_param_sort_keys([_ddp_chunk(params)])
    assert set(sort_keys) == set(params)

    reference, _ = _owners(params, sort_keys, dp_size=4, rank=0)
    rng = random.Random(1234)
    for rank in range(4):
        shuffled = list(params)
        rng.shuffle(shuffled)
        owner_lists, local = _owners(shuffled, sort_keys, dp_size=4, rank=rank)
        assert owner_lists == reference
        assert set(local) == set(reference[rank])
    # Every param has exactly one owner.
    assert sorted(id(p) for owned in reference for p in owned) == sorted(id(p) for p in params)
