# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo
from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer, OptimizerConfig
from megatron.core.optimizer.checkpointing import (
    contains_distributed_optimizer,
    validate_optimizer_checkpoint_topology,
)
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.training import checkpointing


def _tree(*, distributed: bool, active: bool = True) -> MimoOptimizer:
    config = OptimizerConfig()
    compact = Mock(spec=LayerWiseDistributedOptimizer, config=config, model_chunks=[])
    children = [compact]
    if distributed:
        children.append(Mock(spec=DistributedOptimizer, config=config, model_chunks=[]))
    chain = ChainedOptimizer([ChainedOptimizer(children)])
    info = ModuleOptimizerInfo(optimizer=chain, grid=None, pg_collection=None, is_active=active)
    return MimoOptimizer({"encoder": info}, config)


@pytest.mark.parametrize(
    "distributed,active,expected", [(True, True, True), (False, True, False), (True, False, False)]
)
def test_only_active_distributed_leaves_select_metadata(run_config, distributed, active, expected):
    """Stale top-level flags do not hide an active distributed child in a nested wrapper."""
    optimizer = _tree(distributed=distributed, active=active)
    run_config.optimizer.use_distributed_optimizer = False
    metadata = checkpointing._build_sharded_state_dict_metadata(
        dp_cp_group=Mock(), optimizer=optimizer
    )
    assert contains_distributed_optimizer(optimizer) is expected
    assert metadata.get("distrib_optim_sharding_type") == ("dp_reshardable" if expected else None)


@pytest.mark.parametrize("current", [(2, 1), (1, 2)])
@pytest.mark.parametrize(
    "runtime, saved_leaf, global_layout, declared, rejects",
    [
        (True, None, "dp_reshardable", False, True),  # Mixed target, no saved tags.
        (False, "dp_reshardable", "fully_reshardable", False, True),  # Saved child is decisive.
        (False, None, "dp_zero_gather_scatter", True, True),  # Legacy ordinary DistOpt.
        (False, None, "dp_reshardable", False, False),  # Compact LayerWise only.
        (True, "fully_reshardable", "fully_reshardable", False, False),
        (True, None, "fully_sharded_model_space", True, False),
    ],
)
def test_resharding_considers_both_optimizer_trees(
    current, runtime, saved_leaf, global_layout, declared, rejects
):
    """Each tensor-sharded layout involved in the load must support a model topology change."""
    optimizer = _tree(distributed=runtime)
    state = {"encoder": ([{}, {"param_state_sharding_type": saved_leaf}],)}
    kwargs = dict(
        saved_tp_pp=(1, 1),
        current_tp_pp=current,
        sharding_type=global_layout,
        distributed_optimizer_configured=declared,
    )
    if rejects:
        with pytest.raises(
            RuntimeError, match=r"checkpoint\.dist_ckpt_optim_fully_reshardable=True"
        ):
            validate_optimizer_checkpoint_topology(optimizer, state, **kwargs)
        validate_optimizer_checkpoint_topology(None, state, **kwargs)
        validate_optimizer_checkpoint_topology(
            optimizer, state, **{**kwargs, "current_tp_pp": (1, 1)}
        )
    else:
        validate_optimizer_checkpoint_topology(optimizer, state, **kwargs)


@pytest.mark.parametrize("legacy", [False, True], ids=["mixed-tree", "legacy-source-only"])
def test_training_loader_rejects_before_reading_shards(run_config, monkeypatch, legacy):
    """Check a mixed target or a legacy DistOpt source with classic DistOpt disabled today."""
    run_config.checkpoint.ckpt_format = "torch_dist"
    run_config.checkpoint.load = "/checkpoint"
    run_config.optimizer.use_distributed_optimizer = False
    run_config.optimizer.use_layer_wise_distributed_optimizer = True
    args = SimpleNamespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        optimizer="adam",
        use_distributed_optimizer=legacy,
        ckpt_fully_parallel_save=False,
    )
    metadata = None if legacy else {"distrib_optim_sharding_type": "dp_reshardable"}
    layout = "dp_zero_gather_scatter" if legacy else "dp_reshardable"
    state = {"args": args, "optimizer": {}, "checkpoint_version": 3.1}
    read = Mock(
        return_value=(state, "/checkpoint/iter_0000002", False, checkpointing.CheckpointType.GLOBAL)
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(checkpointing, "_load_base_checkpoint", read)
    monkeypatch.setattr(checkpointing, "_read_checkpoint_run_config", lambda *args: None)
    monkeypatch.setattr(
        checkpointing.dist_checkpointing, "load_content_metadata", lambda **kwargs: metadata
    )
    with pytest.raises(RuntimeError, match=rf"{layout}.*--dist-ckpt-optim-fully-reshardable"):
        checkpointing.load_checkpoint(
            [], _tree(distributed=not legacy), None, load_rng=False, restore_training_state=True
        )
    read.assert_called_once()
