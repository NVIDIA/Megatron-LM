# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert-parallel non-emerging (Adam) param groups on the layer-wise + DistOpt path.

In layer-wise mode the emerging optimizer (Muon) owns the matrix parameters through
``LayerWiseDistributedOptimizer``, and everything else is handed to a sibling
``DistributedOptimizer``. That sibling used to be dense-only, so a parameter that is both
expert-parallel and non-Muon had nowhere to go. It is not a hypothetical combination: a hashed
n-gram / PLE memory is EP-sharded (``allreduce=False``) *and* tagged
``is_embedding_or_output_parameter``, so it is expert-parallel and Adam-managed at once.

Tested here:

1. such a model builds a three-way chain and the expert sibling shards over the expert-DP group;
2. a Muon model *without* expert-parallel non-Muon params builds exactly the chain it built
   before -- the containment property that keeps this change off every other training route;
3. a pipeline stage that holds none of the expert-parallel Adam params still builds the (empty)
   expert sibling, so every rank builds the same chain and can step.
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size
from tests.unit_tests.test_utilities import Utils

# The layer-wise optimizer tests do not run on the LTS image (as in test_layer_wise_optimizer.py).
_skip_on_lts = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)


class _MatrixOnlyModel(nn.Module):
    """Plain 2-D weights: every weight is Muon-eligible, biases go to the scalar optimizer."""

    def __init__(self, input_size=32, hidden_size=16, output_size=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class _ExpertTableModel(_MatrixOnlyModel):
    """Adds an EP-sharded, embedding-tagged table -- expert-parallel *and* Adam-managed.

    The two tags are what a real hashed-memory module sets: ``allreduce=False`` makes the
    parameter expert-parallel, and ``is_embedding_or_output_parameter=True`` keeps it out of
    Newton-Schulz orthogonalization.
    """

    def __init__(self, table_rows=12, table_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.table = nn.Parameter(torch.empty(table_rows, table_dim))
        nn.init.normal_(self.table, std=0.02)
        self.table.allreduce = False
        self.table.is_embedding_or_output_parameter = True

    def forward(self, x):
        return super().forward(x) + self.table.sum()


def _build_optimizer(model_class):
    """Wrap ``model_class`` in DDP the way the layer-wise path requires, and build the optimizer."""
    from megatron.training.training import wrap_model_chunks_with_ddp

    model = model_class().bfloat16().cuda()
    model.requires_grad_(True)

    transformer_config = TransformerConfig(num_attention_heads=1, num_layers=1)
    # use_layer_wise_distributed_optimizer=True makes wrap_model_chunks_with_ddp force
    # ddp_config.use_distributed_optimizer=True, which is what selects the split path.
    model = wrap_model_chunks_with_ddp(
        [model],
        transformer_config,
        DistributedDataParallelConfig(),
        use_layer_wise_distributed_optimizer=True,
    )[0]
    model.broadcast_params()

    optimizer_config = OptimizerConfig(
        optimizer='muon',
        lr=0.01,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=False,
        clip_grad=1.0,
        muon_tp_mode="duplicated",
        use_layer_wise_distributed_optimizer=True,
    )

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

    optimizer = get_megatron_optimizer(
        optimizer_config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
    )
    return model, optimizer, pg_collection


def _train_step(model, optimizer):
    """One forward, backward and optimizer step on a random batch."""
    optimizer.zero_grad()
    out = model(torch.randn(4, 32, dtype=torch.bfloat16, device='cuda'))
    out.sum().backward()
    model.finish_grad_sync()
    return optimizer.step()


def _ranks(group):
    return torch.distributed.get_process_group_ranks(group)


def _owned_numel(distopt):
    """Number of main-param elements this rank's shard of ``distopt`` holds."""
    return sum(p.numel() for group in distopt.optimizer.param_groups for p in group['params'])


@_skip_on_lts
class TestLayerWiseExpertDistOpt:
    """Expert-parallel Adam groups get their own DistOpt instance on the layer-wise path."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("expert_model_parallel_size=2 needs an even world size")
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    def test_expert_parallel_adam_group_gets_its_own_distopt(self):
        """An EP-sharded Adam table builds a third chain member on the expert-DP group.

        Before the fix this raised "Non-emerging expert-parallel param groups are not yet
        supported"; there was nowhere to put the table.
        """
        _, optimizer, pg_collection = _build_optimizer(_ExpertTableModel)

        assert isinstance(optimizer, ChainedOptimizer)
        children = optimizer.chained_optimizers
        assert len(children) == 3, (
            "expected LayerWise + dense DistOpt + expert DistOpt, got "
            f"{[type(c).__name__ for c in children]}"
        )
        assert isinstance(children[0], LayerWiseDistributedOptimizer)

        # The two siblings must shard over different data-parallel groups: the dense one over
        # DP, the expert one over expert-DP -- the groups DDP reduce-scatters each buffer over.
        dense, expert = children[1], children[2]
        assert _ranks(dense.data_parallel_group) == _ranks(
            parallel_state.get_data_parallel_group(with_context_parallel=True)
        )
        assert _ranks(expert.data_parallel_group) == _ranks(
            parallel_state.get_expert_data_parallel_group()
        )
        # The index keys the dp_reshardable checkpoint objects, so it must differ between
        # expert-DP groups: the rank in the expert TP/PP group, as in the standard path. The
        # dense model-parallel rank would collide across EP ranks.
        assert dense.data_parallel_group_idx == get_pg_rank(pg_collection.mp)
        assert expert.data_parallel_group_idx == get_pg_rank(pg_collection.tp_ep_pp_with_egtp_remat)

    def test_expert_table_state_is_sharded_over_expert_dp(self):
        """The point of the exercise: the table's optimizer state is split, not replicated."""
        model, optimizer, _ = _build_optimizer(_ExpertTableModel)

        expt_dp_size = get_pg_size(parallel_state.get_expert_data_parallel_group())
        if expt_dp_size == 1:
            pytest.skip("expert-DP size 1 shards nothing; run with EP < world size")

        expert = optimizer.chained_optimizers[2]
        # Sum of this rank's shard across the param groups the expert sibling owns.
        owned = _owned_numel(expert)
        table_numel = model.module.table.numel()
        assert owned < table_numel, (
            f"expert sibling holds {owned} elements of a {table_numel}-element table; "
            "expected a shard, so the state is still replicated"
        )

    def test_step_runs(self):
        """A full step works end to end and updates the table identically on its replicas."""
        model, optimizer, _ = _build_optimizer(_ExpertTableModel)
        table = model.module.table
        table_before = table.detach().clone()

        update_successful, grad_norm, _ = _train_step(model, optimizer)

        assert update_successful
        assert grad_norm is None or torch.isfinite(torch.as_tensor(grad_norm))
        assert not torch.equal(table.detach(), table_before), "the expert sibling did not step"
        # The sibling reduce-scatters the table gradient and all-gathers the update over
        # expert-DP, so the table's expert-DP replicas must hold the same values.
        expt_dp_group = parallel_state.get_expert_data_parallel_group()
        replicas = [torch.empty_like(table) for _ in range(get_pg_size(expt_dp_group))]
        torch.distributed.all_gather(replicas, table.detach(), group=expt_dp_group)
        for replica in replicas[1:]:
            torch.testing.assert_close(replica, replicas[0], rtol=0, atol=0)


@_skip_on_lts
class TestLayerWiseExpertDistOptContainment:
    """A Muon model with no expert-parallel Adam params must be unaffected by the change."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("expert_model_parallel_size=2 needs an even world size")
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    def test_no_expert_group_builds_the_original_two_member_chain(self):
        """Without an expert-parallel non-Muon param the chain is LayerWise + one dense DistOpt.

        This is the regression guard for the containment property: the expert partition must
        stay empty and contribute no optimizer, exactly as before the change.
        """
        _, optimizer, pg_collection = _build_optimizer(_MatrixOnlyModel)

        assert isinstance(optimizer, ChainedOptimizer)
        children = optimizer.chained_optimizers
        assert len(children) == 2, (
            "a model with no expert-parallel non-Muon params must build the same two-member "
            f"chain it built before the change, got {[type(c).__name__ for c in children]}"
        )
        assert isinstance(children[0], LayerWiseDistributedOptimizer)
        dense = children[1]
        assert _ranks(dense.data_parallel_group) == _ranks(
            parallel_state.get_data_parallel_group(with_context_parallel=True)
        )
        assert dense.data_parallel_group_idx == get_pg_rank(pg_collection.mp)


@_skip_on_lts
class TestLayerWiseExpertDistOptPipeline:
    """Every pipeline stage builds the same chain, whether or not it holds the table."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        if Utils.world_size < 4 or Utils.world_size % 4:
            pytest.skip("pipeline_model_parallel_size=2 x expert_model_parallel_size=2")
        Utils.initialize_model_parallel(
            pipeline_model_parallel_size=2, expert_model_parallel_size=2
        )
        yield
        Utils.destroy_model_parallel()

    def test_stage_without_the_table_builds_an_empty_expert_distopt(self):
        """A stage that holds no table still builds the expert sibling, with nothing in it.

        Param-group keys are aligned across ranks, so that stage still sees the table's group,
        with no params, and must build a matching sibling to keep the chains in step. Every
        stage then steps, which takes the empty sibling through grad-norm and update as well.
        """
        has_table = parallel_state.is_pipeline_first_stage()
        model, optimizer, _ = _build_optimizer(_ExpertTableModel if has_table else _MatrixOnlyModel)

        children = optimizer.chained_optimizers
        assert len(children) == 3, (
            "every stage must build LayerWise + dense DistOpt + expert DistOpt, got "
            f"{[type(c).__name__ for c in children]} (has_table={has_table})"
        )
        expert = children[2]
        assert _ranks(expert.data_parallel_group) == _ranks(
            parallel_state.get_expert_data_parallel_group()
        )
        if not has_table:
            assert _owned_numel(expert) == 0

        update_successful, grad_norm, _ = _train_step(model, optimizer)
        assert update_successful
        assert grad_norm is None or torch.isfinite(torch.as_tensor(grad_norm))
