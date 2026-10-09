# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sharded state dicts compute replica ids over the DP x CP group of the module that builds them.

Called without metadata, ``sharded_state_dict`` used to fill in the global
``get_data_parallel_group(with_context_parallel=True)``. A model on its own grid then claimed
replicas by another grid's DP ranks, so several ranks or none wrote each shard.
"""

from unittest import mock

import pytest

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import utils as transformer_utils
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
from tests.unit_tests.test_utilities import Utils


class _ParallelStateWithoutGlobalDpCpGroup:
    """``parallel_state`` as ``transformer.utils`` sees it, without the global DP x CP group."""

    def __getattr__(self, name):
        if name == 'get_data_parallel_group':
            raise AssertionError("read the global data-parallel group")
        return getattr(parallel_state, name)


def forbid_global_dp_cp_group():
    """Make the global DP x CP fallback in ``transformer.utils`` raise."""
    return mock.patch.object(
        transformer_utils, 'parallel_state', _ParallelStateWithoutGlobalDpCpGroup()
    )


def shard_ids(sharded_state_dict):
    """``(key, global_offset, replica_id)`` of every sharded entry, sorted."""
    return sorted(
        (entry.key, getattr(entry, 'global_offset', None), entry.replica_id)
        for entry in nested_values(sharded_state_dict)
    )


class TestEnsureMetadataHasDpCpGroup:

    def test_fills_missing_metadata_with_the_given_group(self):
        group = object()
        metadata = {'singleton_local_shards': True}

        with forbid_global_dp_cp_group():
            assert ensure_metadata_has_dp_cp_group(None, group) == {'dp_cp_group': group}
            assert ensure_metadata_has_dp_cp_group(metadata, group) is metadata

        assert metadata == {'singleton_local_shards': True, 'dp_cp_group': group}

    def test_group_in_metadata_takes_precedence(self):
        own_group, given_group = object(), object()

        with forbid_global_dp_cp_group():
            metadata = ensure_metadata_has_dp_cp_group({'dp_cp_group': own_group}, given_group)

        assert metadata['dp_cp_group'] is own_group

    def test_without_a_group_uses_the_global_dp_cp_group(self):
        Utils.initialize_model_parallel()
        try:
            metadata = ensure_metadata_has_dp_cp_group(None)
            assert metadata['dp_cp_group'] is parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
        finally:
            Utils.destroy_model_parallel()


class TestModulesSupplyTheirOwnDpCpGroup:

    def setup_method(self, method):
        if Utils.world_size % 4 != 0:
            pytest.skip("needs TP=2 with at least two data-parallel replicas")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_transformer_block_and_mlp_without_metadata(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=2,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        block = TransformerBlock(
            config, get_gpt_layer_with_transformer_engine_spec(), pg_collection=pg_collection
        )
        mlp = block.layers[0].mlp
        own_metadata = {'dp_cp_group': pg_collection.dp_cp_gtp_remat}
        expected_block = shard_ids(block.sharded_state_dict(metadata=dict(own_metadata)))
        expected_mlp = shard_ids(mlp.sharded_state_dict(metadata=dict(own_metadata)))

        with forbid_global_dp_cp_group():
            block_sharded_state_dict = block.sharded_state_dict()
            mlp_sharded_state_dict = mlp.sharded_state_dict()

        assert shard_ids(block_sharded_state_dict) == expected_block
        assert shard_ids(mlp_sharded_state_dict) == expected_mlp
