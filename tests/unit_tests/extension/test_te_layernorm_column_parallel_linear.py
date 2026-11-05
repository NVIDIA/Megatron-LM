# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the Transformer Engine fused layernorm and column-parallel linear wrapper."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TELayerNormColumnParallelLinear,
    TENorm,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_sharded_state_dict_dp_cp_group import (
    forbid_global_dp_cp_group,
    shard_ids,
)


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
def test_fp32_residual_input_is_cast_without_mutating_residual():
    """Keep the residual FP32 while running the fused layer in BF16."""
    Utils.initialize_model_parallel(1, 1)

    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp32_residual_connection=True,
        )
        layer = TELayerNormColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=32,
            config=config,
            init_method=init_method_normal(config.init_method_std),
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        hidden_states = torch.randn(
            4, 2, config.hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
        )
        residual = hidden_states
        residual_before = residual.detach().clone()

        output, _ = layer(hidden_states)

        assert layer.layer_norm_weight.dtype == torch.bfloat16
        assert output.dtype == torch.bfloat16
        assert residual.dtype == torch.float32
        assert residual.data_ptr() == hidden_states.data_ptr()
        assert torch.equal(residual.detach(), residual_before)

        output.float().sum().backward()
        assert hidden_states.grad.dtype == torch.float32
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_rmsnorm_gamma_marker(normalization, zero_centered_gamma):
    """Standalone and fused TE norms mark only RMSNorm gamma for optimizer routing."""
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            normalization=normalization,
            layernorm_zero_centered_gamma=zero_centered_gamma,
            use_cpu_initialization=True,
        )
        norm = TENorm(config=config, hidden_size=config.hidden_size)
        layer = TELayerNormColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=32,
            config=config,
            init_method=init_method_normal(config.init_method_std),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
        )
        expected = normalization == "RMSNorm"
        assert getattr(norm.weight, "is_rmsnorm_weight", False) == expected
        assert getattr(layer.layer_norm_weight, "is_rmsnorm_weight", False) == expected
        assert not getattr(layer.weight, "is_rmsnorm_weight", False)
        assert not getattr(layer.bias, "is_rmsnorm_weight", False)
        if normalization == "LayerNorm":
            assert not getattr(norm.bias, "is_rmsnorm_weight", False)
            assert not getattr(layer.layer_norm_bias, "is_rmsnorm_weight", False)

        if expected:
            config.fused_residual_rmsnorm = True
            residual_norm = TENorm(config=config, hidden_size=config.hidden_size, has_residual=True)
            assert residual_norm.weight.is_rmsnorm_weight
    finally:
        Utils.destroy_model_parallel()


def _all_ranks(ids):
    """``ids`` of every rank, in rank order."""
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, ids)
    return gathered


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestShardedStateDictDpCpGroup:
    """Without metadata, replica ids span the DP x CP group of the layer's own collection."""

    def setup_method(self, method):
        if Utils.world_size % 4 != 0:
            pytest.skip("needs TP=2 with at least two data-parallel replicas")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _layer(pg_collection, tensor_model_parallel_size=2):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )
        return TELayerNormColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=32,
            config=config,
            init_method=init_method_normal(config.init_method_std),
            gather_output=False,
            bias=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=None if pg_collection is None else pg_collection.tp,
            pg_collection=pg_collection,
        ).cuda()

    def test_uses_the_collection_dp_cp_group(self):
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        layer = self._layer(pg_collection)
        own_metadata = {'dp_cp_group': pg_collection.dp_cp_gtp_remat}
        expected = shard_ids(layer.sharded_state_dict(metadata=own_metadata))

        with forbid_global_dp_cp_group():
            sharded_state_dict = layer.sharded_state_dict()

        assert shard_ids(sharded_state_dict) == expected

    def test_follows_a_grid_unlike_the_global_one(self):
        """On its own TP=1 grid the layer's ids differ from those of the global TP=2 grid."""
        world_size = torch.distributed.get_world_size()
        tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [[rank] for rank in range(world_size)]
        )
        dp_cp_group = torch.distributed.new_group(list(range(world_size)))
        pg_collection = ProcessGroupCollection(
            tp=tp_group, dp_cp=dp_cp_group, dp_cp_gtp_remat=dp_cp_group
        )
        layer = self._layer(pg_collection, tensor_model_parallel_size=1)
        expected = shard_ids(layer.sharded_state_dict(metadata={'dp_cp_group': dp_cp_group}))
        global_default = shard_ids(
            layer.sharded_state_dict(
                metadata={
                    'dp_cp_group': parallel_state.get_data_parallel_group(
                        with_context_parallel=True
                    )
                }
            )
        )

        with forbid_global_dp_cp_group():
            sharded_state_dict = layer.sharded_state_dict()

        assert shard_ids(sharded_state_dict) == expected
        # Rank 0 is rank 0 in both groups, so compare the ids of every rank.
        assert _all_ranks(expected) != _all_ranks(global_default)

    def test_without_a_collection_uses_the_global_dp_cp_group(self):
        layer = self._layer(None)
        global_metadata = {
            'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)
        }
        expected = shard_ids(layer.sharded_state_dict(metadata=global_metadata))

        assert shard_ids(layer.sharded_state_dict()) == expected
