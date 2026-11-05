# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``GPTModel.sharded_state_dict()`` without metadata uses the model's own DP x CP group."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_sharded_state_dict_dp_cp_group import (
    forbid_global_dp_cp_group,
    shard_ids,
)


def _gpt_model(tp_size, pg_collection):
    # Dimensions stay multiples of the GTP alignment, so no padding is added.
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=8,
        kv_channels=8,
        ffn_hidden_size=128,
        use_cpu_initialization=False,
        tensor_model_parallel_size=tp_size,
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=128,
        max_sequence_length=4,
        pg_collection=pg_collection,
    )


def _all_ranks(ids):
    """``ids`` of every rank, in rank order."""
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, ids)
    return gathered


class TestGPTShardedStateDictDpCpGroup:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "tp_size, gtp_size, model_gtp",
        [
            (2, 1, False),
            # GTP grid; the model keeps its weights whole, so they replicate over GTP peers too.
            (1, 2, False),
            # GTP grid with GTP-sharded weights.
            (1, 2, True),
        ],
    )
    def test_without_metadata_matches_the_global_default(self, tp_size, gtp_size, model_gtp):
        """Same keys, offsets and replica ids as the global default, with no global read.

        On a GTP grid the replica ids must come from the GTP-inclusive group: the replicate
        ``dp_cp`` group would make GTP peers claim the same replica.
        """
        if Utils.world_size % (2 * tp_size * gtp_size) != 0:
            pytest.skip("needs at least two replicate data-parallel ranks")
        if model_gtp and not HAVE_GTP:
            pytest.skip("GTP requires TransformerEngine >= 2.19")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size, gtp_remat_size=gtp_size)
        model_parallel_cuda_manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        if not model_gtp:
            pg_collection.gtp_remat = None
            pg_collection.expt_gtp_remat = None
        model = _gpt_model(tp_size, pg_collection)
        global_dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        expected = shard_ids(model.sharded_state_dict(metadata={'dp_cp_group': global_dp_cp_group}))

        with forbid_global_dp_cp_group():
            sharded_state_dict = model.sharded_state_dict()

        assert shard_ids(sharded_state_dict) == expected
        if gtp_size > 1:
            replicate_dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, with_gtp_remat=False
            )
            replicate_ids = shard_ids(
                model.sharded_state_dict(metadata={'dp_cp_group': replicate_dp_cp_group})
            )
            # Some ranks get the same ids from both groups (rank 0 is rank 0 in each), so compare
            # the ids of every rank.
            assert _all_ranks(replicate_ids) != _all_ranks(expected)
