# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TECudaGraphHelper uses the process groups its caller passes.

When the caller omits them, the helper warns and falls back to the same groups of the global
parallel state as before.
"""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.transformer.cuda_graphs import HAVE_TE_GRAPHS, TECudaGraphHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import (
    Utils,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)

pytestmark = pytest.mark.skipif(not HAVE_TE_GRAPHS, reason="needs Transformer Engine CUDA graphs")

# The groups that TECudaGraphHelper reads from its collection.
HELPER_GROUPS = ("tp", "dp_cp", "pp", "tp_cp", "tp_dp_cp")


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _helper(pg_collection=None):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        use_cpu_initialization=True,
        cuda_graph_impl="transformer_engine",
        bf16=True,
        pipeline_dtype=torch.bfloat16,
    )
    # A model chunk without a decoder has no graphable layers, which is enough to build the
    # helper and resolve its groups.
    return TECudaGraphHelper(
        model=[torch.nn.Linear(4, 4)],
        config=config,
        seq_length=8,
        micro_batch_size=1,
        pg_collection=pg_collection,
    )


@pytest.mark.skipif(Utils.world_size < 4 or Utils.world_size % 4, reason="needs a multiple of 4")
class TestTECudaGraphHelperProcessGroups:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_given_collection_is_used_without_global_reads(self):
        global_groups = vars(
            ProcessGroupCollection.use_mpu_process_groups(required_pgs=list(HELPER_GROUPS))
        )
        pg_collection = ProcessGroupCollection(
            **{name: new_group_with_same_ranks(global_groups[name]) for name in HELPER_GROUPS}
        )

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            helper = _helper(pg_collection)

        assert helper.pg_collection is pg_collection
        assert helper.tp_group is pg_collection.tp
        assert helper.dp_cp_group is pg_collection.dp_cp
        assert helper.pp_group is pg_collection.pp
        for name in HELPER_GROUPS:
            torch.distributed.destroy_process_group(vars(pg_collection)[name])

    def test_omitted_collection_warns_and_uses_the_global_groups(self):
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            helper = _helper()

        messages = [
            str(w.message)
            for w in record
            if "TECudaGraphHelper was called without `pg_collection`" in str(w.message)
        ]
        assert len(messages) == 1
        assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in messages[0]
        assert helper.tp_group is parallel_state.get_tensor_model_parallel_group()
        assert helper.dp_cp_group is parallel_state.get_data_parallel_group(
            with_context_parallel=True, with_gtp_remat=False
        )
        assert helper.pp_group is parallel_state.get_pipeline_model_parallel_group()
        assert helper.pg_collection.tp_cp is parallel_state.get_amax_reduction_group(
            with_context_parallel=True, tp_only_amax_red=True
        )
        assert helper.pg_collection.tp_dp_cp is parallel_state.get_amax_reduction_group(
            with_context_parallel=True, tp_only_amax_red=False
        )
