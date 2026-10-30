# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The config logger names each dump by the ranks of the logging model's process groups.

When the caller passes no complete collection, it warns and names the dump by the ranks of the
global parallel state, as before.
"""

import os
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import config_logger, parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from tests.unit_tests.test_utilities import (
    Utils,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)

RANK_GROUPS = ("tp", "gtp_remat", "dp_gtp_remat", "cp", "pp", "ep")


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _dump(directory, **kwargs):
    """Log one dump into ``directory`` and return the rank string of its file name."""
    config = SimpleNamespace(config_logger_dir=str(directory))
    config_logger.log_config_to_disk(config, {"key": 1}, prefix="dump", **kwargs)
    (name,) = os.listdir(directory)
    assert name.startswith("dump.rank_") and name.endswith(".iter0.json")
    return name[len("dump.rank_") : -len(".iter0.json")]


@pytest.mark.skipif(Utils.world_size < 2, reason="needs at least 2 ranks")
class TestConfigLoggerRankString:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_complete_collection_names_the_dump_without_global_reads(self, tmp_path):
        global_groups = vars(
            ProcessGroupCollection.use_mpu_process_groups(required_pgs=list(RANK_GROUPS))
        )
        pg_collection = ProcessGroupCollection(**global_groups)
        # The model's tensor-parallel group spans every rank, unlike the global one (TP=1).
        world = list(range(torch.distributed.get_world_size()))
        pg_collection.tp, _ = torch.distributed.new_subgroups_by_enumeration([world])
        pg_collection.ep = new_group_with_same_ranks(global_groups["ep"])

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            rank_str = _dump(tmp_path, pg_collection=pg_collection)

        expected = "_".join(
            "0" if vars(pg_collection)[name] is None else str(vars(pg_collection)[name].rank())
            for name in RANK_GROUPS
        )
        assert rank_str == expected
        assert rank_str.split("_")[0] == str(torch.distributed.get_rank())
        torch.distributed.destroy_process_group(pg_collection.tp)
        torch.distributed.destroy_process_group(pg_collection.ep)

    def test_omitted_collection_warns_and_uses_the_global_ranks(self, tmp_path):
        with pytest.warns(
            ProcessGroupFallbackWarning,
            match="log_config_to_disk was called without `pg_collection`",
        ):
            rank_str = _dump(tmp_path)

        assert rank_str == parallel_state.get_all_ranks()

    def test_partial_collection_warns_and_uses_the_global_ranks(self, tmp_path):
        pg_collection = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group())

        with pytest.warns(ProcessGroupFallbackWarning, match="log_config_to_disk"):
            rank_str = _dump(tmp_path, pg_collection=pg_collection)

        assert rank_str == parallel_state.get_all_ranks()

    def test_explicit_rank_string_reads_no_group(self, tmp_path):
        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert _dump(tmp_path, rank_str="custom") == "custom"

    def test_ddp_passes_its_collection(self):
        """DistributedDataParallel names its dump by the collection it was given."""
        from megatron.core.distributed import (
            DistributedDataParallel,
            DistributedDataParallelConfig,
            distributed_data_parallel,
        )
        from megatron.core.transformer import TransformerConfig

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(num_attention_heads=1, num_layers=1, config_logger_dir="unused")
        module = torch.nn.Linear(8, 8).cuda()

        with mock.patch.object(distributed_data_parallel, "log_config_to_disk") as log_config:
            DistributedDataParallel(
                config, DistributedDataParallelConfig(), module, pg_collection=pg_collection
            )

        log_config.assert_called_once()
        assert log_config.call_args.kwargs["pg_collection"] is pg_collection

    @pytest.mark.parametrize("float16", [False, True])
    def test_optimizers_pass_their_collection(self, float16):
        """The FP32 and mixed-precision optimizers name their dumps by their collection."""
        from megatron.core.optimizer import OptimizerConfig
        from megatron.core.optimizer import optimizer as optimizer_module

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config = OptimizerConfig(bf16=float16, config_logger_dir="unused")

        # A stub optimizer (None) is enough: the dump is logged before any parameter is touched.
        with mock.patch.object(optimizer_module, "log_config_to_disk") as log_config:
            if float16:
                optimizer_module.Float16OptimizerWithFloat16Params(
                    None, config, None, None, pg_collection=pg_collection
                )
            else:
                optimizer_module.FP32Optimizer(None, config, None, pg_collection=pg_collection)

        log_config.assert_called_once()
        assert log_config.call_args.kwargs["pg_collection"] is pg_collection
