# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The config logger names each dump by the ranks of the logging model's process groups.

When the caller passes no complete collection, it warns and names the dump by the ranks of the
global parallel state, as before.
"""

import json
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


class _StopAfterDump(Exception):
    """Ends the construction of a model once it has logged its config."""


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

    @pytest.mark.parametrize(
        "module_name, class_name, arguments",
        [
            ("megatron.core.models.bert.bert_model", "BertModel", (0, None, 8, 4)),
            ("megatron.core.models.hybrid.hybrid_model", "HybridModel", (None, 8, 4)),
            ("megatron.core.models.T5.t5_model", "T5LMHead", (False, 8)),
            ("megatron.core.models.vision.clip_vit_model", "CLIPViTModel", (None,)),
            ("megatron.core.models.vision.radio", "RADIOViTModel", (None,)),
            (
                "megatron.core.models.multimodal.llava_model",
                "LLaVAModel",
                (None, 8, 4, None, None, False, None, None),
            ),
        ],
        ids=["bert", "hybrid", "t5-lm-head", "clip-vit", "radio-vit", "llava"],
    )
    def test_models_pass_their_collection(self, module_name, class_name, arguments):
        """Each of these models names its dump by the collection it was given.

        They log before they build a layer or read another argument, so the logger ends the
        construction and the arguments after the config are placeholders.
        """
        import importlib

        from megatron.core.transformer import TransformerConfig

        module = importlib.import_module(module_name)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(num_attention_heads=1, num_layers=1, config_logger_dir="unused")

        with mock.patch.object(
            module, "log_config_to_disk", side_effect=_StopAfterDump
        ) as log_config:
            with pytest.raises(_StopAfterDump):
                getattr(module, class_name)(config, *arguments, pg_collection=pg_collection)

        log_config.assert_called_once()
        assert log_config.call_args.kwargs["pg_collection"] is pg_collection

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

    @pytest.mark.parametrize("distributed", [False, True], ids=["fp32", "distributed"])
    def test_get_megatron_optimizer_passes_its_collection_to_the_dumps(self, distributed):
        """The optimizers that get_megatron_optimizer builds name their dumps by its collection."""
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import (
            OptimizerConfig,
            distrib_optimizer,
            get_megatron_optimizer,
        )
        from megatron.core.optimizer import optimizer as optimizer_module
        from megatron.core.transformer import TransformerConfig

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(use_distributed_optimizer=distributed),
            torch.nn.Linear(16, 16).cuda(),
            pg_collection=pg_collection,
        )
        config = OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            use_distributed_optimizer=distributed,
            config_logger_dir="unused",
        )

        with (
            mock.patch.object(optimizer_module, "log_config_to_disk") as base_log,
            mock.patch.object(distrib_optimizer, "log_config_to_disk") as distributed_log,
        ):
            get_megatron_optimizer(
                config, [model], use_gloo_process_groups=False, pg_collection=pg_collection
            )

        # A distributed optimizer logs its own dump and the dump of its mixed-precision base.
        assert base_log.call_count == 1
        assert distributed_log.call_count == (1 if distributed else 0)
        for call in base_log.call_args_list + distributed_log.call_args_list:
            assert call.kwargs["pg_collection"] is pg_collection

    def test_factory_passes_its_collection_to_the_fully_sharded_optimizer(self):
        """The MFSDP v2 branch of the optimizer factory passes the collection on as well."""
        import megatron.core.optimizer as optimizer_package

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # __new__ gives a chunk that passes the factory's isinstance check without a model.
        chunk_class = optimizer_package.FullyShardedDataParallelV2
        model_chunk = chunk_class.__new__(chunk_class)

        with mock.patch.object(optimizer_package, "FullyShardedOptimizer") as fully_sharded:
            optimizer_package._get_megatron_optimizer_based_on_param_groups(
                config=optimizer_package.OptimizerConfig(use_distributed_optimizer=False),
                model_chunks=[model_chunk],
                param_groups=[],
                pg_collection=pg_collection,
            )

        fully_sharded.assert_called_once()
        assert fully_sharded.call_args.kwargs["pg_collection"] is pg_collection

    def test_fully_sharded_optimizer_passes_its_collection(self):
        """The MFSDP v2 optimizer names its dump, and the dump of its base, by its collection."""
        from megatron.core.optimizer import OptimizerConfig, fully_sharded_optimizer
        from megatron.core.optimizer import optimizer as optimizer_module

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config = OptimizerConfig(
            lr=1.0, use_distributed_optimizer=False, config_logger_dir="unused"
        )
        # A dense stand-in for an MFSDP v2 model chunk, with the attributes the optimizer reads.
        model_chunk = torch.nn.Linear(2, 2, bias=False)
        model_chunk.config = SimpleNamespace(num_moe_experts=None)
        model_chunk.ddp_config = SimpleNamespace()

        with (
            mock.patch.object(optimizer_module, "log_config_to_disk") as base_log,
            mock.patch.object(fully_sharded_optimizer, "log_config_to_disk") as fsdp_log,
        ):
            fully_sharded_optimizer.FullyShardedOptimizer(
                torch.optim.SGD(model_chunk.parameters(), lr=1.0),
                config,
                None,
                None,
                model_chunks=[model_chunk],
                pg_collection=pg_collection,
            )

        for log_config in (base_log, fsdp_log):
            log_config.assert_called_once()
            assert log_config.call_args.kwargs["pg_collection"] is pg_collection

    def test_optimizer_dump_holds_its_collection(self, tmp_path):
        """An optimizer built on a collection writes its dump, with the collection in it."""
        from megatron.core.optimizer import OptimizerConfig
        from megatron.core.optimizer.optimizer import FP32Optimizer

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        config = OptimizerConfig(config_logger_dir=str(tmp_path))

        # A stub optimizer (None) is enough: the dump is logged before any parameter is touched.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            FP32Optimizer(None, config, None, pg_collection=pg_collection)

        (name,) = os.listdir(tmp_path)
        assert name.startswith("FP32Optimizer.rank_") and name.endswith(".iter0.json")
        with open(tmp_path / name) as dump_file:
            dump = json.load(dump_file)
        # The process groups cannot be copied into the dump, so the collection is dumped as
        # its repr, which names the groups it sets and their sizes.
        assert dump["pg_collection"] == repr(pg_collection)
