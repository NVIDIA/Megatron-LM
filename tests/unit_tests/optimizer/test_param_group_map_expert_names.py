# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The parameter-to-group dump names experts by the caller's expert-parallel layout."""

import torch

import megatron.core.optimizer as optimizer_module
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestParamGroupMapExpertNames:

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_dump_passes_the_collection_expert_layout(self, monkeypatch):
        world_size, rank = Utils.world_size, torch.distributed.get_rank()
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(),
            torch.nn.Linear(16, 16, device='cuda'),
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # An expert-parallel layout that differs from the global one (EP=1).
        pg_collection.ep = torch.distributed.new_group(ranks=list(range(world_size)))

        naming_kwargs = []

        def record_name(model_chunks, param, **kwargs):
            naming_kwargs.append(kwargs)
            return f'param{len(naming_kwargs)}'

        monkeypatch.setattr(optimizer_module, 'get_global_unique_param_name', record_name)
        monkeypatch.setattr(torch.distributed.checkpoint, 'save', lambda **kwargs: None)

        get_megatron_optimizer(
            OptimizerConfig(optimizer='adam', lr=1e-3),
            [model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
            dump_param_to_param_group_map='unused',
        )

        assert naming_kwargs
        assert all(kwargs == {'ep_rank': rank, 'ep_size': world_size} for kwargs in naming_kwargs)
