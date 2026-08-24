# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def test_mimo_optimizer_scopes_param_group_alignment_to_module(mocker):
    intra_dist_opt = object()
    pg_collection = SimpleNamespace(intra_dist_opt=intra_dist_opt)
    module = SimpleNamespace(
        pg_collection=pg_collection,
        ddp_config=SimpleNamespace(
            overlap_param_gather=True, num_distributed_optimizer_instances=1
        ),
    )
    grid = SimpleNamespace(is_current_rank_in_grid=lambda: True)
    mimo_model = SimpleNamespace(
        mimo_config=SimpleNamespace(module_to_grid_map={MIMO_LANGUAGE_MODULE_KEY: grid}),
        language_model=module,
        modality_submodules={},
    )
    optimizer_factory = mocker.patch(
        "megatron.core.optimizer.get_megatron_optimizer", return_value=MagicMock()
    )

    get_mimo_optimizer(mimo_model, OptimizerConfig())

    assert optimizer_factory.call_args.kwargs['param_group_process_group'] is intra_dist_opt
    assert optimizer_factory.call_args.kwargs['pg_collection'] is pg_collection
