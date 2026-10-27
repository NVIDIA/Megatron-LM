# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Distributed checkpoint compatibility for split wide-residual connections."""

import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.mapping import ShardedBase
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import wide_residual_hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def test_wide_residual_loads_legacy_distributed_checkpoint(tmp_path_dist_ckpt):
    """Load pre-split attention, Mamba, and MLP weights through the real disk loader."""
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
    )
    model = HybridStack(
        config=config,
        submodules=wide_residual_hybrid_stack_spec.submodules,
        layer_config_list=validate_segment_layers("*M-", config),
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    # Distinct values catch missing read/write/retention tensors, including parameters
    # whose default initialization would otherwise be restored by doing nothing.
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) / 128)
    expected = {name: p.detach().clone() for name, p in model.named_parameters()}

    legacy_state = model.sharded_state_dict()
    for value in legacy_state.values():
        if isinstance(value, ShardedBase):
            # Reproduce the on-disk names before the read/write split, independently
            # of the production compatibility map. Non-residual names stay unchanged.
            value.key = value.key.replace(".residual_read", ".residual_connection").replace(
                ".residual_write", ".residual_connection"
            )

    with TempNamedDir(tmp_path_dist_ckpt / "wide_residual_legacy") as legacy_dir:
        save(legacy_state, legacy_dir)
        legacy_keys = set(load_sharded_metadata(legacy_dir))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.load_state_dict(load(model.sharded_state_dict(), legacy_dir), strict=True)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

        # New saves must retain the same external tensor names, not merely accept
        # old names in Module.load_state_dict after the distributed read has finished.
        with TempNamedDir(tmp_path_dist_ckpt / "wide_residual_resaved") as resaved_dir:
            save(model.sharded_state_dict(), resaved_dir)
            assert set(load_sharded_metadata(resaved_dir)) == legacy_keys
    Utils.destroy_model_parallel()
