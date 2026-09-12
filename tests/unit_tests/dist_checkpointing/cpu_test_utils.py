# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Synchronous checkpoint I/O for CPU layout and resharding regression tests."""

from pathlib import Path

import torch.distributed.checkpoint as torch_dcp

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreSavePlanner,
    TorchDistSaveShardedStrategy,
    _replace_state_dict_keys_with_sharded_keys,
    mcore_to_pyt_state_dict,
)


class TorchDistCPUSaveShardedStrategy(TorchDistSaveShardedStrategy):
    """Use PyTorch's synchronous writer; MCore's async writer requires CUDA staging.

    Pass this strategy to ``dist_checkpointing.save`` to retain the normal MCore
    preprocessing and access-integrity checks with actual on-disk DCP tensors.
    """

    def save(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> None:
        """Save with the normal MCore tensor conversion and PyTorch save planner."""
        grouped, _, _ = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        torch_dcp.save(
            mcore_to_pyt_state_dict(grouped, False),
            checkpoint_id=checkpoint_dir,
            planner=MCoreSavePlanner(
                dedup_replicated_tensors=not self.keep_only_main_replica,
                flatten_state_dict=False,
                flatten_sharded_tensors=False,
            ),
        )
