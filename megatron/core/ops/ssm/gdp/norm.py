# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gated RMSNorm with GDP checkpoint sharding; loaded only when selected."""

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm

from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)


class ExtendedRMSNorm(RMSNorm):
    """Shard the norm's weight over the owning mixer's explicit process groups."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Shard the weight along axis zero without changing its checkpoint key."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
