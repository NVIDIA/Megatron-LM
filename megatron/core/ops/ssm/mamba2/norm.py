# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

"""Mamba gated RMSNorm and its checkpoint sharding, loaded only when selected."""

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm

from megatron.core import parallel_state
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


class ExtendedRMSNorm(RMSNorm):
    """RMSNorm with the existing Mamba weight-sharding contract."""

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Shard the weight along axis zero without changing its checkpoint key."""
        if not hasattr(self, 'tp_group'):
            # Preserve the legacy fallback for callers that did not supply a TP group.
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(prefix="", keep_vars=True),
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
