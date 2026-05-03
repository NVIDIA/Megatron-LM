"""
Parameter Group for FSDP

Groups parameters that share the same (device, dtype, requires_grad) and
manages their buffers collectively. This enables efficient memory management
and collective operations across parameters.
"""

import math
from typing import Dict, List, Optional

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate, Shard

from ..uneven_dtensor import make_uneven_dtensor
from .allocator import TemporaryBucketAllocator
from .dp_buffer import DataParallelBuffer
from .utils import ParamGroupIdx


class ParameterGroup:
    """
    Groups parameters sharing same properties for collective buffer management.

    All parameters in a group have the same:
    - device (cuda device)
    - dtype (data type)
    - requires_grad (whether gradients are needed)

    The group manages:
    - model_weight_buffer: stores sharded model weights
    - main_weight_buffer: optional high-precision copy for mixed precision
    - main_grad_buffer: accumulates gradients before reduction
    - dist_params: DTensor views into the buffer
    - dist_grads: DTensor gradient views
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_group_id: ParamGroupIdx,
        *,
        mesh: Optional[DeviceMesh] = None,
        sharding_strategy: str = "optim_grads_params",
        main_params_dtype: Optional[torch.dtype] = None,
        main_grads_dtype: Optional[torch.dtype] = None,
        gradient_scaling_factor: Optional[float] = None,
        allocator: Optional[TemporaryBucketAllocator] = None,
    ):
        self.params = params
        self.param_idx: Dict[torch.nn.Parameter, int] = {p: i for i, p in enumerate(params)}

        # Assume all params have same device/dtype/require_grad
        # TODO: validate all params have same properties
        self.device = params[0].device
        self.dtype = params[0].dtype
        self.requires_grad = params[0].requires_grad

        # Setup device mesh and derived process group
        self.mesh = mesh
        if mesh is not None:
            assert mesh.ndim == 1, "Only 1D mesh is supported"
            self.dp_group = mesh.get_group()
        else:
            self.dp_group = torch.distributed.group.WORLD

        self.sharding_strategy = sharding_strategy
        self.param_group_id = param_group_id

        # Compute chunk size factor for alignment
        # LCM ensures params align to common boundary for efficient sharding
        if len(params) > 0 and any(p.shape[1:].numel() > 0 for p in params):
            self.chunk_size_factor = max(1, math.lcm(*[p.shape[1:].numel() for p in params]))
        else:
            self.chunk_size_factor = 1

        self.main_params_dtype = main_params_dtype
        self.main_grads_dtype = main_grads_dtype
        self.gradient_scaling_factor = gradient_scaling_factor
        self.allocator = allocator

        # Buffer references (initialized in _init_buffers)
        self.model_weight_buffer: Optional[DataParallelBuffer] = None
        self.transpose_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_grad_buffer: Optional[DataParallelBuffer] = None
        self.hsdp_wbuf: Optional[DataParallelBuffer] = None
        self.hsdp_gbuf: Optional[DataParallelBuffer] = None
        self.hsdp_comm_gbuf: Optional[DataParallelBuffer] = None

        # Initialize buffers and distributed parameters
        self._init_buffers()

    def _create_buffer(self, dtype: torch.dtype, is_distributed: bool) -> DataParallelBuffer:
        """Create a DataParallelBuffer with the given settings."""
        return DataParallelBuffer(
            params=self.params,
            param_idx=self.param_idx,
            dtype=dtype,
            device=self.device,
            dp_group=self.dp_group,
            allocator=self.allocator,
            is_distributed=is_distributed,
            param_group_id=self.param_group_id,
            gradient_scaling_factor=self.gradient_scaling_factor,
            chunk_size_factor=self.chunk_size_factor,
            sharding_strategy=self.sharding_strategy,
        )

    def _init_buffers(self) -> None:
        """
        Initialize all buffers based on sharding strategy.

        Buffer creation logic:
        - model_weight_buffer: always created unless "no_shard"
        - main_weight_buffer: created if main_params_dtype specified
        - main_grad_buffer: created if requires_grad
        """
        s = self.sharding_strategy
        shard_weights = s == "optim_grads_params"
        shard_main_weights = s != "no_shard"
        shard_grads = s in ("optim", "optim_grads", "optim_grads_params")

        # Create model weight buffer
        if s != "no_shard":
            wbuf = self._create_buffer(self.dtype, shard_weights)
            wbuf.init_data(torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device))
            for i, p in enumerate(self.params):
                wbuf.set_item(i, p.detach())
            self.model_weight_buffer = wbuf

        # Create main weight buffer for mixed precision
        if self.main_params_dtype is not None:
            mbuf = self._create_buffer(self.main_params_dtype, shard_main_weights)
            mbuf.init_data(torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device))
            for i, p in enumerate(self.params):
                mbuf.set_item(i, p.detach().to(self.main_params_dtype))
            self.main_weight_buffer = mbuf

        # Create gradient buffer
        if self.requires_grad:
            if self.main_grads_dtype is not None:
                gbuf_dtype = self.main_grads_dtype
            elif self.main_params_dtype is not None:
                gbuf_dtype = self.main_params_dtype
            else:
                gbuf_dtype = self.dtype
            gbuf = self._create_buffer(gbuf_dtype, shard_grads)
            gbuf.init_data(torch.zeros(gbuf.data_size, dtype=gbuf.dtype, device=self.device))
            self.main_grad_buffer = gbuf

        # Create distributed parameter views
        self._init_dist_params()

    def unshard(self, async_op: bool = False):
        """
        Unshard model weights by all-gathering from sharded buffer.

        After unshard, self.params.data points to full (unsharded) tensors.
        """
        _, work = self.model_weight_buffer.unshard(async_op=async_op)
        return work

    def reshard(self):
        """Reshard model weights by releasing unsharded buffer."""
        self.model_weight_buffer.reshard()

    def reduce_grad(self):
        """
        Reduce gradients across DP ranks.

        For distributed buffers: reduce-scatter the full gradient
        For non-distributed buffers: all-reduce in-place
        """
        self.main_grad_buffer.reduce_grad()

    def release_grad_buffer(self):
        """Release the main gradient buffer to free memory."""
        if self.main_grad_buffer is not None:
            # Drop weight.main_grad views that layers.py stores during gradient-accumulation-fusion
            # backward.  Those views keep _unsharded_buffer alive even after reshard() sets the
            # internal reference to None, causing the grad buffer to leak until the next backward.
            for param in self.params:
                if hasattr(param, 'main_grad'):
                    del param.main_grad
            self.main_grad_buffer.reshard()

    def _init_dist_params(self):
        """
        Initialize distributed parameter views (DTensors) into the buffers.

        Creates DTensor views of model weights and gradients based on sharding strategy:
        - "optim_grads_params": both weights and grads sharded
        - "optim_grads": only grads sharded
        - "optim": only weights sharded
        - "no_shard": replicated (no sharding)
        """
        self.dist_params = []
        self.dist_grads = []
        s = self.sharding_strategy

        # Determine placement based on sharding strategy
        is_param_shard = s in ("optim", "optim_grads", "optim_grads_params")
        placements = [Shard(dim=0)] if is_param_shard else [Replicate()]

        # Create parameter DTensor views
        for param in self.params:
            if self.main_weight_buffer is not None:
                mbuf = self.main_weight_buffer
                data = mbuf.get_item(self.param_idx[param], only_shard=is_param_shard)
            elif self.model_weight_buffer is not None:
                wbuf = self.model_weight_buffer
                data = wbuf.get_item(self.param_idx[param], only_shard=is_param_shard)
            else:
                data = param.data.detach()

            dist_param = torch.nn.Parameter(
                make_uneven_dtensor(data, param.shape, self.mesh, placements)
            )
            # Mark as FSDP parameter for special handling
            setattr(param, "__fsdp_param__", True)
            setattr(dist_param, "__fsdp_param__", True)
            self.dist_params.append(dist_param)

        # Create gradient DTensor views
        is_grad_shard = is_param_shard
        for p in self.params:
            if p.requires_grad:
                gbuf = self.main_grad_buffer
                grad_data = gbuf.get_item(self.param_idx[p], only_shard=is_grad_shard)
                self.dist_grads.append(
                    make_uneven_dtensor(grad_data, p.shape, self.mesh, placements)
                )
            else:
                self.dist_grads.append(None)
