# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .dp_buffer import DataParallelBuffer
from .mixed_precision import FullyShardMixedPrecisionPolicy
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
        mp_policy: FullyShardMixedPrecisionPolicy,
        mesh: Optional[DeviceMesh] = None,
        sharding_strategy: str = "optim_grads_params",
        gradient_scaling_factor: Optional[float] = None,
        allocator: Optional[BucketAllocator] = None,
    ):
        self.params = params
        self.param_idx: Dict[torch.nn.Parameter, int] = {p: i for i, p in enumerate(params)}

        # Assume all params have same device/dtype/require_grad
        # TODO: validate all params have same properties
        self.device = params[0].device
        self.dtype = params[0].dtype
        self.requires_grad = params[0].requires_grad
        self.mp_policy = mp_policy
        self.is_fp8_group = self.mp_policy.is_fp8_param(params[0])
        self.needs_transpose_weight_buffer = self.mp_policy.needs_transpose_weight_buffer(params[0])
        assert all(
            self.mp_policy.is_fp8_param(p) == self.is_fp8_group for p in params
        ), "FP8 and non-FP8 parameters must not share a ParameterGroup"

        # Setup device mesh and derived process group
        if mesh is None:
            mesh = DeviceMesh(
                self.device.type,
                list(range(torch.distributed.get_world_size(torch.distributed.group.WORLD))),
            )
        assert mesh.ndim == 1, "Only 1D mesh is supported"
        self.mesh = mesh
        self.dp_group = mesh.get_group()

        self.sharding_strategy = sharding_strategy
        self.param_group_id = param_group_id

        # Compute chunk size factor for alignment
        # LCM ensures params align to common boundary for efficient sharding
        if len(params) > 0 and any(p.shape[1:].numel() > 0 for p in params):
            self.chunk_size_factor = max(1, math.lcm(*[p.shape[1:].numel() for p in params]))
        else:
            self.chunk_size_factor = 1

        self.gradient_scaling_factor = gradient_scaling_factor
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()

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

    def set_allocator(self, allocator: BucketAllocator) -> None:
        self.allocator = allocator
        for buffer in (
            self.model_weight_buffer,
            self.transpose_weight_buffer,
            self.main_weight_buffer,
            self.main_grad_buffer,
        ):
            if buffer is not None:
                buffer.allocator = allocator

    def _create_buffer(
        self, dtype: torch.dtype, is_distributed: bool, role: str
    ) -> DataParallelBuffer:
        """Create a buffer and namespace its temporary bucket by role."""
        return DataParallelBuffer(
            params=self.params,
            param_idx=self.param_idx,
            dtype=dtype,
            device=self.device,
            dp_group=self.dp_group,
            allocator=self.allocator,
            buffer_role=role,
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
        - main_weight_buffer: created if mp_policy.main_params_dtype is specified
        - main_grad_buffer: created if requires_grad
        """
        s = self.sharding_strategy
        shard_weights = s == "optim_grads_params"
        shard_main_weights = s != "no_shard"
        shard_grads = s in ("optim_grads", "optim_grads_params")

        # Create model weight buffer. FP8/MXFP8 parameters store their TE raw
        # payload in the model buffer, so the buffer dtype is uint8 while the
        # logical compute dtype remains on the parameter object.
        if s != "no_shard":
            model_weight_dtype = self.mp_policy.model_weight_buffer_dtype(self.params[0])
            wbuf = self._create_buffer(model_weight_dtype, shard_weights, "model_weight")
            wbuf.init_data(torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device))
            for i, p in enumerate(self.params):
                wbuf.set_item(i, self.mp_policy.get_param_data(p))
            self.model_weight_buffer = wbuf

            if self.needs_transpose_weight_buffer:
                tbuf = self._create_buffer(torch.uint8, shard_weights, "transpose_weight")
                tbuf.init_data(torch.empty(tbuf.data_size, dtype=tbuf.dtype, device=self.device))
                for i, p in enumerate(self.params):
                    tbuf.set_item(i, self.mp_policy.get_param_data(p, transpose=True))
                self.transpose_weight_buffer = tbuf

        # Create main weight buffer for mixed precision
        main_params_dtype = self.mp_policy.main_params_dtype_for_param(self.params[0])
        if main_params_dtype is not None:
            mbuf = self._create_buffer(main_params_dtype, shard_main_weights, "main_weight")
            mbuf.init_data(torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device))
            for i, p in enumerate(self.params):
                item = self.mp_policy.initial_main_weight(p)
                mbuf.set_item(i, item.detach().to(main_params_dtype))
            self.main_weight_buffer = mbuf

        # Free the original full parameter tensors now that their data has been
        # copied into the weight buffers. The module holds DTensor shard views and
        # unshard() rebinds .data to the all-gathered buffer, so the original
        # storage is never accessed again.
        for p in self.params:
            if (
                self.is_fp8_group
                or self.model_weight_buffer is None
                and self.main_weight_buffer is None
            ):
                continue
            _free_storage(p.data)

        # Create gradient buffer
        if self.requires_grad:
            main_grads_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
            gbuf = self._create_buffer(main_grads_dtype, shard_grads, "main_grad")
            gbuf.init_data(torch.zeros(gbuf.data_size, dtype=gbuf.dtype, device=self.device))
            self.main_grad_buffer = gbuf

        # Create distributed parameter views
        self._init_dist_params()

    def unshard(self, async_op: bool = False, is_bwd: bool = False):
        """
        Unshard model weights by all-gathering from sharded buffer.

        After unshard, self.params.data points to full (unsharded) tensors.
        """
        if is_bwd and self.transpose_weight_buffer is not None:
            buffers = [(self.transpose_weight_buffer, True)]
        else:
            buffers = [(self.model_weight_buffer, False)]

        work = None
        for weight_buffer, use_transpose_buffer in buffers:
            full_weight_buffer, weight_work = weight_buffer.unshard(
                async_op=async_op, bind_params=not self.is_fp8_group
            )
            if work is None:
                work = weight_work

            if self.is_fp8_group and full_weight_buffer is not None:
                for p in self.params:
                    item_id = self.param_idx[p]
                    offset, size = weight_buffer.buffer_index._get_item_offset(item_id)
                    self.mp_policy.set_unsharded_weight(
                        p,
                        full_weight_buffer[offset : offset + size].view(p.shape),
                        transpose=use_transpose_buffer,
                    )

        if self.is_fp8_group:
            self.mp_policy.post_unshard(self.params, is_bwd=is_bwd)
        return work

    def reshard(self):
        """Reshard model weights by releasing unsharded buffer."""
        self.model_weight_buffer.reshard()
        if self.transpose_weight_buffer is not None:
            self.transpose_weight_buffer.reshard()
        self.mp_policy.post_reshard(self.params)

    @torch.no_grad()
    def copy_main_weights_to_model_weights(self):
        """Install optimized main weights into model compute weights."""
        if self.main_weight_buffer is None:
            return
        if not self.is_fp8_group:
            self.model_weight_buffer.data.copy_(self.main_weight_buffer.data)
            return

        assert self.model_weight_buffer is not None, "FP8 param gather requires a model buffer"
        fp8_params = []
        main_params = []
        start_offsets = []
        model_param_shards = []
        for p in self.params:
            item_id = self.param_idx[p]
            model_shard = self.model_weight_buffer.get_item(item_id, only_shard=True)
            if model_shard.numel() == 0:
                fp8_params.append(p)
                main_params.append(None)
                start_offsets.append(None)
                model_param_shards.append((None, None))
                continue

            transpose_shard = None
            if self.transpose_weight_buffer is not None:
                transpose_shard = self.transpose_weight_buffer.get_item(item_id, only_shard=True)
            main_weight = self.main_weight_buffer.get_item(item_id, only_shard=True)
            start_offset, _ = self.model_weight_buffer.buffer_index._get_item_slice_in_shard(
                item_id
            )
            fp8_params.append(p)
            main_params.append(main_weight)
            start_offsets.append(start_offset)
            model_param_shards.append((model_shard, transpose_shard))

        self.mp_policy.quantize_main_weights_to_model(
            fp8_params, main_params, start_offsets, self.dp_group, model_param_shards
        )

    def reduce_grad(self):
        """
        Reduce gradients across DP ranks.

        For distributed buffers: reduce-scatter the full gradient
        For non-distributed buffers: all-reduce in-place
        """
        self.main_grad_buffer.reduce_grad(grad_comm_dtype=self.mp_policy.grad_comm_dtype)

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
        - "optim_grads_params": weights and grads sharded, full ZeRO-3
        - "optim_grads": grads sharded, weights replicated (ZeRO-2)
        - "optim": optimizer state sharding only, weights and grads replicated (ZeRO-1)
        - "no_shard": replicated, no sharding (DDP-equivalent)
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
                make_uneven_dtensor(data, param.shape, self.mesh, placements),
                requires_grad=param.requires_grad,
            )
            # Mark as FSDP parameter for special handling
            setattr(param, "__fsdp_param__", True)
            setattr(dist_param, "__fsdp_param__", True)
            self.dist_params.append(dist_param)

        # Create gradient DTensor views. Some groups, e.g. uint8 FP8 model
        # payloads, do not require grads and therefore have no grad buffer.
        if self.main_grad_buffer is None:
            self.dist_grads = [None for _ in self.params]
            return

        is_grad_shard = is_param_shard
        for p in self.params:
            gbuf = self.main_grad_buffer
            grad_data = gbuf.get_item(self.param_idx[p], only_shard=is_grad_shard)
            # NOTE: Do not remove the grad_data.numel() > 0 check.
            # Empty local grad shards are semantically no-ops, but materializing
            # them as DTensor grads can pass zero-numel tensors into fused
            # multi-tensor optimizers such as TE FusedAdam. That can break
            # updates for neighboring non-empty shards.
            if p.requires_grad and grad_data.numel() > 0:
                self.dist_grads.append(
                    make_uneven_dtensor(grad_data, p.shape, self.mesh, placements)
                )
            else:
                self.dist_grads.append(None)
