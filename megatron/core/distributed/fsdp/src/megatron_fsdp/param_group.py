from typing import Dict, List, Optional

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate, Shard

from .allocator import TemporaryBucketAllocator
from .dp_buffer import DataParallelBuffer
from .uneven_dtensor import make_uneven_dtensor


class ParameterGroup:

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        fsdp_unit_id: int,
        *,
        mesh: Optional[DeviceMesh] = None,
        sharding_strategy: str = "optim_grads_params",
        param_group_id: int = 0,
        chunk_size_factor: int = 1,
        main_params_dtype: Optional[torch.dtype] = None,
        gradient_scaling_factor: Optional[float] = None,
        allocator: Optional[TemporaryBucketAllocator] = None,
    ):
        self.params = params
        self.param_idx: Dict[torch.nn.Parameter, int] = {p: i for i, p in enumerate(params)}
        # TODO: validate that all params have the same device/dtype/require_grad
        self.device = params[0].device
        self.dtype = params[0].dtype
        self.requires_grad = params[0].requires_grad
        self.fsdp_unit_id = fsdp_unit_id

        self.mesh = mesh
        if mesh is not None:
            assert mesh.ndim == 1, "Only 1D mesh is supported for now"
            self.dp_group = mesh.get_group()
        else:
            self.dp_group = torch.distributed.group.WORLD

        self.sharding_strategy = sharding_strategy
        self.param_group_id = param_group_id
        self.chunk_size_factor = chunk_size_factor
        self.main_params_dtype = main_params_dtype
        self.gradient_scaling_factor = gradient_scaling_factor
        self.allocator = allocator

        self.model_weight_buffer: Optional[DataParallelBuffer] = None
        self.transpose_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_weight_buffer: Optional[DataParallelBuffer] = None
        self.main_grad_buffer: Optional[DataParallelBuffer] = None
        self.hsdp_wbuf: Optional[DataParallelBuffer] = None
        self.hsdp_gbuf: Optional[DataParallelBuffer] = None
        self.hsdp_comm_gbuf: Optional[DataParallelBuffer] = None

        self._init_buffers()

    def _create_buffer(self, dtype: torch.dtype, is_distributed: bool) -> DataParallelBuffer:
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
        s = self.sharding_strategy
        shard_weights = s == "optim_grads_params"
        shard_main_weights = s != "no_shard"
        shard_grads = s in ("optim_grads", "optim_grads_params")

        if s != "no_shard":
            wbuf = self._create_buffer(self.dtype, shard_weights)
            wbuf.init_data(torch.empty(wbuf.data_size, dtype=wbuf.dtype, device=self.device))
            for i, p in enumerate(self.params):
                wbuf.set_item(i, p.detach())
            self.model_weight_buffer = wbuf

        if self.main_params_dtype is not None:
            mbuf = self._create_buffer(self.main_params_dtype, shard_main_weights)
            mbuf.init_data(torch.empty(mbuf.data_size, dtype=mbuf.dtype, device=self.device))
            for i, p in enumerate(self.params):
                mbuf.set_item(i, p.detach().to(self.main_params_dtype))
            self.main_weight_buffer = mbuf

        if self.requires_grad:
            gbuf = self._create_buffer(self.dtype, shard_grads)
            gbuf.init_data(torch.zeros(gbuf.data_size, dtype=gbuf.dtype, device=self.device))
            self.main_grad_buffer = gbuf

        self._init_dist_params()

    def unshard(self):
        self.model_weight_buffer.unshard()

    def reshard(self):
        self.model_weight_buffer.reshard()

    def reduce_grad(self):
        self.main_grad_buffer.reduce_grad()

    def _init_dist_params(self):
        self.dist_params = []
        s = self.sharding_strategy
        is_param_shard = s == "optim_grads_params"
        if is_param_shard:
            placements = [Shard(dim=0)]
        else:
            placements = [Replicate()]
        for p in self.params:
            if s != "no_shard":
                wbuf = self.model_weight_buffer
                data = wbuf.get_item(self.param_idx[p], only_shard=is_param_shard)
            else:
                data = p.detach()

            self.dist_params.append(
                torch.nn.Parameter(make_uneven_dtensor(data, p.shape, self.mesh, placements))
            )
