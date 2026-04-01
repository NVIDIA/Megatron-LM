from typing import Dict, List, Optional

import torch

from megatron.core.distributed.fsdp_refactor.src.allocator import TemporaryBucketAllocator
from megatron.core.distributed.fsdp_refactor.src.dp_buffer import DataParallelBuffer


class ParameterGroup:

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        fsdp_unit_id: int,
        dp_group: torch.distributed.ProcessGroup,
        device: torch.device,
        sharding_strategy: str = "no_shard",
        param_group_id: int = 0,
        chunk_size_factor: int = 1,
        main_params_dtype: Optional[torch.dtype] = None,
        gradient_scaling_factor: Optional[float] = None,
        allocator: Optional[TemporaryBucketAllocator] = None,
    ):
        self.params = params
        self.param_idx: Dict[torch.nn.Parameter, int] = {
            p: i for i, p in enumerate(params)
        }
        self.dtype = params[0].dtype
        self.requires_grad = params[0].requires_grad
        self.fsdp_unit_id = fsdp_unit_id

        self.dp_group = dp_group
        self.device = device
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

    def init_buffers(self) -> None:
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
