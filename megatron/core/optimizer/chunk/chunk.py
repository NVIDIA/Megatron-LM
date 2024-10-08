# Copyright (c) 2024 Alibaba PAI, ColossalAI and Nvidia Megatron-LM Team.
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

@dataclass
class TensorInfo:
    offset: int
    end: int


class ChunkFullError(Exception):
    pass

def _to_device_obj(maybe_device: Union[int, str, torch.device]):
    if isinstance(maybe_device, int):
        return torch.device(f'cuda:{maybe_device}')
    elif isinstance(maybe_device, str):
        return torch.device(maybe_device)
    elif isinstance(maybe_device, torch.device):
        return maybe_device
    raise ValueError()

class Chunk:

    def __init__(
        self,
        chunk_size: int,
        dtype: torch.dtype,
        init_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        release_cpu_mem_in_cuda_chunk: bool = False
    ) -> None:
        """
        Chunk: A container owning a piece of contiguous memory space for tensors
        Here we use all-gather operation to gather the whole chunk.
        It is designed to make the full use of communication and PCIE bandwidth.

        Args:
            chunk_size (int): the number of elements in the chunk
            dtype (torch.dtype): the data type of the chunk
            init_device (torch.device): optional, During the chunk construction process, where the tensor is stored.
                The default value is None, which is the current GPU
            pin_memory (bool): optional, if True, this chunk always has a shard copied in pinned CPU memory
        """
        self.chunk_size = chunk_size
        self.utilized_size = 0

        self.dtype = dtype
        self._target_device = _to_device_obj(init_device or torch.cuda.current_device())

        # NOTE: always initialize on CPU but move to the target device if needed
        self._cpu_data = torch.zeros(chunk_size, dtype=self.dtype)
        self._cuda_data = None  # keep all zero
        self._closed = False
        self.chunk_mem = self.chunk_size * self._cpu_data.element_size()

        # each tensor is associated with a TensorInfo to track its meta info
        # (offset, end)
        self.tensors_info: Dict[torch.Tensor, TensorInfo] = {}
        # the total number of tensors in the chunk
        self.num_tensors = 0

        # if pin_memory is True, we allocate a piece of CPU pin-memory
        # for it all the time, only be used when device is CPU
        self.pin_memory = pin_memory
        self.release_cpu_mem_in_cuda_chunk = release_cpu_mem_in_cuda_chunk

    @property
    def data(self) -> torch.Tensor:
        assert self._closed, "Chunk.data must be used after chunk-closing!"
        if self.device_type == 'cuda':
            return self._cuda_data
        return self._cpu_data
    
    @data.setter
    def set_data(self, value):
        assert self._closed, "Chunk.data must be used after chunk-closing!"
        if self.device_type == 'cuda':
            self._cuda_data = value
        else:
            self._cpu_data = value

    @property
    def memory_usage(self) -> Dict[str, int]:
        cuda_memory = 0
        cpu_memory = 0

        # Only CUDA Chunk consumes GPU 
        if self.device_type == 'cuda':
            cuda_memory += self.chunk_mem
        
        if self._cpu_data is not None:
            cpu_memory += self.chunk_mem

        return dict(cuda=cuda_memory, cpu=cpu_memory)

    @property
    def device_type(self) -> str:
        """
            Whether closed or not, returns the current device of chunk.
        """
        if self._closed:
            return self._target_device.type
        return "cpu"

    def reset_device(self, device: torch.DeviceObjType, async_move: bool=True):
        self._target_device = _to_device_obj(device)
        if self._closed:
            self.__move_to_target_device(async_move=async_move)

    def append_tensor(self, tensor: torch.Tensor, is_async: bool=False):
        """Add a tensor to the chunk.

        Args:
            tensor (torch.Tensor): a tensor to be added to the chunk
        """
        # sanity check
        # assert tensor.device.type == 'cuda', "Only support append cuda tensor!"
        assert tensor.dtype == self._cpu_data.dtype
        assert not self._closed
        new_utilized_size = self.utilized_size + tensor.numel()
        # raise exception when the chunk size is exceeded
        if new_utilized_size > self.chunk_size:
            raise ChunkFullError

        # Here is possibly a D2D, D2H, H2D copy
        self._cpu_data[self.utilized_size : new_utilized_size].copy_(
            tensor.data.flatten(), non_blocking=is_async
        )

        tensor.data = self._cpu_data[self.utilized_size : new_utilized_size].view(tensor.shape)

        # record all the information about the tensor
        self.num_tensors += 1
        self.tensors_info[tensor] = TensorInfo(self.utilized_size, new_utilized_size)
        self.utilized_size = new_utilized_size

    def alloc_tensor(self, tensor_size: torch.Size) -> torch.Tensor:
        assert not self._closed
        new_utilized_size = self.utilized_size + tensor_size.numel()
        # raise exception when the chunk size is exceeded
        if new_utilized_size > self.chunk_size:
            raise ChunkFullError

        new_tensor = self._cpu_data[self.utilized_size : new_utilized_size].view(tensor_size)
        self.num_tensors += 1
        self.tensors_info[new_tensor] = TensorInfo(self.utilized_size, new_utilized_size)
        self.utilized_size = new_utilized_size
        return new_tensor

    def close_chunk(self):
        """Close the chunk. Any tensor can't be appended to a closed chunk later."""
        assert self._closed is False, "The chunk is already closed!"
        self.__move_to_target_device()
        self._closed = True

    def __move_to_target_device(self, async_move: bool=True):
        if self._target_device.type == "cpu":
            if self._cuda_data is not None:
                if self._cpu_data is None:
                    self._cpu_data = torch.empty(self._cuda_data.shape, dtype=self.dtype, pin_memory=self.pin_memory)
                self._cpu_data.data.copy_(self._cuda_data.data, non_blocking=async_move)
                self.__update_tensors_ptr(self._cpu_data)
                self._cuda_data = None
            # already on cpu, do nothing
            return
        # Target is CUDA
        if self._cpu_data is not None:
            self._cuda_data = torch.empty(self._cpu_data.shape, dtype=self.dtype, device=self._target_device)
            self._cuda_data.data.copy_(self._cpu_data.data, non_blocking=async_move)
            self.__update_tensors_ptr(self._cuda_data)
            if self.release_cpu_mem_in_cuda_chunk or not self.pin_memory:
                self._cpu_data = None

    def get_valid_length(self) -> int:
        """Get the valid length of the chunk's payload."""
        return self.utilized_size

    def get_tensors(self) -> List[torch.Tensor]:
        return list(self.tensors_info.keys())

    def __update_tensors_ptr(self, src_chunk: torch.Tensor) -> None:
        """
            Update tensors ptr to the location of source chunk
        """
        for tensor, tensor_info in self.tensors_info.items():
            tensor.data = src_chunk[tensor_info.offset : tensor_info.end].view(tensor.shape)

    def __hash__(self) -> int:
        return hash(id(self))

    def __repr__(self):
        output = [
            "Chunk Information:\n",
            "\tchunk size: {}, chunk dtype: {}\n".format(
                self.chunk_size, self.dtype
            ),
            "\t# of tensors: {}, utilized size: {}, utilized percentage: {:.2f}\n".format(
                self.num_tensors, self.utilized_size, self.utilized_size / self.chunk_size
            ),
        ]
     
        memory_info = self.memory_usage
        output.append("\tmemory usage: cuda {}, cpu {}\n".format(memory_info["cuda"], memory_info["cpu"]))
        return "".join(output)

    def clone(self, dtype=None) -> 'Chunk':
        """
            Return a clone of self, each tensor in the clone is also a clone of raw tensor
        """
        if dtype is None:
            dtype = self.dtype

        cloned = Chunk(
            self.chunk_size,
            dtype,
            self._target_device,
            self.pin_memory,
            self.release_cpu_mem_in_cuda_chunk
        )

        for tensor in self.get_tensors():
            cloned.alloc_tensor(tensor.size())
        
        if self._closed:
            cloned.close_chunk()
        return cloned
