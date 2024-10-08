# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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
from collections import deque, defaultdict
from typing import Deque, Dict, Iterable, Optional, Set, Tuple, List
import numpy as np
import torch

import torch.distributed

from .chunk import Chunk, ChunkFullError, _to_device_obj


__all__ = ['ChunkManager']

MiB = 1024 * 1024

class ChunkManager:
    """
    A manager class to manipulate the tensors in chunks.

    Args:
        init_device (torch.device): optional, the device on which the chunk is initialized. The default is None.
    """

    def __init__(
        self,
        chunk_size: int,
        init_device: Optional[torch.device] = None,
        max_prefetch: int = 0,
        pin_memory: bool = True,
        is_fp32_grad: bool = True
    ) -> None:
        
        self.__alloc_cuda_grad = not is_fp32_grad
        self.chunk_size = chunk_size
        # NOTE: only used for chunk init, NOT REPRESENT TENSOR DEVICE!
        self.device = init_device or _to_device_obj(torch.cuda.current_device())
        self.chunk_groups: Dict[str, Deque[Chunk]] = dict()
        
        self.param_to_grad_chunk_map: Dict[Chunk, Chunk] = dict()

        # tensor to its corresponding chunk
        self.tensor_chunk_map: Dict[torch.Tensor, Chunk] = dict()

        # NOTE: (param chunk -> (param tensor -> param grad))
        self.tensor_grad_map: Dict[Chunk, Dict[torch.Tensor, torch.Tensor]] = defaultdict(dict)

        # for optimizer state
        self.paired_chunk_map: Dict[Chunk, List[Chunk]] = defaultdict(list)
        
        self.pin_memory = pin_memory

        # accessed_chunks are those chunk currently on cuda
        self.accessed_chunks: Set[Chunk] = set()
        self.total_mem: Dict[str, int] = {"cpu": 0, "cuda": 0}

    def __find_best_chunk_and_append(
            self, 
            tensor: torch.Tensor,
            group_name: str,
            pin_memory: bool = None
        ):
            # to make sure all tensor in the group have the same dtype
            group_name = group_name + '_' + str(tensor.dtype)
            chunk_group = self.__get_chunk_group(group_name)
            chunk_size = self.chunk_size
            # NOTE: Next-Fit
            try:
                # append the tensor to the last chunk
                chunk_group[-1].append_tensor(tensor)
            except (IndexError, ChunkFullError):
                if pin_memory is None:
                    pin_memory = self.pin_memory
                # the except statement will be triggered when there is no chunk or
                # the last chunk in the chunk group is full
                # this will create a new chunk and allocate this chunk to its corresponding process
                if chunk_group:
                    # the chunk group is not empty
                    # close the last chunk
                    self.__close_one_chunk(chunk_group[-1])

                if tensor.numel() > chunk_size:
                    chunk_size = tensor.numel()

                chunk = Chunk(
                    chunk_size=chunk_size,
                    init_device=self.device,
                    dtype=tensor.dtype,
                    pin_memory=pin_memory,
                )

                chunk_group.append(chunk)
                chunk.append_tensor(tensor)
                self.__add_memory_usage(chunk.memory_usage)

            self.tensor_chunk_map[tensor] = chunk_group[-1]
            return tensor

    def register_tensor(
        self,
        tensor: torch.Tensor,
        group_type: str,
        pin_memory: bool = True,
    ) -> torch.Tensor:
        """
        Register a tensor to the chunk manager.
        Then, the tensor should be accessed by `get_chunks`.

        Args:
            tensor: the tensor appended to the chunk
            group_type: the annotation of the group, e.g., data type. The data type in the
                group should be same.
            pin_memory: whether the chunk is pinned in the cpu memory
        """
        assert tensor not in self.tensor_chunk_map
        assert isinstance(tensor, torch.Tensor), "Please feed Tensor to this ChunkManager"

        rank = get_rank()
        group_name = "{}_{}".format(group_type, rank)

        self.__find_best_chunk_and_append(
            tensor,
            group_name,
            pin_memory
        )

    def close_all_groups(self):
        """Close all the chunks of all groups."""
        for group_name in self.chunk_groups:
            self.__close_one_chunk(self.chunk_groups[group_name][-1])

    def move_chunk(self, chunk: Chunk, device: torch.device, async_move=False) -> None:
        """Move the shard of the chunk to the target device."""

        device = _to_device_obj(device)
        # NOTE: if a param chunk is moved to cuda, the cpu grad chunk may be dropped
        self.__sub_memory_usage(chunk.memory_usage)
        chunk.reset_device(device, async_move=async_move)
        self.__add_memory_usage(chunk.memory_usage)

        # NOTE: drop or create grad chunk
        # __alloc_cuda_grad:
        # False, cpu -> cuda [maybe release cpu chunk], cuda -> cpu [maybe create chunk]
        # True, cpu -> cuda [move chunk], cuda -> cpu [move chunk]

        if self.__alloc_cuda_grad:
            grad_chunk = self.param_to_grad_chunk_map[chunk]
            self.__sub_memory_usage(grad_chunk.memory_usage)
            self.param_to_grad_chunk_map[grad_chunk].reset_device(device, async_move=async_move)
            self.__add_memory_usage(grad_chunk.memory_usage)
        elif device.type == 'cuda':
            if chunk in self.param_to_grad_chunk_map:                
                grad_chunk = self.param_to_grad_chunk_map.pop(chunk)
                assert grad_chunk.device_type == 'cpu'
                self.tensor_grad_map.pop(chunk)
                self.__sub_memory_usage(grad_chunk.memory_usage)
            # otherwise try to move a cuda grad chunk to cuda, do nothing
        else:
            if chunk not in self.param_to_grad_chunk_map:
                self.create_grad(chunk)
            # otherwise try to move a cpu grad chunk to cpu, do nothing

        # NOTE: move optimizer state chunks
        for paired_chunk in self.paired_chunk_map[chunk]:
            self.__sub_memory_usage(paired_chunk.memory_usage)
            paired_chunk.reset_device(device, async_move=async_move)
            self.__add_memory_usage(paired_chunk.memory_usage)

        device = _to_device_obj(device)
        if device.type == 'cuda':
            self.accessed_chunks.add(chunk)
        else:
            self.accessed_chunks.remove(chunk)

    def create_grads(self):
        for group_name, chunk_group in self.chunk_groups.items():
            if 'param' not in group_name:
                continue
            for chunk in chunk_group:
                if chunk.device_type == 'cpu' or self.__alloc_cuda_grad:
                    self.create_grad(chunk)
                                
    def create_grad(self, param_chunk: Chunk):
        grad_chunk = param_chunk.clone()
        self.__add_memory_usage(grad_chunk.memory_usage)
        self.param_to_grad_chunk_map[param_chunk] = grad_chunk
        
        for param, grad in zip(param_chunk.get_tensors(), grad_chunk.get_tensors()):
            self.tensor_grad_map[param_chunk][param] = grad

    def attach_grad(self):
        for param_grad_map in self.tensor_grad_map.values():
            for param, grad in param_grad_map.items():
                assert param.shape == grad.shape
                param.grad = grad

    def __repr__(self) -> str:
        msg = [
            "Chunk Manager Information:\n",
            "Total memory: " + ", ".join([f"{k}={v}B" for k, v in self.total_mem.items()]) + "\n",
        ]
        for group_name, group in self.chunk_groups.items():
            msg.append(f"Group {group_name}:\n")
            for i, chunk in enumerate(group):
                msg.append(f"[{i}] {chunk}\n")
        return "".join(msg)

    def __get_chunk_group(self, group_name: str) -> Deque[Chunk]:
        """Register a chunk group."""
        if group_name not in self.chunk_groups:
            self.chunk_groups[group_name] = deque()
        return self.chunk_groups[group_name]

    def __close_one_chunk(self, chunk: Chunk):
        self.__sub_memory_usage(chunk.memory_usage)
        chunk.close_chunk()
        self.__add_memory_usage(chunk.memory_usage)

    def __sub_memory_usage(self, usage: Dict[str, int]):
        for k, v in usage.items():
            self.total_mem[k] -= v

    def __add_memory_usage(self, usage: Dict[str, int]):
        for k, v in usage.items():
            self.total_mem[k] += v

    @staticmethod
    def find_best_chunk_size(
        tensor_numels: np.ndarray,
        search_range_m: float,
        search_interval: int = 32 * MiB,  # hidden size is the best value for the interval
        min_chunk_size_m: float = 32,
        filter_exlarge_params: bool = True,
    ) -> int:
        """
            Brute Search to get a best chunk size.
            search_range_m (float): searching range divided by 2^20.
            search_interval (int): searching interval. (not divided!)
            min_chunk_size_m (float, optional): the minimum size of a distributed chunk, divided by 2^20..
            filter_exlarge_params (bool, optional): filter extreme large parameters. Defaults to True.
        """

        if tensor_numels.size == 0:
            return 32 * 1024 ** 2 # default value, should not be used
        std = np.std(tensor_numels)
        mean = np.mean(tensor_numels)
        upper_limit = round(mean + 3 * std)


        if filter_exlarge_params:
            tensor_numels = tensor_numels[tensor_numels < upper_limit]

        start_numel = round(min_chunk_size_m * MiB) 
        end_numel = round((min_chunk_size_m + search_range_m) * MiB) + 2

        wasted_numel = 0

        results = []
        for chunk_size in range(start_numel, end_numel, search_interval):
            chunks = []
            for n_param in tensor_numels:
                chunks, waste_diff = _next_fit_simulation(
                    chunks,
                    chunk_size,
                    n_param
                )
                wasted_numel += waste_diff 
            results.append((chunk_size, wasted_numel))
        # NOTE: Automatically select the chunk size with minimal waste.
        # But also print the detailed waste of Top-10 chunk_size in the rank 0
        results = sorted(results, key=lambda x: (x[1], -x[0]))[:10]

        if get_rank() == 0:
            print("Chunk Size Eval Results (chunk_size, waste):", results)
        
        return results[0][0]

    def alloc_paired_tensors(
        self,
        dtype: torch.dtype,
    ) -> Dict[torch.Tensor, torch.Tensor]:
        """
            Alloc paired tensors for each parameter in self.chunk_groups (with same device)

        """
        tensor_map = {}
        # NOTE: freeze chunk group for iteration
        for _, chunk_group in list(self.chunk_groups.items()):
            for chunk in chunk_group:
                paired = chunk.clone(dtype)
                self.__add_memory_usage(paired.memory_usage)
                for tensor, p in zip(chunk.get_tensors(), paired.get_tensors()):
                    tensor_map[tensor] = p
                self.paired_chunk_map[chunk].append(paired)

        return tensor_map

    def get_offload_ratio(self):
        n_param_cpu = [0, 0]
        n_param_total = [0, 0]
        for chunk_group in self.chunk_groups.values():
            for chunk in chunk_group:
                n_param_total[0] += chunk.chunk_mem
                n_param_total[1] += chunk.get_valid_length() * chunk.dtype.itemsize
                if chunk.device_type == 'cpu':
                    n_param_cpu[0] += chunk.chunk_mem
                    n_param_cpu[1] += chunk.get_valid_length() * chunk.dtype.itemsize
                
        return {
            'Chunk Offload Ratio': n_param_cpu[0] / (n_param_total[0] + 1e-6) * 100,
            'Tensor Offload Ratio': n_param_cpu[1] / (n_param_total[1] + 1e-6) * 100,
        }

    def calc_size_in_device(self, chunk: Chunk, device_type: str):
        """
        Given a chunk, get the total memory size it required in the target device
        """
        total = 0
        numel = chunk.chunk_size
        total += numel * chunk.dtype.itemsize
        if self.__alloc_cuda_grad and device_type =='cuda':
            total += numel * 4
        
        for pair_chunk in self.paired_chunk_map[chunk]:
            total += pair_chunk.chunk_mem

        return total

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def _next_fit_simulation(
    chunks_list: List[int],
    chunk_size: int,
    n_param: int
) -> Tuple[List[int], int]:
    if len(chunks_list) == 0:
        chunks_list.append(n_param)
        waste_diff = max(0, chunk_size - n_param) # a.k.a space
        return chunks_list, waste_diff
    
    cur_size = chunks_list[-1]
    valid_space = max(0, chunk_size - cur_size)
    if valid_space < n_param:
        chunks_list.append(n_param)
        waste_diff = max(0, valid_space) + max(0, chunk_size - n_param) # a.k.a space
        return chunks_list, waste_diff
    else:
        chunks_list[-1] += n_param
        return chunks_list, -n_param


