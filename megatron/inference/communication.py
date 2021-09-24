# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Communications utilities."""


import torch



def broadcast_tensor(size, dtype, tensor=None, rank=0):
    """ Given size and type of a tensor on all ranks and the tensor value
        only on a specific rank, broadcast from that rank to all other ranks.
    """

    if torch.distributed.get_rank() == rank:
        assert tensor is not None
        assert tensor.is_cuda
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())

    torch.distributed.broadcast(tensor, rank)

    return tensor


def broadcast_list(size, dtype, list_values=None, rank=0):
    """Broadcast a list of values with a given type."""

    tensor = None
    if torch.distributed.get_rank() == rank:
        tensor = torch.tensor(list_values, dtype=dtype,
                              device=torch.cuda.current_device())

    return broadcast_tensor(size, dtype, tensor=tensor, rank=rank)


def broadcast_int_list(size, int_list=None, rank=0):
    """Broadcast a list of interger values."""

    return broadcast_list(size, torch.int64, list_values=int_list, rank=rank)


def broadcast_float_list(size, float_list=None, rank=0):
    """Broadcast a list of float values."""

    return broadcast_list(size, torch.float32, list_values=float_list,
                          rank=rank)
