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


def broadcast_int_list(size, int_list=None, rank=0):
    """Broadcast a list of interger values."""

    long_tensor = None
    if torch.distributed.get_rank() == rank:
        long_tensor = torch.tensor(int_list, dtype=torch.int64,
                                   device=torch.cuda.current_device())

    return broadcast_tensor(size, torch.int64, tensor=long_tensor, rank=rank)
