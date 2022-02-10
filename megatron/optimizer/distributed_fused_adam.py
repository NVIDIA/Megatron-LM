# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

? ? ?

import math
import torch

from megatron import mpu

# >>>
from lutil import pax, tp
# <<<

class DistributedFusedAdam(torch.optim.Optimizer):

    def __init__(self, params):

        super().__init__(params, defaults = {})

        self.initialized = False
        # self.params_32 = None
        # self.grads_32 = None
        # self.opt_m = None
        # self.opt_v = None

        # pax(0, {
        #     "param_groups" : self.param_groups,
        #     "param_groups / 0" : self.param_groups[0],
        #     "param_groups / 1" : self.param_groups[1],
        #     "param_groups / 0 / params" : self.param_groups[0]["params"],
        #     # "param_groups / params" : [ g["params"] for g in self.param_groups ],
        # })

    def initialize(self):

        if self.initialized:
            raise Exception("initialization worked.")
            return
        self.initialized = True

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        total_param_size = sum(
            p.numel()
            for g in self.param_groups
            for p in g["params"]
        )
        shard_size = int(math.ceil(total_param_size / data_parallel_world_size))
        shard_start_index = data_parallel_rank * shard_size
        shard_end_index = min(total_param_size, shard_start_index + shard_size)
        shard_size = shard_end_index - shard_start_index

        allocate_shard = lambda dtype : torch.empty(
            [shard_size],
            dtype = dtype,
            device = torch.cuda.current_device())

        self.main_param_shard = allocate_shard(torch.float)
        self.main_grad_shard = allocate_shard(torch.float)
        self.adam_m_shard = allocate_shard(torch.float)
        self.adam_v_shard = allocate_shard(torch.float)

        # pax(2, {
        #     "data_parallel_rank" : data_parallel_rank,
        #     "data_parallel_world_size" : data_parallel_world_size,
        #     "total_param_size" : total_param_size,
        #     "shard_size" : shard_size,
        #     "shard" : "%d [ %d, %d ]" % (
        #         shard_size,
        #         shard_start_index,
        #         shard_end_index,
        #     ),
        # })

    def step(self):

        self.initialize()

        raise Exception("what's next?")

# >>>
# eof
# <<<
