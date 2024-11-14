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
from abc import ABC

import torch


class ParamGenerator(ABC):
    def append(self, param: torch.nn.Parameter):
        pass

    def generate(self):
        pass

    def clear(self):
        pass


class OrderedParamGenerator(ParamGenerator):
    """OrderedParamGenerator

    Contain the order of parameters visited during runtime.
    """

    def __init__(self) -> None:
        self.param_visited_order = []

    def append(self, param: torch.nn.Parameter):
        self.param_visited_order.append(param)

    def generate(self):
        visited_set = set()
        for p in self.param_visited_order:
            if p not in visited_set:
                yield p
            visited_set.add(p)
        del visited_set

    def is_empty(self):
        return len(self.param_visited_order) == 0

    def clear(self):
        self.param_visited_order = []
