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
from typing import List, Optional

import torch

from .param_runtime_order import OrderedParamGenerator


class MemStats(object):
    def __init__(self) -> None:
        """
        Store the non model data statistics used for Hybrid Optimizer.
        """
        # (preop_step, List[param])
        self._step_param_dict = dict()
        # (param, List[preop_step])
        self._param_step_dict = dict()
        # (preop_step, non_model_data) non model data used during preop_step ~ (preop_step+1)
        self._step_nmd_dict = dict()
        self._param_runtime_order = OrderedParamGenerator()

        self._preop_step = 0

        self._prev_overall_cuda = -1
        self._max_overall_cuda = 0
        self._prev_md_cuda = -1

        # old version
        self._model_data_cuda_list = []
        self._model_data_cpu_list = []

        self._overall_cuda_list = []
        self._overall_cpu_list = []

        self._non_model_data_cuda_list = []
        self._non_model_data_cpu_list = []

    def calc_max_cuda_non_model_data(self):
        if self._prev_overall_cuda != -1 and self._prev_md_cuda != -1:
            max_cuda_non_model_data = self._prev_overall_cuda - self._prev_md_cuda
            # compatibility of the old version.
            self._non_model_data_cuda_list.append(max_cuda_non_model_data)

    def record_max_cuda_model_data(self, val):
        self._prev_md_cuda = val

    def record_max_cuda_overall_data(self, val):
        self._prev_overall_cuda = val
        self._max_overall_cuda = max(self._max_overall_cuda, val)

    @property
    def max_overall_cuda(self):
        return self._max_overall_cuda

    def non_model_data_list(self, device_type: str) -> List[int]:
        if device_type == "cuda":
            return self._non_model_data_cuda_list
        elif device_type == "cpu":
            return self._non_model_data_cpu_list
        else:
            raise TypeError

    def max_non_model_data(self, device_type: str) -> float:
        if device_type == "cuda":
            return max(self._non_model_data_cuda_list)
        elif device_type == "cpu":
            return max(self._non_model_data_cpu_list)
        else:
            raise TypeError

    def clear(self):
        self._model_data_cuda_list = []
        self._overall_cuda_list = []

        self._model_data_cpu_list = []
        self._overall_cpu_list = []

        self._non_model_data_cpu_list = []
        self._non_model_data_cuda_list = []

        self._param_runtime_order.clear()
        self._step_param_dict.clear()
        self._param_step_dict.clear()
        self._step_nmd_dict.clear()
        self._preop_step = 0

        self._prev_overall_cuda = -1
        self._prev_md_cuda = -1
