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
import time
from typing import Optional, List
import torch

class MemStats(object):
    def __init__(self) -> None:
        """
        Store the non model data statistics used for Hybrid Optimizer.
        """
        self._model_data_cuda_list = []
        self._non_model_data_cuda_list = []

        self._prev_overall_cuda = -1
        self._max_overall_cuda = 0
        self._prev_md_cuda = -1

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
        return self._non_model_data_cuda_list

    def max_non_model_data(self, device_type: str) -> float:
        return max(self._non_model_data_cuda_list)

    def clear(self):
        self._model_data_cuda_list = []
        self._non_model_data_cuda_list = []

        self._prev_overall_cuda = -1
        self._max_overall_cuda = 0
        self._prev_md_cuda = -1


class SyncCudaMemoryMonitor:
    """
    A synchronized cuda memory monitor.
    It only record the maximum allocated cuda memory from start point to finish point.
    """

    def __init__(self, power: int = 10):
        self.time_stamps = []
        self.mem_stats = []

        self.keep_measuring = False

    def __len__(self):
        return len(self.mem_stats)
    
    def cur_memory(self):
        return torch.cuda.max_memory_allocated()

    def start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def finish(self) -> int:
        """
        return max gpu memory used since latest `start()`.

        Returns:
            int: max GPU memory
        """
        torch.cuda.synchronize()
        self.time_stamps.append(time())
        max_usage = torch.cuda.max_memory_allocated()
        self.mem_stats.append(max_usage)
        return max_usage

    def clear(self):
        self.mem_stats.clear()
        self.time_stamps.clear()

class MemStatsCollector:
    """
    A Memory statistic collector.
    It works in two phases.
    Phase 1. Collection Phase: collect memory usage statistics of CPU and GPU.
    The first iteration of DNN training.
    Phase 2. Runtime Phase: use the read-only collected stats
    The rest iterations of DNN training.

    It has a Sampling counter which is reset after DNN training iteration.
    """

    def __init__(self) -> None:
        self._mem_monitor = SyncCudaMemoryMonitor()
        self._sampling_time = []

        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0
        self.use_outside_memstats = False
        self._memstats = MemStats()
        self._warmup = True
        self.warmup_memstats = MemStats()
        self.warmup_memstats._non_model_data_cuda_list.append(0)

    @property
    def sampling_time(self):
        return [t - self._sampling_time[0] for t in self._sampling_time]

    def start_collection(self):
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        self.sample_overall_data()
        self._step_total = len(self._memstats.non_model_data_list("cuda"))
        self._start_flag = False

    def record_model_data_volume(
        self,
        current_model_data_volume: int,
        current_optimizer_data_volume: int,
    ) -> None:
        """
        Sampling model data statistics.
        """
        if self._start_flag and not self.use_outside_memstats:
            self._memstats.record_max_cuda_model_data(
                current_model_data_volume + current_optimizer_data_volume
            )

    def sample_overall_data(self) -> None:
        """
        Sampling overall and non model data cuda memory statistics.
        """
        if self._start_flag and not self.use_outside_memstats:
            cuda_overall = self._mem_monitor.finish()
            self._memstats.record_max_cuda_overall_data(cuda_overall)
            self._memstats.calc_max_cuda_non_model_data()

            self._mem_monitor.start()

        if self._start_flag:
            self._sampling_time.append(time.time())

    def clear(self) -> None:
        self._memstats.clear()
        self._start_flag = False
        self._step_idx = 0
        self._step_total = 0

