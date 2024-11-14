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
from typing import Optional

from .memory_monitor import SyncCudaMemoryMonitor
from .memory_stats import MemStats


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

    # NOTE: called the object with start_collection --> sample_overall_data --> finish_collection
    # Then self._memstats.non_model_data_list("cuda") will contains non_model_data size during each step
    def start_collection(self):
        self._start_flag = True
        self._mem_monitor.start()

    def finish_collection(self):
        self.sample_overall_data()
        # self._step_total = len(self._sampling_time)
        self._step_total = len(self._memstats.non_model_data_list("cuda"))
        self._start_flag = False
        # print(f"finish_collection {self._step_total}")

    # deprecated
    def record_model_data_volume(
        self,
        current_model_data_volume: int,
        current_optimizer_data_volume: int,
    ) -> None:
        """
        Sampling model data statistics.
        """
        # TODO: remove colossal-ai dependency
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

    def on_iter_end(self) -> bool:
        """
        During warmup, the non-model data is regarded as the amount of non-model data in each step.
        This strategy is to avoid OOM in the next iteration.

        The warmup ends only when the current max non-model data is no more than amount of last iteration.
        """
        in_warmup = self._warmup
        nmd_cur_iter = self._memstats.max_non_model_data('cuda')
        nmd_warmup = self.warmup_memstats.max_non_model_data('cuda')
        self._warmup = nmd_cur_iter > nmd_warmup
        if self._warmup:
            self.warmup_memstats.record_max_cuda_model_data(self._memstats._prev_md_cuda)
            self.warmup_memstats.non_model_data_list('cuda').append(
                sum(self._memstats.non_model_data_list('cuda'))
            )
        # clean memstats
        self._memstats.clear()
        return in_warmup
