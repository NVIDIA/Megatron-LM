# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import sys
import os
from .utils import is_real_cuda_device_available

on_step_begin = []
on_step_end = []

def trigger(phase):
    [f() for f in phase]

def setup_profiler(profile_type,
                   profile_ranks,
                   profile_step_start,
                   profile_step_end,
                   tensorboard_dir):
    if profile_type is None or not torch.distributed.get_rank() in profile_ranks:
        return

    start_step, end_step = profile_step_start, profile_step_end
    active_steps = end_step - start_step + 1
    cur_step = 0
    cuda_available = is_real_cuda_device_available()

    def on_step_begin_fn():
        nonlocal cur_step
        cur_step = cur_step + 1
    on_step_begin.append(on_step_begin_fn)

    def when(cond, clbk):
        def fn():
            if cond():
                clbk()
        return fn

    def is_start_step():
        return cur_step == start_step

    def is_end_step():
        return cur_step == end_step

    def is_capture_step():
        return cur_step >= start_step and cur_step <= end_step

    if profile_type.startswith('pt'):
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=active_steps, repeat=1)
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.extend([torch.profiler.ProfilerActivity.HPU] if not cuda_available else [])
        activities.extend([torch.profiler.ProfilerActivity.CUDA] if cuda_available else [])
        full = profile_type == 'pt-full'

        profiler = torch.profiler.profile(
            schedule=schedule,
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir, use_gzip=True),
            with_stack=full)

        on_step_begin.append(when(is_start_step, profiler.start))
        on_step_end.append(when(is_capture_step, profiler.step))
        on_step_end.append(when(is_end_step, profiler.stop))

    elif profile_type == 'hltv':
        sys.path.append(os.environ['PYTORCH_MODULES_ROOT_PATH'])
        from topologies.tools import SynapseProfilerApi, TraceType
        api = SynapseProfilerApi()

        def on_start_step():
            nonlocal api
            api.profiler_start(TraceType.TraceAll, 0)

        def on_end_step():
            nonlocal api
            import habana_frameworks.torch.hpu as hpu
            hpu.synchronize()
            api.profiler_stop(TraceType.TraceAll, 0)
            api.profiler_get_trace_json(TraceType.TraceAll, 0)

        on_step_begin.append(when(is_start_step, on_start_step))
        on_step_end.append(when(is_end_step, on_end_step))