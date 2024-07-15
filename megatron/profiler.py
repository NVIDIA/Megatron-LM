# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import torch

on_step_begin = []
on_step_end = []

def trigger(phase):
    [f() for f in phase]

def setup_profiler(args, device):
    if args.profile is None:
        return

    start_step, end_step = map(int, args.profile_steps.split(','))
    active_steps = end_step - start_step + 1
    cur_step = 0

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

    if args.profile.startswith('pt'):
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=active_steps, repeat=1)
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.extend([torch.profiler.ProfilerActivity.HPU] if device.startswith("hpu") else [])
        activities.extend([torch.profiler.ProfilerActivity.CUDA] if device.startswith("cuda") else [])
        full = args.profile == 'pt-full'

        profiler = torch.profiler.profile(
            schedule=schedule,
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.tensorboard_dir, use_gzip=True),
            with_stack=full)

        on_step_begin.append(when(is_start_step, profiler.start))
        on_step_end.append(when(is_capture_step, profiler.step))
        on_step_end.append(when(is_end_step, profiler.stop))
