# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime profiling driven exclusively by explicit configuration and progress."""

from pathlib import Path

import torch

from megatron.core.utils import configure_nvtx_profiling
from megatron.training.config import ProfilingConfig


class TrainingProfiler:
    """Keep profiler handles separate from serializable profiling settings.

    Args:
        config: Authoritative profiling settings.
        rank: Current global rank.
        tensorboard_dir: Trace output location, supplied by the logging owner.
    """

    def __init__(self, config: ProfilingConfig, *, rank: int, tensorboard_dir: str | None) -> None:
        self.config = config
        self.rank = rank
        self.tensorboard_dir = tensorboard_dir
        self.profiler = None
        self.nvtx_context = None

    def _enabled(self) -> bool:
        return self.config.use_nsys_profiler and (
            not self.config.profile_ranks or self.rank in self.config.profile_ranks
        )

    def start(self) -> None:
        """Start PyTorch profiling before the loop, preserving its schedule."""
        config = self.config
        if not self._enabled() or not config.use_pytorch_profiler:
            return
        config.validate()
        if config.pytorch_profiler_collect_chakra:
            trace_dir = Path(f"{self.tensorboard_dir}/../chakra")
            trace_dir.mkdir(parents=True, exist_ok=True)
            observer = torch.profiler.ExecutionTraceObserver().register_callback(
                f"{trace_dir}/rank-{self.rank}.json.gz"
            )
        else:
            observer = None

        def trace_handler(profiler):
            profile_dir = Path(f"{self.tensorboard_dir}/../torch_profile")
            profile_dir.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(f"{profile_dir}/rank-{self.rank}.json.gz")

        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(config.profile_step_start - 1, 0),
                warmup=1 if config.profile_step_start > 0 else 0,
                active=config.profile_step_end - config.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
            record_shapes=config.pytorch_profiler_collect_shapes,
            with_stack=config.pytorch_profiler_collect_callstack,
            execution_trace_observer=observer,
        )
        self.profiler.start()

    def step(self, iteration: int) -> None:
        """Apply profiling at the start of the current training iteration."""
        config = self.config
        if not self._enabled():
            return
        if iteration == config.profile_step_start and config.nvtx_ranges:
            configure_nvtx_profiling(True)
        if config.use_pytorch_profiler:
            self.profiler.step()
        elif iteration == config.profile_step_start:
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
            if config.record_shapes:
                self.nvtx_context = torch.autograd.profiler.emit_nvtx(
                    record_shapes=config.record_shapes
                )
                self.nvtx_context.__enter__()

    def stop(self, iteration: int) -> None:
        """Stop at the configured end iteration, after the training step."""
        config = self.config
        if not self._enabled() or iteration != config.profile_step_end:
            return
        if config.nvtx_ranges:
            configure_nvtx_profiling(False)
        if config.use_pytorch_profiler:
            assert self.profiler is not None
            self.profiler.stop()
            if self.profiler.execution_trace_observer is not None:
                self.profiler.execution_trace_observer.unregister_callback()
        else:
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
            if self.nvtx_context is not None:
                self.nvtx_context.__exit__(None, None, None)
