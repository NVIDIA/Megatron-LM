# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class RNGConfig:
    """Configuration settings for random number generation."""

    seed: int = 1234
    """Random seed used for python, numpy, pytorch, and cuda."""

    te_rng_tracker: bool = False
    """Use the Transformer Engine version of the random number generator.
    Required for CUDA graphs support."""

    inference_rng_tracker: bool = False
    """Use a random number generator configured for inference."""

    data_parallel_random_init: bool = False
    """Enable random initialization of params across data parallel ranks"""


@dataclass(kw_only=True)
class ProfilingConfig:
    """Configuration settings for profiling the training process."""

    use_nsys_profiler: bool = field(default=False, metadata={"argparse_meta": {"arg_names": ["--profile"], "dest": "profile"}})
    """Enable nsys profiling. When using this option, nsys options should be specified in
    commandline. An example nsys commandline is
    `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true
    --capture-range=cudaProfilerApi --capture-range-end=stop`.
    """

    profile_step_start: int = 10
    """Global step to start profiling."""

    profile_step_end: int = 12
    """Global step to stop profiling."""

    use_pytorch_profiler: bool = False
    """Use the built-in pytorch profiler. Useful if you wish to view profiles in tensorboard."""

    profile_ranks: list[int] = field(default_factory=lambda: [0])
    """Global ranks to profile."""

    record_memory_history: bool = False
    """Record memory history in last rank."""

    memory_snapshot_path: str = "snapshot.pickle"
    """Specifies where to dump the memory history pickle."""

    record_shapes: bool = False
    """Record shapes of tensors."""

    nvtx_ranges: bool = False
    """Enable NVTX range annotations for profiling. When enabled, inserts NVTX markers
    to categorize execution in profiler output."""
