# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from dataclasses import dataclass, field
from typing import Literal
import os

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

    pytorch_profiler_collect_shapes: bool = False
    """Collect tensor shape in pytorch profiler."""
  
    pytorch_profiler_collect_callstack: bool = False
    """Collect callstack in pytorch profiler."""
  
    pytorch_profiler_collect_chakra: bool = False                
    """Collect chakra trace in pytorch profiler."""

    profile_ranks: list[int] = field(default_factory=lambda: [])
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


@dataclass(kw_only=True)
class DistributedInitConfig:
    """Configuration settings for distributed training initialization."""

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    """Which backend to use for distributed training."""

    distributed_timeout_minutes: int = 10
    """Timeout minutes for torch.distributed."""

    align_grad_reduce: bool = True
    """If not set, all PP stages will launch gradient reduces simultaneously.
    Otherwise, each PP stage will independently launch as needed.
    """

    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    """local rank passed from distributed launcher."""

    lazy_mpu_init: bool = False
    """If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead.
    Also turns on --use-cpu-initialization flag. This is for external DDP manager."""

    use_megatron_fsdp: bool = False
    """Use Megatron's Fully Sharded Data Parallel. Cannot be used together with use_torch_fsdp2."""

    use_torch_fsdp2: bool = False
    """Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel.
    It is still not in a stable release stage, and may therefore contain bugs or other
    potential issues."""

    nccl_communicator_config_path: str | None = None
    """Path to the yaml file with NCCL communicator configurations. The number of min/max thread
    groups and thread group cluster size of each communicator can be configured by setting
    `min_ctas`, `max_ctas`, and `cga_cluster_size`."""

    use_tp_pp_dp_mapping: bool = False
    """If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.
    """

    enable_gloo_process_groups: bool = field(default=True, metadata={"argparse_meta": {"arg_names": ["--disable-gloo-process-groups"]}})
    """If enabled, create Gloo process groups for communications."""

    use_sharp: bool = False
    """Set the use of SHARP for the collective communications of data-parallel process groups.
    When `True`, run barrier within each data-parallel process group,
    which specifies the SHARP application target groups.
    """

    sharp_enabled_group: Literal["dp", "dp_replica"] | None = None
    """IB SHARP can be enabled from only one communication group.
    By default, it is enabled from dp group if not specified and use_sharp=True.
    Available options: [dp, dp_replica]
    """

    high_priority_stream_groups: list[str] | None = field(default_factory=list)
    """Specify which communicator groups should use high priority streams during creation.
    Assigning high priority to communication streams ensures that communication kernels
    are scheduled with higher priority, minimizing the exposed communication when it is
    overlapped with other computation kernels.
    """

    distributed_timeout_seconds_after_init: int | None = None
    """Timeout in seconds for process groups after initialization. This timeout is applied to all process groups after initialization and the first iteration completes."""

    disable_jit_fuser: bool = False
    """Disable the JIT fuser."""
