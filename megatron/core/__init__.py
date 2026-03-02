# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import signal
import sys

import torch.distributed as dist

import megatron.core.tensor_parallel
import megatron.core.utils
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.inference_params import InferenceParams
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)
from megatron.core.timers import Timers
from megatron.core.utils import is_torch_min_version

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
    "DistributedDataParallel",
    "InferenceParams",
    "ModelParallelConfig",
    "Timers",
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
    "__shortversion__",
    "__version__",
]

from .safe_globals import register_safe_globals

if is_torch_min_version("2.6a0"):
    register_safe_globals()


def graceful_shutdown(signum, frame):
    """
    Signal handler for user-initiated termination (SIGINT / SIGTERM).

    This handler attempts a best-effort graceful shutdown:
      - Logs a single termination message from rank 0
      - Synchronizes all ranks (barrier)
      - Destroys the distributed process group
      - Exits the process cleanly
    """
    from megatron.training import print_rank_0

    print_rank_0("\nTermination requested. Performing orderly shutdown.")

    try:
        if dist.is_available() and dist.is_initialized():
            # synchronize all ranks before exiting
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass

    sys.exit(0)


# Register signal handlers for both:
#  - SIGINT  (Ctrl+C from user)
#  - SIGTERM (sent by torchrun to worker processes)
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
