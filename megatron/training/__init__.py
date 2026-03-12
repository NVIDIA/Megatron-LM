# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import signal
import sys

import torch
import torch.distributed as dist
from datetime import timedelta

from .global_vars import get_args
from .global_vars import get_signal_handler
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_wandb_writer
from .global_vars import get_one_logger
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .initialize  import initialize_megatron
from .training import pretrain, get_model, get_train_valid_test_num_samples, set_startup_timestamps

from .utils import (print_rank_0,
                    is_last_rank,
                    print_rank_last)


def graceful_shutdown(signum, frame):
    """
    Signal handler for user-initiated termination (SIGINT / SIGTERM).

    This handler attempts a best-effort graceful shutdown:
      - Logs a single termination message from rank 0
      - Synchronizes all ranks (barrier)
      - Destroys the distributed process group
      - Exits the process cleanly
    """
    print_rank_0("\nTermination requested. Performing orderly shutdown.")

    try:
        if dist.is_available() and dist.is_initialized():
            # synchronize all ranks before exiting
            try:
                # avoid deadlock if ranks don't all reach here
                dist.barrier(timeout=timedelta(seconds=5))
            except Exception:
                pass

            dist.destroy_process_group()
    except Exception:
        pass

    sys.exit(0)


# Register signal handlers for both:
#  - SIGINT  (Ctrl+C from user)
#  - SIGTERM (sent by torchrun to worker processes)
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)
