# Copyright (c) 2025, NVIDIA CORPORATION.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/decorators/torchrun.py
# Source license: Apache-2.0.

import os
import traceback
from functools import wraps

import torch
from torch.distributed.elastic.multiprocessing.errors import record


def torchrun_main(fn):
    """
    A decorator that wraps the main function of a torchrun script. It uses
    the `torch.distributed.elastic.multiprocessing.errors.record` decorator
    to record any exceptions and ensures that the distributed process group
    is properly destroyed on successful completion. In case of an exception,
    it prints the traceback and performs a hard exit, allowing torchrun to
    terminate all other processes.
    """
    recorded_fn = record(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return_value = recorded_fn(*args, **kwargs)
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return return_value
        except Exception:
            # The 'record' decorator might only log the exception to a file.
            # Print it to stderr as well to make sure it's visible.
            traceback.print_exc()
            # Use os._exit(1) for a hard exit. A regular sys.exit(1) might
            # not be enough to terminate a process stuck in a bad C++ state
            # (e.g., after a NCCL error), which can cause the job to hang.
            os._exit(1)

    return wrapper
