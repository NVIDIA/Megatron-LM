# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from .distributed_data_parallel import DistributedDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .finalize_model_grads import finalize_model_grads

# For backwards compatibility. ParamAndGradBuffer will be deprecated in future release.
# ParamAndGradBuffer (which is an alias of _ParamAndGradBuffer) is not intended to be
# consumed directly by external code.
from .param_and_grad_buffer import ParamAndGradBuffer
