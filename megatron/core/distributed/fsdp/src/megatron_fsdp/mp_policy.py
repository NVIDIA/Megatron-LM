# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Megatron-FSDP Mixed Precision Dataclass"""

    main_params_dtype: Optional[torch.dtype] = torch.float32
    """Data type for the main weight buffer utilized for distributed optimization
      and quantization with Megatron-FSDP. If set to None, the model compute weight
      buffer will take the role of the main weights, or when no sharding is applied,
      the native model weights become the main weights. Defaults to torch.float32.
    """

    main_grads_dtype: Optional[torch.dtype] = torch.float32
    """Data type for the main gradient buffer utilized for distributed optimization with
      Megatron-FSDP. If set to None, main gradients will match the dtype of the model
      compute parameters specified by the user model. Defaults to torch.float32.
    """

    grad_comm_dtype: Optional[torch.dtype] = torch.bfloat16
    """Data type for gradient gather / scatter communications. Can be utilized to reduce
      communication latency, but adds overhead for type-casting and copy operations.
      If set to None, the native model gradient dtype is used. Defaults to torch.bfloat16.
    """

    grad_accum_dtype: Optional[torch.dtype] = torch.float32
    """Data type for gradient reduction and accumulation to control accumulation precision.
      Specifically, gradients will be reduced at this precision, but accumulated either at
      this precision or higher precision based on type-promotion with the main_grads_dtype.
      If set to None, type-promotion with respect to the main_grads_dtype will determine
      the data-type when accumulating. Defaults to torch.float32.
    """
