# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/common/unimodal.py
# Source license: Apache-2.0.

from megatron.training.models.dist_utils import _ddp_wrap  # noqa: F401
from megatron.training.models.dist_utils import _print_num_params  # noqa: F401
from megatron.training.models.dist_utils import _wrap_with_mp_wrapper  # noqa: F401
from megatron.training.models.dist_utils import build_virtual_pipeline_stages  # noqa: F401
from megatron.training.models.dist_utils import to_empty_if_meta_device  # noqa: F401
from megatron.training.models.dist_utils import unimodal_build_distributed_models  # noqa: F401
