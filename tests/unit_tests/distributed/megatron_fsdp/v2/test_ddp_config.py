# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from megatron.core.distributed import DistributedDataParallelConfig


def test_use_megatron_fsdp_v2_implies_megatron_fsdp():
    ddp_config = DistributedDataParallelConfig(use_megatron_fsdp_v2=True)

    assert ddp_config.use_megatron_fsdp
