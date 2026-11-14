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

"""Parameter-group membership policies, independent of buffer distribution."""

import enum


class ParameterGrouping(enum.Enum):
    """Choose which physical parameters share FSDP communication buffers."""

    PACKED = "packed"
    """Group parameters with the same effective dtype and requires-grad setting."""

    PER_PARAMETER = "per_parameter"
    """Create one group per physical parameter, keeping its tied aliases together."""
