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

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fully_shard


@pytest.mark.parametrize(
    ("arg_name", "arg_value"),
    [
        ("reshard_after_forward", False),
        ("shard_placement_fn", lambda param: None),
        ("offload_policy", object()),
    ],
)
def test_fully_shard_rejects_unsupported_pytorch_api_args(arg_name, arg_value):
    module = torch.nn.Linear(1, 1)

    with pytest.raises(
        NotImplementedError, match=f"Megatron FSDP v2 does not support `{arg_name}` yet."
    ):
        fully_shard(module, **{arg_name: arg_value})

    assert type(module) is torch.nn.Linear
