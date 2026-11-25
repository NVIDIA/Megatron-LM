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

# Adapted from NVIDIA-NeMo/Megatron-Bridge (Apache-2.0).
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/utils.py

"""Explicit process-group access for adapter modules."""


def _get_pg_collection(pg_collection=None, source=None, *, required_pgs):
    if pg_collection is None:
        raise ValueError("Adapter construction requires an explicit pg_collection")
    for name in required_pgs:
        if getattr(pg_collection, name, None) is None:
            raise ValueError(f"Adapter construction requires pg_collection.{name}")
    return pg_collection


def _get_process_group(pg_collection, *names):
    for name in names:
        group = getattr(pg_collection, name, None)
        if group is not None:
            return group
    return None


def _process_group_size(group, fallback=1):
    return group.size() if group is not None else fallback


def _process_group_rank(group, fallback=0):
    return group.rank() if group is not None else fallback


def _get_tensor_parallel_group(pg_collection, *, is_expert=False):
    return _get_process_group(pg_collection, "expt_tp" if is_expert else "tp")


def _get_tensor_parallel_group_from_module(module, *, is_expert=False, pg_collection=None):
    groups = _get_pg_collection(pg_collection, required_pgs=["expt_tp" if is_expert else "tp"])
    return _get_tensor_parallel_group(groups, is_expert=is_expert)
