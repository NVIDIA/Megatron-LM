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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/utils/instantiate_utils.py
# Source license: Apache-2.0.

"""Use MLM's config reconstruction and target validation without replacing its resolver."""

from megatron.training.config.instantiate_utils import (  # noqa: F401
    InstantiationException,
    InstantiationMode,
    _call_target,
    _convert_node,
    _convert_target_to_string,
    _extract_pos_args,
    _filter_kwargs_for_target,
    _is_target,
    _Keys,
    _locate,
    _prepare_input_dict_or_list,
    _resolve_target,
    instantiate,
    instantiate_node,
    target_allowlist,
)

_ALLOWED_TARGET_PREFIXES: set[str] = set()


def register_allowed_target_prefix(prefix: str) -> None:
    """Explicitly opt a trusted application module into native config loading."""
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"Prefix must be a non-empty string, got {prefix!r}")
    target_allowlist.add_prefix(prefix if prefix.endswith(".") else prefix + ".")
    _ALLOWED_TARGET_PREFIXES.add(prefix)
