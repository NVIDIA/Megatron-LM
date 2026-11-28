# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/src/megatron/bridge/models/hf_pretrained/utils.py
# Source license: Apache-2.0.
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

import logging

logger = logging.getLogger(__name__)


def is_safe_repo(hf_path: str, trust_remote_code: bool | None) -> bool:
    """
    Decide whether remote code execution should be enabled for a Hugging Face
    model or dataset repository.

    This function follows two rules:
        1. If `trust_remote_code` is explicitly provided (True/False), its value
            takes precedence.
        2. If `trust_remote_code` is None, remote code execution is disabled.

    Args:
        hf_path (str):
            The Hugging Face repository identifier (e.g., "org/model_name").
        trust_remote_code (bool | None):
            If True, always allow remote code execution.
            If False, always disable it.
            If None, disable remote code execution.

    Returns:
        bool: Whether remote code execution should be enabled.
    """
    if trust_remote_code is not None:
        if trust_remote_code is False:
            logger.warning(
                "`trust_remote_code=False`. Remote code may not be executed. "
                "Set `trust_remote_code=True` only if you fully trust the Hugging Face repository."
            )
        return trust_remote_code

    logger.warning(
        "`trust_remote_code` was not explicitly set for %s. Remote code execution is disabled by default. "
        "Set `trust_remote_code=True` only if you fully trust the Hugging Face repository.",
        hf_path,
    )
    return False
