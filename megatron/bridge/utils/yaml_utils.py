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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/utils/yaml_utils.py
# Source license: Apache-2.0.

from typing import Any, Optional

import yaml

from megatron.training.config.yaml_utils import _enum_representer  # noqa: F401
from megatron.training.config.yaml_utils import _function_representer  # noqa: F401
from megatron.training.config.yaml_utils import _generation_config_representer  # noqa: F401
from megatron.training.config.yaml_utils import _partial_representer  # noqa: F401
from megatron.training.config.yaml_utils import _pretrained_config_representer  # noqa: F401
from megatron.training.config.yaml_utils import _safe_object_representer  # noqa: F401
from megatron.training.config.yaml_utils import _torch_dtype_representer  # noqa: F401
from megatron.training.config.yaml_utils import safe_yaml_representers


def dump_dataclass_to_yaml(obj: Any, filename: Optional[str] = None) -> Optional[str]:
    """Dump a dataclass object or other Python object to a YAML file or string.

    Uses safe representers to handle common types.

    Args:
        obj: The object to dump.
        filename: If provided, the path to the file where YAML should be written.
                  If None, returns the YAML string directly.

    Returns:
        If filename is None, returns the YAML string representation of the object.
        Otherwise, returns None.
    """
    with safe_yaml_representers():
        if filename is not None:
            with open(filename, "w+") as f:
                yaml.safe_dump(obj, f)
        else:
            return yaml.safe_dump(obj)
