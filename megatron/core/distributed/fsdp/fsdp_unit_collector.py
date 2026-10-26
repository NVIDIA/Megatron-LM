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

"""Select MCore modules for MFSDP v2 sharding."""

from torch import nn

from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer


class FsdpUnitCollector:
    """Collect dense and expert modules in pre-order, including the root."""

    def __init__(self, fsdp_unit_modules: tuple[type[nn.Module], ...]) -> None:
        self._fsdp_unit_modules = fsdp_unit_modules
        self._units_to_shard: list[nn.Module] = []

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules in pre-order, parents before children."""
        self._units_to_shard = [module]
        self._visit(module, inside_shortcut=False)
        return self._units_to_shard

    def _visit(self, module: nn.Module, inside_shortcut: bool) -> None:
        # Shortcut blocks call these containers through helpers, bypassing their hooks.
        if (
            isinstance(module, self._fsdp_unit_modules)
            and not (
                inside_shortcut and isinstance(module, (TransformerLayer, MambaLayer, MoELayer))
            )
            and module not in self._units_to_shard
        ):
            self._units_to_shard.append(module)
        if isinstance(module, MoELayer):
            if module.experts not in self._units_to_shard:
                self._units_to_shard.append(module.experts)

        inside_shortcut = inside_shortcut or isinstance(module, ShortcutMoEBlock)
        for child in module.children():
            self._visit(child, inside_shortcut)
