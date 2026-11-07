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

from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.transformer_layer import TransformerLayer


class FsdpUnitCollector:
    """Collect FSDP units children before parents.

    Known layers select their own inner units; other modules are traversed recursively.
    """

    def __init__(self, fsdp_unit_module_types: tuple[type[nn.Module], ...]) -> None:
        self._fsdp_unit_module_types = fsdp_unit_module_types
        self._units: list[nn.Module] = []

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules children before parents, including the root last."""
        self._units = []
        self._visit(module)
        if not isinstance(module, self._fsdp_unit_module_types):
            self._units.append(module)
        return self._units

    def _visit(self, module: nn.Module) -> None:
        if isinstance(module, HyperConnectionHybridLayer):
            self._visit_mhc(module)
        elif isinstance(module, ShortcutMoEBlock):
            self._visit_shortcut(module)
        elif isinstance(module, TransformerLayer):
            self._visit_transformer(module)
        elif isinstance(module, MambaLayer):
            self._visit_mamba(module)
        else:
            self._visit_children(module)
            self._collect(module)

    def _collect(self, module: nn.Module) -> None:
        if isinstance(module, self._fsdp_unit_module_types):
            self._units.append(module)

    def _visit_transformer(self, layer: TransformerLayer) -> None:
        self._collect_routed_experts(layer)
        self._collect(layer)

    def _visit_mamba(self, layer: MambaLayer) -> None:
        self._collect(layer)

    def _visit_children(self, module: nn.Module) -> None:
        for child in module.children():
            self._visit(child)

    def _visit_shortcut(self, block: ShortcutMoEBlock) -> None:
        # Helpers bypass both inner layers' __call__, but experts run normally.
        self._collect_routed_experts(block.moe_layer)
        self._collect(block)

    def _visit_mhc(self, mhc_layer: HyperConnectionHybridLayer) -> None:
        inner_layer = mhc_layer.inner_layer
        if isinstance(inner_layer, TransformerLayer):
            config = mhc_layer.config
            recompute_norms = config.recompute_granularity == "selective" and "mhc" in (
                config.recompute_modules or []
            )
            if recompute_norms:
                # Early replay requires parameterized norms to have their own hooks.
                for norm in (inner_layer.input_layernorm, inner_layer.pre_mlp_layernorm):
                    if next(norm.parameters(), None) is not None:
                        self._units.append(norm)
            self._collect_routed_experts(inner_layer)
        self._collect(mhc_layer)

    def _collect_routed_experts(self, layer: TransformerLayer) -> None:
        if layer.is_moe_layer:
            self._units.append(layer.mlp.experts)
