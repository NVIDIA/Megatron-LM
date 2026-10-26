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
    """Collect selected units and their routed experts in pre-order."""

    def __init__(self, fsdp_unit_module_types: tuple[type[nn.Module], ...]) -> None:
        self._fsdp_unit_module_types = fsdp_unit_module_types
        self._collected_units_in_preorder: dict[nn.Module, None] = {}

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules in pre-order, parents before children."""
        self._collected_units_in_preorder = {module: None}
        self._visit(module, under_shortcut=False)
        return list(self._collected_units_in_preorder)

    def _visit(self, module: nn.Module, under_shortcut: bool) -> None:
        if self._should_shard(module, under_shortcut):
            self._collected_units_in_preorder[module] = None

        if MoELayer in self._fsdp_unit_module_types and isinstance(module, MoELayer):
            self._collected_units_in_preorder[module.experts] = None

        under_shortcut = under_shortcut or isinstance(module, ShortcutMoEBlock)
        for child in module.children():
            self._visit(child, under_shortcut)

    def _should_shard(self, module: nn.Module, under_shortcut: bool) -> bool:
        """Whether this module should be an independent FSDP unit."""
        if not isinstance(module, self._fsdp_unit_module_types):
            return False

        # ShortcutMoEBlock interleaves stages during training by calling
        # compute_layer.forward_pre_attn_and_core_attn()/forward_post_core_attn(),
        # moe_layer._pre_mlp_layernorm_and_residual(), and moe_layer.mlp.route()/dispatch().
        # These helper calls bypass the __call__ hooks of compute_layer,
        # moe_layer, and moe_layer.mlp. Sharding these modules separately would leave
        # their weights ungathered. Leave their parameters with the enclosing FSDP unit.
        # We still visit their children. In particular,
        # moe_layer.mlp.experts runs through experts(...), which invokes its hooks,
        # so selecting MoELayer also selects its experts for the expert-DP mesh.
        if under_shortcut and isinstance(module, (TransformerLayer, MambaLayer, MoELayer)):
            return False
        return True
