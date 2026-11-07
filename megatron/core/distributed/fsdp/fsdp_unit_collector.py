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


def _recomputed(module: nn.Module, wrapper: nn.Module | None) -> bool:
    """Whether recomputation replays this module before the wrapper's pre_backward()."""
    if not isinstance(wrapper, HyperConnectionHybridLayer):
        return False
    config = wrapper.config
    if config.recompute_granularity != "selective" or "mhc" not in (config.recompute_modules or []):
        return False
    return module in (
        getattr(wrapper.inner_layer, "input_layernorm", None),
        getattr(wrapper.inner_layer, "pre_mlp_layernorm", None),
    )


class FsdpUnitCollector:
    """Collect selected FSDP units in wrapping order, children before parents."""

    def __init__(self, fsdp_unit_module_types: tuple[type[nn.Module], ...]) -> None:
        self._fsdp_unit_module_types = fsdp_unit_module_types
        self._collected_units_in_preorder: dict[nn.Module, None] = {}

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules in reverse pre-order, including the root last."""
        self._collected_units_in_preorder = {module: None}
        self._visit(module, wrapper=None)
        return list(reversed(self._collected_units_in_preorder))

    def _visit(self, module: nn.Module, wrapper: nn.Module | None) -> None:
        if self._should_shard(module, wrapper):
            self._collected_units_in_preorder[module] = None

        if isinstance(module, (ShortcutMoEBlock, HyperConnectionHybridLayer)):
            wrapper = module
        for child in module.children():
            self._visit(child, wrapper)

    def _should_shard(self, module: nn.Module, wrapper: nn.Module | None) -> bool:
        """Whether this module should be an independent FSDP unit."""
        # Replayed modules need their own hooks to gather parameters and reduce replay gradients.
        if _recomputed(module, wrapper) and next(module.parameters(), None) is not None:
            return True

        if not isinstance(module, self._fsdp_unit_module_types):
            return False

        # ShortcutMoEBlock and mHC invoke inner-layer helpers that bypass __call__.
        # Keep those parameters with the enclosing FSDP unit. Still visit children:
        # routed experts run through experts(...), so they retain their own hooks.
        if wrapper is not None and isinstance(module, (TransformerLayer, MambaLayer)):
            return False
        return True
