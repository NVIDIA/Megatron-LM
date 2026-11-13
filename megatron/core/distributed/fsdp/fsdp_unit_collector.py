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

from collections.abc import Sequence
from enum import IntEnum

from torch import nn

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.gated_delta_net.common import _GDNBase
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer


class RecomputeType(IntEnum):
    """Recompute requirements, ordered so internal replay takes precedence when combined."""

    NONE = 0
    FULL = 1  # Whole layers replay through their forward hooks.
    PARTIAL = 2  # Internal replay needs weights without re-entering the owning unit.


class FsdpUnitCollector:
    """Collect FSDP units children before parents.

    Known layers select their own inner units; other modules are traversed recursively.
    """

    def __init__(self, fsdp_unit_module_types: Sequence[type[nn.Module]] | None = None) -> None:
        if fsdp_unit_module_types is None:
            fsdp_unit_module_types = (
                TransformerLayer,
                MambaLayer,
                ShortcutMoEBlock,
                HyperConnectionHybridLayer,
            )
        self._fsdp_unit_module_types = tuple(fsdp_unit_module_types)
        self._units: list[nn.Module] = []
        self._partially_recomputed_units: set[nn.Module] = set()

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules children before parents, including the root last."""
        self._units = []
        self._partially_recomputed_units = set()
        self._visit(module, is_root=True)
        return self._units

    def is_partially_recomputed(self, unit: nn.Module) -> bool:
        """Whether this unit needs MXFP8 rowwise planes during backward recomputation."""
        return unit in self._partially_recomputed_units

    def _visit(self, module: nn.Module, *, is_root: bool = False) -> RecomputeType:
        """Collect this unit, or propagate its recompute type to its parent."""
        if isinstance(module, HyperConnectionHybridLayer):
            recompute_type = self._visit_mhc(module)
        elif isinstance(module, ShortcutMoEBlock):
            recompute_type = self._visit_shortcut(module)
        elif isinstance(module, TransformerLayer):
            recompute_type = self._visit_transformer(module)
        elif isinstance(module, MambaLayer):
            recompute_type = self._visit_mamba(module)
        elif isinstance(module, (TransformerBlock, HybridStack)):
            recompute_type = self._visit_children(module)
            config = module.config
            if config.recompute_granularity == "full" and (
                config.recompute_method != "block" or config.recompute_num_layers
            ):
                block_recompute_type = (
                    RecomputeType.PARTIAL
                    if any(layer not in self._units for layer in module.layers)
                    else RecomputeType.FULL
                )
                recompute_type = max(recompute_type, block_recompute_type)
        elif isinstance(module, MultiTokenPredictionLayer):
            recompute_type = self._visit_children(module)
            # GPT MTP checkpoints its projection; Hybrid MTP delegates to its inner stack.
            if module.config.recompute_granularity == "full" and not module.is_hybrid_mtp:
                recompute_type = RecomputeType.PARTIAL
        else:
            recompute_type = max(
                self._visit_children(module), self._component_recompute_type(module)
            )

        if is_root or isinstance(module, self._fsdp_unit_module_types):
            self._units.append(module)
            if recompute_type is RecomputeType.PARTIAL:
                self._partially_recomputed_units.add(module)
            return RecomputeType.NONE
        return recompute_type

    def _component_recompute_type(self, module: nn.Module) -> RecomputeType:
        """Check the current component's checkpoint flags without visiting its children."""
        if isinstance(module, MoELayer):
            # The flags already encode selective recompute. Whole-MoE replay uses the
            # router's weights even though routed experts have their own FSDP hooks.
            if module.moe_layer_recompute:
                return RecomputeType.PARTIAL
            if module.shared_experts_recompute and module.shared_experts is not None:
                return RecomputeType.PARTIAL
            return RecomputeType.NONE
        elif isinstance(module, Attention):
            # MLA variants share Attention but no narrower base. They expose this checkpoint
            # flag; other attention implementations may omit it.
            partially_recomputed = getattr(module, "recompute_up_proj", False)
        elif isinstance(module, _GDNBase):
            partially_recomputed = (
                module.config.recompute_granularity != "full" and module.recompute_gdn
            )
        elif isinstance(module, GatedDeltaProductMixer):
            partially_recomputed = (
                module.config.recompute_granularity != "full" and module.recompute_in_proj
            )
        else:
            return RecomputeType.NONE
        return RecomputeType.PARTIAL if partially_recomputed else RecomputeType.NONE

    def _visit_children(self, module: nn.Module) -> RecomputeType:
        recompute_type = RecomputeType.NONE
        for child in module.children():
            recompute_type = max(recompute_type, self._visit(child))
        return recompute_type

    def _visit_transformer(self, layer: TransformerLayer) -> RecomputeType:
        self._collect_routed_experts(layer)
        recompute_type = (
            RecomputeType.FULL
            if layer.config.recompute_granularity == "full"
            else RecomputeType.NONE
        )
        # These components belong to the layer; check their flags without collecting them.
        for child in layer.children():
            recompute_type = max(recompute_type, self._component_recompute_type(child))
        if layer.config.recompute_granularity != "full" and (
            (isinstance(layer, MoETransformerLayer) and layer.moe_layer_recompute)
            or (layer.recompute_mlp and next(layer.mlp.parameters(), None) is not None)
        ):
            recompute_type = RecomputeType.PARTIAL
        return recompute_type

    def _visit_mamba(self, layer: MambaLayer) -> RecomputeType:
        return max(
            (
                RecomputeType.FULL
                if layer.config.recompute_granularity == "full"
                else RecomputeType.NONE
            ),
            self._component_recompute_type(layer.mixer),
        )

    def _visit_shortcut(self, block: ShortcutMoEBlock) -> RecomputeType:
        # Helpers bypass both inner layers' __call__; their experts are still collected.
        return max(
            self._visit_inner_layer(block.compute_layer), self._visit_inner_layer(block.moe_layer)
        )

    def _visit_mhc(self, mhc_layer: HyperConnectionHybridLayer) -> RecomputeType:
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
        return self._visit_inner_layer(inner_layer)

    def _visit_inner_layer(self, layer: nn.Module) -> RecomputeType:
        """Visit a wrapper-owned layer without collecting it as an independent unit."""
        if isinstance(layer, TransformerLayer):
            return self._visit_transformer(layer)
        if isinstance(layer, MambaLayer):
            return self._visit_mamba(layer)
        return self._component_recompute_type(layer)

    def _collect_routed_experts(self, layer: TransformerLayer) -> None:
        if layer.is_moe_layer:
            self._units.append(layer.mlp.experts)
