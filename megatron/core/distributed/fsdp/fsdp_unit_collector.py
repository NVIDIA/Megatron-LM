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

from torch import nn

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.gated_delta_net.common import _GDNBase
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridAttention,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_latent_attention import MultiLatentAttention
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import MoETransformerLayer, TransformerLayer


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

    def _visit(self, module: nn.Module, *, is_root: bool = False) -> bool:
        """Collect this unit, or propagate its recomputation requirement to its parent."""
        if isinstance(module, HyperConnectionHybridLayer):
            partially_recomputed = self._visit_mhc(module)
        elif isinstance(module, ShortcutMoEBlock):
            partially_recomputed = self._visit_shortcut(module)
        elif isinstance(module, TransformerLayer):
            partially_recomputed = self._visit_transformer(module)
        elif isinstance(module, MambaLayer):
            partially_recomputed = self._has_partial_recompute(module.mixer)
        elif isinstance(module, (TransformerBlock, HybridStack)):
            partially_recomputed = self._visit_children(module)
            config = module.config
            partially_recomputed |= (
                config.recompute_granularity == "full"
                and (config.recompute_method != "block" or bool(config.recompute_num_layers))
                and any(layer not in self._units for layer in module.layers)
            )
        elif isinstance(module, MultiTokenPredictionLayer):
            partially_recomputed = self._visit_children(module)
            # GPT MTP checkpoints its projection; Hybrid MTP delegates to its inner stack.
            partially_recomputed |= (
                module.config.recompute_granularity == "full" and not module.is_hybrid_mtp
            )
        else:
            partially_recomputed = self._visit_children(module)
            partially_recomputed |= self._has_partial_recompute(module)

        if is_root or isinstance(module, self._fsdp_unit_module_types):
            self._units.append(module)
            if partially_recomputed:
                self._partially_recomputed_units.add(module)
            return False
        return partially_recomputed

    def _has_partial_recompute(self, module: nn.Module) -> bool:
        """Check the current component's checkpoint flags without visiting its children."""
        if isinstance(module, MoELayer):
            return module.config.recompute_granularity != "full" and (
                (module.moe_layer_recompute and self._mlp_has_owned_parameters(module))
                or (
                    module.shared_experts_recompute
                    and module.shared_experts is not None
                    and module.shared_experts not in self._units
                    and next(module.shared_experts.parameters(), None) is not None
                )
            )
        if isinstance(
            module, (MultiLatentAttention, AbsorbedMLASelfAttention, DSv4HybridAttention)
        ):
            return module.config.recompute_granularity != "full" and module.recompute_up_proj
        if isinstance(module, _GDNBase):
            return module.config.recompute_granularity != "full" and module.recompute_gdn
        if isinstance(module, GatedDeltaProductMixer):
            return module.config.recompute_granularity != "full" and module.recompute_in_proj
        return False

    def _visit_children(self, module: nn.Module) -> bool:
        partially_recomputed = False
        for child in module.children():
            partially_recomputed |= self._visit(child)
        return partially_recomputed

    def _visit_transformer(self, layer: TransformerLayer) -> bool:
        self._collect_routed_experts(layer)
        partially_recomputed = False
        # These components belong to the layer; check their flags without collecting them.
        for child in layer.children():
            partially_recomputed |= self._has_partial_recompute(child)
        if layer.config.recompute_granularity != "full":
            partially_recomputed |= (
                isinstance(layer, MoETransformerLayer) and layer.moe_layer_recompute
            ) or (layer.recompute_mlp and self._mlp_has_owned_parameters(layer.mlp))
        return partially_recomputed

    def _mlp_has_owned_parameters(self, mlp: nn.Module) -> bool:
        if isinstance(mlp, MoELayer):
            # Routed experts have their own hooks; the other children stay with the layer.
            return any(
                next(child.parameters(), None) is not None
                for child in mlp.children()
                if child not in self._units
            )
        return next(mlp.parameters(), None) is not None

    def _visit_shortcut(self, block: ShortcutMoEBlock) -> bool:
        # Helpers bypass both inner layers' __call__; their experts are still collected.
        return self._visit_inner_layer(block.compute_layer) | self._visit_inner_layer(
            block.moe_layer
        )

    def _visit_mhc(self, mhc_layer: HyperConnectionHybridLayer) -> bool:
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

    def _visit_inner_layer(self, layer: nn.Module) -> bool:
        """Visit a wrapper-owned layer without collecting it as an independent unit."""
        if isinstance(layer, TransformerLayer):
            return self._visit_transformer(layer)
        if isinstance(layer, MambaLayer):
            return self._has_partial_recompute(layer.mixer)
        return self._has_partial_recompute(layer)

    def _collect_routed_experts(self, layer: TransformerLayer) -> None:
        if layer.is_moe_layer:
            self._units.append(layer.mlp.experts)
