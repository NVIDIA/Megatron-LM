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
        self._units: dict[nn.Module, None] = {}
        self._partially_recomputed_units: set[nn.Module] = set()

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules children before parents, including the root last."""
        self._units = {}
        self._partially_recomputed_units = set()
        partially_recomputed = self._visit(module)
        if module not in self._units:
            self._units[module] = None
            if partially_recomputed:
                self._partially_recomputed_units.add(module)
        return list(self._units)

    def is_partially_recomputed(self, unit: nn.Module) -> bool:
        """Whether this unit needs MXFP8 rowwise planes during backward recomputation."""
        return unit in self._partially_recomputed_units

    def _visit(self, module: nn.Module) -> bool:
        """Return whether recomputation needs weights left with the enclosing unit."""
        if isinstance(module, HyperConnectionHybridLayer):
            return self._visit_mhc(module)
        if isinstance(module, ShortcutMoEBlock):
            return self._visit_shortcut(module)
        if isinstance(module, TransformerLayer):
            self._collect_routed_experts(module)
            return self._collect(module)
        if isinstance(module, MambaLayer):
            return self._collect(module)
        partially_recomputed = False
        for child in module.children():
            partially_recomputed |= self._visit(child)
        return self._collect(module, partially_recomputed)

    def _collect(self, module: nn.Module, partially_recomputed: bool = False) -> bool:
        partially_recomputed |= self._needs_all_planes(module)
        if isinstance(module, self._fsdp_unit_module_types):
            self._units[module] = None
            if partially_recomputed:
                self._partially_recomputed_units.add(module)
            return False
        # An unselected layer leaves its weights and recompute requirement with its parent.
        return partially_recomputed

    def _needs_all_planes(self, module: nn.Module) -> bool:
        """Check the known layer's config and the submodules it owns."""
        if isinstance(module, TransformerLayer):
            if module.config.recompute_granularity == "full":
                return False
            if isinstance(module, MoETransformerLayer) and module.moe_layer_recompute:
                return True
            return self._mlp_needs_all_planes(module.mlp, module.recompute_mlp) or (
                self._needs_all_planes(module.self_attention)
            )
        if isinstance(module, MambaLayer):
            return self._needs_all_planes(module.mixer)
        if isinstance(module, ShortcutMoEBlock):
            return self._needs_all_planes(module.compute_layer) or self._needs_all_planes(
                module.moe_layer
            )
        if isinstance(module, HyperConnectionHybridLayer):
            return self._needs_all_planes(module.inner_layer)
        if isinstance(module, (TransformerBlock, HybridStack)):
            config = module.config
            return (
                config.recompute_granularity == "full"
                and (config.recompute_method != "block" or bool(config.recompute_num_layers))
                and any(layer not in self._units for layer in module.layers)
            )
        if isinstance(module, MultiTokenPredictionLayer):
            # GPT MTP checkpoints its projection inside forward; Hybrid MTP delegates
            # full recompute to its inner stack.
            return module.config.recompute_granularity == "full" and not module.is_hybrid_mtp
        if isinstance(module, MoELayer):
            return self._mlp_needs_all_planes(module)
        if isinstance(
            module, (MultiLatentAttention, AbsorbedMLASelfAttention, DSv4HybridAttention)
        ):
            return module.config.recompute_granularity != "full" and module.recompute_up_proj
        if isinstance(module, _GDNBase):
            return module.config.recompute_granularity != "full" and module.recompute_gdn
        if isinstance(module, GatedDeltaProductMixer):
            return module.config.recompute_granularity != "full" and module.recompute_in_proj
        return False

    def _mlp_needs_all_planes(self, mlp: nn.Module, recompute_mlp: bool = False) -> bool:
        if isinstance(mlp, MoELayer):
            if mlp.config.recompute_granularity == "full":
                return False
            if recompute_mlp or mlp.moe_layer_recompute:
                # Routed experts have their own hooks; router, shared experts and latent
                # projections stay with the enclosing layer.
                return any(
                    next(child.parameters(), None) is not None
                    for child in mlp.children()
                    if child not in self._units
                )
            return (
                mlp.shared_experts_recompute
                and mlp.shared_experts is not None
                and next(mlp.shared_experts.parameters(), None) is not None
            )
        return recompute_mlp and next(mlp.parameters(), None) is not None

    def _visit_shortcut(self, block: ShortcutMoEBlock) -> bool:
        # Helpers bypass both inner layers' __call__, but experts run normally.
        self._collect_routed_experts(block.moe_layer)
        return self._collect(block)

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
                        self._units[norm] = None
            self._collect_routed_experts(inner_layer)
        return self._collect(mhc_layer)

    def _collect_routed_experts(self, layer: TransformerLayer) -> None:
        if layer.is_moe_layer:
            self._units[layer.mlp.experts] = None
