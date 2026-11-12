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
        # Checkpointed regions, keyed by the module that creates the checkpoint.
        self._recomputed_submodules: dict[nn.Module, list[nn.Module]] = {}
        # Maps each collected unit to itself and its descendants outside nested units.
        self._owned_modules_by_unit: dict[nn.Module, dict[nn.Module, None]] = {}

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules children before parents, including the root last."""
        self._units = []
        self._recomputed_submodules = {}
        self._visit(module)
        if not isinstance(module, self._fsdp_unit_module_types):
            self._units.append(module)
        self._owned_modules_by_unit = {unit: {} for unit in self._units}
        for unit, owned in self._owned_modules_by_unit.items():
            owned[unit] = None
            pending = list(unit.children())
            while pending:
                submodule = pending.pop()
                if submodule not in self._owned_modules_by_unit:
                    owned[submodule] = None
                    pending.extend(submodule.children())
        return self._units

    def is_partially_recomputed(self, unit: nn.Module) -> bool:
        """Whether activation recomputation reruns part of this unit's forward without calling
        the unit.

        The rerun uses the unit's weights without its hooks, so the unit needs
        SchedulePolicy.unshard_all_planes_in_backward. Modules inside a nested unit don't
        count: their own hooks run during recomputation.
        """
        owned = self._owned_modules_by_unit[unit]
        for module in owned:
            for rerun in self._recomputed_submodules.get(module, []):
                for submodule in rerun.modules():
                    # Skip modules of nested units, which own their weights.
                    if submodule not in owned:
                        continue
                    # Skip parameterless modules, e.g. an IdentityOp MLP.
                    if next(submodule.parameters(recurse=False), None) is not None:
                        return True
        return False

    def _visit(self, module: nn.Module, collect_units: bool = True) -> None:
        """Record known checkpoint regions, optionally selecting independent units."""
        if isinstance(module, HyperConnectionHybridLayer):
            self._visit_mhc(module, collect_units)
            return
        if isinstance(module, ShortcutMoEBlock):
            self._visit_shortcut(module, collect_units)
            return
        if isinstance(module, TransformerLayer):
            self._visit_transformer(module, collect_units)
            return
        if isinstance(module, MambaLayer):
            self._visit_mamba(module, collect_units)
            return

        if isinstance(module, (TransformerBlock, HybridStack)):
            config = module.config
            if config.recompute_granularity == "full":
                # FP8 can shift the checkpointed prefix at runtime; include all layers.
                if config.recompute_method != "block" or config.recompute_num_layers:
                    self._recomputed_submodules[module] = list(module.layers)
        elif isinstance(module, MultiTokenPredictionLayer):
            # GPT MTP checkpoints eh_proj and its inner layer. Hybrid MTP delegates
            # full recompute to its inner stack, which the visitor handles separately.
            if module.config.recompute_granularity == "full" and not module.is_hybrid_mtp:
                self._recomputed_submodules[module] = [module]
        elif isinstance(module, MoELayer):
            if module.config.recompute_granularity != "full":
                if module.moe_layer_recompute:
                    self._recomputed_submodules[module] = [module]
                elif module.shared_experts_recompute and module.shared_experts is not None:
                    self._recomputed_submodules[module] = [module.shared_experts]
        elif isinstance(
            module, (MultiLatentAttention, AbsorbedMLASelfAttention, DSv4HybridAttention)
        ):
            if module.config.recompute_granularity != "full" and module.recompute_up_proj:
                # These implementations checkpoint an internal projection helper rather
                # than the attention module's __call__. Conservatively include its weights.
                self._recomputed_submodules[module] = [module]
        elif isinstance(module, _GDNBase):
            if module.config.recompute_granularity != "full" and module.recompute_gdn:
                # GDN/GDN2 checkpoint the two-stage helpers; KDA checkpoints _forward_compute.
                self._recomputed_submodules[module] = [module]
        elif isinstance(module, GatedDeltaProductMixer):
            if module.config.recompute_granularity != "full" and module.recompute_in_proj:
                self._recomputed_submodules[module] = [module.in_proj]

        self._visit_children(module, collect_units)
        if collect_units:
            self._collect(module)

    def _collect(self, module: nn.Module) -> None:
        if isinstance(module, self._fsdp_unit_module_types):
            self._units.append(module)

    def _visit_transformer(self, layer: TransformerLayer, collect_units: bool) -> None:
        if layer.config.recompute_granularity != "full":
            if layer.recompute_mlp:
                self._recomputed_submodules[layer] = [layer.mlp]
            if isinstance(layer, MoETransformerLayer) and layer.moe_layer_recompute:
                # Partial CUDA-graph replay also reruns the pre-MLP norm through helpers.
                self._recomputed_submodules[layer] = [layer.pre_mlp_layernorm, layer.mlp]
        # Known layers own their remaining internals. Visit them for recomputation
        # metadata without selecting extra units from the configured module types.
        self._visit_children(layer, collect_units=False)
        if collect_units:
            self._collect_routed_experts(layer)
            self._collect(layer)

    def _visit_mamba(self, layer: MambaLayer, collect_units: bool) -> None:
        self._visit_children(layer, collect_units=False)
        if collect_units:
            self._collect(layer)

    def _visit_children(self, module: nn.Module, collect_units: bool) -> None:
        for child in module.children():
            self._visit(child, collect_units)

    def _visit_shortcut(self, block: ShortcutMoEBlock, collect_units: bool) -> None:
        # Helpers bypass both inner layers' __call__, but experts run normally.
        self._visit_children(block, collect_units=False)
        if collect_units:
            self._collect_routed_experts(block.moe_layer)
            self._collect(block)

    def _visit_mhc(self, mhc_layer: HyperConnectionHybridLayer, collect_units: bool) -> None:
        self._visit_children(mhc_layer, collect_units=False)
        inner_layer = mhc_layer.inner_layer
        if collect_units:
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
