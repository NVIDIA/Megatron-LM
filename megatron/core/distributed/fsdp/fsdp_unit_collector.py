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

from collections.abc import Iterator, Sequence
from typing import NamedTuple

from torch import nn

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer


class _SelectiveRecompute(NamedTuple):
    """A rule for a module's internal selective checkpoint.

    The checkpoint replays an internal callable without re-entering the module's
    forward hooks. The region conservatively identifies parameters to inspect.
    """

    # Name of the bool attribute a module sets to True at construction when it recomputes,
    # e.g. "recompute_mlp" for TransformerLayer.
    recompute_flag_name: str
    # Name of the module's attribute holding the region root, e.g. "mlp" for
    # TransformerLayer, or None to inspect the current module's subtree.
    recomputed_submodule_name: str | None


_SELECTIVE_RECOMPUTES = (
    _SelectiveRecompute("recompute_mlp", "mlp"),  # TransformerLayer, "mlp"
    _SelectiveRecompute("moe_layer_recompute", None),  # MoELayer or MoETransformerLayer, "moe"
    _SelectiveRecompute("shared_experts_recompute", "shared_experts"),  # MoELayer
    _SelectiveRecompute("recompute_up_proj", None),  # MLA variants, "mla_up_proj"
    _SelectiveRecompute("recompute_gdn", None),  # Gated delta net variants, "gdn"
    _SelectiveRecompute("recompute_in_proj", "in_proj"),  # GatedDeltaProductMixer, "gdp_in_proj"
)


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

        # Collect FSDP units.
        self._visit(module)
        if not isinstance(module, self._fsdp_unit_module_types):
            self._units.append(module)

        # Mark units whose owned parameters are replayed by their internal checkpoints.
        for unit in self._units:
            owned_parameters = self._collect_owned_parameters(unit)
            recomputed_parameters = self._collect_recomputed_parameters(unit)
            if not owned_parameters.isdisjoint(recomputed_parameters):
                self._partially_recomputed_units.add(unit)
        return self._units

    def is_partially_recomputed(self, unit: nn.Module) -> bool:
        """Whether activation recomputation reruns part of this unit's forward without calling
        the unit.

        The rerun uses the unit's weights without its hooks, so the unit needs
        SchedulePolicy.unshard_all_planes_in_backward. Modules inside a nested unit don't
        count: their own hooks run during recomputation.
        """
        return unit in self._partially_recomputed_units

    def _collect_owned_parameters(self, unit: nn.Module) -> set[nn.Parameter]:
        """Collect this unit's parameters, excluding nested FSDP units."""
        parameters = set()
        for owned in self._owned_modules(unit):
            parameters.update(owned.parameters(recurse=False))
        return parameters

    def _collect_recomputed_parameters(self, unit: nn.Module) -> set[nn.Parameter]:
        """Collect parameter candidates from checkpoints defined inside this unit.

        Region parameters include nested units' weights; the caller filters by ownership.
        """
        recomputed = set()
        for owned in self._owned_modules(unit):
            for region in _rerun_submodules(owned):
                recomputed.update(region.parameters())
        return recomputed

    def _owned_modules(self, unit: nn.Module) -> Iterator[nn.Module]:
        """Yield this unit and its descendants outside independent child units."""
        units = set(self._units)
        stack = [unit]
        while stack:
            owned = stack.pop()
            yield owned
            for child in reversed(list(owned.children())):
                if child not in units:
                    stack.append(child)

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


def _rerun_submodules(module: nn.Module) -> list[nn.Module]:
    """Return parameter-region roots for checkpoints defined inside this module."""
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    # Read the module's own config: hybrid layers may each have one.
    config = getattr(module, "config", None)
    if config is not None and config.recompute_granularity == "full":
        if isinstance(module, (TransformerBlock, HybridStack)):
            # Block recompute may checkpoint a prefix of the layers; FP8 can shift it at
            # runtime, so conservatively take all layers.
            if config.recompute_method == "block" and not config.recompute_num_layers:
                return []
            return list(module.layers)
        # GPT MTP layers checkpoint eh_proj together with their inner layer. Hybrid MTP
        # layers delegate full recompute to their inner stack.
        if isinstance(module, MultiTokenPredictionLayer) and not module.is_hybrid_mtp:
            return [module]
        return []

    # Selective recompute flags are set from each module's own config.
    rerun = []
    for recompute in _SELECTIVE_RECOMPUTES:
        if getattr(module, recompute.recompute_flag_name, False) is not True:
            continue
        if recompute.recomputed_submodule_name is None:
            rerun.append(module)
            continue
        # The submodule may be absent, e.g. shared_experts is None without shared experts.
        submodule = getattr(module, recompute.recomputed_submodule_name, None)
        if submodule is not None:
            rerun.append(submodule)
    return rerun
