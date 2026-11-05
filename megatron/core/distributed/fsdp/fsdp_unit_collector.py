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

from typing import NamedTuple, Sequence

from torch import nn

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.transformer.transformer_layer import TransformerLayer


class _SelectiveRecompute(NamedTuple):
    """A module's selective recompute, which reruns part of its forward in backward.

    The rerun recomputes activations with the rerun part's weights but does not call the
    forward hooks of the FSDP unit owning those weights.
    """

    # Name of the bool attribute a module sets to True at construction when it recomputes,
    # e.g. "recompute_mlp" for TransformerLayer.
    recompute_flag_name: str
    # Name of the module's attribute holding the child that the recompute reruns, e.g. "mlp"
    # for TransformerLayer, or None when the module reruns itself.
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
    """Collect selected FSDP units in wrapping order, children before parents."""

    def __init__(self, fsdp_unit_module_types: Sequence[type[nn.Module]] | None = None) -> None:
        """Initialize the collector.

        Args:
            fsdp_unit_module_types: Module types to shard as child FSDP units.
        """
        if fsdp_unit_module_types is None:
            from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP

            fsdp_unit_module_types = (
                TransformerLayer,
                MambaLayer,
                ShortcutMoEBlock,
                TEGroupedMLP,
                SequentialMLP,
            )
        self._fsdp_unit_module_types = tuple(fsdp_unit_module_types)
        # Maps each collected unit, in pre-order, to the modules it owns: itself and its
        # descendants outside nested units.
        self._owned_modules_by_unit: dict[nn.Module, dict[nn.Module, None]] = {}

    def collect(self, module: nn.Module) -> list[nn.Module]:
        """Return selected modules in reverse pre-order, including the root last."""
        self._owned_modules_by_unit = {}
        self._visit(module, owner=module, under_shortcut=False)
        return list(reversed(self._owned_modules_by_unit))

    def is_partially_recomputed(self, unit: nn.Module) -> bool:
        """Whether activation recomputation reruns part of this unit's forward without calling
        the unit.

        The rerun uses the unit's weights without its hooks, so the unit needs
        SchedulePolicy.unshard_all_planes_in_backward. Modules inside a nested unit don't
        count: their own hooks run during recomputation.
        """
        owned = self._owned_modules_by_unit[unit]
        for module in owned:
            for rerun in _rerun_submodules(module):
                for submodule in rerun.modules():
                    # Skip modules of nested units, which own their weights.
                    if submodule not in owned:
                        continue
                    # Skip parameterless modules, e.g. an IdentityOp MLP.
                    if next(submodule.parameters(recurse=False), None) is not None:
                        return True
        return False

    def _visit(self, module: nn.Module, owner: nn.Module, under_shortcut: bool) -> None:
        # The root is always a unit.
        if module is owner or self._should_shard(module, under_shortcut):
            owner = module
            self._owned_modules_by_unit[owner] = {}
        self._owned_modules_by_unit[owner][module] = None

        under_shortcut = under_shortcut or isinstance(module, ShortcutMoEBlock)
        for child in module.children():
            self._visit(child, owner, under_shortcut)

    def _should_shard(self, module: nn.Module, under_shortcut: bool) -> bool:
        """Whether this module should be an independent FSDP unit."""
        if not isinstance(module, self._fsdp_unit_module_types):
            return False

        # ShortcutMoEBlock interleaves stages during training by calling
        # compute_layer.forward_pre_attn_and_core_attn()/forward_post_core_attn() and
        # moe_layer._pre_mlp_layernorm_and_residual(). These helper calls bypass the
        # layers' __call__ hooks. Sharding these modules separately would leave
        # their weights ungathered. Leave their parameters with the enclosing FSDP unit.
        # We still visit their children. In particular,
        # moe_layer.mlp.experts runs through experts(...), which invokes its hooks,
        # so the expert container can remain a separate FSDP unit.
        if under_shortcut and isinstance(module, (TransformerLayer, MambaLayer)):
            return False
        return True


def _rerun_submodules(module: nn.Module) -> list[nn.Module]:
    """Return the parts of this module's forward that its own checkpoints rerun in backward."""
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
    from megatron.core.transformer.transformer_block import TransformerBlock

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
