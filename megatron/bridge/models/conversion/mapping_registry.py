# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/conversion/mapping_registry.py
# Source license: Apache-2.0.

import re
from typing import Any, List, Optional

from megatron.bridge.models.conversion.param_mapping import AutoMapping, MegatronParamMapping
from megatron.bridge.models.conversion.quant_mapping import (
    convert_to_amax_map,
    derive_kv_bmm_amax_map,
)


class MegatronMappingRegistry:
    """
    Registry for weight mappings between model formats with pattern matching support.

    This class serves as a registry of weight mappings between Megatron and external
    (typically HuggingFace) model formats. It provides efficient pattern matching
    for parameter names using glob-like wildcards (*) and supports both forward
    (Megatron → HF) and reverse (HF → Megatron) lookups.

    The registry pre-compiles regex patterns for efficient repeated lookups and
    handles the resolution of wildcards in parameter names.

    Args:
        *mappings: Variable number of MegatronParamMapping objects defining
            the individual weight mappings

    Example:
        >>> # Create a mapping registry with various mappings
        >>> mapping_registry = MegatronMappingRegistry(
        ...     AutoMapping(
        ...         megatron_param="embedding.word_embeddings.weight",
        ...         hf_param="model.embed_tokens.weight",
        ...     ),
        ...     QKVMapping(
        ...         megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
        ...         q="model.layers.*.self_attn.q_proj.weight",
        ...         k="model.layers.*.self_attn.k_proj.weight",
        ...         v="model.layers.*.self_attn.v_proj.weight",
        ...     ),
        ... )

        >>> # Query for a specific layer (wildcards are resolved)
        >>> mapping = mapping_registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_qkv.weight")
        >>> print(mapping.hf_param)  # Will show resolved HF names for layer 0

        >>> # Reverse lookup from HF name
        >>> mapping = mapping_registry.hf_to_megatron_lookup("model.layers.5.self_attn.q_proj.weight")
        >>> print(mapping.megatron_param)  # Shows corresponding Megatron name

        >>> # Build from a list
        >>> mappings = [bridge1, bridge2, bridge3]
        >>> mapping_registry = MegatronMappingRegistry(*mappings)

    Note:
        Wildcard patterns support:
        - '*' matches any sequence of digits (0-9) - designed for layer indices
        - '**' matches any sequence of characters - designed for nested paths
    """

    # Mapping aliases for separate LayerNorm params vs fused TE LayerNormLinear params.
    # Source: https://github.com/NVIDIA/Megatron-LM/blob/3cf7a63fe9316102a498690ae4cf2b22d5ff4be0/megatron/core/post_training/modelopt/gpt/state_dict_hooks.py#L38-L51
    _SEPARATE_LAYERNORM_REWRITES = [
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("self_attention.linear_q_up_proj.layer_norm_weight", "self_attention.q_layernorm.weight"),
        ("self_attention.linear_q_up_proj.layer_norm_bias", "self_attention.q_layernorm.bias"),
        (
            "self_attention.linear_kv_up_proj.layer_norm_weight",
            "self_attention.kv_layernorm.weight",
        ),
        ("self_attention.linear_kv_up_proj.layer_norm_bias", "self_attention.kv_layernorm.bias"),
        ("mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight"),
        ("mlp.linear_fc1.layer_norm_bias", "pre_mlp_layernorm.bias"),
        ("mixer.in_proj.layer_norm_weight", "norm.weight"),
    ]

    def _add_separate_layernorm_mappings(self) -> None:
        """Add mapping aliases for separate LayerNorm module naming.

        Some Megatron implementations keep LayerNorm weights inside fused TE
        linear modules (e.g., `self_attention.linear_qkv.layer_norm_weight`),
        while others expose them as standalone modules (e.g., `input_layernorm.weight`).
        To avoid requiring every bridge to add both names, duplicate any matching
        mapping so both naming conventions resolve to the same HF parameter.
        """
        original_mappings = list(self.mappings)
        existing_names = {mapping.megatron_param for mapping in self.mappings}
        extra_mappings = []

        for mapping in original_mappings:
            for old_name, new_name in self._SEPARATE_LAYERNORM_REWRITES:
                if not mapping.megatron_param.endswith(f"*.{old_name}"):
                    continue
                new_megatron_param = mapping.megatron_param[: -len(old_name)] + new_name
                if new_megatron_param in existing_names:
                    break
                if isinstance(mapping, AutoMapping):
                    new_mapping = AutoMapping(
                        new_megatron_param, mapping.hf_param, mapping.permute_dims
                    )
                else:
                    print(
                        f"Unrecognized mapping type for {mapping.megatron_param} -> {mapping.hf_param}"
                    )
                    break
                extra_mappings.append(new_mapping)
                existing_names.add(new_megatron_param)
                break

        if extra_mappings:
            self.mappings.extend(extra_mappings)

    def _build_quantization_mappings(self) -> List[MegatronParamMapping]:
        """Derive ModelOpt quantizer-state (amax) mappings from the weight mappings."""
        return [
            *convert_to_amax_map(self.mappings, ".weight_quantizer._amax"),
            *convert_to_amax_map(self.mappings, ".input_quantizer._amax"),
            *derive_kv_bmm_amax_map(self.mappings),
        ]

    def _convert_pattern_to_regex(self, pattern: str) -> str:
        """Convert a pattern with wildcards to regex pattern.

        Args:
            pattern: Pattern string with * and ** wildcards

        Returns:
            Regex pattern string

        Note:
            ** must be processed before * to avoid conflicts.
            ** becomes (.*) - matches any characters including dots
            * becomes (\\d+) - matches digits only for layer indices
        """
        # Escape the pattern first
        regex_pattern = re.escape(pattern)

        # Process ** before * to avoid conflicts
        # Replace \*\* with (.*)
        regex_pattern = regex_pattern.replace(r"\*\*", r"(.*)")

        # Replace remaining \* with (\d+)
        regex_pattern = regex_pattern.replace(r"\*", r"(\d+)")

        return regex_pattern

    def __init__(self, *mappings: MegatronParamMapping):
        """
        Initialize MegatronMappingRegistry with weight mappings.

        Args:
            *mappings: MegatronParamMapping objects
        """
        self.mappings = list(mappings)
        self._pg_collection = None
        self._add_separate_layernorm_mappings()
        # Derived mappings for ModelOpt quantizer state: lookups resolve them, but they are not
        # listed with the weight mappings, so registries rebuilt from those do not duplicate them.
        self._quantization_mappings = self._build_quantization_mappings()

        # Pre-compile patterns for efficiency
        self._compiled_patterns = []
        self._reverse_patterns = []  # For hf_param -> megatron lookups

        for mapping in [*self.mappings, *self._quantization_mappings]:
            # Compile source patterns
            if "*" in mapping.megatron_param:
                # Convert glob pattern to regex with support for * and **
                pattern = self._convert_pattern_to_regex(mapping.megatron_param)
                self._compiled_patterns.append((re.compile(f"^{pattern}$"), mapping))
            else:
                self._compiled_patterns.append((None, mapping))

            # Compile destination patterns for reverse lookups
            if isinstance(mapping.hf_param, str):
                if "*" in mapping.hf_param:
                    pattern = self._convert_pattern_to_regex(mapping.hf_param)
                    self._reverse_patterns.append((re.compile(f"^{pattern}$"), mapping))
                else:
                    self._reverse_patterns.append((None, mapping))
            else:
                # For dict destinations, compile patterns for each value
                reverse_dict_patterns = {}
                for key, hf_pattern in mapping.hf_param.items():
                    if "*" in hf_pattern:
                        pattern = self._convert_pattern_to_regex(hf_pattern)
                        reverse_dict_patterns[key] = re.compile(f"^{pattern}$")
                    else:
                        reverse_dict_patterns[key] = None
                self._reverse_patterns.append((reverse_dict_patterns, mapping))

    def set_process_groups_from_pg_collection(self, pg_collection: Any) -> None:
        """Install a process-group collection on all mappings returned by this registry."""
        self._pg_collection = pg_collection
        for mapping in [*self.mappings, *self._quantization_mappings]:
            mapping.set_process_groups_from_pg_collection(pg_collection)

    def _prepare_mapping(self, mapping: MegatronParamMapping) -> MegatronParamMapping:
        mapping.set_process_groups_from_pg_collection(self._pg_collection)
        return mapping

    def resolve_mapping(
        self, mapping: MegatronParamMapping, captures: tuple[str, ...]
    ) -> MegatronParamMapping:
        """Resolve a mapping while preserving this registry's process groups."""
        return self._prepare_mapping(mapping.resolve(captures))

    def megatron_to_hf_lookup(self, megatron_param_name: str) -> Optional[MegatronParamMapping]:
        """
        Get mapping for a Megatron parameter name.

        This method performs efficient lookups by first checking for exact matches,
        then falling back to pattern matching using pre-compiled regex patterns.
        When a pattern match is found, wildcards are automatically resolved.

        Args:
            megatron_param_name: Megatron parameter name to look up
                Example: "decoder.layers.0.self_attention.linear_qkv.weight"

        Returns:
            MegatronParamMapping: Bridge instance with resolved wildcards, or None
                if no matching mapping is found. The returned bridge will have
                all wildcards replaced with actual values.

        Example:
            >>> # Query with exact layer number
            >>> bridge = state_map.megatron_to_hf_lookup("decoder.layers.5.mlp.linear_fc1.weight")
            >>> if bridge:
            ...     print(f"Maps to: {bridge.hf_param}")  # Shows HF name for layer 5
        """
        for pattern, mapping in self._compiled_patterns:
            if pattern is None:
                # Direct match
                if mapping.megatron_param == megatron_param_name:
                    return self._prepare_mapping(mapping)
            else:
                # Pattern match
                match = pattern.match(megatron_param_name)
                if match:
                    # Return resolved mapping with wildcards replaced
                    return self.resolve_mapping(mapping, match.groups())
        return None

    def hf_to_megatron_lookup(self, hf_param_name: str) -> Optional[MegatronParamMapping]:
        """
        Get mapping for a destination parameter name (reverse lookup).

        This is useful when you have a destination name and want to find
        the corresponding megatron name.

        Args:
            hf_param_name: Destination parameter name to look up

        Returns:
            MegatronParamMapping with resolved wildcards, or None if no match found
        """
        for pattern_info, mapping in self._reverse_patterns:
            if isinstance(mapping.hf_param, str):
                # Simple string destination
                pattern = pattern_info
                if pattern is None:
                    # Direct match
                    if mapping.hf_param == hf_param_name:
                        return self._prepare_mapping(mapping)
                else:
                    # Pattern match
                    match = pattern.match(hf_param_name)
                    if match:
                        return self.resolve_mapping(mapping, match.groups())
            else:
                # Dict destination - check each pattern
                patterns_dict = pattern_info
                for key, pattern in patterns_dict.items():
                    if pattern is None:
                        # Direct match
                        if mapping.hf_param[key] == hf_param_name:
                            # Create a simplified mapping for this specific key
                            return self.resolve_mapping(mapping, ())
                    else:
                        # Pattern match
                        match = pattern.match(hf_param_name)
                        if match:
                            return self.resolve_mapping(mapping, match.groups())
        return None

    def get_all_mappings(self) -> List[MegatronParamMapping]:
        """Get all mappings in this MegatronMappingRegistry."""
        return self.mappings.copy()

    def get_mappings_by_pattern(self, pattern: str) -> List[MegatronParamMapping]:
        """
        Get all mappings that match a given pattern.

        Args:
            pattern: Pattern to match (supports * and ** wildcards)

        Returns:
            List of matching MegatronParamMapping objects
        """
        # Convert pattern to regex using the same logic as _convert_pattern_to_regex
        # but for this method we want both * and ** to match anything for search purposes
        regex_pattern = re.escape(pattern)
        regex_pattern = regex_pattern.replace(r"\*\*", r".*")
        regex_pattern = regex_pattern.replace(r"\*", r".*")
        compiled_pattern = re.compile(f"^{regex_pattern}$")

        matches = []
        for mapping in self.mappings:
            if compiled_pattern.match(mapping.megatron_param):
                matches.append(mapping)

        return matches

    def __len__(self) -> int:
        """Return number of mappings."""
        return len(self.mappings)

    def __iter__(self):
        """Iterate over mappings."""
        return iter(self.mappings)

    def __repr__(self) -> str:
        """String representation of MegatronMappingRegistry."""
        return f"MegatronMappingRegistry({len(self.mappings)} mappings)"

    def describe(self) -> str:
        """
        Get a human-readable description of all mappings.

        Returns:
            Formatted string describing all weight mappings
        """
        lines = [f"MegatronMappingRegistry with {len(self.mappings)} mappings:"]
        for i, mapping in enumerate(self.mappings):
            lines.append(f"\n{i + 1}. {mapping.megatron_param}")
            if isinstance(mapping.hf_param, str):
                lines.append(f"   → {mapping.hf_param}")
            else:
                lines.append("   → {")
                for key, value in mapping.hf_param.items():
                    lines.append(f"       {key}: {value}")
                lines.append("     }")

            # Show bridge type
            lines.append(f"   bridge: {type(mapping).__name__}")

        return "\n".join(lines)
