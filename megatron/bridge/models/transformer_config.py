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

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/transformer_config.py
# Source license: Apache-2.0.

"""Bridge wrapper classes for Megatron Core transformer configurations.

These classes provide deferred post-initialization to support the Bridge configuration
override system while maintaining compatibility with Megatron Core's post_init behavior.
"""

import copy
from dataclasses import dataclass, field, fields, is_dataclass

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig as MCoreHeterogeneousTransformerConfig,
)
from megatron.core.transformer.transformer_config import (
    MLATransformerConfig as MCoreMLATransformerConfig,
)
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig


def _safe_asdict(obj, skip_keys: set[str]) -> dict:
    """Shallow asdict variant that preserves handles like process groups.

    dataclasses.asdict performs a deep copy of every leaf value, which breaks objects that
    should remain shared references (e.g., ProcessGroupCollection). This helper mirrors the
    structure of asdict but returns leaf objects as-is so they are not deep-copied.
    """
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = value if f.name in skip_keys else _safe_asdict(value, skip_keys)
        return result
    if isinstance(obj, (list, tuple)):
        return obj.__class__(_safe_asdict(v, skip_keys) for v in obj)
    if isinstance(obj, dict):
        return obj.__class__(
            (_safe_asdict(k, skip_keys), _safe_asdict(v, skip_keys)) for k, v in obj.items()
        )
    return obj


def _resolve_string_fields(config: MCoreTransformerConfig) -> None:
    """Resolve string-valued fields to their runtime types.

    Handles ``activation_func`` (e.g. ``model.activation_func=silu``) and dtype
    fields ``params_dtype`` / ``pipeline_dtype`` (e.g. ``model.params_dtype=bf16``)
    when they arrive as strings via CLI overrides.
    """
    if isinstance(config.activation_func, str):
        from megatron.bridge.utils.activation_map import str_to_callable

        config.activation_func = str_to_callable(config.activation_func)

    if isinstance(config.params_dtype, str):
        from megatron.bridge.utils.activation_map import str_to_dtype

        config.params_dtype = str_to_dtype(config.params_dtype)

    if isinstance(config.pipeline_dtype, str):
        from megatron.bridge.utils.activation_map import str_to_dtype

        config.pipeline_dtype = str_to_dtype(config.pipeline_dtype)


def _normalize_attention_backend(config: MCoreTransformerConfig) -> None:
    """Resolve an unset recipe backend to MCore's supported automatic selection.

    Explicit backend choices and Transformer Engine environment variables are
    left unchanged. MCore validates that those process-wide settings agree.
    """
    if config.attention_backend is None:
        config.attention_backend = AttnBackend.auto


_HYBRIDEP_PADDING_FIELDS = (
    "moe_hybridep_pad_uneven_dispatch_inputs",
    "moe_hybridep_pad_variable_tokens",
)


def _set_moe_expert_tensor_parallel_default(config: MCoreTransformerConfig) -> None:
    """Default expert tensor parallelism to one when expert parallelism is enabled.

    Megatron Core resolves an unset expert tensor parallel size to the tensor model
    parallel size. Bridge MoE configurations instead scale expert layers with expert
    parallelism by default, while preserving explicit expert tensor parallel values.
    """
    if config.expert_tensor_parallel_size is None and config.expert_model_parallel_size > 1:
        config.expert_tensor_parallel_size = 1


def _enable_safe_hybridep_dispatch(config: MCoreTransformerConfig, *, uses_thd: bool) -> None:
    """Enable uneven-input padding for an eager HybridEP THD recipe.

    The combined training recipe owns the tensor layout, so it requests this
    path only when its dataset produces THD packed-sequence metadata. CUDA-graph
    configs retain their explicit setting because the padding path's host scalar
    synchronization is not capture-safe; those configs require equal inputs.
    """

    def _uses_legacy_full_iteration(value: object) -> bool:
        values = value if isinstance(value, list) else [value]
        return any(
            (
                "full_iteration" in item.split(",")
                if isinstance(item, str)
                else getattr(item, "name", None) == "full_iteration"
            )
            for item in values
        )

    has_cuda_graph_impl = hasattr(config, "cuda_graph_impl")
    cuda_graph_impl = getattr(config, "cuda_graph_impl", "none")
    uses_legacy_full_iteration = not has_cuda_graph_impl and (
        _uses_legacy_full_iteration(getattr(config, "cuda_graph_modules", None))
        or _uses_legacy_full_iteration(getattr(config, "cuda_graph_scope", None))
    )
    cuda_graphs_enabled = (
        cuda_graph_impl not in (None, "none")
        or getattr(config, "enable_cuda_graph", False)
        or getattr(config, "external_cuda_graph", False)
        or uses_legacy_full_iteration
    )
    if (
        not uses_thd
        or getattr(config, "moe_token_dispatcher_type", None) != "flex"
        or getattr(config, "moe_flex_dispatcher_backend", None) != "hybridep"
        or cuda_graphs_enabled
    ):
        return

    padding_fields = tuple(
        field_name for field_name in _HYBRIDEP_PADDING_FIELDS if hasattr(config, field_name)
    )
    if not padding_fields:
        raise AttributeError(
            "Megatron Core TransformerConfig does not expose a HybridEP uneven-input padding field"
        )
    for padding_field in padding_fields:
        if not getattr(config, padding_field):
            setattr(config, padding_field, True)


def _patch_yarn_concentration_factor() -> None:
    """Patch MCore _yarn_get_concentration_factor_from_config for None handling.

    TransformerConfig declares yarn_rotary_scaling_factor as ``float | None = None``,
    but MCore uses hasattr(), which returns True for dataclass fields set to None.
    This causes a crash for non-YaRN models. Use getattr + is not None instead.

    TODO: Remove once upstream MCore merges the fix.
    """
    try:
        import megatron.core.models.common.embeddings.yarn_rotary_pos_embedding as _yarn_mod
        import megatron.core.transformer.attention as _attn_mod

        _get_factor = _yarn_mod._yarn_get_concentration_factor

        def _fixed_from_config(config):
            yarn_scaling = getattr(config, "yarn_rotary_scaling_factor", None)
            if yarn_scaling is not None:
                return _get_factor(
                    yarn_scaling,
                    getattr(config, "yarn_mscale", None),
                    getattr(config, "yarn_mscale_all_dim", None),
                )
            return 1.0

        _yarn_mod._yarn_get_concentration_factor_from_config = _fixed_from_config
        _attn_mod._yarn_get_concentration_factor_from_config = _fixed_from_config
    except ImportError:
        pass


# Applied at import so every config that declares the YaRN fields below is covered.
_patch_yarn_concentration_factor()


@dataclass
class TransformerConfig(MCoreTransformerConfig):
    """Megatron Core TransformerConfig with deferred post-init.

    This class inherits from Megatron Core's TransformerConfig but defers the
    execution of post_init() until finalize() is explicitly called. This allows
    for field modifications after construction but before computed fields are
    calculated.

    Usage:
        # Create config with deferred post-init
        config = TransformerConfig(num_layers=32, hidden_size=4096)

        # Modify fields as needed
        config.seq_length = 8192
        config.tensor_model_parallel_size = 2

        # Finalize to compute derived fields
        config.finalize()
    """

    _NO_COPY_KEYS = {"_pg_collection"}

    # Generalized tensor-parallel metadata was added after the frozen MCore dev pin.
    # Keep it on the Bridge wrapper so one configuration remains usable with both pins.
    tensor_parallel_num_weight_shards: int | None = None
    expert_tensor_parallel_num_weight_shards: int | None = None
    gtp_weight_remat_size: int = field(init=False, default=1)
    expert_gtp_weight_remat_size: int = field(init=False, default=1)

    # YaRN (Yet another RoPE extensioN) parameters, used when position_embedding_type == "yarn".
    # Megatron Core models read these from their transformer config, so builder-backed
    # configs declare them here rather than on the outer model config.
    yarn_rotary_scaling_factor: float | None = None
    yarn_original_max_position_embeddings: int | None = None
    yarn_beta_fast: float | None = None
    yarn_beta_slow: float | None = None
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_correction_range_round_to_int: bool | None = None

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        This allows for field modifications after construction without
        invalidating computed fields.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core TransformerConfig.__post_init__()
        to compute derived fields based on the current field values. It can be
        called multiple times safely.
        """
        _resolve_string_fields(self)
        _normalize_attention_backend(self)
        if self.pipeline_model_parallel_size > 1 and self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
            self.sequence_parallel = False
        _set_moe_expert_tensor_parallel_default(self)
        self._finalize_gtp_weight_shards()
        MCoreTransformerConfig.__post_init__(self)

        # In-batch packing produces variable-length packed sequences across microbatches,
        # so PP stages must communicate tensor shapes dynamically instead of using static
        # buffers.  Set *after* __post_init__ to avoid the false-positive MoE allgather
        # dispatcher check (irrelevant for non-MoE models).
        if (
            getattr(self, "_enable_in_batch_packing", False)
            and self.pipeline_model_parallel_size > 1
        ):
            self.variable_seq_lengths = True

    def _finalize_gtp_weight_shards(self) -> None:
        """Derive rematerialization sizes from optional logical weight-shard counts."""
        for field_name, parallel_size_name, output_name in (
            (
                "tensor_parallel_num_weight_shards",
                "tensor_model_parallel_size",
                "gtp_weight_remat_size",
            ),
            (
                "expert_tensor_parallel_num_weight_shards",
                "expert_tensor_parallel_size",
                "expert_gtp_weight_remat_size",
            ),
        ):
            num_weight_shards = getattr(self, field_name, None)
            if num_weight_shards is None:
                continue
            parallel_size = getattr(self, parallel_size_name) or 1
            if num_weight_shards < parallel_size or num_weight_shards % parallel_size:
                raise ValueError(
                    f"{field_name} must be divisible by and at least {parallel_size_name}"
                )
            setattr(self, output_name, num_weight_shards // parallel_size)

    def __deepcopy__(self, memo):
        """Custom deepcopy to preserve process group handles when cloning configs.

        Certain attributes (_pg_collection, etc.) should remain shared references
        rather than being wiped or re-created during deepcopy.
        TODO: This is a temporary hack. Once providers stop embedding the Transformer
        config and instead hold the MCore config as an attribute, we can remove this.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key in self._NO_COPY_KEYS:
                # Keep the same reference to avoid losing initialized process groups.
                setattr(result, key, value)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result

    def asdict(self) -> dict:
        """Return a dict view without deep-copying shared handles (e.g., process groups)."""
        return _safe_asdict(self, self._NO_COPY_KEYS)


@dataclass
class MLATransformerConfig(TransformerConfig, MCoreMLATransformerConfig):
    """Megatron Core MLATransformerConfig with deferred post-init.

    This class inherits from Megatron Core's MLATransformerConfig but defers the
    execution of post_init() until finalize() is explicitly called. This allows
    for field modifications after construction but before computed fields are
    calculated.

    Usage:
        # Create config with deferred post-init
        config = MLATransformerConfig(num_layers=32, hidden_size=4096)

        # Modify fields as needed
        config.q_lora_rank = 1536
        config.kv_lora_rank = 512

        # Finalize to compute derived fields
        config.finalize()
    """

    # Redeclared so the reverse-MRO dataclass field walk cannot revert MCore's MLA default to False.
    multi_latent_attention: bool = True

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        This allows for field modifications after construction without
        invalidating computed fields.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core MLATransformerConfig.__post_init__()
        to compute derived fields based on the current field values. It can be
        called multiple times safely.
        """
        _resolve_string_fields(self)
        _normalize_attention_backend(self)
        if self.pipeline_model_parallel_size > 1 and self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
            self.sequence_parallel = False
        _set_moe_expert_tensor_parallel_default(self)
        self._finalize_gtp_weight_shards()
        MCoreMLATransformerConfig.__post_init__(self)

        if (
            getattr(self, "_enable_in_batch_packing", False)
            and self.pipeline_model_parallel_size > 1
        ):
            self.variable_seq_lengths = True


@dataclass
class HeterogeneousTransformerConfig(TransformerConfig, MCoreHeterogeneousTransformerConfig):
    """Megatron Core HeterogeneousTransformerConfig with deferred post-init.

    This class inherits from both our lazy TransformerConfig and Megatron Core's
    HeterogeneousTransformerConfig. The MRO ensures that our lazy post-init behavior
    is preserved while maintaining all heterogeneous functionality.

    CRITICAL: The inheritance order is important for MRO:
    1. TransformerConfig (our lazy version) comes first
    2. MCoreHeterogeneousTransformerConfig comes second

    Usage:
        # Create config with deferred post-init
        config = HeterogeneousTransformerConfig(
            num_layers=32,
            hidden_size=4096,
            heterogeneous_layers_config_encoded_json=json_string
        )

        # Modify fields as needed
        config.seq_length = 8192
        config.tensor_model_parallel_size = 2

        # Finalize to compute derived fields and parse heterogeneous config
        config.finalize()
    """

    def __post_init__(self) -> None:
        """Skip MCore post_init during initial construction.

        The original post_init logic is deferred until finalize() is called.
        This allows for field modifications after construction without
        invalidating computed fields.
        """
        pass

    def finalize(self) -> None:
        """Execute the deferred MCore post-init logic.

        This method calls the original Megatron Core HeterogeneousTransformerConfig.__post_init__()
        to compute derived fields and parse the heterogeneous block configurations.
        It can be called multiple times safely.
        """
        _resolve_string_fields(self)
        _normalize_attention_backend(self)
        if self.pipeline_model_parallel_size > 1 and self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
            self.sequence_parallel = False
        _set_moe_expert_tensor_parallel_default(self)
        self._finalize_gtp_weight_shards()
        MCoreHeterogeneousTransformerConfig.__post_init__(self)
        if (
            getattr(self, "_enable_in_batch_packing", False)
            and self.pipeline_model_parallel_size > 1
        ):
            self.variable_seq_lengths = True

    def get_config_for_layer(self, layer_number: int) -> MCoreTransformerConfig:
        """Return a layer-specific TransformerConfig without deep-copying process groups."""
        # TODO: This is a temporary hack; replace once providers hold the MCore config directly.

        layer_idx = layer_number - 1  # layer number starts from 1
        if layer_idx < 0 or layer_idx >= len(self.per_block_parameters):
            raise ValueError(
                f"Invalid layer number: {layer_number}. Should be in range [1, {len(self.per_block_parameters)}]."
            )
        block_config = self.per_block_parameters[layer_idx]

        keys_to_update = {}

        # attention config updates
        if block_config.attention.num_query_groups is not None:
            assert (
                not block_config.attention.replace_with_linear and not block_config.attention.no_op
            )
            keys_to_update["num_query_groups"] = block_config.attention.num_query_groups

        # mlp config updates
        if block_config.mlp.ffn_hidden_size is not None:
            assert not block_config.mlp.replace_with_linear and not block_config.mlp.no_op
            keys_to_update["ffn_hidden_size"] = block_config.mlp.ffn_hidden_size

        transformer_config_dict = self.asdict()

        # remove keys that are not in TransformerConfig
        transformer_config_field_names = {f.name for f in fields(MCoreTransformerConfig)}
        transformer_config_dict = {
            k: v for k, v in transformer_config_dict.items() if k in transformer_config_field_names
        }

        transformer_config_dict.update(keys_to_update)

        return MCoreTransformerConfig(**transformer_config_dict)
