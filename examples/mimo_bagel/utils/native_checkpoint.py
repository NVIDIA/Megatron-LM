# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Strict adapter from a native BAGEL initialization checkpoint to MCore.

Native BAGEL stores separate Q/K/V and gate/up projections.  MCore stores those
projections in group-interleaved QKV and concatenated GLU tensors, respectively.
This module owns that representation change and verifies it by applying the
inverse mapping to every fused destination tensor after it is copied.

The adapter intentionally supports only an unsharded TP=1, PP=1, non-VP model.
Loading a full native tensor into a partial MCore model would otherwise appear
to work until a missing or incorrectly sharded parameter is used.
"""

import os
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import Mapping

import torch

_FORMAT_VERSION = 1
_HIGH_PRECISION_VALUE_ATTR = "_bagel_native_high_precision_init_val"
_ACTIVE_TARGETS: ContextVar[dict[int, str] | None] = ContextVar(
    "bagel_native_checkpoint_targets", default=None
)

_AUXILIARY_KEYS = frozenset(
    {
        "time_embedder.mlp.0.weight",
        "time_embedder.mlp.0.bias",
        "time_embedder.mlp.2.weight",
        "time_embedder.mlp.2.bias",
        "vae2llm.weight",
        "vae2llm.bias",
        "llm2vae.weight",
        "llm2vae.bias",
        "connector.fc1.weight",
        "connector.fc1.bias",
        "connector.fc2.weight",
        "connector.fc2.bias",
        "latent_pos_embed.pos_embed",
        "vit_pos_embed.pos_embed",
    }
)

_LANGUAGE_GLOBAL_KEYS = frozenset(
    {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.model.norm_moe_gen.weight",
        "language_model.lm_head.weight",
    }
)

_LANGUAGE_LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "input_layernorm_moe_gen.weight",
    "post_attention_layernorm.weight",
    "post_attention_layernorm_moe_gen.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj_moe_gen.weight",
    "self_attn.q_proj_moe_gen.bias",
    "self_attn.k_proj_moe_gen.weight",
    "self_attn.k_proj_moe_gen.bias",
    "self_attn.v_proj_moe_gen.weight",
    "self_attn.v_proj_moe_gen.bias",
    "self_attn.o_proj_moe_gen.weight",
    "self_attn.q_norm_moe_gen.weight",
    "self_attn.k_norm_moe_gen.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp_moe_gen.gate_proj.weight",
    "mlp_moe_gen.up_proj.weight",
    "mlp_moe_gen.down_proj.weight",
)


@dataclass(frozen=True)
class NativeCheckpointLoadReport:
    """Counts proving complete source consumption and target verification."""

    source_tensors_consumed: int
    target_tensors_verified: int
    fp32_main_tensors_preserved: int


class _TensorSource:
    """Track exact, single-use consumption of a safetensors file."""

    def __init__(self, checkpoint) -> None:
        self._checkpoint = checkpoint
        self.keys = frozenset(checkpoint.keys())
        self._consumed: set[str] = set()

    def take(self, key: str) -> torch.Tensor:
        if key not in self.keys:
            raise KeyError(f"Native BAGEL checkpoint is missing tensor {key!r}")
        if key in self._consumed:
            raise RuntimeError(f"Native BAGEL checkpoint tensor was consumed twice: {key!r}")
        self._consumed.add(key)
        return self._checkpoint.get_tensor(key)

    def require_exact_keys(self, expected: set[str]) -> None:
        missing = sorted(expected.difference(self.keys))
        unexpected = sorted(self.keys.difference(expected))
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise KeyError(
                "Native BAGEL checkpoint tensor set differs from the model: " + "; ".join(details)
            )

    def assert_all_consumed(self) -> None:
        unused = sorted(self.keys.difference(self._consumed))
        if unused:
            raise RuntimeError(f"Native BAGEL checkpoint tensors were not consumed: {unused}")

    @property
    def consumed_count(self) -> int:
        return len(self._consumed)


class _StateDictCheckpoint:
    """Expose a tensor state dict through the safetensors reader interface."""

    def __init__(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._state_dict = state_dict

    def keys(self):
        return self._state_dict.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._state_dict[key]


def _fuse_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> torch.Tensor:
    """Convert native [Q, K, V] tensors to MCore's group-interleaved layout."""

    if num_query_groups <= 0 or num_attention_heads % num_query_groups != 0:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} must be divisible by "
            f"num_query_groups={num_query_groups}"
        )
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if query.ndim == 0 or key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("Q, K, and V must have the same positive rank")
    if query.shape[1:] != key.shape[1:] or query.shape[1:] != value.shape[1:]:
        raise ValueError(
            "Q, K, and V trailing shapes differ: "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}"
        )

    queries_per_group = num_attention_heads // num_query_groups
    expected_q = num_attention_heads * head_dim
    expected_kv = num_query_groups * head_dim
    if query.shape[0] != expected_q or key.shape[0] != expected_kv or value.shape[0] != expected_kv:
        raise ValueError(
            "Q/K/V leading dimensions do not match the attention config: "
            f"got {query.shape[0]}/{key.shape[0]}/{value.shape[0]}, "
            f"expected {expected_q}/{expected_kv}/{expected_kv}"
        )

    trailing_shape = tuple(query.shape[1:])
    grouped_query = query.reshape(num_query_groups, queries_per_group, head_dim, *trailing_shape)
    grouped_key = key.reshape(num_query_groups, 1, head_dim, *trailing_shape)
    grouped_value = value.reshape(num_query_groups, 1, head_dim, *trailing_shape)
    return torch.cat((grouped_query, grouped_key, grouped_value), dim=1).reshape(
        (num_attention_heads + 2 * num_query_groups) * head_dim, *trailing_shape
    )


def _unfuse_qkv(
    fused: torch.Tensor, *, num_attention_heads: int, num_query_groups: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert :func:`_fuse_qkv` without relying on the source tensors."""

    if num_query_groups <= 0 or num_attention_heads % num_query_groups != 0:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} must be divisible by "
            f"num_query_groups={num_query_groups}"
        )
    if fused.ndim == 0:
        raise ValueError("Fused QKV tensor must have positive rank")
    queries_per_group = num_attention_heads // num_query_groups
    expected = (num_attention_heads + 2 * num_query_groups) * head_dim
    if fused.shape[0] != expected:
        raise ValueError(f"Fused QKV leading dimension is {fused.shape[0]}, expected {expected}")

    trailing_shape = tuple(fused.shape[1:])
    grouped = fused.reshape(num_query_groups, queries_per_group + 2, head_dim, *trailing_shape)
    query = grouped[:, :queries_per_group].reshape(num_attention_heads * head_dim, *trailing_shape)
    key = grouped[:, queries_per_group : queries_per_group + 1].reshape(
        num_query_groups * head_dim, *trailing_shape
    )
    value = grouped[:, queries_per_group + 1 :].reshape(
        num_query_groups * head_dim, *trailing_shape
    )
    return query, key, value


def _fuse_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Convert native gate/up projections to MCore's [gate; up] FC1."""

    if gate.shape != up.shape:
        raise ValueError(
            f"GLU gate/up shapes differ: gate={tuple(gate.shape)}, up={tuple(up.shape)}"
        )
    if gate.ndim == 0:
        raise ValueError("GLU gate/up tensors must have positive rank")
    return torch.cat((gate, up), dim=0)


def _unfuse_glu(fused: torch.Tensor, *, ffn_hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert :func:`_fuse_glu` into native gate and up projections."""

    if fused.ndim == 0:
        raise ValueError("Fused GLU tensor must have positive rank")
    if fused.shape[0] != 2 * ffn_hidden_size:
        raise ValueError(
            f"Fused GLU leading dimension is {fused.shape[0]}, expected {2 * ffn_hidden_size}"
        )
    return fused[:ffn_hidden_size], fused[ffn_hidden_size:]


def _reference_on_destination(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    return source.to(device=destination.device, dtype=destination.dtype)


def _assert_equal_to_source(destination: torch.Tensor, source: torch.Tensor, name: str) -> None:
    reference = _reference_on_destination(source, destination)
    if not torch.equal(destination.detach(), reference):
        raise ValueError(f"Native BAGEL remap verification failed for {name}")


def _get_high_precision_init_val(parameter: torch.nn.Parameter) -> torch.Tensor:
    return getattr(parameter, _HIGH_PRECISION_VALUE_ATTR)


def _clear_high_precision_init_val(parameter: torch.nn.Parameter) -> None:
    delattr(parameter, _HIGH_PRECISION_VALUE_ATTR)
    delattr(parameter, "get_high_precision_init_val")
    delattr(parameter, "clear_high_precision_init_val")


def _preserve_fp32_main_init(destination: torch.Tensor, source: torch.Tensor, name: str) -> None:
    """Expose the FP32 source through Megatron-FSDP's main-weight init hook."""

    if not isinstance(destination, torch.nn.Parameter) or not destination.requires_grad:
        return
    if source.dtype != torch.float32:
        return
    if hasattr(destination, "get_high_precision_init_val"):
        raise ValueError(f"MCore destination already owns a high-precision init value for {name}")

    # Keep the temporary full-precision copy on host memory. Megatron-FSDP's
    # DataParallelBuffer.set_item() accepts a CPU source for its GPU copy, and
    # this avoids retaining another full FP32 model on every GPU before wrap.
    high_precision_value = source.detach().to(device="cpu", dtype=torch.float32, copy=True)
    setattr(destination, _HIGH_PRECISION_VALUE_ATTR, high_precision_value)
    destination.get_high_precision_init_val = MethodType(  # type: ignore[attr-defined]
        _get_high_precision_init_val, destination
    )
    destination.clear_high_precision_init_val = MethodType(  # type: ignore[attr-defined]
        _clear_high_precision_init_val, destination
    )


def _copy_parameter(destination: torch.Tensor, source: torch.Tensor, name: str) -> None:
    if destination is None:
        raise ValueError(f"MCore destination is absent for {name}")
    if destination.is_meta:
        raise ValueError(f"MCore destination is still on the meta device for {name}")
    if destination.shape != source.shape:
        raise ValueError(
            f"Cannot initialize {name}: MCore shape {tuple(destination.shape)} "
            f"does not match native shape {tuple(source.shape)}"
        )
    destination.copy_(_reference_on_destination(source, destination))
    _assert_equal_to_source(destination, source, name)
    active_targets = _ACTIVE_TARGETS.get()
    if active_targets is not None:
        previous_name = active_targets.setdefault(id(destination), name)
        if previous_name != name:
            raise ValueError(
                f"MCore destination was initialized twice: {previous_name!r} and {name!r}"
            )
    _preserve_fp32_main_init(destination, source, name)


def _copy_linear(reader: _TensorSource, module, prefix: str) -> None:
    _copy_parameter(module.weight, reader.take(f"{prefix}.weight"), f"{prefix}.weight")
    bias_key = f"{prefix}.bias"
    if module.bias is None:
        if bias_key in reader.keys:
            raise ValueError(f"Native BAGEL has a bias for bias-free MCore module {prefix}")
    else:
        _copy_parameter(module.bias, reader.take(bias_key), bias_key)


def _require_bias_free(module, name: str) -> None:
    bias = getattr(module, "bias", None)
    # Transformer Engine represents ``bias=False`` with a zero-length tensor
    # on some versions.  It is neither a parameter nor a state_dict entry and
    # is not used by the forward pass, so it is semantically bias-free.
    if bias is not None and (not isinstance(bias, torch.Tensor) or bias.numel() != 0):
        raise ValueError(f"MCore module {name} has an unmapped bias")


def _pre_mlp_norm_weight(layer, branch_suffix: str) -> torch.Tensor:
    explicit_norm = getattr(layer, f"pre_mlp_layernorm{branch_suffix}", None)
    if hasattr(explicit_norm, "weight"):
        return explicit_norm.weight
    mlp = getattr(layer, f"mlp{branch_suffix}")
    if hasattr(mlp.linear_fc1, "layer_norm_weight"):
        return mlp.linear_fc1.layer_norm_weight
    branch_name = "generation" if branch_suffix else "understanding"
    raise AttributeError(f"Cannot locate the {branch_name} pre-MLP norm weight")


def _copy_attention_branch(
    reader: _TensorSource,
    mcore_attention,
    source_layer_prefix: str,
    *,
    source_suffix: str,
    destination_suffix: str,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> None:
    source_attention = f"{source_layer_prefix}.self_attn"
    qkv = getattr(mcore_attention, f"linear_qkv{destination_suffix}")
    projection = getattr(mcore_attention, f"linear_proj{destination_suffix}")
    query_norm = getattr(mcore_attention, f"q_layernorm{destination_suffix}")
    key_norm = getattr(mcore_attention, f"k_layernorm{destination_suffix}")

    def source_name(branch: str, field: str) -> str:
        return f"{source_attention}.{branch}{source_suffix}.{field}"

    qkv_prefix = f"{source_attention}.qkv{source_suffix}"
    query = reader.take(source_name("q_proj", "weight"))
    key = reader.take(source_name("k_proj", "weight"))
    value = reader.take(source_name("v_proj", "weight"))
    fused_weight = _fuse_qkv(
        query,
        key,
        value,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        head_dim=head_dim,
    )
    _copy_parameter(qkv.weight, fused_weight, f"{qkv_prefix} fused QKV weight")
    remapped_weight = _unfuse_qkv(
        qkv.weight.detach(),
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        head_dim=head_dim,
    )
    for branch_name, actual, source in zip(
        ("query", "key", "value"), remapped_weight, (query, key, value)
    ):
        _assert_equal_to_source(
            actual, source, f"{qkv_prefix} inverse-remapped {branch_name} weight"
        )

    if qkv.bias is None:
        raise ValueError(f"MCore QKV module is bias-free for {qkv_prefix}")
    query_bias = reader.take(source_name("q_proj", "bias"))
    key_bias = reader.take(source_name("k_proj", "bias"))
    value_bias = reader.take(source_name("v_proj", "bias"))
    fused_bias = _fuse_qkv(
        query_bias,
        key_bias,
        value_bias,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        head_dim=head_dim,
    )
    _copy_parameter(qkv.bias, fused_bias, f"{qkv_prefix} fused QKV bias")
    remapped_bias = _unfuse_qkv(
        qkv.bias.detach(),
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        head_dim=head_dim,
    )
    for branch_name, actual, source in zip(
        ("query", "key", "value"), remapped_bias, (query_bias, key_bias, value_bias)
    ):
        _assert_equal_to_source(actual, source, f"{qkv_prefix} inverse-remapped {branch_name} bias")

    _copy_parameter(
        projection.weight,
        reader.take(source_name("o_proj", "weight")),
        source_name("o_proj", "weight"),
    )
    _require_bias_free(projection, source_name("o_proj", "weight"))
    _copy_parameter(
        query_norm.weight,
        reader.take(source_name("q_norm", "weight")),
        source_name("q_norm", "weight"),
    )
    _copy_parameter(
        key_norm.weight,
        reader.take(source_name("k_norm", "weight")),
        source_name("k_norm", "weight"),
    )


def _copy_mlp_branch(reader: _TensorSource, mcore_mlp, source_prefix: str) -> None:
    gate = reader.take(f"{source_prefix}.gate_proj.weight")
    up = reader.take(f"{source_prefix}.up_proj.weight")
    fused = _fuse_glu(gate, up)
    _copy_parameter(mcore_mlp.linear_fc1.weight, fused, f"{source_prefix} fused GLU weight")
    actual_gate, actual_up = _unfuse_glu(
        mcore_mlp.linear_fc1.weight.detach(), ffn_hidden_size=gate.shape[0]
    )
    _assert_equal_to_source(actual_gate, gate, f"{source_prefix} inverse-remapped gate weight")
    _assert_equal_to_source(actual_up, up, f"{source_prefix} inverse-remapped up weight")
    _require_bias_free(mcore_mlp.linear_fc1, f"{source_prefix}.gate/up projection")
    _copy_parameter(
        mcore_mlp.linear_fc2.weight,
        reader.take(f"{source_prefix}.down_proj.weight"),
        f"{source_prefix}.down_proj.weight",
    )
    _require_bias_free(mcore_mlp.linear_fc2, f"{source_prefix}.down_proj")


def _validate_unsharded_model(language_model) -> None:
    for stage_flag in ("pre_process", "post_process"):
        if hasattr(language_model, stage_flag) and not getattr(language_model, stage_flag):
            raise NotImplementedError(
                f"Native BAGEL initialization requires a complete model with {stage_flag}=True"
            )
    if getattr(language_model, "vp_stage", None) is not None:
        raise NotImplementedError("Native BAGEL initialization does not support VP model chunks")
    if not torch.distributed.is_initialized():
        return
    pg_collection = getattr(language_model, "pg_collection", None)
    if pg_collection is None:
        raise ValueError("MCore language model has no ProcessGroupCollection")
    for name in ("tp", "pp"):
        group = getattr(pg_collection, name, None)
        size = torch.distributed.get_world_size(group) if group is not None else 1
        if size != 1:
            raise NotImplementedError(
                f"Native BAGEL initialization requires TP=1 and PP=1; {name.upper()}={size}"
            )


def _language_keys(num_layers: int) -> set[str]:
    keys = set(_LANGUAGE_GLOBAL_KEYS)
    for layer_index in range(num_layers):
        prefix = f"language_model.model.layers.{layer_index}"
        keys.update(f"{prefix}.{suffix}" for suffix in _LANGUAGE_LAYER_SUFFIXES)
    return keys


def _metadata_int(metadata: Mapping[str, str], key: str) -> int:
    value = metadata.get(key)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Native BAGEL checkpoint metadata {key}={value!r} is not an integer"
        ) from error


def _validate_metadata(
    metadata: Mapping[str, str] | None, *, expected_model_seed: int, expected_world_size: int
) -> None:
    if metadata is None:
        raise ValueError("Native BAGEL checkpoint has no safetensors metadata")
    actual_format = _metadata_int(metadata, "format_version")
    if actual_format != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported native BAGEL checkpoint format_version={actual_format}; "
            f"expected {_FORMAT_VERSION}"
        )
    for key, expected in (("model_seed", expected_model_seed), ("world_size", expected_world_size)):
        actual = _metadata_int(metadata, key)
        if actual != expected:
            raise ValueError(
                f"Native BAGEL checkpoint metadata {key}={actual}; expected {expected}"
            )


def _initialize_language(reader: _TensorSource, language_model, llm_config) -> None:
    layers = language_model.decoder.layers
    configured_layers = int(llm_config.num_hidden_layers)
    if len(layers) != configured_layers:
        raise ValueError(
            f"MCore has {len(layers)} local language layers but native config has "
            f"{configured_layers}; partial PP/VP models are unsupported"
        )

    num_attention_heads = int(llm_config.num_attention_heads)
    num_query_groups = int(llm_config.num_key_value_heads)
    configured_head_dim = getattr(llm_config, "head_dim", None)
    head_dim = (
        int(configured_head_dim)
        if configured_head_dim is not None
        else int(llm_config.hidden_size) // num_attention_heads
    )

    _copy_parameter(
        language_model.embedding.word_embeddings.weight,
        reader.take("language_model.model.embed_tokens.weight"),
        "language_model.model.embed_tokens.weight",
    )
    for layer_index, mcore_layer in enumerate(layers):
        source_layer = f"language_model.model.layers.{layer_index}"
        _copy_parameter(
            mcore_layer.input_layernorm.weight,
            reader.take(f"{source_layer}.input_layernorm.weight"),
            f"{source_layer}.input_layernorm.weight",
        )
        _copy_parameter(
            mcore_layer.input_layernorm_gen.weight,
            reader.take(f"{source_layer}.input_layernorm_moe_gen.weight"),
            f"{source_layer}.input_layernorm_moe_gen.weight",
        )
        _copy_parameter(
            _pre_mlp_norm_weight(mcore_layer, ""),
            reader.take(f"{source_layer}.post_attention_layernorm.weight"),
            f"{source_layer}.post_attention_layernorm.weight",
        )
        _copy_parameter(
            _pre_mlp_norm_weight(mcore_layer, "_gen"),
            reader.take(f"{source_layer}.post_attention_layernorm_moe_gen.weight"),
            f"{source_layer}.post_attention_layernorm_moe_gen.weight",
        )
        _copy_attention_branch(
            reader,
            mcore_layer.self_attention,
            source_layer,
            source_suffix="",
            destination_suffix="",
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            head_dim=head_dim,
        )
        _copy_attention_branch(
            reader,
            mcore_layer.self_attention,
            source_layer,
            source_suffix="_moe_gen",
            destination_suffix="_gen",
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            head_dim=head_dim,
        )
        _copy_mlp_branch(reader, mcore_layer.mlp, f"{source_layer}.mlp")
        _copy_mlp_branch(reader, mcore_layer.mlp_gen, f"{source_layer}.mlp_moe_gen")

    _copy_parameter(
        language_model.decoder.final_layernorm.weight,
        reader.take("language_model.model.norm.weight"),
        "language_model.model.norm.weight",
    )
    _copy_parameter(
        language_model.decoder.final_layernorm_gen.weight,
        reader.take("language_model.model.norm_moe_gen.weight"),
        "language_model.model.norm_moe_gen.weight",
    )
    _copy_parameter(
        language_model.output_layer.weight,
        reader.take("language_model.lm_head.weight"),
        "language_model.lm_head.weight",
    )
    _require_bias_free(language_model.output_layer, "language_model.lm_head")


def _initialize_vision(reader: _TensorSource, mimo_model) -> None:
    vision_encoder = mimo_model.modality_submodules["images"].encoders["vision_encoder"]
    target_state = vision_encoder.encoder.state_dict(keep_vars=True)
    for target_key, destination in target_state.items():
        source_key = f"vit_model.{target_key}"
        _copy_parameter(destination, reader.take(source_key), source_key)


def _initialize_auxiliary(reader: _TensorSource, mimo_model) -> None:
    diffusion = mimo_model.modality_submodules["diffusion"]
    images = mimo_model.modality_submodules["images"]
    timestep = diffusion.encoders["timestep"].mlp
    vision_mlp = images.input_projections[0].encoder

    _copy_linear(reader, timestep[0], "time_embedder.mlp.0")
    _copy_linear(reader, timestep[2], "time_embedder.mlp.2")
    _copy_linear(reader, diffusion.input_projections[0], "vae2llm")
    _copy_linear(reader, diffusion.output_projections[0], "llm2vae")
    _copy_linear(reader, vision_mlp.linear_fc1, "connector.fc1")
    _copy_linear(reader, vision_mlp.linear_fc2, "connector.fc2")
    _copy_parameter(
        diffusion.encoders["latent_position_ids"].pos_embed,
        reader.take("latent_pos_embed.pos_embed"),
        "latent_pos_embed.pos_embed",
    )
    _copy_parameter(
        images.encoders["vision_encoder"].vit_pos_embed.pos_embed,
        reader.take("vit_pos_embed.pos_embed"),
        "vit_pos_embed.pos_embed",
    )


def _assert_exact_target_coverage(
    mimo_model: torch.nn.Module, initialized_targets: Mapping[int, str]
) -> None:
    """Require every semantic MCore parameter and position buffer exactly once."""

    # Some focused unit tests use a structural SimpleNamespace. Production
    # callers are nn.Module instances and receive the full identity check.
    if not isinstance(mimo_model, torch.nn.Module):
        return

    expected_targets = {id(parameter): name for name, parameter in mimo_model.named_parameters()}
    latent_position = (
        mimo_model.modality_submodules["diffusion"].encoders["latent_position_ids"].pos_embed
    )
    expected_targets.setdefault(
        id(latent_position), "modality_submodules.diffusion.encoders.latent_position_ids.pos_embed"
    )

    missing = sorted(
        name for target_id, name in expected_targets.items() if target_id not in initialized_targets
    )
    unexpected = sorted(
        name for target_id, name in initialized_targets.items() if target_id not in expected_targets
    )
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(
            "Native BAGEL target coverage differs from the MCore model: " + "; ".join(details)
        )

    missing_main_values = sorted(
        name
        for name, parameter in mimo_model.named_parameters()
        if parameter.requires_grad and not hasattr(parameter, _HIGH_PRECISION_VALUE_ATTR)
    )
    if missing_main_values:
        raise ValueError(
            "Native BAGEL did not preserve FP32 main-weight initialization for trainable "
            f"targets: {missing_main_values}"
        )


def initialize_bagel_from_native_checkpoint(
    mimo_model: torch.nn.Module,
    checkpoint_path: str,
    *,
    expected_model_seed: int,
    expected_world_size: int,
    llm_config,
) -> NativeCheckpointLoadReport:
    """Initialize an unsharded MCore BAGEL model from native BAGEL safetensors.

    The checkpoint must contain the complete native ``Bagel.state_dict()`` and
    safetensors metadata ``format_version=1``, ``model_seed``, and
    ``world_size``.  Every tensor must map exactly once; missing and additional
    tensors are rejected before any parameter is modified.
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Native BAGEL checkpoint not found: {checkpoint_path}")

    from safetensors import safe_open

    language_model = mimo_model.language_model
    _validate_unsharded_model(language_model)

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        _validate_metadata(
            checkpoint.metadata(),
            expected_model_seed=expected_model_seed,
            expected_world_size=expected_world_size,
        )
        reader = _TensorSource(checkpoint)
        vision_encoder = mimo_model.modality_submodules["images"].encoders["vision_encoder"]
        vision_keys = {f"vit_model.{key}" for key in vision_encoder.encoder.state_dict()}
        expected_keys = (
            _language_keys(int(llm_config.num_hidden_layers)) | vision_keys | set(_AUXILIARY_KEYS)
        )
        reader.require_exact_keys(expected_keys)

        initialized_targets: dict[int, str] = {}
        target_token = _ACTIVE_TARGETS.set(initialized_targets)
        try:
            with torch.no_grad():
                _initialize_language(reader, language_model, llm_config)
                _initialize_vision(reader, mimo_model)
                _initialize_auxiliary(reader, mimo_model)
        finally:
            _ACTIVE_TARGETS.reset(target_token)
        reader.assert_all_consumed()
        _assert_exact_target_coverage(mimo_model, initialized_targets)
        fp32_main_tensors_preserved = (
            sum(
                hasattr(parameter, _HIGH_PRECISION_VALUE_ATTR)
                for parameter in mimo_model.parameters()
            )
            if isinstance(mimo_model, torch.nn.Module)
            else 0
        )
        return NativeCheckpointLoadReport(
            source_tensors_consumed=reader.consumed_count,
            target_tensors_verified=len(initialized_targets),
            fp32_main_tensors_preserved=fp32_main_tensors_preserved,
        )


def initialize_bagel_auxiliary_from_native(
    mimo_model,
    checkpoint_path: str,
    *,
    expected_model_seed: int,
    expected_world_size: int,
) -> NativeCheckpointLoadReport:
    """Initialize BAGEL connector, diffusion, and position modules from a native export.

    Auxiliary-only exports use a small ``torch.save`` container with the same
    format/seed/world-size metadata as the full native safetensors checkpoint.
    The tensor set is validated before any destination is modified.
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Native BAGEL auxiliary checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Native BAGEL auxiliary checkpoint must contain a mapping")
    _validate_metadata(
        payload,
        expected_model_seed=expected_model_seed,
        expected_world_size=expected_world_size,
    )

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Native BAGEL auxiliary checkpoint has no state_dict mapping")
    reader = _TensorSource(_StateDictCheckpoint(state_dict))
    reader.require_exact_keys(set(_AUXILIARY_KEYS))

    initialized_targets: dict[int, str] = {}
    target_token = _ACTIVE_TARGETS.set(initialized_targets)
    try:
        with torch.no_grad():
            _initialize_auxiliary(reader, mimo_model)
    finally:
        _ACTIVE_TARGETS.reset(target_token)
    reader.assert_all_consumed()

    fp32_main_tensors_preserved = (
        sum(
            hasattr(parameter, _HIGH_PRECISION_VALUE_ATTR)
            for parameter in mimo_model.parameters()
        )
        if isinstance(mimo_model, torch.nn.Module)
        else 0
    )
    return NativeCheckpointLoadReport(
        source_tensors_consumed=reader.consumed_count,
        target_tensors_verified=len(initialized_targets),
        fp32_main_tensors_preserved=fp32_main_tensors_preserved,
    )
