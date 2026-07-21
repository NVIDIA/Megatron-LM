# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from examples.mimo_bagel.utils.native_checkpoint import (
    _copy_parameter,
    _fuse_glu,
    _fuse_qkv,
    _require_bias_free,
    _unfuse_glu,
    _unfuse_qkv,
    initialize_bagel_from_native_checkpoint,
)
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked

_HIDDEN_SIZE = 8
_NUM_HEADS = 4
_NUM_QUERY_GROUPS = 2
_HEAD_DIM = 2
_FFN_HIDDEN_SIZE = 6
_VOCAB_SIZE = 11


class _WeightOnlyNorm(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(size))


class _TinyVisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)
        self.norm = torch.nn.LayerNorm(4)


def _linear(in_features, out_features, *, bias=True):
    return torch.nn.Linear(in_features, out_features, bias=bias)


def _make_attention():
    fused_size = (_NUM_HEADS + 2 * _NUM_QUERY_GROUPS) * _HEAD_DIM

    def branch():
        return {
            "qkv": _linear(_HIDDEN_SIZE, fused_size),
            "projection": _linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False),
            "query_norm": _WeightOnlyNorm(_HEAD_DIM),
            "key_norm": _WeightOnlyNorm(_HEAD_DIM),
        }

    understanding = branch()
    generation = branch()
    return SimpleNamespace(
        linear_qkv=understanding["qkv"],
        linear_proj=understanding["projection"],
        q_layernorm=understanding["query_norm"],
        k_layernorm=understanding["key_norm"],
        linear_qkv_gen=generation["qkv"],
        linear_proj_gen=generation["projection"],
        q_layernorm_gen=generation["query_norm"],
        k_layernorm_gen=generation["key_norm"],
    )


def _make_mlp(*, fused_pre_mlp_norm=True):
    linear_fc1 = _linear(_HIDDEN_SIZE, 2 * _FFN_HIDDEN_SIZE, bias=False)
    if fused_pre_mlp_norm:
        # Exercise the real TE fused-layernorm location by default.
        linear_fc1.layer_norm_weight = torch.nn.Parameter(torch.zeros(_HIDDEN_SIZE))
    return SimpleNamespace(
        linear_fc1=linear_fc1, linear_fc2=_linear(_FFN_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False)
    )


def _make_layer(*, explicit_pre_mlp_norm=False):
    pre_mlp_layernorm = (
        _WeightOnlyNorm(_HIDDEN_SIZE) if explicit_pre_mlp_norm else SimpleNamespace()
    )
    pre_mlp_layernorm_gen = (
        _WeightOnlyNorm(_HIDDEN_SIZE) if explicit_pre_mlp_norm else SimpleNamespace()
    )
    return SimpleNamespace(
        input_layernorm=_WeightOnlyNorm(_HIDDEN_SIZE),
        input_layernorm_gen=_WeightOnlyNorm(_HIDDEN_SIZE),
        pre_mlp_layernorm=pre_mlp_layernorm,
        pre_mlp_layernorm_gen=pre_mlp_layernorm_gen,
        self_attention=_make_attention(),
        mlp=_make_mlp(fused_pre_mlp_norm=not explicit_pre_mlp_norm),
        mlp_gen=_make_mlp(fused_pre_mlp_norm=not explicit_pre_mlp_norm),
    )


def _make_model(num_layers=2, *, explicit_pre_mlp_norm=False):
    language_model = SimpleNamespace(
        embedding=SimpleNamespace(word_embeddings=torch.nn.Embedding(_VOCAB_SIZE, _HIDDEN_SIZE)),
        decoder=SimpleNamespace(
            layers=[
                _make_layer(explicit_pre_mlp_norm=explicit_pre_mlp_norm)
                for _ in range(num_layers)
            ],
            final_layernorm=_WeightOnlyNorm(_HIDDEN_SIZE),
            final_layernorm_gen=_WeightOnlyNorm(_HIDDEN_SIZE),
        ),
        output_layer=_linear(_HIDDEN_SIZE, _VOCAB_SIZE, bias=False),
        pg_collection=SimpleNamespace(tp=None, pp=None),
    )

    vision_encoder = SimpleNamespace(
        encoder=_TinyVisionEncoder(),
        vit_pos_embed=SimpleNamespace(pos_embed=torch.nn.Parameter(torch.zeros(5, _HIDDEN_SIZE))),
    )
    images = SimpleNamespace(
        encoders={"vision_encoder": vision_encoder},
        input_projections=[
            SimpleNamespace(
                encoder=SimpleNamespace(
                    linear_fc1=_linear(6, _HIDDEN_SIZE),
                    linear_fc2=_linear(_HIDDEN_SIZE, _HIDDEN_SIZE),
                )
            )
        ],
    )
    diffusion = SimpleNamespace(
        encoders={
            "timestep": SimpleNamespace(
                mlp=torch.nn.Sequential(
                    _linear(3, _HIDDEN_SIZE), torch.nn.SiLU(), _linear(_HIDDEN_SIZE, _HIDDEN_SIZE)
                )
            ),
            "latent_position_ids": SimpleNamespace(
                pos_embed=torch.nn.Parameter(torch.zeros(4, _HIDDEN_SIZE))
            ),
        },
        input_projections=[_linear(4, _HIDDEN_SIZE)],
        output_projections=[_linear(_HIDDEN_SIZE, 4)],
    )
    return SimpleNamespace(
        language_model=language_model,
        modality_submodules={"images": images, "diffusion": diffusion},
    )


def _make_source_state(model, num_layers=2):
    state = {}
    value_offset = 1

    def tensor(key, shape):
        nonlocal value_offset
        numel = int(torch.tensor(shape).prod().item())
        state[key] = torch.arange(value_offset, value_offset + numel, dtype=torch.float32).reshape(
            shape
        )
        value_offset += numel

    tensor("language_model.model.embed_tokens.weight", (_VOCAB_SIZE, _HIDDEN_SIZE))
    tensor("language_model.lm_head.weight", (_VOCAB_SIZE, _HIDDEN_SIZE))
    for layer_index in range(num_layers):
        prefix = f"language_model.model.layers.{layer_index}"
        tensor(f"{prefix}.input_layernorm.weight", (_HIDDEN_SIZE,))
        tensor(f"{prefix}.input_layernorm_moe_gen.weight", (_HIDDEN_SIZE,))
        tensor(f"{prefix}.post_attention_layernorm.weight", (_HIDDEN_SIZE,))
        tensor(f"{prefix}.post_attention_layernorm_moe_gen.weight", (_HIDDEN_SIZE,))
        for suffix in ("", "_moe_gen"):
            attention = f"{prefix}.self_attn"
            tensor(f"{attention}.q_proj{suffix}.weight", (_NUM_HEADS * _HEAD_DIM, _HIDDEN_SIZE))
            tensor(f"{attention}.q_proj{suffix}.bias", (_NUM_HEADS * _HEAD_DIM,))
            for projection in ("k_proj", "v_proj"):
                tensor(
                    f"{attention}.{projection}{suffix}.weight",
                    (_NUM_QUERY_GROUPS * _HEAD_DIM, _HIDDEN_SIZE),
                )
                tensor(f"{attention}.{projection}{suffix}.bias", (_NUM_QUERY_GROUPS * _HEAD_DIM,))
            tensor(f"{attention}.o_proj{suffix}.weight", (_HIDDEN_SIZE, _HIDDEN_SIZE))
            tensor(f"{attention}.q_norm{suffix}.weight", (_HEAD_DIM,))
            tensor(f"{attention}.k_norm{suffix}.weight", (_HEAD_DIM,))
        for mlp_name in ("mlp", "mlp_moe_gen"):
            mlp = f"{prefix}.{mlp_name}"
            tensor(f"{mlp}.gate_proj.weight", (_FFN_HIDDEN_SIZE, _HIDDEN_SIZE))
            tensor(f"{mlp}.up_proj.weight", (_FFN_HIDDEN_SIZE, _HIDDEN_SIZE))
            tensor(f"{mlp}.down_proj.weight", (_HIDDEN_SIZE, _FFN_HIDDEN_SIZE))
    tensor("language_model.model.norm.weight", (_HIDDEN_SIZE,))
    tensor("language_model.model.norm_moe_gen.weight", (_HIDDEN_SIZE,))

    vision_encoder = model.modality_submodules["images"].encoders["vision_encoder"].encoder
    for key, destination in vision_encoder.state_dict().items():
        tensor(f"vit_model.{key}", tuple(destination.shape))

    auxiliary_shapes = {
        "time_embedder.mlp.0.weight": (_HIDDEN_SIZE, 3),
        "time_embedder.mlp.0.bias": (_HIDDEN_SIZE,),
        "time_embedder.mlp.2.weight": (_HIDDEN_SIZE, _HIDDEN_SIZE),
        "time_embedder.mlp.2.bias": (_HIDDEN_SIZE,),
        "vae2llm.weight": (_HIDDEN_SIZE, 4),
        "vae2llm.bias": (_HIDDEN_SIZE,),
        "llm2vae.weight": (4, _HIDDEN_SIZE),
        "llm2vae.bias": (4,),
        "connector.fc1.weight": (_HIDDEN_SIZE, 6),
        "connector.fc1.bias": (_HIDDEN_SIZE,),
        "connector.fc2.weight": (_HIDDEN_SIZE, _HIDDEN_SIZE),
        "connector.fc2.bias": (_HIDDEN_SIZE,),
        "latent_pos_embed.pos_embed": (4, _HIDDEN_SIZE),
        "vit_pos_embed.pos_embed": (5, _HIDDEN_SIZE),
    }
    for key, shape in auxiliary_shapes.items():
        tensor(key, shape)
    return state


def _save_checkpoint(path, state, **metadata_overrides):
    metadata = {"format_version": "1", "model_seed": "35168", "world_size": "8"}
    metadata.update({key: str(value) for key, value in metadata_overrides.items()})
    save_file(state, str(path), metadata=metadata)


def _config(num_layers=2):
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_QUERY_GROUPS,
        head_dim=_HEAD_DIM,
    )


@pytest.mark.parametrize("trailing_shape", [(), (7,)])
def test_qkv_group_permutation_is_bit_exact_and_invertible(trailing_shape):
    q_shape = (_NUM_HEADS * _HEAD_DIM, *trailing_shape)
    kv_shape = (_NUM_QUERY_GROUPS * _HEAD_DIM, *trailing_shape)
    query = torch.arange(torch.tensor(q_shape).prod()).reshape(q_shape)
    key = 1000 + torch.arange(torch.tensor(kv_shape).prod()).reshape(kv_shape)
    value = 2000 + torch.arange(torch.tensor(kv_shape).prod()).reshape(kv_shape)

    fused = _fuse_qkv(
        query,
        key,
        value,
        num_attention_heads=_NUM_HEADS,
        num_query_groups=_NUM_QUERY_GROUPS,
        head_dim=_HEAD_DIM,
    )
    actual = _unfuse_qkv(
        fused,
        num_attention_heads=_NUM_HEADS,
        num_query_groups=_NUM_QUERY_GROUPS,
        head_dim=_HEAD_DIM,
    )

    assert all(torch.equal(left, right) for left, right in zip(actual, (query, key, value)))
    # First group has two Q heads followed by one K and one V head.
    grouped = fused.reshape(_NUM_QUERY_GROUPS, 4, _HEAD_DIM, *trailing_shape)
    assert torch.equal(grouped[0, 0], query.reshape(2, 2, _HEAD_DIM, *trailing_shape)[0, 0])
    assert torch.equal(grouped[0, 2], key.reshape(2, _HEAD_DIM, *trailing_shape)[0])
    assert torch.equal(grouped[0, 3], value.reshape(2, _HEAD_DIM, *trailing_shape)[0])


def test_glu_fusion_is_gate_then_up_and_invertible():
    gate = torch.arange(12).reshape(3, 4)
    up = torch.arange(100, 112).reshape(3, 4)

    fused = _fuse_glu(gate, up)
    actual_gate, actual_up = _unfuse_glu(fused, ffn_hidden_size=3)

    assert torch.equal(fused[:3], gate)
    assert torch.equal(fused[3:], up)
    assert torch.equal(actual_gate, gate)
    assert torch.equal(actual_up, up)


def test_transformer_engine_zero_length_bias_sentinel_is_bias_free():
    module = SimpleNamespace(bias=torch.empty(0))

    _require_bias_free(module, "te_linear")


def test_nonempty_unmapped_bias_is_rejected():
    module = SimpleNamespace(bias=torch.zeros(1))

    with pytest.raises(ValueError, match="unmapped bias"):
        _require_bias_free(module, "linear")


@pytest.mark.parametrize("destination_dtype", [torch.bfloat16, torch.float32])
def test_fp32_source_is_preserved_for_fsdp_main_weight_initialization(destination_dtype):
    source = torch.tensor([1.0001, -2.0003], dtype=torch.float32)
    destination = torch.nn.Parameter(torch.zeros(2, dtype=destination_dtype))

    with torch.no_grad():
        _copy_parameter(destination, source, "weight")

    assert torch.equal(destination, source.to(destination_dtype))
    assert torch.equal(destination.get_high_precision_init_val(), source)
    destination.clear_high_precision_init_val()
    assert not hasattr(destination, "get_high_precision_init_val")


def test_fp32_main_hook_survives_float16_module_conversion():
    module = torch.nn.Linear(2, 1, bias=False)
    source = torch.tensor([[1.0001, -2.0003]], dtype=torch.float32)
    original_parameter = module.weight

    with torch.no_grad():
        _copy_parameter(module.weight, source, "weight")
    convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)

    assert module.weight is original_parameter
    assert module.weight.dtype == torch.bfloat16
    assert torch.equal(module.weight.get_high_precision_init_val(), source)
    module.weight.clear_high_precision_init_val()


def test_full_native_checkpoint_maps_language_vision_and_auxiliary(tmp_path):
    model = _make_model()
    source = _make_source_state(model)
    checkpoint = tmp_path / "native-init.safetensors"
    _save_checkpoint(checkpoint, source)

    initialize_bagel_from_native_checkpoint(
        model,
        str(checkpoint),
        expected_model_seed=35168,
        expected_world_size=8,
        llm_config=_config(),
    )

    assert torch.equal(
        model.language_model.embedding.word_embeddings.weight,
        source["language_model.model.embed_tokens.weight"],
    )
    for layer_index, layer in enumerate(model.language_model.decoder.layers):
        prefix = f"language_model.model.layers.{layer_index}"
        for source_suffix, destination_suffix in (("", ""), ("_moe_gen", "_gen")):
            attention = layer.self_attention
            qkv = getattr(attention, f"linear_qkv{destination_suffix}")
            actual_qkv = _unfuse_qkv(
                qkv.weight,
                num_attention_heads=_NUM_HEADS,
                num_query_groups=_NUM_QUERY_GROUPS,
                head_dim=_HEAD_DIM,
            )
            expected_qkv = tuple(
                source[f"{prefix}.self_attn.{projection}{source_suffix}.weight"]
                for projection in ("q_proj", "k_proj", "v_proj")
            )
            assert all(
                torch.equal(actual, expected) for actual, expected in zip(actual_qkv, expected_qkv)
            )
            mlp_name = "mlp" if not destination_suffix else "mlp_moe_gen"
            mlp = getattr(layer, f"mlp{destination_suffix}")
            gate, up = _unfuse_glu(mlp.linear_fc1.weight, ffn_hidden_size=_FFN_HIDDEN_SIZE)
            assert torch.equal(gate, source[f"{prefix}.{mlp_name}.gate_proj.weight"])
            assert torch.equal(up, source[f"{prefix}.{mlp_name}.up_proj.weight"])

    vision = model.modality_submodules["images"].encoders["vision_encoder"]
    for key, value in vision.encoder.state_dict().items():
        assert torch.equal(value, source[f"vit_model.{key}"])
    assert torch.equal(vision.vit_pos_embed.pos_embed, source["vit_pos_embed.pos_embed"])
    diffusion = model.modality_submodules["diffusion"]
    assert torch.equal(
        diffusion.encoders["latent_position_ids"].pos_embed, source["latent_pos_embed.pos_embed"]
    )
    assert torch.equal(diffusion.output_projections[0].weight, source["llm2vae.weight"])


def test_full_native_checkpoint_maps_local_explicit_pre_mlp_norms(tmp_path):
    """The strict loader supports local norms as well as TE's fused FC1 norm."""

    model = _make_model(explicit_pre_mlp_norm=True)
    source = _make_source_state(model)
    checkpoint = tmp_path / "native-local-init.safetensors"
    _save_checkpoint(checkpoint, source)

    initialize_bagel_from_native_checkpoint(
        model,
        str(checkpoint),
        expected_model_seed=35168,
        expected_world_size=8,
        llm_config=_config(),
    )

    for layer_index, layer in enumerate(model.language_model.decoder.layers):
        prefix = f"language_model.model.layers.{layer_index}"
        assert torch.equal(
            layer.pre_mlp_layernorm.weight,
            source[f"{prefix}.post_attention_layernorm.weight"],
        )
        assert torch.equal(
            layer.pre_mlp_layernorm_gen.weight,
            source[f"{prefix}.post_attention_layernorm_moe_gen.weight"],
        )


def test_checkpoint_metadata_mismatch_fails_before_modifying_model(tmp_path):
    model = _make_model()
    source = _make_source_state(model)
    checkpoint = tmp_path / "wrong-seed.safetensors"
    _save_checkpoint(checkpoint, source, model_seed=999)
    original_embedding = model.language_model.embedding.word_embeddings.weight.detach().clone()

    with pytest.raises(ValueError, match="model_seed=999; expected 35168"):
        initialize_bagel_from_native_checkpoint(
            model,
            str(checkpoint),
            expected_model_seed=35168,
            expected_world_size=8,
            llm_config=_config(),
        )

    assert torch.equal(model.language_model.embedding.word_embeddings.weight, original_embedding)


def test_unexpected_source_tensor_fails_before_modifying_model(tmp_path):
    model = _make_model()
    source = _make_source_state(model)
    source["unexpected.weight"] = torch.ones(1)
    checkpoint = tmp_path / "unexpected.safetensors"
    _save_checkpoint(checkpoint, source)
    original_embedding = model.language_model.embedding.word_embeddings.weight.detach().clone()

    with pytest.raises(KeyError, match="unexpected.weight"):
        initialize_bagel_from_native_checkpoint(
            model,
            str(checkpoint),
            expected_model_seed=35168,
            expected_world_size=8,
            llm_config=_config(),
        )

    assert torch.equal(model.language_model.embedding.word_embeddings.weight, original_embedding)
