# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from types import SimpleNamespace

import pytest
import torch

from examples.inference.advanced import vlm_dynamic_inference as vlm
from examples.multimodal.mimo_checkpoint_model import (
    MimoCheckpointLLaVAModel,
    _unfused_fast_gelu,
    vision_config,
    vision_projection_config,
)
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.transformer.transformer_config import TransformerConfig

_MODALITY = 'modality_submodules.images.module.module.'
_LANGUAGE_KEY = 'language_model.module.module.embedding.word_embeddings.weight'
_ENCODER_KEY = f'{_MODALITY}encoders.vision_encoder.patch_embed.weight'
_PROJECTION_KEY = f'{_MODALITY}input_projections.0.encoder.linear_fc1.weight'
_PREFIX_MAP = {
    'language_model.': 'language_model.module.module.',
    'vision_model.': f'{_MODALITY}encoders.vision_encoder.',
    'vision_projection.': f'{_MODALITY}input_projections.0.',
}


@pytest.fixture
def checkpoint_keys(monkeypatch):
    """Serve the given keys as the tensor keys of the loaded checkpoint."""
    keys = [_LANGUAGE_KEY]
    monkeypatch.setattr(vlm, '_checkpoint_tensor_keys', lambda args: keys)
    return keys


def test_checkpoint_detection(checkpoint_keys):
    args = SimpleNamespace(moe_router_enable_expert_bias=False)
    assert vlm._mimo_checkpoint_prefix_map(args) is None
    vlm._enable_checkpoint_expert_bias(args)
    assert not args.moe_router_enable_expert_bias

    checkpoint_keys += [
        _ENCODER_KEY,
        _PROJECTION_KEY,
        'language_model.module.module.decoder.layers.1.mlp.router.expert_bias',
    ]
    assert vlm._mimo_checkpoint_prefix_map(args) == _PREFIX_MAP
    # Trained biases in the checkpoint win over its saved args.
    vlm._enable_checkpoint_expert_bias(args)
    assert args.moe_router_enable_expert_bias

    checkpoint_keys += [f'{_MODALITY}encoders.other_encoder.patch_embed.weight']
    with pytest.raises(ValueError, match='one vision encoder'):
        vlm._mimo_checkpoint_prefix_map(args)


def test_sharded_state_dict_uses_checkpoint_keys(monkeypatch):
    names = ('language_model.embedding.weight', 'vision_model.patch_embed.weight')
    monkeypatch.setattr(
        LLaVAModel,
        'sharded_state_dict',
        lambda self, prefix='', *_: {
            name: ShardedTensor.from_rank_offsets(name, torch.zeros(1)) for name in names
        },
    )
    # Skip the full model build; only the key mapping is under test.
    model = MimoCheckpointLLaVAModel.__new__(MimoCheckpointLLaVAModel)
    object.__setattr__(model, 'checkpoint_prefix_map', _PREFIX_MAP)

    keys = {value.key for value in model.sharded_state_dict().values()}
    assert keys == {'language_model.module.module.embedding.weight', _ENCODER_KEY}
    assert model.sharded_checkpoint_keys == keys  # Recorded for the post-load check.


def test_resolve_mimo_vision_args(tmp_path):
    with pytest.raises(ValueError, match='--vision-model-type'):
        vlm._resolve_mimo_vision_args(SimpleNamespace(), SimpleNamespace(), set())

    # Without tokenizer declarations: CLI values win, the registry fills the image geometry and
    # the checkpoint the image token.
    args = SimpleNamespace(vision_model_type='pixtral-vit-large', patch_dim=16, pixel_shuffle=True)
    vlm._resolve_mimo_vision_args(
        args, SimpleNamespace(image_token_id=10), {'vision_model_type', 'patch_dim'}
    )
    assert (args.patch_dim, args.img_h, args.image_token_id) == (16, 1540, 10)
    assert args.conv_merging and not args.pixel_shuffle

    # The image tokens the tokenizer declares (possibly nested) win over the checkpoint's.
    tokens = {'10': '<|im_start|>', '18': '<img>', '19': '</img>', '20': '<image_break>'}
    (tmp_path / 'tokenizer_config.json').write_text(
        json.dumps(
            {
                'added_tokens_decoder': {i: {'content': t} for i, t in tokens.items()},
                'processor': {
                    'image_token': '<img>',
                    'image_break_token': '<image_break>',
                    'image_end_token': '</img>',
                },
            }
        )
    )
    args = SimpleNamespace(vision_model_type='pixtral-vit-large', tokenizer_model=str(tmp_path))
    vlm._resolve_mimo_vision_args(args, SimpleNamespace(image_token_id=10), {'vision_model_type'})
    assert (args.image_token_id, args.image_break_token_id, args.image_end_token_id) == (18, 20, 19)
    sources = {record['attr']: record['source'] for record in args._vlm_arg_resolution}
    assert (sources['img_h'], sources['image_token_id']) == ('encoder registry', 'tokenizer')


def test_check_mimo_checkpoint_fully_loaded(monkeypatch, checkpoint_keys):
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(
        torch.distributed, 'all_gather_object', lambda out, obj: out.__setitem__(0, obj)
    )
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda objs, src: None)
    args = SimpleNamespace(
        mimo_checkpoint_prefix_map={'language_model.': _PREFIX_MAP['language_model.']}
    )
    factory_key = 'language_model.module.module.decoder.layers.0.mixer.in_proj.weight'
    model = SimpleNamespace(sharded_checkpoint_keys={_LANGUAGE_KEY, factory_key})
    checkpoint_keys += [
        f'{factory_key}.z',  # A factory sub-key of a requested key.
        'language_model.module.module.decoder.layers.1.mlp.router.qb_bin_bounds',
        _ENCODER_KEY,  # Outside the mapped prefixes.
    ]
    vlm._check_mimo_checkpoint_fully_loaded(args, model)

    checkpoint_keys += ['language_model.module.module.output_layer.weight']
    with pytest.raises(RuntimeError, match='output_layer'):
        vlm._check_mimo_checkpoint_fully_loaded(args, model)


def test_mimo_vision_configs():
    language_config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4)
    args = SimpleNamespace(
        vision_model_type='pixtral-vit-large',
        vision_num_layers=2,
        vision_hidden_size=64,
        vision_ffn_hidden_size=128,
        vision_num_attention_heads=4,
        vision_num_query_groups=None,
        vision_kv_channels=None,
        vision_projection_activation='fast_gelu',
    )
    config = vision_config(args, language_config)
    # Sizes come from the arguments, head layout follows them, structure from the registry.
    assert (config.num_layers, config.hidden_size, config.ffn_hidden_size) == (2, 64, 128)
    assert (config.num_query_groups, config.kv_channels) == (4, 16)
    assert config.normalization == 'RMSNorm' and not config.add_bias_linear

    projection = vision_projection_config(args, language_config, config.add_bias_linear)
    assert (projection.hidden_size, projection.ffn_hidden_size) == (32, 32)
    assert projection.activation_func is _unfused_fast_gelu
    # Native mcore ViTs have no class tokens.
    assert get_num_image_embeddings(1540, 1540, 14, 'pixtral-vit-large', False, 1, False) == 110**2


@pytest.mark.parametrize(
    "break_id, end_id, expanded, mask",
    [
        (None, None, [5, -1, -1, -1, -1, -1, -1, 7], [None, 0, 1, 2, 3, 4, 5, None]),
        (
            20,
            19,
            [5, -1, -1, -1, 20, -1, -1, -1, 19, 7],
            [None, 0, 1, 2, None, 3, 4, 5, None, None],
        ),
    ],
)
def test_dynamic_image_rows(break_id, end_id, expanded, mask):
    wrapper = object.__new__(VLMInferenceWrapper)
    wrapper.model = SimpleNamespace(
        image_token_index=18,
        dynamic_resolution=True,
        patch_dim=14,
        _pixel_shuffle=False,
        _conv_merging=True,
        _drop_vision_class_token=True,
    )
    image_config = SimpleNamespace(image_break_token_id=break_id, image_end_token_id=end_id)
    wrapper.inference_context = SimpleNamespace(
        config=SimpleNamespace(image_preprocessing_config=image_config)
    )
    # 56x84 pixels -> 4x6 patches -> 2 rows of 3 merged tokens.
    assert wrapper.expand_image_tokens(
        [[5, 18, 7]], imgs_sizes=torch.tensor([[56, 84]]), image_token_id=18
    ) == ([expanded], [mask])
