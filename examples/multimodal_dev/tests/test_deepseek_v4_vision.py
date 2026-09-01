# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused architecture tests for native DeepSeek-V4-Flash-Vision support."""

import inspect
import math
from types import SimpleNamespace

import torch

from examples.multimodal_dev.data.deepseek_v4_mock import MockDeepSeekV4VisionDataset
from examples.multimodal_dev.models.deepseek_v4.configuration import (
    DEEPSEEK_V4_FLASH_VISION_COMPRESS_RATIOS,
    DEEPSEEK_V4_FLASH_VISION_HYBRID_PATTERN,
    DEEPSEEK_V4_VOCAB_SIZE,
    IMAGE,
    IMAGE_END,
    IMAGE_START,
    build_image_block,
    build_image_token_visibility,
    get_deepseek_v4_vision_config,
)
from examples.multimodal_dev.models.deepseek_v4.model import DeepSeekV4VisionModel
from examples.multimodal_dev.models.deepseek_v4.vision_encoder import DeepSeekV4VisionEncoder
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.transformer.transformer_config import TransformerConfig


def test_official_vision_config_and_hybrid_pattern():
    config = get_deepseek_v4_vision_config()
    layers = validate_segment_layers(DEEPSEEK_V4_FLASH_VISION_HYBRID_PATTERN)

    assert config.num_layers == 32
    assert config.hidden_size == 1024
    assert config.num_attention_heads == 16
    assert config.ffn_hidden_size == 2816
    assert config.vision_patch_size == 14
    assert config.vision_downsample_ratio == 3
    assert len(layers) == 86
    assert layers[1::2] == [Symbols.MOE] * 43
    assert DEEPSEEK_V4_FLASH_VISION_COMPRESS_RATIOS == [0, 0, 4] + [128, 4] * 20


def test_image_block_has_official_n_layout_and_permutation():
    n_llm_h, n_llm_w = 3, 2
    types, permutation = build_image_block(n_llm_h, n_llm_w, start_pos=5)

    assert (types == IMAGE_START).sum().item() == 1
    assert types[2].item() == IMAGE_START
    assert types[-1].item() == IMAGE_END
    assert (types == IMAGE).sum().item() == n_llm_h * n_llm_w
    assert torch.equal(permutation.sort().values, torch.arange(n_llm_h * n_llm_w))


def test_image_visibility_matches_bidirectional_span_contract():
    start = DEEPSEEK_V4_VOCAB_SIZE + IMAGE_START
    end = DEEPSEEK_V4_VOCAB_SIZE + IMAGE_END
    input_ids = torch.tensor([[11, start, DEEPSEEK_V4_VOCAB_SIZE + IMAGE, end, 12]])

    visibility = build_image_token_visibility(input_ids, max_image_tokens=16)

    assert torch.equal(visibility.left, torch.tensor([[0, 0, 1, 2, 0]], dtype=torch.int32))
    assert torch.equal(visibility.right, torch.tensor([[0, 2, 1, 0, 0]], dtype=torch.int32))


def _tiny_vision_config() -> TransformerConfig:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=4,
        ffn_hidden_size=12,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        use_cpu_initialization=True,
    )
    config.vision_patch_size = 2
    config.vision_downsample_ratio = 2
    config.vision_rope_theta = 10_000.0
    config.vision_out_hidden_size = 8
    config.vision_max_image_tokens = 16
    return config


def test_tiny_vision_encoder_shape_and_backward():
    config = _tiny_vision_config()
    encoder = DeepSeekV4VisionEncoder(config)
    n_h, n_w = 3, 5
    patches = torch.randn(n_h * n_w, 3 * config.vision_patch_size**2, requires_grad=True)
    grid = torch.tensor([[1, n_h, n_w]])

    output = encoder(patches, grid)

    assert output.shape == (
        math.ceil(n_h / config.vision_downsample_ratio)
        * math.ceil(n_w / config.vision_downsample_ratio),
        config.vision_out_hidden_size,
    )
    output.sum().backward()
    assert patches.grad is not None


def test_vision_parameter_precision_contract():
    config = _tiny_vision_config()
    config.num_layers = 0
    config.params_dtype = torch.bfloat16

    encoder = DeepSeekV4VisionEncoder(config)

    assert encoder.patch_embed.proj.weight.dtype == torch.bfloat16
    assert encoder.aligner.w1.weight.dtype == torch.bfloat16
    assert encoder.norm.weight.dtype == torch.float32
    assert encoder.norm.weight.keep_in_fp32


def test_forward_exposes_future_mdp_embedding_injection_seam():
    parameters = inspect.signature(DeepSeekV4VisionModel.forward).parameters

    assert "vision_embeddings" in parameters
    assert "decoder_input" in parameters


def test_external_vision_rows_share_native_special_embedding_merge():
    model = DeepSeekV4VisionModel.__new__(DeepSeekV4VisionModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(sequence_parallel=False)
    model.actual_vocab_size = 10
    model.image_start = torch.nn.Parameter(torch.full((3,), 1.0))
    model.image_pad = torch.nn.Parameter(torch.full((3,), 2.0))
    model.image_newline = torch.nn.Parameter(torch.full((3,), 4.0))
    model.image_end = torch.nn.Parameter(torch.full((3,), 5.0))
    input_ids = torch.tensor(
        [[1, 10 + IMAGE_START, 10 + IMAGE, 10 + IMAGE_END, 2]], dtype=torch.long
    )
    text_embeddings = torch.zeros(5, 1, 3)
    vision_embeddings = torch.full((1, 3), 3.0, requires_grad=True)

    merged = model._scatter_vision_embeddings(
        input_ids, text_embeddings, vision_embeddings, vision_token_indices=torch.tensor([[0, 2]])
    )

    assert torch.equal(merged[:, 0, 0], torch.tensor([0.0, 1.0, 3.0, 5.0, 0.0]))
    merged.sum().backward()
    assert torch.equal(vision_embeddings.grad, torch.ones_like(vision_embeddings))


def test_sparse_visibility_metadata_is_safe_for_full_recompute(monkeypatch):
    from megatron.core import recompute

    visibility = build_image_token_visibility(
        torch.tensor(
            [[1, DEEPSEEK_V4_VOCAB_SIZE + IMAGE_START, DEEPSEEK_V4_VOCAB_SIZE + IMAGE_END]]
        )
    )
    seen = {}

    class Layer:
        layer_number = 1

        def __call__(self, hidden_states, attention_mask, **kwargs):
            seen["attention_mask"] = attention_mask
            return hidden_states + 1

    stack = SimpleNamespace(
        config=SimpleNamespace(
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=False,
            fp8=False,
            fp4=None,
        ),
        layers=[Layer()],
        num_layers_per_pipeline_rank=1,
        pg_collection=SimpleNamespace(tp=None),
    )

    def checkpoint_without_autograd(function, distribute_saved_activations, *args):
        del distribute_saved_activations
        assert all(argument is None or isinstance(argument, torch.Tensor) for argument in args)
        return function(*args)

    monkeypatch.setattr(recompute.tensor_parallel, "checkpoint", checkpoint_without_autograd)
    hidden_states = torch.zeros(3, 1, 2)

    output = recompute.checkpointed_forward(
        stack,
        hidden_states=hidden_states,
        attention_mask=visibility,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        attention_bias=None,
        packed_seq_params=None,
        use_inner_quantization_context=False,
        padding_mask=None,
    )

    assert seen["attention_mask"] is visibility
    assert torch.equal(output, hidden_states + 1)


def test_mock_dataset_never_uses_synthetic_ids_as_lm_targets():
    sample = MockDeepSeekV4VisionDataset(num_samples=1, seq_length=128, image_size=42)[0]

    assert torch.all(sample["labels"][sample["loss_mask"] == 0] == -100)
    assert torch.all(sample["labels"][sample["loss_mask"] == 1] < DEEPSEEK_V4_VOCAB_SIZE)
