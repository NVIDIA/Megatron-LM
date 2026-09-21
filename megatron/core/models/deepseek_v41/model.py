# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek-V4.1 model using HybridModel's embedding, head and checkpoint interface."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from megatron.core.models.deepseek_v41.dspark import DSpark, DSparkOutput
from megatron.core.models.deepseek_v41.engram import Engram, EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import build_compressed_token_map
from megatron.core.models.deepseek_v41.hybrid_adapter import (
    DeepSeekV41ForwardContext,
    deepseek_v41_multimodal_stack_spec,
    deepseek_v41_stack_spec,
)
from megatron.core.models.deepseek_v41.image_processing import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
)
from megatron.core.models.deepseek_v41.vision import Aligner, ViT
from megatron.core.models.hybrid.hybrid_model import HybridModel


@dataclass
class DeepSeekV41TrainingOutput:
    """Separate backbone and draft objectives; the draft objective never trains the backbone."""

    backbone: torch.Tensor
    draft: DSparkOutput
    draft_loss: torch.Tensor


class DeepSeekV41Model(HybridModel):
    """Compose the V4.1 backbone, conditional memory, vision encoder and draft model."""

    def __init__(
        self,
        config,
        vocab_size,
        max_sequence_length,
        *,
        pg_collection,
        token_map=None,
        tokenizer=None,
        **kwargs,
    ) -> None:
        if kwargs.get("share_embeddings_and_output_weights", False):
            raise ValueError("The released V4.1 architecture uses untied embedding/output weights")
        hasher = None
        if getattr(config, "engram_config", None):
            if token_map is None:
                if tokenizer is None:
                    raise ValueError(
                        "Engram requires the released tokenizer or an explicit compressed token map"
                    )
                token_map, _ = build_compressed_token_map(tokenizer)
                token_map = torch.tensor(token_map, dtype=torch.long)
            hasher = EngramHasher(config.engram_config, token_map)
            ids = hasher.layout.layer_ids
            if len(set(ids)) != len(ids) or any(
                i < 0 or i >= (config.num_layers // 2) for i in ids
            ):
                raise ValueError("Engram layers must be distinct zero-based backbone blocks")
        if getattr(config, "vision_config", None):
            v = config.vision_config
            if (
                v.hidden_size % v.num_attention_heads
                or (v.hidden_size // v.num_attention_heads) % 4
            ):
                raise ValueError("Vision heads require a dimension divisible by four for 2D RoPE")
        if getattr(config, "dspark_config", None):
            d = config.dspark_config
            if not 0 <= d.noise_token_id < vocab_size:
                raise ValueError("DSpark noise token must belong to the model vocabulary")
            if d.target_layer_ids != sorted(set(d.target_layer_ids)):
                raise ValueError("DSpark target layers must be strictly increasing")
        super().__init__(
            config=config,
            hybrid_stack_spec=(
                deepseek_v41_multimodal_stack_spec
                if config.vision_config
                else deepseek_v41_stack_spec
            ),
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=config.hybrid_pattern,
            position_embedding_type="none",
            pg_collection=pg_collection,
            **kwargs,
        )
        self.engram_hash = hasher
        if hasher is not None:
            for i in self.engram_hash.layout.layer_ids:
                self.decoder.layers[2 * i].engram = Engram(
                    config, self.engram_hash.layout, i, pg_collection
                )
        self.vision = None
        if getattr(config, "vision_config", None):
            v = config.vision_config
            self.vision_args = SimpleNamespace(
                dim=config.hidden_size,
                vision_n_layers=v.num_hidden_layers,
                vision_dim=v.hidden_size,
                vision_n_heads=v.num_attention_heads,
                vision_inter_dim=v.intermediate_size,
                vision_patch_size=v.patch_size,
                vision_rope_theta=v.rope_theta,
                vision_downsample_ratio=v.downsample_ratio,
                vision_max_n_token=v.max_image_tokens,
                vision_min_pixels=v.min_pixels,
                vision_max_wh_ratio=v.max_wh_ratio,
            )
            self.vision = ViT(self.vision_args)
            self.aligner = Aligner(self.vision_args)
            for name in ("image_start", "image_end", "image_newline"):
                parameter = nn.Parameter(torch.empty(config.hidden_size, dtype=config.params_dtype))
                if config.perform_initialization:
                    config.init_method(parameter)
                self.register_parameter(name, parameter)
        self.dspark = None
        if getattr(config, "dspark_config", None) and config.dspark_config.num_layers:
            self.dspark = DSpark(config, vocab_size, pg_collection)
        self._draft_only_training = False

    def freeze_backbone_for_draft_training(self) -> None:
        """Freeze backbone weights and routing biases before constructing the draft optimizer."""
        if self.dspark is None:
            raise ValueError("Draft-only training requires a DSpark configuration")
        self._draft_only_training = True
        for name, parameter in self.named_parameters():
            if not name.startswith("dspark."):
                parameter.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval mode during a dedicated DSpark stage."""
        super().train(mode)
        if mode and getattr(self, "_draft_only_training", False):
            for name, module in self.named_children():
                if name != "dspark":
                    module.eval()
        return self

    def encode_image(self, image):
        """Encode an ImageInput while retaining gradients through the ViT and projector."""
        if self.vision is None:
            raise ValueError("This configuration has no vision encoder")
        parameter = self.vision.patch_embed.proj.weight
        patches = image.patches.to(device=parameter.device, dtype=parameter.dtype)
        return self.aligner(
            self.vision(patches, image.n_vit_h, image.n_vit_w), image.n_vit_h, image.n_vit_w
        )

    def forward_features(
        self,
        input_ids,
        position_ids,
        *,
        images=None,
        attention_mask=None,
        padding_mask=None,
        target_layer_ids=(),
    ):
        """Return backbone hidden states and optional detached DSpark target features."""
        hidden = self.embedding(input_ids, position_ids)
        image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if images is not None:
            if len(images) != input_ids.shape[0]:
                raise ValueError("Image metadata must have one entry per batch element")
            hidden = hidden.clone()
            for b, sample in enumerate(images):
                for image in sample or ():
                    end = image.start + image.types.numel()
                    if (
                        image.start < 0
                        or end > input_ids.shape[1]
                        or image_mask[b, image.start : end].any()
                    ):
                        raise ValueError("Image spans must be in range and non-overlapping")
                    types = image.types.to(hidden.device)
                    features = self.encode_image(image).to(hidden.dtype)
                    if (types == IMAGE).sum() != features.shape[0]:
                        raise ValueError(
                            "Image span patch slots do not match projected image features"
                        )
                    span = hidden[image.start : end, b]
                    span[types == IMAGE] = features
                    for kind, name in (
                        (IMAGE_START, "image_start"),
                        (IMAGE_END, "image_end"),
                        (IMAGE_NEW_LINE, "image_newline"),
                    ):
                        span[types == kind] = getattr(self, name).to(hidden.dtype)
                    image_mask[b, image.start : end] = True
        hashes = self.engram_hash(input_ids, ~image_mask) if self.engram_hash is not None else None
        return self.decoder(
            hidden,
            attention_mask,
            padding_mask=padding_mask,
            forward_context=DeepSeekV41ForwardContext(
                engram_hashes=hashes,
                token_mask=~image_mask,
                image_mask=image_mask if self.vision is not None else None,
                return_target_hidden=True,
                target_layer_ids=tuple(target_layer_ids),
            ),
        )

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask=None,
        *,
        labels=None,
        images=None,
        padding_mask=None,
        packed_seq_params=None,
        draft_anchor_positions=None,
        **kwargs,
    ):
        """Training forward returning token losses or batch-major vocabulary logits."""
        if packed_seq_params is not None or kwargs:
            raise NotImplementedError("V4.1 currently supports full, unpacked training forwards")
        if draft_anchor_positions is not None and self.dspark is None:
            raise ValueError("Draft anchors require a DSpark configuration")
        target_layers = self.dspark.target_layer_ids if draft_anchor_positions is not None else ()
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self._draft_only_training):
            hidden, features = self.forward_features(
                input_ids,
                position_ids,
                images=images,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                target_layer_ids=target_layers,
            )
            logits, _ = self.output_layer(hidden)
            backbone = (
                self.compute_language_model_loss(labels, logits)
                if labels is not None
                else logits.transpose(0, 1).contiguous()
            )
        if draft_anchor_positions is None:
            return backbone
        draft = self.dspark(
            input_ids,
            features,
            draft_anchor_positions,
            self.embedding.word_embeddings.weight,
            self.output_layer.weight,
        )
        positions = draft_anchor_positions[..., None] + torch.arange(
            self.dspark.block_size, device=input_ids.device
        )
        teacher = logits.transpose(0, 1).gather(
            1, positions.flatten(1)[..., None].expand(-1, -1, logits.shape[-1])
        )
        teacher = teacher.reshape(*positions.shape, logits.shape[-1])
        return DeepSeekV41TrainingOutput(backbone, draft, draft.loss(teacher))
