# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

# RADIO reference code: https://github.com/NVlabs/RADIO


class RADIOViTModel(VisionModule):
    """RADIO ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        ln_post_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_post.
        use_mask_token (bool, optional): Whether to use RADIO mask token. Default to False.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
        max_img_h (int): Max input image height.
        max_img_w (int): Max input image width.
        pos_dropout (int): Positional encoding dropout value. Defaults to 0.
        has_cpe: (bool): Whether to use conditional positional encoding. Defaults to True.
        embedder_bias: (bool): Bias in embedder linear. Defaults to False.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        ln_pre_impl: Union[ModuleSpec, type] = None,
        ln_post_impl: Union[ModuleSpec, type] = None,
        use_mask_token: bool = False,
        add_class_token: bool = True,
        class_token_len: int = 8,
        patch_dim: int = 16,
        img_h: int = 224,
        img_w: int = 224,
        max_img_h: int = 2048,
        max_img_w: int = 2048,
        pos_dropout: int = 0,
        has_cpe: bool = True,
        embedder_bias: bool = False,
    ) -> None:
        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0

        self.input_dims = (img_h // patch_dim, img_w // patch_dim)

        # used for positional embedding
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_num_rows = max_img_h // patch_dim
        self.max_num_cols = max_img_w // patch_dim
        self.max_num_patches = self.max_num_rows * self.max_num_cols

        # TODO: are we actually going to use this anywhere?
        self.use_mask_token = use_mask_token
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, self.visual_hidden_size))

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len
        if self.add_class_token:
            self.class_token = nn.Parameter(
                torch.randn(self.class_token_len, self.visual_hidden_size)
            )

        self.seq_length = (img_h // self.patch_dim) * (img_w // self.patch_dim) + (
            self.class_token_len if self.add_class_token else 0
        )

        pos_scale = self.visual_hidden_size**-0.5
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.max_num_patches, self.visual_hidden_size) * pos_scale
        )
        self.pos_dropout = pos_dropout
        self.has_cpe = has_cpe

        # Using non-TE version so we can force gather_output
        self.embedder = ColumnParallelLinear(
            input_size=3 * self.patch_dim * self.patch_dim,
            output_size=self.visual_hidden_size,
            bias=embedder_bias,
            config=transformer_config,
            gather_output=True,
            init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=1.0),
        )

        self.model_type = ModelType.encoder_or_decoder

        self.ln_pre = None
        self.ln_post = None
        if ln_pre_impl is not None:
            self.ln_pre = build_module(
                ln_pre_impl,
                config=transformer_config,
                hidden_size=self.visual_hidden_size,
                eps=transformer_config.layernorm_epsilon,
            )
        if ln_post_impl is not None:
            self.ln_post = build_module(
                ln_post_impl,
                config=transformer_config,
                hidden_size=self.visual_hidden_size,
                eps=transformer_config.layernorm_epsilon,
            )

        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the RADIO ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        input_size = x.shape[2:]
        py = x.shape[-2] // self.patch_dim
        px = x.shape[-1] // self.patch_dim
        x = rearrange(
            x,
            'b c (py yy) (px xx) -> b (py px) (c yy xx)',
            py=py,
            yy=self.patch_dim,
            px=px,
            xx=self.patch_dim,
        )
        x, _ = self.embedder(x)  # [batch, seq_length, hidden_size]

        x, _ = self.apply_pos_enc(x, input_size=input_size)

        if self.add_class_token:
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]

            x = torch.cat(
                [class_token, x], dim=1
            )  # [batch, seq_length + class_token_len, hidden_size]

        assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"

        if self.ln_pre:
            x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        x = x.contiguous()

        x = self.decoder(x, attention_mask=attention_mask)

        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()

        if self.ln_post:
            x = self.ln_post(x)

        return x

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Apply positional encoding to patches"""
        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)

        if self.training and self.pos_dropout > 0:
            keeps = (
                torch.rand(patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device)
                > self.pos_dropout
            )
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Get positional encoding for certain input size"""
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_dim for d in input_size)

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)

        if patch_idxs is None:
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])

        pos_embed = torch.gather(
            pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs
        )
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]):
        """Get RADIO absolute positional embeddings"""
        if (self.max_num_rows, self.max_num_cols) == input_dims:
            return self.position_embeddings

        pos_embed = self.position_embeddings.reshape(
            1, self.max_num_rows, self.max_num_cols, -1
        ).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.has_cpe:
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = (
                    torch.rand(batch_size, 1, 1, device=pos_embed.device) * (1 - min_scale)
                    + min_scale
                )
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = torch.exp(
                    torch.rand(batch_size, 1, 1, device=pos_embed.device)
                    * (aspect_max - aspect_min)
                    + aspect_min
                )

                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = torch.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)

                pos_xy = torch.rand(batch_size, 1, 1, 2, device=pos_embed.device) * (1 - scale_xy)

                lin_x = torch.linspace(0, 1, steps=input_dims[1], device=pos_embed.device)[
                    None, None
                ].expand(batch_size, input_dims[0], -1)
                lin_y = torch.linspace(0, 1, steps=input_dims[0], device=pos_embed.device)[
                    None, :, None
                ].expand(batch_size, -1, input_dims[1])

                lin_xy = torch.stack([lin_x, lin_y], dim=-1)

                grid_xy = lin_xy * scale_xy + pos_xy

                # Convert to [-1, 1] range
                grid_xy.mul_(2).sub_(1)

                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(
                    pos_embed.float(), size=(max_dim, max_dim), align_corners=True, mode='bilinear'
                ).to(pos_embed.dtype)

                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=True, mode='bilinear'
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed
