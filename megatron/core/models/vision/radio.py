# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

# RADIO reference code: https://github.com/NVlabs/RADIO

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


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
        has_cpe: (bool): Whether to use cropped position embeddings. Defaults to True.
        embedder_bias: (bool): Bias in embedder linear. Defaults to False.
        dynamic_resolution: (bool): Whether to use dynamic resolution. Defaults to False.
        force_eval_mode: (bool): Force the model to stay in eval mode. Defaults to False.
        force_cpe_eval_mode: (bool): Force eval mode only for CPE.
            Defaults to False.
        interpolate_only_cpe: (bool): Interpolate position embeddings to
            input size without cropping. Defaults to False.
        cpe_aspect_ratio_select: (bool): Select position embeddings based
            on aspect ratio. Defaults to False.
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
        dynamic_resolution: bool = False,
        force_eval_mode: bool = False,
        force_cpe_eval_mode: bool = False,
        interpolate_only_cpe: bool = False,
        cpe_aspect_ratio_select: bool = False,
        temporal_patch_dim: int = 1,
        temporal_ckpt_compat: bool = False,  # allow_checkpoint_without_temporal_compression
        separate_video_embedder: bool = False,
        pg_collection=None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.temporal_patch_dim = temporal_patch_dim
        self.img_h = img_h
        self.img_w = img_w

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0

        self.input_dims = (img_h // patch_dim, img_w // patch_dim)

        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_num_rows = max_img_h // patch_dim
        self.max_num_cols = max_img_w // patch_dim
        self.max_num_patches = self.max_num_rows * self.max_num_cols

        self.use_mask_token = use_mask_token
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, self.visual_hidden_size))

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len
        if self.add_class_token:
            self.class_token = nn.Parameter(
                torch.randn(
                    self.class_token_len,
                    self.visual_hidden_size,
                    dtype=transformer_config.params_dtype,
                )
            )
            if transformer_config.fp8:
                self.register_load_state_dict_pre_hook(fp8_pad_hook)

        self.seq_length = (img_h // self.patch_dim) * (img_w // self.patch_dim) + (
            self.class_token_len if self.add_class_token else 0
        )

        pos_scale = self.visual_hidden_size**-0.5
        self.position_embeddings = nn.Parameter(
            torch.randn(
                1,
                self.max_num_patches,
                self.visual_hidden_size,
                dtype=transformer_config.params_dtype,
            )
            * pos_scale
        )
        self.pos_dropout = pos_dropout
        self.has_cpe = has_cpe
        self.dynamic_resolution = dynamic_resolution
        self.force_eval_mode = force_eval_mode
        self.force_cpe_eval_mode = force_cpe_eval_mode
        self.interpolate_only_cpe = interpolate_only_cpe
        self.cpe_aspect_ratio_select = cpe_aspect_ratio_select

        orig_sequence_parallel = transformer_config.sequence_parallel
        transformer_config.sequence_parallel = False

        self.separate_video_embedder = separate_video_embedder

        if separate_video_embedder and self.temporal_patch_dim > 1:
            self.embedder = ColumnParallelLinear(
                input_size=3 * self.patch_dim * self.patch_dim,
                output_size=self.visual_hidden_size,
                bias=embedder_bias,
                config=transformer_config,
                gather_output=True,
                disable_grad_reduce=True,
                init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=1.0),
            )
            self.video_embedder = ColumnParallelLinear(
                input_size=3 * self.temporal_patch_dim * self.patch_dim * self.patch_dim,
                output_size=self.visual_hidden_size,
                bias=embedder_bias,
                config=transformer_config,
                gather_output=True,
                disable_grad_reduce=True,
                init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=1.0),
            )
        else:
            self.embedder = ColumnParallelLinear(
                input_size=3 * self.temporal_patch_dim * self.patch_dim * self.patch_dim,
                output_size=self.visual_hidden_size,
                bias=embedder_bias,
                config=transformer_config,
                gather_output=True,
                disable_grad_reduce=True,
                init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=1.0),
            )
        transformer_config.sequence_parallel = orig_sequence_parallel

        if temporal_ckpt_compat and self.temporal_patch_dim > 1:
            if separate_video_embedder:
                self.register_load_state_dict_pre_hook(
                    self._state_dict_pre_hook_init_video_embedder
                )
            else:
                self.register_load_state_dict_pre_hook(self._state_dict_pre_hook_init_embedder)

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

        self.force_cpe_eval_mode = force_cpe_eval_mode
        if self.force_eval_mode:
            self.eval()

    def train(self, mode: bool = True) -> 'RADIOViTModel':
        """Override train to respect force_eval_mode."""
        if mode and self.force_eval_mode:
            logging.getLogger(__name__).warning(
                "RADIOViTModel has force_eval_mode=True. Keeping model in eval mode."
            )
            return self
        return super().train(mode)

    def _state_dict_pre_hook_init_embedder(self, module, state_dict, prefix, *args, **kwargs):
        """Convert 2D embedder weights to 3D for temporal compression."""  # noqa: C0116
        key = prefix + "embedder.weight"
        if key not in state_dict:
            return

        weight = state_dict[key]
        expected_2d_size = 3 * self.patch_dim * self.patch_dim

        if weight.shape[1] == expected_2d_size:
            weight_3d = weight.repeat(1, self.temporal_patch_dim)
            weight_3d = weight_3d / self.temporal_patch_dim
            state_dict[key] = weight_3d
            logging.getLogger(__name__).info(
                f"Converted embedder weights from 2D ({weight.shape}) to 3D ({weight_3d.shape}) "
                f"for temporal_patch_dim={self.temporal_patch_dim}"
            )

    def _state_dict_pre_hook_init_video_embedder(self, module, state_dict, prefix, *args, **kwargs):
        embedder_key = prefix + "embedder.weight"
        video_embedder_key = prefix + "video_embedder.weight"

        if embedder_key not in state_dict:
            return

        embedder_weight = state_dict[embedder_key]
        expected_2d_size = 3 * self.patch_dim * self.patch_dim
        expected_3d_size = 3 * self.temporal_patch_dim * self.patch_dim * self.patch_dim

        if video_embedder_key not in state_dict:
            if embedder_weight.shape[1] == expected_2d_size:
                video_weight = embedder_weight.repeat(1, self.temporal_patch_dim)
                video_weight = video_weight / self.temporal_patch_dim
                state_dict[video_embedder_key] = video_weight
                logging.getLogger(__name__).info(
                    f"Created video_embedder weights ({video_weight.shape}) from embedder weights "
                    f"({embedder_weight.shape}) for temporal_patch_dim={self.temporal_patch_dim}"
                )
            elif embedder_weight.shape[1] == expected_3d_size:
                state_dict[video_embedder_key] = embedder_weight.clone()
                embedder_2d = embedder_weight.view(
                    embedder_weight.shape[0], self.temporal_patch_dim, -1
                ).mean(dim=1)
                state_dict[embedder_key] = embedder_2d
                logging.getLogger(__name__).info(
                    f"Split combined 3D embedder ({embedder_weight.shape}) into "
                    f"image ({embedder_2d.shape}) and video ({embedder_weight.shape})"
                )

        embedder_bias_key = prefix + "embedder.bias"
        video_embedder_bias_key = prefix + "video_embedder.bias"
        if embedder_bias_key in state_dict and video_embedder_bias_key not in state_dict:
            state_dict[video_embedder_bias_key] = state_dict[embedder_bias_key].clone()

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model."""
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        imgs_sizes: Optional[Union[List[Tuple[int, int]], torch.Tensor]] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        num_frames: Optional[List[int]] = None,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, Union[List[Tuple[int, int]], torch.Tensor], List[int]]
    ]:
        """Forward function of the RADIO ViT Model.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w] or
                [1, total_patches, C*P*P] for dynamic res.
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.
            imgs_sizes: Sizes of the images, for dynamic resolution.
            packed_seq_params (PackedSeqParams): Packed sequence params for attention.
            num_frames (List[int]): Number of frames per media item.

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        if not self.dynamic_resolution:
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

        is_image = None
        if self.temporal_patch_dim > 1:
            assert num_frames is not None, "num_frames is required when temporal_patch_dim > 1"
            if not self.dynamic_resolution:
                raise NotImplementedError(
                    "Temporal compression is only supported for dynamic resolution"
                )
            x, imgs_sizes, num_frames, packed_seq_params, is_image = self._apply_temporal_grouping(
                x,
                imgs_sizes,
                num_frames,
                packed_seq_params,
                skip_image_duplication=self.separate_video_embedder,
            )

        if self.separate_video_embedder and self.temporal_patch_dim > 1:
            embedded_chunks = []
            for chunk, is_img in zip(x, is_image):
                if is_img:
                    emb, _ = self.embedder(chunk)
                else:
                    emb, _ = self.video_embedder(chunk)
                embedded_chunks.append(emb)
            x = torch.cat(embedded_chunks, dim=1)
        else:
            x, _ = self.embedder(x)

        if self.dynamic_resolution:
            if torch.is_tensor(imgs_sizes):
                seq_lens = torch.prod(imgs_sizes // self.patch_dim, dim=-1).tolist()
                sizes_iter = [tuple(sz.tolist()) for sz in imgs_sizes]
            else:
                seq_lens = [(h // self.patch_dim) * (w // self.patch_dim) for h, w in imgs_sizes]
                sizes_iter = imgs_sizes

            assert sum(seq_lens) == x.shape[1], f"{sum(seq_lens)} != {x.shape[1]}"

            chunks = torch.split(x, seq_lens, dim=1)
            chunks = [
                self.apply_pos_enc(chunk, input_size=size)[0]
                for chunk, size in zip(chunks, sizes_iter)
            ]
            x = torch.cat(chunks, dim=1)
        else:
            x, pos_enc = self.apply_pos_enc(x, input_size=input_size)

        if self.add_class_token:
            class_token = self.class_token.expand(x.shape[0], -1, -1)
            if self.dynamic_resolution:
                out = []
                current_length = 0
                for input_size in imgs_sizes:
                    seq_length = input_size[0] // self.patch_dim * input_size[1] // self.patch_dim
                    out.append(class_token)
                    out.append(x[:, current_length : current_length + seq_length, :])
                    current_length += seq_length
                x = torch.cat(out, dim=1)
                add_cu = torch.full_like(
                    packed_seq_params.cu_seqlens_q, self.class_token_len, dtype=torch.int32
                )
                add_cu[0] = 0
                add_cu = torch.cumsum(add_cu, dim=-1, dtype=torch.int32)
                packed_seq_params.cu_seqlens_q = packed_seq_params.cu_seqlens_q + add_cu
                packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_kv + add_cu
                packed_seq_params.max_seqlen_q = (
                    packed_seq_params.max_seqlen_q + self.class_token_len
                )
                packed_seq_params.max_seqlen_kv = (
                    packed_seq_params.max_seqlen_kv + self.class_token_len
                )
            else:
                x = torch.cat([class_token, x], dim=1)

        if not self.dynamic_resolution:
            assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"

        if self.ln_pre:
            x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = x.contiguous()

        x = self.decoder(x, attention_mask=attention_mask, packed_seq_params=packed_seq_params)

        x = x.permute(1, 0, 2)
        x = x.contiguous()

        if self.ln_post:
            x = self.ln_post(x)

        if self.temporal_patch_dim > 1:
            return x, imgs_sizes, num_frames
        else:
            return x

    def _apply_temporal_grouping(
        self,
        x: torch.Tensor,
        imgs_sizes: Union[List[Tuple[int, int]], torch.Tensor],
        num_frames: Union[List[int], torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams] = None,
        skip_image_duplication: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, List[torch.Tensor]],
        Union[List[Tuple[int, int]], torch.Tensor],
        List[int],
        Optional[PackedSeqParams],
        List[bool],
    ]:
        """Group consecutive video frames into tubelets for temporal compression."""
        T = self.temporal_patch_dim
        total_frames = sum(num_frames)
        num_imgs_sizes = imgs_sizes.shape[0]
        expected_tubelets = sum(1 if nf == 1 else math.ceil(nf / T) for nf in num_frames)

        assert total_frames == num_imgs_sizes, (
            f"imgs_sizes must have one entry per frame (ungrouped). "
            f"Got {num_imgs_sizes} entries but sum(num_frames)={total_frames}."
        )
        C_P_P = 3 * self.patch_dim * self.patch_dim

        if torch.is_tensor(imgs_sizes):
            imgs_sizes_list = [tuple(sz.tolist()) for sz in imgs_sizes]
        else:
            imgs_sizes_list = list(imgs_sizes)

        seq_lens = [(h // self.patch_dim) * (w // self.patch_dim) for h, w in imgs_sizes_list]
        chunks = torch.split(x, seq_lens, dim=1)

        grouped_chunks = []
        new_imgs_sizes_list = []
        new_num_frames = []
        is_image = []
        tile_idx = 0

        for nf in num_frames:
            if nf == 1:
                chunk = chunks[tile_idx]
                if skip_image_duplication:
                    grouped_chunks.append(chunk)
                else:
                    duplicated = chunk.repeat(1, 1, T)
                    grouped_chunks.append(duplicated)
                new_imgs_sizes_list.append(imgs_sizes_list[tile_idx])
                new_num_frames.append(1)
                is_image.append(True)
                tile_idx += 1
            else:
                padded_nf = nf if nf % T == 0 else nf + (T - nf % T)
                for group_start in range(0, padded_nf, T):
                    group_frames = []
                    for t in range(T):
                        frame_idx = group_start + t
                        if frame_idx < nf:
                            group_frames.append(chunks[tile_idx + frame_idx])
                        else:
                            group_frames.append(chunks[tile_idx + nf - 1])
                    grouped = torch.cat(group_frames, dim=-1)
                    grouped_chunks.append(grouped)
                    new_imgs_sizes_list.append(imgs_sizes_list[tile_idx])
                    is_image.append(False)
                new_num_frames.append(padded_nf // T)
                tile_idx += nf

        if skip_image_duplication:
            x_grouped = grouped_chunks
        else:
            x_grouped = torch.cat(grouped_chunks, dim=1)

        if torch.is_tensor(imgs_sizes):
            new_imgs_sizes = torch.tensor(
                new_imgs_sizes_list, dtype=imgs_sizes.dtype, device=imgs_sizes.device
            )
        else:
            new_imgs_sizes = new_imgs_sizes_list

        new_packed_seq_params = None
        if packed_seq_params is not None:
            seq_lens = [
                (h // self.patch_dim) * (w // self.patch_dim) for h, w in new_imgs_sizes_list
            ]
            cu_seqlens = [0]
            for sl in seq_lens:
                cu_seqlens.append(cu_seqlens[-1] + sl)
            device = packed_seq_params.cu_seqlens_q.device
            dtype = packed_seq_params.cu_seqlens_q.dtype
            cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=dtype, device=device)
            max_seqlen = torch.tensor(max(seq_lens) if seq_lens else 0, dtype=dtype, device=device)
            new_packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens_tensor,
                cu_seqlens_kv=cu_seqlens_tensor,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
                qkv_format=packed_seq_params.qkv_format,
            )

        return x_grouped, new_imgs_sizes, new_num_frames, new_packed_seq_params, is_image

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Apply positional encoding to patches."""
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
        """Get positional encoding for certain input size."""
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
        """Get RADIO absolute positional embeddings."""
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

        def aspect_ratio_select(pos_embed):
            (pos_H, pos_W) = pos_embed.shape[-2:]
            (input_H, input_W) = input_dims
            if input_H == input_W:
                return pos_embed
            (crop_H, crop_W) = (pos_H, pos_W)
            if input_W < input_H:
                crop_W = min(pos_W, math.ceil(pos_W * (input_W / input_H)))
            else:
                crop_H = min(pos_H, math.ceil(pos_H * (input_H / input_W)))
            return pos_embed[..., :crop_H, :crop_W]

        if self.interpolate_only_cpe:
            if self.cpe_aspect_ratio_select:
                pos_embed = aspect_ratio_select(pos_embed)
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
            ).to(pos_embed.dtype)

        elif self.has_cpe:
            if self.training and not self.force_cpe_eval_mode:
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

                grid_xy.mul_(2).sub_(1)

                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(pos_embed.dtype)

            else:
                max_dim = max(input_dims)

                (B, C, _H, _W) = pos_embed.shape
                aspect_ratio_select_required = B * C * max_dim**2 >= torch.iinfo(torch.int32).max
                if aspect_ratio_select_required or self.cpe_aspect_ratio_select:
                    pos_embed = aspect_ratio_select(pos_embed)
                    pos_embed = F.interpolate(
                        pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
                    ).to(pos_embed.dtype)
                else:
                    pos_embed = F.interpolate(
                        pos_embed.float(),
                        size=(max_dim, max_dim),
                        align_corners=False,
                        mode="bilinear",
                    ).to(pos_embed.dtype)
                    pos_embed = window_select(pos_embed)

        elif self.cpe_aspect_ratio_select:
            pos_embed = aspect_ratio_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed


def fp8_pad_hook(
    module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    """FP8 requires class token length to be a multiple of 16 (for this model).

    Original model checkpoint may not be padded for FP8 so pad it here.
    """
    if not "vision_model.class_token" in state_dict:
        return

    pad = 32 if module.config.fp8_recipe == "mxfp8" else 16

    class_token = state_dict["vision_model.class_token"]
    if class_token.shape[0] % pad != 0:
        pad_len = pad - (class_token.shape[0] % pad)
        pad_tensor = torch.randn(
            pad_len, class_token.shape[-1], dtype=class_token.dtype, device=class_token.device
        )
        class_token = torch.cat([pad_tensor, class_token], dim=0)
        state_dict["vision_model.class_token"] = class_token

    return
