# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
import logging
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

    Recommended CPE mode:
    1. CPE w/ force eval mode:
        - Maintains aspect ratio, maps long edge to 1, resizes then crops
        - Req. flags
            - `force_eval_mode=True` (for pre-training)
            - `force_cpe_eval_mode=True` (for SFT)
        - Expected defaults:
            - `has_cpe=True`
            - `cpe_aspect_ratio_select=False`
            - `interpolate_only_cpe=False`
        - To enable randomness, leave `force_eval_mode=False` and `force_cpe_eval_mode=False`
            - This is the "default" mode for backwards compatibility (not recommended)

    Alternative CPE modes (for baselines)
    2. CPE w/ force eval mode, cpe_aspect_ratio_select=True:
        - Maintains aspect ratio, maps long edge to 1, crops then resizes
            - Slightly worse performance due to border effects along the short edge
        - Req. flags
            - `force_eval_mode=True` (for pre-training)
            - `force_cpe_eval_mode=True` (for SFT)
            - `cpe_aspect_ratio_select=True`
        - Expected defaults:
            - `has_cpe=True`
            - `interpolate_only_cpe=False`
        - To enable randomness, leave `force_eval_mode=False` and `force_cpe_eval_mode=False`
            - This mode has not been trained/evaluated due to worse performance over #1 above
    3. Interpolate only (no CPE):
        - Doesn't maintain aspect ratio, maps long edge to 1, directly resizes to input size
        - Req. flags:
            - `interpolate_only_cpe=True`
        - Expected defaults:
            - cpe_aspect_ratio_select=False`
            - Others ignored: `has_cpe`, `force_eval_mode`, `force_cpe_eval_mode`

    Other CPE modes (not trained/evaluated):
    4. Interpolate only (no CPE), cpe_aspect_ratio_select=True:
        - Maintains aspect ratio, maps long edge to 1, crops then resizes
        - Same as #2 above, just a different code path to reduce confusing flag settings in init
    5. Disabled CPE:
        - Maintains aspect ratio, doesn't map long edge to 1 (if long edge < pos embed long edge), crops then resizes
        - Req. flags:
            - `has_cpe=False`
        - Expected defaults:
            - `cpe_aspect_ratio_select=False`
            - Others ignored: `force_eval_mode`, `force_cpe_eval_mode`, `interpolate_only_cpe`

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
        force_eval_mode: (bool): Force the model to stay in eval mode, usually for pre-training. Defaults to False.
        force_cpe_eval_mode: (bool): Force the model to effectively use eval mode only for CPE. Defaults to False.
        interpolate_only_cpe: (bool): Interpolate the position embeddings to input size, without any cropping. Defaults to False.
        cpe_aspect_ratio_select: (bool): Select position embeddings based on aspect ratio so long edge always mapped to 1. Defaults to False.
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
        allow_checkpoint_without_temporal_compression: bool = False,
        separate_video_embedder: bool = False,
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

        # Using non-TE version so we can force gather_output
        orig_sequence_parallel = transformer_config.sequence_parallel
        transformer_config.sequence_parallel = False

        self.separate_video_embedder = separate_video_embedder

        if separate_video_embedder and self.temporal_patch_dim > 1:
            # Separate embedders for images and videos
            # Image embedder: C * P * P (e.g., 3*16*16=768)
            self.embedder = ColumnParallelLinear(
                input_size=3 * self.patch_dim * self.patch_dim,
                output_size=self.visual_hidden_size,
                bias=embedder_bias,
                config=transformer_config,
                gather_output=True,
                disable_grad_reduce=True,
                init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=1.0),
            )
            # Video embedder: C * T * P * P (e.g., 3*2*16*16=1536 for T=2)
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
            # Single embedder for both images and videos (original behavior)
            # Embedder input size: C * T * P * P (e.g., 3*1*16*16=768 for images, 3*2*16*16=1536 for video tubelets)
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

        # Register hooks for loading checkpoints
        if allow_checkpoint_without_temporal_compression and self.temporal_patch_dim > 1:
            if separate_video_embedder:
                # Create video_embedder weights from embedder weights if missing
                self.register_load_state_dict_pre_hook(self._state_dict_pre_hook_init_video_embedder)
            else:
                # Convert 2D embedder weights to 3D (original behavior)
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

        self.force_cpe_eval_mode = force_cpe_eval_mode  # Simluate eval mode for CPE code only
        if self.force_eval_mode:  # Eval mode for whole model
            self.eval()

    def train(self, mode: bool = True) -> 'RADIOViTModel':
        if mode and self.force_eval_mode:
            logging.getLogger(__name__).warning(
                "RADIOViTModel has force_eval_mode=True. Keeping model in eval mode."
            )
            return self
        return super().train(mode)

    def _state_dict_pre_hook_init_embedder(self, module, state_dict, prefix, *args, **kwargs):
        """Hook to convert 2D embedder weights (T=1) to 3D (T>1) by duplication.

        This is only called if allow_checkpoint_without_temporal_compression=True and
        temporal_patch_dim > 1. If the checkpoint already has 3D weights, this is a no-op.

        Args:
            module: The module being loaded into (same as self for bound methods)
            state_dict: The state dict being loaded
            prefix: Key prefix (e.g., 'vision_model.')
            *args: (local_metadata, strict, missing_keys, unexpected_keys)
        """
        key = prefix + "embedder.weight"
        if key not in state_dict:
            return  # Let normal error handling take over

        weight = state_dict[key]  # [hidden, input_dim]
        expected_2d_size = 3 * self.patch_dim * self.patch_dim  # C*P*P (768 for P=16)

        if weight.shape[1] == expected_2d_size:
            # Checkpoint is 2D (T=1), duplicate to 3D; this matches _apply_temporal_grouping()
            weight_3d = weight.repeat(1, self.temporal_patch_dim)
            # Normalize to maintain scale (averaging the duplicated weights). This ensures that if
            # we duplicate images and apply the (normalized) duplicated weights the output is the same
            weight_3d = weight_3d / self.temporal_patch_dim
            state_dict[key] = weight_3d
            logging.getLogger(__name__).info(
                f"Converted embedder weights from 2D ({weight.shape}) to 3D ({weight_3d.shape}) "
                f"for temporal_patch_dim={self.temporal_patch_dim}"
            )
        # else: weight already has correct shape (T>1 checkpoint), no action needed

    def _state_dict_pre_hook_init_video_embedder(self, module, state_dict, prefix, *args, **kwargs):
        """Hook to create video_embedder weights from image embedder weights if missing.

        This is only called if separate_video_embedder=True, allow_checkpoint_without_temporal_compression=True,
        and temporal_patch_dim > 1. If the checkpoint already has video_embedder weights, this is a no-op.

        The video embedder weights are created by duplicating the image embedder weights T times along the
        input dimension and normalizing, similar to _state_dict_pre_hook_init_embedder.

        Args:
            module: The module being loaded into (same as self for bound methods)
            state_dict: The state dict being loaded
            prefix: Key prefix (e.g., 'vision_model.')
            *args: (local_metadata, strict, missing_keys, unexpected_keys)
        """
        embedder_key = prefix + "embedder.weight"
        video_embedder_key = prefix + "video_embedder.weight"

        if embedder_key not in state_dict:
            return  # Let normal error handling take over

        embedder_weight = state_dict[embedder_key]  # [hidden, input_dim]
        expected_2d_size = 3 * self.patch_dim * self.patch_dim  # C*P*P (768 for P=16)
        expected_3d_size = 3 * self.temporal_patch_dim * self.patch_dim * self.patch_dim  # C*T*P*P

        # Check if video_embedder weights need to be created
        if video_embedder_key not in state_dict:
            # Video embedder weights don't exist in checkpoint
            if embedder_weight.shape[1] == expected_2d_size:
                # Embedder is 2D (image embedder), create video embedder by duplication
                video_weight = embedder_weight.repeat(1, self.temporal_patch_dim)
                # Normalize to maintain scale
                video_weight = video_weight / self.temporal_patch_dim
                state_dict[video_embedder_key] = video_weight
                logging.getLogger(__name__).info(
                    f"Created video_embedder weights ({video_weight.shape}) from embedder weights "
                    f"({embedder_weight.shape}) by duplicating and normalizing for "
                    f"temporal_patch_dim={self.temporal_patch_dim}"
                )
            elif embedder_weight.shape[1] == expected_3d_size:
                # Embedder is 3D (old combined embedder), use as video embedder and create image embedder
                # This handles the case of loading a checkpoint that used combined embedder (without separate_video_embedder)
                state_dict[video_embedder_key] = embedder_weight.clone()
                # Create 2D embedder by averaging the T duplicates
                embedder_2d = embedder_weight.view(embedder_weight.shape[0], self.temporal_patch_dim, -1).mean(dim=1)
                state_dict[embedder_key] = embedder_2d
                logging.getLogger(__name__).info(
                    f"Split combined 3D embedder weights ({embedder_weight.shape}) into "
                    f"image embedder ({embedder_2d.shape}) and video_embedder ({embedder_weight.shape})"
                )

        # Also handle bias if present
        embedder_bias_key = prefix + "embedder.bias"
        video_embedder_bias_key = prefix + "video_embedder.bias"
        if embedder_bias_key in state_dict and video_embedder_bias_key not in state_dict:
            # Bias is the same shape for both embedders (output dimension)
            state_dict[video_embedder_bias_key] = state_dict[embedder_bias_key].clone()
            logging.getLogger(__name__).info(
                f"Copied embedder bias to video_embedder bias"
            )

    def _apply_temporal_grouping(
        self,
        x: torch.Tensor,
        imgs_sizes: Union[List[Tuple[int, int]], torch.Tensor],
        num_frames: Union[List[int], torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams] = None,
        skip_image_duplication: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[List[Tuple[int, int]], torch.Tensor], List[int], Optional[PackedSeqParams], List[bool]]:
        """Group consecutive video frames into tubelets for temporal compression.

        For dynamic resolution, x is [1, total_patches, C*P*P]. This method:
        1. Splits patches by tile using imgs_sizes
        2. Groups consecutive video frames (T at a time)
        3. Concatenates patches along feature dim: [patches, C*P*P] -> [patches, C*T*P*P]
        4. For images (or if frames not divisible by T), duplicates frames as needed
           (unless skip_image_duplication=True, in which case images keep C*P*P)
        5. Updates packed_seq_params if provided

        Args:
            x: Input tensor [1, total_patches, C*P*P]
            imgs_sizes: List of (H, W) per tile, or tensor of shape [num_tiles, 2]
            num_frames: Number of frames per media item, or tensor of shape [num_tiles]
            packed_seq_params: Optional packed sequence params for attention
            skip_image_duplication: If True, images are not duplicated along the temporal dimension
                (they keep C*P*P instead of becoming C*T*P*P). Used with separate_video_embedder.

        Returns:
            x_grouped: Tensor [1, reduced_patches, C*T*P*P] (or mixed C*P*P and C*T*P*P if skip_image_duplication)
            new_imgs_sizes: Updated sizes (one per tubelet)
            new_num_frames: Updated frame counts (divided by T, rounded up)
            new_packed_seq_params: Updated packed_seq_params (or None if not provided)
            is_image: List of booleans, True for image media items, False for video tubelets
        """
        T = self.temporal_patch_dim

        # Validate: imgs_sizes should have one entry per frame (ungrouped)
        # RADIO does the temporal grouping, data loader should NOT pre-group
        # Expect 1 tubelet for images (nf == 1) and ceil(nf / T) tubelets for videos (nf > 1)
        total_frames = sum(num_frames)
        num_imgs_sizes = imgs_sizes.shape[0]
        expected_tubelets = sum(1 if nf == 1 else math.ceil(nf / T) for nf in num_frames)

        assert total_frames == num_imgs_sizes, (
            f"imgs_sizes must have one entry per frame (ungrouped). "
            f"Got {num_imgs_sizes} entries but sum(num_frames)={total_frames}. "
            f"Data appears {'pre-grouped by tubelets' if num_imgs_sizes == expected_tubelets else 'corrupted'}. "
            f"num_frames={num_frames}, T={T}, expected_tubelets={expected_tubelets}."
        )
        C_P_P = 3 * self.patch_dim * self.patch_dim  # Feature dim per patch (without temporal)

        # Convert imgs_sizes to list for easier manipulation
        if torch.is_tensor(imgs_sizes):
            imgs_sizes_list = [tuple(sz.tolist()) for sz in imgs_sizes]
        else:
            imgs_sizes_list = list(imgs_sizes)

        # Compute sequence lengths per frame
        seq_lens = [(h // self.patch_dim) * (w // self.patch_dim) for h, w in imgs_sizes_list]

        # Split x into per-frame chunks
        chunks = torch.split(x, seq_lens, dim=1)  # List of [1, patches_i, C*P*P]

        # Process each media item
        grouped_chunks = []
        new_imgs_sizes_list = []
        new_num_frames = []
        is_image = []  # Track which chunks are images vs videos
        tile_idx = 0

        for nf in num_frames:
            if nf == 1:
                # Single image
                chunk = chunks[tile_idx]  # [1, patches, C*P*P]
                if skip_image_duplication:
                    # Keep image as-is (C*P*P) for separate image embedder
                    grouped_chunks.append(chunk)
                else:
                    # Duplicate to create T copies (original behavior)
                    duplicated = chunk.repeat(1, 1, T)  # [1, patches, C*P*P] -> [1, patches, T*C*P*P]
                    grouped_chunks.append(duplicated)
                new_imgs_sizes_list.append(imgs_sizes_list[tile_idx])
                new_num_frames.append(1)
                is_image.append(True)
                tile_idx += 1
            else:
                # Video: group T consecutive frames
                # Pad to make divisible by T if needed
                padded_nf = nf if nf % T == 0 else nf + (T - nf % T)

                for group_start in range(0, padded_nf, T):
                    group_frames = []
                    for t in range(T):
                        frame_idx = group_start + t
                        if frame_idx < nf:
                            # Use actual frame
                            group_frames.append(chunks[tile_idx + frame_idx])
                        else:
                            # Duplicate last frame for padding
                            group_frames.append(chunks[tile_idx + nf - 1])

                    # All frames in a video should have same size
                    # Concatenate along feature dimension
                    # Each chunk is [1, patches, C*P*P]
                    grouped = torch.cat(group_frames, dim=-1)  # [1, patches, C*P*P] -> [1, patches, T*C*P*P]
                    grouped_chunks.append(grouped)
                    new_imgs_sizes_list.append(imgs_sizes_list[tile_idx])  # Adds one entry in place of T
                    is_image.append(False)

                new_num_frames.append(padded_nf // T)
                tile_idx += nf

        # Note: When skip_image_duplication=True, chunks have different feature dimensions
        # (C*P*P for images, C*T*P*P for videos). We cannot concatenate them directly.
        # Return grouped_chunks separately in this case, handled by caller.
        if skip_image_duplication:
            # Return list of chunks instead of concatenated tensor
            x_grouped = grouped_chunks
        else:
            # Concatenate all grouped chunks across seq dim (original behavior)
            x_grouped = torch.cat(grouped_chunks, dim=1)

        # Convert back to tensor if original was tensor
        if torch.is_tensor(imgs_sizes):
            new_imgs_sizes = torch.tensor(new_imgs_sizes_list, dtype=imgs_sizes.dtype, device=imgs_sizes.device)
        else:
            new_imgs_sizes = new_imgs_sizes_list

        # Update packed_seq_params if provided
        new_packed_seq_params = None
        if packed_seq_params is not None:
            # Compute new sequence lengths (patches per tubelet)
            seq_lens = [(h // self.patch_dim) * (w // self.patch_dim) for h, w in new_imgs_sizes_list]

            # Build cumulative sequence lengths
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

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        imgs_sizes: Optional[Union[List[Tuple[int, int]], torch.Tensor]] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        num_frames: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[List[Tuple[int, int]], torch.Tensor], List[int]]]:
        """Forward function of the RADIO ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w] or [1, total_patches, C*P*P] for dynamic res.
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.
            imgs_sizes (Union[List(Tuple[int, int]), torch.Tensor]): Sizes of the images, for dynamic resolution.
            packed_seq_params (PackedSeqParams): Packed sequence params for attention.
            num_frames (List[int]): Number of frames per media item. Images have 1 frame, videos have >1.
                Used for temporal compression to group consecutive video frames.

        Returns:
            If temporal_patch_dim > 1: Tuple of (x, imgs_sizes, num_frames) where imgs_sizes and num_frames
                are updated for temporal compression (fewer entries, one per tubelet instead of per frame).
            Otherwise: x (torch.Tensor): output after final transformer block of shape [b, s, h].
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

        # Apply temporal grouping for video frames (dynamic resolution only for now)
        is_image = None
        if self.temporal_patch_dim > 1:
            assert num_frames is not None, "num_frames is required when temporal_patch_dim > 1"
            if not self.dynamic_resolution:
                raise NotImplementedError("Temporal compression is only supported for dynamic resolution")

            x, imgs_sizes, num_frames, packed_seq_params, is_image = self._apply_temporal_grouping(
                x, imgs_sizes, num_frames, packed_seq_params,
                skip_image_duplication=self.separate_video_embedder,
            )

        # Apply embedder(s)
        if self.separate_video_embedder and self.temporal_patch_dim > 1:
            # x is a list of chunks with different feature dimensions
            # Apply appropriate embedder to each chunk
            embedded_chunks = []
            for chunk, is_img in zip(x, is_image):
                if is_img:
                    # Image: use image embedder (C*P*P -> hidden)
                    emb, _ = self.embedder(chunk)
                else:
                    # Video tubelet: use video embedder (C*T*P*P -> hidden)
                    emb, _ = self.video_embedder(chunk)
                embedded_chunks.append(emb)
            x = torch.cat(embedded_chunks, dim=1)  # [batch, total_seq_length, hidden_size]
        else:
            x, _ = self.embedder(x)  # [batch, seq_length, hidden_size]

        # in radio pos embedding added before class token
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
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]
            if self.dynamic_resolution:
                # TODO: Leverage pre-computed seq lengths from above
                out = []
                current_length = 0
                for input_size in imgs_sizes:
                    seq_length = input_size[0] // self.patch_dim * input_size[1] // self.patch_dim
                    out.append(class_token)
                    out.append(x[:, current_length : current_length + seq_length, :])
                    current_length += seq_length
                x = torch.cat(out, dim=1)
                # Not using += to avoid double adding in situations
                # where cu_seqlens_q and cu_seqlens_kv are same underlying tensor
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
                x = torch.cat(
                    [class_token, x], dim=1
                )  # [batch, seq_length + class_token_len, hidden_size]

        if not self.dynamic_resolution:
            assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"

        if self.ln_pre:
            x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        x = x.contiguous()

        x = self.decoder(x, attention_mask=attention_mask, packed_seq_params=packed_seq_params)

        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()

        if self.ln_post:
            x = self.ln_post(x)

        # Return updated metadata when temporal compression is active
        if self.temporal_patch_dim > 1:
            return x, imgs_sizes, num_frames
        else:
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

        pos_embed = self.position_embeddings.reshape(  # (B,L,C) -> (B,C,H,W)
            1, self.max_num_rows, self.max_num_cols, -1
        ).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:  # H
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:  # W
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        def aspect_ratio_select(pos_embed):
            (pos_H, pos_W) = pos_embed.shape[-2:]
            (input_H, input_W) = input_dims

            # If image is square, return full pos_embed to interpolate
            if input_H == input_W:
                return pos_embed

            # Crop the pos_embeds to produce same aspect ratio as original image
            (crop_H, crop_W) = (pos_H, pos_W)
            if input_W < input_H:
                crop_W = min(pos_W, math.ceil(pos_W * (input_W / input_H)))
            else:  # H < W
                crop_H = min(pos_H, math.ceil(pos_H * (input_H / input_W)))
            return pos_embed[..., : crop_H, : crop_W]

        if self.interpolate_only_cpe:
            if self.cpe_aspect_ratio_select:
                # Ensures the long edge is always mapped to 1 and the aspect ratio is maintained
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

                # Convert to [-1, 1] range
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
                    # If interpolate output tensor size (numel) >= INT_MAX, interpolate fails so we
                    #   have to take aspect-ratio crop first then upsample (req. for extreme aspect ratios)
                    #   e.g. image HxW = 23424x224 -> dims 1464x14 w/ B=1, C=1280 (real example)
                    # This can optionally be done everytime, but it tends to perform slightly worse
                    #   because we miss out on averaging the position along the border with the
                    #   values just outside the border. Recommend keeping cpe_aspect_ratio_select=False
                    #   and only doing this when required.
                    pos_embed = aspect_ratio_select(pos_embed)
                    pos_embed = F.interpolate(
                        pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
                    ).to(pos_embed.dtype)
                else:
                    # Normal CPE eval mode
                    pos_embed = F.interpolate(
                        pos_embed.float(), size=(max_dim, max_dim), align_corners=False, mode="bilinear"
                    ).to(pos_embed.dtype)
                    pos_embed = window_select(pos_embed)

        elif self.cpe_aspect_ratio_select:
            # NOTE: This produces the same result as `interpolate_only_cpe` + `cpe_aspect_ratio_select`
            #       But it's here to support aspect_ratio_select when has_cpe=False
            pos_embed = aspect_ratio_select(pos_embed)
        else:
            # Normal non-CPE mode
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear"
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed

def fp8_pad_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """FP8 requires class token length to be a multiple of 16 (for this model).

    Original model checkpoint may not be padded for FP8 so pad it here.
    """
    if not "vision_model.class_token" in state_dict:
        return

    class_token = state_dict["vision_model.class_token"]
    if class_token.shape[0] % 16 != 0:
        pad_len = 16 - (class_token.shape[0] % 16)
        pad_tensor = torch.randn(pad_len, class_token.shape[-1], dtype=class_token.dtype, device=class_token.device)
        class_token = torch.cat([pad_tensor, class_token], dim=0)
        state_dict["vision_model.class_token"] = class_token

    return
