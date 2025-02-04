# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from typing import Optional, Literal, Dict

import math
import numpy as np
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange



# ---------------------- Time Embedding -----------------------
class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the `st_conv` flag. This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches and embedding each
    patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    - keep_spatio (bool): If True, the spatial dimensions are kept separate in the output tensor, otherwise, they are flattened. Default: False.
    - st_conv (bool): If True, uses a 3D convolution to process spatial-temporal patches. Otherwise, uses a 2D convolution for spatial patches only.

    The module supports two modes of operation:
    1. Spatial-temporal convolution (`st_conv`=True): Applies a 3D convolution across the temporal and spatial dimensions.
    2. Spatial convolution (`st_conv`=False): Applies a 2D convolution only across the spatial dimensions, dividing `out_channels` by `temporal_patch_size`.

    The output shape of the module depends on the `keep_spatio` flag. If `keep_spatio`=True, the output retains the spatial dimensions.
    Otherwise, the spatial dimensions are flattened into a single dimension.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
        bias=True,
        keep_spatio=False,
        st_conv: bool = True,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.keep_spatio = keep_spatio
        self.st_conv = st_conv

        if st_conv:
            self.proj = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
                bias=bias,
            )
            temporal_patch_size = 1
        else:
            assert out_channels % temporal_patch_size == 0
            self.proj = nn.Conv2d(
                in_channels,
                out_channels // temporal_patch_size,
                kernel_size=spatial_patch_size,
                stride=spatial_patch_size,
                bias=bias,
            )

        if keep_spatio:
            self.out = Rearrange("b c (t r) h w -> b t h w (c r)", r=temporal_patch_size)
        else:
            self.out = Rearrange("b c (t r) h w -> b t (h w) (c r)", r=temporal_patch_size)
            
    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape determined by `keep_spatio` and `st_conv` flags.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        if self.st_conv:
            x = self.proj(x)
        else:
            x = self.proj(rearrange(x, "b c t h w -> (b t) c h w"))
            x = rearrange(x, "(b t) c h w -> b c t h w", t=T)
        return self.out(x)

class SDXLTimesteps(nn.Module):
    def __init__(self, num_channels: int = 320):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        in_dype = timesteps.dtype
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb.to(in_dype)


class SDXLTimestepEmbedding(nn.Module):
    def __init__(self, in_features, out_features, seed=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.linear_1.reset_parameters()
                self.linear_2.reset_parameters()

        setattr(self.linear_1.weight, "pipeline_parallel", True)
        setattr(self.linear_1.bias, "pipeline_parallel", True)
        setattr(self.linear_2.weight, "pipeline_parallel", True)
        setattr(self.linear_2.bias, "pipeline_parallel", True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.activation(sample)
        sample = self.linear_2(sample)

        return sample


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x

# ---------------------- Positional Embedding -----------------------

from torch import Tensor
from torch.distributed import ProcessGroup, all_gather, get_process_group_ranks, get_world_size


def split_inputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Split input tensor along the sequence dimension for checkpoint parallelism.

    This function divides the input tensor into equal parts along the specified
    sequence dimension, based on the number of ranks in the checkpoint parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        x: Input tensor to be split.
        seq_dim: The dimension along which to split the input (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A slice of the input tensor corresponding to the current rank.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    cp_ranks = get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)

    assert x.shape[seq_dim] % cp_size == 0, f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    x = x.view(*x.shape[:seq_dim], cp_size, x.shape[seq_dim] // cp_size, *x.shape[(seq_dim + 1) :])
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    # Note that the new sequence length is the original sequence length / cp_size
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x

class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.cp_group = None

    def enable_context_parallel(self, cp_group: ProcessGroup):
        self.cp_group = cp_group

    def disable_context_parallel(self):
        self.cp_group = None

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=Optional[torch.Tensor]) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split. It delegates the embedding generation to generate_embeddings function.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self.cp_group is not None:
            cp_ranks = get_process_group_ranks(self.cp_group)
            cp_size = len(cp_ranks)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size, H, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps)
        if self.cp_group is not None:
            embeddings = split_inputs_cp(x=embeddings, seq_dim=1, cp_group=self.cp_group)
        return embeddings

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]):
        raise NotImplementedError


class PosEmb3D:
    def __init__(self, *, max_t=96, max_h=960, max_w=960):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        self.grid = torch.stack(torch.meshgrid(
            torch.arange(self.max_t, device='cpu'),
            torch.arange(self.max_h, device='cpu'), 
            torch.arange(self.max_w, device='cpu'), 
            ), dim=-1)
    
    def get_pos_id_3d(self, *, t, h, w):
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[ :t, :h, :w]
    
class SinCosPosEmb3D(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        is_learnable: bool = False,
        interpolation: Literal["crop", "resize", "crop_resize"] = "crop",
        spatial_interpolation_scale=1.0,
        temporal_interpolation_scale=1.0,
        init_length_for_resize: int = 16,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): "crop", "resize", "crop_resize". "crop" means we crop the positional embedding to the length of the input sequence. "resize" means we resize the positional embedding to the length of the input sequence. "crop_resize" (inference only) means we first crop the positional embedding to init_length_for_resize, then resize it to the length of the input sequence.
            init_length_for_resize (int): used when interpolation is "crop_resize", where we "resize" embedding during inference for model trained with "crop". We first "crop" the pos_embed to this length (used during training), then run the "resize", default 16
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        self.init_length_for_resize = init_length_for_resize
        param = get_3d_sincos_pos_embed(
            model_channels, len_h, len_w, len_t, spatial_interpolation_scale, temporal_interpolation_scale
        )
        param = rearrange(param, "(h w t) c -> 1 t h w c", h=len_h, w=len_w)
        if is_learnable:
            self.pos_embed = nn.Parameter(
                torch.from_numpy(param).float(),
            )
        else:
            self.register_buffer("pos_embed", torch.from_numpy(param).float(), persistent=False)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor]) -> torch.Tensor:
        B, T, H, W, C = B_T_H_W_C
        if self.interpolation == "crop":
            return self.pos_embed[:, :T, :H, :W]
        if self.interpolation == "resize":
            return rearrange(
                F.interpolate(
                    rearrange(self.pos_embed, "1 t h w c -> 1 c h w t"),
                    size=(H, W, T),
                    mode="linear",
                    align_corners=False,
                ),
                "1 c h w t -> 1 t h w c",
            )
        if self.interpolation == "crop_resize":
            pos_embed_crop = self.pos_embed[:, : self.init_length_for_resize, :H, :W]  # B,T,H,W,C
            _, t, h, w, c = pos_embed_crop.shape

            pos_embed_crop_resize_t = rearrange(
                F.interpolate(
                    rearrange(pos_embed_crop, "1 t h w c -> 1 (c h w) t"),
                    size=(T),
                    mode="linear",
                ),
                "1 (c h w) t -> 1 t h w c",
                c=c,
                h=h,
                w=w,
            )
            pos_embed_crop_resize = rearrange(
                F.interpolate(
                    rearrange(pos_embed_crop_resize_t, "1 t h w c -> 1 (c t) h w"),
                    size=(H, W),
                    mode="bilinear",
                ),
                "1 (c t) h w -> 1 t h w c",
                c=c,
            )
            return pos_embed_crop_resize

        raise ValueError(f"Unknown interpolation method {self.interpolation}")



def get_3d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, grid_size_t, spatial_interpolation_scale, temporal_interpolation_scale, concat=True):
    grid_h = np.arange(grid_size_h, dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(grid_size_w, dtype=np.float32) / spatial_interpolation_scale
    grid_t = np.arange(grid_size_t, dtype=np.float32) / temporal_interpolation_scale

    grid = np.meshgrid(grid_w, grid_h, grid_t, indexing="ij")
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(3, 1, grid_size_h, grid_size_w, grid_size_t)

    if concat:
        per_axis = embed_dim // 3
        per_axis = (per_axis // 2) * 2  # make it even (for sin/cos split)
        dim_h, dim_w = per_axis, per_axis
        dim_t = embed_dim - dim_h - dim_w
        emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[0])  # (H*W, D/3)
        emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[1])  # (H*W, D/3)
        emb_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid[2])  # (H*W, D/3)

        return np.concatenate([emb_h, emb_w, emb_t], axis=1)  # (H*W*T, D)
    else:
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim, grid[0])  # (H*W)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim, grid[1])  # (H*W)
        emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim, grid[2])  # (H*W)

        return emb_h + emb_w + emb_t  # (H*W*T, D)

# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#-------------------------Official DiT's embedding----------------------
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob, seed=None):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.seed = seed
        if seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.embedding_table.reset_parameters()

        setattr(self.embedding_table.weight, "pipeline_parallel", True)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class TimestepLabelEmbedder(nn.Module):
    """
    Embedder module which holds all the embedder together, for we need to do the grad reduce according to the parameters name with 't_embedder'.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob, seed=None):
        super().__init__()
        self.label_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob, seed=seed)
        self.time_embedder = torch.nn.Sequential(
            SDXLTimesteps(hidden_size),
            SDXLTimestepEmbedding(hidden_size, hidden_size, seed=seed),
        )

    def forward(self, timestep, labels):
        t_B_D = self.time_embedder(timestep.flatten())                   
        y_B_D = self.label_embedder(labels.flatten(), self.training)    
        c_B_D = t_B_D + y_B_D  
        return c_B_D 