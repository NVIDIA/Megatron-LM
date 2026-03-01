# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.

import torch
import torch.nn as nn

class VideoPatchEmbed(nn.Module):
    """
    3D Patch Embedding for Video Diffusion Transformers (DiT).

    Converts a video tensor of shape:
        [B, C, T, H, W]
    into a sequence of tokens:
        [B, N, D]

    where:
        N = (T // pt) * (H // ph) * (W // pw)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()

        # Validate patch_size
        if (
            not isinstance(patch_size, tuple)
            or len(patch_size) != 3
            or not all(isinstance(x, int) and x > 0 for x in patch_size)
        ):
            raise ValueError(f"patch_size must be a tuple of three positive integers, got {patch_size}")

        self.patch_size = patch_size

        # Conv3D performs linear projection over 3D patches
        self.proj: nn.Conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape [B, C, T, H, W]

        Returns:
            tokens: Tensor of shape [B, N, D]
        """
        # Input-dimension checks
        pt, ph, pw = self.patch_size
        T, H, W = x.shape[2], x.shape[3], x.shape[4]
        if T % pt != 0:
            raise ValueError(f"Input temporal dim T={T} not divisible by patch_size[0]={pt}")
        if H % ph != 0:
            raise ValueError(f"Input height H={H} not divisible by patch_size[1]={ph}")
        if W % pw != 0:
            raise ValueError(f"Input width W={W} not divisible by patch_size[2]={pw}")

        # [B, D, T', H', W']
        x = self.proj(x)

        # Flatten spatiotemporal dimensions into tokens
        x = x.flatten(2)        # [B, D, N]
        x = x.transpose(1, 2)   # [B, N, D]

        return x