# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.

import torch
from megatron.core.vision.video_patch_embed import VideoPatchEmbed


def test_video_patch_embed_shapes():
    embed = VideoPatchEmbed(
        in_channels=3,
        embed_dim=64,
        patch_size=(2, 16, 16),
    )

    x = torch.randn(2, 3, 8, 64, 64)
    out = embed(x)

    # (8/2) * (64/16) * (64/16) = 4 * 4 * 4 = 64 tokens
    assert out.shape == (2, 64, 64)


def test_video_patch_embed_non_divisible():
    embed = VideoPatchEmbed(
        in_channels=3,
        embed_dim=32,
        patch_size=(2, 16, 16),
    )
    # T=7 not divisible by 2, H=63 not divisible by 16, W=64 is fine
    x = torch.randn(1, 3, 7, 63, 64)
    try:
        embed(x)
    except ValueError as e:
        assert "not divisible" in str(e)
    else:
        assert False, "Expected ValueError for non-divisible input dims"


def test_video_patch_embed_various_configs():
    configs = [
        ((2, 8, 8), 32),
        ((4, 16, 16), 128),
        ((1, 32, 32), 16),
    ]
    for patch_size, embed_dim in configs:
        embed = VideoPatchEmbed(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        T, H, W = 8, 64, 64
        pt, ph, pw = patch_size
        x = torch.randn(2, 3, T, H, W)
        out = embed(x)
        N = (T // pt) * (H // ph) * (W // pw)
        assert out.shape == (2, N, embed_dim)


def test_video_patch_embed_gradient_flow():
    embed = VideoPatchEmbed(
        in_channels=3,
        embed_dim=16,
        patch_size=(2, 8, 8),
    )
    x = torch.randn(2, 3, 4, 16, 16, requires_grad=True)
    out = embed(x)
    loss = out.sum()
    loss.backward()
    for name, param in embed.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"