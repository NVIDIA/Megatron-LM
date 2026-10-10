# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command-line arguments for two-tower diffusion training.

Registers ``--tt-diffusion-*`` flags that configure the two-tower
block-wise diffusion architecture on top of a Mamba-hybrid backbone.
Called by ``pretrain_mamba_tt_diffusion.py`` via ``extra_args_provider``.
"""


def add_two_tower_diffusion_args(parser):
    """Register two-tower diffusion arguments with *parser*.

    All flags are prefixed with ``--tt-diffusion-`` to avoid collisions with
    the base Mamba/Megatron argument namespace.

    Args:
        parser (argparse.ArgumentParser): Megatron argument parser.

    Returns:
        argparse.ArgumentParser: The same parser, with a new argument group added.
    """
    group = parser.add_argument_group(title="Two-Tower Diffusion")

    group.add_argument(
        "--tt-diffusion",
        default=False,
        action="store_true",
        help="Use Two-Tower architecture for block-wise diffusion training.",
    )
    group.add_argument(
        "--tt-diffusion-block-size",
        type=int,
        default=1,
        help="Tokens per diffusion block. All MambaMixer layers have their "
        "chunk_size overridden to this value so block and chunk boundaries "
        "align for state extraction.",
    )
    group.add_argument(
        "--tt-diffusion-tied-towers",
        action="store_true",
        default=False,
        help="Share weights between context and denoiser towers.",
    )
    group.add_argument(
        "--tt-diffusion-no-freeze-context",
        action="store_true",
        default=False,
        help="Train the context tower body (default: frozen). "
        "When False the context tower is frozen and only the denoiser "
        "tower receives gradients.",
    )
    group.add_argument(
        "--tt-diffusion-time-conditioning",
        action="store_true",
        default=False,
        help="Enable PixArt-alpha adaLN-single timestep conditioning on "
        "the denoiser tower. Adds a TimestepEmbedder, a global modulation "
        "MLP, and per-layer scale/shift/gate tables.",
    )
    group.add_argument(
        "--tt-diffusion-bidirectional-mamba",
        action="store_true",
        default=False,
        help="Run each Mamba layer in the denoiser tower bidirectionally. "
        "A second reversed-sequence SSM pass with zero initial states is "
        "averaged with the forward output. Context tower is unaffected.",
    )
    group.add_argument(
        "--tt-diffusion-context-ar-loss",
        action="store_true",
        default=False,
        help="Add a next-token-prediction loss from the context tower. "
        "Requires --tt-diffusion-tied-towers and --tt-diffusion-no-freeze-context "
        "so the shared tower body and context output head both receive gradients.",
    )
    group.add_argument(
        "--tt-diffusion-mask-token-id",
        type=int,
        default=0,
        help="Token ID used for masking in mask_diffusion mode.",
    )
    return parser
