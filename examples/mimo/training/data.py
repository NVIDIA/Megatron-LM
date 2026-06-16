# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Role-aware data iterator selection for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
import math
from typing import Optional

import torch

from examples.mimo.training.topology import HeteroTopology
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage


def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the MIMO data-selection args (stock extra_args_provider compatible).

    Only the mock provider is supported for now; energon/mistral backends are
    deferred to a later PR.
    """
    group = parser.add_argument_group("mimo data")
    group.add_argument(
        "--dataset-provider",
        choices=["mock"],
        default="mock",
        help="Which MIMO dataset backend to build (mock only for now).",
    )
    return parser


def select_data_iterator(args: argparse.Namespace, topology: HeteroTopology) -> Optional[object]:
    """Create the per-role data iterator this rank needs, or None if it consumes no data."""
    if getattr(args, "dataset_provider", "mock") != "mock":
        # Energon/mistral provider backends are deferred to a later PR.
        raise ValueError(f"unsupported dataset provider: {args.dataset_provider}")
    return select_mock_data_iterator(args, topology)


def select_mock_data_iterator(
    args: argparse.Namespace, topology: HeteroTopology
) -> Optional["MockVLMIterator"]:
    """Pick the mock iterator for this rank's role: encoder PP-first or language PP-edge stages."""
    encoder_name = _encoder_name(topology)
    llm_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    llm_pgc = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
    llm_mbs = args.micro_batch_size

    llm_needs_data = llm_grid.is_current_rank_in_grid() and (
        is_pp_first_stage(llm_pgc.pp) or is_pp_last_stage(llm_pgc.pp)
    )

    if encoder_name is None:
        if llm_needs_data:
            return MockVLMIterator(
                args, llm_mbs, encoder_name, get_mock_data_seed(args, llm_pgc, 100_000)
            )
        return None

    encoder_grid = topology.grids[encoder_name]
    encoder_pgc = topology.module_pgs[encoder_name]
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("micro_batch_size * llm_dp must be divisible by encoder_dp")
    encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp

    encoder_needs_data = encoder_grid.is_current_rank_in_grid() and is_pp_first_stage(
        encoder_pgc.pp
    )

    # A rank in the language grid always feeds the language batch (PP-edge stages); a pure-encoder
    # rank feeds the encoder batch. Colocated ranks (in both) follow the language schedule.
    if llm_needs_data:
        return MockVLMIterator(
            args, llm_mbs, encoder_name, get_mock_data_seed(args, llm_pgc, 100_000)
        )
    if encoder_needs_data:
        return MockVLMIterator(
            args, encoder_mbs, encoder_name, get_mock_data_seed(args, encoder_pgc, 0)
        )
    return None


def get_mock_data_seed(args: argparse.Namespace, pg_collection, module_seed_offset: int) -> int:
    """Seed mock data per DP lane so PP/TP stages in a lane see coherent batches."""
    dp_lane = pg_collection.dp.rank() if pg_collection.dp is not None else 0
    return args.seed + module_seed_offset + dp_lane


def _encoder_name(topology: HeteroTopology) -> Optional[str]:
    """Return the single modality (encoder) grid name, or None for a language-only run."""
    modality = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    return modality[0] if modality else None


def _even_patch_grid(num_patches: int) -> tuple[int, int]:
    """Factor ``num_patches`` into an even (rows, cols) patch grid.

    The dynamic-resolution RADIO path runs ``_pixel_shuffle_dynamic_res``, which
    reshapes each image's patches into ``(rows, cols, c)`` and then halves both
    axes (scale_factor=0.5). Both ``rows`` and ``cols`` must therefore be even.
    A near-square factorization keeps the synthetic aspect ratio sane and the CPE
    position-embedding interpolation well-conditioned.
    """
    assert num_patches % 4 == 0, (
        f"per-image patch count {num_patches} must be divisible by 4 so the "
        "pixel-shuffle (0.5x on each axis) produces an integer token count"
    )
    cols = int(math.isqrt(num_patches))
    while cols > 1 and (num_patches % cols != 0 or (num_patches // cols) % 2 != 0 or cols % 2 != 0):
        cols -= 1
    rows = num_patches // cols
    assert rows % 2 == 0 and cols % 2 == 0 and rows * cols == num_patches
    return rows, cols


class MockVLMIterator:
    """Infinite iterator yielding synthetic VLM-like next-token microbatches.

    Minimal self-contained mock so data selection is testable without a provider backend.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        micro_batch_size: int,
        encoder_name: Optional[str],
        seed: int,
    ) -> None:
        self.args = args
        self.micro_batch_size = micro_batch_size
        self.encoder_name = encoder_name
        self.image_seq_length = args.image_seq_length or args.seq_length // 2
        self.vision_encoder_key = getattr(args, "vision_encoder_key", "clip_encoder")
        # Pixel-input encoders (e.g. the Nemotron RADIO provider sets
        # ``vision_input_mode="pixels"``) take a raw image tensor as the
        # positional ``x`` arg of their wrapper's forward. Hidden-state encoders
        # (the CLIP/mock path) instead receive a precomputed ``hidden_states``
        # tensor. Feeding the wrong key/shape makes ``ModalitySubmodules.encode``
        # call the encoder with kwargs its forward never accepts, which aborts
        # the encoder forward before it reaches send_forward.
        self.vision_input_mode = getattr(args, "vision_input_mode", "hidden_states")
        self.dynamic_resolution = bool(getattr(args, "dynamic_resolution", False))
        self.patch_dim = getattr(args, "patch_dim", 16)
        self.num_image_tiles = getattr(args, "num_image_tiles", 1)
        self.dtype = torch.float32 if args.fp32 else torch.bfloat16
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(seed)
        if self.image_seq_length >= args.seq_length:
            raise ValueError("--image-seq-length must be smaller than --seq-length")

    def __iter__(self):
        return self

    def __next__(self):
        args = self.args
        image_tokens = torch.full(
            (self.micro_batch_size, self.image_seq_length),
            args.image_token_id,
            dtype=torch.long,
            device="cuda",
        )
        text_tokens = torch.randint(
            1,
            args.vocab_size,
            (self.micro_batch_size, args.seq_length - self.image_seq_length),
            device="cuda",
            generator=self.generator,
        )
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
        labels[(labels == args.image_token_id)] = -100
        loss_mask = (labels != -100).to(dtype=torch.float32)
        modality_inputs = {}
        if self.encoder_name is not None:
            if self.vision_input_mode == "pixels" and self.dynamic_resolution:
                # Dynamic-resolution RADIO (the Nemotron6-MoE VLM preset sets
                # ``dynamic_resolution=True``) does NOT accept a fixed-tile
                # (N, 3, H, W) pixel batch. Its forward expects the packed-patch
                # format produced by DynamicResolutionImageTilingStrategy.stack:
                #   x:                [1, total_patches, 3*p*p]   (pre-patchified)
                #   imgs_sizes:       (N_images, 2) int32 (H, W) in *pixels*
                #   packed_seq_params: THD PackedSeqParams over the per-image
                #                      patch counts (variable-length attention).
                # Feeding the fixed-tile (N,3,H,W) tensor with imgs_sizes=None
                # drives RADIOViTModel.forward down its dynamic branch with no
                # per-image splits, stalling/desyncing the encoder before it can
                # send_forward. Replicate the prototype's packed mock exactly.
                encoder_inputs = self._build_dynamic_resolution_pixels(args)
            elif self.vision_input_mode == "pixels":
                # Fixed-tile RADIO (``--no-dynamic-resolution``): the wrapper's
                # forward patchifies a raw (N, 3, img_h, img_w) image tensor.
                encoder_inputs = {
                    self.vision_encoder_key: {
                        "x": torch.randn(
                            self.micro_batch_size * self.num_image_tiles,
                            3,
                            args.img_h,
                            args.img_w,
                            device="cuda",
                            dtype=self.dtype,
                            generator=self.generator,
                        )
                    }
                }
            else:
                # Hidden-state encoders (CLIP/mock) consume a precomputed
                # encoder hidden-state tensor of shape (image_seq_length,
                # micro_batch_size, hidden_size).
                encoder_hidden_states = torch.randn(
                    self.image_seq_length,
                    self.micro_batch_size,
                    args.hidden_size,
                    device="cuda",
                    dtype=self.dtype,
                    generator=self.generator,
                )
                encoder_inputs = {
                    self.vision_encoder_key: {
                        "hidden_states": encoder_hidden_states,
                        "attention_mask": None,
                    }
                }
            modality_inputs[self.encoder_name] = encoder_inputs
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(args.seq_length, device="cuda")
            .unsqueeze(0)
            .expand(self.micro_batch_size, -1)
            .clone(),
            "modality_inputs": modality_inputs,
        }

    def _build_dynamic_resolution_pixels(self, args: argparse.Namespace) -> dict:
        """Build the packed-patch RADIO inputs for the dynamic-resolution path.

        Emits ``micro_batch_size * num_image_tiles`` synthetic images whose
        per-image RADIO output token count sums to exactly ``image_seq_length``
        per sample, so the count matches the ``image_seq_length`` placeholder
        image tokens MimoModel scatters embeddings into (a hard equality check
        in ``combine_embeddings``). RADIO drops ``class_token_len`` tokens per
        image and applies a 0.5x pixel shuffle on each axis (output = patches/4),
        so per-image patches = 4 * (image_seq_length / num_image_tiles).

        Returns the encoder-input dict with ``x``/``imgs_sizes``/
        ``packed_seq_params`` keyed under ``vision_encoder_key``, matching
        DynamicResolutionImageTilingStrategy.stack.
        """
        num_images = self.num_image_tiles
        if self.image_seq_length % num_images != 0:
            raise ValueError(
                f"image_seq_length ({self.image_seq_length}) must be divisible by "
                f"num_image_tiles ({num_images}) for the dynamic-resolution mock"
            )
        out_tokens_per_image = self.image_seq_length // num_images
        patches_per_image = out_tokens_per_image * 4  # undo 0.5x-per-axis pixel shuffle
        rows, cols = _even_patch_grid(patches_per_image)
        h_pix = rows * self.patch_dim
        w_pix = cols * self.patch_dim

        # Per-image patchified features: rearrange (c, py*p, px*p) -> (py*px, c*p*p).
        # The mock emits random patch features directly (shape-equivalent to the
        # tiling-strategy rearrange) since RADIO's embedder only sees the last dim.
        feat_dim = 3 * self.patch_dim * self.patch_dim
        total_patches = self.micro_batch_size * num_images * patches_per_image
        x = torch.randn(
            1,
            total_patches,
            feat_dim,
            device="cuda",
            dtype=self.dtype,
            generator=self.generator,
        )

        n_images_total = self.micro_batch_size * num_images
        # imgs_sizes and the packed-seq cu_seqlens/max_seqlen are *metadata*, not
        # data, and must live on CPU to match the prototype's energon tiling
        # strategy (DynamicResolutionImageTilingStrategy.stack builds plain CPU
        # tensors). RADIOViTModel.forward iterates imgs_sizes in Python and uses
        # the per-image patch counts as slice bounds and as the THD cu_seqlens fed
        # to TE attention; placing them on CUDA forces a blocking device->host
        # sync on every iteration (and TE expects int32 cu_seqlens on CPU), which
        # is the kind of degenerate host<->device round-trip that stalls the
        # encode before the transformer layers. Only ``x`` (the real activation)
        # is created on CUDA.
        imgs_sizes = torch.tensor(
            [[h_pix, w_pix]] * n_images_total, dtype=torch.int32
        )
        cu_seqlens = torch.arange(
            0,
            (n_images_total + 1) * patches_per_image,
            patches_per_image,
            dtype=torch.int32,
        )
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=torch.tensor(patches_per_image, dtype=torch.int32),
            max_seqlen_kv=torch.tensor(patches_per_image, dtype=torch.int32),
        )
        return {
            self.vision_encoder_key: {
                "x": x,
                "imgs_sizes": imgs_sizes,
                "packed_seq_params": packed_seq_params,
            }
        }
