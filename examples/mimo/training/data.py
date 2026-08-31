# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Role-aware external DataLoaders for heterogeneous MIMO mock training."""

from __future__ import annotations

import argparse
from math import isqrt
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from examples.mimo.training.topology import HeteroTopology
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_pg_rank

_ENCODER_SEED_OFFSET = 10_000
_LANGUAGE_SEED_OFFSET = 20_000
_SPLIT_SEED_OFFSETS = (0, 100_000, 200_000)


def add_mock_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the mock-dataset arguments consumed by this module's loaders."""
    group = parser.add_argument_group("mimo mock data")
    group.add_argument("--dataset-provider", choices=("mock",), default="mock")
    group.add_argument("--image-token-id", type=int, default=511)
    group.add_argument("--image-seq-length", type=int, default=None)
    group.add_argument("--mock-dataset-size", type=int, default=10_000)
    return parser


def _dynamic_patch_grid(num_patches: int, require_even: bool) -> tuple[int, int]:
    """Factor a patch budget into the nearest-to-square valid grid."""
    for rows in range(isqrt(num_patches), 0, -1):
        if num_patches % rows:
            continue
        cols = num_patches // rows
        if require_even and (rows % 2 or cols % 2):
            continue
        return rows, cols
    qualifier = " even-by-even" if require_even else ""
    raise ValueError(f"cannot factor {num_patches} input patches into a{qualifier} patch grid")


class _MockVLMDataset(Dataset):
    """Synthetic samples matching the heterogeneous Nemotron RADIO VLM input schema."""

    def __init__(
        self,
        *,
        size: int,
        seq_len: int,
        image_seq_length: int,
        vocab_size: int,
        pad_token_id: int,
        image_token_id: int,
        encoder_name: Optional[str],
        seed: int,
        dtype: torch.dtype,
        dynamic_resolution: bool,
        patch_dim: int,
        img_h: int,
        img_w: int,
        pixel_shuffle: bool,
        num_image_tiles: int,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.image_seq_length = image_seq_length
        self.image_token_id = image_token_id
        self.encoder_name = encoder_name
        self.seed = seed
        self.dtype = dtype
        self.dynamic_resolution = dynamic_resolution
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        self.pixel_shuffle = pixel_shuffle
        self.num_image_tiles = num_image_tiles

        if self.seq_len <= self.image_seq_length:
            raise ValueError(
                f"image_seq_length ({self.image_seq_length}) must be less than "
                f"seq_len ({self.seq_len})"
            )
        if self.patch_dim <= 0:
            raise ValueError(f"patch_dim must be positive, got {self.patch_dim}")
        if self.num_image_tiles <= 0:
            raise ValueError(f"num_image_tiles must be positive, got {self.num_image_tiles}")

        self._text_token_ids = torch.arange(1, vocab_size, dtype=torch.long)
        self._text_token_ids = self._text_token_ids[
            (self._text_token_ids != self.image_token_id) & (self._text_token_ids != pad_token_id)
        ]

        if self.dynamic_resolution:
            if self.image_seq_length % self.num_image_tiles:
                raise ValueError(
                    f"image_seq_length ({self.image_seq_length}) must be divisible by "
                    f"num_image_tiles ({self.num_image_tiles})"
                )
            emitted_per_tile = self.image_seq_length // self.num_image_tiles
            patches_per_tile = emitted_per_tile * (4 if self.pixel_shuffle else 1)
            self.patch_rows, self.patch_cols = _dynamic_patch_grid(
                patches_per_tile, require_even=self.pixel_shuffle
            )
        else:
            if self.img_h % self.patch_dim or self.img_w % self.patch_dim:
                raise ValueError(
                    f"img_h ({self.img_h}) and img_w ({self.img_w}) must be divisible by "
                    f"patch_dim ({self.patch_dim})"
                )
            self.patch_rows = self.img_h // self.patch_dim
            self.patch_cols = self.img_w // self.patch_dim

        if self.encoder_name is not None and not self.dynamic_resolution:
            if self.pixel_shuffle and self.patch_rows != self.patch_cols:
                raise ValueError(
                    "fixed-resolution RADIO pixel shuffle requires a square patch grid, "
                    f"got {self.patch_rows}x{self.patch_cols}"
                )
            if self.pixel_shuffle and (self.patch_rows % 2 or self.patch_cols % 2):
                raise ValueError(
                    "pixel shuffle requires an even patch grid in both dimensions, "
                    f"got {self.patch_rows}x{self.patch_cols}"
                )
            patches = self.num_image_tiles * self.patch_rows * self.patch_cols
            emitted_tokens = patches // 4 if self.pixel_shuffle else patches
            if self.image_seq_length != emitted_tokens:
                raise ValueError(
                    f"fixed-resolution mode emits {emitted_tokens} image tokens, "
                    f"got image_seq_length={self.image_seq_length}"
                )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, object]:
        input_ids = self._mock_tokenize(idx)
        labels = torch.full_like(input_ids, -100)
        labels[:-1] = input_ids[1:]
        labels[labels == self.image_token_id] = -100
        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": (labels != -100).float(),
            "position_ids": torch.arange(len(input_ids), dtype=torch.long),
            "modality_inputs": {},
        }
        if self.encoder_name is not None:
            sample["modality_inputs"] = {
                self.encoder_name: {self.encoder_name: self._encoder_inputs()}
            }
        return sample

    def _mock_tokenize(self, idx: int) -> torch.Tensor:
        image_tokens = torch.full((self.image_seq_length,), self.image_token_id, dtype=torch.long)
        num_text_tokens = self.seq_len - self.image_seq_length
        if num_text_tokens and self._text_token_ids.numel() == 0:
            raise ValueError(
                "vocab_size must contain at least one non-padding token distinct from "
                "image_token_id"
            )
        generator = torch.Generator().manual_seed(self.seed + idx)
        choices = torch.randint(
            self._text_token_ids.numel(), (num_text_tokens,), generator=generator, dtype=torch.long
        )
        return torch.cat((image_tokens, self._text_token_ids[choices]), dim=0)

    def _encoder_inputs(self) -> dict[str, torch.Tensor]:
        if not self.dynamic_resolution:
            return {
                "x": torch.zeros(self.num_image_tiles, 3, self.img_h, self.img_w, dtype=self.dtype)
            }

        patches_per_tile = self.patch_rows * self.patch_cols
        return {
            "x": torch.zeros(
                1, self.num_image_tiles * patches_per_tile, 3 * self.patch_dim**2, dtype=self.dtype
            ),
            "imgs_sizes": torch.tensor(
                [[self.patch_rows * self.patch_dim, self.patch_cols * self.patch_dim]]
                * self.num_image_tiles,
                dtype=torch.int32,
            ),
        }


def _build_mock_vlm_dataloader(
    *,
    batch_size: int,
    dataset_size: int,
    seq_len: int,
    image_seq_length: int,
    vocab_size: int,
    pad_token_id: int,
    image_token_id: int,
    encoder_name: Optional[str],
    seed: int,
    dtype: torch.dtype,
    dynamic_resolution: bool,
    patch_dim: int,
    img_h: int,
    img_w: int,
    pixel_shuffle: bool,
    num_image_tiles: int,
) -> DataLoader:
    """Create synthetic data matching the heterogeneous Nemotron RADIO VLM input schema."""
    dataset = _MockVLMDataset(
        size=dataset_size,
        seq_len=seq_len,
        image_seq_length=image_seq_length,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        image_token_id=image_token_id,
        encoder_name=encoder_name,
        seed=seed,
        dtype=dtype,
        dynamic_resolution=dynamic_resolution,
        patch_dim=patch_dim,
        img_h=img_h,
        img_w=img_w,
        pixel_shuffle=pixel_shuffle,
        num_image_tiles=num_image_tiles,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_mock_batch
    )


def _collate_mock_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    collated = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "loss_mask": torch.stack([item["loss_mask"] for item in batch]),
        "position_ids": torch.stack([item["position_ids"] for item in batch]),
        "modality_inputs": {},
    }
    for modality_name, encoders in batch[0]["modality_inputs"].items():
        collated["modality_inputs"][modality_name] = {}
        for encoder_name in encoders:
            encoder_items = [item["modality_inputs"][modality_name][encoder_name] for item in batch]
            x = encoder_items[0]["x"]
            encoder_batch = {
                "x": torch.cat([item["x"] for item in encoder_items], dim=1 if x.ndim == 3 else 0)
            }
            if "imgs_sizes" in encoder_items[0]:
                imgs_sizes = torch.cat([item["imgs_sizes"] for item in encoder_items])
                patch_dim = isqrt(x.shape[-1] // 3)
                if 3 * patch_dim**2 != x.shape[-1]:
                    raise ValueError(
                        f"dynamic encoder feature size ({x.shape[-1]}) is not 3 * patch_dim^2"
                    )
                seq_lens = torch.prod(imgs_sizes // patch_dim, dim=-1, dtype=torch.int32)
                cu_seqlens = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32),
                        torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
                    )
                )
                max_seqlen = int(seq_lens.max().item())
                encoder_batch.update(
                    {
                        "imgs_sizes": imgs_sizes,
                        "packed_seq_params": PackedSeqParams(
                            qkv_format="thd",
                            cu_seqlens_q=cu_seqlens,
                            cu_seqlens_kv=cu_seqlens.clone(),
                            max_seqlen_q=max_seqlen,
                            max_seqlen_kv=max_seqlen,
                        ),
                    }
                )
            collated["modality_inputs"][modality_name][encoder_name] = encoder_batch
    return collated


def build_train_valid_test_data_loaders(
    args: argparse.Namespace, topology: HeteroTopology
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build independent mock DataLoaders for the data-consuming rank role."""
    if getattr(args, "dataset_provider", "mock") != "mock":
        raise ValueError(f"unsupported dataset provider: {args.dataset_provider}")

    encoder_name = _encoder_name(topology)
    if encoder_name is not None and (args.micro_batch_size * args.llm_dp) % args.encoder_dp:
        raise ValueError("micro_batch_size * llm_dp must be divisible by encoder_dp")

    language_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    language_pgc = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
    language_needs_data = language_grid.is_current_rank_in_grid() and (
        is_pp_first_stage(language_pgc.pp) or is_pp_last_stage(language_pgc.pp)
    )

    encoder_needs_data = False
    encoder_pgc = None
    if encoder_name is not None:
        encoder_pgc = topology.module_pgs[encoder_name]
        rank_in_encoder = topology.grids[encoder_name].is_current_rank_in_grid()
        if rank_in_encoder and not getattr(args, "disable_vision_class_token", False):
            raise ValueError("RADIO mock data requires --disable-vision-class-token")
        encoder_needs_data = rank_in_encoder and is_pp_first_stage(encoder_pgc.pp)

    if encoder_needs_data and language_needs_data:
        raise ValueError("the external DataLoader adapter requires non-colocated module grids")
    if encoder_needs_data:
        encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp
        return _build_split_loaders(
            args,
            batch_size=encoder_mbs,
            pg_collection=encoder_pgc,
            module_seed_offset=_ENCODER_SEED_OFFSET,
            encoder_name=encoder_name,
        )
    if language_needs_data:
        return _build_split_loaders(
            args,
            batch_size=args.micro_batch_size,
            pg_collection=language_pgc,
            module_seed_offset=_LANGUAGE_SEED_OFFSET,
            encoder_name=None,
        )
    return (None, None, None)


def _build_split_loaders(
    args: argparse.Namespace,
    *,
    batch_size: int,
    pg_collection,
    module_seed_offset: int,
    encoder_name: Optional[str],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build split-local datasets with deterministic module/DP/split seeds."""
    base_seed = args.seed + module_seed_offset + get_pg_rank(pg_collection.dp)
    common = _mock_loader_kwargs(args, encoder_name)
    return tuple(
        _build_mock_vlm_dataloader(
            batch_size=batch_size,
            dataset_size=getattr(args, "mock_dataset_size", 10_000),
            seed=base_seed + split_offset,
            **common,
        )
        for split_offset in _SPLIT_SEED_OFFSETS
    )


def _mock_loader_kwargs(args: argparse.Namespace, encoder_name: Optional[str]) -> dict:
    """Translate parsed training arguments to the reusable mock loader."""
    seq_len = args.seq_length
    dtype = getattr(args, "params_dtype", None)
    if dtype is None:
        dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float32

    image_size = getattr(args, "image_size", 224)
    img_h = getattr(args, "img_h", image_size)
    img_w = getattr(args, "img_w", image_size)
    patch_dim = getattr(args, "patch_dim", 16)
    num_image_tiles = getattr(args, "num_image_tiles", 1)
    pixel_shuffle = bool(getattr(args, "pixel_shuffle", False))
    dynamic_resolution = bool(getattr(args, "dynamic_resolution", False))
    image_seq_length = getattr(args, "image_seq_length", None)
    if image_seq_length is None:
        image_seq_length = (
            seq_len // 2
            if dynamic_resolution
            else _fixed_image_seq_length(img_h, img_w, patch_dim, num_image_tiles, pixel_shuffle)
        )

    return {
        "seq_len": seq_len,
        "image_seq_length": image_seq_length,
        "vocab_size": args.vocab_size,
        "pad_token_id": getattr(args, "pad_token_id", 0),
        "image_token_id": args.image_token_id,
        "encoder_name": encoder_name,
        "dtype": dtype,
        "dynamic_resolution": dynamic_resolution,
        "patch_dim": patch_dim,
        "img_h": img_h,
        "img_w": img_w,
        "pixel_shuffle": pixel_shuffle,
        "num_image_tiles": num_image_tiles,
    }


def _fixed_image_seq_length(
    img_h: int, img_w: int, patch_dim: int, num_image_tiles: int, pixel_shuffle: bool
) -> int:
    """Derive fixed-resolution RADIO output tokens from image geometry."""
    if patch_dim <= 0 or img_h % patch_dim or img_w % patch_dim:
        raise ValueError("fixed RADIO image dimensions must be divisible by patch_dim")
    patches = num_image_tiles * (img_h // patch_dim) * (img_w // patch_dim)
    return patches // 4 if pixel_shuffle else patches


def _encoder_name(topology: HeteroTopology) -> Optional[str]:
    """Return the example's optional single encoder module name."""
    names = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    if len(names) > 1:
        raise ValueError("this example's mock data supports at most one encoder module")
    return names[0] if names else None
