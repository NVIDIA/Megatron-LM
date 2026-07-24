# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed-window mock image-text data for Qwen3.5-VL training.

Generates deterministic fixed-length packed THD windows from a
configurable short/long document mixture (document-count weights), with
text-only documents and length-scaled multimodal image density.

One dataset item is one full ``seq_length``-token training window sliced
from a mock document stream planned by
:class:`~examples.multimodal_dev.data.packed_window_plan.PackedWindowPlanGenerator`
(the same plan source the CPU calibration simulator uses). This module is
a token/pixel adapter over that plan; window-level statistics (segments
per window, image counts, vision share) are emergent from the document
layer and are measured, not configured.

The generic text-only ``MockVarlenDataset`` cannot transport ragged vision
payloads through the core packing scheduler, so this provider keeps the
raw per-sample contract and leaves multimodal packing to
``multimodal_dev.forward_step.pack_or_pad_batch``. Fixed-shape single-image
scenarios are served by ``--dataset-provider mock`` instead.
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from examples.multimodal_dev.models.qwen35_vl.configuration import (
    QWEN35_VL_IMAGE_TOKEN_ID,
    QWEN35_VL_VIDEO_TOKEN_ID,
    QWEN35_VL_VISION_START_TOKEN_ID,
)
from megatron.training.datasets.utils import load_json_arg

_MAX_TORCH_SEED = 2**63 - 1

_TEXT_TOKEN_STREAM = 3
_PIXEL_VALUE_STREAM = 5


def _seed_sequence(seed: int, idx: int, stream: int, item: int = 0) -> np.random.SeedSequence:
    """Return an access-order-independent RNG namespace for one sample stream."""
    return np.random.SeedSequence([int(seed), int(idx), int(stream), int(item)])


class PackedWindowQwen35VLDataset(Dataset):
    """Fixed-length packed windows sliced from a mock document stream.

    Contract (six fields): ``input_ids``/``labels``/``loss_mask`` of shape
    ``[seq_length]``, ``pixel_values [total_raw_patches, pixel_dim]``,
    ``image_grid_thw [num_images, 3]``, and ``seq_lens [num_segments]``
    with ``seq_lens.sum() == seq_length``. Labels are next-token targets;
    each segment's final position is ``-100`` (no cross-document
    prediction), as are targets that land on image or vision-start tokens.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        seq_length: int,
        window_config: dict[str, Any],
        seed: int = 1234,
        vocab_size: int = 248320,
        image_token_id: int = QWEN35_VL_IMAGE_TOKEN_ID,
        video_token_id: int = QWEN35_VL_VIDEO_TOKEN_ID,
        vision_start_token_id: int = QWEN35_VL_VISION_START_TOKEN_ID,
        image_size_config: dict[str, Any] | None = None,
        max_raw_patches_per_window: int | None = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
    ) -> None:
        from examples.multimodal_dev.data.packed_window_plan import PackedWindowPlanGenerator

        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
        if patch_size <= 0 or temporal_patch_size <= 0 or spatial_merge_size <= 0:
            raise ValueError(
                "patch_size, temporal_patch_size, and spatial_merge_size must be positive."
            )
        if (
            not isinstance(image_size_config, dict)
            or image_size_config.get("mode") != "buckets"
            or not image_size_config.get("resolutions")
        ):
            raise ValueError(
                "packed_window mode requires --mock-image-size-config-json with "
                '{"mode":"buckets","resolutions":[...]} (optional "weights").'
            )

        import numbers

        block = patch_size * spatial_merge_size
        grids: list[tuple[int, int, int]] = []
        merged_tokens: list[int] = []
        raw_patches: list[int] = []
        for index, resolution in enumerate(image_size_config["resolutions"]):
            if (
                not isinstance(resolution, (list, tuple))
                or len(resolution) != 2
                or not all(
                    isinstance(side, numbers.Integral) and not isinstance(side, bool)
                    for side in resolution
                )
                or not all(int(side) > 0 for side in resolution)
            ):
                raise ValueError(
                    f"Bucket resolution at index {index} must be exactly two positive "
                    f"integers [height, width]; got {resolution!r}."
                )
            height, width = int(resolution[0]), int(resolution[1])
            if height % block or width % block:
                raise ValueError(
                    f"Bucket resolution {height}x{width} must be divisible by "
                    f"patch_size*spatial_merge_size={block}."
                )
            grid_h, grid_w = height // patch_size, width // patch_size
            grids.append((1, grid_h, grid_w))
            merged_tokens.append((grid_h // spatial_merge_size) * (grid_w // spatial_merge_size))
            raw_patches.append(grid_h * grid_w)
        weights = image_size_config.get("weights") or [1.0] * len(grids)
        if len(weights) != len(grids):
            raise ValueError(
                f"Bucket 'weights' must match 'resolutions' in length; got "
                f"{len(weights)} weights for {len(grids)} resolutions."
            )

        special_ids = {image_token_id, video_token_id, vision_start_token_id}
        # Token ID 0 is reserved for packing padding; a special ID of 0 could
        # be miscounted as an image placeholder after collate padding.
        if any(not 0 < token_id < vocab_size for token_id in special_ids):
            raise ValueError(
                f"All multimodal token IDs must be in [1, vocab_size={vocab_size}); "
                f"got {sorted(special_ids)}."
            )
        self.safe_text_token_id = next(
            (token_id for token_id in range(1, vocab_size) if token_id not in special_ids), None
        )
        if self.safe_text_token_id is None:
            raise ValueError("vocab_size does not contain a usable non-special text token ID.")

        self.num_samples = int(num_samples)
        self.seq_length = int(seq_length)
        self.seed = int(seed)
        self.vocab_size = int(vocab_size)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.vision_start_token_id = int(vision_start_token_id)
        self.special_ids = special_ids
        self.grids = grids
        self.pixel_dim = 3 * temporal_patch_size * patch_size * patch_size
        if max_raw_patches_per_window is not None and int(max_raw_patches_per_window) <= 0:
            raise ValueError(
                "max_raw_patches_per_window must be a positive integer or None "
                f"(0 does not silently disable it); got {max_raw_patches_per_window!r}."
            )
        self.max_raw_patches_per_window = (
            int(max_raw_patches_per_window) if max_raw_patches_per_window is not None else None
        )
        # The plan pool bounds construction time and memory independently of
        # the virtual dataset length Megatron requests for the full training
        # schedule: indices wrap onto the pool for the window LAYOUT while
        # token/pixel content stays keyed by the virtual index.
        window_config = dict(window_config)
        pool_windows = int(window_config.pop("plan_pool_windows", 2048))
        if pool_windows <= 0:
            raise ValueError(f"plan_pool_windows must be positive, got {pool_windows}.")
        self.plan_pool_windows = min(pool_windows, num_samples) if num_samples else 0
        self.plan = (
            PackedWindowPlanGenerator(
                seq_length=seq_length,
                num_windows=self.plan_pool_windows,
                seed=seed,
                config=window_config,
                bucket_merged_tokens=merged_tokens,
                bucket_raw_patches=raw_patches,
                bucket_weights=weights,
            )
            if num_samples > 0
            else None
        )

    def __len__(self) -> int:
        return self.num_samples

    def _generator(self, idx: int, stream: int, item: int = 0) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        seed_state = _seed_sequence(self.seed, idx, stream, item).generate_state(1, dtype=np.uint64)
        generator.manual_seed(int(seed_state[0]) % _MAX_TORCH_SEED)
        return generator

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.num_samples == 0:
            raise IndexError("Cannot index an empty PackedWindowQwen35VLDataset.")
        idx = int(idx) % self.num_samples
        window = self.plan.window(idx % self.plan_pool_windows)
        # Enforce the patch budget from the plan geometry BEFORE any pixel
        # tensor is materialized: the packer-level guard alone cannot prevent
        # the DataLoader/host-memory peak of a heavy window (a 128K tail
        # window is multiple GiB of fp32 pixels).
        total_raw_patches = sum(atom.raw_patches for atom in window.atoms)
        if (
            self.max_raw_patches_per_window is not None
            and total_raw_patches > self.max_raw_patches_per_window
        ):
            raise ValueError(
                f"Window {idx} carries {total_raw_patches} raw vision patches, exceeding "
                f"max_raw_patches_per_window={self.max_raw_patches_per_window} "
                f"({len(window.atoms)} images). Long-window profiles require chunked "
                "vision-encoder execution (Phase B) before raising this budget."
            )

        input_ids = torch.randint(
            1,
            self.vocab_size,
            (self.seq_length,),
            dtype=torch.long,
            generator=self._generator(idx, stream=_TEXT_TOKEN_STREAM),
        )
        for special_id in self.special_ids:
            input_ids[input_ids == special_id] = self.safe_text_token_id
        for atom in window.atoms:
            input_ids[atom.offset] = self.vision_start_token_id
            input_ids[atom.offset + 1 : atom.offset + 1 + atom.merged_tokens] = self.image_token_id

        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        # No cross-document prediction: the last position of every segment
        # has no target inside its own document.
        boundary = 0
        for _, segment_length in window.segments:
            boundary += segment_length
            labels[boundary - 1] = -100
        target_is_special = labels == -100
        for special_id in self.special_ids:
            target_is_special |= labels == special_id
        labels[target_is_special] = -100
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        loss_mask[target_is_special] = 0.0

        if window.atoms:
            # One preallocated buffer, filled per image: no per-image tensor
            # list + concat, so the host peak is the payload itself rather
            # than twice the payload.
            pixel_values = torch.empty((total_raw_patches, self.pixel_dim), dtype=torch.float32)
            row = 0
            for ordinal, atom in enumerate(window.atoms):
                pixel_values[row : row + atom.raw_patches].normal_(
                    generator=self._generator(idx, stream=_PIXEL_VALUE_STREAM, item=ordinal)
                )
                row += atom.raw_patches
            image_grid_thw = torch.tensor(
                [self.grids[atom.bucket_index] for atom in window.atoms], dtype=torch.long
            )
        else:
            pixel_values = torch.empty((0, self.pixel_dim), dtype=torch.float32)
            image_grid_thw = torch.empty((0, 3), dtype=torch.long)

        seq_lens = torch.tensor([length for _, length in window.segments], dtype=torch.long)
        if int(seq_lens.sum().item()) != self.seq_length:
            raise RuntimeError(
                f"Window {idx} segment lengths sum to {int(seq_lens.sum().item())}; "
                f"expected {self.seq_length}."
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "seq_lens": seq_lens,
        }


def train_valid_test_varlen_datasets_provider(
    train_val_test_num_samples,
) -> tuple[PackedWindowQwen35VLDataset, PackedWindowQwen35VLDataset, PackedWindowQwen35VLDataset]:
    """Provide packed-window mock train, validation, and test datasets."""
    from megatron.training import get_args

    args = get_args()
    if getattr(args, "use_varlen_dataset", False):
        raise ValueError(
            "The multimodal mock_varlen provider is incompatible with --use-varlen-dataset; "
            "vision payloads are packed by multimodal_dev.forward_step instead."
        )
    if getattr(args, "sequence_packing_scheduler", None) is not None:
        raise ValueError(
            "The multimodal mock_varlen provider is incompatible with "
            "--sequence-packing-scheduler; vision payloads are packed by "
            "multimodal_dev.forward_step instead."
        )
    uses_hybridep = (
        getattr(args, "use_packed_sequence", False)
        and getattr(args, "moe_token_dispatcher_type", None) == "flex"
        and getattr(args, "moe_flex_dispatcher_backend", None) == "hybridep"
    )
    if uses_hybridep and not getattr(args, "moe_hybridep_pad_variable_tokens", False):
        raise ValueError(
            "The multimodal mock_varlen provider requires "
            "--moe-hybridep-pad-variable-tokens with packed THD + HybridEP; "
            "locally packed token counts can differ across the HybridEP group."
        )
    if not getattr(args, "use_packed_sequence", False):
        raise ValueError(
            "The multimodal mock_varlen packed_window provider requires "
            "--use-packed-sequence: windows carry multiple document segments "
            "(seq_lens) and the padded BSHD layout has no segment representation."
        )
    if not getattr(args, "use_vanilla_collate_fn", False):
        raise ValueError(
            "The multimodal mock_varlen provider requires --use-vanilla-collate-fn "
            "so variable-length samples remain a list until multimodal packing."
        )

    total_seq_length = int(getattr(args, "total_seq_length", 1024))
    model_seq_length = getattr(args, "seq_length", None)
    if model_seq_length is None or total_seq_length != int(model_seq_length):
        raise ValueError(
            "The multimodal mock_varlen provider requires --total-seq-length to equal "
            f"--seq-length; got total_seq_length={total_seq_length}, "
            f"seq_length={model_seq_length}."
        )

    window_config = load_json_arg(getattr(args, "varlen_mock_dataset_config_json", None))
    if not isinstance(window_config, dict) or window_config.get("mode") != "packed_window":
        raise ValueError(
            'mock_varlen supports only {"mode": "packed_window"} in '
            "--varlen-mock-dataset-config-json (the legacy distribution/file "
            "sample modes were removed; use --dataset-provider mock for "
            "fixed-shape data)."
        )
    micro_batch_size = int(getattr(args, "micro_batch_size", 1) or 1)
    if micro_batch_size != 1:
        raise ValueError(
            "packed_window mode requires micro_batch_size == 1 for training and "
            f"evaluation: one item already is a full {total_seq_length}-token "
            f"window; got micro_batch_size={micro_batch_size}."
        )

    kwargs = dict(
        seq_length=total_seq_length,
        window_config={key: value for key, value in window_config.items() if key != "mode"},
        vocab_size=getattr(args, "padded_vocab_size", 248320),
        image_token_id=getattr(args, "image_token_id", QWEN35_VL_IMAGE_TOKEN_ID),
        image_size_config=load_json_arg(getattr(args, "mock_image_size_config_json", None)),
        # Mirror the packer guard at the dataset so over-budget windows fail
        # before pixels are materialized on the host.
        max_raw_patches_per_window=getattr(args, "max_vision_patches_per_microbatch", None),
    )
    seed = int(getattr(args, "seed", 1234))
    return tuple(
        PackedWindowQwen35VLDataset(
            num_samples=train_val_test_num_samples[split], seed=seed + split, **kwargs
        )
        for split in range(3)
    )
