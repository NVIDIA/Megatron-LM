# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-width image/text mock pretraining for the native Bridge VLM builders.

Uses MLM's existing callback dataset and masked-loss interfaces. Qwen receives
patchified pixels and grid metadata; Super VL receives pixels and image sizes.
As in Bridge's model-specific steps, the models own modality insertion and
multimodal positions. This initial path supports PP=CP=1, with ordinary TP/EP/DP.
It does not implement conversation tokenization, SFT packing, video, or Energon.
"""

from collections.abc import Callable, Iterator, Sequence
from functools import partial

import torch
from torch.utils.data import Dataset

from megatron.bridge.models.nemotron_3.super_vl_config import Nemotron35SuperVLModelConfig
from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig
from megatron.core.datasets.utils import Split
from megatron.training import get_timers
from megatron.training.config import PretrainConfigContainer
from megatron.training.global_vars import get_run_config


def validate_vlm_run(config: PretrainConfigContainer) -> None:
    """Reject layouts not implemented by this fixed-width VLM input path."""
    if not isinstance(config.model, (Nemotron35SuperVLModelConfig, Qwen35VLModelConfig)):
        raise TypeError("This forward step requires a Qwen3.5 VL or Nemotron3.5 Super VL model")
    transformer = config.model.transformer
    if transformer.pipeline_model_parallel_size != 1 or transformer.context_parallel_size != 1:
        raise ValueError("The initial native VLM recipe step requires PP=1 and CP=1")
    if transformer.dynamic_context_parallel or transformer.sequence_packing_scheduler is not None:
        raise ValueError("The initial native VLM recipe step requires fixed-width unpacked inputs")


class MockVisionLanguageDataset(Dataset):
    """One synthetic 256x256 image followed by random text, with shifted targets.

    This is an input/gradient/throughput fixture, not a convergence dataset. Each
    index has its own RNG, independent of loader workers, TP rank, and training
    RNG state. MLM's ordinary sampler owns DP sharding and resume offsets.
    The model execution bound is the physical row width; no text is truncated.
    """

    def __init__(
        self,
        model_config: Nemotron35SuperVLModelConfig | Qwen35VLModelConfig,
        *,
        num_samples: int,
        seed: int,
        split: Split,
    ) -> None:
        self.model_config = model_config
        self.num_samples = num_samples
        self.seed = seed
        self.split = split
        self.sequence_length = model_config.transformer.max_seq_length_per_sample
        self.image_size = 256
        if isinstance(model_config, Qwen35VLModelConfig):
            self.patch_size = model_config.vision_config.patch_size
            merge = model_config.vision_config.spatial_merge_size
            self.image_tokens = (self.image_size // self.patch_size // merge) ** 2
        elif isinstance(model_config, Nemotron35SuperVLModelConfig):
            # RADIO's 16x16 patches are spatially downsampled 2x2 by the projector.
            self.patch_size = 16
            self.image_tokens = (self.image_size // self.patch_size // 2) ** 2
        else:
            raise TypeError(f"Unsupported mock VLM model: {type(model_config).__name__}")
        if self.sequence_length is None or self.sequence_length <= self.image_tokens + 4:
            raise ValueError("The execution length must fit the image prefix and text targets")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = torch.Generator().manual_seed(self.seed + index)
        model = self.model_config
        # Keep ordinary random text away from both models' reserved media IDs.
        tokens = torch.randint(
            100, min(model.vocab_size, 10000), (self.sequence_length + 1,), generator=rng
        )
        if isinstance(model, Qwen35VLModelConfig):
            tokens[0] = model.vision_start_token_id
            tokens[1 : self.image_tokens + 1] = model.image_token_id
            tokens[self.image_tokens + 1] = model.vision_end_token_id
            grid = self.image_size // self.patch_size
            width = 3 * model.vision_config.temporal_patch_size * self.patch_size**2
            media = {
                "pixel_values": torch.randn(grid * grid, width, generator=rng),
                "image_grid_thw": torch.tensor([1, grid, grid], dtype=torch.long),
            }
            prefix = self.image_tokens + 2
        else:
            tokens[: self.image_tokens] = model.image_token_index
            media = {
                "images": torch.randn(3, self.image_size, self.image_size, generator=rng),
                "imgs_sizes": torch.tensor([self.image_size, self.image_size], dtype=torch.long),
                "num_frames": torch.tensor(1, dtype=torch.long),
            }
            prefix = self.image_tokens
        labels = tokens[1:].clone()
        loss_mask = torch.ones(self.sequence_length, dtype=torch.float32)
        loss_mask[: prefix - 1] = 0
        labels[: prefix - 1] = 0
        return {"input_ids": tokens[:-1], "labels": labels, "loss_mask": loss_mask, **media}


def mock_vlm_datasets_provider(
    train_val_test_num_samples: Sequence[int], vp_stage: int | None = None
) -> tuple[MockVisionLanguageDataset | None, ...]:
    """Build image/text fixtures on every model-parallel rank using the same seed."""
    cfg = get_run_config()
    validate_vlm_run(cfg)
    if cfg.data.dataset is not None:
        raise ValueError("Mock VLM recipes use a loader-only data section, not a GPT dataset")
    return tuple(
        (
            MockVisionLanguageDataset(
                cfg.model, num_samples=count, seed=cfg.rng.seed + split.value * 1000000, split=split
            )
            if count
            else None
        )
        for split, count in zip(Split, train_val_test_num_samples)
    )


mock_vlm_datasets_provider.is_distributed = True


def forward_step(
    data_iterator: Iterator[dict[str, torch.Tensor]], model: torch.nn.Module
) -> tuple[torch.Tensor, Callable]:
    """Run image-conditioned causal training and MLM's existing masked loss."""
    from pretrain_gpt import loss_func

    cfg = get_run_config()
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    batch = {name: value.cuda(non_blocking=True) for name, value in next(data_iterator).items()}
    timers("batch-generator").stop()
    if isinstance(cfg.model, Qwen35VLModelConfig):
        batch["pixel_values"] = batch["pixel_values"].flatten(0, 1)
        batch["image_grid_thw"] = batch["image_grid_thw"].reshape(-1, 3)
    else:
        batch["position_ids"] = torch.arange(
            batch["input_ids"].shape[1], device=batch["input_ids"].device
        ).expand_as(batch["input_ids"])
    batch["attention_mask"] = None
    output = model(**batch)
    loss_mask = batch["loss_mask"]
    if isinstance(output, tuple):
        output, model_loss_mask = output
        if model_loss_mask is not None:
            loss_mask = model_loss_mask
    return output, partial(loss_func, loss_mask.contiguous(), model=model)
