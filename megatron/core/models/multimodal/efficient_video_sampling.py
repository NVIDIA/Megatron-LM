import abc
import dataclasses
from functools import partial
import math
from math import ceil
from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
import einops
import torch.nn.functional as F
from scipy.stats import beta

from megatron.core.packed_seq_params import PackedSeqParams


def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - y)

def plain_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y

def cosine(x: torch.Tensor, y: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    cosine_dissimilarity = 1 - torch.nn.functional.cosine_similarity(x, y, dim=dim)
    if keepdim:
        cosine_dissimilarity = cosine_dissimilarity.unsqueeze(dim)
    return cosine_dissimilarity


predefined_distance_metrics = {
    "l1": l1,
    "plain_diff": plain_diff,
    "cos": partial(cosine, dim=1, keepdim=True),
}


def _calculate_mean_temporal_diff(
        x: torch.Tensor,
        *,
        patch_size: int,
        tubelet_size: int = 1,
        distance_metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
):
    """
    Inspired by: https://github.com/rccchoudhury/rlt/blob/69ec8b4b8bf8fc44252108801f095936d9845dba/src/models/static_token_utils.py#L7
    Args:
     - x (torch.Tensor): A tensor of shape [B, C, T, H, W] or [B, C, T, seq].
     - patch_size (int): The patch size to use. The resulting mask would be of size H/patch_size x W/patch_size.
     - tubelet_size (int): The temporal length of a token.
     - distance_metric (str): The distance metric to use.
    """
    if x.ndim == 4:  # [B, C, T, seq] feature level
        pool = F.avg_pool2d
        kernel_size = (1, patch_size)
    elif x.ndim == 5:  # [B, C, T, H, W] pixel level
        pool = F.avg_pool3d
        kernel_size = (1, patch_size, patch_size)
    else:
        raise ValueError(f"Expected 4 or 5 dimensions, got {x.shape=}")

    if distance_metric is None:
        distance_metric = l1
    if isinstance(distance_metric, str):
        distance_metric = predefined_distance_metrics[distance_metric]

    x = x.type(torch.float32)

    # Calculate differences between frames with a step of tubelet_size, ensuring batch dimension is preserved
    # Compare "front" of first token to "back" of second token
    first_tubes = x[:, :, (2 * tubelet_size - 1)::tubelet_size]
    next_tubes = x[:, :, :-tubelet_size:tubelet_size]

    diffs = distance_metric(first_tubes, next_tubes)

    # Apply average pooling over spatial dimensions while keeping the batch dimension intact
    avg_pool_blocks = pool(diffs, kernel_size=kernel_size, ceil_mode=True)
    # Compute the mean along the channel dimension, preserving the batch dimension
    avg_pool_blocks = torch.mean(avg_pool_blocks, dim=1, keepdim=True)
    return avg_pool_blocks

def pairwise_find_idxs_to_keep(
        x: torch.Tensor,
        *,
        patch_size: int,
        tubelet_size: int = 1,
        distance_metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        threshold: Optional[float] = None,
        quantile: Optional[float] = None,
) -> torch.Tensor:
    """

    Args:
     - x (torch.Tensor): A tensor of shape [B, C, T, H, W] or [B, C, T, seq].
     - patch_size (int): The patch size to use. The resulting mask would be of size H/patch_size x W/patch_size.
     - tubelet_size (int): The temporal length of a token.
     - distance_metric (str | Callable): The distance metric to use.
     - threshold (float): The mean intensity threshold for considering a token as static.
     - quantile (float): The quantile of tokens to consider static, based on intensity.
    Returns:
     - mask (torch.Tensor): A bool tensor of shape [B, T, H, W] or [B, T, seq] that selects tokens that are not repeated.

    """
    # TODO(nbagrov): we can support threshold with quantile as upper bound
    assert (threshold is not None) ^ (quantile is not None), f"Exactly one of {threshold=} or {quantile=} must be specified"

    avg_pool_blocks = _calculate_mean_temporal_diff(x, patch_size=patch_size, tubelet_size=tubelet_size, distance_metric=distance_metric)

    # Create a dummy first frame for each item in the batch
    first_frame = torch.ones_like(avg_pool_blocks[:, :, 0:1]) * 255
    # Concatenate the dummy first frame with the rest of the frames, preserving the batch dimension
    avg_pool_blocks = torch.cat([first_frame, avg_pool_blocks], dim=2).squeeze(1)
    # Determine indices to keep based on the threshold, ensuring the operation is applied across the batch
    if threshold is not None:
        keep_idxs = avg_pool_blocks > threshold
    elif quantile is not None:
        q_value = torch.quantile(avg_pool_blocks, quantile)
        keep_idxs = avg_pool_blocks > q_value
    else:
        raise NotImplementedError

    return keep_idxs


def iterative_find_idxs_to_keep(
        x: torch.Tensor,
        *,
        patch_size: int,
        tubelet_size: int = 1,
        distance_metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "plain_diff",
        threshold: Optional[float] = None,
        quantile: Optional[float] = None,
) -> torch.Tensor:
    """
    Inspired by: https://github.com/rccchoudhury/rlt/blob/69ec8b4b8bf8fc44252108801f095936d9845dba/src/models/static_token_utils.py#L7

    Args:
     - x (torch.Tensor): A tensor of shape [B, C, T, H, W] or [B, C, T, seq].
     - patch_size (int): The patch size to use. The resulting mask would be of size H/patch_size x W/patch_size.
     - tubelet_size (int): The temporal length of a token.
     - distance_metric (str | Callable): The distance metric to use.
     - threshold (float): The mean intensity threshold for considering a token as static.
     - quantile (float): The quantile of tokens to consider static, based on intensity.
    Returns:
     - mask (torch.Tensor): A bool tensor of shape [B, T, H, W] or [B, T, seq] that selects tokens that are not repeated.

    """

    # TODO(nbagrov): we can support threshold with quantile as upper bound
    assert (threshold is not None) ^ (quantile is not None), f"Exactly one of {threshold=} or {quantile=} must be specified"

    avg_pool_blocks = _calculate_mean_temporal_diff(x, patch_size=patch_size, tubelet_size=tubelet_size, distance_metric=distance_metric)

    cum_diffs = torch.cumsum(avg_pool_blocks, dim=1)  # B x (T-1) x H' x W' Cumulative sum along the temporal dimension
    cum_diffs = torch.cat([torch.zeros_like(cum_diffs[:, :1]), cum_diffs], dim=1)  # Add a dummy first frame with zeros

    (B, T), other_shapes = cum_diffs.shape[:2], cum_diffs.shape[2:]

    # The first frame is always our keyframe, so we initialize the keyframe index tensor
    keyframe_indexes = torch.zeros((B, 1, *other_shapes), dtype=torch.long, device=cum_diffs.device)

    keep_masks = [
        torch.ones((B, 1, *other_shapes), dtype=torch.bool, device=cum_diffs.device)
    ]

    for i in range(1, T):  # Iterate over the video duration
        prev = torch.gather(cum_diffs, 1, keyframe_indexes)
        curr = cum_diffs[:, i:i + 1]
        diff = torch.abs(curr - prev)
        if threshold is not None:
            keep_idxs = diff > threshold
        elif quantile is not None:
            q_value = torch.quantile(diff, quantile)
            keep_idxs = diff > q_value
        else:
            raise NotImplementedError

        keyframe_indexes[keep_idxs] = i  # Update keyframe indexes where the condition is met
        keep_masks.append(keep_idxs)

    keep_idxs = torch.cat(keep_masks, dim=1)  # Stack the masks along the temporal dimension
    return keep_idxs


class FloatSampler(abc.ABC):
    """
    An abstract class for sampling a pruning rate from a certain distribution.
    """
    @abc.abstractmethod
    def sample(self, *, n_frames: int) -> float:
        """
        Sample a pruning rate from a distribution.
        Args:
            n_frames (int): The number of frames in the video.
            is_training (bool): Whether the model is in training mode.
        Returns:
            float: The sampled pruning rate.
        """
        ...

    @abc.abstractmethod
    def get_default_quantile(self) -> float:
        ...

class BetaDistribution:
    class Sampler(FloatSampler):
        def __init__(self, mode, min_val, max_val):
            super().__init__()
            self._alpha, self._beta = BetaDistribution.get_beta_params_from_mode(mode, min_val, max_val)
            self._min_val = min_val
            self._max_val = max_val
            self._mode = mode

        def get_default_quantile(self):
            return self._mode

        def sample(self, *, n_frames):
            return self._min_val + beta.rvs(self._alpha, self._beta, size=1)[0] * (self._max_val - self._min_val)

    @staticmethod
    def get_beta_params_from_mode(mode, min_val, max_val, grid_points=200, skew_bias=4):
        target_mode_unit = (mode - min_val) / (max_val - min_val)
        grid = np.linspace(1.01, 10, grid_points)
        alpha_grid, beta_grid = np.meshgrid(grid, grid)

        # Compute mode for each pair
        mode_grid = (alpha_grid - 1) / (alpha_grid + beta_grid - 2)
        loss = (mode_grid - target_mode_unit) ** 2

        # Add skew penalty to favor asymmetry (higher skew_bias favors more skew)
        skew_penalty = (alpha_grid / beta_grid) ** skew_bias
        loss *= skew_penalty

        idx = np.unravel_index(np.argmin(loss), loss.shape)
        return alpha_grid[idx], beta_grid[idx]

class FrameBasedSampler(FloatSampler):
    def __init__(self, quantile: float):
        super().__init__()
        self._quantile = quantile
        self._mapper = {}

    def get_default_quantile(self):
        return self._quantile

    def sample(self, *, n_frames:int):
        n_frames = max(n_frames, 2)  # just a safety mechanism
        if n_frames not in self._mapper:
            max_val = (n_frames - 1) / n_frames
            mode = min(max_val, self._quantile)  # just a safety mechanism
            self._mapper[n_frames] = BetaDistribution.Sampler(min_val=0, max_val=max_val, mode=mode)
        return self._mapper[n_frames].sample(n_frames=n_frames)


class DeltaDistribution:
    class Sampler(FloatSampler):
        def __init__(self, mode):
            self._mode = mode

        def get_default_quantile(self) -> float:
            return self._mode

        def sample(self, *, n_frames:int) -> float:
            return self._mode


@dataclasses.dataclass
class EVSConfig:
    method: str
    distance_metric: str
    quantile_sampler: FloatSampler
    level: str
    patch_size: int
    position_ids_handling: str


class EVSVariant:
    """
    Format: {method}+{quantile}+{level}+p{patch_size}+pos-{position_ids_handling}.
    Example: `keyframe+beta-0.75+pix+p32+pos-ignore`
             `pairwise+frame-based-0.75+features-cos+p1+pos-preserve`

    """
    feature_location_in_variant = {
        "method": 0,
        "quantile": 1,
        "level": 2,
        "patch_size": 3,
        "posids_handling": 4,
    }

    method_to_callable = {
        "pairwise": pairwise_find_idxs_to_keep,
        "iterative": iterative_find_idxs_to_keep
    }

    @classmethod
    def from_string(cls, variant: Optional[str]):
        if not cls._should_init(variant):
            return None

        method = cls._process_method(variant)
        level, distance_metric = cls._process_level(variant)
        config = EVSConfig(
            method=method,
            quantile_sampler=cls._process_quantile(variant),
            level=level,
            distance_metric=distance_metric,
            patch_size=cls._process_patch_size(variant),
            position_ids_handling=cls._process_position_ids_handling(variant),
        )
        return cls(variant=variant, config=config)

    @classmethod
    def uses_special_position_ids(cls, variant: Optional[str]):
        return cls._should_init(variant) and cls._process_position_ids_handling(variant) == "preserve"

    @classmethod
    def _should_init(cls, variant: Optional[str]):
        return variant not in [None, "None", "none", "Off", "off", False, "False", "false"]

    def __init__(self, *, variant: str, config: EVSConfig):
        self._config = config
        self.variant = variant  # for debugging, nice name, prints, etc
        self._level_to_callable = {
            "pixels": self._create_pixel_level_evs_mask,
            "features": self._create_feature_level_evs_mask,
        }
        self.enabled = True
        print(f"{self.__class__.__name__} initialized: {variant}")

    @classmethod
    def extract_from_variant(cls, variant: str, feature: str):
        try:
            return variant.split('+')[cls.feature_location_in_variant[feature]]
        except (KeyError, IndexError):
            raise ValueError(f"Could not extract '{feature}' from {variant=}, based on mapping {cls.feature_location_in_variant}. "
                             f"Either mapping or variant does not contain the feature")

    @classmethod
    def _process_method(cls, variant: str):
        method = cls.extract_from_variant(variant, feature='method')

        if method in ["random", "frames", "noop"]:  # Here are dummy "methods", used for plug-in baselines
            return method

        supported = list(cls.method_to_callable.keys())
        if method not in supported:
            raise ValueError(f"Only {supported} are supported, got: {method} (from {variant})")

        return method

    @classmethod
    def _process_position_ids_handling(cls, variant: str):
        _, posids_handling = cls.extract_from_variant(variant, feature="posids_handling").split("-")
        assert _ == "pos"
        supported = ["ignore", "preserve"]
        if posids_handling not in supported:
            raise ValueError(f"Only {supported} are supported, got: {posids_handling} (from {variant})")
        return posids_handling

    @classmethod
    def _process_patch_size(cls, variant):
        patch = cls.extract_from_variant(variant, feature='patch_size')
        assert patch[0] == "p"
        return int(patch[1:])

    @classmethod
    def _process_level(cls, variant) -> Tuple[str, str]:
        level = cls.extract_from_variant(variant, feature="level")
        supported = {  # aliases
            "pixels": ["pixels", "pixel", "pix"],
            "features": ["features", "feature", "feat", "feats"],
        }

        if "-" in level:
            level, distance_metric = level.split("-")
        else:
            distance_metric = "l1"

        if distance_metric not in predefined_distance_metrics:
            raise ValueError(f"Unknown distance metric '{distance_metric}'. ")

        found = False
        for key, aliases in supported.items():
            if level in aliases:
                found = key
                break

        if not found:
            raise ValueError(f"Only {list(supported.keys())} are supported, got: {level} (from {variant})")

        return found, distance_metric

    @classmethod
    def _process_quantile(cls, variant: str):
        q_variant_to_sampler = {
            "static": DeltaDistribution.Sampler,
            "beta": partial(BetaDistribution.Sampler, min_val=0, max_val=0.83),
        }
        variant_and_value = cls.extract_from_variant(variant, feature='quantile')  # example: random-0.75, static-0.25, ...

        if variant_and_value.startswith("frame-based-"):
            quantile = float(variant_and_value.split("-")[-1])
            return FrameBasedSampler(quantile)

        q_variant, quantile = variant_and_value.split("-")

        if q_variant not in q_variant_to_sampler:
            raise NotImplementedError(f"Only {list(q_variant_to_sampler.keys())} are supported, got: {q_variant} (from {variant})")
        return q_variant_to_sampler[q_variant](float(quantile))

    @property
    def distance_metric(self) -> str:
        return self._config.distance_metric

    @property
    def patch_size(self) -> int:
        return self._config.patch_size

    @property
    def level(self) -> str:
        return self._config.level

    @property
    def method(self) -> str:
        return self._config.method

    @property
    def position_ids_handling(self) -> str:
        return self._config.position_ids_handling

    def sample_quantile(self, n_frames:int, is_training: bool) -> float:
        if not is_training:
            return self._config.quantile_sampler.get_default_quantile()
        return self._config.quantile_sampler.sample(n_frames=n_frames)

    def _create_media_according_to_evs_level(self, images: Optional[torch.Tensor], embeddings: Optional[torch.Tensor], num_tiles, num_frames):
        if self.level == "pixels":
            if images is None:
                raise ValueError("For pixel level, please provide images")
            split_media = EVSHelper.prepare_for_mask_computation(images, num_tiles, num_frames)
            expected_sequence_length = (ceil(images.shape[-2] / self.patch_size) * ceil(images.shape[-1] / self.patch_size))  # h*w / patch^2
        elif self.level == "features":
            if embeddings is None:
                raise ValueError("For features level, please provide embeddings")
            embeddings = embeddings.permute(1, 0, 2)  # [seq, tiles*frames, feats] -> [tiles*frames, seq, feats]
            split_media = EVSHelper.prepare_for_mask_computation(embeddings, num_tiles, num_frames)
            expected_sequence_length = ceil(embeddings.shape[-2] / self.patch_size)  # seq / patch
        else:
            raise ValueError(f"Level '{self.level}' is not supported.")

        return split_media, expected_sequence_length

    def _create_pixel_level_evs_mask(self, pixels: torch.Tensor, *, frames: int, tiles: int, is_training:bool) -> torch.Tensor:
        """
        In: shape=[frames*tiles, channels, height, width] type=torch.float (32, 16, whatever)
        Out: shape=[frames*tiles, height * width], type=torch.bool
        """
        x = einops.rearrange(pixels, '(T t) C H W -> t C T H W', T=frames, t=tiles)  # trick here is to address tiles like "batch"
        assert x.shape == (tiles, pixels.shape[1], frames, *pixels.shape[-2:])
        masks = self.method_to_callable[self.method](
            x,
            threshold=None,
            quantile=self.sample_quantile(frames, is_training),
            tubelet_size=1,
            patch_size=self.patch_size,
            distance_metric=self.distance_metric,
        )
        masks = einops.rearrange(masks, 't T H W -> (T t) (H W)')  # Tile-first to Frame-first
        return masks

    def _create_feature_level_evs_mask(self, embeddings: torch.Tensor, *, frames: int, tiles: int, is_training:bool) -> torch.Tensor:
        """
        In: shape=[frames*tiles, seq, features] type=torch.float (32, 16, whatever)
        Out: shape=[frames*tiles, seq], type=torch.bool
        """

        x = einops.rearrange(embeddings, '(T t) S C -> t C T S', T=frames, t=tiles)
        assert x.shape == (tiles, embeddings.shape[-1], frames, embeddings.shape[1])
        masks = self.method_to_callable[self.method](
            x,
            threshold=None,
            quantile=self.sample_quantile(frames, is_training),
            tubelet_size=1,
            patch_size=self.patch_size,
            distance_metric=self.distance_metric,
        )
        masks = einops.rearrange(masks, 't T S -> (T t) S')  # Tile-first to Frame-first
        return masks

    def calculate_mask(self, *, images: Optional[torch.Tensor], embeddings: Optional[torch.Tensor],
                       num_tiles: torch.Tensor, num_frames: torch.Tensor, is_video: List[bool], is_training:bool) -> Tuple[List[torch.Tensor], int]:
        """
        images.shape = (tiles*frames, C, H, W)
        embeddings.shape = (seq, tiles*frames, features)
        """

        split_media, expected_sequence_length = self._create_media_according_to_evs_level(
            images=images, embeddings=embeddings, num_tiles=num_tiles, num_frames=num_frames
        )

        if len(split_media) != len(is_video):
            raise ValueError(f"The number of split media ({len(split_media)}) does not match the length of the media filter ({len(is_video)})")

        device = images.device if images is not None else embeddings.device
        masks = []
        for media, media_is_video in zip(split_media, is_video):
            mask = torch.ones((media.tensor.shape[0], expected_sequence_length), dtype=torch.bool, device=device)
            if not media_is_video or self.method == "noop" or not self.enabled:
                masks.append(mask)
                continue

            if self.method == "random":
                flat_mask = mask.flatten()
                flat_mask[:int(flat_mask.numel() * self.sample_quantile(media.frames, is_training=is_training))] = False
                flat_mask = flat_mask[torch.randperm(media.tensor.shape[0] * expected_sequence_length, device=images.device)]  # shuffle
                mask = flat_mask.reshape(media.tensor.shape[0], expected_sequence_length)
                masks.append(mask)
                print(f"Debug (eval): {self.method=}: {(mask.sum() / mask.numel())=}")
                continue

            if self.method == "frames":
                remaining = int(media.tensor.shape[0] * (1 - self.sample_quantile(media.frames, is_training=is_training)))  # example: 8 frames * (1-0.75) = 2
                keep_indices = set(i * (media.tensor.shape[0] // remaining) for i in range(remaining))  # [0, 1] -> [0, 4] - the frames we keep
                mask_indices = set(range(media.tensor.shape[0])) - keep_indices
                mask[list(mask_indices), ...] = False
                masks.append(mask)
                print(f"Debug (eval): {self.method=}: {(mask.sum() / mask.numel())=}")
                continue

            # Here, `method` is to actually apply EVS
            # The `method` is baked inside `level` callable
            assert self.method in self.method_to_callable

            mask = self._level_to_callable[self.level](
                media.tensor, frames=media.frames, tiles=media.tiles_per_frame, is_training=is_training
            )
            assert mask.shape == (media.tensor.shape[0], expected_sequence_length)
            masks.append(mask)

        return masks, expected_sequence_length

    def mask_embeddings(
            self,
            *,
            embeddings: torch.Tensor,
            evs_mask: torch.Tensor,
            packed_seq_params: Optional["PackedSeqParams"] = None,
            per_sample_pad_to_divisibility: Optional[int] = None,
            sequence_pad_to_divisibility: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], "PackedSeqParams"]:
        assert embeddings.ndim == 3, f"Expected 3D inputs: [batch, seq, hidden], got: {embeddings.shape}"
        batch_size = embeddings.size(0)

        # Handle embeddings
        final_embedding_list = []
        adjusted_packed_seq_params = None
        for i in range(batch_size):
            if packed_seq_params is not None:
                assert batch_size == 1
                assert packed_seq_params.qkv_format == 'thd'  # Wasn't tested with other formats

            kept_embeddings_sequence_padded, adjusted_packed_seq_params = EVSHelper.masked_select(  # note, we override `packed_seq_params`
                embeddings[i], evs_mask[i], packed_seq_params=packed_seq_params,
                per_sample_pad_to_divisibility=per_sample_pad_to_divisibility,
                sequence_pad_to_divisibility=sequence_pad_to_divisibility,
            )
            final_embedding_list.append(kept_embeddings_sequence_padded)
        final_embedding = torch.nn.utils.rnn.pad_sequence(final_embedding_list, batch_first=True)

        # Handle position ids
        final_position_ids = None
        final_position_ids_list = []
        if self.position_ids_handling == "preserve":
            if packed_seq_params is not None:
                position_ids = self._build_position_ids_for_packed_sequence(packed_seq_params).to(device=embeddings.device)
            else:
                position_ids = torch.arange(embeddings.size(1), device=embeddings.device).unsqueeze(0).repeat(batch_size, 1)

            assert position_ids.shape[0] == embeddings.shape[0]
            assert position_ids.shape[1] == embeddings.shape[1]

            for i in range(batch_size):
                kept_position_ids_sequence_padded, _ = EVSHelper.masked_select(
                    position_ids[i], evs_mask[i], packed_seq_params=packed_seq_params,
                    per_sample_pad_to_divisibility=per_sample_pad_to_divisibility,
                    sequence_pad_to_divisibility=sequence_pad_to_divisibility,
                )
                final_position_ids_list.append(kept_position_ids_sequence_padded)
            final_position_ids = torch.nn.utils.rnn.pad_sequence(final_position_ids_list, batch_first=True)

        return final_embedding, final_position_ids, adjusted_packed_seq_params

    @staticmethod
    def _build_position_ids_for_packed_sequence(packed_seq_params: PackedSeqParams) -> torch.Tensor:
        """
        Given an input packed sequence params object, that represents sequences A,B,C..
        we build the position ids for A,B,C independently and then concatenate them.
        :return: shape [1, seq]
        """
        chunks = packed_seq_params.cu_seqlens_q[1:] - packed_seq_params.cu_seqlens_q[:-1] # [Num sequences] where each element is the length of the sequence
        position_ids = [torch.arange(chunk_len, device=packed_seq_params.cu_seqlens_q.device) for chunk_len in chunks]
        return torch.cat(position_ids, dim=0).view(1, -1)


    def mask_labels_and_loss_mask(  # noqa: compains that can be static
            self,
            *,
            labels: torch.Tensor,
            loss_mask: torch.Tensor,
            evs_mask: torch.Tensor,
            packed_seq_params: Optional["PackedSeqParams"] = None,
            per_sample_pad_to_divisibility: Optional[int] = None,
            sequence_pad_to_divisibility: Optional[int] = None,
            labels_padding_value: Optional[int] = None,
            loss_padding_value: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert labels.ndim == loss_mask.ndim == 2, f"Expected 2D inputs: [batch, seq], got: {labels.shape=}, {loss_mask.shape=}"
        batch_size = labels.size(0)

        final_labels_list = []
        final_loss_mask_list = []
        for i in range(batch_size):
            kept_labels_sequence, _ = EVSHelper.masked_select(
                labels[i], evs_mask[i], packed_seq_params=packed_seq_params,
                per_sample_pad_to_divisibility=per_sample_pad_to_divisibility,
                sequence_pad_to_divisibility=sequence_pad_to_divisibility,
                padding_value=labels_padding_value,
            )

            kept_loss_mask_sequence, _ = EVSHelper.masked_select(
                loss_mask[i], evs_mask[i], packed_seq_params=packed_seq_params,
                per_sample_pad_to_divisibility=per_sample_pad_to_divisibility,
                sequence_pad_to_divisibility=sequence_pad_to_divisibility,
                padding_value=loss_padding_value
            )

            final_labels_list.append(kept_labels_sequence)
            final_loss_mask_list.append(kept_loss_mask_sequence)

        final_labels = torch.nn.utils.rnn.pad_sequence(final_labels_list, batch_first=True, padding_value=labels_padding_value)
        final_loss_mask = torch.nn.utils.rnn.pad_sequence(final_loss_mask_list, batch_first=True, padding_value=loss_padding_value)
        return final_labels, final_loss_mask


class EVSHelper:
    @dataclasses.dataclass
    class Media:
        tensor: torch.Tensor
        tiles_per_frame: int
        frames: int

        def __post_init__(self):
            assert len(self.tensor) == self.frames * self.tiles_per_frame

    @staticmethod
    def prepare_for_mask_computation(tensors: torch.Tensor, num_tiles: torch.Tensor, num_frames: torch.Tensor) -> List["EVSHelper.Media"]:
        """
        Splits `tensors` (images or embeddings) based on `num_tiles` and `num_frames`, and return a list of tensors, where each tensor belongs to a media -
        Media can be an image, video, etc. The important thing is that it "reverses" the batching and packing procedure.
        """
        if len(tensors) != sum(num_tiles):
            raise ValueError(f"The number of image tensors ({len(tensors)}) does not match the total number of tiles ({len(num_tiles)})")

        if len(num_tiles) != sum(num_frames):
            raise ValueError(f"The number of image tiles ({len(num_tiles)}) does not match the total number of frames ({len(num_frames)})")

        split = []
        for nf in num_frames:
            curr_tiles, num_tiles = num_tiles[:nf], num_tiles[nf:]

            # if we work with a video, all frames should have same amount of tiles. If we work with images, then we have a single frame
            assert len(torch.unique(curr_tiles)) == 1
            needed_amount = sum(curr_tiles)
            tensor_tiles, tensors = tensors[:needed_amount], tensors[needed_amount:]

            # To comply with both tensors during training and ints during eval. Don't ask...
            ct = curr_tiles[0].item() if isinstance(curr_tiles[0], torch.Tensor) else curr_tiles[0]
            nf = nf.item() if isinstance(nf, torch.Tensor) else nf

            split.append(EVSHelper.Media(tensor=tensor_tiles, tiles_per_frame=ct, frames=nf))

        # we consume everything
        assert len(num_tiles) == len(tensors) == 0
        return split

    @staticmethod
    def pad_right(tensor, *, padding_dim: int = 0, padding_value: float | int | bool = 0, required_padding=None, required_divisibility=None):
        if required_padding is None and required_divisibility is None:
            raise ValueError("Either `padding` or `divisibility` must be specified")
        if required_padding is not None and required_divisibility is not None:
            raise ValueError("Only one of `padding` or `divisibility` must be specified")
        if required_divisibility == 0 or required_padding == 0:
            return tensor, 0

        if required_padding is None:
            assert required_divisibility is not None
            tensor_len = tensor.size(padding_dim)
            tensor_len_padded = int(math.ceil(tensor_len / required_divisibility) * required_divisibility)
            required_padding = tensor_len_padded - tensor_len

        padding_tuple_prefix = (0, 0) * (tensor.ndim - padding_dim - 1)
        if required_padding > 0:
            padding_tuple = padding_tuple_prefix + (0, required_padding)
            tensor_padded = torch.nn.functional.pad(tensor, padding_tuple, value=padding_value)
        else:
            tensor_padded = tensor

        return tensor_padded, required_padding

    @staticmethod
    def _safety_checks(tensor: torch.Tensor, evs_mask: torch.Tensor, packed_seq_params: Optional["PackedSeqParams"] = None):
        if evs_mask.ndim > 1:
            raise ValueError(f"EVS mask has more than 1 dimension: {evs_mask.ndim}")
        if evs_mask.size(0) != tensor.size(0):
            raise ValueError(f"EVS mask has a different number of elements than the tensor: {evs_mask.size(0)=} != {tensor.size(0)=}")
        if evs_mask.dtype != torch.bool:
            raise TypeError(f"EVS mask has a non-bool type: {evs_mask.dtype}")
        if evs_mask.device != tensor.device:
            raise ValueError(f"EVS mask must be on the same device as the tensor: {evs_mask.device=} != {tensor.device=}")

        if packed_seq_params is None:
            return

        # Here, we have `packed_seq_params`. Validate packed sequence bounds before using them as indices
        if not torch.equal(packed_seq_params.cu_seqlens_q, packed_seq_params.cu_seqlens_q_padded):
            raise NotImplementedError(f"At the moment, we assuming padded and not padded are equal,"
                                      f" got: {packed_seq_params.cu_seqlens_q_padded=} != {packed_seq_params.cu_seqlens_q=}")

        if packed_seq_params.cu_seqlens_q.device != evs_mask.device:
            raise ValueError(f"PackedSeqParams are not on the same device as the tensor: {packed_seq_params.cu_seqlens_q.device=} != {evs_mask.device=}")

        max_seq_idx = packed_seq_params.cu_seqlens_q[-1].item()
        if max_seq_idx > tensor.size(0):
            raise ValueError(f"Packed sequence cumulative length ({max_seq_idx}) exceeds tensor size ({tensor.size(0)})")
        if max_seq_idx > evs_mask.size(0):
            raise ValueError(f"Packed sequence cumulative length ({max_seq_idx}) exceeds EVS mask size ({evs_mask.size(0)})")


    @staticmethod
    def masked_select(tensor: torch.Tensor, evs_mask: torch.Tensor, *, packed_seq_params: Optional["PackedSeqParams"] = None,
                      per_sample_pad_to_divisibility: Optional[int] = None,
                      sequence_pad_to_divisibility: Optional[int] = None,
                      padding_dim: int = 0, padding_value: float | int | bool = 0):

        EVSHelper._safety_checks(tensor, evs_mask, packed_seq_params)

        if padding_dim >= tensor.ndim:
            raise ValueError(f"Padding dimension {padding_dim} is greater than the number of dimensions of the tensor {tensor.ndim}")

        original_length = tensor.size(0)

        if packed_seq_params is None:
            post_evs_tensor = tensor[evs_mask]
            if per_sample_pad_to_divisibility is not None or sequence_pad_to_divisibility is not None:
                post_evs_tensor, _ = EVSHelper.pad_right(
                    post_evs_tensor,
                    required_divisibility=max(per_sample_pad_to_divisibility or 0, sequence_pad_to_divisibility or 0),
                    padding_dim=padding_dim, padding_value=padding_value
                )
            assert post_evs_tensor.size(0) <= original_length
            return post_evs_tensor, packed_seq_params

        # We need to update the packed_seq_params to reflect the new sequence length
        # How: keep_embeddings_mask is a boolean mask that is True for the embeddings that should be kept
        # A packed_seq_params contains indexes like [0, 1002, 4123, 4321]
        # We take each slice and count how many values are False in that range. And subtract that value from the end of the slice.
        # We repeat this for each slice.
        # This way we can keep the packed_seq_params consistent with the new sequence length.

        post_evs_tensor = []
        evs_seqlens = torch.zeros_like(packed_seq_params.cu_seqlens_q)
        evs_seqlens_padding = torch.zeros_like(packed_seq_params.cu_seqlens_q)
        max_seq_len = 0
        for i in range(len(packed_seq_params.cu_seqlens_q) - 1):
            start_idx = packed_seq_params.cu_seqlens_q[i]
            end_idx = packed_seq_params.cu_seqlens_q[i + 1]
            assert 0 <= start_idx < end_idx <= tensor.size(0)
            seq_len = evs_mask[start_idx:end_idx].sum()
            if seq_len == 0:
                raise AssertionError("This should never happen, because there must be some text in the sequence!")
            evs_seqlens[i + 1] = seq_len
            relevant_sequence = tensor[start_idx:end_idx]
            kept_sequence_after_evs = relevant_sequence[evs_mask[start_idx:end_idx]]
            assert len(kept_sequence_after_evs) == seq_len
            if per_sample_pad_to_divisibility is not None:
                kept_sequence_after_evs, padding = EVSHelper.pad_right(
                    kept_sequence_after_evs,
                    required_divisibility=per_sample_pad_to_divisibility,
                    padding_dim=padding_dim, padding_value=padding_value
                )
                evs_seqlens_padding[i + 1] = padding
                seq_len = len(kept_sequence_after_evs)
            if sequence_pad_to_divisibility is not None and i == len(packed_seq_params.cu_seqlens_q) - 2:  # we now operate on the last sample
                # we should calculate required padding based on the entire sequence, not the current sample
                remainder = torch.sum(evs_seqlens + evs_seqlens_padding) % sequence_pad_to_divisibility
                required_padding = sequence_pad_to_divisibility - remainder if remainder != 0 else 0
                kept_sequence_after_evs, padding = EVSHelper.pad_right(
                    kept_sequence_after_evs,
                    required_padding=required_padding,
                    padding_dim=padding_dim, padding_value=padding_value
                )
                evs_seqlens_padding[i + 1] += padding
                seq_len = len(kept_sequence_after_evs)

            max_seq_len = max(max_seq_len, seq_len)
            post_evs_tensor.append(kept_sequence_after_evs)

        post_evs_tensor = torch.cat(post_evs_tensor, dim=0)
        evs_cu_seqlens = evs_seqlens.cumsum(dim=0)
        assert evs_cu_seqlens[0] == 0

        evs_cu_seqlens_padded = evs_cu_seqlens + evs_seqlens_padding.cumsum(0)

        # (@nbagrov, @ekhvedchenia): cu_seqlens should also be padded for some reason...
        evs_cu_seqlens = torch.clone(evs_cu_seqlens_padded)
        assert evs_cu_seqlens[0] == evs_cu_seqlens_padded[0] == 0

        modified_packed_seq_params = PackedSeqParams(
            qkv_format=packed_seq_params.qkv_format,
            max_seqlen_q=max_seq_len,
            max_seqlen_kv=max_seq_len,
            cu_seqlens_q=evs_cu_seqlens.to(dtype=torch.int32),
            cu_seqlens_kv=evs_cu_seqlens.to(dtype=torch.int32),
            cu_seqlens_q_padded=evs_cu_seqlens_padded.to(dtype=torch.int32),
            cu_seqlens_kv_padded=evs_cu_seqlens_padded.to(dtype=torch.int32),
        )

        return post_evs_tensor, modified_packed_seq_params
