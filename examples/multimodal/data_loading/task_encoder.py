# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import ast
import dataclasses
import hashlib
import importlib
import json
import os
import random
import re
import threading
from collections import defaultdict
from functools import partial
from typing import List, Literal, Tuple, TypedDict, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.energon import (
    Batch,
    Cooker,
    DefaultTaskEncoder,
    Lazy,
    MapDataset,
    Sample,
    SampleDecoder,
)
from megatron.energon import __version__ as energon_version
from megatron.energon import stateless
from megatron.energon.av import AVDecoder
from megatron.energon.edataclass import edataclass
from megatron.training import get_args, get_tokenizer

from .conversation_sample import ConversationSample, ImageMedia, Media, VideoFrameMedia, VideoMedia
from .cookers.conversation import (
    EXPLICIT_ASSISTANT_LOSS_FIELD,
    EXPLICIT_ASSISTANT_LOSS_MODE,
    _validate_loss_mask_subflavors,
    cook_conversation,
    cook_general_conversations_jsonl,
    cook_general_conversations_jsonl_explicit_loss_v1,
    cook_general_conversations_webdataset,
    cook_openai_messages_jsonl,
    cook_openai_messages_offline_packed_jsonl,
)
from .cookers.eagle import cook_eagle
from .image_processing import ImageTilingParams, create_image_tiling_strategy
from .knapsacks import (
    balanced_greedy_knapsack,
    bucketing_greedy_knapsack,
    greedy_knapsack,
    streaming_prompt_dedup_first_fit_knapsack,
)

IDENTITY_FILTER_DATASETS = ("apps, taco", "new sft problems", "NemotronX RL")
_ENERGON_AV_OPEN_LOCK = threading.Lock()


def _parse_packing_algorithm_parameters(raw_parameters: object) -> dict[str, str]:
    """Parse packing algorithm parameters from a dict-like or key=value string."""
    if raw_parameters is None:
        return {}
    if isinstance(raw_parameters, dict):
        parameters = raw_parameters
    elif isinstance(raw_parameters, str):
        raw_parameters = raw_parameters.strip()
        if not raw_parameters:
            return {}
        if raw_parameters.startswith("{"):
            try:
                parameters = json.loads(raw_parameters)
            except json.JSONDecodeError:
                try:
                    parameters = ast.literal_eval(raw_parameters)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(
                        "packing_algorithm_parameters must be a JSON/Python dict or "
                        "a comma-separated key=value string"
                    ) from exc
            if not isinstance(parameters, dict):
                raise ValueError("packing_algorithm_parameters dict input must parse to a dict")
        else:
            parameters = {}
            for item in raw_parameters.replace(",", " ").split():
                if "=" not in item:
                    raise ValueError(
                        "packing_algorithm_parameters entries must use key=value syntax"
                    )
                key, value = item.split("=", 1)
                parameters[key] = value
    else:
        raise TypeError("packing_algorithm_parameters must be a string or dict")

    normalized_parameters = {}
    for key, value in parameters.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("packing_algorithm_parameters contains an empty key")
        normalized_parameters[normalized_key] = str(value).strip()
    return normalized_parameters


def _pop_int_packing_algorithm_parameter(parameters: dict[str, str], key: str, default: int) -> int:
    raw_value = parameters.pop(key, None)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"packing algorithm parameter {key} must be an integer") from exc
    if value < 0:
        raise ValueError(f"packing algorithm parameter {key} must be non-negative")
    return value


def _clean_think(match: re.Match, *, newline_before_close: bool) -> str:
    """Strip whitespace inside <think> tags and format a non-empty trace."""
    clean_content = match.group(1).strip()
    if not clean_content:
        return "<think></think>"

    close_prefix = "\n" if newline_before_close else ""
    return f"<think>\n{clean_content}{close_prefix}</think>"


def _normalize_thinking_trace(content: str, *, thinking_trace_format: str) -> str:
    """Normalize a tagged thinking trace for the Nemotron 6 serialization contract."""
    ultra_format = thinking_trace_format == "ultra"
    content = re.sub(
        r"<think>(.*?)</think>",
        partial(_clean_think, newline_before_close=not ultra_format),
        content,
        flags=re.DOTALL,
    )

    if ultra_format:
        # Ultra serializes the answer directly after the closing thinking tag.
        replacement = "</think>"
    else:
        replacement = "</think>\n"
    return re.sub(r"</think>\s*", replacement, content)


def _has_filtered_identity_keyword(sample: ConversationSample, keywords: list[str]) -> bool:
    """Match configured identity keywords for datasets that need identity filtering."""
    normalized_keywords = tuple(keyword.lower() for keyword in keywords if keyword)
    if not normalized_keywords:
        return False
    if sample.__subflavors__.get("dataset") not in IDENTITY_FILTER_DATASETS:
        return False

    for message in sample.conversation:
        if message.sender != "assistant":
            continue
        for fragment in message.fragments:
            if isinstance(fragment, str):
                content_lower = fragment.lower()
                if any(keyword in content_lower for keyword in normalized_keywords):
                    return True
    return False


try:
    from megatron.core.models.multimodal.context_parallel import get_padding
except ImportError:
    get_padding = None


class NoTrainableTokensError(ValueError):
    """Raised when an SFT sample has no labels that should contribute to loss."""


@edataclass
class ConversationTaskSample(Sample):
    # (c, h, w)
    imgs: List[torch.Tensor]
    num_tiles: torch.Tensor
    tokens: torch.Tensor
    total_len: int  # Total token count in the sample, including text and image tokens
    total_len_padded: int  # Total token count in the sample, including text and image tokens, padded to the context parallel size
    labels: torch.Tensor = None


@edataclass
class PreEncodedTaskSample(Sample):
    tokens: torch.Tensor
    labels: torch.Tensor
    images: list[ImageTilingParams]
    total_len: int
    total_len_padded: int
    num_frames: list[int]
    prompt_hash: str


@edataclass
class PreEncodedOfflinePackedTextSample(Sample):
    tokens: torch.Tensor
    labels: torch.Tensor
    max_length: int
    cu_lengths: torch.Tensor
    cu_lengths_padded: torch.Tensor
    samples_seen: torch.Tensor
    sample_lengths: torch.Tensor


@edataclass
class PackedTaskSample(Sample):
    """Dataclass to store a single packed sample (not a batch).

    P = Number of sub-samples in the packed sample
    seq_len = Total sequence length
    num_imgs = Number of images across all samples in the packed sample
    """

    # Sample name
    __key__: list[str]
    # Input tokens packed into a single tensor (seq_len,)
    tokens: torch.Tensor
    # Target tokens packed into a single tensor (seq_len,)
    labels: torch.Tensor
    # Maximum length across sub-samples.
    max_length: int
    # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    cu_lengths: list[int]
    # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    cu_lengths_padded: list[int]

    # Input images
    imgs: list[torch.Tensor]
    # Number of tiles for each image of each sample (num_imgs)
    num_tiles: list[int]

    # Number of frames used per VideoMedia / ImageMedia (1 frame for ImageMedia)
    num_frames: list[int]

    # Number of samples in the packed sample
    samples_seen: int
    # Real lengths of the original samples before packing.
    sample_lengths: torch.Tensor

# Typing for the resulting batch data after encode_batch()
@edataclass
class BatchedPackedTaskSample(Batch):
    """Dataclass to store a batch of packed samples.

    N = Batch size
    P = Number of samples in the packed sample
    seq_len = Maximum sequence length
    num_imgs = Number of images across all samples in the packed sample
    """

    # Input tokens packed and padded (N, seq_len)
    tokens: torch.Tensor
    # Target tokens packed and padded (N, seq_len)
    labels: torch.Tensor
    # Maximum length across sub-samples (N,)
    max_lengths: list[int]
    # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    cu_lengths: list[list[int]]
    # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    cu_lengths_padded: list[list[int]]

    # All image tiles stacked into a single tensor (num_tiles, C, H, W)
    imgs: torch.Tensor
    # Number of tiles per image (N, num_imgs)
    num_tiles: list[list[int]]
    # Size of each image tile
    imgs_sizes: list[tuple[int, int]]
    # Maximum length across sub-samples. (N,)
    vision_max_lengths: list[int]
    # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (N, num_imgs)
    vision_cu_lengths: list[list[int]]

    # Number of samples in the packed batch
    samples_seen: int
    # Original sample lengths, padded with zeros along the batch dimension.
    sample_lengths: torch.Tensor

    # Whether the batch has a padded image
    has_pad_img: bool

    # "Batched" version of number of frames used per VideoMedia / ImageMedia (1 frame for ImageMedia)
    num_frames: list[list[int]]

class ConversationTurn(TypedDict):
    """One turn of a conversation passed to the multimodal tokenizer."""

    role: Literal["user", "assistant", "system", "tool"]
    content: Union[str, list[dict[str, Union[str, int]]]]


class MultiModalTaskEncoder(
    DefaultTaskEncoder[
        ConversationSample,
        Union[ConversationTaskSample, PackedTaskSample],
        BatchedPackedTaskSample,
        dict,
    ]
):
    """A simple task encoder for VLMs (LlavaSample only)."""

    decoder = SampleDecoder(image_decode="pil", av_decode="AVDecoder", guess_content=True)

    cookers = [
        Cooker(cook_eagle, has_subflavors={"cook": "eagle"}),
        Cooker(cook_conversation, has_subflavors={"cook": "conversation"}),
        Cooker(
            cook_general_conversations_webdataset,
            has_subflavors={"cook": "general_conversations_webdataset"},
        ),
        Cooker(
            cook_general_conversations_jsonl, has_subflavors={"cook": "general_conversations_jsonl"}
        ),
        Cooker(
            cook_general_conversations_jsonl_explicit_loss_v1,
            has_subflavors={"cook": "general_conversations_jsonl_explicit_loss_v1"},
        ),
        Cooker(cook_openai_messages_jsonl, has_subflavors={"cook": "openai_messages_jsonl"}),
        Cooker(
            cook_openai_messages_offline_packed_jsonl,
            has_subflavors={"cook": "openai_messages_offline_packed_jsonl"},
        ),
    ]

    def __init__(self, is_val: bool = False, tiling_augment_prob: float = 0.4):
        super().__init__()
        self.is_val = is_val
        self.args = get_args()
        self.tokenizer = get_tokenizer()
        with open(self.args.prompt_path, "r") as f:
            self.manual_prompts = json.load(f)
        # TODO: There is four seq lengths now: seq_length, decoder_seq_length, dataloader_seq_length, packing_seq_length
        # Why do we need all of these? Can we simplify?
        # The docs are not clear.
        self.dataloader_seq_length = self.args.dataloader_seq_length
        # TODO: It's unclear why we need this separately. This should be the same as dataloader_seq_length which should be decoder_seq_length?
        self.packing_seq_length = self.args.packing_seq_length
        self.is_packing_enabled = (
            self.args.packing_buffer_size is not None
            and self.args.packing_buffer_size > 0
            and not is_val
        )
        if self.dataloader_seq_length and self.packing_seq_length:
            assert (
                self.dataloader_seq_length >= self.packing_seq_length
            ), "dataloader sequence length must be greater than or equal to the packing sequence length"

        if self.is_packing_enabled:
            assert self.packing_seq_length > 0, "packing sequence length must be set"

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w
        self.img_token_id = self.tokenizer.image_token_index
        # This map is used to reduce the number of tiles used per image if the number of tokens is
        # larger than the decoder_seq_length.
        self.num_tiles_degradation_map = {12: 8, 8: 6, 6: 4, 4: 2, 2: 1, 1: 1}

        self.tiling_augment_prob = tiling_augment_prob

        # Create the image tiling strategy using the refactored function
        self.image_tiling_strategy = create_image_tiling_strategy(self.args)

        # Validate temporal compression settings
        temporal_patch_size = getattr(self.args, 'video_temporal_patch_size', 1)
        if temporal_patch_size > 1:
            # video_min_num_frames must be at least temporal_patch_size
            assert self.args.video_min_num_frames >= temporal_patch_size, (
                f"video_min_num_frames ({self.args.video_min_num_frames}) must be >= "
                f"video_temporal_patch_size ({temporal_patch_size})"
            )
            # video_max_num_frames should be a multiple of temporal_patch_size
            assert self.args.video_max_num_frames % temporal_patch_size == 0, (
                f"video_max_num_frames ({self.args.video_max_num_frames}) must be a multiple of "
                f"video_temporal_patch_size ({temporal_patch_size})"
            )

        packing_algorithm_parameters = _parse_packing_algorithm_parameters(
            getattr(self.args, "packing_algorithm_parameters", "")
        )
        if self.args.packing_knapsack_algorithm == "greedy_knapsack":
            self.packing_knapsack_algorithm = greedy_knapsack
        elif self.args.packing_knapsack_algorithm == "balanced_greedy_knapsack":
            balanced_knapsack_delta = _pop_int_packing_algorithm_parameter(
                packing_algorithm_parameters, "balanced_knapsack_delta", 20
            )
            self.packing_knapsack_algorithm = partial(
                balanced_greedy_knapsack, delta=balanced_knapsack_delta
            )
        elif self.args.packing_knapsack_algorithm == "bucketing_greedy_knapsack":
            self.packing_knapsack_algorithm = bucketing_greedy_knapsack
        elif self.args.packing_knapsack_algorithm == "streaming_prompt_dedup_first_fit_knapsack":
            self.packing_knapsack_algorithm = streaming_prompt_dedup_first_fit_knapsack
        else:
            raise ValueError(f"Unknown knapsack algorithm: {self.args.packing_knapsack_algorithm}")
        if packing_algorithm_parameters:
            unused_parameters = ", ".join(sorted(packing_algorithm_parameters))
            raise ValueError(f"Unused packing algorithm parameter(s): {unused_parameters}")
        self.shuffle_packed_samples = (
            self.args.packing_knapsack_algorithm != "streaming_prompt_dedup_first_fit_knapsack"
        )

        print(f"{type(self).__name__} initialized. Energon Version: {energon_version}")

    @staticmethod
    def get_seq_frames_v3(
        total_duration: float, desired_num_frames: int = 0, temporal_jitter: bool = False
    ) -> torch.Tensor:
        """
        Calculate the timestamps of frames to extract from a video.

        Parameters:
            total_duration: Total duration of the video.
            desired_num_frames: Desired number of frames to extract.
            temporal_jitter: Whether to jitter the frames.

        Returns:
            List of timestamps of frames to extract.
        """
        # Calculate the size of each segment from which a frame will be extracted
        seg_size = float(total_duration - 1) / desired_num_frames
        # print(f"seg_size: {seg_size}")

        # Middle of each segment
        seq = seg_size * (torch.arange(desired_num_frames).to(dtype=torch.float32) + 0.5)

        if temporal_jitter:
            jitter_size = seg_size * 0.5
            # Generate random shifts for all frames at once
            seq += torch.rand(len(seq), dtype=torch.float32) * (jitter_size * 2) - jitter_size
            # Clip values to valid range
            seq = torch.clamp(seq, 0, total_duration)

        return seq

    def video_to_frames(self, video: VideoMedia) -> Tuple[list[Media], int]:
        """Convert a video to a list of video frame and text according to the settings."""
        video_duration = video.metadata["video_duration"]
        video_num_frames = video.metadata["video_num_frames"]

        if video_num_frames is None or video_duration is None:
            raise ValueError(
                f"Missing video metadata (num_frames={video_num_frames}, duration={video_duration}) for {video.value}"
            )

        start_time = 0
        if video.start_time is not None:
            if video.end_time is not None:
                video_duration = video.end_time - video.start_time
            else:
                video_duration = video_duration - video.start_time
            start_time = video.start_time
        elif video.end_time is not None:
            video_duration = video.end_time

        temporal_patch_size = getattr(self.args, 'video_temporal_patch_size', 1)

        # Build unified set of possible aug_scale_frames_up values and draw once.
        # aug_scale_frames_up > 1 = scale UP frames (more frames, lower resolution per frame)
        # aug_scale_frames_up < 1 = scale DOWN frames (fewer frames, higher resolution per frame)
        # aug_scale_frames_up = 1 = identity (no augmentation)
        # E.g. --video-aug-scale-frames-up 4 --video-aug-scale-resolution-up 3
        #   gives {1/3, 1/2, 1, 2, 3, 4} (6 values, uniform draw)
        video_aug_scale_frames_up = getattr(self.args, 'video_aug_scale_frames_up', None)
        video_aug_scale_resolution_up = getattr(self.args, 'video_aug_scale_resolution_up', None)

        scale_values = [1.0]
        if video_aug_scale_frames_up is not None and video_aug_scale_frames_up > 1:
            scale_values.extend(float(s) for s in range(2, video_aug_scale_frames_up + 1))
        if video_aug_scale_resolution_up is not None and video_aug_scale_resolution_up > 1:
            scale_values.extend(1.0 / s for s in range(2, video_aug_scale_resolution_up + 1))

        aug_scale_frames_up = random.choice(scale_values)

        resolution_only = getattr(self.args, 'video_aug_scale_resolution_only', False)

        if video_num_frames < self.args.video_min_num_frames:
            # Some videos are too short or low-fps, just use the whole video, like sthv2, smit, llava-hound (2fps)
            sample_num_frames = video_num_frames
            aug_scale_frames_up = 1.0  # Reset to not change image size unnecessarily
            effective_max_num_frames = self.args.video_max_num_frames
            effective_fps = self.args.video_default_fps
        else:
            # We sample frames like: sample_num_frames = min(max(fps * duration, min_frames), max_frames)
            #   where `fps` and `max_frames` are hyperparameters. To always increase `sample_num_frames`
            #   by `scale`, we have to scale both `fps` and `max_frames` and then we apply the min/max
            #   which ensures exactly one is chosen (and either way the scaling factor is applied)
            if resolution_only:
                # Resolution-only mode: only change patch count, keep frame count unchanged
                effective_max_num_frames = self.args.video_max_num_frames
                effective_fps = self.args.video_default_fps
            else:
                # aug_scale_frames_up > 1: more frames, lower resolution per frame
                # aug_scale_frames_up < 1: fewer frames, higher resolution per frame
                effective_max_num_frames = max(
                    self.args.video_min_num_frames,
                    int(self.args.video_max_num_frames * aug_scale_frames_up),
                )
                effective_fps = max(1, int(self.args.video_default_fps * aug_scale_frames_up))

            default_sample_num_frames = int(effective_fps * video_duration)
            sample_num_frames = min(
                max(default_sample_num_frames, self.args.video_min_num_frames),
                effective_max_num_frames,
            )

        # Round to multiple of temporal patch size for temporal compression
        # Only round if not already a multiple; prefer rounding up if within limits, else round down
        if temporal_patch_size > 1:
            if sample_num_frames % temporal_patch_size != 0:
                rounded_down = (sample_num_frames // temporal_patch_size) * temporal_patch_size
                rounded_up = rounded_down + temporal_patch_size
                if rounded_up <= video_num_frames and rounded_up <= effective_max_num_frames:
                    sample_num_frames = rounded_up
                else:
                    sample_num_frames = max(temporal_patch_size, rounded_down)

            # Verify the final num frames is valid (whether we rounded or not)
            assert sample_num_frames % temporal_patch_size == 0, (
                f"sample_num_frames ({sample_num_frames}) must be a multiple of "
                f"temporal_patch_size ({temporal_patch_size})"
            )
            assert sample_num_frames >= temporal_patch_size, (
                f"sample_num_frames ({sample_num_frames}) must be at least "
                f"temporal_patch_size ({temporal_patch_size})"
            )
            assert sample_num_frames <= effective_max_num_frames, (
                f"sample_num_frames ({sample_num_frames}) exceeds "
                f"effective_max_num_frames ({effective_max_num_frames})"
            )

        frame_timestamps = self.get_seq_frames_v3(
            video_duration, sample_num_frames, self.args.video_frame_temporal_jitter
        )
        frame_timestamps += start_time

        video_fps = video.video_fps
        video_prompt_version = getattr(self.args, 'video_prompt_version', 2)

        def make_video_frame_media(i, timestamp):
            metadata = {"video_width": video.video_width, "video_height": video.video_height}
            if aug_scale_frames_up != 1.0:
                metadata["video_aug_scale_frames_up"] = aug_scale_frames_up
            return VideoFrameMedia(
                value=video.value,
                timestamp=float(timestamp),
                frame_index=float(timestamp * video_fps),
                sample_index=i,
                metadata=metadata,
            )

        if video_prompt_version == 1 or temporal_patch_size == 1:
            # Version 1 (previous): Each frame on its own line.
            #
            # This returns a list like:
            #   ["This is a video:\n", "Frame 1 sampled at 0.00 seconds: ", VideoFrameMedia(0), "\n",
            #    "Frame 2 sampled at 0.88 seconds: ", VideoFrameMedia(1), "\n", ...]
            #
            # During preencode_sample(), each VideoFrameMedia is processed:
            #   - If (sample_index + 1) % temporal_patch_size == 0: adds IMAGE_TOKEN to text
            #   - Always appends to image_media list (for image processing)
            #   - This works because we ensure len(frame_timestamps) % temporal_patch_size == 0
            #
            # So with temporal_patch_size=2, the final TEXT prompt becomes:
            #   "This is a video:\n"
            #   "Frame 1 sampled at 0.00 seconds: <image>\n"  <- token added (0 % 2 == 0)
            #   "Frame 2 sampled at 0.88 seconds: \n"         <- no token (1 % 2 != 0)
            #   "Frame 3 sampled at 1.76 seconds: <image>\n"  <- token added (2 % 2 == 0)
            #   "Frame 4 sampled at 2.64 seconds: \n"         <- no token (3 % 2 != 0)
            #   "Frame 5 sampled at 3.52 seconds: <image>\n"  <- token added (4 % 2 == 0)
            #
            # All 5 frames' image data is still processed; frames 0+1 are combined into tubelet 0,
            # frames 2+3 into tubelet 1, etc. Each tubelet embedding replaces one <image> token.
            return ["This is a video:\n"] + [
                media
                for i, timestamp in enumerate(frame_timestamps)
                for media in (
                    f"Frame {i + 1} sampled at {timestamp:.2f} seconds: ",
                    # TODO: Orginal eagle repro
                    # f"Frame {i + 1} sampled at {timestamp:.2f} seconds: <image-{i}>",
                    make_video_frame_media(i, timestamp),
                    "\n",
                )
            ], len(frame_timestamps)
        elif video_prompt_version == 2 and temporal_patch_size > 1:
            # Version 2 (new default): Group T frames with "and", one <image> per group.
            # This also produces the same output as version 1 if temporal_patch_size == 1

            # This returns a list like:
            #   ["This is a video:\n",
            #    "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.88 seconds: ",
            #    VideoFrameMedia(0), VideoFrameMedia(1), "\n",
            #    "Frame 3 sampled at 1.76 seconds and frame 4 sampled at 2.64 seconds: ",
            #    VideoFrameMedia(2), VideoFrameMedia(3), "\n",
            #    "Frame 5 sampled at 3.52 seconds: ", VideoFrameMedia(4), "\n"]
            #
            # Same processing as version 1: IMAGE_TOKEN added only when sample_index % T == 0.
            # Final TEXT prompt becomes:
            #   "This is a video:\n"
            #   "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.88 seconds: <image>\n"
            #   "Frame 3 sampled at 1.76 seconds and frame 4 sampled at 2.64 seconds: <image>\n"
            #   "Frame 5 sampled at 3.52 seconds: <image>\n"
            result = ["This is a video:\n"]
            T = temporal_patch_size
            for group_start in range(0, len(frame_timestamps), T):
                # Build text for this group
                group_text_parts = []
                group_media = []
                for j in range(T):
                    sample_idx = group_start + j
                    if sample_idx < len(frame_timestamps):
                        timestamp = frame_timestamps[sample_idx]
                        frame_str = "Frame" if j == 0 else "frame"
                        group_text_parts.append(
                            f"{frame_str} {sample_idx + 1} sampled at {timestamp:.2f} seconds"
                        )
                        group_media.append(make_video_frame_media(sample_idx, timestamp))
                if group_text_parts:
                    result.append(" and ".join(group_text_parts) + ": ")
                    result.extend(group_media)
                    result.append("\n")
            return result, len(frame_timestamps)
        else:
            raise NotImplementedError(
                f"Video prompt version {video_prompt_version} with"
                f" temporal patch size {temporal_patch_size} is not implemented"
            )

    @staticmethod
    def _get_prompt_hash(sample: ConversationSample) -> str:
        prompt_text_parts = []
        for message in sample.conversation:
            if message.sender not in ("user", "tool"):
                continue
            prompt_text_parts.extend(
                fragment for fragment in message.fragments if isinstance(fragment, str)
            )
        prompt_text = "".join(prompt_text_parts)
        return hashlib.md5(prompt_text.encode("utf-8")).hexdigest()

    @stateless(restore_seeds=True)
    def preencode_sample(
        self, sample: ConversationSample
    ) -> PreEncodedTaskSample | PreEncodedOfflinePackedTextSample:
        """Encode sample."""

        identity_filter_keywords = getattr(self.args, "filter_identity_keywords", None) or []
        if _has_filtered_identity_keyword(sample, identity_filter_keywords):
            raise ValueError(
                "Sample from identity-filtered dataset contains a filtered "
                f"assistant identity keyword: {sample.__key__}"
            )

        explicit_assistant_loss = _validate_loss_mask_subflavors(sample.__subflavors__)
        if sample.__subflavors__.get("offline_packed_messages", False):
            return self._preencode_offline_packed_text_sample(sample)

        # In-place convert VideoMedia to VideoFrameMedia (and text)
        # Some really large video files cause decoding to take a long time potentially leading to issues.
        allow_large_videos = getattr(self.args, "allow_large_videos", False)
        data_augment = sample.__subflavors__.get("data_augment", False) and not self.is_val
        tiling_augment_prob = sample.__subflavors__.get(
            "tiling_augment_prob", self.tiling_augment_prob
        )
        train_only_on_last_assistant_turn = sample.__subflavors__.get(
            "train_only_on_last_assistant_turn", False
        )
        tool_response_as_turn_boundary = sample.__subflavors__.get(
            "tool_response_as_turn_boundary", False
        )
        skip_chat_template = sample.__subflavors__.get("skip_chat_template", False)
        aggregated_num_frames = []
        prompt_hash = self._get_prompt_hash(sample)

        assistant_turn_loss: list[bool] | None = None
        if explicit_assistant_loss:
            if (
                sample.__subflavors__.get("assistant_loss_mask_field")
                != EXPLICIT_ASSISTANT_LOSS_FIELD
            ):
                raise ValueError(
                    "explicit assistant loss requires "
                    f"assistant_loss_mask_field={EXPLICIT_ASSISTANT_LOSS_FIELD}"
                )
            if train_only_on_last_assistant_turn:
                raise ValueError(
                    "explicit assistant loss is incompatible with "
                    "train_only_on_last_assistant_turn"
                )
            assistant_turn_loss = []
            for message in sample.conversation:
                if message.sender == "assistant":
                    if type(message.loss) is not bool:
                        raise ValueError(
                            "explicit assistant loss requires a boolean loss flag "
                            f"on every assistant turn in sample {sample.__key__}"
                        )
                    assistant_turn_loss.append(message.loss)
                elif message.loss is not None:
                    raise ValueError(
                        "explicit assistant loss flags are only valid on assistant "
                        f"turns in sample {sample.__key__}"
                    )
            if not any(assistant_turn_loss):
                raise ValueError(
                    f"explicit assistant loss has no loss=true turn in sample {sample.__key__}"
                )
        elif any(message.loss is not None for message in sample.conversation):
            raise ValueError(
                "conversation contains explicit loss flags without "
                f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}"
            )

        # We tentatively extract the first message if it's a system prompt and use this rather than
        # the default. After this, we expect no system prompt in the conversation.

        has_system_message = sample.conversation[0].sender == "system"
        system_prompt = ""
        if has_system_message:
            system_prompt = sample.conversation[0].fragments[0]
            sample.conversation = sample.conversation[1:]

        structured_conversation: list[ConversationTurn] = [
            {"role": "system", "content": system_prompt}
        ]

        for message in sample.conversation:
            idx = 0
            while idx < len(message.fragments):
                fragment = message.fragments[idx]

                if isinstance(fragment, VideoMedia):
                    if not allow_large_videos and fragment.clip_duration > 60 * 10:
                        raise ValueError(f"Video is too large: {fragment.value}")

                    frames, num_frames = self.video_to_frames(fragment)
                    message.fragments[idx : idx + 1] = frames
                    idx += len(frames)
                    aggregated_num_frames.append(num_frames)
                else:
                    if isinstance(fragment, (ImageMedia, VideoFrameMedia)):
                        # Image or a single frame
                        aggregated_num_frames.append(1)
                    elif isinstance(fragment, str):
                        # Text fragment
                        pass
                    elif isinstance(fragment, bytes):
                        raise ValueError(
                            f"Could not convert bytes to known media type: {fragment[:100]!r}"
                        )
                    else:
                        raise ValueError(
                            f"Unexpected media type: {type(fragment)}. Fragment: {fragment}"
                        )
                    idx += 1

        image_media: list[ImageMedia | VideoFrameMedia] = []
        # For temporal compression: track video frame count to emit one IMAGE_TOKEN per tubelet
        temporal_patch_size = getattr(self.args, 'video_temporal_patch_size', 1)

        # Format the conversation as a list of "user" / "assistant" turns.
        for message in sample.conversation:
            if not self.args.relax_sender_check:
                assert message.sender in [
                    "user",
                    "assistant",
                    "tool",
                ], f"unexpected sender {message.sender} in {sample.conversation}"

            content_parts: list[dict[str, Union[str, int]]] = []

            for fragment in message.fragments:
                if isinstance(fragment, str):
                    if not fragment:
                        continue
                    content_parts.append({"type": "text", "text": fragment})
                elif isinstance(fragment, ImageMedia):
                    content_parts.append({"type": "image"})
                    image_media.append(fragment)
                elif isinstance(fragment, VideoFrameMedia):
                    # With temporal compression, only add IMAGE_TOKEN at tubelet boundaries
                    #   (every T frames). Use sample_index which is the consecutive index within
                    #   sampled frames (0, 1, 2, ...) and resets to 0 for each video.
                    # This works because:
                    #   1) _group_video_frame_params_into_tubelets() groups per-frame params to match IMAGE_TOKEN count
                    #   2) Grouped params used for sequence length calculations and padding
                    #   3) All frames (un-grouped) are passed as pixels to RADIO.forward()
                    #   4) RADIO._apply_temporal_grouping() combines every T frames
                    #   5) RADIO returns updated imgs_sizes/num_frames for LLaVAModel
                    if temporal_patch_size > 1:
                        # Add the image token for last frame in every group of T
                        if (
                            fragment.sample_index + 1
                        ) % temporal_patch_size == 0:  # Next frame == new group
                            content_parts.append({"type": "image"})
                    else:
                        # Add the image token for every frame
                        content_parts.append({"type": "image"})
                    image_media.append(fragment)
                elif isinstance(fragment, VideoMedia):
                    raise ValueError("VideoMedia should have been converted to VideoFrameMedia.")
            if self.args.only_keep_samples_with_img and len(image_media) == 0:
                raise ValueError(f"Sample has no image: {sample.__key__}")

            prompt_format = self.args.tokenizer_prompt_format

            if (
                not skip_chat_template
                and prompt_format == "nemotron6-moe"
                and message.sender == "assistant"
                and not self.args.relax_thinking_trace_check
            ):
                if any(part["type"] != "text" for part in content_parts):
                    raise ValueError(
                        f"Assistant turns with multimodal content are not supported in sample {sample.__key__}"
                    )
                content = "".join(part["text"] for part in content_parts)
                think_start_count = content.count("<think>")
                think_end_count = content.count("</think>")
                if think_start_count == 0 and think_end_count == 0:
                    # Add think tags in non-reasoning mode, if missing
                    content = "<think></think>" + content.strip()
                else:
                    # There should be exactly one of each, otherwise it's invalid
                    assert think_start_count == 1 and think_end_count == 1, (
                        f"Found sample with {think_start_count} <think> tags and {think_end_count} </think> tags in sample with "
                        f"key: {sample.__key__} and subflavors: {sample.__subflavors__}"
                    )

                    # </think> should come after <think>, otherwise it's invalid
                    start_idx = content.find("<think>")
                    end_idx = content.find("</think>")
                    assert start_idx < end_idx, (
                        f"Found sample with </think> tags before </think> tags in sample with "
                        f"key: {sample.__key__} and subflavors: {sample.__subflavors__}"
                    )

                    content = _normalize_thinking_trace(
                        content,
                        thinking_trace_format=getattr(
                            self.args, "thinking_trace_format", "normalized"
                        ),
                    )
                content_parts = [{"type": "text", "text": content}]

            structured_conversation.append({"role": message.sender, "content": content_parts})

        input_ids, target = self.tokenizer.tokenize_conversation(
            structured_conversation,
            True,
            False,
            train_only_on_last_assistant_turn=train_only_on_last_assistant_turn,
            tool_response_as_turn_boundary=tool_response_as_turn_boundary,
            assistant_turn_loss=assistant_turn_loss,
            skip_chat_template=skip_chat_template,
        )
        input_ids = torch.as_tensor(input_ids)
        target = torch.as_tensor(target)
        if assistant_turn_loss is not None:
            self._validate_explicit_loss_target_spans(
                target,
                expected_span_count=sum(assistant_turn_loss),
                sample_key=sample.__key__,
                phase="tokenization",
            )
        if len(target) == 0 or not bool((target != IGNORE_INDEX).any().item()):
            raise NoTrainableTokensError(
                f"target is empty: {target}, DETOKENIZED:\n\n{self.tokenizer.detokenize(input_ids)}\n\nCONVERSATION:\n\n"
                + "".join([f"{m.sender}: {m.fragments}\n" for m in sample.conversation])
            )

        max_image_token_allowed = self.args.decoder_seq_length - len(input_ids) - 4
        image_media_params = self.image_tiling_strategy.compute_params(
            image_media,
            max_image_token_allowed,
            data_augment=data_augment,
            tiling_augment_prob=tiling_augment_prob,
        )

        # With temporal compression, we emit one IMAGE_TOKEN per tubelet (grouped frame), not per frame.
        # Create grouped params for token counting (one per IMAGE_TOKEN), but keep
        #   ungrouped params for storage (one per frame, needed for frame loading).
        # This is necessary to get accurate token / embedding / image counts when we're calling
        #   compute_params() and process_media() on individual video frames, rather than entire videos
        if temporal_patch_size > 1:
            grouped_params_for_tokens = self._group_video_frame_params_into_tubelets(
                image_media, image_media_params, temporal_patch_size
            )
        else:
            grouped_params_for_tokens = image_media_params

        input_ids, target = self._truncate_to_decoder_seq_len(
            input_ids=input_ids,
            target=target,
            image_tiling_params=grouped_params_for_tokens,
            sample_key=sample.__key__,
            sample_subflavors=sample.__subflavors__,
        )

        if assistant_turn_loss is not None:
            self._validate_explicit_loss_target_spans(
                target,
                expected_span_count=sum(assistant_turn_loss),
                sample_key=sample.__key__,
                phase="truncation",
            )

        # We need to ensure that there are at least some trainable tokens in the sample.
        has_trainable_tokens = self._target_has_trainable_tokens(
            input_ids, target, grouped_params_for_tokens
        )

        if not has_trainable_tokens:
            raise NoTrainableTokensError(
                f"Sample has no trainable tokens: {self.tokenizer.detokenize(input_ids)}"
            )

        total_len, total_len_padded, input_ids, target = self._pad_for_context_parallel_and_fp8(
            input_ids, target, grouped_params_for_tokens
        )

        # Store UNGROUPED params (one per frame) for frame loading in _load_media()
        # The reason we must use ungrouped params for `images` here is that if we used the
        #   grouped params, the line `frame_clips = media_value.get_clips(` in _load_media() would
        #   only load one of the frames in the group (the last one) instead of both, breaking things
        # The reason we must use grouped params for `sample` and the other metadata is that those
        #   are used for the LLM's packed seq params and thus must reflect the sequence lengths and
        #   other metadata that will come out of the vision encoder, post temporal compression
        # This whole metadata tracking for grouped vs. ungrouped params is unfortunate, and only
        #   necessary because we're passing single frames to RADIO and doing the temporal grouping
        #   there. In the near-future we should refactor this so we always pass (T,B,C,H,W) tensors
        #   to the vision encoder, to simplify everything.
        return PreEncodedTaskSample.derive_from(
            sample,  # UNGROUPED (need per-image metadata for vision encoder)
            tokens=input_ids,  # Grouped
            labels=target,  # Grouped
            images=image_media_params,  # UNGROUPED (need per-image metadata for vision encoder)
            total_len=total_len,  # Grouped
            total_len_padded=total_len_padded,  # Grouped
            num_frames=aggregated_num_frames,  # UNGROUPED (need per-image metadata for vision encoder)
            prompt_hash=prompt_hash,
        )

    def _split_offline_packed_conversations(
        self, sample: ConversationSample
    ) -> list[list[ConversationTurn]]:
        conversations: list[list[ConversationTurn]] = []
        current: list[ConversationTurn] = []

        for message in sample.conversation:
            if message.loss is not None:
                raise ValueError(
                    "openai_messages_offline_packed_jsonl does not support "
                    f"explicit assistant loss in sample {sample.__key__}"
                )
            if message.sender == "system" and current:
                conversations.append(current)
                current = []

            if any(not isinstance(fragment, str) for fragment in message.fragments):
                raise ValueError(
                    "openai_messages_offline_packed_jsonl supports text-only messages; "
                    f"got {message.fragments!r} in sample {sample.__key__}"
                )
            current.append({"role": message.sender, "content": "".join(message.fragments)})

        if current:
            conversations.append(current)
        if not conversations:
            raise ValueError(f"Offline packed sample has no conversations: {sample.__key__}")
        return conversations

    def _preencode_offline_packed_text_sample(
        self, sample: ConversationSample
    ) -> PreEncodedOfflinePackedTextSample:
        """Tokenize one already-packed Nano SFT row without online repacking."""
        assert not self.is_packing_enabled, (
            "openai_messages_offline_packed_jsonl rows are already packed; "
            "do not pass --packing-buffer-size with this cooker."
        )

        pack_length = self.args.decoder_seq_length
        pad = self.tokenizer.pad

        pack_tokens: list[int] = []
        pack_targets: list[int] = []
        cu_lengths = [0]
        sample_lengths: list[int] = []

        for conversation in self._split_offline_packed_conversations(sample):
            tokens, targets = self.tokenizer.tokenize_conversation(
                conversation,
                True,
                False,
                train_only_on_last_assistant_turn=sample.__subflavors__.get(
                    "train_only_on_last_assistant_turn", False
                ),
                tool_response_as_turn_boundary=sample.__subflavors__.get(
                    "tool_response_as_turn_boundary", False
                ),
                skip_chat_template=sample.__subflavors__.get("skip_chat_template", False),
            )
            tokens = torch.as_tensor(tokens, dtype=torch.int64)
            targets = torch.as_tensor(targets, dtype=torch.int64)
            if len(targets) == 0 or (targets != IGNORE_INDEX).sum() == 0:
                continue

            tokens_list = [int(token) for token in tokens.tolist()]
            targets_list = [int(target) for target in targets.tolist()]
            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            if getattr(self.args, "context_parallel_size", 1) > 1:
                pad_granularity = self.args.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    pack_tokens.extend([pad] * pad_len)
                    pack_targets.extend([pad] * pad_len)

            current_length = len(pack_tokens)
            cu_lengths.append(current_length)
            sample_lengths.append(current_length - cu_lengths[-2])

            if len(pack_tokens) >= pack_length + 1:
                pack_tokens = pack_tokens[:pack_length]
                pack_targets = pack_targets[:pack_length]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                cu_lengths[-1] = len(pack_tokens) - 1
                sample_lengths[-1] = cu_lengths[-1] - cu_lengths[-2]
                break

        if len(cu_lengths) < 2:
            raise NoTrainableTokensError(
                f"Offline packed sample has no trainable conversations: {sample.__key__}"
            )

        if len(pack_tokens) < pack_length + 1:
            pad_len = pack_length + 1 - len(pack_tokens)
            pack_tokens.extend([pad] * pad_len)
            pack_targets.extend([pad] * pad_len)
            cu_lengths[-1] = len(pack_tokens) - 1
            sample_lengths[-1] = cu_lengths[-1] - cu_lengths[-2]

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1

        cu_lengths_tensor = torch.tensor(cu_lengths, dtype=torch.int32)
        adjacent_diffs = cu_lengths_tensor[1:] - cu_lengths_tensor[:-1]
        max_length = int(adjacent_diffs.max().item())

        return PreEncodedOfflinePackedTextSample.derive_from(
            sample,
            tokens=torch.tensor(pack_tokens[:-1], dtype=torch.int64),
            labels=torch.tensor(pack_targets, dtype=torch.int64),
            max_length=max_length,
            cu_lengths=cu_lengths_tensor,
            cu_lengths_padded=cu_lengths_tensor.clone(),
            samples_seen=torch.tensor(len(sample_lengths), dtype=torch.int32),
            sample_lengths=torch.tensor(sample_lengths, dtype=torch.int32),
        )

    @stateless(restore_seeds=True)
    def preencode_sample_for_packing(self, sample: ConversationSample):
        encoded_sample = self.preencode_sample(sample)
        samples = getattr(encoded_sample, "samples", None)
        if samples is not None:
            yield from samples
        else:
            yield encoded_sample

    def build_encode_sample(self, dataset, *, worker_config):
        if self.is_packing_enabled:
            return MapDataset(
                dataset,
                self.preencode_sample_for_packing,
                worker_config=worker_config,
                stateless_map_fn=True,
            )
        return super().build_encode_sample(dataset, worker_config=worker_config)

    @stateless(restore_seeds=True)
    def postencode_sample(
        self, sample: PreEncodedTaskSample | PreEncodedOfflinePackedTextSample
    ) -> PackedTaskSample:
        if isinstance(sample, PreEncodedOfflinePackedTextSample):
            return PackedTaskSample.derive_from(
                sample,
                __key__=[sample.__key__],
                tokens=sample.tokens,
                labels=sample.labels,
                imgs=[],
                num_tiles=[],
                num_frames=[],
                max_length=sample.max_length,
                cu_lengths=sample.cu_lengths,
                cu_lengths_padded=sample.cu_lengths_padded,
                samples_seen=sample.samples_seen,
                sample_lengths=sample.sample_lengths,
            )

        self._load_media(sample)

        data_augment = sample.__subflavors__.get("data_augment", False) and not self.is_val

        # Transform the images
        image_tiles = []
        for media_idx, media in enumerate(sample.images):
            # Debug: Save images if DEBUG environment variable is set to 1
            if os.environ.get("DEBUG_DATALOADER", "0") == "1":
                try:
                    self._debug_save_image(media, media_idx, sample.__key__, data_augment)
                except Exception as e:
                    print(f"[DEBUG] Failed to save debug image: {e}")

            image_tiles.extend(
                self.image_tiling_strategy.apply_params(media, data_augment=data_augment)
            )

        # Make this a packed sample (if used without packing, it will be the same next code)
        return PackedTaskSample.derive_from(
            sample,
            __key__=[sample.__key__],
            tokens=sample.tokens,
            labels=sample.labels,
            imgs=image_tiles,
            num_tiles=[media.num_tiles for media in sample.images],
            num_frames=sample.num_frames,
            max_length=sample.total_len_padded,
            cu_lengths=torch.tensor([0, sample.total_len], dtype=torch.int32),
            cu_lengths_padded=torch.tensor([0, sample.total_len_padded], dtype=torch.int32),
            samples_seen=torch.tensor(1, dtype=torch.int32),
            sample_lengths=torch.tensor([sample.total_len], dtype=torch.int32),
        )

    def _debug_save_image(self, media, media_idx, sample_key, data_augment):
        """Save debug images with original and transformed sizes."""
        from datetime import datetime

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        # Create debug directory if it doesn't exist
        debug_dir = os.environ.get(
            "DEBUG_DATALOADER_DIR", os.path.join(os.getcwd(), "debug_images")
        )
        os.makedirs(debug_dir, exist_ok=True)

        # Get original image and size
        original_image = media.media.value

        if isinstance(media.media, ImageMedia):
            orig_width, orig_height = media.media.width, media.media.height
            media_type = "image"
            if os.environ.get("DEBUG_DATALOADER_VIDEO_ONLY", "0") == "1":
                return
        elif isinstance(media.media, VideoFrameMedia):
            orig_width, orig_height = media.media.video_width, media.media.video_height
            media_type = "video"
            if os.environ.get("DEBUG_DATALOADER_IMAGE_ONLY", "0") == "1":
                return
        else:
            return  # Skip if not a supported media type

        # Apply the transformation to get the processed tiles
        transformed_tiles = self.image_tiling_strategy.apply_params(
            media, data_augment=data_augment
        )

        # Get the normalization stats for denormalization
        from .image_processing import pixel_statistics

        pixel_mean, pixel_std = pixel_statistics.get(
            self.args.vision_model_type, ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        )

        # Convert lists to tensors for denormalization
        mean = torch.tensor(pixel_mean).view(3, 1, 1)
        std = torch.tensor(pixel_std).view(3, 1, 1)

        # Create a figure with subplots
        num_tiles = len(transformed_tiles)
        fig, axes = plt.subplots(1, num_tiles + 1, figsize=(5 * (num_tiles + 1), 5))
        if num_tiles == 0:
            axes = [axes]

        # Plot original image
        ax = axes[0] if num_tiles > 0 else axes
        ax.imshow(original_image)
        ax.set_title(
            f"Original Image\nSize: {orig_width}x{orig_height}", fontsize=10, fontweight='bold'
        )
        ax.axis('off')

        # Plot transformed tiles
        for tile_idx, tile_tensor in enumerate(transformed_tiles):
            ax = axes[tile_idx + 1]

            # Denormalize the tensor: img = img * std + mean
            denormalized = tile_tensor * std + mean

            # Clamp to [0, 1] range
            denormalized = torch.clamp(denormalized, 0, 1)

            # Convert to numpy and transpose from CxHxW to HxWxC
            tile_image = denormalized.permute(1, 2, 0).cpu().numpy()

            # Get the new size
            new_height, new_width = tile_image.shape[:2]

            ax.imshow(tile_image)
            ax.set_title(
                f"Tile {tile_idx + 1}/{num_tiles}\n"
                f"Size: {tile_tensor.shape[2]}x{tile_tensor.shape[1]}\n"
                f"Original: {orig_width}x{orig_height}",
                fontsize=9,
                fontweight='bold',
            )
            ax.axis('off')

        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_key = str(sample_key).replace("/", "_").replace("\\", "_")[:50]
        filename = f"{safe_key}_{media_type}_media{media_idx}_{timestamp}.png"
        filepath = os.path.join(debug_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"[DEBUG] Saved debug image to: {filepath}")

    @stateless(restore_seeds=True)
    def select_samples_to_pack(
        self, samples: List[PreEncodedTaskSample]
    ) -> List[List[PreEncodedTaskSample]]:
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        lengths = [sample.total_len_padded for sample in samples]

        packed_samples = self.packing_knapsack_algorithm(lengths, samples, self.packing_seq_length)
        if self.shuffle_packed_samples:
            random.shuffle(packed_samples)

        # TODO: Save iterated samples
        # with open(f"tmpdata/packed_samples_{os.getpid()}.json", "a") as f:
        #     f.write(json.dumps(
        #         {
        #             "lengths": [int(sample.total_len_padded) for sample in samples],
        #             "samples": [
        #                 {
        #                     "key": sample.__key__,
        #                     "images": [img.media.value.fname for img in sample.images],
        #                     "image_sizes": [(img.media.metadata.get("width", img.media.metadata.get("video_width")), img.media.metadata.get("height", img.media.metadata.get("video_height"))) for img in sample.images],
        #                     "length": int(sample.total_len_padded),
        #                     "tokens": [int(token) for token in sample.tokens],
        #                     "detokens": [self.tokenizer.detokenize([token]) for token in sample.tokens],
        #                     "tiling": [img.tiling for img in sample.images],
        #                     "num_tiles": sample.num_tiles,
        #                     "name": sample.__subflavors__["name"],
        #                 } for sample in samples
        #             ],
        #         }
        #     ) + "\n")

        print(
            f"[pid={os.getpid()}] Computed packing from {len(samples)} samples to {len(packed_samples)} packed samples (avg {len(samples) / len(packed_samples)} samples per packed sample)"
        )
        print(
            f"[pid={os.getpid()}] Computed packing from {sum(lengths)} tokens (avg {sum(lengths) / len(samples)} tokens per sample, avg {sum(lengths) / len(packed_samples)} tokens per packed sample)"
        )
        print(
            f"[pid={os.getpid()}] Packing efficiency: {sum(lengths)} / {len(packed_samples) * self.packing_seq_length} = {sum(lengths) / (len(packed_samples) * self.packing_seq_length) * 100:.2f}%"
        )
        assert sum(len(s) for s in packed_samples) == len(
            samples
        ), "knapsack discarded some samples"
        # Print the 5% - 50% - 95% percentiles of the lengths
        print(f"[pid={os.getpid()}] 5% percentile of lengths: {np.percentile(lengths, 5)}")
        print(f"[pid={os.getpid()}] 50% percentile of lengths: {np.percentile(lengths, 50)}")
        print(f"[pid={os.getpid()}] 95% percentile of lengths: {np.percentile(lengths, 95)}")

        # trainable_tokens = [len([int(token) for token in sample.labels if token != IGNORE_INDEX]) for sample in samples]
        # print(f"[pid={os.getpid()}] 5% percentile of trainable tokens: {np.percentile(trainable_tokens, 5)}")
        # print(f"[pid={os.getpid()}] 50% percentile of trainable tokens: {np.percentile(trainable_tokens, 50)}")
        # print(f"[pid={os.getpid()}] 95% percentile of trainable tokens: {np.percentile(trainable_tokens, 95)}")

        return packed_samples

    @stateless
    def pack_selected_samples(self, samples: List[PackedTaskSample]) -> PackedTaskSample:
        """
        Function to pack a list of ImageTaskSamplePacked into a single ImageTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

        Args:
            samples: List of ImageTaskSamplePacked instances to pack into one sample.

        Returns:
            ImageTaskSamplePacked instance.
        """
        cu_lengths = [0]
        cu_lengths_padded = [0]
        # Process each sample and build lists that we will concatenate to create the packed sample.
        for sample in samples:
            sample_len = sample.cu_lengths[-1]
            sample_len_padded = sample.cu_lengths_padded[-1]

            cu_lengths.append(cu_lengths_padded[-1] + sample_len)
            cu_lengths_padded.append(cu_lengths_padded[-1] + sample_len_padded)

        assert (
            cu_lengths_padded[-1] <= self.packing_seq_length
        ), f"Packed sample exceeds the maximum sequence length of {self.packing_seq_length}: {samples}"
        assert all(
            isinstance(sample.imgs, list) for sample in samples
        ), "All images must be tensors"
        assert all(
            isinstance(img, torch.Tensor) for sample in samples for img in sample.imgs
        ), f"All images must be tensors: {[type(img) for sample in samples for img in sample.imgs]}"

        if self.args.allow_cross_sample_attention:
            cu_lengths = [0, cu_lengths[-1]]
            cu_lengths_padded = [0, cu_lengths_padded[-1]]
            max_length = sum(sample.max_length for sample in samples)
        else:
            max_length = max(sample.max_length for sample in samples)

        return PackedTaskSample(
            __key__=[k for s in samples for k in s.__key__],
            __restore_key__=(),  # Will be set by energon based on `samples`
            __subflavors__={idx: sample.__subflavors__ for idx, sample in enumerate(samples)},
            tokens=torch.cat([sample.tokens for sample in samples], dim=0),
            labels=torch.cat([sample.labels for sample in samples], dim=0),
            imgs=[img for sample in samples for img in sample.imgs],
            cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
            cu_lengths_padded=torch.tensor(cu_lengths_padded, dtype=torch.int32),
            max_length=max_length,
            num_tiles=[n for s in samples for n in s.num_tiles],
            num_frames=[n for s in samples for n in s.num_frames],
            samples_seen=sum(s.samples_seen for s in samples),
            sample_lengths=torch.cat([s.sample_lengths for s in samples], dim=0),
        )

    def batch(self, samples: List[PackedTaskSample]) -> BatchedPackedTaskSample:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]

        # Pad image packed seq length to be % 16 if using fp8 and dynamic resolution
        has_fp8 = self.args.fp8 is not None
        has_pad_img = torch.tensor(False)
        # CP pads each local vision partition immediately before the encoder.
        no_cp = self.args.context_parallel_size == 1
        if has_fp8 and self.args.dynamic_resolution and no_cp:
            img_seq_len = 0
            for img in imgs:
                img_seq_len += (img.shape[1] // self.args.patch_dim) * (
                    img.shape[2] // self.args.patch_dim
                )
            padding_needed = get_padding(
                img_seq_len,
                self.args.context_parallel_size,
                self.args.tensor_model_parallel_size,
                self.args.sequence_parallel,
                self.args.tp_comm_overlap,
                self.args.decoder_seq_length,
                fp8_enabled=has_fp8,
            )
            if padding_needed > 0:
                pad_img = torch.zeros(
                    [3, self.args.patch_dim, padding_needed * self.args.patch_dim]
                )
                imgs.append(pad_img)
                has_pad_img = torch.tensor(True)

        imgs, imgs_sizes, vision_cu_lengths, vision_max_lengths = self.image_tiling_strategy.stack(
            imgs
        )

        # For batch mode, wrap in additional dimension for consistency
        if imgs is None:
            imgs = torch.tensor([[0]], dtype=torch.float32)
        if imgs_sizes is None:
            imgs_sizes = torch.tensor([[0, 0]], dtype=torch.int32)
        # Set default values if no vision metadata was returned (static resolution case)
        if vision_cu_lengths is None:
            vision_cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        else:
            # Shape: (1, batch_size + 1)
            vision_cu_lengths = vision_cu_lengths.unsqueeze(0)
        if vision_max_lengths is None:
            vision_max_lengths = torch.tensor([[0]], dtype=torch.int32)
        else:
            # Shape: (1,)
            vision_max_lengths = vision_max_lengths.unsqueeze(0)

        # If the user hasn't defined a target dataloader sequence length, then use the max along the sample lengths.
        max_seq_len = self.dataloader_seq_length
        if not max_seq_len:
            max_seq_len = max(len(s.tokens) for s in samples)

        tokens = torch.full((len(samples), max_seq_len), self.tokenizer.pad, dtype=torch.int64)
        # +1 to accommodate shift to left by one later.
        labels = torch.full((len(samples), max_seq_len + 1), self.tokenizer.pad, dtype=torch.int64)

        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.tokens))
            target_len = min(max_seq_len + 1, len(s.labels))

            tokens[i, :text_len] = s.tokens[:text_len]
            labels[i, :target_len] = s.labels[:target_len]

        num_tiles = torch.tensor([n for s in samples for n in s.num_tiles], dtype=torch.int32)
        if len(num_tiles) == 0:
            num_tiles = torch.tensor([[0]], dtype=torch.int32)

        num_frames = torch.tensor([n for s in samples for n in s.num_frames], dtype=torch.int32)
        if len(num_frames) == 0:
            num_frames = torch.tensor([[0]], dtype=torch.int32)

        cu_lengths = torch.stack([s.cu_lengths for s in samples])
        cu_lengths_padded = torch.stack([s.cu_lengths_padded for s in samples])
        max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)
        sample_lengths = torch.nn.utils.rnn.pad_sequence(
            [s.sample_lengths.to(dtype=torch.int32) for s in samples],
            batch_first=True,
            padding_value=0,
        )

        if self.dataloader_seq_length is not None:
            cu_lengths[0][-1] = self.dataloader_seq_length
            cu_lengths_padded[0][-1] = self.dataloader_seq_length
            new_max_length = cu_lengths_padded[0][-1] - cu_lengths[0][-2]
            max_lengths = torch.max(max_lengths, new_max_length)

        # Pad entire sequence to be a multiple of 16 if using fp8.
        if has_fp8:
            total_seq_len = cu_lengths_padded[0][-1]
            padding_needed = get_padding(
                total_seq_len,
                self.args.context_parallel_size,
                self.args.tensor_model_parallel_size,
                self.args.sequence_parallel,
                self.args.tp_comm_overlap,
                self.args.decoder_seq_length,
                fp8_enabled=has_fp8,
            )
            if padding_needed > 0:
                tokens = torch.cat(
                    [
                        tokens,
                        torch.full(
                            (tokens.shape[0], padding_needed), self.tokenizer.pad, dtype=torch.int64
                        ),
                    ],
                    dim=1,
                )
                labels = torch.cat(
                    [
                        labels,
                        torch.full(
                            (labels.shape[0], padding_needed), IGNORE_INDEX, dtype=torch.int64
                        ),
                    ],
                    dim=1,
                )
                cu_lengths[0][-1] += padding_needed
                cu_lengths_padded[0][-1] += padding_needed
                new_max_length = cu_lengths_padded[0][-1] - cu_lengths[0][-2]
                max_lengths = torch.max(max_lengths, new_max_length)

        return BatchedPackedTaskSample(
            __key__=[s.__key__ for s in samples],
            __restore_key__=[s.__restore_key__ for s in samples],
            __subflavors__=samples[0].__subflavors__,
            tokens=tokens,
            labels=labels,
            imgs=imgs,
            num_tiles=num_tiles,
            num_frames=num_frames,
            cu_lengths=cu_lengths,
            cu_lengths_padded=cu_lengths_padded,
            max_lengths=max_lengths,
            imgs_sizes=imgs_sizes,
            vision_cu_lengths=vision_cu_lengths,
            vision_max_lengths=vision_max_lengths,
            has_pad_img=has_pad_img,
            samples_seen=sum(s.samples_seen for s in samples),
            sample_lengths=sample_lengths,
        )

    def encode_batch(self, batch: BatchedPackedTaskSample) -> dict:
        return dataclasses.asdict(batch)

    def _decode_video_frames_with_energon(
        self,
        av_decoder: AVDecoder,
        targets: list[float],
        *,
        thread_count: int = 0,
    ) -> list[Image.Image]:
        """Use Energon's seek policy with optional worker-local codec threads."""
        if thread_count < 0:
            raise ValueError("video_decode_thread_count must be non-negative")
        if not targets:
            return []

        def get_clips():
            return av_decoder.get_clips(
                video_clip_ranges=[(timestamp, timestamp) for timestamp in targets],
                video_unit="seconds",
            )

        if thread_count == 0:
            frame_clips = get_clips()
        else:
            # Patch only the worker-local container open while preserving
            # Energon's indexing, seeking, and frame-selection behavior.
            decoder_module = importlib.import_module(AVDecoder.__module__)
            with _ENERGON_AV_OPEN_LOCK:
                original_av_open = getattr(decoder_module, "av_open", None)
                if not callable(original_av_open):
                    frame_clips = get_clips()
                else:

                    def threaded_av_open(*args, **kwargs):
                        container = original_av_open(*args, **kwargs)
                        try:
                            for media_stream in container.streams:
                                if media_stream.type != "video":
                                    continue
                                codec_context = media_stream.codec_context
                                if codec_context is None:
                                    continue
                                codec_context.thread_type = "FRAME"
                                codec_context.thread_count = thread_count
                        except Exception:
                            container.close()
                            raise
                        return container

                    decoder_module.av_open = threaded_av_open
                    try:
                        frame_clips = get_clips()
                    finally:
                        decoder_module.av_open = original_av_open

                    # Some codecs retain delayed frames with FRAME threading.
                    # Retry only that video through the exact unthreaded path.
                    if len(frame_clips.video_clips) < len(targets):
                        frame_clips = get_clips()

        images = [tensor_to_pil(image[0]) for image in frame_clips.video_clips]
        if not images:
            raise ValueError("Unable to decode video frames: Energon returned no frames")
        if len(images) < len(targets):
            images.extend([images[-1]] * (len(targets) - len(images)))
        return images

    def _decode_video_frames(
        self, av_decoder: AVDecoder, targets: list[float]
    ) -> list[Image.Image]:
        """Decode target frames with Energon-exact worker-local frame threads."""
        thread_count = int(getattr(self.args, "video_decode_thread_count", 8))
        return self._decode_video_frames_with_energon(
            av_decoder, targets, thread_count=thread_count
        )

    def _load_media(self, sample: PreEncodedTaskSample) -> None:
        """Loads all lazy media in the sample."""
        if len(sample.images) > 1:
            medias: dict[Lazy[AVDecoder] | Lazy[Image.Image], list[ImageTilingParams]] = (
                defaultdict(list)
            )
            # Group by video and frame index
            for media in sample.images:
                # video is a Lazy[AVDecoder] (it's hashable, all pointing to the same file are the same object)
                medias[media.media.value].append(media)

            for media, frames in medias.items():
                media_value = media.get(sample)
                if isinstance(media_value, (tuple, list)):
                    media_value = media_value[0]
                if isinstance(media_value, AVDecoder):
                    media_value.suppress_warnings = True
                    images = self._decode_video_frames(
                        media_value, [frame.media.timestamp for frame in frames]
                    )
                    for frame, image in zip(frames, images):
                        frame.media.value = image
                elif isinstance(media_value, Image.Image):
                    for frame in frames:
                        frame.media.value = media_value
                elif isinstance(media_value, str):
                    # Text fragment
                    pass
                elif isinstance(media_value, bytes):
                    raise ValueError(
                        f"Unable to parse bytes as known media type: {media_value[:100]!r}"
                    )
                else:
                    raise ValueError(f"Unexpected media type: {type(media_value)}. Media: {media}")
        else:
            for media in sample.images:
                value = media.media.value.get(sample)
                if isinstance(value, (tuple, list)):
                    value = value[0]
                media.media.value = value
    @staticmethod
    def _count_trainable_target_spans(target: torch.Tensor) -> int:
        """Count contiguous non-ignored target regions."""
        if target.numel() == 0:
            return 0
        trainable = target != IGNORE_INDEX
        span_starts = trainable.clone()
        span_starts[1:] = trainable[1:] & ~trainable[:-1]
        return int(span_starts.sum().item())

    @classmethod
    def _validate_explicit_loss_target_spans(
        cls, target: torch.Tensor, *, expected_span_count: int, sample_key: str, phase: str
    ) -> None:
        """Require every explicitly selected assistant target span to survive."""
        actual_span_count = cls._count_trainable_target_spans(target)
        if actual_span_count != expected_span_count:
            raise ValueError(
                "explicit assistant loss target span mismatch after "
                f"{phase} for sample {sample_key}: expected "
                f"{expected_span_count}, found {actual_span_count}"
            )

    def _group_video_frame_params_into_tubelets(
        self, image_media: list, image_media_params: list, temporal_patch_size: int
    ) -> list:
        """
        Group video frame params into tubelet params to match the number of IMAGE_TOKENs.

        With temporal compression, we emit one IMAGE_TOKEN per tubelet (every T frames),
        not per frame. This function groups the params accordingly, summing num_embeddings
        for each tubelet.

        Note: Because we ensure that the number of video frames is a multiple of T, it each group
        is guaranteed to be from a unique video, even if there are multiple videos in the sample.

        Args:
            image_media: List of ImageMedia and VideoFrameMedia objects.
            image_media_params: List of params, one per media item.
            temporal_patch_size: Number of frames per tubelet (T).

        Returns:
            List of params with video frames grouped into tubelets.
        """
        # Phase 1: Build groups - map each index i in image_media to a group ID
        # Images get their own group, video frames are grouped by tubelet (every T frames)
        groups: list[list[int]] = []  # List of index lists, one per group

        for i, media in enumerate(image_media):
            if isinstance(media, VideoFrameMedia):
                # Video frame: group by tubelet (every T frames)
                if media.sample_index % temporal_patch_size == 0:
                    # Start of a new tubelet
                    groups.append([i])
                else:
                    # Continue current tubelet (add to last group)
                    groups[-1].append(i)
            else:
                # Image: each gets its own group
                groups.append([i])

        # Phase 2: Create grouped params
        # For video tubelets: use first frame's embeddings (E), NOT sum (T × E)
        # Model groups T frames into 1 tubelet with E patches (features are T× larger, count is same)
        grouped_params = []
        for group_indices in groups:
            # Use first item's params as base
            base_params = image_media_params[group_indices[0]]

            # Verify all items in the group have the same embeddings and tiles
            # (all video frames in a tubelet should have same spatial size)
            assert all(
                [
                    image_media_params[idx].num_embeddings == base_params.num_embeddings
                    for idx in group_indices
                ]
            ), (
                f"All items in the group must have the same num_embeddings: {group_indices}, "
                f"got {[image_media_params[idx].num_embeddings for idx in group_indices]}"
            )
            assert all(
                [
                    image_media_params[idx].num_tiles == base_params.num_tiles
                    for idx in group_indices
                ]
            ), f"All items in the group must have the same num_tiles: {group_indices}"

            # Use first frame's embeddings (model produces E patches per tubelet, not T × E)
            grouped_params.append(base_params)

        return grouped_params

    def _target_has_trainable_tokens(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        image_tiling_params: list[ImageTilingParams],
    ) -> bool:
        """
        Check if the target has trainable tokens.

        Args:
            input_ids: Input tokens.
            target: Target tokens.
            image_tiling_params: Image tiling parameters.
        Returns:
            True if the target has trainable tokens, False otherwise.
        """
        # Compute the loss mask based on extending the image tags with the proper
        # number of image tokens, extracting the first self.args.decoder_seq_length tokens, and
        # ensuring that some of these tokens have a loss mask > 0.
        # Note that this is a bit hacky because we reproduce here parts of the logics which are in
        # the model itself. Ideally, the data sampler would return the already processed inputs
        # and targets to avoid this duplication.
        expanded_target = target.clone()
        expanded_target[input_ids == self.img_token_id] = self.img_token_id
        expanded_target = self._replace_value_with_repetition(
            expanded_target,
            self.img_token_id,
            torch.tensor([media.num_embeddings for media in image_tiling_params]),
            IGNORE_INDEX,
        )
        loss_mask = torch.ones(expanded_target.size(), dtype=torch.float)
        loss_mask[expanded_target == self.tokenizer.pad] = 0.0  # mask paddings
        loss_mask[expanded_target == IGNORE_INDEX] = 0.0  # mask prompts
        loss_mask = torch.cat((loss_mask[1:], torch.zeros((1,))))
        loss_mask = loss_mask[: self.args.decoder_seq_length]
        return torch.sum(loss_mask) > 0

    def _replace_value_with_repetition(
        self, arr: torch.Tensor, token_to_replace: int, num_repetition: torch.Tensor, new_token: int
    ) -> torch.Tensor:
        """
        Replace every occurrence of value V in the input array with R repetitions of W.

        Args:
            arr: Input array to be modified
            token_to_replace: Token to be replaced
            num_repetition: Number of repetition of new token. Size must match the number of `token_to_replace` in the
                input array.
            new_token: New token to replace the `token_to_replace` with.

        Returns:
            Array: New array with token_to_replace replaced by num_repetition repetitions of new_token
        """
        assert torch.sum(arr == token_to_replace) == len(
            num_repetition
        ), "The number of tokens to replace must match the length of the tile tensor."

        # Convert to list for easier manipulation
        arr_list = arr.tolist()
        result = []
        idx = 0
        for item in arr_list:
            if item == token_to_replace:
                # If the current item matches token_to_replace, add R copies of new_token
                result.extend([new_token] * num_repetition[idx].item())
                idx += 1
            else:
                # Otherwise, keep the original item
                result.append(item)

        return torch.tensor(result, dtype=arr.dtype, device=arr.device)

    def _pad_for_context_parallel_and_fp8(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        image_tiling_params: list[ImageTilingParams],
    ) -> tuple[int, int, torch.Tensor, torch.Tensor]:
        total_len = self._get_total_seq_length(input_ids, image_tiling_params)
        total_len_padded = total_len
        # With context parallel or sequence parallel, we need to pad individual sequences.
        # However, with FP8, we need to only pad the entire sequence to be a multiple of 16.
        if getattr(self.args, "context_parallel_size", 1) > 1 or self.args.sequence_parallel:
            padding_needed = get_padding(
                total_len,
                self.args.context_parallel_size,
                self.args.tensor_model_parallel_size,
                self.args.sequence_parallel,
                self.args.tp_comm_overlap,
                self.args.decoder_seq_length,
                fp8_enabled=False,
            )
            # Build padding directly on the token/label tensors' dtype and device;
            # torch.ones(...)*value would create default float CPU tensors before cat().
            padding1 = torch.full(
                (padding_needed,),
                self.tokenizer.pad,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            padding2 = torch.full(
                (padding_needed,), IGNORE_INDEX, dtype=target.dtype, device=target.device
            )
            input_ids = torch.cat([input_ids, padding1])
            target = torch.cat([target, padding2])

            # A100 doesn't support cu_seqlens != cu_seqlens_padded. hack=True forces them to be same.
            # hack=False only supported on H100.
            hack = True
            if hack:
                # Pad everything
                total_len = total_len + padding_needed
                total_len_padded = total_len
            else:
                # Pad only padding.
                total_len_padded = total_len + padding_needed
        return total_len, total_len_padded, input_ids, target

    def _get_total_seq_length(
        self,
        input_ids: torch.Tensor,
        image_tiling_params: list[ImageTilingParams],
    ):
        """Calculate expected sequence length given text tokens length and number of tiles."""
        total_num_images = len(image_tiling_params)
        total_num_image_embeddings = sum(media.num_embeddings for media in image_tiling_params)
        total_len = len(input_ids) + total_num_image_embeddings - total_num_images
        return total_len

    def _truncate_to_decoder_seq_len(
        self,
        input_ids: torch.Tensor,
        target: torch.Tensor,
        image_tiling_params: list[ImageTilingParams],
        sample_key: str = "",
        sample_subflavors: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate tokens and labels if they exceed sequence length."""
        total_img_embeddings_len = sum(media.num_embeddings for media in image_tiling_params)
        total_num_images = len(image_tiling_params)
        packing_seq_length = (
            self.args.packing_seq_length
            if self.args.packing_seq_length > 0
            else self.args.decoder_seq_length
        )
        max_text_tokens = packing_seq_length - total_img_embeddings_len + total_num_images

        truncated_input_ids = input_ids[:max_text_tokens]
        truncated_target = target[:max_text_tokens]

        assert len(truncated_input_ids) == len(
            truncated_target
        ), "Input and target must have the same length"
        inputs_are_truncated = len(truncated_input_ids) < len(input_ids)
        if self.args.dynamic_resolution_no_truncate and inputs_are_truncated:
            raise ValueError(
                "Truncation required but --dynamic-resolution-no-truncate is set, skipping sample: \n"
                f"sample_key: {sample_key} \n"
                f"sample_subflavors: {sample_subflavors} \n"
                f"original input length: {len(input_ids)} \n"
                f"truncated input length: {len(truncated_input_ids)} \n"
                f"max_text_tokens: {max_text_tokens} \n"
                f"total_img_embeddings_len: {total_img_embeddings_len} \n"
                f"total_num_images: {total_num_images} \n"
            )

        # If truncate causes all labels to be ignored, then skip the sample
        if len(truncated_target) == 0 or (truncated_target == IGNORE_INDEX).all():
            raise NoTrainableTokensError(
                "All targets will be ignored after truncation: \n"
                f"original input: {input_ids} \n"
                f"original target: {target} \n"
                f"truncated input:  {truncated_input_ids} \n"
                f"truncated target: {truncated_target} \n"
                f"max_text_tokens: {max_text_tokens} \n"
                f"total_img_embeddings_len: {total_img_embeddings_len} \n"
                f"total_num_images: {total_num_images} \n"
            )

        return truncated_input_ids, truncated_target


tensor_to_pil = ToPILImage()
