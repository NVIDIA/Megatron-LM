# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import bisect
import dataclasses
import functools
import json
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from image_processing import ImageTransform, find_closest_aspect_ratio, find_closest_area_weighted_aspect_ratio, process_images
from PIL import Image
from torchvision.transforms import ToPILImage
import numpy as np
from einops import rearrange
import torch

from energon_util import OfflineTargetAspectRatioSample, SampleListSample
from megatron.core.models.multimodal.context_parallel import get_padding
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, IMAGE_TOKEN, VIDEO_TOKEN
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    OCRSample,
    Sample,
    SimilarityInterleavedSample,
    VQASample,
    MultiChoiceVQASample
)
from megatron.energon.task_encoder.base import stateless
from megatron.training import get_args, get_tokenizer


@dataclass
class ImageTaskSample(Sample):
    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Dict
    __subflavors__: Dict
    # (c, h, w)
    imgs: List[torch.Tensor]
    num_tiles: List[int]
    tokens: torch.Tensor
    total_len: int  # Total token count in the sample, including text and image tokens
    total_len_padded: int  # Total padded token count in the sample.
    labels: torch.Tensor = None


@dataclass
class ImageTaskSamplePacked(Sample):
    """Dataclass to store a single packed sample (not a batch).

        P = Number of sub-samples in the packed sample
        seq_len = Total sequence length
        num_imgs = Number of images across all samples in the packed sample
    """

    __key__: str    # Sample name
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Dict     # Sample metadata. Deprecated.
    __subflavors__: Dict    # Sample metadata.
    tokens: torch.Tensor  # Input tokens packed into a single tensor (seq_len,)
    labels: torch.Tensor # Target tokens packed into a single tensor (seq_len,)
    imgs: List[torch.Tensor]    # Input images
    num_tiles: List[int]  # Number of tiles for each image of each sample (num_imgs)
    max_length: int    # Maximum length across sub-samples.
    cu_lengths: List[int]  # Cumulative length of each sub-sample in this packed sample incl. text and image tokens (P,)
    cu_lengths_padded: List[int]  # Cumulative padded length of each sub-sample in this packed sample incl. text and image tokens (P,)


# Typing for the resulting batch data after encode_batch()
@dataclass
class ImageTaskBatchPacked(Batch):
    """Dataclass to store a batch of packed samples.

        N = Batch size
        P = Number of samples in the packed sample
        seq_len = Maximum sequence length
        num_imgs = Number of images across all samples in the packed sample
    """

    __key__: List[str]  # Sample names
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Dict     # Sample metadata. Deprecated.
    __subflavors__: List[Dict]  # Sample metadatas.
    tokens: torch.Tensor  # Input tokens packed and padded (N, seq_len)
    labels: torch.Tensor # Target tokens packed and padded (N, seq_len)
    imgs: torch.Tensor  # All image tiles stacked into a single tensor (num_tiles, C, H, W)
    num_tiles: List[List[int]]  # Number of tiles per image (N, num_imgs)
    max_lengths: List[int]  # Maximum length across sub-samples (N,)
    cu_lengths: List[List[int]]  # Cumulative length of each sub-sample in each packed sample of the batch (N, P)
    cu_lengths_padded: List[List[int]]  # Cumulative padded length of each sub-sample in each packed sample of the batch (N, P)
    imgs_sizes: List[Tuple[int, int]]
    vision_max_lengths: List[int]
    vision_cu_lengths: List[List[int]]
    has_pad_img: bool

    # Sound
    sound_clips: list[torch.Tensor]
    sound_length: list[int]
    sound_timestamps: list[tuple[int, int]]
    num_sound_clips: list[int]


# Based on https://github.com/hiyouga/LLaMA-Factory/blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L19
# Copyright (c) 2024 LLaMA-Factory. Apache license 2.0.
def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


# Based on https://github.com/hiyouga/LLaMA-Factory/blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L27
# Copyright (c) 2024 LLaMA-Factory. Apache license 2.0.
def greedy_knapsack(item_sizes: List[int], samples: List, max_capacity: int) -> List:
    """Greedy algorithm with binary search for the knapsack problem.

    Pack as many samples as possible given a maximum capacity and capacities of individual samples.
    Used if sequence packing is enabled.
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must have the same length."

    knapsacks = []

    if len(item_sizes) == 0:
        return knapsacks

    # Sort sample lengths and samples together.
    sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_item_sizes = list(sorted_item_sizes)
    sorted_samples = list(sorted_samples)

    # Check if all samples fit in the knapsack capacity.
    if sorted_item_sizes[-1] > max_capacity:
        raise ValueError(f"knapsack: A sample is larger {sorted_item_sizes[-1]} than the max_sequence_length {max_capacity}.")

    while sorted_item_sizes:
        current_knapsack = []
        remaining_capacity = max_capacity

        while True:
            idx = search_for_fit(sorted_item_sizes, remaining_capacity)
            if idx == -1:
                break   # Can't fit more samples.

            remaining_capacity -= sorted_item_sizes[idx]

            sorted_item_sizes.pop(idx)
            sample = sorted_samples.pop(idx)
            current_knapsack.append(sample)

        knapsacks.append(current_knapsack)

    return knapsacks


class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatchPacked, dict]):
    """A simple task encoder for VLMs."""

    def __init__(
        self
    ):
        super().__init__()

        self.args = get_args()

        self.tokenizer = get_tokenizer()
        self.manual_prompts = {}
        if self.args.prompt_path:
            with open(self.args.prompt_path, "r") as f:
                self.manual_prompts = json.load(f)
        self.dataloader_seq_length = self.args.dataloader_seq_length  # Always return samples of this length.
        self.packing_seq_length = self.args.packing_seq_length     # Packing sequence length, if packing is enabled.
        self.is_packing_enabled = self.args.packing_buffer_size is not None and self.args.packing_buffer_size > 0

        if self.dataloader_seq_length and self.packing_seq_length:
            assert self.dataloader_seq_length >= self.packing_seq_length, "dataloader sequence length must be greater than or equal to the packing sequence length"

        if self.is_packing_enabled:
            assert self.packing_seq_length > 0, "packing sequence length must be set"

        self.num_image_embeddings_per_tile = get_num_image_embeddings(
            img_h=self.args.img_h,
            img_w=self.args.img_w,
            patch_dim=self.args.patch_dim,
            vision_model_type=self.args.vision_model_type,
            disable_vision_class_token=self.args.disable_vision_class_token,
            class_token_len=1,
            pixel_shuffle=self.args.pixel_shuffle,
            use_tile_tags=self.args.use_tile_tags,
            max_num_tiles=self.args.max_num_tiles,
            tokenizer_type=self.args.tokenizer_prompt_format,
            use_image_break_token=self.args.image_break_token is not None,
            conv_merging=self.args.conv_merging,
        )

        # Create a partial function with all the self.args parameters pre-filled
        # Only img_h and img_w need to be specified when calling this function
        self._get_num_image_embeddings = functools.partial(
            get_num_image_embeddings,
            patch_dim=self.args.patch_dim,
            vision_model_type=self.args.vision_model_type,
            disable_vision_class_token=self.args.disable_vision_class_token,
            class_token_len=1,
            pixel_shuffle=self.args.pixel_shuffle,
            use_tile_tags=self.args.use_tile_tags,
            max_num_tiles=self.args.max_num_tiles,
            tokenizer_type=self.args.tokenizer_prompt_format,
            use_image_break_token=self.args.image_break_token is not None,
            conv_merging=self.args.conv_merging,
        )

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        # This map is used to reduce the number of tiles used per image if the number of tokens is
        # larger than the decoder_seq_length.
        self.num_tiles_degradation_map = {12:8, 8:6, 6:4, 4:2, 2:1, 1:1}

        self.find_closest_aspect_ratio_fn = (
            find_closest_area_weighted_aspect_ratio if self.args.use_area_weighted_aspect_ratio
            else find_closest_aspect_ratio)

        self.transform_img = ImageTransform(
            self.img_h,
            self.args.vision_model_type,
            dynamic_resolution=self.args.dynamic_resolution,
            res_step=self.args.patch_dim,
            min_num_patches=self.args.dynamic_resolution_min_patches,
            max_num_patches=self.args.seq_length - (1 if not self.args.disable_vision_class_token else 0), #TODO: handle class toekn length correctly(not just use 1)
            pixel_shuffle=self.args.pixel_shuffle,
            min_side=self.args.dynamic_resolution_min_side,
            conv_merging=self.args.conv_merging,
            match_tiling_dynamic_resolution=self.args.match_tiling_dynamic_resolution,
            masked_tiling_dynamic_resolution=getattr(self.args, "masked_tiling_dynamic_resolution", False),
            thumbnail_area_threshold=self.args.thumbnail_area_threshold,
        )

    def _verify_no_temporal_compression(self):
        """
        This TaskEncoder is not updated to support these features, but we still construct it during
        training setup even when using MultiModalTaskEncoder, enen though we don't use it. So we
        have to check this during runtime.
        """
        # For tiling only, need a fixed number of embeddings for num_image_embeddings_per_tile
        # We don't pass in `is_video` to calculate `self.num_image_embeddings_per_tile`,
        # so we currently require no temporal compression because we can't verify the number of frames
        video_temporal_patch_size = getattr(self.args, 'video_temporal_patch_size', 1)
        if video_temporal_patch_size != 1:
            raise NotImplementedError(
                f"When using TaskEncoder, temporal compression is not supported."
                f" Found video_temporal_patch_size={video_temporal_patch_size}."
            )

    def _get_total_seq_length(self, input_ids, num_tiles, imgs=None):
        """Calculate expected sequence length given text tokens length and number of tiles."""
        self._verify_no_temporal_compression()

        if self.args.dynamic_resolution:
            assert imgs is not None

            img_seq_len = 0
            img_idx = 0
            # For dynamic resolution, we need to group embeddings by conceptual image
            # since match tiling can return multiple tensors (main + thumbnail) per image
            for num_tiles_for_image in num_tiles:
                # Sum embeddings for all tiles/images belonging to this conceptual image
                for _ in range(num_tiles_for_image):
                    img_seq_len += self._get_num_image_embeddings(
                        img_h=imgs[img_idx].shape[1],
                        img_w=imgs[img_idx].shape[2],
                    )
                    img_idx += 1

            total_len = len(input_ids) + img_seq_len - len(num_tiles)
        else:
            total_num_images = len(num_tiles)
            total_num_tiles = sum(num_tiles)
            total_len = len(input_ids) + total_num_tiles * self.num_image_embeddings_per_tile - total_num_images
        return total_len

    def _truncate_for_packing(self, input_ids, target, num_tiles, imgs):
        """Truncate tokens and labels if they exceed packing sequence length."""
        self._verify_no_temporal_compression()

        total_num_images = len(num_tiles)
        total_num_tiles = sum(num_tiles)
        if self.args.dynamic_resolution:
            assert imgs is not None

            total_img_embeddings_len = 0
            img_idx = 0
            # For dynamic resolution, we need to group embeddings by conceptual image
            # since match tiling can return multiple tensors (main + thumbnail) per image
            for num_tiles_for_image in num_tiles:
                # Sum embeddings for all tiles/images belonging to this conceptual image
                for _ in range(num_tiles_for_image):
                    total_img_embeddings_len += self._get_num_image_embeddings(
                        img_h=imgs[img_idx].shape[1],
                        img_w=imgs[img_idx].shape[2],
                    )
                    img_idx += 1
        else:
            total_img_embeddings_len = total_num_tiles * self.num_image_embeddings_per_tile
        max_text_tokens = self.packing_seq_length - total_img_embeddings_len + total_num_images

        input_ids = input_ids[:max_text_tokens]
        target = target[:max_text_tokens]

        # If truncate causes all labels to be ignored, then skip the sample
        if (target == IGNORE_INDEX).all():
            raise ValueError(f"all targets will be ignored after truncation: {input_ids}")

        return input_ids, target

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: Union[SimilarityInterleavedSample]):
        if isinstance(sample, SimilarityInterleavedSample):
            yield self.encode_llava_sft(sample)
        # Because the SampleListSample is defined in the Megatron module but loaded by the Energon
        # library, we need to resort to the more brittle check:
        elif type(sample).__name__ == "SampleListSample":
            yield self.encode_sample_list(sample)
        else:
            raise NotImplementedError("Sample format not supported", sample)


    def encode_captioning(self, sample: CaptioningSample):
        """Encode CaptioningSample."""
        augment = sample.__subflavors__.get("augmentation")

        imgs = self.transform_img(
            sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
            find_closest_aspect_ratio_fn=self.find_closest_aspect_ratio_fn
        )
        num_tiles = [len(imgs)]

        prompt_list = self.manual_prompts["CaptioningPretraining"]["raw"]

        prompt_idx = np.random.randint(len(prompt_list))
        cur_prompt = prompt_list[prompt_idx]
        cur_prompt = IMAGE_TOKEN + "\n" + cur_prompt + "\n"

        caption = sample.caption.strip()

        split_by_line_flag = sample.__subflavors__.get("SplitByLine")
        if split_by_line_flag:
            caption_list = caption.split('\n')
            caption = np.random.choice(caption_list)

        conv = [
            # Note: no system message.
            {"role": "user", "content": cur_prompt},
            {"role": "assistant", "content": caption},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conv, True, False)

        if self.is_packing_enabled:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles, imgs)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles, imgs),
        )

    def encode_llava_pretrain(self, sample: VQASample):
        """Encode pretrain sample in LLAVA style."""
        augment = sample.__subflavors__.get("augmentation", False)

        imgs = self.transform_img(
            sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
            find_closest_aspect_ratio_fn=self.find_closest_aspect_ratio_fn
        )
        num_tiles = [len(imgs)]

        # LLAVA training: override text-prompt with just the image.
        conv = [
            # Note: no system message.
            {"role": "user", "content": IMAGE_TOKEN + "\n"},
            {"role": "assistant", "content": sample.answers},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conv, True, False)

        if self.is_packing_enabled:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles, imgs)

        assert self._get_total_seq_length(input_ids, num_tiles, imgs) < self.args.decoder_seq_length, f"total sequence length {self._get_total_seq_length(input_ids, num_tiles, imgs)} needs to be less than {self.args.decoder_seq_length}"


        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles, imgs),
        )

    def encode_sample_list(self, samples: SampleListSample):
        """We encode the list of samples using encode_llava_sft on each sample."""
        error_msg = ("You probably don't want to use online packing since SampleListSample is "
                     "usually used along offline packing.")
        assert not self.is_packing_enabled, error_msg
        encoded_samples = []
        current_length = 0
        for idx, sample in enumerate(samples.samples):
            try:
                encoded_sample = self.encode_llava_sft(sample, truncate_for_sample_list_packing=True)
                if current_length + encoded_sample.total_len > self.packing_seq_length:
                    print(f"Encoding list of samples: stopped at {idx} samples to stick to {self.packing_seq_length}. Last sample key: {sample.__key__}")
                    break
                else:
                    encoded_samples.append(encoded_sample)
                    current_length += encoded_sample.total_len
            except Exception as e:
                print(e)
        return self.pack_selected_samples(encoded_samples)

    def encode_llava_sft(self, sample: Union[SimilarityInterleavedSample, OfflineTargetAspectRatioSample], truncate_for_sample_list_packing=False):
        """Encode SFT sample."""
        self._verify_no_temporal_compression()

        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        has_video = sample.__subflavors__['has_video'] if 'has_video' in sample.__subflavors__ else False

        # If the target aspect ratio are provided by the dataset, we use them instead of computing
        # them with the self.find_closest_aspect_ratio_fn function.
        local_find_closest_aspect_ratio_fn = self.find_closest_aspect_ratio_fn
        if type(sample).__name__ == "OfflineTargetAspectRatioSample" and len(sample.target_aspect_ratio) > 0:
            target_aspect_ratio = tuple(sample.target_aspect_ratio[0])
            assert target_aspect_ratio is not None, "Sample of type OfflineTargetAspectRatioSample needs to define the target aspect ratio."
            local_find_closest_aspect_ratio_fn = lambda *args, **kwargs: target_aspect_ratio

        has_image = False
        # We infer whether the sample has image or not.
        if hasattr(sample, "images") and not has_video:
            # If this is a text-only sample and we are freezing the LM,
            # then use a dummy input image.
            if len(sample.images) == 0 and self.args.freeze_LM:
                empty_img = Image.new('RGB', (self.args.img_w, self.args.img_h), (255, 255, 255))
                sample.images.append(empty_img)
            if len(sample.images) > 0:
                has_image = True

        # Note: Some tokenizers may ignore the system prompt.
        if self.args.tokenizer_prompt_format == "nemotron-h-5p5-reasoning":
            conversation = [{"role": "system", "content": "You are a helpful assistant."}]
        else:
            conversation = [{"role": "system", "content": "Answer the questions."}]
        # Format the conversation as a list of "user" / "assistant" turns.
        for text in sample.texts:
            error_msg = f"unexpected role {text['from']} in {sample.texts}"
            assert text["from"] in ["human", "gpt"], error_msg
            if self.args.tokenizer_prompt_format == "nemotron-h-5p5-reasoning":
                if text["from"] == "gpt":
                    # Append empty think tokens if missing
                    if not("<think>" in text["value"] and "</think>" in text["value"]):
                        text["value"] = "<think></think>\n" + text["value"].strip()
                elif text["from"] == "human":
                    # Remove legacy reasoning instructions from training prompt
                    text["value"] = text["value"].replace("detailed thinking on\n", "", 1).strip()
                    assert "detailed thinking on" not in text["value"], f"Found sample with detailed thinking on: {sample.texts} {sample.__key__}"
            conversation.append({
                "role": "user" if text["from"] == "human" else "assistant",
                "content": text["value"]})

        # Replace the image tags <image-idx> with IMAGE_TOKEN and count the number of image tags
        number_image_tags = 0
        image_tag_ids_list = []
        for turn in conversation:
            if turn["role"] == "user":
                image_tag_ids = [int(x) - 1 for x in re.findall(r"<image-(\d+)>", turn["content"])]
                image_tag_ids_list.extend(image_tag_ids)
                turn["content"] = re.sub(r"<image-\d+>", IMAGE_TOKEN, turn["content"])
                # For videos, we use the image token to locate where to put the frames.
                if has_video:
                    turn["content"] = turn["content"].replace(VIDEO_TOKEN, IMAGE_TOKEN)
                number_image_tags += turn["content"].count(IMAGE_TOKEN)

        # We re-order the images in sample.images according to how they appear in the conversation.
        if len(image_tag_ids_list) > 0:
            sample.images = [sample.images[idx] for idx in image_tag_ids_list]

        # If there is only one image, but several image tags, we assume all the tags refer to the
        # same image and duplicate the image:
        if not has_video and len(sample.images) == 1 and number_image_tags > 1:
            sample.images = sample.images * number_image_tags

        # If there are no images in the sample, remove the image tags in the conversation.
        if len(sample.images) == 0:
            for turn in conversation:
                if turn["role"] == "user":
                    turn["content"] = turn["content"].replace(IMAGE_TOKEN, "")
            number_image_tags = 0

        # We currently only support one video per sample.
        number_of_images = 1 if has_video else len(sample.images)
        # Fail if there are more image or video tags than image or videos:
        error_msg = (
            f"Found {number_image_tags} image tags for {number_of_images} images. {sample.texts}")
        assert number_image_tags <= number_of_images, error_msg

        # If there are less image of video tags than image or videos, prepend the tags to the first
        # user message:
        if number_image_tags < number_of_images:
            for turn in conversation:
                if turn["role"] == "user":
                    turn["content"] = IMAGE_TOKEN*(number_of_images-number_image_tags) + "\n" + turn["content"]
                    break

        input_ids, target = self.tokenizer.tokenize_conversation(conversation, True, False)

        if has_image:
            imgs = []
            num_tiles = []
            max_num_tiles = self.args.max_num_tiles
            # We keep a buffer of 4 tokens for the question,
            # the rest can be used for image tokens.
            max_image_token_allowed = self.args.decoder_seq_length - len(input_ids) - 4
            # We start by extracting as many tiles per image as possible, and decrease the max
            # number of tiles if there are too many image tokens.
            while True:
                imgs = []
                num_tiles = []
                for img in sample.images:
                    # This if block is a temporary fix to handle video frames. We hard code
                    # `use_tiling = False` because we don't use tiling for videos frames to keep
                    # the number of tokens to a reasonable value.
                    if isinstance(img, torch.Tensor) or isinstance(img, np.ndarray):
                        if len(img.shape) == 4:
                            assert img.shape[0] == 1, f"When len(img.shape) == 4, we expect the first dimension to be 1, but got img.shape: {img.shape} instead."
                            img = img[0]
                            use_tiling = False
                        to_pil = ToPILImage()
                        img = to_pil(img)
                    img_tiles = self.transform_img(
                        img, self.img_h, self.img_w, self.args.use_tiling, max_num_tiles,
                        self.args.use_thumbnail, augment, find_closest_aspect_ratio_fn=local_find_closest_aspect_ratio_fn)
                    imgs += img_tiles
                    num_tiles += [len(img_tiles)]
                if max_num_tiles == 1 or self.args.dynamic_resolution:
                    break
                if sum(num_tiles) * self.num_image_embeddings_per_tile > max_image_token_allowed:
                    if max_num_tiles in self.num_tiles_degradation_map:
                        max_num_tiles = self.num_tiles_degradation_map[max_num_tiles]
                    else:
                        raise RuntimeError((
                            f"Tried to decrease the number of tiles {max_num_tiles} but it's not ",
                            f"defined in the degradation map {self.num_tiles_degradation_map}"))
                else:
                    break
        elif has_video:
            # We don't use tiling for videos to limit the number of tokens.
            use_tiling=False
            # Grab the selected frames of the video as a tensor with shape
            # fhwc: (num_frames, num_channels, height, width).
            video_fchw = sample.images.frames
            if video_fchw.shape[0] == 0:
                raise ValueError(f"Video {sample.__key__} {sample.__restore_key__} {sample.texts} has no frames.")
            selected_frames = torch.linspace(
                0, video_fchw.shape[0] - 1,
                min(self.args.num_frames, video_fchw.shape[0])).long()
            video_fchw = video_fchw[selected_frames]
            imgs = []
            for video_chw in video_fchw:
                to_pil = ToPILImage()
                video_chw = to_pil(video_chw)
                imgs += self.transform_img(
                    video_chw, self.img_h, self.img_w, use_tiling, self.args.max_num_tiles,
                    self.args.use_thumbnail, augment, find_closest_aspect_ratio_fn=local_find_closest_aspect_ratio_fn)
            num_tiles = [len(imgs)]
        else:
            imgs = num_tiles = []

        if self.is_packing_enabled or truncate_for_sample_list_packing:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles, imgs)

        # Some final checks with respect to the number of image tokens and images on the tokenized
        # conversation. There can still be errors, for instance if a non-video sample happens to
        # have our pre-defined video token, or if the packing truncation removed a necessary image
        # tag.
        number_image_token = np.sum(input_ids == self.img_token_id)
        error_msg = (
            f"Found {number_image_token} image tokens for len({num_tiles}) = {len(num_tiles)} image tiles in {conversation}.")
        assert number_image_token == len(num_tiles), error_msg
        error_msg = (
            f"Found sum({num_tiles}) = {np.sum(num_tiles)} tiles for {len(imgs)} images in {conversation}.")
        assert np.sum(num_tiles) == len(imgs), error_msg

        # We need to ensure that there are at least some trainable tokens in the sample.
        assert self.target_has_trainable_tokens(input_ids, num_tiles, target, imgs), "Sample has no trainable tokens."

        assert self._get_total_seq_length(input_ids, num_tiles, imgs) < self.args.decoder_seq_length, f"total sequence length {self._get_total_seq_length(input_ids, num_tiles, imgs)} needs to be less than {self.args.decoder_seq_length}"

        # Context parallel and FP8 require padding.
        # TODO: Total sample len and padded len are kept the same here.
        # H100 and newer cuDNN versions can handle different values for them.
        total_len = self._get_total_seq_length(input_ids, num_tiles, imgs)

        # Individual samples need to be padded if using context parallel or sequence parallel.
        # Here we don't pad for FP8 because only the final sequence needs to be padded. That is done in batch().
        has_cp = self.args.context_parallel_size > 1
        if has_cp or self.args.sequence_parallel:
            padding_needed = get_padding(total_len, self.args.context_parallel_size, self.args.tensor_model_parallel_size, self.args.sequence_parallel,
                                         self.args.tp_comm_overlap, self.args.decoder_seq_length, fp8_enabled=False)
            padding_input = np.ones(padding_needed) * self.tokenizer.pad
            padding_labels = np.ones(padding_needed) * IGNORE_INDEX
            input_ids = np.concatenate([input_ids, padding_input])
            target = np.concatenate([target, padding_labels])
            total_len = total_len + padding_needed

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=total_len,
            total_len_padded=total_len,
        )

    def target_has_trainable_tokens(self, input_ids, num_tiles, target, imgs):
        self._verify_no_temporal_compression()

        # Compute the loss mask based on extending the image tags with the proper
        # number of image tokens, extracting the first self.args.decoder_seq_length tokens, and
        # ensuring that some of these tokens have a loss mask > 0.
        # Note that this is a bit hacky because we reproduce here parts of the logics which are in
        # the model itself. Ideally, the data sampler would return the already processed inputs
        # and targets to avoid this duplication.
        expanded_target = target.copy()
        expanded_target[input_ids==self.img_token_id] = self.img_token_id
        if self.args.dynamic_resolution:
            img_embeddings_len = []
            img_idx = 0
            # For dynamic resolution, we need to group embeddings by conceptual image
            # since match tiling can return multiple tensors (main + thumbnail) per image
            for num_tiles_for_image in num_tiles:
                total_embeddings_for_image = 0
                # Sum embeddings for all tiles/images belonging to this conceptual image
                for _ in range(num_tiles_for_image):
                    total_embeddings_for_image += self._get_num_image_embeddings(
                        img_h=imgs[img_idx].shape[1],
                        img_w=imgs[img_idx].shape[2],
                    )
                    img_idx += 1
                img_embeddings_len.append(total_embeddings_for_image)
        else:
            img_embeddings_len = np.array(num_tiles) * self.num_image_embeddings_per_tile

        expanded_target = self.replace_value_with_repetition(
            expanded_target, self.img_token_id, img_embeddings_len, IGNORE_INDEX)
        loss_mask = torch.ones(torch.tensor(expanded_target).size(), dtype=torch.float)
        loss_mask[expanded_target == self.tokenizer.pad] = 0.0 # mask paddings
        loss_mask[expanded_target == IGNORE_INDEX] = 0.0 # mask prompts
        loss_mask = torch.cat((loss_mask[1:], torch.zeros((1,))))
        loss_mask = loss_mask[:self.args.decoder_seq_length]
        return torch.sum(loss_mask) > 0

    def replace_value_with_repetition(self, arr, token_to_replace, num_repetition, new_token):
        """
        Replace every occurrence of value V in the input array with R repetitions of W.

        Args:
            arr (Array): Input array to be modified
            token_to_replace: token to be replaced
            new_token: new token
            num_repetition (Array): number of repetition of new token.

        Returns:
            Array: New array with token_to_replace replaced by num_repetition repetitions of
             new_token
        """
        error_msg = "The number of image tokens must match the length of the tile tensor."
        assert np.sum(arr==token_to_replace) == len(num_repetition), error_msg
        result = []
        idx = 0
        for item in arr:
            if item == token_to_replace:
                # If the current item matches token_to_replace, add R copies of W
                result.extend([new_token] * num_repetition[idx])
                idx += 1
            else:
                # Otherwise, keep the original item
                result.append(item)

        return np.array(result)


    def encode_any_single_turn_vqa(self, sample):
        """Encode MultiChoiceVQA or VQA sample."""
        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        has_video = sample.__subflavors__['has_video'] if 'has_video' in sample.__subflavors__ else False

        if has_video:
            # Grab the selected frames of the video as a tensor with shape
            # fhwc: (num_frames, height, width, num_channels).
            video_fhwc = sample.image.permute(0, 2, 3, 1)
            selected_frames = torch.linspace(
                0, video_fhwc.shape[0] - 1, self.args.num_frames).long()
            video_frame_fhwc = video_fhwc[selected_frames]
            imgs = []
            for video_frame_hwc in video_frame_fhwc:
                imgs += self.transform_img(
                    video_frame_hwc, self.img_h, self.img_w,
                    self.args.use_tiling, self.args.max_num_tiles,
                    self.args.use_thumbnail, augment, find_closest_aspect_ratio_fn=self.find_closest_aspect_ratio_fn
                )
        else:
            imgs = self.transform_img(
                sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles,
                self.args.use_thumbnail, augment, find_closest_aspect_ratio_fn=self.find_closest_aspect_ratio_fn
            )

        num_tiles = [len(imgs)]

        if isinstance(sample, MultiChoiceVQASample):
            cur_prompt = format_multichoice_question(sample.context, sample.choices)
            if IMAGE_TOKEN not in cur_prompt:
                cur_prompt = IMAGE_TOKEN + "\n" + cur_prompt
            cur_answer = format_multichoice_answer(sample.correct_choice_idx)
        elif isinstance(sample, VQASample):
            if 'docvqa' in sample.__key__:
                prompt_list = self.manual_prompts["VQASFT"]["docvqa"]
            elif sample.__subflavors__.get("VQASFT"):
                prompt_list = self.manual_prompts["VQASFT"]["raw"]
            else:
                prompt_list = ["{}"]

            prompt_idx = np.random.randint(len(prompt_list))
            cur_prompt = prompt_list[prompt_idx]

            cur_prompt = cur_prompt.format(sample.context)

            if IMAGE_TOKEN not in cur_prompt:
                cur_prompt = IMAGE_TOKEN + "\n" + cur_prompt

            if isinstance(sample.answers, list):
                answer_list = sample.answers
                weight_list = np.array(sample.answer_weights).astype(np.float32)
                weight_list = weight_list / np.sum(weight_list)
                answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
                cur_answer = answer_list[answer_idx]
            else:
                cur_answer = sample.answers
        else:
            raise NotImplementedError("Unsupported data type provided", sample)

        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": cur_prompt},
            {"role": "assistant", "content": str(cur_answer)},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conversation, True, False)

        if self.is_packing_enabled:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles, imgs)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles, imgs),
        )

    def combined_ocr_encoder(self, sample, task_type):
        """Encode OCR samples."""
        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False

        if task_type == "encode_pdf":
            sample, cur_prompt, cur_answer = self.encode_pdf_prompt(sample)
        elif task_type == "encode_ocr_ref":
            sample, cur_prompt, cur_answer = self.encode_ocr_ref_prompt(sample)
        elif task_type == "_encode_ocr":
            sample, cur_prompt, cur_answer = self.encode_ocr_prompt(sample)

        imgs = self.transform_img(
            sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles,
            self.args.use_thumbnail, augment, find_closest_aspect_ratio_fn=self.find_closest_aspect_ratio_fn
        )
        num_tiles = [len(imgs)]

        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": cur_prompt},
            {"role": "assistant", "content": str(cur_answer)},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conversation, True, False)

        if self.is_packing_enabled:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles, imgs)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles, imgs),
        )

    def encode_pdf_prompt(self, sample: OCRSample) -> ImageTaskSample:
        """Encode OCR sample."""
        prompt_list = self.manual_prompts["DocPretraining"]["raw"]
        prompt_idx = np.random.randint(len(prompt_list))
        cur_prompt = prompt_list[prompt_idx]
        if IMAGE_TOKEN not in cur_prompt:
            cur_prompt = IMAGE_TOKEN + "\n" + cur_prompt

        # Make sure there is no extra IMAGE_TOKEN tag.
        sample.text = sample.text.replace(IMAGE_TOKEN, "")

        caption = sample.text.strip()

        split_by_line_flag = sample.__subflavors__.get("SplitByLine")
        if split_by_line_flag:
            caption_list = caption.split('\n')
            caption = np.random.choice(caption_list)
        cur_answer = caption

        return sample, cur_prompt, cur_answer

    def encode_ocr_ref_prompt(self, sample: OCRSample) -> ImageTaskSample:
        """Encode OCR sample."""
        ref = sample.text
        region = sample.words_boxes

        # Make sure there is no extra IMAGE_TOKEN tag
        ref = ref.replace(IMAGE_TOKEN, "")

        if len(region) == 4:
            region = f"<box>({region[0]},{region[1]}),({region[2]},{region[3]})</box>"
        else:
            region = f"<quad>({region[0]},{region[1]}),({region[2]},{region[3]}),({region[4]},{region[5]}),({region[6]},{region[7]})</quad>"

        # Randomly choose between two tasks
        task_idx = np.random.randint(2)
        if task_idx == 0:
            # Referring Grounding
            prompt_list = self.manual_prompts["DocPretraining"]["referring_grounding"]
            prompt_content = ref
            answer = region
        else:
            # Grounded OCR
            prompt_list = self.manual_prompts["DocPretraining"]["grounded_ocr"]
            prompt_content = region
            answer = ref

        prompt_idx = np.random.randint(len(prompt_list))
        cur_prompt = prompt_list[prompt_idx]
        cur_prompt = cur_prompt.format(prompt_content)
        if IMAGE_TOKEN not in cur_prompt:
            cur_prompt = IMAGE_TOKEN + "\n" + cur_prompt

        return sample, cur_prompt, answer

    def bbox_coord_to_label(self, text, bbox):
        """Format bbox coordinates as text."""
        assert len(bbox) == 4 or len(bbox) == 8

        # Make sure there is no extra IMAGE_TOKEN tag
        text = text.replace(IMAGE_TOKEN, "")

        if len(bbox) == 4:
            label_str = f"<ref>{text}</ref><box>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})</box>"
        else:
            label_str = f"<ref>{text}</ref><quad>({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]}),({bbox[4]},{bbox[5]}),({bbox[6]},{bbox[7]})</quad>"

        return label_str

    def encode_ocr_prompt(self, sample: OCRSample) -> ImageTaskSample:
        """Encode OCR sample."""
        if isinstance(sample.words_boxes[0], int):
            answer = self.bbox_coord_to_label(sample.text, sample.words_boxes)
        elif isinstance(sample.words_boxes[0], list):
            answer = ""
            for i, bbox in enumerate(sample.words_boxes):
                answer += self.bbox_coord_to_label(sample.words_text[i], bbox)

        prompt_list = self.manual_prompts["DocPretraining"]["ocr_multi"]
        prompt_idx = np.random.randint(len(prompt_list))
        cur_prompt = prompt_list[prompt_idx]

        if IMAGE_TOKEN not in cur_prompt:
            cur_prompt = IMAGE_TOKEN + "\n" + cur_prompt
        cur_answer = answer

        return sample, cur_prompt, cur_answer

    def batch(self, samples: List[Union[ImageTaskSample, ImageTaskSamplePacked]]) -> ImageTaskBatchPacked:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]

        if len(imgs) > 0 and self.args.dynamic_resolution:
            assert "radio" in self.args.vision_model_type or self.args.vision_model_type in ["clip", "siglip"], (
                "Dynamic resolution currently only works with radio or clip/siglip"
            )

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
            target_len = min(max_seq_len+1, len(s.labels))

            tokens[i, :text_len] = s.tokens[:text_len]
            labels[i, :target_len] = s.labels[:target_len]

        num_tiles = torch.tensor([n for s in samples for n in s.num_tiles], dtype=torch.int32)

        total_seq_len = self._get_total_seq_length(tokens[0], num_tiles, imgs)

        if len(num_tiles) == 0:
            num_tiles = torch.tensor([[0]], dtype=torch.int32)

        # Pad image packed seq length to be % 16 if using fp8 and dynamic resolution
        has_fp8 = self.args.fp8 is not None
        has_pad_img = torch.tensor(False)
        # TODO: Context parallel currently requires padding per CP rank so we do it later if needed.
        no_cp = self.args.context_parallel_size == 1

        if has_fp8 and self.args.dynamic_resolution and no_cp:
            img_seq_len = 0
            for img in imgs:
                img_seq_len += (img.shape[1] // self.args.patch_dim) * (img.shape[2] // self.args.patch_dim)
            padding_needed = get_padding(img_seq_len, self.args.context_parallel_size, self.args.tensor_model_parallel_size, self.args.sequence_parallel,
                                         self.args.tp_comm_overlap, self.args.decoder_seq_length, fp8_enabled=has_fp8)
            if padding_needed > 0:
                pad_img = torch.zeros([3, self.args.patch_dim, padding_needed * self.args.patch_dim])
                imgs.append(pad_img)
                has_pad_img = torch.tensor(True)

        imgs, imgs_sizes, vision_cu_lengths, vision_max_lengths = process_images(
            imgs, self.args.patch_dim, self.args.dynamic_resolution, batch_mode=True
        )

        # Set default values if no vision metadata was returned (static resolution case)
        if vision_cu_lengths is None:
            vision_cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        if vision_max_lengths is None:
            vision_max_lengths = torch.tensor([[0]], dtype=torch.int32)

        # Cumulative sample lengths are needed for packing, otherwise use dummy values.
        cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        cu_lengths_padded = torch.tensor([[0]], dtype=torch.int32)
        max_lengths = torch.tensor([[0]], dtype=torch.int32)

        is_packed = isinstance(samples[0], ImageTaskSamplePacked)
        if is_packed:
            cu_lengths = torch.stack([s.cu_lengths for s in samples])
            cu_lengths_padded = torch.stack([s.cu_lengths_padded for s in samples])
            max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)

            if self.dataloader_seq_length is not None:
                for i in range(len(samples)):
                    cu_lengths[i][-1] = self.dataloader_seq_length
                    cu_lengths_padded[i][-1] = self.dataloader_seq_length
                    new_max_length = cu_lengths_padded[i][-1] - cu_lengths[i][-2]
                    max_lengths[i] = torch.max(max_lengths[i], new_max_length)

        # Pad entire sequence to be a multiple of 16 if using fp8
        if has_fp8:
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
                tokens = torch.cat([tokens, torch.full((tokens.shape[0], padding_needed), self.tokenizer.pad, dtype=torch.int64)], dim=1)
                labels = torch.cat([labels, torch.full((labels.shape[0], padding_needed), IGNORE_INDEX, dtype=torch.int64)], dim=1)
                if is_packed:
                    cu_lengths[0][-1] += padding_needed
                    cu_lengths_padded[0][-1] += padding_needed
                    new_max_length = cu_lengths_padded[0][-1] - cu_lengths[0][-2]
                    max_lengths = torch.max(max_lengths, new_max_length)

        # No sound support in this old dataloading file.
        sound_clips = torch.tensor([[0]], dtype=torch.float32)
        sound_length = torch.tensor([[0]], dtype=torch.int64)
        sound_timestamps = torch.tensor([[0]], dtype=torch.float32)
        num_sound_clips = torch.tensor([[0]], dtype=torch.int64)

        return ImageTaskBatchPacked(
            __key__=[s.__key__ for s in samples],
            __restore_key__=[s.__restore_key__ for s in samples],
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__,
            tokens=tokens,
            labels=labels,
            imgs=imgs,
            num_tiles=num_tiles,
            cu_lengths=cu_lengths,
            cu_lengths_padded=cu_lengths_padded,
            max_lengths=max_lengths,
            imgs_sizes=imgs_sizes,
            vision_cu_lengths=vision_cu_lengths,
            vision_max_lengths=vision_max_lengths,
            has_pad_img=has_pad_img,
            sound_clips=sound_clips,
            sound_length=sound_length,
            sound_timestamps=sound_timestamps,
            num_sound_clips=num_sound_clips,
        )

    def encode_batch(self, batch: ImageTaskBatchPacked) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw

    def select_samples_to_pack(self, samples: List[ImageTaskSample]) -> List[List[ImageTaskSample]]:
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/advanced/packing.html
        """
        lengths = [sample.total_len for sample in samples]

        packed_samples = greedy_knapsack(lengths, samples, self.packing_seq_length)

        return packed_samples

    @stateless
    def pack_selected_samples(self, samples: List[ImageTaskSample]) -> List[ImageTaskSamplePacked]:
        """
        Function to pack a list of ImageTaskSample into a single ImageTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/advanced/packing.html

        Args:
            samples: List of ImageTaskSample instances to pack into one sample.

        Returns:
            ImageTaskSamplePacked instance.
        """
        packing_seq_len = self.packing_seq_length

        packed_tokens = []
        packed_labels = []
        packed_imgs = []

        current_length = 0
        max_length = 0
        cu_lengths = [0]
        cu_lengths_padded = [0]

        # Process each sample and build lists that we will concatenate to create the packed sample.
        for _, sample in enumerate(samples):
            sample_len = sample.total_len

            if sample_len > max_length:
                max_length = sample_len

            # If adding this sample exceeds the max length, stop.
            # This should not happen. The select_samples_to_pack method should have already ensured that the samples fit.
            if current_length + sample_len > packing_seq_len:
                raise ValueError(f"Packed sample exceeds the maximum sequence length of {packing_seq_len}: {samples}")

            # Add the sample's tokens and labels
            packed_tokens.append(sample.tokens)
            packed_labels.append(sample.labels)

            # Add the images
            packed_imgs += sample.imgs

            current_length += sample_len
            cu_lengths.append(current_length)
            cu_lengths_padded.append(current_length)

        # Concatenate packed tokens and labels.
        packed_tokens = torch.cat(packed_tokens, dim=0)
        packed_labels = torch.cat(packed_labels, dim=0)

        return ImageTaskSamplePacked(
            __key__=",".join([s.__key__ for s in samples]),
            __restore_key__=(),  # Will be set by energon based on `samples`
            __subflavor__=None,
            __subflavors__=samples[0].__subflavors__,
            tokens=packed_tokens,
            labels=packed_labels,
            imgs=packed_imgs,
            cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
            cu_lengths_padded=torch.tensor(cu_lengths_padded, dtype=torch.int32),
            max_length=max_length,
            num_tiles=[n for s in samples for n in s.num_tiles],
        )
