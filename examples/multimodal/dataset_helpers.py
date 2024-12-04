# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import bisect
import dataclasses
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from image_processing import get_visual_transform
import numpy as np
import torch

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, IMAGE_TOKEN
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
            self.args.img_h,
            self.args.img_w,
            self.args.patch_dim,
            self.args.vision_model_type,
            self.args.disable_vision_class_token,
            1,
            self.args.pixel_shuffle,
            self.args.use_tile_tags,
        )

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w

    def _get_total_seq_length(self, input_ids, num_tiles):
        """Calculate expected sequence length given text tokens length and number of tiles."""
        total_num_images = len(num_tiles)
        total_num_tiles = sum(num_tiles)
        total_len = len(input_ids) + total_num_tiles * self.num_image_embeddings_per_tile - total_num_images
        return total_len

    def _truncate_for_packing(self, input_ids, target, num_tiles):
        """Truncate tokens and labels if they exceed packing sequence length."""
        total_num_images = len(num_tiles)
        total_num_tiles = sum(num_tiles)
        total_img_embeddings_len = total_num_tiles * self.num_image_embeddings_per_tile
        max_text_tokens = self.packing_seq_length - total_img_embeddings_len + total_num_images

        input_ids = input_ids[:max_text_tokens]
        target = target[:max_text_tokens]

        # If truncate causes all labels to be ignored, then skip the sample
        if (target == IGNORE_INDEX).all():
            raise ValueError(f"all targets will be ignored after truncation: {input_ids}")

        return input_ids, target

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: Union[CaptioningSample, OCRSample, VQASample, SimilarityInterleavedSample]):
        if isinstance(sample, OCRSample):
            if "pdfa" in sample.__key__:
                yield self.combined_ocr_encoder(sample, task_type='encode_pdf')
            elif "multi" in sample.__key__:
                yield self.combined_ocr_encoder(sample, task_type='_encode_ocr')
            else:
                yield self.combined_ocr_encoder(sample, task_type='encode_ocr_ref')
        elif isinstance(sample, CaptioningSample):
            yield self.encode_captioning(sample)
        elif isinstance(sample, VQASample):
            is_llava_training = sample.__subflavors__["is_llava_training"] if "is_llava_training" in sample.__subflavors__ else False

            if "llava" in sample.__key__ or is_llava_training:
                yield self.encode_llava_pretrain(sample)
            else:
                yield self.encode_any_single_turn_vqa(sample)
        elif isinstance(sample, SimilarityInterleavedSample):
            yield self.encode_llava_sft(sample)
        elif isinstance(sample, MultiChoiceVQASample):
            yield self.encode_any_single_turn_vqa(sample)
        else:
            raise NotImplementedError("Sample format not supported", sample)

    def encode_captioning(self, sample: CaptioningSample):
        """Encode CaptioningSample."""
        augment = sample.__subflavors__.get("augmentation")

        imgs = get_visual_transform(
            sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
            self.args.vision_model_type,
        )
        num_tiles = [len(imgs)]

        prompt_list = self.manual_prompts["CaptioningPretraining"]["raw"]

        prompt_idx = np.random.randint(len(prompt_list))
        cur_prompt = prompt_list[prompt_idx]
        cur_prompt = "<image>\n" + cur_prompt + "\n"

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
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles),
        )

    def encode_llava_pretrain(self, sample: VQASample):
        """Encode pretrain sample in LLAVA style."""
        augment = sample.__subflavors__.get("augmentation", False)

        imgs = get_visual_transform(
            sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
            self.args.vision_model_type,
        )
        num_tiles = [len(imgs)]

        # LLAVA training: override text-prompt with just the image.
        conv = [
            # Note: no system message.
            {"role": "user", "content": "<image>\n"},
            {"role": "assistant", "content": sample.answers},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conv, True, False)

        if self.is_packing_enabled:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles),
        )

    def encode_llava_sft(self, sample: SimilarityInterleavedSample):
        """Encode SFT sample."""
        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        has_video = sample.__subflavors__['has_video'] if 'has_video' in sample.__subflavors__ else False
        has_image = sample.__subflavors__['has_image'] if 'has_image' in sample.__subflavors__ else False
        has_image = has_image or (hasattr(sample, "images") and len(sample.images) > 0)

        if has_video:
            # Grab the selected frames of the video as a tensor with shape
            # fhwc: (num_frames, height, width, num_channels).
            video_fhwc = sample.images[0].permute(0, 2, 3, 1)
            selected_frames = torch.linspace(
                0, video_fhwc.shape[0] - 1, self.args.num_frames).long()
            video_frame_fhwc = video_fhwc[selected_frames]
            imgs = []
            for video_frame_hwc in video_frame_fhwc:
                imgs += get_visual_transform(
                    video_frame_hwc, self.img_h, self.img_w,
                    self.args.use_tiling, self.args.max_num_tiles,
                    self.args.use_thumbnail, augment, self.args.vision_model_type)
            num_tiles = [len(imgs)]
        elif has_image:
            imgs = get_visual_transform(
                sample.images[0], self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
                self.args.vision_model_type,
            )
            num_tiles = [len(imgs)]
        else:
            imgs = num_tiles = []
            sample.__key__ = "{}-{}".format("no-image", sample.__key__)

        conversation = []
        # Note: Some tokenizers may ignore the system prompt.
        conversation.append({"role": "system", "content": "Answer the questions."})

        has_image_token = False

        for text in sample.texts:
            if IMAGE_TOKEN in text["value"]:
                has_image_token = True

            if text["from"] == "human":
                role = "user"
            elif text["from"] == "gpt":
                role = "assistant"
            else:
                raise RuntimeError(f"unexpected role {text['from']} in {sample.texts}")

            turn = {"role": role, "content": text["value"]}
            conversation.append(turn)

        # If the sample contains an image but none of the user messages has an image token,
        # then add it to the first user message.
        if len(imgs) > 0 and not has_image_token:
            for turn in conversation:
                if turn["role"] == "user":
                    turn["content"] = f"{IMAGE_TOKEN}\n" + turn["content"]
                    break

        input_ids, target = self.tokenizer.tokenize_conversation(conversation, True, False)

        if self.is_packing_enabled:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles),
        )

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
                imgs += get_visual_transform(
                    video_frame_hwc, self.img_h, self.img_w,
                    self.args.use_tiling, self.args.max_num_tiles,
                    self.args.use_thumbnail, augment, self.args.vision_model_type)
        else:
            imgs = get_visual_transform(
                sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles,
                self.args.use_thumbnail, augment, self.args.vision_model_type,
            )

        num_tiles = [len(imgs)]

        if isinstance(sample, MultiChoiceVQASample):
            cur_prompt = format_multichoice_question(sample.context, sample.choices)
            if "<image>" not in cur_prompt:
                cur_prompt = "<image>\n" + cur_prompt
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

            if "<image>" not in cur_prompt:
                cur_prompt = "<image>\n" + cur_prompt

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
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles),
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

        imgs = get_visual_transform(
                sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles,
                self.args.use_thumbnail, augment, self.args.vision_model_type,
            )
        num_tiles = [len(imgs)]

        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": cur_prompt},
            {"role": "assistant", "content": str(cur_answer)},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conversation, True, False)

        if self.is_packing_enabled:
            input_ids, target = self._truncate_for_packing(input_ids, target, num_tiles)

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            tokens=torch.tensor(input_ids),
            labels=torch.tensor(target),
            total_len=self._get_total_seq_length(input_ids, num_tiles),
        )

    def encode_pdf_prompt(self, sample: OCRSample) -> ImageTaskSample:
        """Encode OCR sample."""
        prompt_list = self.manual_prompts["DocPretraining"]["raw"]
        prompt_idx = np.random.randint(len(prompt_list))
        cur_prompt = prompt_list[prompt_idx]
        if "<image>" not in cur_prompt:
            cur_prompt = "<image>\n" + cur_prompt

        # Make sure there is no extra <image> tag.
        sample.text = sample.text.replace("<image>", "")

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

        # Make sure there is no extra <image> tag
        ref = ref.replace("<image>", "")

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
        if "<image>" not in cur_prompt:
            cur_prompt = "<image>\n" + cur_prompt

        return sample, cur_prompt, answer

    def bbox_coord_to_label(self, text, bbox):
        """Format bbox coordinates as text."""
        assert len(bbox) == 4 or len(bbox) == 8

        # Make sure there is no extra <image> tag
        text = text.replace("<image>", "")

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

        if "<image>" not in cur_prompt:
            cur_prompt = "<image>\n" + cur_prompt
        cur_answer = answer

        return sample, cur_prompt, cur_answer

    def batch(self, samples: List[Union[ImageTaskSample, ImageTaskSamplePacked]]) -> ImageTaskBatchPacked:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]
        if len(imgs) > 0:
            imgs = torch.stack(imgs)
        else:
            imgs = torch.tensor([[0]], dtype=torch.float32)

        # If the user hasn't defined a target dataloader sequence length, then use the max along the sample lengths.
        max_seq_len = self.dataloader_seq_length
        if not max_seq_len:
           max_seq_len = max(len(s.tokens) for s in samples)

        tokens = np.full((len(samples), max_seq_len), self.tokenizer.pad, dtype=np.int64)
        # +1 to accommodate shift to left by one later.
        labels = np.full((len(samples), max_seq_len + 1), self.tokenizer.pad, dtype=np.int64)

        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.tokens))
            target_len = min(max_seq_len+1, len(s.labels))

            tokens[i, :text_len] = s.tokens[:text_len]
            labels[i, :target_len] = s.labels[:target_len]

        num_tiles = torch.tensor([n for s in samples for n in s.num_tiles], dtype=torch.int32)
        if len(num_tiles) == 0:
            num_tiles = torch.tensor([[0]], dtype=torch.int32)

        # Cumulative sample lengths are needed for packing, otherwise use dummy values.
        cu_lengths = torch.tensor([[0]], dtype=torch.int32)
        max_lengths = torch.tensor([[0]], dtype=torch.int32)

        if self.is_packing_enabled:
            cu_lengths = torch.stack([s.cu_lengths for s in samples])
            max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)

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
            max_lengths=max_lengths,
        )

    def encode_batch(self, batch: ImageTaskBatchPacked) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw

    def select_samples_to_pack(self, samples: List[ImageTaskSample]) -> List[List[ImageTaskSample]]:
        """Selects which samples will be packed together.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html
        """
        lengths = [sample.total_len for sample in samples]

        packed_samples = greedy_knapsack(lengths, samples, self.packing_seq_length)

        return packed_samples

    @stateless
    def pack_selected_samples(self, samples: List[ImageTaskSample]) -> List[ImageTaskSamplePacked]:
        """
        Function to pack a list of ImageTaskSample into a single ImageTaskSamplePacked.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/packing.html

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
            max_length=max_length,
            num_tiles=[n for s in samples for n in s.num_tiles],
        )


def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()


def format_multichoice_question(question, multichoice_options):
    """Format multi-choice question."""
    options_text = ["{}. {}\n".format(chr(ord('A') + i), option) for i, option in
                    zip(range(len(multichoice_options)), multichoice_options)]
    options_text = "".join(options_text)

    options_text = f"{options_text}Answer with the option's letter from the given choices directly."

    return "{}\n{}".format(question, options_text)


def format_multichoice_answer(idx):
    """Format multi-choice answer."""
    return chr(ord('A') + idx)
