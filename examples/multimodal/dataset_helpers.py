# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import itertools
import json
import random
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from image_processing import get_visual_transform
import numpy as np
import torch
from torchvision import transforms as T

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX
from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    OCRSample,
    SimilarityInterleavedSample,
    VQASample,
)
from megatron.energon.transforms import CustomTransform, MergeTransform
from megatron.training import get_args, get_tokenizer


class RandomResize(CustomTransform):
    """Resizes the image by a random scale factor in the given interval, but at most max_size"""

    def __init__(self, min_scale: float, max_scale: float, max_size: int):
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_size = max_size

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        scale = random.uniform(self._min_scale, self._max_scale)
        new_size = tuple(int(x * scale) for x in dst_size)

        if max(new_size) > self._max_size:
            scale = self._max_size / max(new_size)
            new_size = tuple(int(x * scale) for x in dst_size)

        matrix = self.scale(scale, scale) @ matrix
        dst_size = np.array(new_size, dtype=dst_size.dtype)

        return matrix, dst_size, (self.__class__.__name__, scale)


class RandomResizeLongEdge(CustomTransform):
    """Resizes the image's longer edge to a random length between min_size and max_size pixels."""

    def __init__(self, min_size: int, max_size: int):
        self._min_size = min_size
        self._max_size = max_size

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        new_long = random.randint(self._min_size, self._max_size)
        if dst_size[0] > dst_size[1]:  # h > w
            new_w, new_h = int(new_long * dst_size[1] / dst_size[0]), new_long
        else:  # w > h
            new_w, new_h = new_long, int(new_long * dst_size[0] / dst_size[1])

        new_size = (new_h, new_w)
        matrix = self.scale(new_w / dst_size[1], new_h / dst_size[0]) @ matrix
        dst_size = np.array(new_size, dtype=dst_size.dtype)

        return matrix, dst_size, (self.__class__.__name__, new_size)


class RandomPad(CustomTransform):
    """Pads the image to the given size, randomly choosing the position of the image within the new larger image.
    If the image is already larger than the given size, it will not be padded in that direction(s)."""

    def __init__(self, size: Tuple[int, int]):
        self._new_size = size  # h, w

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        h_pad = max(self._new_size[0] - dst_size[0], 0)
        w_pad = max(self._new_size[1] - dst_size[1], 0)

        if h_pad == 0 and w_pad == 0:
            return matrix, dst_size, (self.__class__.__name__, None)
        else:
            # TODO: fix me
            # top = random.randint(0, h_pad)
            # left = random.randint(0, w_pad)
            top = 0
            left = 0

            matrix = self.translate(left, top) @ matrix
            dst_size = np.array(self._new_size, dtype=dst_size.dtype)
            return matrix, dst_size, (self.__class__.__name__, (top, left))


def _get_ocr_document_visual_transform(IMG_H=1024, IMG_W=1024):
    document_visual_transform = T.Compose(
        [
            MergeTransform(
                [
                    # T.RandomResizedCrop(size=FINAL_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                    RandomResizeLongEdge(960, 1008),  # Note: 1008 comes from list(range(960, 1024, 16))[-1]
                    T.RandomRotation(5, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomPerspective(distortion_scale=0.1, p=0.1),
                    RandomPad((IMG_H, IMG_W)),
                ]
            ),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0)),
            T.RandomGrayscale(p=0.5),
            T.RandomInvert(p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            # LogImage(),
            # T.ToTensor(),
            # T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return document_visual_transform

def _get_ocr_document_identity_transform(IMG_H=1024, IMG_W=1024):
    long_edge = max(IMG_H, IMG_W)
    document_identity_transform = T.Compose(
        [
            MergeTransform(
                [
                    RandomResizeLongEdge(long_edge, long_edge),
                    RandomPad((long_edge, long_edge)),
                ]
            )
        ]
    )
    return document_identity_transform

def _get_ocr_paragraph_visual_transform(IMG_H=1024, IMG_W=1024):
    paragraph_visual_transform = T.Compose(
        [
            MergeTransform(
                [
                    # T.RandomResizedCrop(size=FINAL_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                    RandomResize(0.5, 2.0, min(IMG_H, IMG_W)), #FINAL_SIZE),
                    T.RandomRotation(1, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomPerspective(distortion_scale=0.1, p=0.1),
                    RandomPad((IMG_H, IMG_W)),
                ]
            ),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0)),
            T.RandomGrayscale(p=0.5),
            T.RandomInvert(p=0.5),
            # T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            # T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            # LogImage(),
            # T.ToTensor(),
            # T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return paragraph_visual_transform

# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavors__: Dict
    # (c, h, w)
    imgs: List[torch.Tensor]
    num_tiles: List[int]
    text: np.ndarray
    target: torch.Tensor = None


# Typing for the resulting batch data after encode_batch()
@dataclass
class ImageTaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (num_tiles, c, h, w)
    imgs: torch.Tensor
    num_tiles: List[int]
    # (n, seq_len)
    text: torch.Tensor
    # (n, seq_len)
    target: torch.Tensor


class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatch, dict]):
    """A simple task encoder for captioning."""

    def __init__(
        self
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()

        self.tokenizer = get_tokenizer()
        self.manual_prompts = json.load(open(self.args.prompt_path))
        self.seq_len = self.args.dataloader_seq_length

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w


    def encode_sample(self, sample: Union[CaptioningSample, OCRSample, VQASample, SimilarityInterleavedSample]):
        if isinstance(sample, CaptioningSample):
            yield self.encode_captioning(sample)
        elif isinstance(sample, VQASample):
            is_llava_training = sample.__subflavors__['is_llava_training'] if 'is_llava_training' in sample.__subflavors__ else False

            if "llava" in sample.__key__ or is_llava_training:
                yield self.encode_llava_pretrain(sample)
            else:
                yield self.encode_vqa(sample)
        elif isinstance(sample, SimilarityInterleavedSample):
            if "llava" or "video" in sample.__key__:
                yield self.encode_llava_sft(sample)
            else:
                raise NotImplementedError('Sample format not supported')
        else:
            raise NotImplementedError('Sample format not supported')

    def encode_captioning(self, sample: CaptioningSample):
        augment = sample.__subflavors__.get("augmentation")

        imgs = get_visual_transform(
            sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
        )
        num_tiles = [len(imgs)]

        prompt_list = self.manual_prompts["CaptioningPretraining"]["llava"]

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

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
        )

    def encode_llava_pretrain(self, sample: VQASample):
        augment = sample.__subflavors__.get("augmentation", False)

        imgs = get_visual_transform(
            sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
        )
        num_tiles = [len(imgs)]

        # LLAVA training: override text-prompt with just the image.
        conv = [
            # Note: no system message.
            {"role": "user", "content": "<image>\n"},
            {"role": "assistant", "content": sample.answers},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conv, True, False)

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
        )

    def encode_llava_sft(self, sample: SimilarityInterleavedSample):
        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        has_image = sample.__subflavors__['has_image'] if 'has_image' in sample.__subflavors__ else False
        has_video = sample.__subflavors__['has_video'] if 'has_video' in sample.__subflavors__ else False

        if has_image:
            imgs = get_visual_transform(
                sample.images[0], self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
            )
            num_tiles = [len(imgs)]
        elif has_video:
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
                    self.args.use_thumbnail, augment=False)
            num_tiles = [len(imgs)]
        else:
            imgs = num_tiles = []
            sample.__key__ = "{}-{}".format("no-image", sample.__key__)

        conversation = []
        # Note: Some tokenizers may ignore the system prompt.
        conversation.append({"role": "system", "content": "Answer the questions."})

        for text in sample.texts:
            if text["from"] == "human":
                role = "user"
            elif text["from"] == "gpt":
                role = "assistant"
            else:
                raise RuntimeError(f"unexpected role {text['from']} in {sample.texts}")

            turn = {"role": role, "content": text["value"]}
            conversation.append(turn)

        input_ids, target = self.tokenizer.tokenize_conversation(conversation, True, False)

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
        )

    def encode_vqa(self, sample: VQASample):
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
                    self.args.use_thumbnail, augment=False)
        else:
            imgs = get_visual_transform(
                sample.image, self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
            )
        num_tiles = [len(imgs)]

        if "<image>" not in sample.context:
            sample.context = "<image>" + sample.context

        if isinstance(sample.answers, list):
            answer_list = sample.answers
            weight_list = np.array(sample.answer_weights).astype(np.float32)
            weight_list = weight_list / np.sum(weight_list)
            answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
            answer = answer_list[answer_idx]
        else:
            answer = sample.answers

        conversation = [
            {"role": "user", "content": sample.context},
            {"role": "assistant", "content": answer},
        ]

        input_ids, target = self.tokenizer.tokenize_conversation(conversation, True, False)

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
        )

    def batch(self, samples: List[ImageTaskSample]) -> ImageTaskBatch:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]
        if len(imgs) > 0:
            imgs = torch.stack(imgs)
        else:
            imgs = torch.tensor([[0]], dtype=torch.float32)

        # Put tile counts to a single tensor. If there are no images (text-only), then use a dummy tensor.
        num_tiles = torch.tensor([n for s in samples for n in s.num_tiles], dtype=torch.int)
        if len(num_tiles) == 0:
            num_tiles = torch.tensor([[0]], dtype=torch.int)

        # If the user hasn't defined a target sequence length, then use the max along the sample lengths.
        max_seq_len = self.seq_len
        if not max_seq_len:
            max_seq_len = max(len(s.text) for s in samples)

        text_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad, dtype=np.int64)
        # +1 to accommodate shift to left by one later.
        target_mat = np.full((len(samples), max_seq_len + 1), self.tokenizer.pad, dtype=np.int64)

        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len+1, len(s.target))

            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]

        batch = ImageTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            imgs=imgs,
            num_tiles=num_tiles,
            text=torch.from_numpy(text_mat),
            target=torch.from_numpy(target_mat),
        )

        return batch

    def encode_batch(self, batch: ImageTaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw


def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
