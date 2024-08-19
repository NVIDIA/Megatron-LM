# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import json
import random
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import conversation as conversation_lib
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms import Compose, RandAugment, RandomResizedCrop, Resize, ToPILImage

from megatron.core import mpu
from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    OCRSample,
    SimilarityInterleavedSample,
    VQASample,
)
from megatron.energon.transforms import CustomTransform, MergeTransform
from megatron.training import get_args
from megatron.training.tokenizer import build_tokenizer

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# Imagenet's mean and std.
pixel_mean = [123.675, 116.28, 103.53]
pixel_std = [58.395, 57.12, 57.375]


def convert_to_rgb(image):
    return image.convert("RGB")

def _transform_train(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0)),
        convert_to_rgb,
    ])

def _transform_train_aug(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0)),
        convert_to_rgb,
        RandAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    ])

def _transform_test(img_h, img_w):
    return Compose([
        ToPILImage(),
        Resize((img_h, img_w)),
        convert_to_rgb,
    ])

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
    img: torch.Tensor
    text: np.ndarray
    prompt_len: np.int64
    target: torch.Tensor = None
    img_size: Optional[tuple] = None


# Typing for the resulting batch data after encode_batch()
@dataclass
class ImageTaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (n, c, h, w)
    img: torch.Tensor
    # (n, seq_len)
    text: torch.Tensor
    # (n, 1)
    prompt_len: torch.Tensor
    # (n, seq_len)
    target: torch.Tensor

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Tokenizer:
    def __init__(self):

        args = get_args()
        self.args = args

        self.IMAGE_TOKEN_INDEX = -200
        self.initializer()

    def initializer(self):
        # Use Encoder class as a container for global data
        Tokenizer.tokenizer = build_tokenizer(self.args)
        if hasattr(Tokenizer.tokenizer, 'eod'):
            self.eod_token = Tokenizer.tokenizer.eod
        elif hasattr(Tokenizer.tokenizer, 'eos_id'):
            self.eod_token = Tokenizer.tokenizer.eos_id
        else:
            raise AttributeError('No eod token found in Tokenizer')
        self.split_token = 313131

        if (
            hasattr(self.args, "split_sentences") and self.args.split_sentences
        ):  # default false
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            library = "tokenizers/punkt/{}.pickle".format("english")
            # print("loading: " + library)
            splitter = nltk.load(library)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Tokenizer.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params, lang_vars=CustomLanguageVars()
                )
            else:
                Tokenizer.splitter = splitter
        else:
            Tokenizer.splitter = IdentitySplitter()

    def __call__(self, text: str, padded: bool = True): # -> torch.Tensor:
        sentence = Tokenizer.splitter.tokenize(text)[0]
        sentence = Tokenizer.tokenizer.tokenize(sentence)
        return sentence

    def pad(self, content, seq_len=1024):
        out = np.pad(content, pad_width=(0,max(0,seq_len-len(content))), mode='constant', constant_values=self.eod_token)

        return out

class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatch, dict]):
    """A simple task encoder for captioning."""

    def __init__(
        self
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()

        self.tokenizer = Tokenizer()
        self.manual_prompts = json.load(open(self.args.prompt_path))
        self.seq_len = self.args.decoder_seq_length - self.args.seq_length
        self.max_seq_len = self.seq_len

        self.txt_to_token_dict = {}

        self.img_h, self.img_w = self.args.img_h, self.args.img_w

        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        self.ocr_document_visual_transform = _get_ocr_document_visual_transform(self.img_h, self.img_w)
        self.ocr_document_identity_transform = _get_ocr_document_identity_transform(self.img_h, self.img_w)
        self.ocr_paragraph_visual_transform = _get_ocr_paragraph_visual_transform(self.img_h, self.img_w)

    def get_visual_transform(self, img_sample, sample_augmentation=False):
        img_sample = np.array(img_sample)

        raw_h, raw_w = img_sample.shape[0], img_sample.shape[1]
        ratio = float(max(self.img_h, self.img_w)) / max(raw_h, raw_w)
        scaled_h, scaled_w = int(raw_h * ratio + 0.5), int(raw_w * ratio + 0.5)

        # if the sample needs augmentation or not
        if sample_augmentation:
            # further check if augmentation is a global flag in args
            if self.args.aug:
                visual_transform = _transform_train_aug(scaled_h, scaled_w)
            else:
                visual_transform = _transform_train(scaled_h, scaled_w)
        else:
            visual_transform = _transform_test(scaled_h, scaled_w)

        img = visual_transform(img_sample)

        # Normalize pixel values.
        img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - self.pixel_mean) / self.pixel_std

        # Pad to target image size.
        delta_h, delta_w = self.img_h - scaled_h, self.img_w - scaled_w
        img = torch.nn.functional.pad(img, (0, delta_w, 0, delta_h))

        return img

    def encode_sample(self, sample: Union[CaptioningSample, OCRSample, VQASample, SimilarityInterleavedSample]):
        if isinstance(sample, OCRSample):
            yield self.encode_ocr(sample)
        elif isinstance(sample, CaptioningSample):
            yield self.encode_captioning(sample)
        elif isinstance(sample, VQASample):
            is_llava_training = sample.__subflavors__['is_llava_training'] if 'is_llava_training' in sample.__subflavors__ else False

            if "llava" in sample.__key__ or is_llava_training:
                yield self.encode_llava_pretrain(sample)
            else:
                yield self.encode_vqa(sample)
        elif isinstance(sample, SimilarityInterleavedSample):
            if "llava" in sample.__key__:
                yield self.encode_llava_sft(sample)
            else:
                raise NotImplementedError('Sample format not supported')
        else:
            raise NotImplementedError('Sample format not supported')

    def encode_captioning(self, sample: CaptioningSample):
        sample_augmentation = sample.__subflavors__.get("augmentation")
        conv_format = sample.__subflavors__['conv_format'] if 'conv_format' in sample.__subflavors__ else 'mistral'
        no_instruction = sample.__subflavors__['no_instruction'] if 'no_instruction' in sample.__subflavors__ else False

        img_size = np.array(sample.image.size)
        img = self.get_visual_transform(
            np.array(sample.image), sample_augmentation=sample_augmentation
        )

        prompt_list = self.manual_prompts["CaptioningPretraining"]["llava"]

        prompt_idx = np.random.randint(len(prompt_list))
        cur_prompt = prompt_list[prompt_idx]
        cur_prompt = "<image>\n" + cur_prompt + "\n"

        caption = sample.caption.strip()

        split_by_line_flag = sample.__subflavors__.get("SplitByLine")
        if split_by_line_flag:
            caption_list = caption.split('\n')
            caption = np.random.choice(caption_list)

        if conv_format == 'llama3_sft':
            conv = conversation_lib.llama3_instruct.copy()
            sep = conv.sep
        elif conv_format == "mistral":
            conv = conversation_lib.mistral_instruct.copy()
            conv = conv.sep2

        conversation = cur_prompt + caption + sep

        input_ids = np.array(tokenizer_image_token(self.args, conversation, self.tokenizer, has_image=True))
        target = input_ids.copy()

        prompt_len = len(tokenizer_image_token(self.args, cur_prompt, self.tokenizer))
        target[:prompt_len] = IGNORE_INDEX

        input_ids = self.tokenizer.pad(input_ids, self.max_seq_len+1) # pad with EOD
        target = self.tokenizer.pad(target, self.max_seq_len+1) #, pad_value=IGNORE_INDEX) # pad with ignore_index. this will be used to create loss_mask

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            img=img,
            text=input_ids,
            prompt_len=prompt_len,
            target=target,
            img_size=img_size
        )

    def encode_llava_pretrain(self, sample: VQASample):
        sample_augmentation = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False

        use_chat_format = sample.__subflavors__['use_chat_format'] if 'use_chat_format' in sample.__subflavors__ else False
        conv_format = sample.__subflavors__['conv_format'] if 'conv_format' in sample.__subflavors__ else "mistral"

        img_size = np.array(sample.image.size)
        img = self.get_visual_transform(sample.image, sample_augmentation=sample_augmentation)

        assert "<image>" in sample.context

        if use_chat_format:
            prompt_idx = np.random.randint(len(self.manual_prompts["Captioning"]["raw"]))
            prompt = self.manual_prompts["Captioning"]["raw"][prompt_idx]

            sample.context = "User: <image>" + "\n" + prompt + " Assistant: "
            conversation = sample.context + sample.answers + conversation_lib.mistral_instruct.sep
        else:
            # LLAVA training: override text-prompt with just IMAGE_TOKEN_INDEX
            sample.context = "<image>" + "\n"
            if conv_format == 'llama3_sft':
                conversation = sample.context + sample.answers + conversation_lib.llama3_instruct.sep
            elif conv_format == "mistral":
                conversation = sample.context + sample.answers + conversation_lib.mistral_instruct.sep2

        input_ids = np.array(tokenizer_image_token(self.args, conversation, self.tokenizer, has_image=True))
        target = input_ids.copy()

        prompt_len = len(tokenizer_image_token(self.args, sample.context, self.tokenizer))
        target[:prompt_len] = IGNORE_INDEX

        input_ids = self.tokenizer.pad(input_ids, self.max_seq_len+1) # pad with EOD
        target = self.tokenizer.pad(target, self.max_seq_len+1) #, pad_value=IGNORE_INDEX) # pad with ignore_index. this will be used to create loss_mask

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            img=img,
            text=input_ids,
            prompt_len=prompt_len,
            target=target,
            img_size=img_size
        )

    # Based on https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/train/train.py#L500
    def encode_llava_sft(self, sample: SimilarityInterleavedSample):
        sample_augmentation = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        use_chat_format = sample.__subflavors__['use_chat_format'] if 'use_chat_format' in sample.__subflavors__ else False
        has_image = sample.__subflavors__['has_image'] if 'has_image' in sample.__subflavors__ else False
        no_instruction = sample.__subflavors__['no_instruction'] if 'no_instruction' in sample.__subflavors__ else False
        conv_format = sample.__subflavors__['conv_format'] if 'conv_format' in sample.__subflavors__ else "mistral"

        if has_image:
            img_size = np.array(sample.images[0].size)
            img = self.get_visual_transform(sample.images[0], sample_augmentation=sample_augmentation)
        else:
            img_size = np.array([0,0])
            img = torch.from_numpy(np.array([-1]).astype(np.float32))
            sample.__key__ = "{}-{}".format("no-image", sample.__key__)

        if conv_format == 'llama3_sft':
            conv = conversation_lib.llama3_instruct.copy()
        elif conv_format == "mistral":
            conv = conversation_lib.mistral_instruct.copy()

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        if use_chat_format:
            source = sample.texts
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], sentence
                conv.append_message(role, sentence["value"])
            conversation = conv.get_prompt()

            ### Tokenize conversations
            input_ids = tokenizer_image_token(self.args, conversation, self.tokenizer, has_image)

            input_ids = torch.LongTensor(input_ids)
            target = input_ids.clone()

            if conv.sep_style == conversation_lib.SeparatorStyle.MPT:
                # Mask targets
                sep = conv.sep + conv.roles[1]

                total_len = int((target != self.tokenizer.eod_token).sum())

                rounds = conversation.split(conv.sep)
                re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
                for conv_idx in range(3, len(rounds), 2):
                    re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt

                cur_len = 0
                target[:cur_len] = IGNORE_INDEX

                for i, rou in enumerate(re_rounds):
                    if rou == "":
                        break

                    rou += conv.sep

                    parts = rou.split(sep)

                    if len(parts) != 2:
                        break
                    parts[0] += sep

                    round_len = len(tokenizer_image_token(self.args, rou, self.tokenizer, has_image))
                    instruction_len = len(tokenizer_image_token(self.args, parts[0], self.tokenizer, has_image))

                    if conv_format == 'llama3_sft' and i > 0:
                        round_len -= 1
                        instruction_len -= 1

                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len

                target[cur_len:] = IGNORE_INDEX

            elif conv.sep_style == conversation_lib.SeparatorStyle.TWO:
                ### Mask targets
                sep = conv.sep + conv.roles[1] + ": "

                total_len = int((target != self.tokenizer.eod_token).sum())

                rounds = conversation.split(conv.sep2)

                cur_len = 0

                for i, rou in enumerate(rounds):
                    if rou == "":
                        break

                    rou += conv.sep2 # put back conv.sep2 since we will lose it while we conversation.split above with conv.sep2

                    parts = rou.split(sep)

                    if len(parts) != 2:
                        break
                    parts[0] += sep

                    round_len = len(tokenizer_image_token(self.args, rou, self.tokenizer, has_image))
                    instruction_len = len(tokenizer_image_token(self.args, parts[0], self.tokenizer, has_image)) - 2

                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                    cur_len += round_len

                target[cur_len:] = IGNORE_INDEX

            elif conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
                raise NotImplementedError("this tokenizer is not supported yet with this data type")

            if cur_len < self.max_seq_len:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX

                    raise Exception(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. Something is wrong, please fix!"
                    )

        else:
            return NotImplementedError

        # pad to max_seq_len
        input_ids = self.tokenizer.pad(input_ids, self.max_seq_len+1) # pad with EOD
        target = self.tokenizer.pad(target, self.max_seq_len+1)

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            img=img,
            text=input_ids,
            prompt_len=instruction_len,
            target=target,
            img_size=img_size
        )

    def encode_vqa(self, sample: VQASample):
        sample_augmentation = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False

        img = self.get_visual_transform(sample.image, sample_augmentation=sample_augmentation)

        img_size = np.array(sample.image.size)

        if sample.context[-1:] != "\n":
            sample.context = sample.context + "\n"

        question_token = self.tokenizer(sample.context)
        if isinstance(sample.answers, list):
            answer_list = sample.answers
            weight_list = np.array(sample.answer_weights).astype(np.float32)
            weight_list = weight_list / np.sum(weight_list)
            answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
            answer = answer_list[answer_idx]
            answer_token = self.tokenizer(answer)
        else:
            answer_token = self.tokenizer(sample.answers)

        prompt_len = len(question_token)

        seq_len = self.max_seq_len + 4

        text_sample = np.concatenate([[IMAGE_TOKEN_INDEX], question_token, answer_token])
        text_sample = self.tokenizer.pad(text_sample, seq_len)

        target = text_sample.copy()
        target[:max(0, prompt_len - 1)] = IGNORE_INDEX

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            img=img,
            text=text_sample,
            prompt_len=prompt_len,
            target=target,
            img_size=img_size
        )

    def encode_ocr(self, sample: OCRSample) -> ImageTaskSample:
        if sample.__subflavors__["type"] == "document":
            visual_transform = self.ocr_document_visual_transform
        elif sample.__subflavors__["type"] == "paragraph":
            visual_transform = self.ocr_paragraph_visual_transform
        elif sample.__subflavors__["augmentation"] == False:
            visual_transform = self.ocr_document_identity_transform
        else:
            raise ValueError(f"Unknown subflavor {sample.__subflavors__}")

        if sample.words_boxes is not None and sample.words_boxes.shape[1] >= 5:
            # Boxes with conf below 0.9 are skipped
            filter_words_mask = sample.words_boxes[:, 4] < 0.9
            filter_boxes = sample.words_boxes[filter_words_mask, :4]
            for x, y, x2, y2 in filter_boxes:
                if isinstance(sample.image, Image.Image):
                    draw = ImageDraw.Draw(sample.image)
                    draw.rectangle([int(x), int(y), (int(x2), int(y2))], fill=0)
                else:
                    sample.image[:, int(y) : int(y2) + 1, int(x) : int(x2) + 1] = 0

            text = " ".join(
                text for skip, text in zip(filter_words_mask, sample.words_text) if not skip
            )
        else:
            text = " ".join(sample.text.splitlines())

        match = re.search(r'"text_sequence": "(.*?)"', text)
        if match:
            text = match.group(1)

        img = visual_transform(sample.image)
        img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - self.pixel_mean) / self.pixel_std
        img = torch.nn.functional.pad(img, (0, self.img_w - img.shape[2], 0, self.img_h - img.shape[1]))

        # randomly select a prompt
        prompt_idx = np.random.randint(len(self.manual_prompts["OCR"]["raw"]))
        cur_prompt = self.manual_prompts["OCR"]["raw"][prompt_idx]

        if cur_prompt not in self.txt_to_token_dict:
            self.txt_to_token_dict[cur_prompt] = self.tokenizer(cur_prompt)
        cur_prompt = self.txt_to_token_dict[cur_prompt]

        text_sample = self.tokenizer(text)
        prompt_len = len(cur_prompt)
        seq_len = self.seq_len + 4
        text_sample = np.concatenate([cur_prompt, text_sample])
        text_sample = self.tokenizer.pad(text_sample, seq_len=seq_len)
        text_sample = text_sample[:seq_len]

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            img=img,
            text=text_sample,
            prompt_len=prompt_len
        )

    def batch(self, samples: List[ImageTaskSample]) -> ImageTaskBatch:
        batch = ImageTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            img=torch.stack([s.img for s in samples]),
            text=torch.from_numpy(np.stack([s.text for s in samples], axis=0).astype(np.int64)),
            prompt_len=torch.from_numpy(np.array([s.prompt_len for s in samples], dtype=np.int64)),
            target=torch.from_numpy(np.stack([s.target for s in samples], axis=0).astype(np.int64)),
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

# From https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/mm_utils.py#L185
def tokenizer_image_token(args, prompt, tokenizer, has_image=True, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

    if not has_image:
        input_ids = tokenizer(prompt)

    else:
        prompt_chunks = [tokenizer(chunk) for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0

        if args.tokenizer_type in ['Llama2Tokenizer', 'Llama3Tokenizer'] and len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

    # # remove BOS token
    # if args.tokenizer_type in ['Llama2Tokenizer', 'Llama3Tokenizer']:
    #     return input_ids[1:]

    return input_ids
