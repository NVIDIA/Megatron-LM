# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from image_processing import get_visual_transform
import numpy as np
import torch

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX
from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    OCRSample,
    SimilarityInterleavedSample,
    VQASample,
    MultiChoiceVQASample
)
from megatron.training import get_args, get_tokenizer


# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __restore_key__: str
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
    __restore_key__: str
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

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
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

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
        )

    def encode_llava_sft(self, sample: SimilarityInterleavedSample):
        """Encode SFT sample."""
        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        has_image = sample.__subflavors__['has_image'] if 'has_image' in sample.__subflavors__ else False
        has_video = sample.__subflavors__['has_video'] if 'has_video' in sample.__subflavors__ else False

        if has_image:
            imgs = get_visual_transform(
                sample.images[0], self.img_h, self.img_w, self.args.use_tiling, self.args.max_num_tiles, self.args.use_thumbnail, augment,
                self.args.vision_model_type,
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
                    self.args.use_thumbnail, augment, self.args.vision_model_type)
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
            __restore_key__=sample.__restore_key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
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

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
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

        return ImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            num_tiles=num_tiles,
            text=input_ids,
            target=target,
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
            __restore_key__=[s.__restore_key__ for s in samples],
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
