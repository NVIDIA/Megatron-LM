# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Protocol, Union

import torch
import torch.nn.utils.rnn as rnn_utils
from PIL import Image
import numpy as np
import soundfile as sf
from scipy import signal
import io
from megatron.training.global_vars import get_tokenizer

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir,
            os.path.pardir,
            "examples/multimodal",
        )
    )
)
from dataloader_provider import train_valid_test_dataloaders_provider
from transformers import AutoProcessor
from examples.mimo.data.avlm_sample_loader import reject_clip_list
from examples.mimo.data.utils.calculate_audio_tokens import (
    capped_whisper_lengths,
    expand_audio_placeholders,
    stack_or_cat,
)

from megatron.energon import (
    DefaultTaskEncoder,
    VQASample,
    WorkerConfig,
    get_loader,
    get_train_dataset,
)
from megatron.energon.task_encoder.base import stateless
from megatron.training import get_args

IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"


@dataclass
class ConversationTemplateConfig:
    system: str = None
    chat_template: str = None


@dataclass
class LlavaConversationTemplateConfig(ConversationTemplateConfig):
    """Default system prompt and chat template for Llava training."""

    system: str = None
    chat_template: str = None


@dataclass
class VisionAudioQASample(VQASample):
    """
    Sample type for vision audio question answering.
    Adding audio to the VQASample class.
    """

    #: The input audio tensor in the shape
    audio: torch.Tensor = None


class AVLMModelType(Enum):
    IMAGE_AUDIO_LLAVA_AVLM = "image_audio_llava_avlm"


class AVLMTaskEncoder(
    DefaultTaskEncoder[
        VisionAudioQASample,
        dict,
        dict,
        dict,
    ]
):
    def __init__(
        self,
        model_type: AVLMModelType,
        processor,
        image_processor,
        audio_processor,
        conversation_template_config=None,
    ):
        self.model_type = model_type
        self.processor = processor
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.conversation_template_config = conversation_template_config

    def apply_prompt_template(self, input_text: VisionAudioQASample):
        """Create conversation prompt string using HF chat template.

        The first user turn always contains an image placeholder, later turns are text-only.
        Returns a *prompt string* that can be fed into the processor together with an image.
        """

        user_msgs = input_text.context
        bot_msgs = input_text.answers

        def _ensure_list_type(value):
            if isinstance(value, list):
                return value
            return [value]

        user_msgs = _ensure_list_type(user_msgs)
        bot_msgs = _ensure_list_type(bot_msgs)

        conversation = []
        for _, (u_txt, b_txt) in enumerate(zip(user_msgs, bot_msgs)):
            conversation.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": u_txt}],
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": b_txt}],
                }
            )

        # Inject optional system message
        if (
            self.conversation_template_config
            and self.conversation_template_config.system
        ):
            conversation.insert(
                0,
                {"role": "system", "content": self.conversation_template_config.system},
            )

        # Select chat template
        if (
            self.conversation_template_config
            and self.conversation_template_config.chat_template
        ):
            self.processor.chat_template = (
                self.conversation_template_config.chat_template
            )
        return self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _find_pattern_indices(
        self, template, pattern, start_idx=0, allow_first_mismatch=False
    ):
        template_len = len(template)
        pat_len = len(pattern)
        for i in range(start_idx, template_len - pat_len + 1):
            match = template[i : i + pat_len] == pattern
            if torch.all(match) or (allow_first_mismatch and torch.all(match[1:])):
                return i, i + pat_len
        return -1, -1

    @stateless
    def encode_sample(self, sample: VisionAudioQASample):
        """Return tokenised multimodal sample."""
        args = get_args()
        prompt = self.apply_prompt_template(sample)
        logging.debug(f"prompt: {prompt}")

        # Convert raw image bytes to tensor
        if sample.image is not None:
            image_io = io.BytesIO(sample.image)
            image = Image.open(image_io)
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # Convert to CxHxW format
            image_tensor = image_tensor.float() / 255.0  # rescale to [0,1] range
        else:
            image_tensor = None

        # Convert raw audio data to tensor. A list is every clip, in file order.
        fixed_sample_rate = 16000
        all_clips = bool(getattr(args, "all_audio_clips", False))
        reject_clip_list(sample.audio, all_clips)

        def _waveform_from_bytes(audio_bytes):
            audio_io = io.BytesIO(audio_bytes)
            waveform, sample_rate = sf.read(audio_io)
            if sample_rate != fixed_sample_rate:
                num_samples = int(len(waveform) * fixed_sample_rate / sample_rate)
                waveform = signal.resample(waveform, num_samples)
            return torch.from_numpy(waveform).float()

        if sample.audio is None:
            audio_tensor = None
        elif isinstance(sample.audio, (list, tuple)):
            audio_tensor = [_waveform_from_bytes(blob) for blob in sample.audio]
        else:
            audio_tensor = _waveform_from_bytes(sample.audio)

        # Process audio + prompt. feature_lengths is the capped mel, not 2 * tokens.
        if audio_tensor is not None:
            waveforms = audio_tensor if isinstance(audio_tensor, list) else [audio_tensor]
            features = []
            feature_lengths = []
            token_counts = []
            for waveform in waveforms:
                processed = self.audio_processor(waveform, sampling_rate=fixed_sample_rate)
                mel_features = torch.tensor(processed["input_features"]).squeeze(0)
                length = int(waveform.shape[-1])
                mel, num_tokens = capped_whisper_lengths(
                    length,
                    args.audio_encoder_model,
                    feature_time=mel_features.shape[-1],
                )
                features.append(mel_features)
                feature_lengths.append(mel)
                token_counts.append(num_tokens)
            # Flag on keeps a clip axis for K=1 so stack_or_cat can cat variable K.
            # Flag off stays rank-2 so a batch still stacks to [B, 80, T].
            if all_clips:
                processed_audios = torch.stack(features, dim=0)
                audios_seq_lengths = torch.tensor(token_counts)
                audios_feature_lengths = torch.tensor(feature_lengths)
            else:
                processed_audios = features[0]
                audios_seq_lengths = torch.tensor(token_counts[0])
                audios_feature_lengths = torch.tensor(feature_lengths[0])
            processed_prompt = expand_audio_placeholders(prompt, token_counts)
        else:
            processed_audios = None
            audios_seq_lengths = None
            audios_feature_lengths = None
            processed_prompt = prompt

        # Process image + prompt
        # Here, we:
        #  + process the image
        #  + use self.processor to automatically calculate the number  
        #    of image tokens, then add them to the prompt
        #    => this step combine adding the corresponding image tokens to the prompt AND
        #       tokenize the prompt after that
        if image_tensor is not None:
            processed_images = self.image_processor(
                images=image_tensor,
                return_tensors="pt",
                do_rescale=False,
            )["pixel_values"]
            processed_images = processed_images.squeeze(0)  # remove batch dim
        else:
            processed_images = None

        processed_prompt_inputs = self.processor(
            images=image_tensor,
            text=processed_prompt,
            add_special_tokens=False,
            return_tensors="pt",
            do_rescale=False,
        )
        
        # Remove batch dim
        for k, v in processed_prompt_inputs.items():
            processed_prompt_inputs[k] = v.squeeze(0)

        # Combine image and audio processed data
        inputs = {
            "input_ids": processed_prompt_inputs["input_ids"],
            "attention_mask": processed_prompt_inputs["attention_mask"],
        }

        if processed_images is not None:
            inputs["images"] = processed_images

        if processed_audios is not None:
            inputs["audios"] = processed_audios
            inputs["audios_seq_lengths"] = audios_seq_lengths
            inputs["audios_feature_lengths"] = audios_feature_lengths

        answers = sample.answers
        if answers:
            if not isinstance(answers, list):
                answers = [answers]
            tokenizer = self.processor.tokenizer
            inputs["labels"] = torch.full_like(inputs["input_ids"], fill_value=-100)
            search_idx = 0
            for ans in answers:
                answer_tokens = tokenizer.encode(
                    ans, add_special_tokens=False, return_tensors="pt"
                )[0]
                s_idx, e_idx = self._find_pattern_indices(
                    inputs["input_ids"], answer_tokens, search_idx
                )
                if s_idx == -1:
                    raise ValueError(f"Answer not found in input_ids: {ans}")
                inputs["labels"][s_idx:e_idx] = inputs["input_ids"][s_idx:e_idx]
                search_idx = e_idx

            # shift inputs and labels by 1
            inputs["input_ids"] = inputs["input_ids"][:-1]
            inputs["labels"] = inputs["labels"][1:]
            inputs["loss_mask"] = (inputs["labels"] != -100).long()

        else:
            inputs["labels"] = None
            inputs["loss_mask"] = None

        return inputs

    def batch(self, samples: List[Dict]) -> Dict:
        """Pad/stack individual samples into a single batch dict."""

        if not samples:
            return {}

        batched: Dict[str, torch.Tensor] = {}
        keys = samples[0].keys()

        for key in keys:
            values = [s[key] for s in samples if key in s and s[key] is not None]

            processor = KEY_PROCESSORS.get(key)
            if processor is not None:
                batched[key] = processor(values)
                continue

            # Fallback behaviours if no specific processor is registered.
            if isinstance(values[0], torch.Tensor):
                batched[key] = torch.stack(values, dim=0)
            else:
                batched[key] = values

        return batched

    def encode_batch_avlm_clip_whisper_llava(self, batch_data: Dict) -> Dict:
        input_ids = batch_data["input_ids"]
        labels = batch_data.get("labels")
        loss_mask = batch_data.get("loss_mask")

        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)

        images = batch_data.get("images")
        audios = batch_data.get("audios")
        audios_seq_lengths = batch_data.get("audios_seq_lengths")
        audios_feature_lengths = batch_data.get("audios_feature_lengths")

        output = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if images is not None:
            output["modality_inputs"] = {
                "images": {"clip_encoder": {"pixel_values": images}}
            }

        if audios is not None:
            if "modality_inputs" not in output:
                output["modality_inputs"] = {}
            output["modality_inputs"]["audios"] = {
                "whisper_encoder": {
                    "input_features": audios,
                    "seq_lengths": audios_seq_lengths,
                    "feature_lengths": audios_feature_lengths,
                }
            }

        return output

    def encode_batch(self, batch_data: Dict) -> dict:
        if self.model_type is AVLMModelType.IMAGE_AUDIO_LLAVA_AVLM:
            return self.encode_batch_avlm_clip_whisper_llava(batch_data)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")


def llava_avlm_dataloader_provider(train_val_test_num_samples):
    args = get_args()

    # update global tokenizer if hf_assign_unused_tokens is set
    if args.hf_assign_unused_tokens:
        _tokenizer = get_tokenizer()._tokenizer
        for token_id_pair in args.hf_assign_unused_tokens:
            token, id_str = token_id_pair.split(',')
            id = int(id_str)
            _tokenizer.add_special_tokens({'additional_special_tokens': [token]})
            _tokenizer.vocab[token] = id
            _tokenizer.added_tokens_encoder[token] = id
            _tokenizer.added_tokens_decoder[id] = token
        get_tokenizer()._tokenizer = _tokenizer

    tokenizer_model_id = args.tokenizer_model
    processor = AutoProcessor.from_pretrained(tokenizer_model_id)
    processor.tokenizer = get_tokenizer()._tokenizer  # update processor to use custom tokenizer
    image_processor = AutoProcessor.from_pretrained(tokenizer_model_id).image_processor
    audio_processor = AutoProcessor.from_pretrained(args.audio_encoder_model)

    return train_valid_test_dataloaders_provider(
        train_val_test_num_samples,
        task_encoder=AVLMTaskEncoder(
            model_type=AVLMModelType.IMAGE_AUDIO_LLAVA_AVLM,
            processor=processor,
            image_processor=image_processor,
            audio_processor=audio_processor,
            conversation_template_config=LlavaConversationTemplateConfig(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path to the dataset directory in energon format",
    )
    args = parser.parse_args()
    
    # for image audio llava avlm
    hf_model_id = "llava-hf/llava-1.5-7b-hf"
    model_type = AVLMModelType.IMAGE_AUDIO_LLAVA_AVLM

    processor = AutoProcessor.from_pretrained(hf_model_id)
    processor.tokenizer = get_tokenizer()._tokenizer
    image_processor = AutoProcessor.from_pretrained(hf_model_id).image_processor
    audio_processor = AutoProcessor.from_pretrained("openai/whisper-small")

    worker_config = WorkerConfig.default_worker_config(0)
    train_loader = get_loader(
        get_train_dataset(
            args.data_path,
            batch_size=8,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=AVLMTaskEncoder(
                model_type=model_type,
                processor=processor,
                image_processor=image_processor,
                audio_processor=audio_processor,
                conversation_template_config=LlavaConversationTemplateConfig(),
            ),
            worker_config=worker_config,
        ),
        worker_config=worker_config,
    )

    print(f"data loader length {len(train_loader)}")
    for index, each_batch in enumerate(train_loader):
        print(f"batch index {index} tokens {each_batch['input_ids']}")
        if 'modality_inputs' in each_batch:
            if 'images' in each_batch['modality_inputs']:
                print(f"images shape: {each_batch['modality_inputs']['images']['clip_encoder']['pixel_values'].shape}")
            if 'audios' in each_batch['modality_inputs']:
                print(f"audios shape: {each_batch['modality_inputs']['audios']['whisper_encoder']['input_features'].shape}")
        break


# -----------------------------------------------------------------------------
# Key processing utilities for batching
# -----------------------------------------------------------------------------


class KeyProcessor(Protocol):
    """Callable that aggregates a list of tensors into a single batched tensor."""

    def __call__(self, values: List[torch.Tensor]) -> torch.Tensor:  # pragma: no cover
        ...


class StackProcessor:
    """Simply stack tensors along a given dimension."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __call__(self, values: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values, dim=self.dim)


class PaddingProcessor:
    """Pad variable-length sequences to the same length."""

    def __init__(self, pad_value: int, batch_first: bool = True):
        self.pad_value = pad_value
        self.batch_first = batch_first

    def __call__(self, values: List[torch.Tensor]) -> torch.Tensor:
        return rnn_utils.pad_sequence(
            values, batch_first=self.batch_first, padding_value=self.pad_value
        )


# Registry mapping sample keys to their corresponding processor.
class StackOrCatProcessor:
    """Stack one clip per sample. Concatenate samples that already hold several clips."""

    def __call__(self, values: List[torch.Tensor]) -> torch.Tensor:
        return stack_or_cat(values)


KEY_PROCESSORS: Dict[str, KeyProcessor] = {
    "images": StackProcessor(),
    "audios": StackOrCatProcessor(),
    "audios_seq_lengths": StackOrCatProcessor(),
    "audios_feature_lengths": StackOrCatProcessor(),
    "input_ids": PaddingProcessor(pad_value=0),
    "attention_mask": PaddingProcessor(pad_value=0),
    "loss_mask": PaddingProcessor(pad_value=0),
    "labels": PaddingProcessor(pad_value=-100),
} 