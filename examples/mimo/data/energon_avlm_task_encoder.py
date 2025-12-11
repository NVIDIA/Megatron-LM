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
from examples.mimo.data.utils.calculate_audio_tokens import calculate_num_audio_tokens

from megatron.energon import (
    DefaultTaskEncoder,
    VQASample,
    WorkerConfig,
    get_loader,
    get_train_dataset,
)
from megatron.energon.task_encoder.base import stateless
from megatron.training import get_args
from megatron.training.tokenizer.multimodal_tokenizer import mistral_custom_template

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

        # Convert raw audio data to tensor
        if sample.audio is not None:
            audio_io = io.BytesIO(sample.audio)
            # Read audio first
            waveform, sample_rate = sf.read(audio_io)
            # Resample if needed
            fixed_sample_rate = 16000
            if sample_rate != fixed_sample_rate:
                # Calculate number of samples for 16kHz
                num_samples = int(len(waveform) * fixed_sample_rate / sample_rate)
                # Resample using scipy's resample
                waveform = signal.resample(waveform, num_samples)
            # Convert to tensor
            audio_tensor = torch.from_numpy(waveform).float()
        else:
            audio_tensor = None

        # Process audio + prompt
        # Here, we:
        #  + process the audio
        #  + manually calculate the number of audio tokens, then add them to the prompt
        if audio_tensor is not None:
            processed_audios = self.audio_processor(audio_tensor, sampling_rate=fixed_sample_rate)
            processed_audios = torch.tensor(processed_audios["input_features"])
            processed_audios = processed_audios.squeeze(0)  # remove batch dim
            num_audio_tokens = calculate_num_audio_tokens(audio_tensor.unsqueeze(0), args.audio_encoder_model)
            audios_seq_lengths = torch.tensor(num_audio_tokens)
            processed_prompt = prompt.replace(AUDIO_TOKEN, AUDIO_TOKEN * num_audio_tokens)
        else:
            processed_audios = None
            audios_seq_lengths = None
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
                    "seq_lengths": audios_seq_lengths
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
KEY_PROCESSORS: Dict[str, KeyProcessor] = {
    "images": StackProcessor(),
    "audios": StackProcessor(),
    "audios_seq_lengths": StackProcessor(),
    "input_ids": PaddingProcessor(pad_value=0),
    "attention_mask": PaddingProcessor(pad_value=0),
    "loss_mask": PaddingProcessor(pad_value=0),
    "labels": PaddingProcessor(pad_value=-100),
} 