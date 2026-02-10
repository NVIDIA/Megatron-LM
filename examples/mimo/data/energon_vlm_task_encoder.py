import argparse
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Protocol, Union

import torch
import torch.nn.utils.rnn as rnn_utils

# TODO: ykarnati, use absolute import or 
# define train_valid_test_dataloaders_provider in here
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


@dataclass
class ConversationTemplateConfig:
    system: str = None
    chat_template: str = None



@dataclass
class LlavaConversationTemplateConfig(ConversationTemplateConfig):
    """Default system prompt and chat template for Llava training."""

    system: str = None
    chat_template: str = None


class ModelType(Enum):
    LLAVA_VLM = "llava_vlm"
    VIDEO_LLAVA_VLM = "video_llava_vlm"

class VLMTaskEncoder(
    DefaultTaskEncoder[
        Union[VQASample],
        dict,
        dict,
        dict,
    ]
):
    def __init__(
        self,
        model_type: ModelType,
        processor,
        conversation_template_config=None,
    ):
        self.model_type = model_type

        self.processor = processor
        self.conversation_template_config = conversation_template_config

    def apply_prompt_template(self, input_text: VQASample):
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
    def encode_sample(self, sample: VQASample):
        """Return tokenised multimodal sample."""
        # Build prompt
        prompt = self.apply_prompt_template(sample)
        logging.debug(f"prompt: {prompt}")

        # Process image + prompt
        inputs = self.processor(
            images=getattr(sample, "image", None),
            text=prompt,
            add_special_tokens=False,
            return_tensors="pt",
            do_rescale=False,
        )
        # HF processor returns a dict with batch dim
        # Remove batch dim
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

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

    def encode_batch_vlm_clip_llava(self, batch_data: Dict) -> Dict:
        input_ids = batch_data["input_ids"]
        labels = batch_data.get("labels")
        loss_mask = batch_data.get("loss_mask")

        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)

        pixel_values = batch_data.get("pixel_values")

        output = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if pixel_values is not None:
            output["modality_inputs"] = {
                "images": {"clip_encoder": {"pixel_values": pixel_values}}
            }

        return output

    def encode_batch_vlm_clip_llava_video(self, batch_data: Dict) -> Dict:
        input_ids = batch_data["input_ids"]
        labels = batch_data.get("labels")
        loss_mask = batch_data.get("loss_mask")

        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).repeat(input_ids.size(0), 1)

        pixel_values_videos = batch_data.get("pixel_values_videos")

        output = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if pixel_values_videos is not None:
            output["modality_inputs"] = {
                "images": {"clip_encoder": {"pixel_values": pixel_values_videos}}
            }

        return output

    def encode_batch(self, batch_data: Dict) -> dict:
        if self.model_type is ModelType.LLAVA_VLM:
            return self.encode_batch_vlm_clip_llava(batch_data)
        elif self.model_type is ModelType.VIDEO_LLAVA_VLM:
            return self.encode_batch_vlm_clip_llava_video(batch_data)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")


def llava_vlm_dataloader_provider(train_val_test_num_samples, is_video_input=False):
    args = get_args()
    tokenizer_model_id = args.tokenizer_model
    processor = AutoProcessor.from_pretrained(tokenizer_model_id)
    if is_video_input:
        model_type = ModelType.VIDEO_LLAVA_VLM
    else:
        model_type = ModelType.LLAVA_VLM
    return train_valid_test_dataloaders_provider(
        train_val_test_num_samples,
        task_encoder=VLMTaskEncoder(
            model_type=model_type,
            processor=processor,
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
    model_name = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_name)
    worker_config = WorkerConfig.default_worker_config(0)
    train_loader = get_loader(
        get_train_dataset(
            args.data_path,
            batch_size=8,
            shuffle_buffer_size=None,
            max_samples_per_sequence=None,
            task_encoder=VLMTaskEncoder(
                model_type=ModelType.LLAVA_VLM,
                processor=processor,
                conversation_template_config=LlavaConversationTemplateConfig(),
            ),
            worker_config=worker_config,
        ),
        worker_config=worker_config,
    )

    print(f"data loader length {len(train_loader)}")
    for index, each_batch in enumerate(train_loader):
        print(
            f"batch index {index} tokens {each_batch['input_ids']} images shape \
               {each_batch['modality_inputs']['images']['clip_encoder']['pixel_values'].shape}"
        )
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
    "pixel_values": StackProcessor(),
    "pixel_values_videos": StackProcessor(),
    "input_ids": PaddingProcessor(pad_value=0),
    "attention_mask": PaddingProcessor(pad_value=0),
    "loss_mask": PaddingProcessor(pad_value=0),
    "labels": PaddingProcessor(pad_value=-100),
}
