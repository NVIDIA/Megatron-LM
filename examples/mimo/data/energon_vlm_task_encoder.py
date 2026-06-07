# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union, Iterable, Tuple, Optional, Protocol
import heapq
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

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.energon import (
    DefaultTaskEncoder,
    VQASample,
    WorkerConfig,
    get_loader,
    get_train_dataset,
)
from megatron.energon.task_encoder.base import stateless
from megatron.training import get_args
from megatron.core.models.multimodal import context_parallel


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

def predict_seq_len_with_padding(instance_tokens: torch.Tensor, pad_to_multiple_of: int = 64) -> int:
    """Get seqlen with padding.
    Args:
        instance_tokens (torch.Tensor): Tensor of instance tokens.
        pad_to_multiple_of (int): Pad to multiple of this value.
    Returns:
        int: Padded sequence length.
    """
    seqlen = len(instance_tokens)
    seqlen_padded = (seqlen + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
    return seqlen_padded

def group_samples(samples: List[Dict[str, torch.Tensor]], 
                  group_size: int, 
                  lengths: List[int],
                  ) -> List[List[Dict[str, torch.Tensor]]]:
    """Group samples into groups of size group_size.

    Args:
        samples (List[Dict[str, torch.Tensor]]): List of samples to group.
        group_size (int): Maximum size of each group.
        lengths (List[int]): List of lengths of each sample.

    Returns:
        List[List[Dict[str, torch.Tensor]]]: List of groups, where each group is a list of samples
                that should be packed together. Each group's total length will not exceed group_size.
    """
    # create a max heap of the lengths
    max_heap: List[Tuple[int, int]] = [(-length, i) for i, length in enumerate(lengths)]
    heapq.heapify(max_heap)

    groups: List[List[Dict[str, torch.Tensor]]] = []
    current_group: List[Dict[str, torch.Tensor]] = []
    current_length: int = 0
    while max_heap:
        neg_length, i = heapq.heappop(max_heap)
        length = -neg_length
        if current_length + length <= group_size:
            current_group.append(samples[i])
            current_length += length
        else:
            groups.append(current_group)
            current_group = [samples[i]]
            current_length = length
    # If we're at the end of the samples, add the last group
    if current_group:
        groups.append(current_group)
    return groups

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
        conversation_template_config: Optional[ConversationTemplateConfig] = None,
        max_seq_length: Optional[int] = None,
    ):
        """Initialize VLMTaskEncoder.

        Args:
            model_name (str): Model name, currently only "llava_vlm" is supported.
            processor: HuggingFace processor for the model.
            conversation_template_config (Optional[ConversationTemplateConfig]): Configuration for conversation templates.
            max_seq_length (Optional[int]): Maximum sequence length for packing. Should be sum of max_text_length
                and image_seq_length. If None, defaults to 4096. This value is used as group_size for sequence packing.
        """
        self.model_type = model_type
        # Use max_seq_length if provided, otherwise default to 4096
        self.group_size = max_seq_length if max_seq_length is not None else 4096
        self.processor = processor
        self.conversation_template_config = conversation_template_config
        # Read parallelism settings directly from training args (these live in TransformerConfig).
        _args = get_args()
        self._cp_size = getattr(_args, 'context_parallel_size', 1)
        self._tp_size = getattr(_args, 'tensor_model_parallel_size', 1)
        self._sequence_parallel = getattr(_args, 'sequence_parallel', False)

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

    def select_samples_to_pack(self, samples: List[Dict[str, torch.Tensor]]) -> List[List[Dict[str, torch.Tensor]]]:
        """Selects which samples will be packed together.
        
        This function receives a list of samples (size according to the selected packing_buffer_size), 
        and partitions those samples into groups that shall be packed together.

        Args:
            samples (List[Dict[str, torch.Tensor]]): List of samples from the buffer, each containing
                tokenized data with keys like 'input_ids', 'labels', 'loss_mask', etc.

        Returns:
            List[List[Dict[str, torch.Tensor]]]: List of groups, where each group is a list of samples
                that should be packed together. Each group's total length will not exceed group_size.

        NOTE: Energon dataloader calls this method internally if packing is used.
        Please see https://nvidia.github.io/Megatron-Energon/advanced/packing.html
        """
        # Group samples into groups of size group_size
        lengths = [predict_seq_len_with_padding(sample["input_ids"]) for sample in samples]
        # lengths = [sample["input_ids"].size(0) for sample in samples]
        packed_samples = group_samples(samples, group_size=self.group_size, lengths=lengths)
        return packed_samples
    
    @stateless
    def pack_selected_samples(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Implements how a group of samples will be mapped to a single sample.
        
        Args:
            samples (List[Dict[str, torch.Tensor]]): List of samples to pack together.

        Returns:
            Dict[str, torch.Tensor]: Packed sample with keys like 'input_ids', 'labels', 'loss_mask', etc.
        """
        # Pad each sample to a multiple of 64, then concatenate
        padded_input_ids = []
        padded_labels = []
        padded_loss_masks = []
        padded_lens = []

        has_labels = "labels" in samples[0] and samples[0]["labels"] is not None
        has_loss_mask = "loss_mask" in samples[0] and samples[0]["loss_mask"] is not None

        for sample in samples:
            original_len = sample["input_ids"].size(0)
            padded_len = predict_seq_len_with_padding(sample["input_ids"])
            padded_lens.append(padded_len)
            pad_amount = padded_len - original_len

            padded_input_ids.append(torch.cat([
                sample["input_ids"],
                torch.zeros(pad_amount, dtype=sample["input_ids"].dtype)
            ]))
            
            if has_labels:
                padded_labels.append(torch.cat([
                    sample["labels"],
                    torch.full((pad_amount,), -100, dtype=sample["labels"].dtype)
                ]))

            if has_loss_mask:
                padded_loss_masks.append(torch.cat([
                    sample["loss_mask"],
                    torch.zeros(pad_amount, dtype=sample["loss_mask"].dtype)
                ]))

        # Concatenate sequences
        input_ids = torch.cat(padded_input_ids)
        labels = torch.cat(padded_labels) if has_labels else None
        loss_mask = torch.cat(padded_loss_masks) if has_loss_mask else None
        
        batched_images = torch.stack([s["pixel_values"] for s in samples], dim=0)   # (B , C , H , W)
        
        # Calculate padding if context parallel or sequence parallel is enabled
        pad_len = 0
        if self._cp_size > 1 or self._sequence_parallel:
            pad_len = context_parallel.get_padding(
                len(input_ids),
                self._cp_size,
                self._tp_size,
                self._sequence_parallel,
            )
        
        # Pad sequences
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
            if labels is not None:
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
            if loss_mask is not None:
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)])
        
        # Generate position_ids after padding
        position_ids = torch.arange(len(input_ids))
        
        # Calculate cu_seqlens using padded lengths
        lens = torch.tensor(padded_lens, dtype=torch.int32)
        cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(lens, dim=0)])
        
        # Calculate padded sequence lengths and cu_seqlens
        seqlens_padded = [l.item() + pad_len for l in lens]
        cu_seqlens_padded = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(torch.tensor(seqlens_padded, dtype=torch.int32), dim=0)])
        
        packing_kwargs = {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_kv": cu_seqlens,
            "cu_seqlens_q_padded": cu_seqlens_padded,
            "cu_seqlens_kv_padded": cu_seqlens_padded,
            "max_seqlen_q": torch.tensor(max(seqlens_padded), dtype=torch.int32),
            "max_seqlen_kv": torch.tensor(max(seqlens_padded), dtype=torch.int32),
        }
        
        packed_result = {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "pixel_values": batched_images,
            "position_ids": position_ids,
            "packing_kwargs": packing_kwargs,
        }
            
        return packed_result

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
        is_packed_sample = "packing_kwargs" in samples[0]
        for key in keys:
            values = [s[key] for s in samples if key in s and s[key] is not None]

            processor = KEY_PROCESSORS.get(key)
            if processor is not None:
               batched[key] = processor(values, max_len=self.group_size, is_packed_sample=is_packed_sample) 
               continue
            
            # Fallback behaviours if no specific processor is registered.
            if isinstance(values[0], torch.Tensor):
               batched[key] = torch.stack(values, dim=0)
            else:
               batched[key] = values
        
        # Add context parallel padding if enabled
        if self._cp_size > 1:
            seq_len = batched["input_ids"].size(1)
            pad_len = context_parallel.get_padding(
                seq_len,
                self._cp_size,
                self._tp_size,
                self._sequence_parallel,
            )
            if pad_len > 0:
                # Pad input_ids
                batched["input_ids"] = torch.cat([
                    batched["input_ids"],
                    torch.zeros(batched["input_ids"].size(0), pad_len, dtype=batched["input_ids"].dtype)
                ], dim=1)
                # Pad labels
                if "labels" in batched:
                    batched["labels"] = torch.cat([
                        batched["labels"],
                        torch.full((batched["labels"].size(0), pad_len), -100, dtype=batched["labels"].dtype)
                    ], dim=1)
                # Pad loss_mask
                if "loss_mask" in batched:
                    batched["loss_mask"] = torch.cat([
                        batched["loss_mask"],
                        torch.zeros(batched["loss_mask"].size(0), pad_len, dtype=batched["loss_mask"].dtype)
                    ], dim=1)
        
        return batched

    def encode_batch_vlm_clip_llava(self, batch_data: Dict) -> Dict:
        input_ids = batch_data["input_ids"]
        labels = batch_data.get("labels")
        loss_mask = batch_data.get("loss_mask")

        # Handle packed-sample case where input_ids is 1-D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # add batch dimension
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)
            if loss_mask is not None and loss_mask.dim() == 1:
                loss_mask = loss_mask.unsqueeze(0)

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

def llava_vlm_dataloader_provider(train_val_test_num_samples, max_seq_length: Optional[int] = None, is_video_input: bool = False):
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
            max_seq_length=max_seq_length,
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path to the dataset directory in energon format",
    )
    parser.add_argument('--total-seq-length', type=int, default=512, help='Maximum text length')
    parser.add_argument('--image-seq-length', type=int, default=197, help='Number of image tokens')
    parser.add_argument('--packing-buffer-size', type=int, default=None, help='Packing buffer size when using sequence packing')
    args = parser.parse_args()
    
    # Calculate max_seq_length as sum of text and image sequence lengths
    max_seq_length = args.max_text_length + args.image_seq_length
    
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
                max_seq_length=max_seq_length,  # Use calculated max_seq_length
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

    def __call__(self, values: List[torch.Tensor], max_len: Optional[int] = None, is_packed_sample: bool = False) -> torch.Tensor:  # pragma: no cover
        ...


class StackProcessor:
    """Simply stack tensors along a given dimension."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __call__(self, values: List[torch.Tensor], max_len: Optional[int] = None, is_packed_sample: bool = False) -> torch.Tensor:
        if values[0].dim() == 3:
            return torch.stack(values, dim=self.dim) # (B , C , H , W)
        else:
            # Concatenate already-batched image tensors along the batch dimension.
            return torch.cat(values, dim=self.dim)  # (B , C , H , W)


class PaddingProcessor:
    """Pad variable-length sequences to the same length."""

    def __init__(self, pad_value: int, batch_first: bool = True):
        self.pad_value = pad_value
        self.batch_first = batch_first
    
    def _pad_and_stack(self, tensors: List[torch.Tensor], max_len: int, pad_val: int) -> torch.Tensor:
        """Pad or truncate a list of 1D tensors to a fixed length and stack them."""
        padded_tensors = []
        for t in tensors:
            current_len = t.size(0)
            if current_len > max_len:
                # Truncate
                padded_tensors.append(t[:max_len])
            else:
                # Pad
                pad_amount = max_len - current_len
                padding = torch.full((pad_amount,), pad_val, dtype=t.dtype, device=t.device)
                padded_tensors.append(torch.cat([t, padding]))
        return torch.stack(padded_tensors, dim=0)

    def __call__(self, values: List[torch.Tensor], max_len: Optional[int] = None, is_packed_sample: bool = False) -> torch.Tensor:
        if is_packed_sample:
            return rnn_utils.pad_sequence(
                    values, batch_first=self.batch_first, padding_value=self.pad_value
                    )
        else:
            return self._pad_and_stack(values, max_len, self.pad_value)

class PackingKwargsProcessor:
    """Extract the value at first index for packing_kwargs"""

    def __call__(self, values: List[torch.Tensor], max_len: Optional[int] = None, is_packed_sample: bool = False) -> torch.Tensor:
        if len(values) == 1:
            return values[0]
        else:
            raise ValueError("Multiple packing_kwargs found in batch; expected only one per batch.")

class GenericStackProcessor:

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __call__(self, values: List[torch.Tensor], max_len: Optional[int] = None, is_packed_sample: bool = False) -> torch.Tensor:
        # Generic stacking for other tensor fields
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values, dim=self.dim)
        else:
            return values

# Registry mapping sample keys to their corresponding processor.
KEY_PROCESSORS: Dict[str, KeyProcessor] = {
    "pixel_values": StackProcessor(),
    "pixel_values_videos": StackProcessor(),
    "input_ids": PaddingProcessor(pad_value=0),
    "attention_mask": PaddingProcessor(pad_value=0),
    "loss_mask": PaddingProcessor(pad_value=0),
    "labels": PaddingProcessor(pad_value=-100),
    "packing_kwargs": PackingKwargsProcessor(),
}
