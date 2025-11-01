# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Supervised Finetuning GPT."""
import itertools
import os
import sys
from functools import partial
from typing import Any, Dict, Optional

import json
import jsonlines

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import datasets
import torch
import transformers

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.model_provider import model_provider
from megatron.post_training.loss_func import loss_func
from megatron.post_training.non_loss_data_func import report_draft_acceptance_length
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    print_rank_0,
    unwrap_model,
)

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def add_finetune_args(parser):
    """Add additional arguments for finetune."""
    group = parser.add_argument_group(title='Finetune')
    group.add_argument("--offline-distillation-data", type=str, help="Path to the offline dataset directory with base model features.")


    add_modelopt_args(parser)
    return parser

def get_eos_id():
    """Return the eos token id.
    
    We insert eos_token between two samples during packing. However, if the eos_token is used in message or after turns,
    we need to replace it with some other special tokens that do not appear in message."""
    tokenizer = get_tokenizer()
    hf_tokenizer = tokenizer._tokenizer

    if hf_tokenizer.eos_token == "<|eot_id|>":
        return 128001
    if hf_tokenizer.eos_token == "<|eot|>":
        return 200001
    if hf_tokenizer.eos_token == "<|im_end|>":
        return 151643
    if hf_tokenizer.eos_token == "<|return|>":
        return 199999

    return hf_tokenizer.eos_token_id


class OfflineDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, num_samples):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.file_paths = []

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isfile(item_path):
                self.file_paths.append(item_path)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        idx = idx % len(self.file_paths)
        file_path = self.file_paths[idx]
        sample = torch.load(file_path)
        return sample

class SFTDataset(torch.utils.data.Dataset):

    hf_dataset_to_kwargs = {
        "Open-Orca/OpenOrca": {"split": "train"},
        "Open-Orca/SlimOrca": {"split": "train"},
        "nvidia/Daring-Anteater": {"split": "train"},
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": {"split": "train"},
        "HuggingFaceH4/ultrachat_200k": {"split": "train_sft"},
    }

    hf_dataset_to_conversation = {
        "Open-Orca/OpenOrca": lambda data: SFTDataset._to_conversation(
            data["question"], data["response"]
        ),
        "Open-Orca/SlimOrca": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "nvidia/Daring-Anteater": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": lambda data: SFTDataset._sharegpt_to_openai_conversations(
            data
        ),
    }

    hf_dataset_to_prompt_template = {
        "Open-Orca/OpenOrca": "{{ messages['question'] + ' ' + messages['response'] + ' ' }}",
    }

    def __init__(
        self,
        num_packed_samples: int,
        data_path: Optional[str],
        tokenizer: transformers.PreTrainedTokenizerBase,
        seq_length: int,
        hf_dataset: Optional[str] = None,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        """A simple dataset implementation for supervised fine-tuning.

        The raw data is processed and packed to an indexed dataset on the fly. Users
        specify the total number of packed samples and the dataloader (or sampler)
        access the packed dataset by indices. When the packed dataset length is smaller
        than the index, the packing process fetches the raw data in a cyclic fashion
        until the packed dataset has sufficient length.

        Args:
            data_path: Path to the json or jsonl file
            num_packed_samples: total number of packed samples (cyclic access)
            tokenizer: hf tokenizer
            seq_length: max sequence length
            hf_dataset: not supported yet
        """
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

        self.num_packed_samples = num_packed_samples
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.hf_dataset = hf_dataset
        self.data_transformation = lambda data: data
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.indexed_dataset = []
        self._raw_sample_index = 0

        # [WAR]: For DeepSeek-V3/R1 tokenizer, we modify the chat_template such that the <think>
        # tokens are preserved for supervised learning.
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

        if data_path is not None:
            if data_path.endswith(".json"):
                self._raw_samples = json.load(open(data_path))
            elif data_path.endswith(".jsonl"):
                with jsonlines.open(data_path, mode='r') as reader:
                    self._raw_samples = [obj for obj in reader]
            else:
                raise ValueError("data_path must be json or jsonl")
        elif self.hf_dataset is not None:
            hf_dataset_kwargs = SFTDataset.hf_dataset_to_kwargs.get(
                self.hf_dataset, {"split": "train"}
            )
            self._raw_samples = datasets.load_dataset(self.hf_dataset, token=os.environ.get("HF_TOKEN", None), **hf_dataset_kwargs)
            self._raw_samples = self._raw_samples.shard(
                num_shards=self.num_shards, index=shard_index
            )

            print(
                "Rank {:3}/{:3} creates SFT data shard {:3}/{:3} with {:10} raw samples".format(
                    torch.distributed.get_rank(),
                    torch.distributed.get_world_size(),
                    self.shard_index,
                    self.num_shards,
                    len(self._raw_samples),
                ),
                flush=True,
            )

        else:
            raise ValueError("Either hf_dataset or data_path must be provided!")

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SFTDataset.hf_dataset_to_prompt_template
        elif self.hf_dataset is not None:
            self.data_transformation = SFTDataset.hf_dataset_to_conversation.get(
                self.hf_dataset, lambda data: data
            )

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

    def __len__(self):
        return self.num_packed_samples

    def __getitem__(self, idx):
        """Get the idx packed data.

        The packed data index is different from the raw data index where a packed sample
        of sequence-length may require concatenting multiple raw data. When all raw data
        are used up, the last packed data is throw away, and we have a packed dataset
        in memory. The packed data index may exceed the length of the packed dataset
        which will just wrap in a cyclic fashion.
        """
        idx = idx // self.num_shards

        while idx >= len(self.indexed_dataset):
            packed_samples = self._process_and_pack_example()
            if packed_samples is None:
                break
            else:
                self.indexed_dataset.append(packed_samples)
            if len(self.indexed_dataset) % 10000 == 0:
                print(
                    "Rank {:3}/{:3} requests {:10}/{:10} packed SFT sample".format(
                        torch.distributed.get_rank(),
                        torch.distributed.get_world_size(),
                        idx,
                        len(self.indexed_dataset),
                    ),
                    flush=True,
                )

        idx = idx % len(self.indexed_dataset)
        torch_sample = {}
        for key, val in self.indexed_dataset[idx].items():
            torch_sample[key] = torch.LongTensor(val)
        return torch_sample

    def _process_and_pack_example(self):
        """Process multiple raw data and pack them into fixed sequence length."""
        required_packed_tokens = self.seq_length + 1
        current_packed_samples = []
        current_packed_samples_token_count = 0

        while current_packed_samples_token_count < required_packed_tokens:
            if self._raw_sample_index >= len(self._raw_samples):
                return None
            raw_sample = self._raw_samples[self._raw_sample_index]
            self._raw_sample_index += 1
            processed_sample = self._process_example(raw_sample)
            if processed_sample is not None:
                current_packed_samples.append(processed_sample)
                current_packed_samples_token_count += processed_sample["token_count"]

        packed_samples = {}

        for key in ['input_ids', 'loss_mask']:
            packed_samples[key] = list(
                itertools.chain.from_iterable([obj[key] for obj in current_packed_samples])
            )

        for key in ['token_count']:
            packed_samples[key] = [obj[key] for obj in current_packed_samples]

        return packed_samples

    def _process_example(self, example: Dict[str, Any]):
        """Apply the chat template and compute the answer-only loss mask."""
        if not isinstance(example, Dict):
            raise ValueError("The sample must be a Dict but got {}".format(type(example)))

        # Several things can happen here after the transformation is applied:
        #
        # 1. If the transformation is identity transformation, then either the chat data
        #    is already in OpenAI chat format or there is a custom prompt template used.
        # 2. Otherwise, the tokenizer must have a default chat template and we are either
        #    converting the ShareGPT chat data or standard SFT data to OpenAI chat data.
        example = self.data_transformation(example)

        # Check if this is OpenAI chat data?
        conversations = example.get("conversations", None)
        if conversations is None:
            conversations = example.get("messages", None)

        # We don't use the data if there is no assistant reply or the conversation that
        # starts with the assistant.
        if conversations is not None:
            example = conversations
            if len(conversations) < 2 or example[0]["role"] == "assistant":
                return None

        # We always add eos between samples for training purpose.
        input_ids = self.tokenizer.apply_chat_template(example)
        current_loss_mask = [1] * len(input_ids)
        input_ids = input_ids + [get_eos_id()]
        current_loss_mask += [0]

        assert len(input_ids) == len(current_loss_mask)

        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            current_loss_mask = current_loss_mask[: self.seq_length]

        processed_example = {
            'input_ids': input_ids,
            'loss_mask': current_loss_mask,
            'token_count': len(input_ids),
        }
        return processed_example

    @classmethod
    def _to_conversation(cls, question, response):
        msg_question = {"role": "user", "content": question}
        msg_response = {"role": "assistant", "content": response}
        return {"conversations": [msg_question, msg_response]}

    @classmethod
    def _sharegpt_to_openai_conversations(cls, data):
        role_mapping = {
            "user": "user",
            "User": "user",
            "human": "user",
            "assistant": "assistant",
            "Assistant": "assistant",
            "gpt": "assistant",
            "system": "system",
            "System": "system",
        }
        processed_data = {"conversations": []}
        for msg in data["conversations"]:
            role = role_mapping[msg["from"]]
            content = msg["value"]
            processed_data["conversations"].append({"role": role, "content": content})
        return processed_data

    @classmethod
    def _special_to_openai_conversations(cls, data):
        processed_data = {"conversations": data["input"]["messages"]}
        return processed_data


def train_valid_test_sft_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples
            in train test and validation.
    """
    print_rank_0("> building train, validation, and test SFT datasets ...")
    args = get_args()
    tokenizer = get_tokenizer()

    if not isinstance(tokenizer._tokenizer, transformers.PreTrainedTokenizerBase):
        raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

    if args.micro_batch_size > 1:
        raise ValueError("SFTDataloader only supports micro_batch_size=1.")

    if args.export_offline_model:
        train_ds = OfflineDataset(os.path.join(args.offline_distillation_data, "train"), train_val_test_num_samples[0])
        valid_ds = OfflineDataset(os.path.join(args.offline_distillation_data, "valid"), train_val_test_num_samples[1])
        test_ds = OfflineDataset(os.path.join(args.offline_distillation_data, "test"), train_val_test_num_samples[2])

        print_rank_0("> finished creating offline SFT datasets ...")
    else:
        kwargs = {
            "tokenizer": tokenizer._tokenizer,
            "seq_length": args.seq_length,
            # Optional kwargs
            "hf_dataset": args.finetune_hf_dataset,
            "num_shards": mpu.get_expert_data_parallel_world_size(),
            "shard_index": mpu.get_expert_data_parallel_rank(),
        }

        data_path = [
            args.train_data_path[0] if args.train_data_path else None,
            args.valid_data_path[0] if args.valid_data_path else None,
            args.test_data_path[0] if args.test_data_path else None,
        ]

        train_ds = SFTDataset(train_val_test_num_samples[0], data_path[0], **kwargs)
        valid_ds = SFTDataset(train_val_test_num_samples[1], data_path[1], **kwargs)
        test_ds = SFTDataset(train_val_test_num_samples[2], data_path[2], **kwargs)

        print_rank_0("> finished creating SFT datasets ...")

    return train_ds, valid_ds, test_ds


def get_batch(data_iterator):
    """Generate a batch.
    
    For OfflineDataset, the aux_hidden_states and final hidden_states from the
    base model are loaded for offline speculative model training."""
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    # Broadcast data since only TP rank-0 has the data_iterator.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    if not args.export_offline_model:
        keys = ["input_ids", "loss_mask"]
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    else:
        keys = ["input_ids"]
        datatype = torch.int64
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
        data_b["loss_mask"] = torch.ones_like(data_b["input_ids"])
        data_b["loss_mask"][data_b["loss_mask"]==get_eos_id()] = 0
        data_b["loss_mask"] = torch.cat([data_b["loss_mask"], torch.zeros(1,1).to(torch.cuda.current_device())], dim=-1)

        keys = ["aux_hidden_states", "hidden_states"]
        datatype = torch.bfloat16
        feature_b = tensor_parallel.broadcast_data(keys, data, datatype)


    # Unpack the data received.
    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, 0 : 0 + args.seq_length].contiguous()
    labels = tokens_[:, 1 : 1 + args.seq_length].contiguous()
    answer_only_loss_mask = data_b["loss_mask"][:, 1 : 1 + args.seq_length].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, get_eos_id(), get_eos_id(), args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss, False
    )
    loss_mask = loss_mask * answer_only_loss_mask.to(dtype=loss_mask.dtype)


    labels = labels.contiguous()
    loss_mask = loss_mask.contiguous()

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    if args.export_offline_model:
        batch["aux_hidden_states"] = feature_b["aux_hidden_states"].transpose(0, 1)[:args.seq_length]
        batch["hidden_states"] = feature_b["hidden_states"].transpose(0, 1)[:args.seq_length]

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch


def non_loss_data_func(model: GPTModel):
    """Callback to compute the acceptance length."""
    args = get_args()
    if not args.export_offline_model:
        report_draft_acceptance_length(model)



def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator: Input data iterator
        model: The GPT Model
    """
    timers = get_timers()

    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)
    tokens = batch["tokens"]
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]
    if args.export_offline_model:
        aux_hidden_states = batch["aux_hidden_states"]
        hidden_states = batch["hidden_states"]
    timers("batch-generator").stop()

    if args.export_offline_model:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, aux_hidden_states=aux_hidden_states, hidden_states=hidden_states,)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask, model=model)


if __name__ == "__main__":
    pretrain(
        train_valid_test_sft_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_finetune_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        non_loss_data_func=non_loss_data_func,
    )
