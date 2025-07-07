# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx]["messages"]


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length

        conversation_list = self.dataset[int(self.indices[idx % len(self.indices)])]
        tokens, target = tokenizer.tokenize_conversation(
            conversation_list, return_target=True, add_generation_prompt=False
        )

        # minus one to insert eos token
        if len(tokens) > max_seq_len - 1:
            if True:  # TODO: when too long to fit in context, truncate left to right
                tokens = tokens[: max_seq_len - 1]
                target = target[: max_seq_len - 1]
            else:  # right to left
                tokens = tokens[-(max_seq_len - 1) :]
                target = target[-(max_seq_len - 1) :]

        # padding
        num_tokens = len(tokens) + 1
        padding_len = max_seq_len - num_tokens
        assert padding_len >= 0
        filler = [tokenizer.pad] * (padding_len + 1)

        tokens = np.array(tokens.tolist() + [tokenizer.eod] + filler, dtype=np.int64)
        target = np.array(target.tolist() + [tokenizer.eod] + filler, dtype=np.int64)

        tokens = torch.tensor(tokens)
        target = torch.tensor(target)

        tokens = tokens[:-1].contiguous()
        target = target[1:].contiguous()

        loss_mask, position_ids, attention_mask = self._get_ltor_masks_and_position_ids(
            max_seq_len, target, tokenizer.pad
        )

        if self.config.create_attention_mask:
            ret = {
                'tokens': tokens,
                'labels': target,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }
        else:
            ret = {
                'tokens': tokens,
                'labels': target,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }

        return ret

    def _get_ltor_masks_and_position_ids(self, max_seq_len, target, pad_token):
        """Build masks and position id for left to right model for SFT"""

        assert not self.config.reset_position_ids and not self.config.reset_attention_mask

        # Position ids.
        position_ids = torch.arange(max_seq_len, dtype=torch.long)

        # Loss mask.
        loss_mask = torch.ones(max_seq_len, dtype=torch.float)
        loss_mask[target == pad_token] = 0.0  # mask paddings
        loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts

        if self.config.create_attention_mask:
            attention_mask = torch.tril(
                torch.ones((seq_length, seq_length), device=data.device)
            ).unsqueeze(0)
            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return loss_mask, position_ids, attention_mask
