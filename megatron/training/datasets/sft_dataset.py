# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit, json
from collections import Counter
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
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
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
        self.packed_sequence_get_count = 0
        self.packed_sequence_truncated_count = 0
        self.packed_sequence_padded_count = 0
        self.conversation_truncated_count = 0
        self.conversation_length_hist = Counter()
        self.truncated_tokens_hist = Counter()
        self.padded_tokens_hist = Counter()
        # TODO(duncan): Add character to token ratio?
        # WARNING: The following code relies on DP=1 and num-workers=0. This will be removed
        # if index_split == Split.train:
        #     atexit.register(self._dump_stats)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            # Whenever we see a new system message, start a new conversation
            if msg["role"] == "system":
                if current:  # If previously accumulating a conversation, then store it
                    split_conversations.append(current)
                current = [msg]  # Then start the new conversation
            else:
                current.append(msg) # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        merged_conversations = self.dataset[int(self.indices[idx % len(self.indices)])]
        split_conversations = self._split_conversations(merged_conversations)

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1]+1, positions[-1]+1+pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        eod = tokenizer.eod
        pad = tokenizer.pad
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        conversation_truncated_count = 0
        truncated_tokens_count = 0
        for conversation in split_conversations:

            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()

            # Add EOD, unless it's already present
            if tokens_list[-1] != eod:
                tokens_list.append(eod)
                targets_list.append(eod)

            self.conversation_length_hist[len(tokens_list)] += 1

            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            if self.config.context_parallel_size > 1:
                pad_granularity = self.config.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                truncated_tokens_count += len(pack_tokens) - pack_length
                conversation_truncated_count += 1
                truncate_left_not_right = True  # TODO(duncan): plumb this switch in
                if truncate_left_not_right:  # Retain existing eod
                    max_body = pack_length
                    pack_tokens = pack_tokens[-max_body:]
                    pack_targets = pack_targets[-max_body:]
                    pack_tokens.append(pad)
                    pack_targets.append(pad)
                else:  # Truncate right (need to add eod)
                    max_body = pack_length - 1
                    pack_tokens = pack_tokens[:max_body]
                    pack_targets = pack_targets[:max_body]
                    pack_tokens.extend([eod, pad])
                    pack_targets.extend([eod, pad])
                pack_positions = pack_positions[:pack_length+1]
                # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        if conversation_truncated_count > 0:
            self.conversation_truncated_count += conversation_truncated_count
            self.packed_sequence_truncated_count += 1

        self.truncated_tokens_hist[truncated_tokens_count] += 1

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1
            self.packed_sequence_padded_count += 1
            self.padded_tokens_hist[pad_len - 1] += 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Align and convert to tensors
        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask.
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0  # Mask paddings
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        # attention_mask = None

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        # Calculating max_seqlen here, rather than incrementally above, because of possible
        # effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor

        self.packed_sequence_get_count += 1

        return {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }

    # TODO(duncan): remove this
    def get_stats(self):
        return {
            "packed_sequence_get_count": self.packed_sequence_get_count,
            "packed_sequence_truncated_count": self.packed_sequence_truncated_count,
            "packed_sequence_padded_count": self.packed_sequence_padded_count,
            "conversation_truncated_count": self.conversation_truncated_count,
            "conversation_length_hist": dict(self.conversation_length_hist),
            "truncated_tokens_hist": dict(self.truncated_tokens_hist),
            "padded_tokens_hist": dict(self.padded_tokens_hist),
        }

    # TODO(duncan): remove this
    def _dump_stats(self):
        stats_path = "/lustre/fsw/portfolios/llmservice/users/duncan/mamba/packed-sequence/megatron-hybrid-fix/stats/stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.get_stats(), f, indent=2, sort_keys=True)
        print(f"\n[SFTDataset] wrote stats -> {stats_path}")
