# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit, json
from collections import Counter
import json
import math
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
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
        # Pre-calculate padding divisor to avoid redundant computation in get_padding_size
        self.padding_divisor = self._calculate_padding_divisor()

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

    def _calculate_padding_divisor(self) -> int:
        """
            Calculate the divisor used for sequence padding.
            tp_pad = tp_size * 2 if tp_size > 1 else 1
            cp_pad = cp_size * 2 if cp_size > 1 else 1
            cp_pad = cp_pad * dp_size if hybrid_cp else cp_pad
            divisor = cp_pad * tp_pad
        """
        if self.config.hybrid_context_parallel:
            # Hybrid CP: consider both CP and DP
            cp_pad = self.config.data_parallel_size * self.config.context_parallel_size * 2
        else:
            # Standard CP: only consider CP
            cp_pad = self.config.context_parallel_size * 2 if self.config.context_parallel_size > 1 else 1
        tp_pad = self.config.sequence_parallel_size if self.config.sequence_parallel_size > 0 else 1
        divisor = cp_pad * tp_pad
        # TODO(tailaim): do we need to pad for FP8 execution?
        # divisor = ((divisor + 15) // 16) * 16
        return divisor
    
    def get_padding_size(
        self,
        seq_len: int,
    ) -> int:
        seq_len_padded = math.ceil(seq_len / self.padding_divisor) * self.padding_divisor
        assert seq_len > seq_len_padded / 2 / self.config.context_parallel_size * (self.config.context_parallel_size - 1), \
        f"sequence length {seq_len} is too short, the divisor is {self.padding_divisor}, that means cp_rank \
        {self.config.context_parallel_size-1} will have no valid tokens"
        return seq_len_padded

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sequence_packing = self.config.sequence_packing
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

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1

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

        return {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }

class MockSFTLowLevelDataset:
    """The low-level mock dataset for SFT

    Args:
        mock_config (dict): The config for mock dataset.
    """

    seed: int = 0
    """The hard-coded random seed to use to set the NumPy RNG"""

    size: int = 1000000
    """The hard-coded number of sequence to generate"""
    
    # This is to maintain consistency with the SFT dataset that uses real data. In the real dataset, an element in the low-level dataset often contains multiple sequences. So here, each element in the mock low-level dataset also contains num_sequence_per_sample sequences. This will be made more reasonable in the future.
    

    def __init__(self, config: Dict) -> None:
        np.random.seed(self.seed)
        # either choose to load sequence lengths from external file, or generate random sequence lengths
        
        assert "mode" in config, f"mode must be set, either 'file' or 'distribution'"
        
        if config["mode"] == "file":
            self.sequence_lengths = np.array(pd.read_csv(config["path"])).flatten()
            self.size = len(self.sequence_lengths)
        elif config["mode"] == "distribution":
            min_seq_len = config["min_seq_len"]
            max_seq_len = config["max_seq_len"]
            mean_seq_len = config["mean_seq_len"]
            if config["type"] == "lognormal":
                lognormal_sigma = config["lognormal_sigma"]
                self.sequence_lengths = self.generate_lognormal_samples(self.size, mean_seq_len,lognormal_sigma, min_seq_len, max_seq_len)
            else:
                raise ValueError(f"Unsupported sequence length distribution type {config['type']}")
        
    def generate_lognormal_samples(self, size, mean, sigma, min_seq_len, max_seq_len):   
        mu = np.log(mean) - sigma**2 / 2
        samples = np.random.lognormal(mu, sigma, size)
        samples = np.clip(samples, min_seq_len, max_seq_len)
        return samples.astype(int)   

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> List[np.ndarray]:
        length = self.sequence_lengths[idx % self.size]
        # the length of sample is 'length', but only length-1 elements are generated here, 
        # because an eod token will be appended at the end later in SFTDataset
        sample = np.arange(1, length, dtype=np.int64)
        return sample
class MockSFTDataset(SFTDataset):
    """The mock dataset used during SFT"""

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
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        mock_config = json.loads(config.sft_mock_dataset_config_json)
        return MockSFTLowLevelDataset(mock_config)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sequence_packing = self.config.sequence_packing
        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length

        tokens = self.dataset[int(self.indices[idx % len(self.indices)])]
        target = np.array(tokens, dtype=np.int64)
        
        force_eod_length = int(tokenizer.force_eod)

        if len(tokens) > max_seq_len - force_eod_length:
            # cut the right side
            tokens = tokens[: max_seq_len - force_eod_length]
            target = target[: max_seq_len - force_eod_length]
            # tokens = tokens[(-max_seq_len + force_eod_length):]
            # target = target[(-max_seq_len + force_eod_length):]

        # padding
        num_tokens = len(tokens) + force_eod_length
        if sequence_packing:
            padding_len = self.get_padding_size(num_tokens) - num_tokens
        else:
            padding_len = max_seq_len - num_tokens
        assert padding_len >= 0
        filler = [tokenizer.eod] * force_eod_length + [tokenizer.pad] * (padding_len + 1)

        tokens = np.array(tokens.tolist() + filler, dtype=np.int64)
        target = np.array(target.tolist() + filler, dtype=np.int64)

        tokens = torch.tensor(tokens)
        target = torch.tensor(target)

        tokens = tokens[:-1].contiguous()
        target = target[1:].contiguous()
        seq_len = tokens.numel()

        loss_mask, position_ids, attention_mask = self._get_ltor_masks_and_position_ids(
            seq_len, target, tokenizer.pad
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

        if sequence_packing:
            # sequence packing need both original sequence length and padded length
            ret['original_seq_len'] = torch.tensor(num_tokens, dtype=torch.int32, device=tokens.device)
            ret['padded_seq_len'] = torch.tensor(seq_len, dtype=torch.int32, device=tokens.device)

        return ret
