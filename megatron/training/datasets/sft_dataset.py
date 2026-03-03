# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit, json
from collections import Counter
import json
import math
from typing import Any, Dict, Optional, List, Union

import numpy as np
import pandas as pd
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
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
        for conversation in split_conversations:

            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()


            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            pad_granularity = self._calculate_padding_divisor()
            mod_token_count = len(pack_tokens) % pad_granularity
            if mod_token_count != 0:
                pad_len = pad_granularity - mod_token_count
                extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                # Truncate on the right
                max_body = pack_length
                pack_tokens = pack_tokens[:max_body]
                pack_targets = pack_targets[:max_body]
                pack_tokens.append(pad)
                pack_targets.append(pad)
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
        mode (str): One of 'file', 'distribution', or 'verification'.
        **kwargs: Additional arguments depending on mode.
            For mode='file': path (str) - path to a CSV file with sequence lengths.
            For mode='distribution': type (str), min_seq_len (int), max_seq_len (int),
                mean_seq_len (int), and distribution-specific params (e.g. lognormal_sigma).
            For mode='verification': data_path (str) - prefix path to an IndexedDataset
                (.bin/.idx files). Optional lognormal distribution params same as
                'distribution' mode (defaults: min_seq_len=100, max_seq_len=4096,
                mean_seq_len=2048, lognormal_sigma=1.1).
        format (str): Output format for MockSFTDataset. Either 'thd' (default, sequence
            packing with cu_seqlens) or 'sbhd' (padded to seq_length, no cu_seqlens).
    """

    seed: int = 0
    """The hard-coded random seed to use to set the NumPy RNG"""

    size: int = 1000000
    """The hard-coded number of sequence to generate"""

    def __init__(self, mode: str, **kwargs) -> None:
        np.random.seed(self.seed)
        self.format = kwargs.get("format", "thd")

        if mode == "file":
            self.sequence_lengths = np.array(pd.read_csv(kwargs["path"])).flatten()
            self.size = len(self.sequence_lengths)
        elif mode == "distribution":
            min_seq_len = kwargs["min_seq_len"]
            max_seq_len = kwargs["max_seq_len"]
            mean_seq_len = kwargs["mean_seq_len"]
            if kwargs["type"] == "lognormal":
                lognormal_sigma = kwargs["lognormal_sigma"]
                self.sequence_lengths = self.generate_lognormal_samples(
                    self.size, mean_seq_len, lognormal_sigma, min_seq_len, max_seq_len
                )
            else:
                raise ValueError(f"Unsupported distribution type {kwargs['type']}")
        elif mode == "verification":
            # Load real tokens from an IndexedDataset for realistic loss curves.
            # Sequence lengths are drawn from a lognormal distribution (same as
            # "distribution" mode) to allow controlled comparison of THD vs SBHD.
            self.indexed_dataset = IndexedDataset(kwargs["data_path"])
            min_seq_len = kwargs.get("min_seq_len", 100)
            max_seq_len = kwargs.get("max_seq_len", 4096)
            mean_seq_len = kwargs.get("mean_seq_len", 2048)
            lognormal_sigma = kwargs.get("lognormal_sigma", 1.1)
            self.sequence_lengths = self.generate_lognormal_samples(
                self.size, mean_seq_len, lognormal_sigma, min_seq_len, max_seq_len
            )
        else:
            raise ValueError(f"Unsupported mode '{mode}', must be 'file', 'distribution', or 'verification'")
        
    def generate_lognormal_samples(self, size, mean, sigma, min_seq_len, max_seq_len):   
        mu = np.log(mean) - sigma**2 / 2
        samples = np.random.lognormal(mu, sigma, size)
        samples = np.clip(samples, min_seq_len, max_seq_len)
        return samples.astype(int)   

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> np.ndarray:
        # The returned sample has 'length-1' tokens; an EOD token is appended
        # later in MockSFTDataset.__getitem__, making the total 'length' tokens.
        length = int(self.sequence_lengths[idx % self.size])
        if hasattr(self, 'indexed_dataset'):
            target = length - 1
            num_docs = len(self.indexed_dataset)
            doc_idx = idx % num_docs
            raw = self.indexed_dataset[doc_idx]
            if len(raw) >= target:
                sample = raw[:target]
            else:
                # Concatenate documents until we reach the target length.
                chunks = [raw]
                total = len(raw)
                next_doc = doc_idx + 1
                while total < target:
                    raw_next = self.indexed_dataset[next_doc % num_docs]
                    need = target - total
                    chunks.append(raw_next[:need])
                    total += min(len(raw_next), need)
                    next_doc += 1
                sample = np.concatenate(chunks)[:target]
            assert len(sample) == target
            return sample.astype(np.int64)
        else:
            return np.arange(1, length, dtype=np.int64)


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
        if config.sft_mock_dataset_config_json is None:
            mock_config = {
                    "mode": "distribution",
                    "type": "lognormal",
                    "min_seq_len": config.sequence_length // 2,
                    "max_seq_len": config.sequence_length,
                    "mean_seq_len": config.sequence_length // 4 * 3,
                    "lognormal_sigma": 1.1,
                }
        else:
            mock_config = json.loads(config.sft_mock_dataset_config_json)
        return MockSFTLowLevelDataset(**mock_config)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length
        eod = tokenizer.eod
        pad = tokenizer.pad

        tokens = self.dataset[int(self.indices[idx % len(self.indices)])]

        # Convert tokens to list and always append EOD to ensure length consistency.
        # The low-level dataset returns length-1 tokens, and we add EOD to make it length tokens.
        tokens_list = tokens.tolist()
        tokens_list.append(eod)

        if self.dataset.format == "sbhd":
            # SBHD format: single padded sequence without cu_seqlens.
            # Long sequences are truncated to pack_length tokens (including EOD).
            if len(tokens_list) >= pack_length + 1:
                tokens_list = tokens_list[:pack_length - 1] + [eod]
            # Pad to pack_length + 1 (offset by 1 for input/label split).
            pad_len = pack_length + 1 - len(tokens_list)
            if pad_len > 0:
                tokens_list = tokens_list + [pad] * pad_len
            assert len(tokens_list) == pack_length + 1
            input_ids    = torch.tensor(tokens_list[:-1], dtype=torch.int64)
            labels       = torch.tensor(tokens_list[1:],  dtype=torch.int64)
            # Position IDs are sequential across the entire sequence including padding,
            # matching GPTDataset behavior for standard (non-packed) training.
            position_ids = torch.arange(pack_length, dtype=torch.int64)
            loss_mask = torch.ones(pack_length, dtype=torch.float32)
            loss_mask[labels == pad] = 0.0
            return {
                'tokens':       input_ids,
                'labels':       labels,
                'loss_mask':    loss_mask,
                'position_ids': position_ids,
            }

        # THD format (sequence packing) below.
        def extend_with_padding(tokens, positions, pad_len):
            tokens.extend([pad] * pad_len)
            positions.extend(range(positions[-1] + 1, positions[-1] + 1 + pad_len))

        pack_tokens = list(tokens_list) + [pad]
        pack_positions = list(range(len(pack_tokens)))

        # Truncate if sequence exceeds pack_length + 1 (need +1 for shift).
        if len(pack_tokens) > pack_length + 1:
            pack_tokens = pack_tokens[:pack_length - 1] + [eod, pad]
            pack_positions = pack_positions[:pack_length + 1]

        # Pad to pad_granularity alignment (tp * cp * 2).
        # We need final length (after shift) to be divisible by pad_granularity.
        pad_granularity = self._calculate_padding_divisor()
        final_len = len(pack_tokens) - 1
        mod_token_count = final_len % pad_granularity
        if mod_token_count != 0:
            pad_len = pad_granularity - mod_token_count
            extend_with_padding(pack_tokens, pack_positions, pad_len)

        # Apply shift for next-token prediction.
        input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(pack_tokens[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        seq_len = len(input_ids)
        cu_seqlens = [0, seq_len]

        # Loss mask: mask padding tokens
        loss_mask = torch.ones(seq_len, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0

        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        max_seqlen = torch.tensor(seq_len, dtype=torch.int32)

        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }
