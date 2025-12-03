# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

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
        sft_sequence_packing = self.config.sft_sequence_packing
        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length

        conversation_list = self.dataset[int(self.indices[idx % len(self.indices)])]
        tokens, target = tokenizer.tokenize_conversation(
            conversation_list, return_target=True, add_generation_prompt=False
        )

        force_eod_length = int(tokenizer.force_eod)

        if len(tokens) > max_seq_len - force_eod_length:
            tokens = tokens[: max_seq_len - force_eod_length]
            target = target[: max_seq_len - force_eod_length]

        # if use sequence packing, pad according to get_padding_size
        # else pad to max_seq_len
        num_tokens = len(tokens) + force_eod_length
        if sft_sequence_packing:
            padding_len = self.get_padding_size(num_tokens) - num_tokens
            # debugmtl
            # padding_len = max_seq_len - num_tokens
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

        if sft_sequence_packing:
            # sequence packing need both original sequence length and padded length
            ret['original_seq_len'] = torch.tensor(num_tokens, dtype=torch.int32, device=tokens.device)

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
                torch.ones((max_seq_len, max_seq_len), device=target.device)
            ).unsqueeze(0)
            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return loss_mask, position_ids, attention_mask

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
        sample = np.arange(2, length + 1 , dtype=np.int64)
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
        sft_sequence_packing = self.config.sft_sequence_packing
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
        if sft_sequence_packing:
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

        if sft_sequence_packing:
            # sequence packing need both original sequence length and padded length
            ret['original_seq_len'] = torch.tensor(num_tokens, dtype=torch.int32, device=tokens.device)

        return ret