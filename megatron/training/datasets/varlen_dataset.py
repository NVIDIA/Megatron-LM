# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Variable-length raw-text datasets for packed THD pretraining.

The real-data path accepts HuggingFace datasets, local parquet files, and local
JSON/JSONL files with a ``text`` column. The mock path generates samples from
file-based or lognormal sequence-length distributions. Both paths emit one
unpacked sample at a time for the sequence-packing scheduler.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset


@dataclass
class VarlenDatasetConfig(GPTDatasetConfig):
    """Configuration for variable-length real and mock pretraining datasets."""

    mock_dataset_config: Optional[Dict[str, Any]] = None
    sbhd_validation: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        assert not self.hybrid_context_parallel, (
            "VarlenDataset uses the dp_balanced static-CP scheduler and cannot use "
            "hybrid context parallelism"
        )


def _looks_like_hf_id(path: str) -> bool:
    """Return whether ``path`` looks like a HuggingFace dataset identifier."""
    if not path or os.path.exists(path) or path.startswith(("/", "./", "../")):
        return False
    return "/" in path


def _raw_text_loader(sample: Dict[str, Any]) -> str:
    """Return and validate a pretraining sample's ``text`` field."""
    text = sample.get("text", "")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise ValueError(
            "VarlenDataset requires the 'text' field to be a string, "
            f"got {type(text).__name__}."
        )
    return text


class VarlenLowLevelDataset:
    """Load variable-length pretraining text from HF, parquet, JSON, or JSONL."""

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import Dataset, load_dataset
        except ImportError as exc:
            raise ImportError(
                "VarlenDataset requires the `datasets` library (pip install datasets)."
            ) from exc

        if _looks_like_hf_id(dataset_path):
            self.dataset = load_dataset(dataset_path, split="train")
        elif dataset_path.endswith(".parquet"):
            self.dataset = load_dataset("parquet", data_files=dataset_path, split="all")
        else:
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError(
                    "VarlenDataset requires `pandas` to load local JSON/JSONL files "
                    "(pip install pandas)."
                ) from exc
            dataframe = pd.read_json(
                dataset_path, lines=not dataset_path.lower().endswith(".json")
            )
            self.dataset = Dataset.from_pandas(dataframe, preserve_index=False)

        if "text" not in self.dataset.column_names:
            raise ValueError(
                "VarlenDataset requires a raw pretraining 'text' column, "
                f"got {sorted(self.dataset.column_names)}."
            )

    @property
    def schema_name(self) -> str:
        """Return the single supported real-data schema."""
        return "pretrain-text"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> str:
        return _raw_text_loader(self.dataset[idx])


class VarlenDataset(MegatronDataset):
    """Variable-length raw-text dataset consumed by the packing scheduler."""

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: VarlenDatasetConfig
    ) -> LowLevelDataset:
        return VarlenLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _calculate_padding_divisor(self) -> int:
        """Return the per-sample alignment required before DP/CP packing."""
        cp_size = self.config.context_parallel_size or 1
        cp_pad = cp_size * 2 if cp_size > 1 else 1
        sp_pad = self.config.sequence_parallel_size or 1
        return cp_pad * sp_pad

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        max_len = self.config.sequence_length
        eod = tokenizer.eod
        pad = tokenizer.pad if tokenizer.pad is not None else eod
        assert eod is not None, "VarlenDataset requires an EOD/EOS token id."
        assert not self.config.reset_position_ids
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask

        text = self.dataset[int(self.indices[idx % len(self.indices)])]
        if not isinstance(text, str):
            raise ValueError(
                "VarlenDataset expects raw pretraining text, "
                f"got {type(text).__name__}."
            )
        tokens_list = list(tokenizer.tokenize(text))
        targets_list = list(tokens_list)

        if not tokens_list:
            tokens_list = [eod, eod]
            targets_list = [eod, eod]

        if len(tokens_list) > max_len + 1:
            tokens_list = tokens_list[: max_len + 1]
            targets_list = targets_list[: max_len + 1]
        if len(tokens_list) == max_len + 1 and tokens_list[-1] != eod:
            tokens_list[-1] = eod
            targets_list[-1] = eod
        if tokens_list[-1] != eod:
            tokens_list.append(eod)
            targets_list.append(eod)

        valid_len = len(tokens_list) - 1

        if self.config.sbhd_validation:
            pad_len = max_len + 1 - len(tokens_list)
            if pad_len > 0:
                tokens_list.extend([pad] * pad_len)
                targets_list.extend([pad] * pad_len)
            assert len(tokens_list) == max_len + 1
            input_ids = torch.tensor(tokens_list[:-1], dtype=torch.int64)
            labels = torch.tensor(targets_list[1:], dtype=torch.int64)
            loss_mask = torch.ones(max_len, dtype=torch.float32)
            loss_mask[valid_len:] = 0.0
            if self.config.eod_mask_loss:
                loss_mask[input_ids == eod] = 0.0
            return {
                "tokens": input_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": torch.arange(max_len, dtype=torch.int64),
            }

        original_seq_len = len(tokens_list) - 1
        padding_divisor = self._calculate_padding_divisor()
        remainder = original_seq_len % padding_divisor
        if remainder:
            pad_len = padding_divisor - remainder
            tokens_list.extend([pad] * pad_len)
            targets_list.extend([pad] * pad_len)
        padded_seq_len = len(tokens_list) - 1

        input_ids = torch.tensor(tokens_list[:-1], dtype=torch.int64)
        labels = torch.tensor(targets_list[1:], dtype=torch.int64)
        loss_mask = torch.ones(padded_seq_len, dtype=torch.float32)
        loss_mask[valid_len:] = 0.0
        if self.config.eod_mask_loss:
            loss_mask[input_ids == eod] = 0.0

        return {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(padded_seq_len, dtype=torch.int64),
            "original_seq_len": torch.tensor([original_seq_len], dtype=torch.int32),
            "padded_seq_len": torch.tensor([padded_seq_len], dtype=torch.int32),
        }


class MockVarlenLowLevelDataset:
    """Generate mock token arrays from file-based or lognormal lengths."""

    seed: int = 0
    size: int = 1_000_000

    def __init__(self, mode: str, **kwargs) -> None:
        np.random.seed(self.seed)
        if mode == "file":
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError(
                    "MockVarlenDataset file mode requires pandas (pip install pandas)."
                ) from exc
            self.sequence_lengths = np.asarray(pd.read_csv(kwargs["path"])).flatten()
            self.size = len(self.sequence_lengths)
        elif mode == "distribution":
            if kwargs["type"] != "lognormal":
                raise ValueError(f"Unsupported distribution type {kwargs['type']}")
            sigma = kwargs["lognormal_sigma"]
            mean = kwargs["mean_seq_len"]
            mu = np.log(mean) - sigma**2 / 2
            samples = np.random.lognormal(mu, sigma, self.size)
            self.sequence_lengths = np.clip(
                samples, kwargs["min_seq_len"], kwargs["max_seq_len"]
            ).astype(int)
        else:
            raise ValueError(f"Unsupported mode '{mode}', must be 'file' or 'distribution'")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> np.ndarray:
        length = self.sequence_lengths[idx % self.size]
        return np.arange(1, length, dtype=np.int64)


class MockVarlenDataset(VarlenDataset):
    """Mock variable-length dataset for the packed THD benchmark path."""

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: VarlenDatasetConfig
    ) -> LowLevelDataset:
        mock_config = config.mock_dataset_config
        if mock_config is None:
            mock_config = {
                "mode": "distribution",
                "type": "lognormal",
                "min_seq_len": config.sequence_length // 2,
                "max_seq_len": config.sequence_length,
                "mean_seq_len": config.sequence_length // 4 * 3,
                "lognormal_sigma": 1.1,
            }
        return MockVarlenLowLevelDataset(**mock_config)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        max_len = self.config.sequence_length
        eod = tokenizer.eod
        pad = tokenizer.pad if tokenizer.pad is not None else eod

        tokens_list = self.dataset[int(self.indices[idx % len(self.indices)])].tolist()
        if not tokens_list:
            tokens_list.append(eod)
        tokens_list.append(eod)
        targets_list = list(tokens_list)

        if len(tokens_list) > max_len + 1:
            tokens_list = tokens_list[:max_len] + [eod]
            targets_list = targets_list[:max_len] + [eod]
        original_seq_len = len(tokens_list) - 1

        padding_divisor = self._calculate_padding_divisor()
        remainder = original_seq_len % padding_divisor
        if remainder:
            pad_len = padding_divisor - remainder
            tokens_list.extend([pad] * pad_len)
            targets_list.extend([pad] * pad_len)
        padded_seq_len = len(tokens_list) - 1

        input_ids = torch.tensor(tokens_list[:-1], dtype=torch.int64)
        labels = torch.tensor(targets_list[1:], dtype=torch.int64)
        loss_mask = torch.ones(padded_seq_len, dtype=torch.float32)
        loss_mask[original_seq_len:] = 0.0
        if self.config.eod_mask_loss:
            loss_mask[input_ids == eod] = 0.0

        return {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(padded_seq_len, dtype=torch.int64),
            "original_seq_len": torch.tensor([original_seq_len], dtype=torch.int32),
            "padded_seq_len": torch.tensor([padded_seq_len], dtype=torch.int32),
        }
