# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Variable-length dataset producers for packed THD training.

The real-data loader accepts Hugging Face dataset ids, local parquet files,
and local JSON/JSONL files. Instruction data is normalized from common
OpenAI, ShareGPT, and Alpaca layouts; a plain ``text`` column is treated as
pretraining data. Each mid-level sample remains unpacked so a downstream
packing scheduler can balance and combine samples across ranks.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from megatron.core.datasets.megatron_dataset import LowLevelDataset
from megatron.core.datasets.utils import Split
from megatron.training.datasets.sft_dataset import (
    IGNORE_INDEX,
    MockSFTDataset,
    MockSFTLowLevelDataset,
    SFTDataset,
    SFTDatasetConfig,
    _calculate_padding_divisor,
    _get_padding_token_id,
    _normalize_mock_token_ids,
)

_INSTRUCTION_FIELDS: Tuple[str, ...] = ("instruction", "prompt", "query", "question")
_OUTPUT_FIELDS: Tuple[str, ...] = ("output", "response", "completion", "answer")
_EXTRA_INPUT_FIELDS: Tuple[str, ...] = ("input", "context")

_SHAREGPT_ROLE_MAP: Dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


@dataclass
class VarlenDatasetConfig(SFTDatasetConfig):
    """Training-local configuration for variable-length dataset producers."""

    varlen_sbhd_validation: bool = False
    """Emit fixed-width SBHD samples instead of variable-length THD samples."""

    def __post_init__(self) -> None:
        """Validate layout-specific constraints after base SFT validation."""
        super().__post_init__()
        if self.varlen_sbhd_validation and self.hybrid_context_parallel:
            raise ValueError("varlen_sbhd_validation is incompatible with hybrid_context_parallel")
        if not self.varlen_sbhd_validation:
            padding_divisor = _calculate_padding_divisor(self)
            if self.sequence_length % padding_divisor != 0:
                raise ValueError(
                    "VarlenDatasetConfig.sequence_length must be divisible by the CP/SP "
                    f"padding divisor ({padding_divisor})"
                )


class _LocalJsonDataset:
    """Small index over a local JSON array or newline-delimited JSON file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._records: list[dict[str, Any]] | None = None
        self._offsets: list[int] | None = None
        self.column_names: list[str] = []

        with open(path, "rb") as stream:
            first_non_whitespace = b""
            while True:
                value = stream.read(1)
                if not value or not value.isspace():
                    first_non_whitespace = value
                    break

        if first_non_whitespace == b"[":
            with open(path, encoding="utf-8") as stream:
                records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON array in {path!r}")
            self._records = [self._validate_record(record, path) for record in records]
            self._collect_column_names(self._records)
        else:
            self._offsets = []
            with open(path, "rb") as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL record at byte offset {offset} in {path!r}"
                        ) from exc
                    record = self._validate_record(record, path)
                    self._offsets.append(offset)
                    self._collect_column_names([record])

        if len(self) == 0:
            raise ValueError(f"Local dataset {path!r} must contain at least one record")

    @staticmethod
    def _validate_record(record: Any, path: str) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError(f"Every record in {path!r} must be a JSON object")
        return record

    def _collect_column_names(self, records: Iterable[dict[str, Any]]) -> None:
        seen = set(self.column_names)
        for record in records:
            for key in record:
                if key not in seen:
                    self.column_names.append(key)
                    seen.add(key)

    def __len__(self) -> int:
        if self._records is not None:
            return len(self._records)
        assert self._offsets is not None
        return len(self._offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._records is not None:
            return self._records[idx]
        assert self._offsets is not None
        offset = self._offsets[idx]
        with open(self.path, "rb") as stream:
            stream.seek(offset)
            return self._validate_record(json.loads(stream.readline()), self.path)


def _looks_like_hf_id(path: str | None) -> bool:
    """Return whether ``path`` resembles a non-local ``owner/repository`` id."""
    local_suffixes = (".json", ".jsonl", ".parquet")
    if (
        not path
        or os.path.exists(path)
        or path.startswith(("/", "./", "../"))
        or path.lower().endswith(local_suffixes)
    ):
        return False
    return True


def _first_present(sample: Dict[str, Any], fields: Iterable[str]) -> str | None:
    """Return the first non-empty string found under ``fields``."""
    for field_name in fields:
        value = sample.get(field_name)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"VarlenDataset field {field_name!r} must be a string, "
                f"got {type(value).__name__}"
            )
        return value
    return None


def _ensure_str_content(content: Any, location: str) -> str:
    """Validate a chat turn's scalar text content."""
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(
            f"VarlenDataset {location} content must be a string, "
            f"got {type(content).__name__}; multimodal content is not supported"
        )
    return content


def _alpaca_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert an Alpaca/Dolly-style record to chat messages."""
    instruction = _first_present(sample, _INSTRUCTION_FIELDS) or ""
    extra_input = _first_present(sample, _EXTRA_INPUT_FIELDS) or ""
    output = _first_present(sample, _OUTPUT_FIELDS) or ""
    user_content = f"{instruction}\n\n{extra_input}" if extra_input else instruction
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def _sharegpt_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a ShareGPT ``conversations`` record to chat messages."""
    conversations = sample.get("conversations") or []
    if not isinstance(conversations, list):
        raise ValueError("VarlenDataset 'conversations' must be a list")

    messages: List[Dict[str, str]] = []
    first_speaker = ""
    if conversations:
        if not isinstance(conversations[0], dict):
            raise ValueError("Every ShareGPT turn must be an object")
        first_speaker = str(conversations[0].get("from") or "").lower()
    if first_speaker != "system":
        messages.append({"role": "system", "content": ""})

    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError("Every ShareGPT turn must be an object")
        speaker = str(turn.get("from") or "").lower()
        role = _SHAREGPT_ROLE_MAP.get(speaker, "user")
        messages.append(
            {
                "role": role,
                "content": _ensure_str_content(
                    turn.get("value"), f"ShareGPT turn with role {role!r}"
                ),
            }
        )
    return messages


def _messages_passthrough(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize an OpenAI ``messages`` record to role/content pairs."""
    raw_messages = sample.get("messages") or []
    if not isinstance(raw_messages, list):
        raise ValueError("VarlenDataset 'messages' must be a list")
    if any(not isinstance(message, dict) for message in raw_messages):
        raise ValueError("Every OpenAI message must be an object")

    if raw_messages and raw_messages[0].get("role") != "system":
        raw_messages = [{"role": "system", "content": ""}, *raw_messages]
    return [
        {
            "role": str(message.get("role") or "user"),
            "content": _ensure_str_content(
                message.get("content"),
                f"OpenAI message with role {message.get('role') or 'user'!r}",
            ),
        }
        for message in raw_messages
    ]


def _raw_text_loader(sample: Dict[str, Any]) -> str:
    """Return a pretraining ``text`` value without chat normalization."""
    text = sample.get("text")
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError(
            f"VarlenDataset pretraining text must be a string, got {type(text).__name__}"
        )
    return text


def _select_converter(column_names: Sequence[str]) -> Tuple[Callable[[Dict[str, Any]], Any], str]:
    """Choose a record converter from the available column names."""
    columns = set(column_names)
    if "messages" in columns:
        return _messages_passthrough, "openai-messages"
    if "conversations" in columns:
        return _sharegpt_to_messages, "sharegpt"
    if any(field in columns for field in _INSTRUCTION_FIELDS) and any(
        field in columns for field in _OUTPUT_FIELDS
    ):
        return _alpaca_to_messages, "alpaca"
    if "text" in columns:
        return _raw_text_loader, "pretrain-text"
    raise ValueError(
        "VarlenDataset cannot infer a supported schema from columns " f"{sorted(columns)}"
    )


class VarlenLowLevelDataset:
    """Load HF, parquet, JSON, or JSONL records and normalize their schema."""

    def __init__(self, dataset_path: str) -> None:
        if _looks_like_hf_id(dataset_path):
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "VarlenLowLevelDataset requires `datasets` for Hugging Face ids"
                ) from exc
            self.dataset = load_dataset(dataset_path, split="train")
        elif dataset_path.endswith(".parquet"):
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "VarlenLowLevelDataset requires `datasets` for parquet files"
                ) from exc
            self.dataset = load_dataset("parquet", data_files=dataset_path, split="all")
        else:
            self.dataset = _LocalJsonDataset(dataset_path)

        _, self._schema_name = _select_converter(self.dataset.column_names)

    @property
    def schema_name(self) -> str:
        """Detected schema: Alpaca, ShareGPT, OpenAI messages, or pretrain text."""
        return self._schema_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        record = self.dataset[idx]
        populated_columns = [key for key, value in record.items() if value is not None]
        converter, _ = _select_converter(populated_columns)
        return converter(record)


class VarlenDataset(SFTDataset):
    """Return one unpacked variable-length sample for downstream THD packing."""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: str | None,
        indices: np.ndarray,
        num_samples: int | None,
        index_split: Split,
        config: VarlenDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: VarlenDatasetConfig
    ) -> VarlenLowLevelDataset:
        """Build a normalized low-level variable-length dataset."""
        del config
        return VarlenLowLevelDataset(dataset_path)

    def _tokenize_item(self, item: Any) -> tuple[list[int], list[int]]:
        """Tokenize one raw-text or normalized conversation item."""
        tokenizer = self.config.tokenizer
        if isinstance(item, str):
            token_ids = tokenizer.tokenize(item)
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            tokens = list(token_ids)
            targets = list(tokens)
        else:
            token_ids, target_ids = tokenizer.tokenize_conversation(
                item, return_target=True, add_generation_prompt=False
            )
            tokens = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
            targets = target_ids.tolist() if hasattr(target_ids, "tolist") else list(target_ids)
        if len(tokens) != len(targets):
            raise ValueError("Tokenizer returned different token and target lengths")
        return tokens, targets

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        sequence_length = self.config.sequence_length
        eod = tokenizer.eod
        if eod is None:
            raise ValueError("VarlenDataset requires an EOD/EOS token id")
        pad = _get_padding_token_id(tokenizer, eod)

        item = self.dataset[int(self.indices[idx % len(self.indices)])]
        if self.config.reset_position_ids:
            raise ValueError("VarlenDataset does not support reset_position_ids")
        if self.config.create_attention_mask or self.config.reset_attention_mask:
            raise ValueError("VarlenDataset requires attention-mask creation to be disabled")

        tokens, targets = self._tokenize_item(item)
        if not tokens:
            tokens = [eod, eod]
            targets = [eod, eod]
        elif tokens[-1] != eod:
            tokens.append(eod)
            targets.append(eod)
        if len(tokens) == 1:
            tokens.append(eod)
            targets.append(eod)

        if len(tokens) > sequence_length + 1:
            tokens = tokens[: sequence_length + 1]
            targets = targets[: sequence_length + 1]

        valid_length = len(tokens) - 1
        if self.config.varlen_sbhd_validation:
            padding_length = sequence_length + 1 - len(tokens)
            tokens.extend([pad] * padding_length)
            targets.extend([pad] * padding_length)
            input_ids = torch.tensor(tokens[:-1], dtype=torch.int64)
            labels = torch.tensor(targets[1:], dtype=torch.int64)
            loss_mask = torch.ones(sequence_length, dtype=torch.float32)
            loss_mask[valid_length:] = 0.0
            loss_mask[labels == IGNORE_INDEX] = 0.0
            return {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'position_ids': torch.arange(sequence_length, dtype=torch.int64),
            }

        padding_length = (-valid_length) % self.padding_divisor
        tokens.extend([pad] * padding_length)
        targets.extend([pad] * padding_length)
        padded_length = len(tokens) - 1

        input_ids = torch.tensor(tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(targets[1:], dtype=torch.int64)
        loss_mask = torch.ones(padded_length, dtype=torch.float32)
        loss_mask[valid_length:] = 0.0
        loss_mask[labels == IGNORE_INDEX] = 0.0
        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': torch.arange(padded_length, dtype=torch.int64),
            'original_seq_len': torch.tensor([valid_length], dtype=torch.int32),
            'padded_seq_len': torch.tensor([padded_length], dtype=torch.int32),
        }


class MockVarlenDataset(MockSFTDataset):
    """Mock producer with the same unpacked THD output as :class:`VarlenDataset`."""

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: VarlenDatasetConfig
    ) -> MockSFTLowLevelDataset:
        """Build a mock source from the inherited flat fields."""
        return MockSFTDataset.build_low_level_dataset(dataset_path, config)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.config.varlen_sbhd_validation:
            raise ValueError("MockVarlenDataset supports only unpacked THD output")

        tokenizer = self.config.tokenizer
        sequence_length = self.config.sequence_length
        eod = tokenizer.eod
        if eod is None:
            raise ValueError("MockVarlenDataset requires an EOD/EOS token id")
        pad = _get_padding_token_id(tokenizer, eod)

        raw_tokens = self.dataset[int(self.indices[idx % len(self.indices)])]
        tokens = _normalize_mock_token_ids(
            raw_tokens, tokenizer, verification=self.dataset.indexed_dataset is not None
        )
        tokens.append(eod)
        if len(tokens) > sequence_length + 1:
            tokens = tokens[: sequence_length + 1]

        valid_length = len(tokens) - 1
        padding_length = (-valid_length) % self.padding_divisor
        tokens.extend([pad] * padding_length)
        padded_length = len(tokens) - 1

        input_ids = torch.tensor(tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(tokens[1:], dtype=torch.int64)
        loss_mask = torch.ones(padded_length, dtype=torch.float32)
        loss_mask[valid_length:] = 0.0
        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': torch.arange(padded_length, dtype=torch.int64),
            'original_seq_len': torch.tensor([valid_length], dtype=torch.int32),
            'padded_seq_len': torch.tensor([padded_length], dtype=torch.int32),
        }
