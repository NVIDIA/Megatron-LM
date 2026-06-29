# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Supervised fine-tuning datasets and typed mock-data producers."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


@dataclass
class SFTDatasetConfig(GPTDatasetConfig):
    """Training-local configuration for SFT datasets and their mock producers.

    Mock configuration is represented by flat, typed fields. ``mock_data_path``
    points to a headerless sequence-length file in ``file`` mode and to an
    ``IndexedDataset`` prefix in ``verification`` mode.
    """

    mock_data_mode: Literal["distribution", "file", "verification"] = "distribution"
    """Source used by :class:`MockSFTLowLevelDataset`."""

    mock_data_path: str | None = None
    """Mode-specific local path; unused in ``distribution`` mode."""

    mock_data_distribution: Literal["lognormal"] = "lognormal"
    """Synthetic sequence-length distribution."""

    mock_data_min_sequence_length: int | None = None
    """Minimum generated sequence length, including the final EOD token."""

    mock_data_max_sequence_length: int | None = None
    """Maximum generated sequence length, including the final EOD token."""

    mock_data_mean_sequence_length: float | None = None
    """Mean of the unclipped synthetic sequence-length distribution."""

    mock_data_lognormal_sigma: float = 1.1
    """Sigma of the synthetic lognormal sequence-length distribution."""

    mock_data_seed: int = 0
    """Seed used by the mock dataset's private NumPy generator."""

    mock_data_size: int = 1_000_000
    """Number of generated samples in ``distribution`` and ``verification`` modes."""

    emit_packing_metadata: bool = False
    """Emit real-token boundaries required by the dynamic packing scheduler."""

    def __post_init__(self) -> None:
        """Set sequence-length defaults and validate the flat mock fields."""
        super().__post_init__()

        if self.sequence_length < 2:
            raise ValueError("SFTDatasetConfig.sequence_length must be at least 2")

        if self.mock_data_min_sequence_length is None:
            self.mock_data_min_sequence_length = max(2, self.sequence_length // 2)
        if self.mock_data_max_sequence_length is None:
            self.mock_data_max_sequence_length = max(2, self.sequence_length)
        if self.mock_data_mean_sequence_length is None:
            self.mock_data_mean_sequence_length = max(2, self.sequence_length * 3 / 4)

        if self.mock_data_mode not in {"distribution", "file", "verification"}:
            raise ValueError(
                "mock_data_mode must be one of 'distribution', 'file', or 'verification'"
            )
        if self.mock_data_mode in {"file", "verification"} and not self.mock_data_path:
            raise ValueError(f"mock_data_path is required in {self.mock_data_mode!r} mode")
        if self.mock_data_distribution != "lognormal":
            raise ValueError("mock_data_distribution currently supports only 'lognormal'")
        if self.mock_data_min_sequence_length < 2:
            raise ValueError("mock_data_min_sequence_length must be at least 2")
        if self.mock_data_max_sequence_length < self.mock_data_min_sequence_length:
            raise ValueError(
                "mock_data_max_sequence_length must be greater than or equal to "
                "mock_data_min_sequence_length"
            )
        if self.mock_data_mean_sequence_length <= 0:
            raise ValueError("mock_data_mean_sequence_length must be positive")
        if self.mock_data_lognormal_sigma < 0:
            raise ValueError("mock_data_lognormal_sigma must be non-negative")
        if self.mock_data_size <= 0:
            raise ValueError("mock_data_size must be positive")


def _calculate_padding_divisor(config: GPTDatasetConfig) -> int:
    """Return the token alignment required by CP and sequence parallelism."""
    context_parallel_size = config.context_parallel_size or 1
    sequence_parallel_size = config.sequence_parallel_size or 1
    if config.hybrid_context_parallel:
        context_parallel_padding = config.data_parallel_size * context_parallel_size * 2
    else:
        context_parallel_padding = context_parallel_size * 2 if context_parallel_size > 1 else 1
    return context_parallel_padding * sequence_parallel_size


def _get_padding_token_id(tokenizer, eod: int) -> int:
    """Return a valid padding id, falling back to EOD when no pad token exists."""

    pad = None
    for attribute in ("pad", "pad_id"):
        try:
            pad = getattr(tokenizer, attribute)
        except (AttributeError, NotImplementedError):
            continue
        if pad is not None:
            break

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(pad, int) or pad < 0 or (vocab_size is not None and pad >= vocab_size):
        return eod
    return pad


def _normalize_mock_token_ids(token_ids: np.ndarray, tokenizer, verification: bool) -> list[int]:
    """Keep synthetic mock ids inside the tokenizer vocabulary."""

    vocab_size = int(tokenizer.vocab_size)
    if vocab_size <= 0:
        raise ValueError("Mock datasets require a positive tokenizer vocabulary size")
    token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    invalid = (token_ids < 0) | (token_ids >= vocab_size)
    if np.any(invalid):
        if verification:
            raise ValueError("Verification tokens must be inside the tokenizer vocabulary")
        token_ids = np.mod(token_ids, vocab_size)
    return token_ids.tolist()


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
        except ImportError as exc:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            ) from exc
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
        self.padding_divisor = _calculate_padding_divisor(config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        """Return the number of records in the low-level SFT dataset."""
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        """Build the JSON-backed low-level SFT dataset."""
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples if self.num_samples is not None else len(self.indices)

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
                current.append(msg)  # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        merged_conversations = self.dataset[int(self.indices[idx % len(self.indices)])]
        split_conversations = self._split_conversations(merged_conversations)

        def extend_with_padding(tokens, targets, positions, padding_flags, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            position_start = positions[-1] + 1 if positions else 0
            positions.extend(range(position_start, position_start + pad_len))
            padding_flags.extend([True] * pad_len)

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        pack_padding_flags = []
        cu_seqlens = [0]
        cu_seqlens_original = [0]
        eod = tokenizer.eod
        if eod is None:
            raise ValueError("SFTDataset requires an EOD/EOS token id")
        pad = _get_padding_token_id(tokenizer, eod)
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        for conversation in split_conversations:
            physical_sequence_start = cu_seqlens[-1]
            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()
            if self.config.emit_packing_metadata and len(tokens_list) < 2:
                # THD real boundaries describe next-token prediction steps, so a
                # one-token conversation has no representable real sequence.
                continue

            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)
            pack_padding_flags.extend([False] * len(tokens_list))

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            mod_token_count = len(pack_tokens) % self.padding_divisor
            if mod_token_count != 0:
                pad_len = self.padding_divisor - mod_token_count
                extend_with_padding(
                    pack_tokens, pack_targets, pack_positions, pack_padding_flags, pad_len
                )

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))
            real_sequence_length = max(0, len(tokens_list) - 1)
            cu_seqlens_original.append(cu_seqlens_original[-1] + real_sequence_length)

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                # Truncate on the right
                max_body = pack_length
                pack_tokens = pack_tokens[:max_body]
                pack_targets = pack_targets[:max_body]
                pack_padding_flags = pack_padding_flags[:max_body]
                pack_tokens.append(pad)
                pack_targets.append(pad)
                pack_padding_flags.append(True)
                pack_positions = pack_positions[: pack_length + 1]
                # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
                cu_seqlens[-1] = len(pack_tokens) - 1
                retained_real_tokens = max(
                    0, min(real_sequence_length, pack_length - physical_sequence_start - 1)
                )
                if retained_real_tokens == 0 and self.config.emit_packing_metadata:
                    # A one-slot tail cannot contain an input/target pair. Fold it into
                    # the previous sequence's physical padding instead of emitting a
                    # zero-length real sequence, which THD attention cannot represent.
                    cu_seqlens[-2] = cu_seqlens[-1]
                    cu_seqlens.pop()
                    cu_seqlens_original.pop()
                else:
                    cu_seqlens_original[-1] = cu_seqlens_original[-2] + retained_real_tokens
                break

        if self.config.emit_packing_metadata and len(cu_seqlens) == 1:
            raise ValueError(
                "Packing metadata requires at least one conversation with two or more tokens"
            )

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(
                pack_tokens, pack_targets, pack_positions, pack_padding_flags, pad_len
            )
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1
        assert len(pack_padding_flags) == pack_length + 1

        # Align and convert to tensors
        input_ids = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
        labels = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask.
        loss_mask = (~torch.tensor(pack_padding_flags[1:], dtype=torch.bool)).to(torch.float32)
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        # attention_mask = None

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        cu_seqlens_original = torch.tensor(cu_seqlens_original, dtype=torch.int32)
        # Calculating max_seqlen here, rather than incrementally above, because of possible
        # effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor

        # Pad cu_seqlens to a fixed length so that default_collate can
        # stack samples with different numbers of documents.  Trailing
        # entries are filled with pack_length; the merge helper strips
        # them later.
        padded_cu_seqlens = torch.full((pack_length + 1,), pack_length, dtype=torch.int32)
        padded_cu_seqlens[: cu_seqlens.numel()] = cu_seqlens
        padded_cu_seqlens_original = torch.full(
            (pack_length + 1,), int(cu_seqlens_original[-1]), dtype=torch.int32
        )
        padded_cu_seqlens_original[: cu_seqlens_original.numel()] = cu_seqlens_original

        sample = {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': padded_cu_seqlens,
            'max_seqlen': max_seqlen,
        }
        if getattr(self.config, "emit_packing_metadata", False):
            sample['cu_seqlens_original'] = padded_cu_seqlens_original
        return sample


class MockSFTLowLevelDataset:
    """Generate mock token arrays from typed sequence-length settings.

    Args:
        mode: ``distribution``, ``file``, or ``verification``.
        path: Headerless sequence-length file for ``file`` mode, or an
            ``IndexedDataset`` prefix for ``verification`` mode.
        distribution: Synthetic distribution name. Only ``lognormal`` is
            currently supported.
        min_sequence_length: Minimum generated length, including EOD.
        max_sequence_length: Maximum generated length, including EOD.
        mean_sequence_length: Mean of the unclipped distribution.
        lognormal_sigma: Sigma of the lognormal distribution.
        seed: Seed for a private NumPy random generator.
        size: Number of generated samples.
    """

    def __init__(
        self,
        mode: Literal["distribution", "file", "verification"],
        path: str | None,
        distribution: Literal["lognormal"],
        min_sequence_length: int,
        max_sequence_length: int,
        mean_sequence_length: float,
        lognormal_sigma: float,
        seed: int,
        size: int,
    ) -> None:
        self.indexed_dataset: IndexedDataset | None = None

        if mode == "file":
            if path is None:
                raise ValueError("path is required in file mode")
            self.sequence_lengths = self._load_sequence_lengths(path)
            self.size = int(self.sequence_lengths.size)
        elif mode in {"distribution", "verification"}:
            if distribution != "lognormal":
                raise ValueError(f"Unsupported distribution {distribution!r}")
            self.sequence_lengths = self._generate_lognormal_samples(
                size,
                mean_sequence_length,
                lognormal_sigma,
                min_sequence_length,
                max_sequence_length,
                seed,
            )
            self.size = size
            if mode == "verification":
                if path is None:
                    raise ValueError("path is required in verification mode")
                self.indexed_dataset = IndexedDataset(path)
                if len(self.indexed_dataset) == 0:
                    raise ValueError("verification IndexedDataset must not be empty")
        else:
            raise ValueError(
                f"Unsupported mode {mode!r}; expected 'distribution', 'file', or 'verification'"
            )

    @staticmethod
    def _load_sequence_lengths(path: str) -> np.ndarray:
        """Read all values from a headerless comma-separated numeric file."""
        try:
            sequence_lengths = np.loadtxt(path, dtype=np.int64, delimiter=",", ndmin=1)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Unable to read headerless integer sequence lengths from {path!r}"
            ) from exc
        sequence_lengths = np.asarray(sequence_lengths, dtype=np.int64).reshape(-1)
        if sequence_lengths.size == 0:
            raise ValueError("Sequence-length file must contain at least one value")
        if np.any(sequence_lengths < 2):
            raise ValueError("Every mock sequence length must be at least 2")
        return sequence_lengths

    @staticmethod
    def _generate_lognormal_samples(
        size: int,
        mean: float,
        sigma: float,
        min_sequence_length: int,
        max_sequence_length: int,
        seed: int,
    ) -> np.ndarray:
        """Generate deterministic clipped lognormal sequence lengths."""
        mu = np.log(mean) - sigma**2 / 2
        rng = np.random.default_rng(seed)
        samples = rng.lognormal(mu, sigma, size)
        return np.clip(samples, min_sequence_length, max_sequence_length).astype(np.int64)

    def __len__(self) -> int:
        return self.size

    def _get_verification_tokens(self, idx: int, target_length: int) -> np.ndarray:
        """Concatenate indexed documents until ``target_length`` tokens are available."""
        assert self.indexed_dataset is not None
        if target_length == 0:
            return np.empty(0, dtype=np.int64)

        chunks = []
        total_length = 0
        document_index = idx % len(self.indexed_dataset)
        empty_documents_seen = 0
        while total_length < target_length:
            document = np.asarray(
                self.indexed_dataset[document_index % len(self.indexed_dataset)], dtype=np.int64
            )
            document_index += 1
            if document.size == 0:
                empty_documents_seen += 1
                if empty_documents_seen >= len(self.indexed_dataset):
                    raise ValueError("verification IndexedDataset contains no tokens")
                continue
            empty_documents_seen = 0
            remaining = target_length - total_length
            chunk = document[:remaining]
            chunks.append(chunk)
            total_length += int(chunk.size)
        return np.concatenate(chunks).astype(np.int64, copy=False)

    def __getitem__(self, idx: int) -> np.ndarray:
        length = int(self.sequence_lengths[idx % self.size])
        target_length = length - 1  # The mid-level dataset appends EOD.
        if self.indexed_dataset is not None:
            return self._get_verification_tokens(idx, target_length)
        return np.arange(1, length, dtype=np.int64)


class MockSFTDataset(SFTDataset):
    """Fixed-width mock SFT dataset for correctness and throughput tests."""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: str | None,
        indices: np.ndarray,
        num_samples: int | None,
        index_split: Split,
        config: SFTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: SFTDatasetConfig
    ) -> MockSFTLowLevelDataset:
        """Build a mock source directly from the config's flat typed fields."""
        del dataset_path
        assert config.mock_data_min_sequence_length is not None
        assert config.mock_data_max_sequence_length is not None
        assert config.mock_data_mean_sequence_length is not None
        return MockSFTLowLevelDataset(
            mode=config.mock_data_mode,
            path=config.mock_data_path,
            distribution=config.mock_data_distribution,
            min_sequence_length=config.mock_data_min_sequence_length,
            max_sequence_length=config.mock_data_max_sequence_length,
            mean_sequence_length=config.mock_data_mean_sequence_length,
            lognormal_sigma=config.mock_data_lognormal_sigma,
            seed=config.mock_data_seed,
            size=config.mock_data_size,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        sequence_length = self.config.sequence_length
        eod = tokenizer.eod
        if eod is None:
            raise ValueError("MockSFTDataset requires an EOD/EOS token id")
        pad = _get_padding_token_id(tokenizer, eod)

        raw_tokens = self.dataset[int(self.indices[idx % len(self.indices)])]
        token_ids = _normalize_mock_token_ids(
            raw_tokens, tokenizer, verification=self.dataset.indexed_dataset is not None
        )
        token_ids.append(eod)

        if len(token_ids) > sequence_length:
            token_ids = token_ids[: sequence_length - 1] + [eod]
        valid_length = len(token_ids) - 1
        token_ids.extend([pad] * (sequence_length + 1 - len(token_ids)))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.int64)
        labels = torch.tensor(token_ids[1:], dtype=torch.int64)
        loss_mask = torch.ones(sequence_length, dtype=torch.float32)
        loss_mask[valid_length:] = 0.0
        position_ids = torch.arange(sequence_length, dtype=torch.int64)

        # Match SFTDataset's fixed-width contract from #5454 so default_collate
        # can stack samples with different numbers of packed subsequences.
        cu_seqlens = torch.full((sequence_length + 1,), sequence_length, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens_original = torch.full((sequence_length + 1,), valid_length, dtype=torch.int32)
        cu_seqlens_original[0] = 0
        sample = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': torch.tensor(sequence_length, dtype=torch.int32),
        }
        if self.config.emit_packing_metadata:
            sample['cu_seqlens_original'] = cu_seqlens_original
        return sample
