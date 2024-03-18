# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.masked_dataset import (
    MaskedWordPieceDataset,
    MaskedWordPieceDatasetConfig,
)
from megatron.core.datasets.utils import Split


@dataclass
class T5MaskedWordPieceDatasetConfig(MaskedWordPieceDatasetConfig):
    """Configuration object for Megatron Core T5 WordPiece datasets

    NB: As a temporary holdover from Megatron-LM. The T5 tokenizer has an attribute which defines
    a number of special sentinel tokens used during sampling. The assert in __post_init__ serves to
    preserve compatibility with Megatron-LM until the T5 tokenizer is in Megatron Core.

    Args:
        sequence_length_encoder (Optional[int]): A sequence_length alias and the sequence length for the encoder

        sequence_length_decoder (int): The sequence length for the decoder
    """

    sequence_length_encoder: Optional[int] = field(init=False, default=None)

    sequence_length_decoder: int = None

    def __post_init__(self) -> None:
        """Do asserts and set fields post init
        """
        super().__post_init__()

        self.sequence_length_encoder = self.sequence_length

        assert self.sequence_length_encoder is not None
        assert self.sequence_length_decoder is not None

        assert len(self.tokenizer.additional_special_tokens_ids) > 0


class T5MaskedWordPieceDataset(MaskedWordPieceDataset):
    """The T5 dataset that assumes WordPiece tokenization

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around which to build the MegatronDataset

        dataset_path (str): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (T5MaskedWordPieceDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: str,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: T5MaskedWordPieceDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

    def _finalize(self) -> None:
        """Abstract method implementation
        """
        self.token_lookup = list(self.config.tokenizer.inv_vocab.keys())
        # Account for the single <bos> and single <eos> token ids
        self.sample_index = self._build_sample_index(self.config.sequence_length - 2, 1)

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Inherited method implementation

        Returns:
            List[str]: The key config attributes
        """
        return super(
            T5MaskedWordPieceDataset, T5MaskedWordPieceDataset
        )._key_config_attributes() + ["sequence_length_decoder",]

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        """Abstract method implementation
 
        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, Union[int, numpy.ndarray]]: The 
        """
        idx_beg, idx_end, target_sequence_length = self.sample_index[idx]
        sample = [self.dataset[i] for i in range(idx_beg, idx_end)]

        numpy_random_state = numpy.random.RandomState(
            seed=(self.config.random_seed + idx) % 2 ** 32
        )

        assert target_sequence_length <= self.config.sequence_length

        # Flatten the sample into a list of tokens
        tokens = [token for sentence in sample for token in sentence]

        # Truncate the list of tokens to a desired length
        truncated = len(tokens) > target_sequence_length
        tokens = tokens[:target_sequence_length]

        # Masking
        (tokens, _, _, _, masked_spans,) = self._create_masked_lm_predictions(
            tokens, target_sequence_length, numpy_random_state
        )

        # Prepare the encoder input and decoder input and output
        sentinels = deque(self.config.tokenizer.additional_special_tokens_ids)
        encoder_input = []
        decoder_input = [self.config.tokenizer.bos]
        decoder_output = []
        idx_beg = 0
        for indices, labels in masked_spans:
            sentinel = sentinels.popleft()

            # set the end index
            idx_end = indices[0]

            encoder_input.extend(tokens[idx_beg:idx_end])
            encoder_input.append(sentinel)

            decoder_input.append(sentinel)
            decoder_input.extend(labels)

            decoder_output.append(sentinel)
            decoder_output.extend(labels)

            # set the start index
            idx_beg = indices[-1] + 1

        encoder_input.extend(tokens[idx_beg:])
        decoder_output.append(self.config.tokenizer.eos)

        # Pad the sequences and convert to NumPy
        length_toks_encoder = len(encoder_input)
        length_toks_decoder = len(decoder_input)
        length_pads_encoder = self.config.sequence_length_encoder - length_toks_encoder
        length_pads_decoder = self.config.sequence_length_decoder - length_toks_decoder
        assert length_pads_encoder >= 0
        assert length_pads_decoder >= 0

        encoder_input = numpy.array(encoder_input, dtype=numpy.int64)
        encoder_input = numpy.pad(
            encoder_input, (0, length_pads_encoder), constant_values=self.config.tokenizer.pad
        )

        decoder_input = numpy.array(decoder_input, dtype=numpy.int64)
        decoder_input = numpy.pad(
            decoder_input, (0, length_pads_decoder), constant_values=self.config.tokenizer.pad
        )

        # Create attention and history masks
        mask_encoder = self._make_attention_mask(encoder_input, encoder_input)
        mask_encoder_decoder = self._make_attention_mask(decoder_input, encoder_input)
        mask_decoder = self._make_attention_mask(decoder_input, decoder_input)
        mask_decoder = mask_decoder * self._make_history_mask(decoder_input)

        # Mask the labels
        decoder_output = numpy.array(decoder_output, dtype=numpy.int64)
        decoder_output = numpy.pad(decoder_output, (0, length_pads_decoder), constant_values=-1)

        # Get the loss mask
        loss_mask = numpy.zeros(self.config.sequence_length_decoder, dtype=numpy.int64)
        loss_mask[:length_toks_decoder] = 1

        return {
            "text_enc": encoder_input,
            "text_dec": decoder_input,
            "labels": decoder_output,
            "loss_mask": loss_mask,
            "truncated": int(truncated),
            "enc_mask": mask_encoder,
            "dec_mask": mask_decoder,
            "enc_dec_mask": mask_encoder_decoder,
        }

    @staticmethod
    def _make_attention_mask(
        source_block: numpy.ndarray, target_block: numpy.ndarray
    ) -> numpy.ndarray:
        """Return a 2-D attention mask

        Args:
            source_block (numpy.ndarray): A 1-D array
            target_block (numpy.ndarray): A 1-D array

        Returns:
            numpy.ndarray: The 2-D attention mask
        """
        mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
        return mask.astype(numpy.int64)

    @staticmethod
    def _make_history_mask(block: numpy.ndarray) -> numpy.ndarray:
        """Return a 2-D history (lower-left-triangular) mask

        Args:
            block (numpy.ndarray): A 1-D array

        Returns:
            numpy.ndarray: The 2-D history (lower-left-triangular) mask
        """
        arange = numpy.arange(block.shape[0])
        mask = arange[None,] <= arange[:, None]
        return mask.astype(numpy.int64)

    def _get_token_mask(self, numpy_random_state: numpy.random.RandomState) -> int:
        """Abstract method implementation

        100% of the time, replace the token id with mask token id.

        Args:
            numpy_random_state (RandomState): The NumPy random state

        Returns:
            int: The mask token id
        """
        return self.config.tokenizer.mask
