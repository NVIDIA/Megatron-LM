# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy
import torch
from packaging.version import Version as PkgVersion

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.masked_dataset import (
    MaskedWordPieceDataset,
    MaskedWordPieceDatasetConfig,
)
from megatron.core.datasets.utils import Split
from megatron.core.utils import get_te_version


@dataclass
class T5MaskedWordPieceDatasetConfig(MaskedWordPieceDatasetConfig):
    """Configuration object for Megatron Core T5 WordPiece datasets

    NB: As a temporary holdover from Megatron-LM. The T5 tokenizer has an attribute which defines
    a number of special sentinel tokens used during sampling. The assert in __post_init__ serves to
    preserve compatibility with Megatron-LM until the T5 tokenizer is in Megatron Core.
    """

    sequence_length_encoder: Optional[int] = field(init=False, default=None)
    """A sequence_length alias and the sequence length for the encoder"""

    sequence_length_decoder: int = None
    """The sequence length for the decoder"""

    def __post_init__(self) -> None:
        """Do asserts and set fields post init"""
        super().__post_init__()

        self.sequence_length_encoder = self.sequence_length

        assert self.sequence_length_encoder is not None
        assert self.sequence_length_decoder is not None

        assert len(self.tokenizer.additional_special_tokens_ids) > 0


class T5MaskedWordPieceDataset(MaskedWordPieceDataset):
    """The T5 dataset that assumes WordPiece tokenization

    Args:
        indexed_dataset (IndexedDataset): The IndexedDataset around
            which to build the MegatronDataset

        dataset_path (str): The real path on disk to the dataset, for bookkeeping

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The number of samples to draw from the indexed
            dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indexed_indices Split

        config (T5MaskedWordPieceDatasetConfig): The config
    """

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: str,
        indexed_indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: T5MaskedWordPieceDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )

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
        )._key_config_attributes() + ["sequence_length_decoder"]

    @staticmethod
    def _build_b1ss_attention_mask(
        source_block: torch.tensor, target_block: torch.tensor, make_history_mask: bool = False
    ) -> torch.tensor:
        """Build an attention-mask having shape (bs, 1, q_len, kv_len)
        from source_block and target_block

        Args:
            source_block (torch.tensor): A 2-D array of tokens (bs, q_len)
            target_block (torch.tensor): A 2-D array of tokens (bs, kv_len)
            make_history_mask (bool): Whether to turn mask into causal mask

        Returns:
            torch.tensor: The 4-D attention mask (bs, 1, q_len, kv_len)
        """
        batch_size = source_block.shape[0]
        attention_mask = []
        for i in range(batch_size):
            source_sample = source_block[i]
            target_sample = target_block[i]
            mask = (target_sample[None, :] >= 1) * (source_sample[:, None] >= 1)
            if make_history_mask:
                arange = numpy.arange(source_sample.shape[0])
                history_mask = arange[None,] <= arange[:, None]
                history_mask = torch.tensor(history_mask).to(mask.device)
                mask = mask * history_mask
            mask = ~(mask)  # flip True to False
            attention_mask.append(mask)
        attention_mask = torch.stack(attention_mask)
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    @staticmethod
    def config_attention_mask(
        encoder_tokens: torch.tensor,
        decoder_tokens: torch.tensor,
        encoder_mask: torch.tensor,
        decoder_mask: torch.tensor,
        use_local: bool = False,
        test_te_version: str = None,
    ) -> torch.tensor:
        """Config attention-mask for encoder_mask, decoder_mask, encoder_decoder_mask
        conditioned on transformer-implementation (e.g. TE vs local), TE versions,
        and TE backends

        Args:
            encoder_tokens (torch.tensor): A 2-D array of tokens (bs, kv_len)
            decoder_tokens (torch.tensor): A 2-D array of tokens (bs, q_len)
            encoder_mask (torch.tensor): A 2-D array of tokens (bs, kv_len)
            decoder_mask (torch.tensor): A 2-D array of tokens (bs, q_len)
            use_local (bool): Whether the current T5 model uses local (vs TE)
                transformer implmentation

        Returns:
            Configured encoder_mask, decoder_mask, encoder_decoder_mask
            torch.tensor: configured encoder attention mask
            torch.tensor: configured decoder attention mask
            torch.tensor: configured encoder-decoder attention mask
        """
        # If using local transformer implementation (not transformer_engine):
        # re-organize all attention masks, because local and transformer_engine
        # backbones use different masks shapes. E.g.:
        # (local: b1ss - transformer_engine: b11s)
        if use_local:
            encoder_mask = T5MaskedWordPieceDataset._build_b1ss_attention_mask(
                encoder_tokens, encoder_tokens
            )
            decoder_mask = T5MaskedWordPieceDataset._build_b1ss_attention_mask(
                decoder_tokens, decoder_tokens, make_history_mask=True
            )
            encoder_decoder_mask = T5MaskedWordPieceDataset._build_b1ss_attention_mask(
                decoder_tokens, encoder_tokens
            )

        else:
            # If using transformer_engine transformer implementation:
            # 1. For TE version >= 1.10, across all 3 backends,
            #    The padding mask is configued as
            #    [bs, 1, 1, seq_len] for self-attention and
            #    ([bs, 1, 1, q_len], [bs, 1, 1, kv_len]) for cross-attention
            # 2. For TE version >=1.7 and <1.10, when using Non-fused backend,
            #    The padding mask is configued as
            #    [bs, 1, q_len, kv_len] for both self-attention and for cross-attention
            # 3. For TE version <1.7, only support Non-fused backend
            #    The padding mask is configued as
            #    [bs, 1, q_len, kv_len] for both self-attention and for cross-attention

            # Process for Flash/Fused
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(1)
            decoder_mask = decoder_mask.unsqueeze(1).unsqueeze(1)
            encoder_decoder_mask = (decoder_mask, encoder_mask)
            # set decoder_mask to None because decoder uses AttnMaskType.causal
            decoder_mask = None

            # get TE version, using test TE version if not None
            if test_te_version is not None:
                te_version = PkgVersion(test_te_version)
            else:
                te_version = get_te_version()

            # Check for older TE version than 1.10, adjust attention mask accordingly
            flash_attention_enabled = os.getenv('NVTE_FLASH_ATTN') == '1'
            fused_attention_enabled = os.getenv('NVTE_FUSED_ATTN') == '1'
            if (te_version < PkgVersion("1.10.0")) and (te_version >= PkgVersion("1.7.0")):
                if not (flash_attention_enabled) and not (fused_attention_enabled):
                    encoder_mask = T5MaskedWordPieceDataset._build_b1ss_attention_mask(
                        encoder_tokens, encoder_tokens
                    )
                    encoder_decoder_mask = T5MaskedWordPieceDataset._build_b1ss_attention_mask(
                        decoder_tokens, encoder_tokens
                    )
                else:
                    pass
            elif te_version < PkgVersion("1.7.0"):
                if not (flash_attention_enabled) and not (fused_attention_enabled):
                    encoder_mask = T5MaskedWordPieceDataset._build_b1ss_attention_mask(
                        encoder_tokens, encoder_tokens
                    )
                    encoder_decoder_mask = T5MaskedWordPieceDataset._build_b1ss_attention_mask(
                        decoder_tokens, encoder_tokens
                    )
                else:
                    assert not flash_attention_enabled and not fused_attention_enabled, (
                        "Flash and fused attention is not supported with transformer "
                        "engine version < 1.7. Set NVTE_FLASH_ATTN=0 and NVTE_FUSED_ATTN=0"
                        "or upgrade transformer engine >= 1.7"
                    )
        return encoder_mask, decoder_mask, encoder_decoder_mask

    def __getitem__(self, idx: int) -> Dict[str, Union[int, numpy.ndarray]]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, Union[int, numpy.ndarray]]: The
        """
        idx_beg, idx_end, target_sequence_length = self.sample_index[idx]
        sample = [self.dataset[i] for i in range(idx_beg, idx_end)]

        numpy_random_state = numpy.random.RandomState(seed=(self.config.random_seed + idx) % 2**32)

        assert target_sequence_length <= self.config.sequence_length

        # Flatten the sample into a list of tokens
        tokens = [token for sentence in sample for token in sentence]

        # Truncate the list of tokens to a desired length
        truncated = len(tokens) > target_sequence_length
        tokens = tokens[:target_sequence_length]

        # Masking
        (tokens, _, _, _, masked_spans) = self._create_masked_lm_predictions(
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
        mask_encoder = numpy.array([1] * length_toks_encoder + [0] * length_pads_encoder)
        mask_decoder = numpy.array([1] * length_toks_decoder + [0] * length_pads_decoder)
        mask_encoder_decoder = None

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
        }

    def _get_token_mask(self, numpy_random_state: numpy.random.RandomState) -> int:
        """Abstract method implementation

        100% of the time, replace the token id with mask token id.

        Args:
            numpy_random_state (RandomState): The NumPy random state

        Returns:
            int: The mask token id
        """
        return self.config.tokenizer.mask
