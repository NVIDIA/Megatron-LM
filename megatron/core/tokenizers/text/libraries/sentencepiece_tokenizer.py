# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import re
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import sentencepiece

    HAVE_SP = True
except ModuleNotFoundError:
    HAVE_SP = False

import torch

from .abstract_tokenizer import MegatronTokenizerTextAbstract
from .chat_template import MegatronTokenizerChatTemplate


class SentencePieceTokenizer(MegatronTokenizerTextAbstract, MegatronTokenizerChatTemplate):
    """Sentencepiecetokenizer https://github.com/google/sentencepiece."""

    def __init__(
        self,
        tokenizer_path: str,
        special_tokens: Optional[Union[Dict[str, str], List[str]]] = None,
        legacy: bool = False,
        ignore_extra_whitespaces: bool = True,
        chat_template: Optional[str] = None,
        trim_spm_separator_after_special_token=True,
        spm_separator='▁',
    ) -> None:
        """
        Args:
            tokenizer_path (str): path to sentence piece tokenizer model.
            special_tokens (Optional[Union[Dict[str, str], List[str]]]):
                either list of special tokens or dictionary of token name to token value
            legacy (bool): when set to True, the previous behavior of the SentecePiece wrapper
                will be restored, including the possibility to add special tokens inside wrapper.
            ignore_extra_whitespaces (bool): whether to ignore extra whitespaces in the
                input text while encoding.
                Note:
                    This is done for the current models tokenizers that don't handle extra
                    whitespaces as by default tokenizer learned to ignore it.
                    To check if the tokenizer by default ignores extra whitespaces refer to
                    `self.removed_extra_spaces` attribute of the tokenizer.
                    We added a parameter to process_asr_tokenizer.py for upcoming models to
                    handle it inbuilt.
            chat_template (Optional[str]): tokenizer chat template in jinja format.
        """

        self.chat_template = chat_template
        if not tokenizer_path or not os.path.exists(tokenizer_path):
            raise ValueError(f"tokenizer_path: {tokenizer_path} is invalid")

        if HAVE_SP:
            self.tokenizer = sentencepiece.SentencePieceProcessor()
        else:
            raise ModuleNotFoundError("sentencepiece library should be installed.")

        self.tokenizer.Load(tokenizer_path)

        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.legacy = legacy
        self.ignore_extra_whitespaces = ignore_extra_whitespaces
        # using special symbol for extra_space token, so it is not likely to be in the vocabulary
        self.extra_space_token = '☯'
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        self.trim_spm_separator_after_special_token = trim_spm_separator_after_special_token
        self.spm_separator_id = self.tokenizer.piece_to_id(spm_separator)
        self.spm_separator = spm_separator

        if special_tokens:
            if not self.legacy:
                raise ValueError(
                    "Special tokens must be None when legacy is set to False. "
                    "Provide special tokens at train time."
                )
            self.add_special_tokens(special_tokens)

        self.removed_extra_spaces = self.tokenizer.encode_as_pieces(
            'x  y'
        ) == self.tokenizer.encode_as_pieces('x y')
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens(
            'x'
        ) + self.text_to_tokens('y')

    def text_to_tokens(self, text: str) -> List[str]:
        """Converts text to tokens."""
        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            text = re.sub(r'(?<= )(?= )|^ | $', f' {self.extra_space_token} ', text)
        if self.legacy:
            tokens = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                tok = self.tokenizer.encode_as_pieces(text[idx:next_idx])
                # Chat-templates insert a space between a special token and first word (e.g.
                # "[INST] who") which is tokenized as <inst-id> <space-id> <who-id> instead of
                # <inst-id> <who-id>.
                if (
                    self.trim_spm_separator_after_special_token
                    and len(tokens) > 0
                    and tokens[-1] in self.special_token_to_id
                    and len(tok) > 0
                    and tok[0] == self.spm_separator
                ):
                    tok.pop(0)
                tokens.extend(tok)
                tokens.append(next_token)
                idx = next_idx + len(next_token)

            tokens.extend(self.tokenizer.encode_as_pieces(text[idx:]))

        else:
            tokens = self.tokenizer.encode_as_pieces(text)

        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            tokens = list(filter(lambda x: x != self.extra_space_token, tokens))
        return tokens

    def text_to_ids(self, text, sample_alpha=None) -> List[int]:
        """Converts text to tokens ids."""
        if isinstance(text, str):
            return self._text_to_ids(text, sample_alpha)
        else:
            raise ValueError(f"Expected str input, but got {type(text)}")

    def _text_to_ids(self, text, sample_alpha=None) -> List[int]:
        """Converts text to tokens ids."""
        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            text = re.sub(r'(?<= )(?= )|^ | $', f' {self.extra_space_token} ', text).rstrip()
        if self.legacy:
            ids = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                text_tokens = self.tokenizer.encode(text[idx:next_idx])
                # Chat-templates insert a space between a special token and first word (e.g.
                # "[INST] who") which is tokenized as <inst-id> <space-id> <who-id> instead of
                # <inst-id> <who-id>.
                if (
                    self.trim_spm_separator_after_special_token
                    and len(ids) > 0
                    and ids[-1] in self.id_to_special_token
                    and len(text_tokens) > 0
                    and text_tokens[0] == self.spm_separator_id
                ):
                    text_tokens.pop(0)
                ids.extend(text_tokens)
                ids.append(self.special_token_to_id[next_token])
                idx = next_idx + len(next_token)

            if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
                ids.extend(self._text_to_ids_extra_space(text[idx:]))
            else:
                ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
            return ids

        if self.removed_extra_spaces and not self.ignore_extra_whitespaces:
            return self._text_to_ids_extra_space(text, sample_alpha)

        if sample_alpha is not None:
            return self.tokenizer.encode_as_ids(
                text, enable_sampling=True, alpha=sample_alpha, nbest_size=-1
            )
        else:
            return self.tokenizer.encode_as_ids(text)

    def _text_to_ids_extra_space(self, text, sample_alpha=None) -> List[int]:
        """Converts text to tokens ids."""
        ids = []
        encoding_kwargs = {}
        if sample_alpha is not None:
            encoding_kwargs = {'enable_sampling': True, 'alpha': sample_alpha, 'nbest_size': -1}
        for part in text.split(self.extra_space_token):
            if not part:
                continue
            part += self.extra_space_token
            part_ids = self.tokenizer.encode_as_ids(part, **encoding_kwargs)
            ids.extend(part_ids[:-1])

        return ids

    def tokens_to_text(self, tokens: List[str]) -> str:
        """Converts list of tokens text."""
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        return self.tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids: List[int]) -> str:
        """Converts list of ids to text."""
        if isinstance(ids, (np.ndarray, torch.Tensor)):
            ids = ids.tolist()

        if self.legacy:
            text = ""
            last_i = 0

            for i, id in enumerate(ids):
                if id in self.id_to_special_token:
                    text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                    text += self.id_to_special_token[id] + " "
                    last_i = i + 1

            text += self.tokenizer.decode_ids(ids[last_i:])
            return text.strip()

        return self.tokenizer.decode_ids(ids)

    def token_to_id(self, token: str) -> int:
        """Converts a single token to it's id."""
        if self.legacy and token in self.special_token_to_id:
            return self.special_token_to_id[token]

        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts list of tokens ids to it's token values."""
        tokens = []
        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.id_to_special_token[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))
        return tokens

    def tokens_to_ids(
        self, tokens: Union[str, List[str]], tokens_to_skip: List[str] = []
    ) -> List[int]:
        """Converts list of tokens to it's ids."""
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            if token not in tokens_to_skip:
                ids.append(self.token_to_id(token))
        return ids

    def add_special_tokens(self, special_tokens: Union[list, dict]) -> None:
        """Adds special tokens to the tokenizer."""
        if not self.legacy:
            raise AttributeError(
                "Special Token addition does not work when legacy is set to False."
            )

        if isinstance(special_tokens, list):
            for token in special_tokens:
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
                elif self.tokenizer.piece_to_id(token) != self.tokenizer.unk_id():
                    self.special_token_to_id[token] = self.tokenizer.piece_to_id(token)
                    self.id_to_special_token[self.special_token_to_id[token]] = token

        elif isinstance(special_tokens, dict):
            for token_name, token in special_tokens.items():
                setattr(self, token_name, token)
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
                elif self.tokenizer.piece_to_id(token) != self.tokenizer.unk_id():
                    self.special_token_to_id[token] = self.tokenizer.piece_to_id(token)
                    self.id_to_special_token[self.special_token_to_id[token]] = token
        else:
            raise ValueError(
                "Expected special_tokens to be a list or a dict " + str(type(special_tokens))
            )

    @property
    def pad_id(self) -> int:
        """Returns id of padding token."""
        if self.legacy:
            pad_id = self.tokens_to_ids([self.pad_token])[0]
        else:
            pad_id = self.tokenizer.pad_id()
        return pad_id

    @property
    def bos_id(self) -> int:
        """Returns id of begginning of sentence token."""
        if self.legacy:
            bos_id = self.tokens_to_ids([self.bos_token])[0]
        else:
            bos_id = self.tokenizer.bos_id()
        return bos_id

    @property
    def eos_id(self) -> int:
        """Returns id of end of sentence token."""
        if self.legacy:
            eos_id = self.tokens_to_ids([self.eos_token])[0]
        else:
            eos_id = self.tokenizer.eos_id()
        return eos_id

    @property
    def sep_id(self) -> int:
        """Returns id of end of SEP token."""
        if self.legacy:
            return self.tokens_to_ids([self.sep_token])[0]
        else:
            raise NameError(
                "Use function token_to_id to retrieve special tokens other than "
                "unk, pad, bos, and eos."
            )

    @property
    def cls_id(self) -> int:
        """Returns id of classification token."""
        if self.legacy:
            return self.tokens_to_ids([self.cls_token])[0]
        else:
            raise NameError(
                "Use function token_to_id to retrieve special tokens other than "
                "unk, pad, bos, and eos."
            )

    @property
    def mask_id(self) -> int:
        """Returns id of mask token."""
        if self.legacy:
            return self.tokens_to_ids([self.mask_token])[0]
        else:
            raise NameError(
                "Use function token_to_id to retrieve special tokens other than "
                "unk, pad, bos, and eos."
            )

    @property
    def unk_id(self) -> int:
        """Returns id of unknown tokens."""
        return self.tokenizer.unk_id()

    @property
    def additional_special_tokens_ids(self) -> list:
        """
        Returns a list of the additional special tokens (excluding bos, eos, pad, unk).
        Used to return sentinel tokens for e.g. T5.
        """
        special_tokens = set(
            [
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.mask_token,
                self.cls_token,
                self.sep_token,
            ]
        )
        return [v for k, v in self.special_token_to_id.items() if k not in special_tokens]

    @property
    def vocab(self) -> list:
        """Returns tokenizer's vocabulary."""
        main_vocab = [
            self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())
        ]
        special_tokens = [
            self.id_to_special_token[self.original_vocab_size + i]
            for i in range(self.vocab_size - self.original_vocab_size)
        ]
        return main_vocab + special_tokens

    @property
    def inv_vocab(self) -> dict:
        """Returns tokenizer vocab with reversed keys and values."""
        return {id: token for id, token in enumerate(self.vocab)}
