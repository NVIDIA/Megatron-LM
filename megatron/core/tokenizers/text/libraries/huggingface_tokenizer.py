# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import List, Optional

try:
    from transformers import AutoTokenizer

    HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    HAVE_TRANSFORMERS = False

from .abstract_tokenizer import MegatronTokenizerTextAbstract

logger = logging.getLogger(__name__)


class HuggingFaceTokenizer(MegatronTokenizerTextAbstract):
    """
    Wrapper of HuggingFace AutoTokenizer
        https://huggingface.co/transformers/model_doc/auto.html#autotokenizer.
    """

    def __init__(
        self,
        tokenizer_path: str,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        additional_special_tokens: Optional[List] = [],
        use_fast: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        include_special_tokens: bool = False,
        chat_template: str = None,
    ):
        """
        Args:
            tokenizer_path: corresponds to HuggingFace-AutoTokenizer's
                'pretrained_model_name_or_path' input argument.
                For more details please refer to
                https://huggingface.co/transformers/_modules/transformers/tokenization_auto.html#AutoTokenizer.from_pretrained.
            vocab_file: path to file with vocabulary which consists
                of characters separated by newlines.
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
            additional_special_tokens: list of other tokens beside standard special tokens
                (bos, eos, pad, etc.).
                For example, sentinel tokens for T5 (<extra_id_0>, <extra_id_1>, etc.)
            use_fast: whether to use fast HuggingFace tokenizer
            include_special_tokens: when True, converting text to ids will include special
                tokens / prompt tokens (if any), yielding self.tokenizer(text).input_ids
        """

        try:
            # this logic deals with different huggingface tokenizers having different args
            if vocab_file is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_path,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                    chat_template=chat_template,
                )
            elif merges_file is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_path,
                    vocab_file=vocab_file,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                    chat_template=chat_template,
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_path,
                    vocab_file=vocab_file,
                    merge_files=merges_file,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                    chat_template=chat_template,
                )
        except Exception as e:
            raise ValueError(
                'Unable to instantiate HuggingFace AutoTokenizer '
                f'for {tokenizer_path}. Exception: {e}'
            )

        self.include_special_tokens = include_special_tokens
        self.original_vocab_size = len(self.tokenizer)
        self.chat_template = chat_template
        self.eos_token = eos_token
        special_tokens_dict = {}

        # # setting special tokens, by default the default model's special tokens will be preserved
        # # unless passes new values to the special tokens
        if unk_token is not None:
            special_tokens_dict["unk_token"] = unk_token
        if mask_token is not None:
            special_tokens_dict["mask_token"] = mask_token
        if pad_token is not None:
            special_tokens_dict["pad_token"] = pad_token

        # if the model does not have eos_token but has sep_token,
        if sep_token is not None:
            special_tokens_dict["sep_token"] = sep_token
        if eos_token is not None:
            special_tokens_dict["eos_token"] = eos_token
        elif self.tokenizer.sep_token is None and self.tokenizer.eos_token:
            special_tokens_dict["sep_token"] = self.tokenizer.eos_token
        elif self.tokenizer.eos_token is None and self.tokenizer.sep_token:
            special_tokens_dict["eos_token"] = self.tokenizer.sep_token

        # if the model does not have bos_token but has cls_token,
        # set bos_token = cls_token, and vice versa
        if bos_token is not None:
            special_tokens_dict["bos_token"] = bos_token
        elif self.tokenizer.bos_token is None and self.tokenizer.cls_token:
            special_tokens_dict["bos_token"] = self.tokenizer.cls_token
        if cls_token is not None:
            special_tokens_dict["cls_token"] = cls_token
        elif self.tokenizer.cls_token is None and self.tokenizer.bos_token:
            special_tokens_dict["cls_token"] = self.tokenizer.bos_token

        # add additional special tokens (not standard special tokens such as bos, eod, sep)
        if additional_special_tokens is not None:
            special_tokens_dict["additional_special_tokens"] = additional_special_tokens

        new_tokens_in_vocab = []
        for token in [mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token]:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)
        for token in additional_special_tokens:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)

        if len(new_tokens_in_vocab) > 0:
            """
            Special tokens that were not previously included in the tokenizer's vocabulary file
            will be added to the vocabulary and, as a result, the model should be resized,
            for example:

            # define your model
            tokenizer_path = 'roberta-base'
            tokenizer = MegatronTokenizer.from_pretrained(tokenizer_path=tokenizer_path)

            special_tokens = {'bos_token': '<BOS>',
                              'cls_token': '<CSL>',
                              'additional_special_tokens': ['<MY_NER_TOKEN>', '<ANOTHER_TOKEN>']}
            tokenizer.add_special_tokens(special_tokens_dict=special_tokens)

            # resize your model so that the embeddings for newly added tokens
            tokenizer.resize_token_embeddings(tokenizer_default.vocab_size)
            """

            logger.warning(
                f'{new_tokens_in_vocab} \n will be added to the vocabulary.\n'
                f'Please resize your model accordingly.'
            )
        self.add_special_tokens(special_tokens_dict)
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens(
            'x'
        ) + self.text_to_tokens('y')
        self._inv_vocab_dict = {}

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...).
        If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).

        Args:
            special_tokens_dict: dict of string.
                Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``,
                ``cls_token``, ``mask_token``, ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary.

        Returns:
            Number of tokens added to the vocabulary.
        """

        num_tokens_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_tokens_added > 0:
            logger.info(f'{num_tokens_added} special tokens added, resize your model accordingly.')
        for k in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, k, getattr(self.tokenizer, k, None))
        return num_tokens_added

    @property
    def additional_special_tokens_ids(self):
        """
        Returns a list of the additional special tokens (excluding bos, eos, pad, unk).
        Used to return sentinel tokens for e.g. T5.
        """
        return [self.token_to_id(token) for token in self.additional_special_tokens]

    def text_to_tokens(self, text: str) -> List[str]:
        """Converts text to tokens."""
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[str]) -> str:
        """Converts list of tokens text."""
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def token_to_id(self, token: str) -> int:
        """Converts a single token to it's id."""
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts list of tokens to it's ids."""
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts list of tokens ids to it's token values."""
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to tokens ids."""
        if self.include_special_tokens:
            return self.tokenizer(text).input_ids
        tokens = self.text_to_tokens(text)
        ids = self.tokens_to_ids(tokens)
        return ids

    def ids_to_text(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        """Converts list of ids to text."""
        tokens = self.ids_to_tokens(ids)
        if remove_special_tokens:
            tokens_clean = [t for t in tokens if t not in self.tokenizer.all_special_tokens]
        else:
            tokens_clean = tokens
        text = self.tokens_to_text(tokens_clean)
        return text

    def apply_chat_template(self, conversation, chat_template, **kwargs):
        """Applies chat template and tokenizes results"""
        return self.tokenizer.apply_chat_template(
            conversation=conversation, chat_template=chat_template, **kwargs
        )

    @property
    def vocab(self) -> list:
        """Returns tokenizer vocab values."""
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def inv_vocab(self) -> dict:
        """Returns tokenizer vocab with reversed keys and values."""
        if self._inv_vocab_dict == {}:
            self._inv_vocab_dict = {v: k for k, v in self.tokenizer.vocab.items()}
        return self._inv_vocab_dict

    @property
    def vocab_size(self) -> int:
        """Returns size of tokenizer vocabulary."""
        return len(self.tokenizer)

    @property
    def pad_id(self) -> int:
        """Returns id of padding token."""
        if getattr(self, 'pad_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'pad_token')])[0]

    @property
    def bos_id(self) -> int:
        """Returns id of beggining of sentence token."""
        if getattr(self, 'bos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'bos_token')])[0]

    @property
    def eos_id(self) -> int:
        """Returns id of end of sentence token."""
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def eod(self) -> int:
        """Returns EOD token id."""
        if getattr(self, 'eos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def sep_id(self) -> int:
        """Returns id of SEP token."""
        if getattr(self, 'sep_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'sep_token')])[0]

    @property
    def cls_id(self) -> int:
        """Returns id of classification token."""
        if getattr(self, 'cls_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'cls_token')])[0]

    @property
    def unk_id(self) -> int:
        """Returns id of unknown tokens."""
        if getattr(self, 'unk_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'unk_token')])[0]

    @property
    def mask_id(self) -> int:
        """Returns id of mask token."""
        if getattr(self, 'mask_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'mask_token')])[0]

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self.tokenizer.save_vocabulary(
            save_directory=save_directory, filename_prefix=filename_prefix
        )

    def save_pretrained(self, save_directory: str):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self.tokenizer.save_pretrained(save_directory)
