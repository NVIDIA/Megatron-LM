# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict
from typing import Dict, List, Optional, Union

from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase
from megatron.core.tokenizers.text.libraries.abstract_tokenizer import MegatronTokenizerTextAbstract

TOKENIZER_MAPPING_LIBRARIES = OrderedDict(
    [
        ("sentencepiece", "SentencePieceTokenizer"),
        ("huggingface", "HuggingFaceTokenizer"),
        ("megatron", "MegatronHFTokenizer"),
        ("tiktoken", "TikTokenTokenizer"),
        ("byte-level", "ByteLevelTokenizer"),
        ("null", "NullTokenizer"),
    ]
)


class MegatronTokenizerText(MegatronTokenizerBase):
    """Base class for Megatron text tokenizers."""

    def __init__(self, path: str, config: dict, **kwargs) -> None:
        """
        Args:
            path (str): path to the tokenizer model.
            config (dict): tokenizer parameters.
                library (str): tokenizer library.
                class_name (str): name of tokenizer class.
                class_path (str): path to tokenizer class.
                model_type (str): type of the model to be used with tokenizer.
                chat_template (str): tokenizer chat template.
        """

        super().__init__(path, config, **kwargs)
        self._tokenizer = self._restore_model(**kwargs)
        self.additional_args = kwargs
        self.path = path
        if (
            config.get("chat_template", None) is None
            and kwargs.get("chat_template", None) is not None
        ):
            self.chat_template = kwargs.get("chat_template", None)
        else:
            self.chat_template = config.get("chat_template", None)

    def _restore_model(self, **kwargs) -> MegatronTokenizerTextAbstract:
        """Returns tokenizer library object."""

        import megatron.core.tokenizers.text.libraries as tokenizers

        library_class = getattr(tokenizers, TOKENIZER_MAPPING_LIBRARIES[self.library])

        if self.library in ['byte-level', 'null']:
            return library_class(**kwargs)
        else:
            return library_class(self.path, **kwargs)

    def tokenize(self, text: str) -> List[int]:
        """
        Text tokenization.

        Args:
            text (str): text to be tokenized.

        Returns:
            list: list of ids.
        """

        return self._tokenizer.text_to_ids(text)

    def detokenize(self, ids: List[int]) -> str:
        """
        Text detokenization.

        Args:
            ids (list): text to be tokenized.

        Returns:
            text: dettokenized text.
        """

        return self._tokenizer.ids_to_text(ids)

    def apply_chat_template(
        self, conversation: List[Dict[str, str]], chat_template: Optional[str] = None, **kwargs
    ) -> Union[str, list]:
        """
        Applies chat template to the conversation.

        Args:
            conversation (list):
            chat_template (Optional[str]): chat template to be use. If not specified,
                tokenizer's chat template will be used.

        Returns:
            Union[str, list]: a chat with applied chat template or a list of token ids.
        """

        # Use tokenizer's chat template if chat template wasn't specified.
        if not chat_template:
            assert self.chat_template is not None, "`chat_template` was not specified."

            chat_template = self.chat_template

        return self._tokenizer.apply_chat_template(
            conversation=conversation, chat_template=chat_template, **kwargs
        )

    def save_pretrained(self, path: str) -> None:
        """
        Saves HF tokenizer files.

        Args:
            path (str): path where to save tokenizer files.
        """

        if self.library in ['huggingface', 'megatron']:
            self._tokenizer.save_pretrained(path)
        else:
            raise ValueError(
                f"save_pretrained method is not supported with {self.library} library."
            )

    def add_special_tokens(self, special_tokens: Union[list, dict]) -> None:
        """
        Adds a dictionary of special tokens (eos, pad, cls...).
            Tokens are only added if they are not already in the vocabulary.
            Indexed starting from the last index of the current vocabulary.

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined
                special attributes: [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``,
                ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``].
        """

        self._tokenizer.add_special_tokens(special_tokens)

    @property
    def additional_special_tokens_ids(self) -> list:
        """Returns a list of the additional special tokens."""
        return self._tokenizer.additional_special_tokens_ids

    @property
    def vocab_size(self) -> int:
        """Returns vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def vocab(self):
        """Returns tokenizer vocabulary."""
        return self._tokenizer.vocab

    @property
    def unique_identifiers(self) -> OrderedDict:
        """Returns a dictionary of unique identifiers."""
        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = f"{type(self).__module__}.{type(self).__qualname__}"
        unique_identifiers["tokenizer_path"] = self.path
        for arg in self.additional_args:
            unique_identifiers[arg] = str(self.additional_args[arg])

        return unique_identifiers

    @property
    def pad(self) -> int:
        """Returns id of padding token."""
        return self._tokenizer.pad_id

    @property
    def pad_id(self) -> int:
        """Returns id of padding token. Need for NeMo."""
        return self._tokenizer.pad_id

    @property
    def eod(self) -> int:
        """Returns id of end of document token."""
        return self._tokenizer.eod

    @property
    def bos(self) -> int:
        """Returns id of beginning of sentence token."""
        return self._tokenizer.bos_id

    @property
    def bos_id(self) -> int:
        """Returns id of beginning of sentence token. Need for NeMo."""
        return self._tokenizer.bos_id

    @property
    def eos_id(self) -> int:
        """Returns id of end of sentence token."""
        return self._tokenizer.eos_id

    @property
    def eos(self) -> int:
        """Returns id of end of sentence token. Need for legacy."""
        return self._tokenizer.eos_id

    @property
    def unk(self) -> int:
        """Returns id of of unknown token."""
        return self._tokenizer.unk_id

    @property
    def unk_id(self) -> int:
        """Returns id of of unknown token. Need for NeMo."""
        return self._tokenizer.unk_id

    @property
    def mask(self) -> int:
        """Returns id of of mask token."""
        return self._tokenizer.mask_id

    @property
    def mask_id(self) -> int:
        """Returns id of of mask token. Need for NeMo."""
        return self._tokenizer.mask_id

    @property
    def cls(self) -> int:
        """Returns id of classification token."""
        return self._tokenizer.cls_id

    @property
    def cls_id(self) -> int:
        """Returns id of classification token. Need for NeMo."""
        return self._tokenizer.cls_id

    @property
    def sep(self) -> int:
        """Returns id of SEP token."""
        return self._tokenizer.sep_id

    @property
    def sep_id(self) -> int:
        """Returns id of SEP token. Need for NeMo."""
        return self._tokenizer.sep_id

    @property
    def vocab_file(self) -> str:
        """Returns vocabulary file path if specified."""
        return self.additional_args.get('vocab_file', None)

    @property
    def merges_file(self) -> str:
        """Returns merges file path if specified."""
        return self.additional_args.get('merges_file', None)

    @property
    def inv_vocab(self) -> dict:
        """Returns tokenizer vocab with reversed keys and values."""
        return self._tokenizer.inv_vocab
