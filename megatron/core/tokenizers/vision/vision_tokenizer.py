# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict
from typing import Dict, List, Union

from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase

TOKENIZER_MAPPING_LIBRARIES = OrderedDict(
    [
        ("multimodal", "MegatronMultimodalTokenizer"),
        ("null-multimodal", "MegatronNullMultimodalTokenizer"),
    ]
)


class MegatronTokenizerVision(MegatronTokenizerBase):
    """Base class for Megatron vision tokenizers."""

    def __init__(self, path: str, config: dict, **kwargs) -> None:
        """
        Args:
            path (str): path to the tokenizer model.
            config (dict): tokenizer parameters.
                library (str): tokenizer library.
                class_name (str): name of tokenizer class.
                class_path (str): path to tokenizer class.
                model_type (str): type of the model to be used with tokenizer.
        """

        super().__init__(path, config, **kwargs)
        self._tokenizer = self._restore_model(**kwargs)
        self.path = path

    def _restore_model(self, **kwargs):
        """Returns tokenizer library object."""

        import megatron.core.tokenizers.vision.libraries as tokenizers

        library_class = getattr(tokenizers, TOKENIZER_MAPPING_LIBRARIES[self.library])

        if self.library in ['null-multimodal']:
            return library_class(**kwargs)
        else:
            return library_class(self.path, **kwargs)

    def tokenize(self, text: Union[str, List[Dict]]) -> List[int]:
        """
        Text tokenization.

        Args:
            text (str | list): text to be tokenized.

        Returns:
            list: list of ids.
        """

        return self._tokenizer.tokenize(text)

    def detokenize(self, ids: List[int]) -> str:
        """
        Text detokenization.

        Args:
            ids (list): text to be tokenized.

        Returns:
            text: detokenized text.
        """

        return self._tokenizer.detokenize(ids)

    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "user", "content": "something"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """

        return self._tokenizer.tokenize_conversation(
            conversation=conversation,
            return_target=return_target,
            add_generation_prompt=add_generation_prompt,
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

    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def apply_chat_template(self):
        """Applies tokenizer's chat template."""
        raise NotImplementedError("This method is not supported for vision tokenizers.")

    def get_special_tokens(self) -> list:
        """Returns a list of the additional special tokens."""
        return self._tokenizer.get_special_tokens()

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Calculate offsets."""
        return self._tokenizer.offsets(ids=ids, text=text)

    @property
    def vocab(self):
        """Tokenizer vocab."""
        return self._tokenizer.vocab

    @property
    def vocab_size(self) -> int:
        """Returns vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def pad(self):
        """Pad token ID."""
        return self._tokenizer.pad

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eod
