# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import List


class MegatronTokenizerTextAbstract(ABC):
    """
    Abstract class for Megatron text tokenizers.
    """

    @abstractmethod
    def text_to_tokens(self, text: str) -> List[str]:
        """
        Converts text to tokens.

        Args:
            text (str): text to be tokenized.

        Returns:
            List[str]: list of tokens.
        """
        pass

    @abstractmethod
    def tokens_to_text(self, tokens: List[str]) -> str:
        """
        Converts tokens to text.

        Args:
            tokens (List[str]): tokens to be detokenized.

        Returns:
            str: detokenized text.
        """
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Converts tokens to ids.

        Args:
            tokens (List[str]): tokens to be converted.

        Returns:
            List[int]: ids of tokens.
        """
        pass

    @abstractmethod
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Converts ids to tokens.

        Args:
            ids (List[int]): ids to be converted.

        Returns:
            List[str]: list of tokens.
        """
        pass

    @abstractmethod
    def text_to_ids(self, text: str) -> List[int]:
        """
        Converts text to ids.

        Args:
            text (str): text to be tokenized.

        Returns:
            List[int]: list of ids.
        """
        pass

    @abstractmethod
    def ids_to_text(self, ids: List[int]) -> str:
        """
        Converts ids to text.

        Args:
            ids (List[int]): ids to be detokenized.

        Returns:
            str: detokenized text.
        """
        pass

    @abstractmethod
    def add_special_tokens(self):
        """Adds special tokens to the tokenizer."""
        pass

    @property
    def cls_id(self) -> int:
        """Property alias to match MegatronTokenizer; returns cls_id if available."""
        if hasattr(self, 'cls_id'):
            return self.cls_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'cls' or 'cls_id'")

    @property
    def sep_id(self) -> int:
        """Property alias to match MegatronTokenizer; returns sep_id if available."""
        if hasattr(self, 'sep_id'):
            return self.sep_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'sep' or 'sep_id'")

    @property
    def pad_id(self) -> int:
        """Property alias to match MegatronTokenizer; returns pad_id if available."""
        if hasattr(self, 'pad_id'):
            return self.pad_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'pad' or 'pad_id'")

    @property
    def eod(self) -> int:
        """Property alias to match MegatronTokenizer; returns eod_id if available."""
        if hasattr(self, 'eod_id'):
            return self.eod_id
        if hasattr(self, 'eos_id'):
            # Default to end-of-sentence id if end-of-document is not defined.
            return self.eos_id
        raise AttributeError(
            f"{type(self).__name__} has no attribute 'eod', 'eod_id', 'eos', or 'eos_id'"
        )

    @property
    def bos_id(self) -> int:
        """Property alias to match MegatronTokenizer; returns bos_id if available."""
        if hasattr(self, 'bos_id'):
            return self.bos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'bos' or 'bos_id'")

    @property
    def eos_id(self) -> int:
        """Property alias to match MegatronTokenizer; returns eos_id if available."""
        if hasattr(self, 'eos_id'):
            return self.eos_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'eos' or 'eos_id'")

    @property
    def mask_id(self) -> int:
        """Property alias to match MegatronTokenizer; returns mask_id if available."""
        if hasattr(self, 'mask_id'):
            return self.mask_id
        raise AttributeError(f"{type(self).__name__} has no attribute 'mask' or 'mask_id'")
