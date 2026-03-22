# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

try:
    from transformers.utils.chat_template_utils import _compile_jinja_template

    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


class MegatronTokenizerTextAbstract(ABC):
    """
    Abstract class for Megatron text tokenizers.
    """

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: str = None,
        tokenize: Optional[bool] = True,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        add_generation_prompt: Optional[bool] = False,
        **kwargs,
    ) -> Union[str, List[int]]:
        """
        Applies tokenizer's chat template to the conversation using Jinja2.

        Args:
            conversation (List[Dict[str, str]]): a list of dicts with "role" and "content" keys,
                representing the chat history so far.
            chat_template (str): Jinja2 chat template string. If not provided, falls back to
                ``self.chat_template``.
            tokenize (bool): whether to tokenize the output. If ``False``,
                the output will be a string.
            truncation (bool): whether to truncate sequences at the maximum length.
                Has no effect if tokenize is ``False``.
            max_length (int): maximum length to use for truncation.
                Has no effect if tokenize is ``False``.
            add_generation_prompt (bool): If set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output.
        """
        if not chat_template:
            chat_template = getattr(self, 'chat_template', None)
        assert chat_template, (
            "Chat template is not defined. "
            "Please, specify tokenizer chat template in the metadata file."
        )
        if truncation:
            assert max_length, "max_length must be specified if truncation is used."

        if HAVE_TRANSFORMERS:
            compiled_template = _compile_jinja_template(chat_template)
            chat_text = compiled_template.render(
                messages=conversation, add_generation_prompt=add_generation_prompt
            )

            if tokenize:
                chat_ids = self.text_to_ids(chat_text)
                if truncation:
                    chat_ids = chat_ids[:max_length]
                return chat_ids

            return chat_text
        else:
            raise ModuleNotFoundError("Please, install transformers library.")

    def token_to_id(self, token: str) -> int:
        """Converts a single token to its ID.

        Concrete default so that every text library tokenizer exposes a canonical
        single-token-to-ID method.  SentencePiece, TikToken, and HuggingFace
        override this with optimized versions; the default delegates to
        ``tokens_to_ids``.
        """
        return self.tokens_to_ids([token])[0]

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
