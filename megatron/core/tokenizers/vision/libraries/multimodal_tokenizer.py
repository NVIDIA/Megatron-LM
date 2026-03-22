# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Union

from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN
from megatron.core.tokenizers.conversation import PROMPT_FORMAT_REGISTRY, tokenize_conversation

try:
    import transformers

    HAVE_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    HAVE_TRANSFORMERS = False


IMAGE_TAGS = {
    "nvlm": ("<Image>", "</Image>"),
    "internvl": ("<img>", "</img>"),
    "": None,  # Image tag not used.
}


class MegatronMultimodalTokenizer:
    """Multimodal Tokenizer."""

    def __init__(
        self,
        path: str,
        prompt_format: str,
        special_tokens: List[str],
        image_tag_type: str,
        force_system_message: bool = False,
        **kwargs,
    ):
        """Tokenizer with a support for non-text inputs.

        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            path (str): Path to the underlying tokenizer.
            prompt_format (str): Prompt format for the tokenizer.
            special_tokens (List[str]): Non-text tokens.
            image_tag_type (str): Image tag to apply, if any. For example <img><image></img>.
        """
        if not HAVE_TRANSFORMERS:
            raise ImportError(
                "MegatronMultimodalTokenizer currently requires "
                "transformers library to be installed."
            )
        if prompt_format == "nvlm-yi-34b":
            kwargs.update({"from_slow": True, "legacy": False, "add_bos_token": True})
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=path, **kwargs
        )

        self._vocab_size = len(tokenizer)

        num_added_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        assert num_added_tokens == len(
            special_tokens
        ), f"failed to add {len(special_tokens)} special tokens; only added {num_added_tokens}"

        self.tokenizer = tokenizer

        if prompt_format not in PROMPT_FORMAT_REGISTRY:
            raise NotImplementedError("unknown multimodal tokenizer type", prompt_format)

        # Build the prompt config from the registry.
        # Qwen formats need the force_system_message flag passed through.
        if prompt_format in ("qwen2p0", "qwen2p5"):
            self._prompt_config = PROMPT_FORMAT_REGISTRY[prompt_format](
                tokenizer, force_system_message=force_system_message
            )
        else:
            self._prompt_config = PROMPT_FORMAT_REGISTRY[prompt_format](tokenizer)

        self._prompt_format = prompt_format
        self._image_tag = IMAGE_TAGS[image_tag_type]

    def _apply_image_tag(self, text: Union[str, List[Dict]]):
        """Surround <image> with image tags such as <img> and </img>."""
        if self._image_tag is None:
            return text

        replacement = f"{self._image_tag[0]}{IMAGE_TOKEN}{self._image_tag[1]}"

        if isinstance(text, list):
            for turn in text:
                turn["content"] = turn["content"].replace(IMAGE_TOKEN, replacement)
        else:
            text = text.replace(IMAGE_TOKEN, replacement)

        return text

    def tokenize(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(text, False, True).tolist()

        return self._encode(text)

    def _encode(self, text: str):
        """Tokenize text input."""
        text = self._apply_image_tag(text)
        return self.tokenizer.encode(text)

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
        return tokenize_conversation(
            tokenizer=self.tokenizer,
            conversation=conversation,
            prompt_config=self._prompt_config,
            return_target=return_target,
            add_generation_prompt=add_generation_prompt,
            apply_image_tag_fn=self._apply_image_tag,
        )

    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, tokens: List[int]):
        """Detokenize tokens."""
        return self.tokenizer.decode(tokens)

    def add_special_tokens(self, special_tokens: List[str]):
        """Add special tokens."""
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

    def get_special_tokens(self):
        """Get special tokens."""
        return self.tokenizer.get_added_vocab()

    @property
    def pad(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size

    @property
    def vocab(self):
        """Tokenizer vocab."""
        return self.tokenizer.get_vocab()
