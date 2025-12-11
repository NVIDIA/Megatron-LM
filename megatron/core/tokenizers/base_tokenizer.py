# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod


class MegatronTokenizerBase(ABC):
    """Abstract class for Megatron tokenizers."""

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

        self.path = path
        for key, value in config.items():
            setattr(self, key, value)

    @abstractmethod
    def tokenize(self):
        """Encoding function."""
        pass

    @abstractmethod
    def detokenize(self):
        """Decoding function."""
        pass

    @abstractmethod
    def vocab(self):
        """Returns tokenizer vocab."""
        pass

    @abstractmethod
    def vocab_size(self):
        """Returns tokenizer vocab size."""
        pass

    @abstractmethod
    def apply_chat_template(self):
        """Applies tokenizer's chat template."""
        pass
