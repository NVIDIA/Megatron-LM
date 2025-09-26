# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText


class RetroTokenizer(MegatronTokenizerText):
    """Base class for Megatron Retro tokenizer."""

    def __init__(self, path: str = None, config: dict = None, **kwargs) -> None:
        config['class_name'] = self.__class__.__name__
        config['class_path'] = self.__class__.__module__
        super().__init__(path, config, **kwargs)
