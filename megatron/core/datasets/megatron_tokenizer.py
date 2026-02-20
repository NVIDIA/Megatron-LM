# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Backward-compatibility shim. The legacy tokenizer base class was moved out of
# megatron.core.datasets in newer Megatron-LM versions.  Megatron-Bridge still
# imports from this path, so we keep a thin re-export here.

import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import numpy

logger = logging.getLogger(__name__)


class MegatronLegacyTokenizer(ABC):
    """Abstract class for tokenizer

    Absent a config or class-specific tracking of which objects are uniquely identifying, we must
    include all key word arguments as unique identifiers

    Args:
        tokenizer_paths (Tuple[str]): All tokenizer source paths or prefixes

        tokenizer_options (Dict[str, Any]): All tokenizer options
    """

    def __init__(self, *tokenizer_paths: str, **tokenizer_options: Any):
        logger.warning(
            "You're using the legacy tokenizer system, which is deprecated "
            "and will be removed in a future release. Please migrate to the new tokenizer system "
            "(`megatron.core.tokenizers.MegatronTokenizer`)."
        )
        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["tokenizer_path"] = list(tokenizer_paths)
        for option in tokenizer_options:
            self.unique_identifiers[option] = str(tokenizer_options[option])

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)

        super().__init__()

    @abstractmethod
    def tokenize(self, text: str) -> numpy.ndarray:
        pass

    def detokenize(self, ids: numpy.ndarray) -> str:
        raise NotImplementedError("{} has no method 'detokenize'".format(type(self).__name__))

    def offsets(self, ids: list[int], text: str) -> list[int]:
        raise NotImplementedError("{} has no method 'offsets'".format(type(self).__name__))

    @property
    @abstractmethod
    def vocab(self):
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    def cls(self):
        raise NotImplementedError("{} has no attribute 'cls'".format(type(self).__name__))

    @property
    def sep(self):
        raise NotImplementedError("{} has no attribute 'sep'".format(type(self).__name__))

    @property
    def pad(self):
        raise NotImplementedError("{} has no attribute 'pad'".format(type(self).__name__))

    @property
    def eod(self):
        raise NotImplementedError("{} has no attribute 'eod'".format(type(self).__name__))

    @property
    def bos(self):
        raise NotImplementedError("{} has no attribute 'bos'".format(type(self).__name__))

    @property
    def eos(self):
        raise NotImplementedError("{} has no attribute 'eos'".format(type(self).__name__))

    @property
    def mask(self):
        raise NotImplementedError("{} has no attribute 'mask'".format(type(self).__name__))


# Older code imported this class under the name ``MegatronTokenizer``.
MegatronTokenizer = MegatronLegacyTokenizer
