# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import shutil
from typing import Optional

import torch
from torch.hub import _get_torch_home

try:
    import wget

    HAVE_WGET = True
except ModuleNotFoundError:
    HAVE_WGET = False

from .huggingface_tokenizer import HuggingFaceTokenizer

logger = logging.getLogger(__name__)
torch_home = _get_torch_home()

if not isinstance(torch_home, str):
    logger.info("Torch home not found, caching megatron in cwd")
    torch_home = os.getcwd()

MEGATRON_CACHE = os.path.join(torch_home, "megatron")

MEGATRON_CONFIG_MAP = {
    "BertWordPieceLowerCase": {
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt",  # pylint: disable=line-too-long
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",  # pylint: disable=line-too-long
        "tokenizer_name": "bert-large-uncased",
    },
    "BertWordPieceCase": {
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/files/release/mp_rank_00/model_optim_rng.pt",  # pylint: disable=line-too-long
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "tokenizer_name": "bert-large-cased",
    },
    "GPT2BPETokenizer": {
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "merges_file": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "tokenizer_name": "gpt2",
    },
    "megatron-gpt-345m": {
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "merges_file": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "tokenizer_name": "gpt2",
    },
    "megatron-bert-345m-uncased": {
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.0/files/release/mp_rank_00/model_optim_rng.pt",  # pylint: disable=line-too-long
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        "tokenizer_name": "bert-large-uncased",
    },
    "megatron-bert-345m-cased": {
        "checkpoint": "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/files/release/mp_rank_00/model_optim_rng.pt",  # pylint: disable=line-too-long
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "tokenizer_name": "bert-large-cased",
    },
    "megatron-bert-uncased": {
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",  # pylint: disable=line-too-long
        "tokenizer_name": "bert-large-uncased",
    },
    "megatron-bert-cased": {
        "vocab": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        "tokenizer_name": "bert-large-cased",
    },
    "biomegatron-bert-345m-uncased": {
        "vocab": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345muncased/versions/0/files/vocab.txt",  # pylint: disable=line-too-long
        "tokenizer_name": "bert-large-uncased",
    },
    "biomegatron-bert-345m-cased": {
        "vocab": "https://api.ngc.nvidia.com/v2/models/nvidia/biomegatron345mcased/versions/0/files/vocab.txt",  # pylint: disable=line-too-long
        "tokenizer_name": "bert-large-cased",
    },
}


class MegatronHFTokenizer(HuggingFaceTokenizer):
    """ """

    def __init__(
        self,
        tokenizer_path: str,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        if tokenizer_path in MEGATRON_CONFIG_MAP.keys():
            tokenizer_name = tokenizer_path
        else:
            raise ValueError(
                f"The name of the tokenizer is incorrect. \
            Please see the list of available models: {self._get_available_models_list()}."
            )

        vocab_file = self._get_vocab_file(tokenizer_name, vocab_file)
        merges_file = self._get_merges_file(tokenizer_name, vocab_file)
        tokenizer_path = MEGATRON_CONFIG_MAP[tokenizer_name]["tokenizer_name"]
        super().__init__(tokenizer_path, vocab_file, merges_file, **kwargs)

    def _get_vocab_file(self, tokenizer_name: str, vocab_file: str = None) -> str:
        """
        Gets vocabulary file from cache or downloads it.

        Args:
            tokenizer_name (str): pretrained model name.
            vocab_file (str): path to the vocab file.

        Returns:
            path: path to the vocab file
        """

        if not vocab_file:
            url = MEGATRON_CONFIG_MAP[tokenizer_name]["vocab"]

            path = os.path.join(MEGATRON_CACHE, tokenizer_name + "_vocab")
            vocab_file = self._download(path, url)

        return vocab_file

    def _get_merges_file(self, tokenizer_name: str, merges_file: str = None) -> str:
        """
        Gets merge file from cache or downloads it.

        Args:
            tokenizer_name (str): pretrained model name.
            merges_file (str): path to the merges file.

        Returns:
            path: path to the vocab file.
        """

        if not merges_file:
            if 'gpt' not in tokenizer_name.lower():
                return None
            url = MEGATRON_CONFIG_MAP[tokenizer_name]["merges_file"]

            path = os.path.join(MEGATRON_CACHE, tokenizer_name + "_merges")
            merges_file = self._download(path, url)

        return merges_file

    def _get_available_models_list(self) -> list:
        """Returns a list of available megatron tokenizers."""

        return list(MEGATRON_CONFIG_MAP.keys())

    def _download(self, path: str, url: str):
        """
        Gets a file from cache or downloads it

        Args:
            path: path to the file in cache
            url: url to the file
        Returns:
            path: path to the file in cache
        """

        if url is None:
            return None

        if (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ) and not os.path.exists(path):
            os.makedirs(MEGATRON_CACHE, exist_ok=True)
            logging.info(f"Downloading from {url} to {path}")
            if HAVE_WGET:
                downloaded_path = wget.download(url)
            else:
                raise ModuleNotFoundError("wget library should be isntalled.")
            if not os.path.exists(downloaded_path):
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")
            shutil.move(downloaded_path, path)
        # wait until the master process downloads the file and writes it to the cache dir
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return path
