# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import importlib
import json
import logging
import os
from collections import OrderedDict
from typing import Optional, Union

from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase

TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("default-text", "DefaultTokenizerText"),
        ("gpt", "GPTTokenizer"),
        ("mamba", "MambaTokenizer"),
        ("bert", "BertTokenizer"),
        ("t5", "T5Tokenizer"),
        ("default-vision", "DefaultTokenizerVision"),
    ]
)

TEXT_LIBRARIES = [
    "sentencepiece",
    "huggingface",
    "megatron",
    "tiktoken",
    "byte-level",
    "null-text",
    "sft",
]
VISION_LIBRARIES = ["multimodal", "null-multimodal"]

logger = logging.getLogger(__name__)


class MegatronTokenizer:
    """Restores model tokenizer."""

    def __init__(self) -> None:
        raise EnvironmentError(
            "MegatronTokenizer is designed to be instantiated using the "
            "`MegatronTokenizer.from_pretrained()` method."
        )

    def from_pretrained(
        tokenizer_path: str = None, metadata_path: Optional[Union[str, dict]] = None, **kwargs
    ) -> MegatronTokenizerBase:
        """
        Args:
            path (str): path to tokenizer file with metadata.json in folder.
            metadata_path (Optional[str]): path to the tokenizer metadata.
                Must be specified when loading the tokenizer from HF.

        Returns:
            MegatronTokenizerBase: tokenizer object.

        Usage:
            MegatronTokenizer.from_pretrained(tokenizer_path='/path/to/tokenzier')
        """

        # Get metadata path
        if not metadata_path:
            metadata_path = _get_metadata_path(tokenizer_path)

        if isinstance(metadata_path, str):
            # Check if metadata file exists
            assert os.path.exists(metadata_path), (
                "Tokenizer metadata file doesn't exist. Please, use "
                "MegatronTokenizer.write_metadata() method to generate metadata file."
            )
            # Load tokenizer metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        elif isinstance(metadata_path, dict):
            metadata = metadata_path
            metadata_path = None
        else:
            raise ValueError(
                f"Expected metadata_path to be str or dict, but got {type(metadata_path)}."
            )

        tokenizer_library = metadata.get('library', None)
        if tokenizer_library not in ['byte-level', 'null-text', 'null-multimodal']:
            assert tokenizer_path, "Tokenizer path must be specified."

        if tokenizer_library in ['multimodal']:
            assert 'prompt_format' in kwargs, "Prompt format (`prompt_format`) must be specified."
            assert (
                'special_tokens' in kwargs
            ), "Special tokens (`special_tokens`) must be specified."
            assert (
                'image_tag_type' in kwargs
            ), "Image tag type (`image_tag_type`) must be specified."

        # Initialize tokenizer object
        tokenizer_cls = _get_tokenizer_model_class(tokenizer_library, metadata)

        metadata['metadata_path'] = metadata_path
        tokenizer = tokenizer_cls(path=tokenizer_path, config=metadata, **kwargs)

        return tokenizer

    def write_metadata(
        tokenizer_path: str,
        tokenizer_library: str,
        model_type: Optional[str] = None,
        tokenizer_class: Optional[MegatronTokenizerBase] = None,
        chat_template: Optional[str] = None,
        overwrite: Optional[bool] = False,
        metadata_path: Optional[str] = None,
    ) -> None:
        """
        Creates metadata file for tokenizer.

        Args:
            tokenizer_path (str): path to tokenizer model.
            tokenizer_library (str): tokenizer model library.
            model_type (str): type of the model to be used with tokenizer.
                list of available model types: [gpt, bert, t5, mamba, default].
                `DefaultTokenizerText` will be used if model_type is not specified.
            tokenizer_class (MegatronTokenizerBase): pre-defined tokenizer class.
            chat_template (str): tokenizer chat template in jinja format.
            overwrite (bool): overwrites existing metadata file if set to True.
            metadata_path (Optional[str]): path where metadata file will be saved. If not specified,
                the metadata file will be stored in the same directory as the tokenizer.

        Usage:
            MegatronTokenizer.write_metadata(
                tokenizer_path='/path/to/tokenzier/model',
                tokenizer_library='sentencepiece',
                model_type='llama',
            )
        """

        assert os.path.exists(
            tokenizer_path
        ), "Tokenizer path doesn't exist. Please, provide the correct path to the tokenizer."
        assert tokenizer_library in TEXT_LIBRARIES or tokenizer_library in VISION_LIBRARIES, (
            "Tokenizer library is not supported. Please, see the list of available "
            f"tokenizer libraries: text: {TEXT_LIBRARIES}, vision: {VISION_LIBRARIES}."
        )
        tokenizer_type = 'text' if tokenizer_library in TEXT_LIBRARIES else 'vision'
        if model_type is None and tokenizer_class is None:
            model_type = f"default-{tokenizer_type}"

        # Write metadata
        if not metadata_path:
            metadata_path = _get_metadata_path(tokenizer_path)
        if os.path.exists(metadata_path) and not overwrite:
            raise ValueError(
                "Metadata file already exists. If you want to overwrite it, "
                "please set overwrite param to True."
            )
        else:
            metadata = {
                'library': tokenizer_library,
                'class_name': tokenizer_class.__name__ if tokenizer_class else None,
                'class_path': tokenizer_class.__module__ if tokenizer_class else None,
                'model_type': model_type,
                'chat_template': chat_template,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            logger.info(f"Metadata file was sucessfully saved: {metadata_path}.")


def _get_metadata_path(tokenizer_path: str) -> str:
    """
    Returns metadata file path.

    Args:
        tokenizer_path (str): path to the tokenizer model.

    Returns:
        str: path to the metadata file.
    """

    # Get metadata file path
    dir_path = os.path.dirname(tokenizer_path) if os.path.isfile(tokenizer_path) else tokenizer_path
    metadata_path = f'{dir_path}/tokenizer_metadata.json'

    return metadata_path


def _get_tokenizer_model_class(library: str, metadata: dict) -> MegatronTokenizerBase:
    """
    Returns a class which corresponds to choosen tokenizer model type.

    Args:
        library (str): tokenizer library.
        metadata (dict): tokenizer metadata.

    Returns:
        MegatronTokenizerBase: class for choosen tokenizer model type.
    """
    # Return tokenizer class if it was specified in metadata.
    if metadata.get('tokenizer_class', None):
        return getattr(metadata['tokenizer_class_path'], metadata['tokenizer_class_name'])

    # Define tokenizer type
    tokenizer_type = 'text' if library in TEXT_LIBRARIES else 'vision'

    module_name = f"megatron.core.tokenizers.{tokenizer_type}.models"
    models = importlib.import_module(module_name)

    model_type = metadata.get("model_type", None)
    if model_type is None:
        model_type = f"default-{tokenizer_type}"

    tokenizer_cls = getattr(models, TOKENIZER_MAPPING_NAMES[model_type])

    return tokenizer_cls
