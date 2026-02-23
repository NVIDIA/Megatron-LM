# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import os
import random
import string
import tempfile
from argparse import Namespace
from collections import defaultdict
from typing import Dict, Optional

import numpy
import pytest
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.sft_dataset import SFTDataset, SFTDatasetConfig
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split, compile_helpers, get_blend_from_list
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.utils import get_blend_and_blend_per_split
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils
from tools.build_sequences_per_dataset import build_sequences_per_dataset

from megatron.core.tokenizers.text.libraries.sft_tokenizer import nemotron_nano_v2_custom_template

# TODO(asolergi-nv): Add chat template to tokens
def create_file_prefixes(tokenizer, number_of_files, maximum_number_of_conversations, maximum_number_of_messages_per_conversation, with_system_message, dataset_dir):
    # Create dataset directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Create file prefixes
    file_prefixes = []
    for i in range(number_of_files):
        file_prefix_path = os.path.join(dataset_dir, f"file_{i}")
        builder = IndexedDatasetBuilder(
            file_prefix_path + ".bin", dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )
        for _ in range(random.randint(10, maximum_number_of_conversations)):
            conversation = []
            if with_system_message:
                conversation.append({
                    "role": "system",
                    "content": "".join(random.choices(string.ascii_letters, k=random.randint(50,300)))
                })
            for _ in range(random.randint(1, maximum_number_of_messages_per_conversation)):
                conversation.append({
                    "role": "user",
                    "content": "".join(random.choices(string.ascii_letters, k=random.randint(50,300)))
                })
                conversation.append({
                    "role": "assistant",
                    "content": "".join(random.choices(string.ascii_letters, k=random.randint(50,300)))
                })

            tokenized_conversation = tokenizer.apply_chat_template(conversation, chat_template=nemotron_nano_v2_custom_template, tokenize=True, add_generation_prompt=False)

            builder.add_document(tokenized_conversation, [len(tokenized_conversation)])
        builder.finalize(file_prefix_path + ".idx")
        file_prefixes.append(file_prefix_path)

    return file_prefixes

@pytest.mark.parametrize("vocab_size", [131072, 20000])
@pytest.mark.parametrize("with_system_message", [True, False])
def test_sft_dataset(
    vocab_size,
    with_system_message,
    tmp_path_dist_ckpt,
    sequence_length: int = 500,
    number_of_files: int = 10,
    number_of_conversations: int = 20,
    maximum_number_of_messages_per_conversation: int = 8,
):
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = build_tokenizer(
        Namespace(
            vocab_size=vocab_size,
            tokenizer_type="NullSFTTokenizer",
            rank=0,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
        )
    )

    with TempNamedDir(tmp_path_dist_ckpt / "test_fast_builder", sync=True) as temp_dir:
        # Created file_prefixes (tokenizer, Number of files, number of documents, path) --> returns file prefixes (list of strings)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            file_prefixes = create_file_prefixes(
                tokenizer, number_of_files, number_of_conversations, maximum_number_of_messages_per_conversation, with_system_message, os.path.join(temp_dir, "dataset")
            )
        else:
            file_prefixes = []
            for i in range(number_of_files):
                file_prefix_path = os.path.join(temp_dir, "dataset", f"file_{i}")
                file_prefixes.append(file_prefix_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        random.seed(1234)  # NOTE(asolergi-nv): re-sync random state across all ranks

        data_cache_path = os.path.join(temp_dir, "cache")

        args = Namespace(
            seed=1234,
            seq_length=sequence_length,
            data_cache_path=data_cache_path,
            split=None,
            data_path=None,
            train_data_path=file_prefixes[0:6],
            valid_data_path=file_prefixes[6:9],
            test_data_path=file_prefixes[9:10],
            per_split_data_args_path=None,
            data_args_path=None,
        )

        blend, blend_per_split = get_blend_and_blend_per_split(args)

        data_args = {
            "random_seed": args.seed,
            "sequence_length": args.seq_length,
            "blend": blend,
            "blend_per_split": blend_per_split,
            "split": args.split,
            "path_to_cache": args.data_cache_path,
            "tokenizer": tokenizer,
            "reset_position_ids": False,
            "reset_attention_mask": False,
            "eod_mask_loss": False,
            "create_attention_mask": False,
        }
        config = SFTDatasetConfig(**data_args)

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            SFTDataset, [100, 10, 10], lambda: True, config
        ).build()

        print(f"train_ds: {train_ds[0]}")


    
    