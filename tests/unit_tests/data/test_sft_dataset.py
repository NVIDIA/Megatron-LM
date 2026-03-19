# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import os
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import numpy
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder
from megatron.core.datasets.sft_dataset import (
    ChatTemplateConfig,
    SFTDataset,
    SFTDatasetConfig,
    extract_segments,
)
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.utils import get_blend_and_blend_per_split
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


@dataclass
class TestChatTemplateConfig(ChatTemplateConfig):
    """Chat template config compatible with NullSFTTokenizer's character-level encoding.

    Marker strings use ASCII characters (token IDs < 128).
    Think markers must be single characters (tokenize(str)[0] returns a single ID).
    Tool call markers end with '\\n' as required by _split_tool_calls.
    """

    system_start_str: str = "[S]"
    user_start_str: str = "[U]"
    assistant_start_str: str = "[A]"
    end_str: str = "[E]"
    think_start_str: str = "{"
    think_end_str: str = "}"
    tool_call_start_str: str = "~\n"
    tool_call_end_str: str = "`\n"
    tool_response_start_str: str = "^\n"
    tool_response_end_str: str = "|\n"


# Safe range for content tokens: [200, 1000) avoids collisions with ASCII marker tokens (< 128)
CONTENT_TOKEN_MIN = 200
CONTENT_TOKEN_MAX = 1000


def _random_content_tokens(min_len=5, max_len=20):
    """Generate random content token IDs in a safe range that won't collide with markers."""
    length = random.randint(min_len, max_len)
    return [random.randint(CONTENT_TOKEN_MIN, CONTENT_TOKEN_MAX - 1) for _ in range(length)]


def build_tokenized_conversation(
    tokenizer, chat_template_config, conversation_type, with_system=True
):
    """Build a pre-tokenized conversation as a flat list of token IDs.

    Args:
        tokenizer: Tokenizer with tokenize() method.
        chat_template_config: TestChatTemplateConfig instance.
        conversation_type: One of 'simple', 'with_thinking', 'with_tool_calls',
                           'with_thinking_and_tool_calls'.
        with_system: Whether to include a system message.

    Returns:
        List[int]: The tokenized conversation.
    """
    cfg = chat_template_config

    sys_start = tokenizer.tokenize(cfg.system_start_str)
    usr_start = tokenizer.tokenize(cfg.user_start_str)
    ast_start = tokenizer.tokenize(cfg.assistant_start_str)
    end = tokenizer.tokenize(cfg.end_str)
    think_start = tokenizer.tokenize(cfg.think_start_str)
    think_end = tokenizer.tokenize(cfg.think_end_str)
    tc_start = tokenizer.tokenize(cfg.tool_call_start_str)
    tc_end = tokenizer.tokenize(cfg.tool_call_end_str)
    tr_start = tokenizer.tokenize(cfg.tool_response_start_str)

    tokens = []

    # Optional system message
    if with_system:
        tokens += sys_start + _random_content_tokens() + end

    if conversation_type == "simple":
        # user + assistant
        tokens += usr_start + _random_content_tokens() + end
        tokens += ast_start + _random_content_tokens() + end

    elif conversation_type == "with_thinking":
        # user + assistant(think + response)
        tokens += usr_start + _random_content_tokens() + end
        tokens += (
            ast_start
            + think_start
            + _random_content_tokens()
            + think_end
            + _random_content_tokens()
            + end
        )

    elif conversation_type == "with_tool_calls":
        # user + assistant(tool_call) + user(tool_response) + assistant(response)
        tokens += usr_start + _random_content_tokens() + end
        tokens += ast_start + tc_start + _random_content_tokens() + tc_end + end
        tokens += usr_start + tr_start + _random_content_tokens() + end
        tokens += ast_start + _random_content_tokens() + end

    elif conversation_type == "with_thinking_and_tool_calls":
        # user + assistant(think + tool_call) + user(tool_response) + assistant(response)
        tokens += usr_start + _random_content_tokens() + end
        tokens += (
            ast_start
            + think_start
            + _random_content_tokens()
            + think_end
            + tc_start
            + _random_content_tokens()
            + tc_end
            + end
        )
        tokens += usr_start + tr_start + _random_content_tokens() + end
        tokens += ast_start + _random_content_tokens() + end

    return tokens


CONVERSATION_TYPES = ["simple", "with_thinking", "with_tool_calls", "with_thinking_and_tool_calls"]


def create_file_prefixes(
    tokenizer, chat_template_config, number_of_files, max_conversations_per_file, dataset_dir
):
    """Create indexed dataset files with pre-tokenized conversations of all types."""
    os.makedirs(dataset_dir, exist_ok=True)

    file_prefixes = []
    for i in range(number_of_files):
        file_prefix_path = os.path.join(dataset_dir, f"file_{i}")
        builder = IndexedDatasetBuilder(
            file_prefix_path + ".bin", dtype=DType.optimal_dtype(tokenizer.vocab_size)
        )
        for _ in range(random.randint(10, max_conversations_per_file)):
            conv_type = random.choice(CONVERSATION_TYPES)
            with_system = random.choice([True, False])
            tokens = build_tokenized_conversation(
                tokenizer, chat_template_config, conv_type, with_system
            )
            builder.add_document(tokens, [len(tokens)])
        builder.finalize(file_prefix_path + ".idx")
        file_prefixes.append(file_prefix_path)

    return file_prefixes


def verify_loss_mask(sample, config):
    """Verify the loss mask of a packed sample against the expected pattern.

    For each document in the packed sample (split by cu_seqlens), runs
    extract_segments and checks that loss_mask values match the flag settings.
    """
    tokens = sample['tokens']
    labels = sample['labels']
    loss_mask = sample['loss_mask']
    cu_seqlens = sample['cu_seqlens']

    # The __getitem__ returns tokens[:-1], labels=tokens[1:], loss_mask[:-1]
    # To reconstruct the full token sequence per document, we use cu_seqlens.
    # cu_seqlens gives cumulative lengths of the original (unshifted) documents.
    # The total packed length is cu_seqlens[-1], and output length is cu_seqlens[-1] - 1.

    num_docs = len(cu_seqlens) - 1
    # Reconstruct the full token sequence (before the [:-1] / [1:] split)
    full_tokens = torch.cat([tokens[:1], labels])  # length = cu_seqlens[-1]
    # Reconstruct the full loss mask (before the [:-1] slice)
    # loss_mask has length cu_seqlens[-1] - 1, original had length cu_seqlens[-1]
    # We don't know the last element, but we don't need it for per-doc verification

    offset = 0
    for doc_idx in range(num_docs):
        doc_len = (cu_seqlens[doc_idx + 1] - cu_seqlens[doc_idx]).item()
        doc_tokens = full_tokens[offset : offset + doc_len].tolist()

        # loss_mask covers positions [0, cu_seqlens[-1] - 1) of the full sequence
        # For this document, loss_mask positions are [offset, offset + doc_len)
        # but the last position of the entire packed sequence has no loss_mask entry
        doc_loss_mask_end = min(offset + doc_len, len(loss_mask))
        doc_loss_mask = loss_mask[offset:doc_loss_mask_end]

        if not config.train_on_assistant_responses_only:
            # All tokens should be trained on
            assert (
                doc_loss_mask == 1.0
            ).all(), f"Doc {doc_idx}: expected all loss_mask=1 when train_on_assistant_responses_only=False"
        else:
            segments = extract_segments(
                doc_tokens,
                config.role_start_tokens,
                config.end_tokens,
                config.think_start_id,
                config.think_end_id,
                config.tool_call_start_tokens,
                config.tool_call_end_tokens,
                config.tool_response_start_tokens,
            )

            for seg in segments:
                # Segment positions are relative to doc_tokens, but loss_mask is
                # indexed from the start of the packed sequence. Adjust with offset.
                seg_start = seg["start"]
                seg_end = min(seg["end"], doc_loss_mask_end - offset)
                if seg_start >= seg_end:
                    continue

                seg_mask = doc_loss_mask[seg_start:seg_end]
                role = seg["role"]

                if role == "assistant":
                    expected = 1.0
                elif role == "reasoning":
                    expected = 1.0 if config.train_on_thinking_traces else 0.0
                elif role == "tool_call":
                    expected = 1.0 if config.train_on_tool_calls else 0.0
                else:
                    # system, user, tool_response — always masked
                    expected = 0.0

                assert (seg_mask == expected).all(), (
                    f"Doc {doc_idx}, segment role='{role}' [{seg_start}:{seg_end}]: "
                    f"expected loss_mask={expected}, got {seg_mask.tolist()}"
                )

        offset += doc_len


@pytest.mark.parametrize("vocab_size", [131072, 20000])
@pytest.mark.parametrize(
    "train_on_assistant_responses_only,train_on_thinking_traces,train_on_tool_calls",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, True),
    ],
)
def test_sft_dataset(
    vocab_size,
    train_on_assistant_responses_only,
    train_on_thinking_traces,
    train_on_tool_calls,
    tmp_path_dist_ckpt,
    sequence_length: int = 1500,
    number_of_files: int = 10,
    max_conversations_per_file: int = 20,
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

    chat_template_config = TestChatTemplateConfig()

    with TempNamedDir(tmp_path_dist_ckpt / "test_fast_builder", sync=True) as temp_dir:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            file_prefixes = create_file_prefixes(
                tokenizer,
                chat_template_config,
                number_of_files,
                max_conversations_per_file,
                os.path.join(temp_dir, "dataset"),
            )
        else:
            file_prefixes = []
            for i in range(number_of_files):
                file_prefix_path = os.path.join(temp_dir, "dataset", f"file_{i}")
                file_prefixes.append(file_prefix_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        random.seed(1234)

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
            "train_on_assistant_responses_only": train_on_assistant_responses_only,
            "train_on_thinking_traces": train_on_thinking_traces,
            "train_on_tool_calls": train_on_tool_calls,
            "chat_template_config": chat_template_config,
        }
        config = SFTDatasetConfig(**data_args)

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            SFTDataset, [100, 10, 10], lambda: True, config
        ).build()

        # Shape invariant checks + loss mask verification
        for sample_idx in [0, 1, -1]:
            sample = train_ds[sample_idx]
            tokens = sample['tokens']
            labels = sample['labels']
            loss_mask = sample['loss_mask']
            cu_seqlens = sample['cu_seqlens']

            # Dtype checks
            assert tokens.dtype == torch.int64
            assert labels.dtype == torch.int64
            assert loss_mask.dtype == torch.float32
            assert cu_seqlens.dtype == torch.int32

            # Shape consistency: tokens, labels, loss_mask must have the same length
            assert tokens.shape == labels.shape == loss_mask.shape

            # Packed length must not exceed sequence_length
            assert tokens.shape[0] <= sequence_length

            # cu_seqlens[-1] == total tokens before the [:-1]/[1:] shift
            assert tokens.shape[0] + 1 == cu_seqlens[-1]

            # Labels are correctly shifted: labels[i] == tokens[i+1] in the original sequence
            # Reconstruct full sequence and verify
            full_tokens = torch.cat([tokens[:1], labels])
            assert (full_tokens[:-1] == tokens).all(), "tokens should be full_tokens[:-1]"
            assert (full_tokens[1:] == labels).all(), "labels should be full_tokens[1:]"

            # Token values are in valid vocab range
            assert (tokens >= 0).all() and (tokens < vocab_size + 1).all()
            assert (labels >= 0).all() and (labels < vocab_size + 1).all()

            # cu_seqlens invariants: starts at 0, monotonically increasing
            assert cu_seqlens[0] == 0
            doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            assert (doc_lengths > 0).all(), "Each document must have positive length"
            assert (
                doc_lengths <= sequence_length
            ).all(), "No document should exceed sequence_length"

            # Loss mask is binary (only 0.0 or 1.0)
            assert ((loss_mask == 0.0) | (loss_mask == 1.0)).all(), "Loss mask must be binary"

            # Loss mask is non-trivial when train_on_assistant_responses_only=True
            if train_on_assistant_responses_only:
                assert (loss_mask == 0.0).any(), "Expected some masked positions"
                assert (loss_mask == 1.0).any(), "Expected some unmasked positions"

            # Per-document loss mask verification via extract_segments
            verify_loss_mask(sample, config)
