# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import math

from megatron.core.tokenizers import MegatronTokenizer

MEGATRON_TOKENIZERS = ['BertWordPieceLowerCase', 'BertWordPieceCase', 'GPT2BPETokenizer']

SP_TOKENIZERS = ['SentencePieceTokenizer', 'GPTSentencePieceTokenizer', 'Llama2Tokenizer']

logger = logging.getLogger(__name__)


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""
    kwargs = {}
    tokenizer_library = None
    tokenizer_path = None
    if args.tokenizer_type in MEGATRON_TOKENIZERS:
        tokenizer_library = 'megatron'
        tokenizer_path = args.tokenizer_type
        kwargs['additional_special_tokens'] = (
            args.tokenizer_special_tokens if args.tokenizer_special_tokens else []
        )
        if tokenizer_path == 'BertWordPieceCase':
            special_tokens = {}
            special_tokens['additional_special_tokens'] = [f'<extra_id_{i}>' for i in range(100)]
            kwargs = special_tokens
        kwargs['vocab_file'] = args.vocab_file
        kwargs['merges_file'] = args.merge_file
        kwargs['use_fast'] = not args.tokenizer_hf_no_use_fast
        kwargs['trust_remote_code'] = args.trust_remote_code
        kwargs['include_special_tokens'] = not args.tokenizer_hf_no_include_special_tokens
    elif args.tokenizer_type in SP_TOKENIZERS:
        tokenizer_library = 'sentencepiece'
        tokenizer_path = args.tokenizer_model
        kwargs['legacy'] = args.tokenizer_sentencepiece_legacy
        kwargs['special_tokens'] = args.tokenizer_special_tokens
    elif args.tokenizer_type == 'TikTokenizer':
        tokenizer_library = 'tiktoken'
        tokenizer_path = args.tokenizer_model
        if args.tiktoken_pattern:
            kwargs['pattern'] = args.tiktoken_pattern
        if args.vocab_size:
            kwargs['vocab_size'] = args.vocab_size
        kwargs['num_special_tokens'] = args.tiktoken_num_special_tokens
        kwargs['special_tokens'] = args.tokenizer_special_tokens
    elif args.tokenizer_type == 'HuggingFaceTokenizer':
        tokenizer_library = 'huggingface'
        tokenizer_path = args.tokenizer_model
        kwargs['vocab_file'] = args.vocab_file
        kwargs['merges_file'] = args.merge_file
        kwargs['additional_special_tokens'] = (
            args.tokenizer_special_tokens if args.tokenizer_special_tokens else []
        )
        kwargs['use_fast'] = not args.tokenizer_hf_no_use_fast
        kwargs['trust_remote_code'] = args.trust_remote_code
        kwargs['include_special_tokens'] = not args.tokenizer_hf_no_include_special_tokens
    elif args.tokenizer_type == 'MultimodalTokenizer':
        tokenizer_library = 'multimodal'
        kwargs['prompt_format'] = args.tokenizer_prompt_format
        kwargs['special_tokens'] = args.special_tokens
        kwargs['image_tag_type'] = args.image_tag_type
        kwargs['force_system_message'] = args.force_system_message
    elif args.tokenizer_type == 'SFTTokenizer':
        tokenizer_library = 'sft'
        tokenizer_path = args.tokenizer_model
        kwargs['prompt_format'] = args.sft_tokenizer_prompt_format
    elif args.tokenizer_type in ['NullTokenizer', 'NullMultimodalTokenizer']:
        tokenizer_library = (
            'null-text' if args.tokenizer_type == 'NullTokenizer' else 'null-multimodal'
        )
        metadata = {'library': tokenizer_library}
        if args.vocab_size:
            kwargs['vocab_size'] = args.vocab_size
        tokenizer = MegatronTokenizer.from_pretrained(metadata_path=metadata, **kwargs)

        # Add vocab size (if not already set from a checkpoint).
        _set_padded_vocab_size(args, tokenizer)

        return tokenizer

    if args.tokenizer_metadata:
        metadata = args.tokenizer_metadata
    else:
        metadata = {'library': tokenizer_library}
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=tokenizer_path, metadata_path=metadata, **kwargs
    )

    # Add vocab size (if not already set from a checkpoint).
    _set_padded_vocab_size(args, tokenizer)

    return tokenizer


def vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        logger.info(
            f' > padded vocab (size: {orig_vocab_size}) with '
            f'{after - orig_vocab_size} dummy tokens '
            f'(new size: {after})'
        )
    return after


def _set_padded_vocab_size(args, tokenizer):
    """Sets padded vocab size if None."""
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = vocab_size_with_padding(tokenizer.vocab_size, args)
