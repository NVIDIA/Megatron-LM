# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import math

from megatron.core.tokenizers import MegatronTokenizer

logger = logging.getLogger(__name__)


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer from command-line arguments.

    Uses --tokenizer-library and --tokenizer-mode (new API) or falls back to
    --tokenizer-type (deprecated, mapped in argument validation).
    """
    build_kwargs = _build_library_kwargs(args)
    build_kwargs.update(_build_mode_kwargs(args))
    build_kwargs.update(kwargs)  # Allow caller overrides

    if getattr(args, 'tokenizer_metadata', None):
        metadata = args.tokenizer_metadata
    else:
        metadata = {'library': _resolve_library(args)}

    tokenizer_path = _resolve_tokenizer_path(args)

    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=tokenizer_path, metadata_path=metadata, **build_kwargs
    )

    # Add vocab size (if not already set from a checkpoint).
    _set_padded_vocab_size(args, tokenizer)

    return tokenizer


def _resolve_library(args):
    """Map tokenizer-library + tokenizer-mode to the internal library string."""
    mode = getattr(args, 'tokenizer_mode', 'text')
    lib = getattr(args, 'tokenizer_library', None)

    if mode == 'sft':
        # SFT is now a capability, not a library. Use the actual library.
        # Default to 'huggingface' for backward compat.
        return lib or 'huggingface'
    elif mode == 'multimodal':
        return 'multimodal' if lib != 'null' else 'null-multimodal'
    elif lib == 'null':
        return 'null-text'
    return lib


def _resolve_tokenizer_path(args):
    """Resolve tokenizer path from args."""
    lib = getattr(args, 'tokenizer_library', None)

    if lib == 'megatron':
        # For 'megatron' library, the tokenizer_type or tokenizer_model is the
        # predefined model name (e.g., "GPT2BPETokenizer", "BertWordPieceCase").
        return getattr(args, 'tokenizer_type', None) or args.tokenizer_model
    return args.tokenizer_model


def _build_library_kwargs(args):
    """Build kwargs specific to the tokenizer library."""
    build_kwargs = {}
    lib = getattr(args, 'tokenizer_library', None)

    if lib in ('huggingface', 'megatron'):
        # Special case for BertWordPieceCase: add extra_id tokens.
        if (
            getattr(args, 'tokenizer_type', None) == 'BertWordPieceCase'
            or getattr(args, 'tokenizer_model', None) == 'BertWordPieceCase'
        ):
            build_kwargs['additional_special_tokens'] = [f'<extra_id_{i}>' for i in range(100)]
        else:
            build_kwargs['additional_special_tokens'] = (
                args.tokenizer_special_tokens if args.tokenizer_special_tokens else []
            )
        build_kwargs['vocab_file'] = args.vocab_file
        build_kwargs['merges_file'] = args.merge_file
        build_kwargs['use_fast'] = not getattr(args, 'tokenizer_hf_no_use_fast', False)
        build_kwargs['trust_remote_code'] = getattr(args, 'trust_remote_code', False)
        build_kwargs['include_special_tokens'] = not getattr(
            args, 'tokenizer_hf_no_include_special_tokens', False
        )
    elif lib == 'sentencepiece':
        build_kwargs['legacy'] = getattr(args, 'tokenizer_sentencepiece_legacy', False)
        build_kwargs['special_tokens'] = args.tokenizer_special_tokens
    elif lib == 'tiktoken':
        if getattr(args, 'tiktoken_pattern', None):
            build_kwargs['pattern'] = args.tiktoken_pattern
        if getattr(args, 'vocab_size', None):
            build_kwargs['vocab_size'] = args.vocab_size
        build_kwargs['num_special_tokens'] = getattr(args, 'tiktoken_num_special_tokens', 1000)
        build_kwargs['special_tokens'] = args.tokenizer_special_tokens
    elif lib == 'null':
        if getattr(args, 'vocab_size', None):
            build_kwargs['vocab_size'] = args.vocab_size

    return build_kwargs


def _build_mode_kwargs(args):
    """Build kwargs specific to the tokenizer mode."""
    build_kwargs = {}
    mode = getattr(args, 'tokenizer_mode', 'text')

    if mode == 'sft':
        prompt_format = getattr(args, 'tokenizer_prompt_format', None)
        if prompt_format is None:
            prompt_format = getattr(args, 'sft_tokenizer_prompt_format', 'nemotron-h-aligned')
        build_kwargs['prompt_format'] = prompt_format
    elif mode == 'multimodal':
        build_kwargs['prompt_format'] = getattr(args, 'tokenizer_prompt_format', None)
        build_kwargs['special_tokens'] = getattr(args, 'special_tokens', [])
        build_kwargs['image_tag_type'] = getattr(args, 'image_tag_type', '')
        build_kwargs['force_system_message'] = getattr(args, 'force_system_message', False)

    return build_kwargs


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
