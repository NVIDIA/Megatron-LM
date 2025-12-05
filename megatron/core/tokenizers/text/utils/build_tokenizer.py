# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from megatron.core.tokenizers import MegatronTokenizer

MEGATRON_TOKENIZERS = ['BertWordPieceLowerCase', 'BertWordPieceCase', 'GPT2BPETokenizer']

SP_TOKENIZERS = ['SentencePieceTokenizer', 'GPTSentencePieceTokenizer', 'Llama2Tokenizer']


def build_tokenizer(args):
    """ """
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
        kwargs['use_fast'] = args.tokenizer_hf_use_fast
        kwargs['trust_remote_code'] = args.trust_remote_code
        kwargs['include_special_tokens'] = args.tokenizer_hf_include_special_tokens
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
        kwargs['use_fast'] = args.tokenizer_hf_use_fast
        kwargs['trust_remote_code'] = args.trust_remote_code
        kwargs['include_special_tokens'] = args.tokenizer_hf_include_special_tokens
    elif args.tokenizer_type == 'NullTokenizer':
        tokenizer_library = 'null'
        metadata = {'library': tokenizer_library}
        if args.vocab_size:
            kwargs['vocab_size'] = args.vocab_size
        tokenizer = MegatronTokenizer.from_pretrained(metadata_path=metadata, **kwargs)

        return tokenizer

    if args.tokenizer_metadata:
        metadata = args.tokenizer_metadata
    else:
        metadata = {'library': tokenizer_library}
    tokenizer = MegatronTokenizer.from_pretrained(
        tokenizer_path=tokenizer_path, metadata_path=metadata, **kwargs
    )

    return tokenizer
