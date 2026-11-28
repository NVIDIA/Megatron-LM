# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import sys

from megatron.post_training.arguments import add_modelopt_args
from megatron.training.arguments import parse_args, validate_args


def make_modelopt_test_args(extra_argv=(), **overrides):
    sys.argv = ["modelopt_post_training_test", *extra_argv]
    args = parse_args(extra_args_provider=add_modelopt_args)
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 8
    args.micro_batch_size = 1
    args.seq_length = 128
    args.max_position_embeddings = 128
    args.padded_vocab_size = 32000
    args.position_embedding_type = "rope"
    args.apply_rope_fusion = False
    for name, value in overrides.items():
        setattr(args, name, value)
    validate_args(args)
    return args
