# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import json
import os
import sys
import torch
import types

from schema_core import get_model_schema
from loader_base import MegatronCheckpointLoaderBase


def add_arguments(parser):
    """Add command-line arguments relevant to Megatron model loading."""
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original size of vocab; if specified, trims padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to a vocab file. If specified, determines vocab size to trim padding.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--position-embedding-type',
                       type=str,
                       default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Type of position embedding.')
    group.add_argument('--loader-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


class MegatronCheckpointLoaderLLM(MegatronCheckpointLoaderBase):
    """
    Orchestrates loading a Megatron checkpoint and sending
    model parameters over a given multiprocessing queue.

    Args:
        args: argparse Namespace with Megatron checkpoint configurations.
        queue: A multiprocessing.Queue (or similar) used to send out loaded tensors.
    """

    def build_sys_argv(self):
        """
        Construct a sys.argv list for Megatron's argument parser.
        This centralizes the hack of overwriting sys.argv.
        """

        return [
            *super().build_sys_argv(),
            '--position-embedding-type', self.args.position_embedding_type,
        ]

    def import_model_provider(self):
        """Return the correct model_provider function depending on GPT vs. BERT."""
        if self.args.model_type == 'GPT':
            from pretrain_gpt import model_provider
            return model_provider
        elif self.args.model_type == 'BERT':
            from pretrain_bert import model_provider
            return model_provider
        else:
            raise Exception(f"Unrecognized model type: {self.args.model_type}")


    def send_model_over_queue(self):
        self.send_metadata_over_queue()
        # Model schema.
        schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
        )
        self.send_llm_over_queue(schema)
        self.queue.put("done")


def load_checkpoint(queue, args):
    """
    Required top-level function that creates the loader,
    calls its .load(), and handles exceptions by signaling 'exit'.
    """
    loader = MegatronCheckpointLoaderLLM(args, queue)
    try:
        loader.load()
    except Exception as e:
        queue.put("exit")
        raise e
