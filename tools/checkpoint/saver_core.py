# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import torch
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from schema_core import get_model_schema
from saver_base import MegatronCheckpointSaverBase


def add_arguments(parser):
    group = parser.add_argument_group(title='M-Core saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Target expert model parallel size, default to 1')
    group.add_argument('--saver-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


class MegatronCheckpointSaverLLM(MegatronCheckpointSaverBase):
    def import_model_provider(self):
        try:
            from megatron.core.enums import ModelType
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        if self.md.model_type == 'GPT':
            from pretrain_gpt import model_provider
            self.model_provider = model_provider
            self.margs.model_type = ModelType.encoder_or_decoder
        elif self.md.model_type == 'BERT':
            from pretrain_bert import model_provider
            self.model_provider = model_provider
            self.margs.model_type = ModelType.encoder_or_decoder
        else:
            raise Exception(f'unrecognized model type: {self.args.model_type}')

    def receive_model(self):
        # Model schema.
        schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
        )
        self.receive_lm(schema)

def save_checkpoint(queue, args):
    """
    Required top-level function that creates the saver and calls its .save().
    """
    saver = MegatronCheckpointSaverLLM(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e
