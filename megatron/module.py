# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron Module"""

import torch

from megatron import get_args
from megatron import mpu


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module."""

    def __init__(self):
        super(MegatronModule, self).__init__()

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)


class PipelinedMegatronModule(MegatronModule):
    """Pipelining specific extensions of MegatronModule."""

    def __init__(self, share_word_embeddings=True):
        super(PipelinedMegatronModule, self).__init__()
        args = get_args()
        self.share_word_embeddings = share_word_embeddings

    def word_embeddings_weight(self):
        if mpu.is_pipeline_first_stage():
            return self.language_model.embedding.word_embeddings.weight
        if mpu.is_pipeline_last_stage():
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last stage, '
                                'but share_word_embeddings is false')
            return self.word_embeddings.weight
        raise Exception('word_embeddings_weight() should be '
                        'called for first and last stage only')

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_word_embeddings is false')
        # Parameters are shared between the word embeddings layer, and the heads at
        # the end of the model. In a pipelined setup with more than one stage, the
        # initial embedding layer and the head are on different workers, so we do
        # the following:
        # 1. Create a second copy of word_embeddings on the last stage, with initial
        #    parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that the
        #    two copies of word_embeddings start off with the same parameter values.
        # 3. In the training loop, before an all-reduce between the grads of the two
        #    word_embeddings layers to ensure that every applied weight update is the
        #    same on both stages.
        if mpu.is_pipeline_last_stage():
            if not mpu.is_pipeline_first_stage():
                self._word_embeddings_for_head_key = 'word_embeddings_for_head'
                # If first and last stages are different, set word_embeddings
                # weights to 0 here, then copy first stage's weights using all_reduce
                # below.
                self.word_embeddings = mpu.VocabParallelEmbedding(
                    args.padded_vocab_size, args.hidden_size,
                    init_method=init_method_normal(args.init_method_std))
                self.word_embeddings.weight.data.fill_(0)
        # Ensure that first and last stages have the same initial parameter values.
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            torch.distributed.all_reduce(self.word_embeddings_weight().data,
                                         group=mpu.get_embedding_group())
