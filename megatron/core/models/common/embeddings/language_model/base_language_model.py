import logging

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule


class BaseLanguageModel(MegatronModule):
    def __init__(self, config):
        super(BaseLanguageModel, self).__init__(config=config)

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt'
        self.decoder.set_input_tensor(input_tensor[0])

    def compute_language_model_loss(self, labels, logits):
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss

    def initialize_last_stage_with_word_embeddings(self, llm_model):

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism and sharing word
        # embeddings. Nothing to do if we aren't sharing weights or aren't using
        # pipeline parallelism.
        if not self.share_embeddings_and_output_weights or (self.pre_process and self.post_process):
            return

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.output_layer.weight.data.fill_(0)
            self.output_layer.weight.shared = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                weight = self.shared_embedding_or_output_weight()
                torch.distributed.all_reduce(
                    weight.data, group=parallel_state.get_embedding_group()
                )

        elif not getattr(llm_model, "embedding_warning_printed", False):
            logging.getLogger(__name__).warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            llm_model.embedding_warning_printed = True
