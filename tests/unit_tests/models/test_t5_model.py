# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
import pytest

import torch
import megatron.core.parallel_state as ps

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.T5.t5_model import T5Model
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.T5.t5_spec import (get_t5_encoder_with_transformer_engine_block_spec,
                                            get_t5_decoder_with_transformer_engine_block_spec,
                                            get_t5_encoder_with_local_block_spec,
                                            get_t5_decoder_with_local_block_spec)

class TestT5Model:

    def setup_method(self, method):
        Utils.initialize_model_parallel(2, 2)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=12, hidden_size=768, num_attention_heads=12, kv_channels=64, ffn_hidden_size=3072,
            use_cpu_initialization=True, pipeline_dtype=torch.bfloat16,
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2,
        )
        rank = ps.get_pipeline_model_parallel_rank()
        world_size = Utils.world_size
        en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(12)
        de_block_spec = get_t5_decoder_with_transformer_engine_block_spec(12)

        first_decoder_rank = 1
        pre_process = rank == 0 or rank == first_decoder_rank
        post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
        add_encoder = ps.is_inside_encoder(rank)
        add_decoder = ps.is_inside_decoder(rank)

        encoder_config = deepcopy(transformer_config)
        encoder_config.pipeline_model_parallel_size = 1

        self.t5_model = T5Model(
            encoder_config=encoder_config, config=transformer_config, transformer_encoder_layer_spec=en_block_spec,
            transformer_decoder_layer_spec=de_block_spec,  vocab_size=29184, max_sequence_length=4,
            pre_process=pre_process, post_process=post_process,
            add_encoder=add_encoder, add_decoder=add_decoder,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.t5_model, T5Model)

        assert self.t5_model.max_sequence_length == 4

    def test_set_input_tensor(self):
        config: TransformerConfig = self.t5_model.config
        sequence_length = self.t5_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.t5_model.set_input_tensor(input_tensor)

        assert self.t5_model.encoder.input_tensor.shape[0] == sequence_length
        assert self.t5_model.encoder.input_tensor.shape[1] == micro_batch_size
        assert self.t5_model.encoder.input_tensor.shape[2] == config.hidden_size

    def test_post_process_forward(self):
        config: TransformerConfig = self.t5_model.config
        sequence_length = self.t5_model.max_sequence_length
        micro_batch_size = 2

        self.t5_model.cuda()

        data = list(range(sequence_length))
        encoder_input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        decoder_input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        encoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()
        decoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()
        encoder_decoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()

        output = self.t5_model.forward(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_decoder_attn_mask=encoder_decoder_attn_mask
        )
        if self.t5_model.post_process:
            logits = output
            assert logits.shape[0] == micro_batch_size
            assert logits.shape[1] == sequence_length
            assert logits.shape[2] == self.t5_model.vocab_size // 2
        else:
            encoder_hidden_states = output
            assert encoder_hidden_states.shape[0] == sequence_length
            assert encoder_hidden_states.shape[1] == micro_batch_size
            assert encoder_hidden_states.shape[2] == config.hidden_size


    def test_forward_output_encoder_hidden_only(self):
        config: TransformerConfig = self.t5_model.config
        sequence_length = self.t5_model.max_sequence_length
        micro_batch_size = 2

        self.t5_model.cuda()

        data = list(range(sequence_length))
        encoder_input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        decoder_input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        encoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()
        decoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()
        encoder_decoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()

        encoder_hidden_states = self.t5_model.forward(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_decoder_attn_mask=encoder_decoder_attn_mask,
            output_encoder_hidden_only=True
        )
        assert encoder_hidden_states.shape[0] == sequence_length
        assert encoder_hidden_states.shape[1] == micro_batch_size
        assert encoder_hidden_states.shape[2] == config.hidden_size

    def test_forward_with_encoder_hidden_states(self):
        config: TransformerConfig = self.t5_model.config
        sequence_length = self.t5_model.max_sequence_length
        micro_batch_size = 2

        self.t5_model.cuda()

        data = list(range(sequence_length))
        encoder_input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        decoder_input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        encoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()
        decoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()
        encoder_decoder_attn_mask = torch.ones((1, sequence_length, sequence_length), dtype=bool).cuda()
        encoder_hidden_states = torch.zeros((sequence_length, micro_batch_size, config.hidden_size), dtype=torch.float32).cuda()

        output = self.t5_model.forward(
            encoder_input_ids=None,
            decoder_input_ids=decoder_input_ids,
            encoder_attn_mask=encoder_attn_mask,
            decoder_attn_mask=decoder_attn_mask,
            encoder_decoder_attn_mask=encoder_decoder_attn_mask,
            encoder_hidden_states=encoder_hidden_states
        )
        if self.t5_model.post_process:
            logits = output
            assert logits.shape[0] == micro_batch_size
            assert logits.shape[1] == sequence_length
            assert logits.shape[2] == self.t5_model.vocab_size // 2
        else:
            encoder_hidden_states = output
            assert encoder_hidden_states.shape[0] == sequence_length
            assert encoder_hidden_states.shape[1] == micro_batch_size
            assert encoder_hidden_states.shape[2] == config.hidden_size

    def test_no_post_process_forward(self):
        pass

    def test_no_preprocess_forward(self):
        pass

    def test_state_dict_for_save_checkpoint(self):
        pass

    def test_load_state_dict(self):
        pass

