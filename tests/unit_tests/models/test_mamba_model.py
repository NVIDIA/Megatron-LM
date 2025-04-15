# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts import BaseInferenceContext, StaticInferenceContext
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestMambaModel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        model_config = TransformerConfig(
            num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
            hidden_size=256,  # The Mamba layer places several constraints on this
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        self.model = MambaModel(
            config=model_config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_attention_ratio=0.3,
            hybrid_mlp_ratio=0.3,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, MambaModel)

        assert self.model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1774872

    def test_set_input_tensor(self):
        config: TransformerConfig = self.model.config
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape[0] == sequence_length
        assert self.model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.model.decoder.input_tensor.shape[2] == config.hidden_size

    def test_forward(self):
        config: TransformerConfig = self.model.config
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        self.model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.model.vocab_size

    def test_inference(self):
        config: TransformerConfig = self.model.config
        micro_batch_size = 2
        inference_context: BaseInferenceContext = StaticInferenceContext(
            max_batch_size=micro_batch_size, max_sequence_length=self.model.max_sequence_length
        )
        prompt_length = self.model.max_sequence_length - 1

        self.model.cuda()

        # load-context/first-output-token, step/generate
        for offset in (0, prompt_length):
            if offset == 0:
                sequence_length = prompt_length
            else:
                sequence_length = 1
            inference_context.sequence_len_offset = offset

            data = list(range(sequence_length))
            input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
            position_ids = (
                torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
            )
            attention_mask = torch.ones(
                (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
            ).cuda()

            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inference_context=inference_context,
            )

            assert logits.shape[0] == micro_batch_size
            assert logits.shape[1] == sequence_length
            assert logits.shape[2] == self.model.vocab_size

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))

    def test_layer_numbers(self):
        """
        The layer numbers should start at one (for the embedding # layer) and go up
        incrementally from there. This is required for PEFT to work.
        """
        model = self.model
        for expected, layer in enumerate(model.decoder.layers, start=1):
            assert expected == layer.layer_number, "layer numbers are incorrect"
