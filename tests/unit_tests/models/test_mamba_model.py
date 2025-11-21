# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
from datetime import timedelta

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.contexts import BaseInferenceContext, StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import (
    divide,
    get_mamba_inference_state_config_from_model,
    is_fa_min_version,
    is_torch_min_version,
)
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

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
    )
    @pytest.mark.parametrize("tp_size,cp_size,pp_size", [(2, 1, 4), (1, 1, 8), (8, 1, 1)])
    def test_with_custom_process_groups(self, tmp_path, tp_size, cp_size, pp_size):
        """Test MambaModel with custom process groups."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=pp_size,
        )

        # Create device mesh for custom process groups
        assert torch.distributed.get_world_size() == 8, "Test requires 8 GPUs"

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimensions tp, cp, pp (reversed from device mesh order)
        grid = HyperCommGrid([tp_size, cp_size, pp_size], ["tp", "cp", "pp"])

        pp_group = grid.create_pg("pp")
        cp_group = grid.create_pg("cp")
        tp_group = grid.create_pg("tp")
        embd_group_ranks = parallel_state.default_embedding_ranks(
            torch.distributed.get_process_group_ranks(pp_group)
        )
        embd_group = torch.distributed.new_group(
            ranks=embd_group_ranks, timeout=timedelta(minutes=30)
        )

        # Create model with custom process groups
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg_collection = ProcessGroupCollection(
            tp=tp_group, cp=cp_group, pp=pp_group, embd=embd_group
        )

        # Configure model with appropriate sizes for parallelism
        model_config = TransformerConfig(
            num_layers=3 * pp_size,  # Scale layers with PP size
            hidden_size=256 * tp_size,
            num_attention_heads=4 * tp_size,  # Scale heads with TP size
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=pp_size,
            pipeline_dtype=torch.bfloat16,
        )

        model = MambaModel(
            config=model_config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=128,
            max_sequence_length=4,
            hybrid_attention_ratio=0.3,
            hybrid_mlp_ratio=0.3,
            pg_collection=pg_collection,
        )

        # Basic forward test
        micro_batch_size = 2
        sequence_length = model.max_sequence_length

        model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == divide(model.vocab_size, tp_size)


class TestMambaWithDynamicInference:
    """Tests MambaModel with dynamic inference."""

    @torch.inference_mode()
    def setup_method(self, method):
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        transformer_config = TransformerConfig(
            num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
            hidden_size=256,
            mamba_num_heads=16,
            num_attention_heads=16,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid",
            fp8_recipe="tensorwise",
        )

        self.mamba_model = MambaModel(
            config=transformer_config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=128,
            max_sequence_length=DynamicInferenceContext.TOKEN_ROUNDER,
            hybrid_attention_ratio=0.3,
            hybrid_mlp_ratio=0.3,
            parallel_output=True,
        )
        self.mamba_model = Float16Module(self.mamba_model.config, self.mamba_model)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_dynamic_inference_padding_with_fp8(self):
        """
        Tests that logits for padded tokens are zeroed out for fp8 inference.
        """
        self.mamba_model.cuda()
        self.mamba_model.eval()
        config = self.mamba_model.config

        # Mamba specific: Retrieve inference state config
        mamba_inference_state_config = get_mamba_inference_state_config_from_model(
            self.mamba_model.module
        )

        inference_context = DynamicInferenceContext(
            params_dtype=config.params_dtype,
            num_layers=config.num_layers,
            kv_channels=config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            max_sequence_length=self.mamba_model.module.max_sequence_length,
            buffer_size_gb=1.0,
            block_size_tokens=256,
            materialize_only_last_token_logits=False,
            mamba_inference_state_config=mamba_inference_state_config,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        # Add a request with 10 tokens. Since 10 is not a multiple of the rounder,
        # this will create padding up to the padded length.
        active_token_count = 10
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.arange(0, active_token_count, dtype=torch.long, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        inference_context.add_request(request)

        # Prepares the context, including calculating the padded token count.
        inference_context.initialize_attention_state()

        assert inference_context.active_token_count == active_token_count
        assert inference_context.padded_active_token_count == DynamicInferenceContext.TOKEN_ROUNDER

        # Prepare inputs for the forward pass.
        padded_token_count = inference_context.padded_active_token_count
        input_ids, position_ids = inference_context.current_input_and_position_ids()

        # Run the forward pass with inference parameters.
        logits = self.mamba_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            inference_context=inference_context,
            runtime_gather_output=True,
        )

        # Verify the output shape.
        assert logits.shape[0] == 1
        assert logits.shape[1] == padded_token_count
        assert logits.shape[2] == self.mamba_model.module.vocab_size

        # Extract the logits corresponding to the padding tokens.
        padding_start_idx = inference_context.active_token_count
        padding_end_idx = inference_context.padded_active_token_count
        padding_logits = logits[0, padding_start_idx:padding_end_idx, :]

        # Assert that all padding logits are zero.
        assert torch.all(padding_logits == 0.0), "Logits for padding tokens are not all zero."
