# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts import BaseInferenceContext, StaticInferenceContext
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import divide, is_torch_min_version
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
        from torch.distributed import DeviceMesh

        mesh = torch.distributed.init_device_mesh(
            "cuda", (pp_size, cp_size, tp_size), mesh_dim_names=["pp", "cp", "tp"]
        )
        pp_group = mesh.get_group(mesh_dim="pp")
        cp_group = mesh.get_group(mesh_dim="cp")
        tp_group = mesh.get_group(mesh_dim="tp")

        # Create model with custom process groups
        from megatron.core.process_groups_config import ModelCommProcessGroups

        model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group, pp=pp_group)

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
            model_comm_pgs=model_comm_pgs,
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
