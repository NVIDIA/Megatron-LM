# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear


class TestMultimodalProjector:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True)
        mlp_layer_spec = _get_mlp_module_spec().submodules
        
        affine_layer_spec = MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=None,
            )
        self.mlp = MultimodalProjector(config = transformer_config, submodules = mlp_layer_spec, projector_type = "mlp", input_size = 1024)
        self.affine = MultimodalProjector(config = transformer_config, submodules = affine_layer_spec, projector_type = "affine", input_size = 1024)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.mlp, MultimodalProjector)
        assert isinstance(self.affine, MultimodalProjector)

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == 280896

        num_weights = sum([p.numel() for p in self.affine.parameters()])
        assert num_weights == 65600

    def test_forward(self):
        self.mlp.cuda()
        self.affine.cuda()

        image_projection = torch.zeros((2, 1024)).cuda()

        logits = self.mlp.forward(image_projection)
        assert len(logits) == 2
        assert logits.shape == torch.Size([2, 64])

        logits = self.affine.forward(image_projection)
        assert len(logits) == 2
        assert logits.shape == torch.Size([2, 64])

    def test_save_load(self, tmp_path):
        path = tmp_path / "mlp.pt"
        torch.save(self.mlp.state_dict(), path)

        self.mlp.load_state_dict(torch.load(path))

        path = tmp_path / "affine.pt"
        torch.save(self.affine.state_dict(), path)

        self.affine.load_state_dict(torch.load(path))

