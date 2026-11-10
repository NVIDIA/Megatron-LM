# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestMultimodalProjector:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        mlp_layer_spec = get_mlp_module_spec().keywords['submodules']

        affine_layer_spec = MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=None)
        self.mlp = MultimodalProjector(
            config=transformer_config,
            submodules=mlp_layer_spec,
            projector_type="mlp",
            input_size=1024,
        )
        self.affine = MultimodalProjector(
            config=transformer_config,
            submodules=affine_layer_spec,
            projector_type="affine",
            input_size=1024,
        )

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

    def test_zero_token_gtp_lane_uses_zero_padding(self, monkeypatch):
        class RecordingEncoder(torch.nn.Module):
            def forward(self, hidden_states):
                self.input = hidden_states.clone()
                return hidden_states.new_zeros(hidden_states.shape[0], 32), None

        projector = MultimodalProjector.__new__(MultimodalProjector)
        torch.nn.Module.__init__(projector)
        projector.config = SimpleNamespace(fp8="e4m3", fp8_recipe="mxfp8", gtp_weight_remat_size=2)
        projector.encoder = RecordingEncoder()
        monkeypatch.setattr(
            "megatron.core.models.vision.multimodal_projector.get_fp8_context",
            lambda config: nullcontext(),
        )
        monkeypatch.setattr(
            "megatron.core.models.vision.multimodal_projector.get_fp8_align_size", lambda recipe: 32
        )

        output = projector(torch.empty(0, 16))

        assert projector.encoder.input.shape == (32, 16)
        assert torch.count_nonzero(projector.encoder.input) == 0
        assert output.shape == (0, 32)

    def test_zero_token_gtp_lane_completes_forward_backward(self):
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

        if torch.distributed.get_world_size() < 2:
            pytest.skip("requires at least two torchrun ranks")
        if not HAVE_GTP:
            pytest.skip("GTP requires a supported Transformer Engine version")

        Utils.initialize_model_parallel(gtp_remat_size=2)
        model_parallel_cuda_manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp", "pp", "gtp_remat", "expt_gtp_remat"]
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            add_bias_linear=False,
            gated_linear_unit=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            gtp_weight_remat_size=2,
            use_cpu_initialization=False,
        )
        projector = MultimodalProjector(
            config=config,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=None),
            projector_type="affine",
            input_size=64,
            pg_collection=pg_collection,
        ).cuda()
        for parameter in projector.parameters():
            if getattr(parameter, "is_gtp_weight_remat", False):
                parameter.main_grad = torch.zeros_like(parameter, dtype=torch.bfloat16)

        token_count = 0 if parallel_state.get_gtp_weight_remat_rank() == 0 else 32
        hidden_states = torch.randn(
            token_count, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        output = projector(hidden_states)
        output.sum().backward()

        assert output.shape == (token_count, 64)
        assert hidden_states.grad is not None
