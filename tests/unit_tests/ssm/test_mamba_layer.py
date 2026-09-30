# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.wide_residual_mamba_layer import WideResidualMambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestMambaLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=1,
            num_attention_heads=1,
            layernorm_epsilon=1e-6,
            use_cpu_initialization=True,
        )
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        # Use an explicit norm so the test can verify the configured epsilon.
        mamba_submodules = replace(
            hybrid_stack_spec.submodules.mamba_layer.submodules, norm=WrappedTorchNorm
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.layer = MambaLayer(transformer_config, mamba_submodules, pg_collection=pg_collection)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_configured_layernorm_epsilon(self):
        assert self.layer.norm.eps == self.layer.config.layernorm_epsilon

    def test_post_core_preserves_ordinary_argument_binding(self, monkeypatch):
        ssm_output, residual, projected = (torch.randn(2, 3, 256) for _ in range(3))
        mixer_output = (projected, None)
        project = Mock(return_value=mixer_output)
        write = Mock(return_value=projected)
        monkeypatch.setattr(self.layer.mixer, "forward_post_core_attn", project)
        monkeypatch.setattr(self.layer, "_apply_mixer_bda", write)
        inference_context, padding_mask = object(), object()

        for args, kwargs in (
            ((), {}),
            ((inference_context, padding_mask), {}),
            ((), dict(inference_context=inference_context, padding_mask=padding_mask)),
        ):
            assert (
                self.layer.forward_post_core_attn(ssm_output, residual, *args, **kwargs)
                is projected
            )
            project.assert_called_with(ssm_output)
            write.assert_called_with(mixer_output, residual)

        with pytest.raises(TypeError, match="multiple values.*inference_context"):
            self.layer.forward_post_core_attn(
                ssm_output, residual, inference_context, inference_context=inference_context
            )

    @pytest.mark.parametrize("fp32_residual", [False, True], ids=["bf16", "fp32"])
    def test_wide_mamba_two_stage_forward_backward(self, fp32_residual, monkeypatch):
        config = replace(
            self.layer.config,
            wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
            hidden_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp32_residual_connection=fp32_residual,
        )
        layer = (
            WideResidualMambaLayer(
                config,
                self.layer.submodules_config,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                ),
            )
            .cuda()
            .bfloat16()
        )
        dtype = torch.float32 if fp32_residual else torch.bfloat16
        hidden_states = torch.randn(16, 2, 3 * config.hidden_size, device="cuda", dtype=dtype)
        atomic_input = hidden_states.detach().clone().requires_grad_(True)
        atomic_output = layer(atomic_input)
        atomic_output.float().square().mean().backward()
        atomic_grads = {
            name: param.grad.clone()
            for name, param in layer.named_parameters()
            if param.grad is not None
        }

        layer.zero_grad(set_to_none=True)
        split_input = hidden_states.detach().clone().requires_grad_(True)
        stage_state = layer.forward_pre_attn_and_core_attn(split_input)
        assert len(stage_state) == 3
        split_output = layer.forward_post_core_attn(*stage_state)
        split_output.float().square().mean().backward()
        split_grads = {
            name: param.grad for name, param in layer.named_parameters() if param.grad is not None
        }

        torch.testing.assert_close(split_output, atomic_output, rtol=0, atol=0)
        torch.testing.assert_close(split_input.grad, atomic_input.grad, rtol=0, atol=0)
        assert split_grads.keys() == atomic_grads.keys()
        for name, grad in split_grads.items():
            torch.testing.assert_close(grad, atomic_grads[name], rtol=0, atol=0, msg=name)

        with pytest.raises(TypeError, match="connection_state"):
            layer.forward_post_core_attn(*stage_state[:2])

        write = Mock(return_value=split_output)
        monkeypatch.setattr(layer, "_apply_mixer_bda", write)
        recompute_context = object()
        assert (
            layer.forward_post_core_attn(
                *stage_state, residual_stream_recompute_context=recompute_context
            )
            is split_output
        )
        assert write.call_args.args[1] is stage_state[1]
        assert write.call_args.args[2] is stage_state[2]
        assert write.call_args.kwargs["recompute_context"] is recompute_context

    def test_gpu_forward(self):
        layer = self.layer
        layer.cuda()
        layer.eval()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, layer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = layer(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == layer.config.hidden_size
        assert output.dtype == torch.float32
