# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility tests for MTP and MIMO with streamwise wide residuals."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid import hybrid_block as hybrid_block_module
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from tests.unit_tests.test_utilities import Utils


def _wide_config(
    *, num_layers: int, hidden_size: int, mtp_num_layers: int | None = None, with_moe: bool = False
):
    moe_config = {}
    if with_moe:
        moe_config = {
            "ffn_hidden_size": 2 * hidden_size,
            "moe_ffn_hidden_size": 2 * hidden_size,
            "num_moe_experts": 8,
            "moe_router_topk": 2,
            "moe_grouped_gemm": True,
            "add_bias_linear": False,
        }
    return TransformerConfig(
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
        mtp_loss_scaling_factor=0.1,
        mtp_use_repeated_layer=mtp_num_layers is not None,
        hidden_size=hidden_size,
        num_attention_heads=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cpu_initialization=True,
        recompute_granularity="selective",
        recompute_modules=["residual_stream"],
        residual_stream_recompute_num_layers=1,
        wide_residual=WideResidualConfig(
            num_streams=3,
            streamwise_sigmoid_init_scale=0.01,
            learned_retention=True,
            retention_init=0.999,
            retention_max_forget=0.10,
        ),
        **moe_config,
    )


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestWideResidualMTPAndMIMO:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        MTPLossLoggingHelper.tracker = {}

    def teardown_method(self):
        MTPLossLoggingHelper.tracker = {}
        Utils.destroy_model_parallel()

    def test_gpt_mtp_keeps_auxiliary_layer_at_backbone_width(self):
        config = _wide_config(num_layers=2, hidden_size=64, mtp_num_layers=2)
        layer_spec = get_gpt_layer_local_spec()
        model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            mtp_block_spec=get_gpt_mtp_block_spec(
                config=config, spec=layer_spec, use_transformer_engine=False
            ),
            vocab_size=128,
            max_sequence_length=4,
            position_embedding_type="none",
        ).cuda()
        input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device="cuda")
        position_ids = torch.arange(4, device="cuda").unsqueeze(0).expand(2, -1)
        labels = input_ids.roll(-1, dims=1)
        loss_mask = torch.ones_like(labels, dtype=torch.float32)

        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )
        loss.mean().backward()

        mtp_layer = model.mtp.layers[0]
        assert model.decoder.residual_stream_readout is not None
        assert model.decoder.layers[0].residual_connection_self_attn is not None
        assert len(model.mtp.layers) == 1
        assert mtp_layer.mtp_model_layer.residual_connection_self_attn is None
        assert mtp_layer.mtp_model_layer.residual_connection_mlp is None
        assert mtp_layer.mtp_model_layer.residual_stream_hidden_size == config.hidden_size
        assert mtp_layer.eh_proj.weight.grad is not None
        assert model.embedding.word_embeddings.weight.grad is not None
        assert model.decoder.residual_stream_readout.exit_map.logit.grad is not None
        assert loss.shape == input_ids.shape

    def test_hybrid_mtp_replays_only_the_main_wide_decoder(self, monkeypatch):
        replay_plan_calls = []
        build_replay_plan = hybrid_block_module.build_residual_stream_recompute_plan

        def track_replay_plan(num_layers, block_size):
            replay_plan_calls.append((num_layers, block_size))
            return build_replay_plan(num_layers, block_size)

        monkeypatch.setattr(
            hybrid_block_module, "build_residual_stream_recompute_plan", track_replay_plan
        )
        config = _wide_config(num_layers=2, hidden_size=256, mtp_num_layers=2, with_moe=True)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=4,
            hybrid_layer_pattern="ME/*E/*E",
            position_embedding_type="none",
        ).cuda()
        input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device="cuda")
        position_ids = torch.arange(4, device="cuda").unsqueeze(0).expand(2, -1)
        labels = input_ids.roll(-1, dims=1)
        loss_mask = torch.ones_like(labels, dtype=torch.float32)

        loss = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
        )
        loss.mean().backward()

        mtp_stack = model.mtp.layers[0].mtp_model_layer
        assert replay_plan_calls == [(len(model.decoder.layers), 1)]
        assert model.decoder.uses_wide_residual_stream
        assert model.decoder.residual_stream_readout is not None
        assert model.decoder.layers[0].residual_connection is not None
        assert model.decoder.layers[1].residual_connection_mlp is not None
        assert not mtp_stack.uses_wide_residual_stream
        assert mtp_stack.residual_stream_readout is None
        assert mtp_stack.layers[0].is_mtp_layer
        assert mtp_stack.layers[0].residual_connection_self_attn is None
        assert mtp_stack.layers[1].is_mtp_layer
        assert mtp_stack.layers[1].residual_connection_mlp is None
        assert mtp_stack.layers[0].residual_stream_hidden_size == config.hidden_size
        assert mtp_stack.layers[1].residual_stream_hidden_size == config.hidden_size
        assert model.mtp.layers[0].eh_proj.weight.grad is not None
        assert model.embedding.word_embeddings.weight.grad is not None
        assert loss.shape == input_ids.shape

    def test_mimo_composes_at_backbone_width_before_decoder_expansion(self):
        config = _wide_config(num_layers=1, hidden_size=64)
        layer_spec = get_gpt_layer_local_spec()
        language_spec = ModuleSpec(
            module=GPTModel,
            params={
                "config": config,
                "transformer_layer_spec": layer_spec,
                "vocab_size": 128,
                "max_sequence_length": 4,
                "position_embedding_type": "none",
            },
        )
        model = MimoModel(
            MimoModelConfig(
                language_model_spec=language_spec, modality_submodules_spec={}, special_token_ids={}
            )
        ).cuda()
        input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], device="cuda")
        position_ids = torch.arange(4, device="cuda").unsqueeze(0).expand(2, -1)
        labels = input_ids.roll(-1, dims=1)
        loss_mask = torch.ones_like(labels, dtype=torch.float32)

        with patch.object(
            model.language_model.decoder, "forward", wraps=model.language_model.decoder.forward
        ) as decoder_forward:
            loss, returned_loss_mask = model(
                input_ids=input_ids, position_ids=position_ids, labels=labels, loss_mask=loss_mask
            )
        loss.mean().backward()

        decoder_input = decoder_forward.call_args.kwargs["hidden_states"]
        assert decoder_input.shape == (4, 2, config.hidden_size)
        assert model.language_model.decoder.residual_stream_readout is not None
        assert model.language_model.embedding.word_embeddings.weight.grad is not None
        assert model.language_model.decoder.residual_stream_readout.exit_map.logit.grad is not None
        assert returned_loss_mask is loss_mask
        assert loss.shape == input_ids.shape
