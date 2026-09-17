# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compatibility tests for MTP and MIMO with streamwise wide residuals."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_mtp_block_spec,
    get_gpt_wide_residual_layer_local_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid import hybrid_block as hybrid_block_module
from megatron.core.models.hybrid.hybrid_layer_specs import wide_residual_hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.wide_residual_mamba_layer import WideResidualMambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.wide_residual_layer import WideResidualTransformerLayer
from tests.unit_tests.test_utilities import Utils


def _wide_config(
    *,
    num_layers: int,
    hidden_size: int,
    mtp_num_layers: int | None = None,
    with_moe: bool = False,
    with_shortcut: bool = False,
    fp32_residual_connection: bool = False,
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
    if with_shortcut:
        moe_config.update(
            moe_shortcut_connection=True,
            moe_shortcut_post_norm=True,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shared_expert_intermediate_size=2 * hidden_size,
        )
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
        bf16=fp32_residual_connection,
        params_dtype=torch.bfloat16 if fp32_residual_connection else torch.float32,
        fp32_residual_connection=fp32_residual_connection,
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


def _move_model_to_configured_dtype(model, config):
    """Convert this test model to its configured parameter dtype."""

    return model.cuda().to(dtype=config.params_dtype)


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

    @pytest.mark.parametrize(
        "fp32_residual_connection", [False, True], ids=["native-residual", "fp32-residual"]
    )
    def test_gpt_mtp_keeps_auxiliary_layer_at_backbone_width(self, fp32_residual_connection):
        config = _wide_config(
            num_layers=2,
            hidden_size=64,
            mtp_num_layers=2,
            fp32_residual_connection=fp32_residual_connection,
        )
        layer_spec = get_gpt_wide_residual_layer_local_spec()
        model = _move_model_to_configured_dtype(
            GPTModel(
                config=config,
                transformer_layer_spec=layer_spec,
                mtp_block_spec=get_gpt_mtp_block_spec(
                    config=config, spec=layer_spec, use_transformer_engine=False
                ),
                vocab_size=128,
                max_sequence_length=4,
                position_embedding_type="none",
            ),
            config,
        )
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
        assert type(model.decoder.layers[0]) is WideResidualTransformerLayer
        assert model.decoder.layers[0].residual_connection_self_attn is not None
        assert len(model.mtp.layers) == 1
        assert type(mtp_layer.mtp_model_layer) is TransformerLayer
        assert mtp_layer.mtp_model_layer._get_self_attention_residual_connection() is None
        assert mtp_layer.mtp_model_layer._get_mlp_residual_connection() is None
        assert mtp_layer.eh_proj.weight.grad is not None
        assert model.embedding.word_embeddings.weight.grad is not None
        assert model.decoder.residual_stream_readout.exit_map.logit.grad is not None
        assert loss.shape == input_ids.shape

    @pytest.mark.parametrize(
        "fp32_residual_connection", [False, True], ids=["native-residual", "fp32-residual"]
    )
    def test_hybrid_mtp_replays_only_the_main_wide_decoder(
        self, monkeypatch, fp32_residual_connection
    ):
        replay_plan_calls = []
        build_replay_plan = hybrid_block_module.build_residual_stream_recompute_plan

        def track_replay_plan(num_layers, block_size, *, atomic_layer_pairs=()):
            atomic_layer_pairs = tuple(atomic_layer_pairs)
            replay_plan_calls.append((num_layers, block_size, atomic_layer_pairs))
            return build_replay_plan(num_layers, block_size, atomic_layer_pairs=atomic_layer_pairs)

        monkeypatch.setattr(
            hybrid_block_module, "build_residual_stream_recompute_plan", track_replay_plan
        )
        config = _wide_config(
            num_layers=2,
            hidden_size=256,
            mtp_num_layers=2,
            with_moe=True,
            fp32_residual_connection=fp32_residual_connection,
        )
        model = _move_model_to_configured_dtype(
            HybridModel(
                config=config,
                hybrid_stack_spec=wide_residual_hybrid_stack_spec,
                vocab_size=128,
                max_sequence_length=4,
                hybrid_layer_pattern="ME/ME/ME",
                position_embedding_type="none",
            ),
            config,
        )
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
        assert replay_plan_calls == [
            (model.decoder.num_layers_per_pipeline_rank, 1, model.decoder._shortcut_layer_pairs())
        ]
        assert model.decoder.uses_wide_residual_stream
        assert model.decoder.residual_stream_readout is not None
        assert type(model.decoder.layers[0]) is WideResidualMambaLayer
        assert model.decoder.layers[0].residual_connection is not None
        assert type(model.decoder.layers[1]) is WideResidualTransformerLayer
        assert model.decoder.layers[1].residual_connection_mlp is not None
        assert not mtp_stack.uses_wide_residual_stream
        assert mtp_stack.residual_stream_readout is None
        assert mtp_stack.layers[0].is_mtp_layer
        assert type(mtp_stack.layers[0]) is MambaLayer
        assert mtp_stack.layers[0]._get_residual_connection() is None
        assert mtp_stack.layers[1].is_mtp_layer
        assert isinstance(mtp_stack.layers[1], TransformerLayer)
        assert not isinstance(mtp_stack.layers[1], WideResidualTransformerLayer)
        assert mtp_stack.layers[1]._get_mlp_residual_connection() is None
        assert model.mtp.layers[0].eh_proj.weight.grad is not None
        assert model.embedding.word_embeddings.weight.grad is not None
        assert loss.shape == input_ids.shape

    @pytest.mark.parametrize(
        "fp32_residual_connection", [False, True], ids=["native-residual", "fp32-residual"]
    )
    def test_mimo_composes_at_backbone_width_before_decoder_expansion(
        self, fp32_residual_connection
    ):
        config = _wide_config(
            num_layers=1, hidden_size=64, fp32_residual_connection=fp32_residual_connection
        )
        layer_spec = get_gpt_wide_residual_layer_local_spec()
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
        model = _move_model_to_configured_dtype(
            MimoModel(
                MimoModelConfig(
                    language_model_spec=language_spec,
                    modality_submodules_spec={},
                    special_token_ids={},
                )
            ),
            config,
        )
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
        expected_decoder_dtype = (
            torch.float32 if config.fp32_residual_connection else config.params_dtype
        )
        assert decoder_input.dtype == expected_decoder_dtype
        assert model.language_model.decoder.residual_stream_readout is not None
        assert model.language_model.embedding.word_embeddings.weight.grad is not None
        assert model.language_model.decoder.residual_stream_readout.exit_map.logit.grad is not None
        assert returned_loss_mask is loss_mask
        assert loss.shape == input_ids.shape

    def test_mimo_hybrid_shortcut_mtp_wide_fp32_forward_backward(self):
        config = _wide_config(
            num_layers=4,
            hidden_size=256,
            mtp_num_layers=2,
            with_moe=True,
            with_shortcut=True,
            fp32_residual_connection=True,
        )
        language_spec = ModuleSpec(
            module=HybridModel,
            params={
                "config": config,
                "hybrid_stack_spec": wide_residual_hybrid_stack_spec,
                "vocab_size": 128,
                "max_sequence_length": 4,
                "hybrid_layer_pattern": "MEME/*E/*E",
                "position_embedding_type": "none",
            },
        )
        model = _move_model_to_configured_dtype(
            MimoModel(
                MimoModelConfig(
                    language_model_spec=language_spec,
                    modality_submodules_spec={},
                    special_token_ids={},
                )
            ),
            config,
        )

        shortcuts = model.language_model.decoder.layers
        assert len(shortcuts) == 2
        assert all(isinstance(shortcut, ShortcutMoEBlock) for shortcut in shortcuts)
        second_shortcut_input_dtypes = []

        def record_second_shortcut_input_dtype(_module, inputs):
            second_shortcut_input_dtypes.append(inputs[0].dtype)

        shortcuts[1].shortcut_residual_read.register_forward_pre_hook(
            record_second_shortcut_input_dtype
        )
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
        mtp_stack = model.language_model.mtp.layers[0].mtp_model_layer
        assert decoder_input.dtype == torch.float32
        assert second_shortcut_input_dtypes
        assert all(dtype == torch.float32 for dtype in second_shortcut_input_dtypes)
        for shortcut in shortcuts:
            assert shortcut.shortcut_residual_read is not None
            assert shortcut.shortcut_residual_read.read_map.logit.grad is not None
            assert shortcut.attn_layer.residual_connection.read_map.logit.grad is not None
            assert shortcut.moe_layer.residual_connection_mlp.read_map.logit.grad is not None
            assert shortcut.shortcut_post_norm.weight.grad is not None
        assert not mtp_stack.uses_wide_residual_stream
        assert len(mtp_stack.layers) == 1
        assert isinstance(mtp_stack.layers[0], ShortcutMoEBlock)
        assert mtp_stack.layers[0].is_mtp_layer
        assert mtp_stack.layers[0].shortcut_residual_read is None
        assert mtp_stack.layers[0].shortcut_post_norm.weight.grad is not None
        assert model.language_model.mtp.layers[0].eh_proj.weight.grad is not None
        assert model.language_model.embedding.word_embeddings.weight.grad is not None
        assert returned_loss_mask is loss_mask
        assert loss.shape == input_ids.shape
