# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import logging
import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from packaging import version
from pytest import approx
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
    get_mlp_module_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("layout, pp_size", [("Et|mL", 2), ("Et|m|L", 3)])
def test_gpt_mhc_rejects_standalone_mtp_pipeline_chunks(layout, pp_size):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=pp_size,
        pipeline_model_parallel_layout=layout,
        enable_hyper_connections=True,
        mtp_num_layers=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="requires MTP and the final decoder layer"):
        GPTModel(
            config,
            get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=8,
        )


def test_mhc_mtp_preserves_multistream_on_pipeline_stages():
    """Dev's PP path must pass pre-contraction streams to the real MTP block."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            enable_hyper_connections=True,
            num_residual_streams=4,
            mtp_num_layers=1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        first_stage = pg_collection.pp.rank() == 0
        layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
        mtp_spec = get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True)
        model = GPTModel(
            config,
            layer_spec,
            vocab_size=128,
            max_sequence_length=16,
            pre_process=first_stage,
            post_process=not first_stage,
            share_embeddings_and_output_weights=False,
            mtp_block_spec=mtp_spec,
            pg_collection=pg_collection,
        ).cuda()
        tokens = torch.arange(32, device="cuda").reshape(2, 16)
        positions = torch.arange(16, device="cuda").expand_as(tokens)
        multistream_shape = (16, 2, config.num_residual_streams * config.hidden_size)
        captured = {}
        pipeline_input = None
        if not first_stage:
            pipeline_input = torch.randn(
                multistream_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            model.set_input_tensor(pipeline_input)

            def capture_mtp_input(module, args, kwargs):
                captured["contracted"] = kwargs["hidden_states"].shape
                captured["multistream"] = kwargs["mhc_multistream"].shape

            model.mtp.register_forward_pre_hook(capture_mtp_input, with_kwargs=True)

        output = model(tokens, positions, attention_mask=None, labels=tokens)
        assert torch.isfinite(output).all()
        if first_stage:
            assert output.shape == multistream_shape
        else:
            assert output.shape == tokens.shape
            assert captured["contracted"] == (16, 2, config.hidden_size)
            assert captured["multistream"] == multistream_shape
        output.float().square().mean().backward()
        if pipeline_input is not None:
            assert pipeline_input.grad is not None
            assert torch.isfinite(pipeline_input.grad).all()
            assert torch.count_nonzero(pipeline_input.grad) > 0
            mtp_grads = [p.grad for p in model.mtp.parameters() if p.grad is not None]
            assert mtp_grads
            assert all(torch.isfinite(grad).all() for grad in mtp_grads)
            assert any(torch.count_nonzero(grad) > 0 for grad in mtp_grads)
    finally:
        Utils.destroy_model_parallel()


class TestGPTModel:

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            embedding_init_method_std=1.0,  # Test that we can initialize the embedding weights to something else.
        )
        with patch('megatron.core.models.gpt.gpt_model.log_single_rank') as mock_log_single_rank:
            self.gpt_model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=100,
                max_sequence_length=4,
            )
        self.mock_log_single_rank = mock_log_single_rank

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        self.mock_log_single_rank.assert_called_once()
        _, level, message = self.mock_log_single_rank.call_args.args
        assert level == logging.WARNING
        assert message == (
            "GPTModel IS DEPRECATED. GPTModel is only accepting critical bug fixes, no new "
            "features. Please reference the migration guide "
            "`docs/user-guide/hybrid-model-migration.md` for details on how to use `HybridModel`"
        )

        assert isinstance(self.gpt_model, GPTModel)

        assert self.gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.gpt_model.parameters()])
        assert num_weights == 6240

    @pytest.mark.parametrize("add_bias_linear", [False, True])
    def test_hash_routed_moe_receives_token_ids_and_backpropagates(self, add_bias_linear):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            num_moe_experts=4,
            moe_num_hash_layers=1,
            hash_moe_vocab_size=100,
            use_cpu_initialization=True,
            add_bias_linear=add_bias_linear,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=True),
            vocab_size=128,
            max_sequence_length=8,
        ).cuda()
        router = model.decoder.layers[0].mlp.router
        assert router.is_hash_layer
        assert not model.decoder.layers[1].mlp.router.is_hash_layer
        received = []
        router.register_forward_pre_hook(
            lambda module, args, kwargs: received.append(kwargs['input_ids'].clone()),
            with_kwargs=True,
        )
        tokens = torch.arange(16, device='cuda').reshape(2, 8)
        positions = torch.arange(8, device='cuda').expand_as(tokens)
        loss = model(tokens, positions, None, labels=tokens)
        torch.testing.assert_close(received[0], tokens)
        assert torch.isfinite(loss).all()
        loss.mean().backward()
        grad = model.embedding.word_embeddings.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0

    @pytest.mark.parametrize("mtp_num_layers", [None, 0, 1])
    def test_hsm_requires_two_mtp_layers(self, mtp_num_layers):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            mtp_num_layers=mtp_num_layers,
            mtp_hsm=True,
        )
        with pytest.raises(ValueError, match="mtp_hsm=True requires mtp_num_layers >= 2"):
            GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=100,
                max_sequence_length=4,
            )

        assert config.mtp_hsm is True

    def test_rejects_hybrid_mtp_override(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            mtp_hybrid_override_pattern="*",
        )
        with pytest.raises(
            ValueError, match="mtp_hybrid_override_pattern is not supported by GPTModel"
        ):
            GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=100,
                max_sequence_length=4,
            )

    @pytest.mark.internal
    def test_set_input_tensor(self):
        config: TransformerConfig = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.gpt_model.set_input_tensor(input_tensor)

        assert self.gpt_model.decoder.input_tensor.shape[0] == sequence_length
        assert self.gpt_model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.gpt_model.decoder.input_tensor.shape[2] == config.hidden_size

    def test_embedding_init(self):
        """Test that we can initialize the embedding weights to something else. This test could be added to any model."""
        config: TransformerConfig = self.gpt_model.config
        assert self.gpt_model.embedding.word_embeddings.weight.std().cpu().item() == approx(
            config.embedding_init_method_std, abs=1e-1
        )
        assert self.gpt_model.embedding.word_embeddings.weight.mean().cpu().item() == approx(
            0.0, abs=1e-1
        )

    @pytest.mark.internal
    def test_post_process_forward(self):
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size

    @pytest.mark.parametrize("sequence_parallel", [False, True])
    @pytest.mark.parametrize("has_padding_mask", [False, True])
    def test_mtp_keeps_raw_padding_mask_after_decoder_preprocess(
        self, sequence_parallel, has_padding_mask
    ):
        """Decoder SP transforms must not change the MTP sequence-roll source layout."""
        input_ids = torch.arange(8, dtype=torch.long).view(2, 4)
        position_ids = input_ids.clone()
        raw_padding_mask = (
            torch.tensor([[False, False, True, True], [False, True, False, True]])
            if has_padding_mask
            else None
        )
        transformed_padding_mask = raw_padding_mask
        if sequence_parallel and raw_padding_mask is not None:
            transformed_padding_mask = raw_padding_mask[:, :2].contiguous()
        decoder_input = torch.zeros(4, 2, self.gpt_model.config.hidden_size)
        decoder_output = torch.ones_like(decoder_input)
        self.gpt_model.config.sequence_parallel = sequence_parallel

        with (
            patch.object(
                self.gpt_model,
                "_preprocess",
                return_value=(decoder_input, None, None, None, None, transformed_padding_mask),
            ),
            patch.object(self.gpt_model.decoder, "forward", return_value=decoder_output) as decoder,
            patch.object(
                self.gpt_model, "_postprocess", return_value=decoder_output
            ) as postprocess,
        ):
            self.gpt_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                padding_mask=raw_padding_mask,
            )

        assert decoder.call_args.kwargs["padding_mask"] is transformed_padding_mask
        # Since dev #8105 the post-process gets the caller's unsharded mask, not the decoder's.
        assert postprocess.call_args.kwargs["padding_mask"] is raw_padding_mask
        assert postprocess.call_args.kwargs["mtp_padding_mask"] is raw_padding_mask

    def test_postprocess_reuses_supplied_mtp_sequence_roll_context(self):
        """Fine-grained callers can reuse one prepared context without rebuilding it."""

        class RecordingMTP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.kwargs = None

            def forward(self, **kwargs):
                self.kwargs = kwargs
                return kwargs["hidden_states"]

        self.gpt_model.config.mtp_num_layers = 1
        self.gpt_model.post_process = False
        self.gpt_model.mtp = RecordingMTP()
        supplied_context = object()
        hidden_states = torch.zeros(4, 2, self.gpt_model.config.hidden_size)
        input_ids = torch.zeros(2, 4, dtype=torch.long)
        position_ids = torch.zeros_like(input_ids)
        local_padding_mask = torch.zeros(2, 2, dtype=torch.bool)
        raw_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        with patch(
            "megatron.core.models.gpt.gpt_model.prepare_mtp_sequence_roll_context",
            side_effect=AssertionError("A supplied MTP context must not be rebuilt."),
        ):
            output = self.gpt_model._postprocess(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=None,
                rotary_pos_emb=None,
                rotary_pos_cos=None,
                rotary_pos_sin=None,
                mtp_in_postprocess=True,
                attention_mask=None,
                padding_mask=local_padding_mask,
                mtp_padding_mask=raw_padding_mask,
                sequence_roll_context=supplied_context,
            )

        assert output is hidden_states
        assert self.gpt_model.mtp.kwargs["sequence_roll_context"] is supplied_context
        assert self.gpt_model.mtp.kwargs["padding_mask"] is local_padding_mask
        assert self.gpt_model.mtp.kwargs["sequence_roll_padding_mask"] is raw_padding_mask

    @pytest.mark.internal
    def test_output_processor_forward(self):
        config: TransformerConfig = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()
        self.gpt_model.eval()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = (input_ids + 1) % self.gpt_model.vocab_size
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        context = {"selected_token_positions": torch.tensor([0, 2], device="cuda")}
        created = {}
        seen = {}

        def output_processor(**kwargs):
            seen.update(kwargs)
            logits, _ = kwargs["output_layer"](
                kwargs["hidden_states"],
                weight=kwargs["output_weight"],
                runtime_gather_output=kwargs["runtime_gather_output"],
            )
            logits = kwargs["scale_logits"](logits).transpose(0, 1).contiguous().float()
            token_logprobs = F.log_softmax(logits, dim=-1).gather(
                dim=-1, index=kwargs["labels"].unsqueeze(-1)
            )
            token_logprobs = token_logprobs.squeeze(-1)
            result = {
                "payload": token_logprobs.index_select(
                    1, kwargs["context"]["selected_token_positions"]
                ),
                "tag": "structured",
            }
            created["result"] = result
            return result

        with torch.no_grad():
            logits = self.gpt_model.forward(
                input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
            ).float()
            expected = F.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1))
            expected = expected.squeeze(-1).index_select(1, context["selected_token_positions"])

            output = self.gpt_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_processor=output_processor,
                output_processor_context=context,
            )

        assert output is created["result"]
        assert isinstance(output, dict)
        assert torch.allclose(output["payload"], expected)
        assert output["tag"] == "structured"
        assert seen["context"] is context
        assert seen["output_layer"] is self.gpt_model.output_layer
        assert seen["output_weight"] is None
        assert seen["output_layer"].weight is not None
        assert seen["labels"] is labels
        assert seen["runtime_gather_output"] is None
        assert seen["config"] is config
        assert seen["compute_language_model_loss"].__self__ is self.gpt_model
        assert seen["scale_logits"].__self__ is self.gpt_model

    @pytest.mark.internal
    def test_build_schedule_plan_threads_output_processor(self):
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        context = {"loss_mask": torch.ones_like(labels)}

        seen = {}

        def output_processor(**kwargs):
            seen.update(kwargs)
            return kwargs["hidden_states"].sum()

        schedule_plan = self.gpt_model.build_schedule_plan(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_processor=output_processor,
            output_processor_context=context,
        )

        assert schedule_plan.state.output_processor is output_processor
        assert schedule_plan.state.output_processor_context is context

        schedule_plan.state.decoder_input = None
        schedule_plan.state.rotary_pos_emb = None
        schedule_plan.state.rotary_pos_cos = None
        schedule_plan.state.rotary_pos_sin = None
        schedule_plan.state.sequence_len_offset = None

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            self.gpt_model.config.hidden_size,
            device="cuda",
            requires_grad=True,
        )
        output = schedule_plan.post_process.forward(hidden_states)

        assert output.item() == pytest.approx(hidden_states.sum().item())
        assert seen["context"] is context
        assert seen["labels"] is labels
        assert seen["output_layer"] is self.gpt_model.output_layer


def test_get_mlp_module_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_mlp_module_spec)

    # Define the expected signature
    expected_params = {
        "use_te": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "num_experts": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "moe_grouped_gemm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "fp8": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "use_te_op_fuser": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {
        "use_te": True,
        "num_experts": None,
        "moe_grouped_gemm": False,
        "fp8": None,
        "use_te_op_fuser": False,
    }

    # Check expected parameters are in function signature
    for param_name, param_kind in expected_params.items():
        assert param_name in sig.parameters, f"Unexpected parameter: {param_name}"
        assert (
            param_kind is sig.parameters[param_name].kind
        ), f"Wrong kind for parameter: {param_name}"

    # Check default values
    sig_defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }
    for k, v in expected_defaults.items():
        assert (
            k in sig_defaults and v == sig_defaults[k]
        ), f"Default value of {sig_defaults[k]} does not match the expected value of {v} for parameter {k}."


@pytest.mark.skipif(
    not is_te_min_version("1.13.0"), reason="TEFusedMLP is only supported with TE 1.13+."
)
class TestGPTWithFusedOps:
    """GPT model with Transformer Engine operation-based API"""

    def setup_method(self, method) -> None:
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(use_te_op_fuser=True),
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method) -> None:
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_forward(self) -> None:
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.gpt_model.vocab_size


@pytest.mark.skipif(
    not is_te_min_version("1.13.0"), reason="TEFusedMLP is only supported with TE 1.13+."
)
@pytest.mark.parametrize("num_experts", [None, 4])
@pytest.mark.parametrize("gated_linear_unit", [True, False])
def test_gpt_with_te_activation_func(num_experts, gated_linear_unit):
    """Test GPT model with Transformer Engine activation function"""

    # setup
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=4,
        use_cpu_initialization=True,
        add_bias_linear=False,
        use_te_activation_func=True,
        bias_activation_fusion=False,
        gated_linear_unit=gated_linear_unit,
        num_moe_experts=num_experts,
        moe_grouped_gemm=(num_experts is not None),
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts, use_te_activation_func=True
        ),
        vocab_size=128,
        max_sequence_length=128,
    )

    # test
    sequence_length = gpt_model.max_sequence_length
    micro_batch_size = 2

    gpt_model.cuda()

    data = list(range(sequence_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
    attention_mask = torch.ones(
        (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
    ).cuda()

    logits = gpt_model.forward(
        input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
    )

    assert logits.shape[0] == micro_batch_size
    assert logits.shape[1] == sequence_length
    assert logits.shape[2] == gpt_model.vocab_size

    # teardown
    Utils.destroy_model_parallel()


class TestGPTModelWithCustomPG:
    def setup_method(self, method):
        Utils.destroy_model_parallel()
        Utils.initialize_distributed()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "tp_size, dp_size, cp_size", [(1, 8, 1), (2, 4, 1)]  # TP 1, DP 8, CP 1  # TP 2, DP 4, CP 1
    )
    def test_gpt_model_with_custom_pg(self, tp_size, dp_size, cp_size):

        # Create HyperCommGrid with dimensions tp, cp, ep, pp, dp (reversed from device mesh order)
        grid = HyperCommGrid([tp_size, cp_size, 1, 1, dp_size], ["tp", "cp", "ep", "pp", "dp"])

        tp_group = grid.create_pg("tp")
        cp_group = grid.create_pg("cp")
        pp_group = grid.create_pg("pp")
        ep_group = grid.create_pg("ep")
        embd_group_ranks = parallel_state.default_embedding_ranks(
            torch.distributed.get_process_group_ranks(pp_group)
        )
        embd_group = torch.distributed.new_group(
            ranks=embd_group_ranks, timeout=timedelta(minutes=30)
        )
        pg_collection = ProcessGroupCollection(
            tp=tp_group, cp=cp_group, pp=pp_group, ep=ep_group, embd=embd_group
        )

        model_parallel_cuda_manual_seed(
            1234, tp_rank=tp_group.rank(), ep_rank=ep_group.rank(), etp_rank=tp_group.rank()
        )
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=1024, num_attention_heads=16, use_cpu_initialization=False
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=512,
            pg_collection=pg_collection,
            post_process=False,
        )

        # Check that model weights are distributed as expected when using TP
        assert (
            self.gpt_model.decoder.layers[0].self_attention.linear_qkv.weight.shape[0]
            == (1024 * 3) / tp_size
        )
        assert self.gpt_model.decoder.layers[0].self_attention.linear_qkv.weight.shape[1] == 1024
        assert self.gpt_model.decoder.layers[0].self_attention.linear_proj.weight.shape[0] == 1024
        assert (
            self.gpt_model.decoder.layers[0].self_attention.linear_proj.weight.shape[1]
            == 1024 / tp_size
        )

        # Check that the logits output shape is correct
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        self.gpt_model.cuda()

        input_ids = torch.ones(micro_batch_size, sequence_length, dtype=torch.int64, device="cuda")
        position_ids = torch.ones(
            micro_batch_size, sequence_length, dtype=torch.int64, device="cuda"
        )

        logits = self.gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )

        assert logits.shape[0] == sequence_length
        assert logits.shape[1] == micro_batch_size
        assert logits.shape[2] == self.gpt_model.config.hidden_size


class TestGPTWithDynamicInference:
    """Tests GPTModel with dynamic inference."""

    @torch.inference_mode()
    def setup_method(self, method):
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        transformer_config = TransformerConfig(
            num_layers=8,
            hidden_size=256,
            num_attention_heads=8,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid",
            fp8_recipe="tensorwise",
        )

        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=DynamicInferenceContext.TOKEN_ROUNDER,
            parallel_output=True,
        )
        self.gpt_model = Float16Module(self.gpt_model.config, self.gpt_model)

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
        self.gpt_model.cuda()
        self.gpt_model.eval()
        config = self.gpt_model.config

        inference_context = DynamicInferenceContext(
            model_config=TransformerConfig(
                params_dtype=config.params_dtype,
                num_layers=config.num_layers,
                kv_channels=config.hidden_size // config.num_attention_heads,
                num_attention_heads=config.num_attention_heads,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=self.gpt_model.module.max_sequence_length,
                buffer_size_gb=1.0,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
            ),
        )

        # Add a request with 10 tokens. Since 10 is not a multiple of 64,
        # this will create padding up to the padded length of 64.
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
        with InferenceMode.active():
            logits = self.gpt_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                inference_context=inference_context,
                runtime_gather_output=True,
            )

        # Verify the output shape.
        assert logits.shape[0] == 1
        assert logits.shape[1] == padded_token_count
        assert logits.shape[2] == self.gpt_model.module.vocab_size

        # Extract the logits corresponding to the padding tokens (from index 10 to 63).
        padding_start_idx = inference_context.active_token_count
        padding_end_idx = inference_context.padded_active_token_count
        padding_logits = logits[0, padding_start_idx:padding_end_idx, :]

        # Assert that all padding logits are zero.
        assert torch.all(padding_logits == 0.0), "Logits for padding tokens are not all zero."


def test_get_transformer_layer_spec_forwards_use_te_activation_func():
    """Test that _get_transformer_layer_spec forwards use_te_activation_func.

    Regression test for https://github.com/NVIDIA/Megatron-LM/issues/2770
    The --use-te-activation-func flag was silently ignored for non-MoE GPT
    models because _get_transformer_layer_spec did not forward the parameter
    to get_gpt_layer_with_transformer_engine_spec.
    """
    mock_config = MagicMock()
    mock_config.use_te_activation_func = True
    mock_config.use_kitchen = False
    mock_config.use_kitchen_attention = False
    mock_config.kitchen_attention_backend = "sdpa"

    mock_args = MagicMock()
    mock_args.num_experts = None
    mock_args.moe_grouped_gemm = False
    mock_args.qk_layernorm = False
    mock_args.multi_latent_attention = False
    mock_args.experimental_attention_variant = None
    mock_args.qk_l2_norm = False

    with (
        patch('gpt_builders.get_args', return_value=mock_args),
        patch('gpt_builders.get_gpt_layer_with_transformer_engine_spec') as mock_spec_fn,
    ):
        from gpt_builders import _get_transformer_layer_spec

        _get_transformer_layer_spec(use_te=True, config=mock_config)

        mock_spec_fn.assert_called_once()
        _, call_kwargs = mock_spec_fn.call_args
        assert (
            call_kwargs.get('use_te_activation_func') is True
        ), "use_te_activation_func must be forwarded from config"


def test_gpt_builder_forwards_rope_scaling_factor():
    """Test that gpt_builder forwards rope_scaling_factor to GPTModel.

    Regression test for https://github.com/NVIDIA/Megatron-LM/issues/6305
    The --rope-scaling-factor flag was silently ignored because gpt_builder
    passed rope_scaling but not rope_scaling_factor, so GPTModel always fell
    back to its default factor of 8.0.
    """
    mock_config = MagicMock()

    mock_args = MagicMock()
    mock_args.spec = None
    mock_args.transformer_impl = "transformer_engine"
    mock_args.experimental_attention_variant = None
    mock_args.num_experts = None
    mock_args.heterogeneous_layers_config_path = None
    mock_args.mtp_num_layers = None
    mock_args.use_rope_scaling = True
    mock_args.rope_scaling_factor = 32.0

    with (
        patch('gpt_builders.GPTModel') as mock_gpt_model,
        patch('gpt_builders._get_transformer_layer_spec'),
    ):
        from gpt_builders import gpt_builder

        gpt_builder(mock_args, pre_process=True, post_process=True, config=mock_config)

        mock_gpt_model.assert_called_once()
        _, call_kwargs = mock_gpt_model.call_args
        assert (
            call_kwargs.get('rope_scaling_factor') == 32.0
        ), "rope_scaling_factor must be forwarded from args"


def test_gpt_builder_uses_experimental_mtp_spec_for_empty_decoder_stage():
    """An MTP-only PP stage must not fall back to standard attention."""
    args = MagicMock()
    args.yaml_cfg = None
    args.spec = None
    args.transformer_impl = "transformer_engine"
    args.experimental_attention_variant = "dsa"
    args.mtp_num_layers = 7
    config = MagicMock()
    config.transformer_impl = "transformer_engine"

    empty_block = MagicMock()
    empty_block.layer_specs = []
    expected_spec = MagicMock()

    with (
        patch('gpt_builders.core_transformer_config_from_args', return_value=config),
        patch(
            'gpt_builders.get_transformer_block_with_experimental_attention_variant_spec',
            return_value=empty_block,
        ),
        patch(
            'gpt_builders.get_transformer_layer_with_experimental_attention_variant_spec',
            return_value=[MagicMock(), expected_spec],
        ) as mock_experimental_specs,
        patch('gpt_builders._get_transformer_layer_spec') as mock_generic_spec,
        patch('gpt_builders.get_gpt_mtp_block_spec', return_value=MagicMock()) as mock_get_mtp,
        patch('gpt_builders.GPTModel', return_value=MagicMock()),
    ):
        from gpt_builders import gpt_builder

        gpt_builder(args, pre_process=False, post_process=True, vp_stage=None)

    mock_experimental_specs.assert_called_once_with(config=config)
    mock_generic_spec.assert_not_called()
    assert mock_get_mtp.call_args.args[1] is expected_spec
