# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy
import inspect
import os
from datetime import timedelta

import pytest
import torch
from packaging import version
from pytest import approx
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_mlp_module_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.test_utilities import Utils


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
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.gpt_model, GPTModel)

        assert self.gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.gpt_model.parameters()])
        assert num_weights == 6240

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


def test_get_mlp_module_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_mlp_module_spec)

    # Define the expected signature
    expected_params = {
        "use_te": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "num_experts": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "moe_grouped_gemm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "fp8": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "moe_use_legacy_grouped_gemm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "use_te_op_fuser": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {
        "use_te": True,
        "num_experts": None,
        "moe_grouped_gemm": False,
        "fp8": None,
        "moe_use_legacy_grouped_gemm": False,
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


@pytest.mark.skipif(
    not is_te_min_version("1.2.0"), reason="TE fused attention test requires TE >= 1.2.0"
)
@pytest.mark.parametrize("attn_mask_type", [AttnMaskType.no_mask, AttnMaskType.causal])
def test_gptmodel_te_fused_attention_mask_matches_local_attention_mask(
    attn_mask_type: AttnMaskType,
):
    """Verify TE fused attention with mask->bias conversion matches native attention."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    try:
        seed = 1234
        model_parallel_cuda_manual_seed(seed)

        seq_len = 4096
        batch_size = 1
        hidden_size = 2048
        num_attention_heads = 16
        vocab_size = 128

        local_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_attention_heads,
            kv_channels=hidden_size // num_attention_heads,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            sequence_parallel=False,
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            transformer_impl="native",
            use_cpu_initialization=False,
            mtp_num_layers=1,
        )
        te_config = copy.deepcopy(local_config)
        te_config.transformer_impl = "transformer_engine"

        # Build local model with the same attn_mask_type
        local_spec = get_gpt_layer_local_spec()
        local_spec.submodules.self_attention.params["attn_mask_type"] = attn_mask_type
        local_model = (
            GPTModel(
                config=local_config,
                transformer_layer_spec=local_spec,
                vocab_size=vocab_size,
                max_sequence_length=seq_len,
                pre_process=True,
                post_process=True,
            )
            .cuda()
            .bfloat16()
        )
        # Initialize embedding weights following normal distribution
        # to improve the accuracy of the test.
        local_model.embedding.word_embeddings.weight.data = torch.randn(
            vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
        )

        # Build TE model with the same attn_mask_type
        te_spec = get_gpt_layer_with_transformer_engine_spec()
        te_spec.submodules.self_attention.params["attn_mask_type"] = attn_mask_type
        te_model = (
            GPTModel(
                config=te_config,
                transformer_layer_spec=te_spec,
                vocab_size=vocab_size,
                max_sequence_length=seq_len,
                pre_process=True,
                post_process=True,
            )
            .cuda()
            .bfloat16()
        )
        te_model.load_state_dict(local_model.state_dict(), strict=False)

        local_model.eval()
        te_model.eval()

        # Prepare input_ids
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long
        )
        position_ids = (
            torch.arange(seq_len, device="cuda", dtype=torch.long)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # Prepare attention mask
        attention_mask = torch.randn(1, 1, seq_len, seq_len, device="cuda", dtype=torch.float32) > 0
        if "causal" in attn_mask_type.name:
            # Legal causal mask: upper-triangular (mask out future tokens), broadcasted on batch.
            triangular_mask = torch.triu(
                torch.ones((1, 1, seq_len, seq_len), device="cuda", dtype=torch.bool), diagonal=1
            )
            attention_mask = attention_mask | triangular_mask
        else:
            # Avoid fully-masked rows for the non-causal random-mask case.
            attention_mask[..., 0] = False
        attention_mask = attention_mask.expand(batch_size, -1, -1, -1)

        local_out = local_model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        te_out = te_model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        local_out.mean().backward()
        te_out.mean().backward()

        def assert_similar(actual, expected, name, threshold=0.9999):
            def _calculate_tensor_similarity(x, y):
                denominator = (x * x + y * y).sum()
                if denominator == 0:
                    return 1
                sim = 2 * (x * y).sum() / denominator
                return sim

            actual = actual.data.double()
            expected = expected.data.double()
            cosine_sim = torch.nn.functional.cosine_similarity(
                actual.flatten().unsqueeze(0), expected.flatten().unsqueeze(0)
            ).item()
            tensor_sim = _calculate_tensor_similarity(actual, expected)
            assert cosine_sim > threshold, f"{name} cosine similarity = {cosine_sim} < {threshold}"
            assert tensor_sim > threshold, f"{name} tensor similarity = {tensor_sim} < {threshold}"

        # Compare outputs between local and TE models.
        assert_similar(te_out, local_out, "GPT output with attention_mask", threshold=0.9999)
        # Compare gradients for overlapping parameters between local and TE models.
        local_named_params = dict(local_model.named_parameters())
        te_named_params = dict(te_model.named_parameters())
        for param_name, local_param in local_named_params.items():
            te_param = te_named_params.get(param_name)
            if te_param is None:
                continue
            if local_param.grad is None or te_param.grad is None:
                continue
            if local_param.grad.shape != te_param.grad.shape:
                continue
            assert_similar(te_param.grad, local_param.grad, f"parameter grad for {param_name}")
    finally:
        Utils.destroy_model_parallel()
