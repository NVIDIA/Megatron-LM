# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os
from datetime import timedelta

import pytest
import torch
from packaging import version
from pytest import approx

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_mlp_module_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
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
