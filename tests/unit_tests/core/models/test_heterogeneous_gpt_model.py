# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import json

import pytest
import torch

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.test_utilities import Utils

TORCH_VERSION_GE_2_4 = is_torch_min_version("2.4.0")

first_layer = {
    "attention": {"no_op": False, "replace_with_linear": False, "num_query_groups": 8},
    "mlp": {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 14336},
}


@pytest.fixture
def heterogeneous_gpt_model(request, tmp_path):
    (
        attention_no_op,
        attention_replace_with_linear,
        attention_num_query_groups,
        mlp_no_op,
        mlp_replace_with_linear,
        mlp_ffn_hidden_size,
        use_transformer_engine,
    ) = request.param

    second_layer_config = {
        "attention": {
            "no_op": attention_no_op,
            "replace_with_linear": attention_replace_with_linear,
            "num_query_groups": attention_num_query_groups,
        },
        "mlp": {
            "no_op": mlp_no_op,
            "replace_with_linear": mlp_replace_with_linear,
            "ffn_hidden_size": mlp_ffn_hidden_size,
        },
    }

    block_config_data = {"block_configs": [first_layer, second_layer_config]}
    block_config_file = tmp_path / "config.json"
    block_config_file.write_text(json.dumps(block_config_data))

    transformer_config = HeterogeneousTransformerConfig(
        num_layers=2,
        hidden_size=4096,
        add_bias_linear=False,
        normalization="RMSNorm",
        gated_linear_unit=True,
        num_attention_heads=32,
        use_cpu_initialization=True,
        perform_initialization=False,
        heterogeneous_layers_config_path=str(block_config_file),
    )

    return GPTModel(
        transformer_config,
        transformer_layer_spec=get_gpt_heterogeneous_layer_spec(
            transformer_config, use_te=use_transformer_engine
        ),
        vocab_size=128256,
        position_embedding_type="rope",
        max_sequence_length=4,
    )


@pytest.mark.parametrize(
    "heterogeneous_gpt_model, expected_num_parameters",
    [
        ((False, False, 8, False, False, 14336, True), 1486901248),  # regular TE
        pytest.param(
            (False, False, 8, False, False, 14336, False),
            1486901248,
            marks=pytest.mark.skipif(not TORCH_VERSION_GE_2_4, reason="Requires PyTorch >= 2.4.0"),
        ),  # regular local
        ((True, False, None, False, False, 14336, True), 1444954112),  # attn no-op TE
        pytest.param(
            (True, False, None, False, False, 14336, False),
            1444954112,
            marks=pytest.mark.skipif(not TORCH_VERSION_GE_2_4, reason="Requires PyTorch >= 2.4.0"),
        ),  # attn no-op local
        ((False, False, 8, True, False, None, True), 1310736384),  # mlp no-op TE
        pytest.param(
            (False, False, 8, True, False, None, False),
            1310736384,
            marks=pytest.mark.skipif(not TORCH_VERSION_GE_2_4, reason="Requires PyTorch >= 2.4.0"),
        ),  # mlp no-op local
        ((False, True, None, False, False, 14336, True), 1461735424),  # attn replace with linear TE
        pytest.param(
            (False, True, None, False, False, 14336, False),
            1461735424,
            marks=pytest.mark.skipif(not TORCH_VERSION_GE_2_4, reason="Requires PyTorch >= 2.4.0"),
        ),  # attn replace with linear local
        ((False, False, 8, False, True, None, True), 1327517696),  # mlp replace with linear TE
        pytest.param(
            (False, False, 8, False, True, None, False),
            1327517696,
            marks=pytest.mark.skipif(not TORCH_VERSION_GE_2_4, reason="Requires PyTorch >= 2.4.0"),
        ),  # mlp replace with linear local
    ],
    indirect=["heterogeneous_gpt_model"],
)
class TestHeterogeneousGPTModel:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self, heterogeneous_gpt_model, expected_num_parameters):
        assert isinstance(heterogeneous_gpt_model, GPTModel)

        assert heterogeneous_gpt_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in heterogeneous_gpt_model.parameters()])
        assert num_weights == expected_num_parameters

    def test_post_process_forward(self, heterogeneous_gpt_model, expected_num_parameters):
        sequence_length = heterogeneous_gpt_model.max_sequence_length
        micro_batch_size = 2

        heterogeneous_gpt_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = heterogeneous_gpt_model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == heterogeneous_gpt_model.vocab_size
