# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
import inspect
import tempfile

import pytest
import torch
from packaging.version import Version

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.post_training.modelopt.gpt.state_dict_hooks import (
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.post_training.modelopt.hybrid.model_specs import get_hybrid_stack_modelopt_spec
from megatron.core.post_training.modelopt.layers import Linear, Norm
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, DSAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import get_te_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def model_forward(model: torch.nn.Module, config: TransformerConfig, micro_batch_size: int = 2):
    inference_context: StaticInferenceContext = StaticInferenceContext(
        max_batch_size=micro_batch_size, max_sequence_length=model.max_sequence_length
    )
    prompt_length = model.max_sequence_length - 1

    with InferenceMode.active():
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

            logits = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inference_context=inference_context,
                runtime_gather_output=True,
            )

            assert logits.shape[0] == micro_batch_size
            # StaticInferenceContext always sets materialize_only_last_token_logits=True.
            assert logits.shape[1] == 1
            assert logits.shape[2] == model.vocab_size


class TestModelOptGPTModel:

    _test_inference = True

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self._dist_checkpoint_name = "standard_gpt_model"

        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.default_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )
        # Ensure that a GPTModel can be built with the modelopt spec.
        self.modelopt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_modelopt_spec(
                transformer_config, remap_te_layernorm=True
            ),
            vocab_size=100,
            max_sequence_length=4,
        )

    def test_sharded_state_dict_restore(self, tmp_path_dist_ckpt):
        """Save with the default TE spec and restore using the ModelOpt spec."""
        _dist_checkpoint_name = "default_model"
        metadata = {
            "dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)
        }
        te_fused_sharded_state_dict = self.default_model.sharded_state_dict(metadata=metadata)
        modelopt_sharded_state_dict = self.modelopt_model.sharded_state_dict(metadata=metadata)

        with TempNamedDir(tmp_path_dist_ckpt / _dist_checkpoint_name, sync=True) as tmpdirname:
            dist_checkpointing.save(te_fused_sharded_state_dict, tmpdirname)
            state_dict = dist_checkpointing.load(modelopt_sharded_state_dict, tmpdirname)
            self.modelopt_model.load_state_dict(state_dict)

    def test_inference(self):
        if not self._test_inference:
            return
        config: TransformerConfig = self.modelopt_model.config
        model = self.modelopt_model.cuda()
        model_forward(model, config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestModelOptMLAMoE(TestModelOptGPTModel):

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        # Early version of TE DotProductAttention does not support
        # q, k, v to have different shapes.
        self._test_inference = get_te_version() > Version("1.10")

        transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            add_bias_linear=False,
            num_moe_experts=2,
            moe_layer_freq=[0, 1],
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=128,
            qk_layernorm=True,
            use_cpu_initialization=True,
        )
        default_spec = get_gpt_decoder_block_spec(transformer_config, use_transformer_engine=True)
        self.default_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=default_spec,
            vocab_size=100,
            max_sequence_length=8,
        )
        modelopt_spec = get_gpt_modelopt_spec(transformer_config, remap_te_layernorm=True)
        # Ensure that a GPTModel can be built with the modelopt spec.
        self.modelopt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=modelopt_spec,
            vocab_size=100,
            max_sequence_length=8,
        )


class TestModelOptLlama4MoE(TestModelOptGPTModel):

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        # Early version of TE DotProductAttention does not support
        # q, k, v to have different shapes.
        self._test_inference = get_te_version() > Version("1.10")

        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            add_bias_linear=False,
            num_moe_experts=2,
            moe_layer_freq=[0, 1],
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=128,
            qk_layernorm=True,
            qk_l2_norm=True,
            use_cpu_initialization=True,
        )
        default_spec = get_gpt_decoder_block_spec(
            transformer_config, use_transformer_engine=True, qk_l2_norm=True
        )
        self.default_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=default_spec,
            vocab_size=100,
            max_sequence_length=8,
        )
        modelopt_spec = get_gpt_modelopt_spec(
            transformer_config, remap_te_layernorm=True, qk_l2_norm=True
        )
        # Ensure that a GPTModel can be built with the modelopt spec.
        self.modelopt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=modelopt_spec,
            vocab_size=100,
            max_sequence_length=8,
        )


class TestModelOptHybridModel(TestModelOptGPTModel):

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )

        # A Hybrid HybridModel using fused-TE spec (default)
        self.default_model = HybridModel(
            config=transformer_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

        # A Hybrid HybridModel using ModelOpt spec (local + TENorm).
        self.modelopt_model = HybridModel(
            config=transformer_config,
            hybrid_stack_spec=get_hybrid_stack_modelopt_spec(remap_te_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )


def test_get_gpt_modelopt_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_gpt_modelopt_spec)

    # Define the expected signature
    expected_params = {
        "config": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "local_core_attention": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "remap_te_layernorm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "real_quant_cfg": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "qk_l2_norm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "use_arbitrary_attention_mask": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {
        "local_core_attention": False,
        "remap_te_layernorm": False,
        "real_quant_cfg": "None",
        "qk_l2_norm": False,
        "use_arbitrary_attention_mask": False,
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


def test_get_hybrid_stack_modelopt_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_hybrid_stack_modelopt_spec)

    # Define the expected signature
    expected_params = {
        "local_core_attention": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "remap_te_layernorm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "use_default_te_spec": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {
        "local_core_attention": False,
        "remap_te_layernorm": False,
        "use_default_te_spec": False,
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


def test_get_hybrid_stack_modelopt_spec_use_default_te_spec():
    """Test that use_default_te_spec=True returns the standard hybrid_stack_spec."""
    spec = get_hybrid_stack_modelopt_spec(use_default_te_spec=True)
    assert spec is hybrid_stack_spec


def test_get_hybrid_stack_modelopt_spec_local_feature_specs():
    """The local ModelOpt HybridStack spec covers all HybridModel layer families."""
    spec = get_hybrid_stack_modelopt_spec()
    submodules = spec.submodules

    gdn_layer = submodules.gdn_layer
    assert gdn_layer.module is TransformerLayer
    assert gdn_layer.submodules.input_layernorm is Norm
    assert gdn_layer.submodules.self_attention.module is GatedDeltaNet
    assert gdn_layer.submodules.self_attention.submodules.in_proj is ColumnParallelLinear
    assert gdn_layer.submodules.self_attention.submodules.out_norm is Norm
    assert gdn_layer.submodules.self_attention.submodules.out_proj is RowParallelLinear

    dsa_layer = submodules.dsa_layer
    assert dsa_layer.module is TransformerLayer
    assert dsa_layer.submodules.input_layernorm is Norm
    assert dsa_layer.submodules.self_attention.module is MLASelfAttention
    assert dsa_layer.submodules.self_attention.submodules.q_layernorm is IdentityOp
    assert dsa_layer.submodules.self_attention.submodules.kv_layernorm is IdentityOp
    dsa_attention = dsa_layer.submodules.self_attention.submodules.core_attention
    assert dsa_attention.module is DSAttention
    indexer = dsa_attention.submodules.indexer
    assert indexer.module is DSAIndexer
    assert issubclass(indexer.submodules.linear_wq_b, Linear)
    assert "parallel_mode" in inspect.signature(indexer.submodules.linear_wq_b).parameters
    assert issubclass(indexer.submodules.linear_wk, Linear)
    assert indexer.submodules.k_norm is Norm
    assert issubclass(indexer.submodules.linear_weights_proj, Linear)

    mtp_block_spec = submodules.mtp_block_spec
    assert mtp_block_spec.module is MultiTokenPredictionBlock
    mtp_layer_spec = mtp_block_spec.submodules.layer_specs[0]
    assert mtp_layer_spec.module is MultiTokenPredictionLayer
    assert mtp_layer_spec.submodules.enorm is Norm
    assert mtp_layer_spec.submodules.hnorm is Norm
    assert mtp_layer_spec.submodules.eh_proj is ColumnParallelLinear
    assert mtp_layer_spec.submodules.layer_norm is Norm


def test_get_hybrid_stack_modelopt_spec_remaps_gdn_layernorm():
    """GDN local spec can load checkpoints saved from the fused TE GDN spec."""
    spec = get_hybrid_stack_modelopt_spec(remap_te_layernorm=True)
    assert spec.submodules.gdn_layer.submodules.sharded_state_dict_keys_map == {
        'input_layernorm.': 'self_attention.in_proj.layer_norm_'
    }
