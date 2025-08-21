# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import inspect
import tempfile

import pytest
import torch
from packaging.version import Version

from megatron.core import dist_checkpointing
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.post_training.modelopt.gpt.state_dict_hooks import (
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import get_te_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def model_forward(model: torch.nn.Module, config: TransformerConfig, micro_batch_size: int = 2):
    inference_context: StaticInferenceContext = StaticInferenceContext(
        max_batch_size=micro_batch_size, max_sequence_length=model.max_sequence_length
    )
    prompt_length = model.max_sequence_length - 1

    # load-context/first-output-token, step/generate
    for offset in (0, prompt_length):
        if offset == 0:
            sequence_length = prompt_length
        else:
            sequence_length = 1
        inference_context.sequence_len_offset = offset

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inference_context=inference_context,
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
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
        te_fused_sharded_state_dict = self.default_model.sharded_state_dict()
        modelopt_sharded_state_dict = self.modelopt_model.sharded_state_dict()

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


class TestModelOptMambaModel(TestModelOptGPTModel):

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )

        # A Hybrid MambaModel using fused-TE spec (default)
        self.default_model = MambaModel(
            config=transformer_config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_override_pattern="M*-",
        )

        # A Hybrid MambaModel using ModelOpt spec (local + TENorm).
        self.modelopt_model = MambaModel(
            config=transformer_config,
            mamba_stack_spec=get_mamba_stack_modelopt_spec(remap_te_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            hybrid_override_pattern="M*-",
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


def test_get_mamba_stack_modelopt_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_mamba_stack_modelopt_spec)

    # Define the expected signature
    expected_params = {
        "local_core_attention": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "remap_te_layernorm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {"local_core_attention": False, "remap_te_layernorm": False}

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
