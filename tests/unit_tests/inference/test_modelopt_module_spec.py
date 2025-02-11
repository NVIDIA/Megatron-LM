# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import inspect
import tempfile

import pytest
import torch

from megatron.core import InferenceParams, dist_checkpointing
from megatron.core.inference.modelopt_support.gpt.model_specs import get_gpt_layer_modelopt_spec
from megatron.core.inference.modelopt_support.gpt.state_dict_hooks import (
    mcore_gpt_load_te_state_dict_pre_hook,
)
from megatron.core.inference.modelopt_support.mamba.model_specs import get_mamba_stack_modelopt_spec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def model_forward(model: torch.nn.Module, config: TransformerConfig, micro_batch_size: int = 2):
    inference_params: InferenceParams = InferenceParams(
        max_batch_size=micro_batch_size, max_sequence_length=model.max_sequence_length
    )
    prompt_length = model.max_sequence_length - 1

    # load-context/first-output-token, step/generate
    for offset in (0, prompt_length):
        if offset == 0:
            sequence_length = prompt_length
        else:
            sequence_length = 1
        inference_params.sequence_len_offset = offset

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
            inference_params=inference_params,
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == model.vocab_size


class TestModelOptGPTModel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )
        # Ensure that a GPTModel can be built with the modelopt spec.
        self.modelopt_gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_modelopt_spec(remap_te_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
        )

    def test_load_te_state_dict_pre_hook(self):
        handle = self.modelopt_gpt_model._register_load_state_dict_pre_hook(
            mcore_gpt_load_te_state_dict_pre_hook
        )
        self.modelopt_gpt_model.load_state_dict(self.gpt_model.state_dict())
        handle.remove()

    def test_sharded_state_dict_restore(self, tmp_path_dist_ckpt):
        te_fused_sharded_state_dict = self.gpt_model.sharded_state_dict()
        modelopt_sharded_state_dict = self.modelopt_gpt_model.sharded_state_dict()

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_sharded_state_dict_restore', sync=True
        ) as tmpdirname:
            dist_checkpointing.save(te_fused_sharded_state_dict, tmpdirname)
            state_dict = dist_checkpointing.load(modelopt_sharded_state_dict, tmpdirname)
            self.modelopt_gpt_model.load_state_dict(state_dict)

    def test_inference(self):
        config: TransformerConfig = self.modelopt_gpt_model.config
        model = self.modelopt_gpt_model.cuda()
        model_forward(model, config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestModelOptMambaModel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )

        # A Hybrid MambaModel using fused-TE spec (default)
        self.mamba_model = MambaModel(
            config=transformer_config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_override_pattern="M*-",
        )

        # A Hybrid MambaModel using ModelOpt spec (local + TENorm).
        self.modelopt_mamba_model = MambaModel(
            config=transformer_config,
            mamba_stack_spec=get_mamba_stack_modelopt_spec(remap_te_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            hybrid_override_pattern="M*-",
        )

    def test_sharded_state_dict_restore(self, tmp_path_dist_ckpt):
        te_fused_sharded_state_dict = self.mamba_model.sharded_state_dict()
        modelopt_sharded_state_dict = self.modelopt_mamba_model.sharded_state_dict()

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_sharded_state_dict_restore', sync=True
        ) as tmpdirname:
            dist_checkpointing.save(te_fused_sharded_state_dict, tmpdirname)
            state_dict = dist_checkpointing.load(modelopt_sharded_state_dict, tmpdirname)
            self.modelopt_mamba_model.load_state_dict(state_dict)

    def test_inference(self):
        config: TransformerConfig = self.modelopt_mamba_model.config
        model = self.modelopt_mamba_model.cuda()
        model_forward(model, config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


def test_get_gpt_layer_modelopt_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_gpt_layer_modelopt_spec)

    # Define the expected signature
    expected_params = {
        "num_experts": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "local_core_attention": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "moe_grouped_gemm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "remap_te_layernorm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "qk_layernorm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {
        "num_experts": None,
        "local_core_attention": False,
        "moe_grouped_gemm": False,
        "remap_te_layernorm": False,
        "qk_layernorm": False,
    }

    # Check parameter kinds
    for param_name, param in sig.parameters.items():
        assert param_name in expected_params.keys(), f"Unexpected parameter: {param_name}"
        assert param.kind is expected_params[param_name], f"Wrong kind for parameter: {param_name}"

    # Check default values
    defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }
    assert defaults == expected_defaults, "Default values do not match the expected ones."


def test_get_mamba_stack_modelopt_spec_interface():
    # Get the function signature
    sig = inspect.signature(get_mamba_stack_modelopt_spec)

    # Define the expected signature
    expected_params = {
        "local_core_attention": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "remap_te_layernorm": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }

    expected_defaults = {"local_core_attention": False, "remap_te_layernorm": False}

    # Check parameter kinds
    for param_name, param in sig.parameters.items():
        assert param_name in expected_params.keys(), f"Unexpected parameter: {param_name}"
        assert param.kind is expected_params[param_name], f"Wrong kind for parameter: {param_name}"

    # Check default values
    defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }
    assert defaults == expected_defaults, "Default values do not match the expected ones."
