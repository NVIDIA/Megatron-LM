# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.fine_grained_schedule import (
    TransformerLayerSchedulePlan,
    schedule_layer_1f1b,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


@dataclass
class DummyState:
    """
    A dummy state class that holds various attention-related parameters.

    This class is used to simulate the state object that would normally be passed
    to transformer layers in a real model.
    """

    attention_mask = None
    attention_bias = None
    inference_params = None
    packed_seq_params = None
    sequence_len_offset = None
    rotary_pos_emb = None
    rotary_pos_cos = None
    rotary_pos_sin = None


def build_data():
    """
    Creates a random tensor for testing purposes.

    Returns:
        torch.Tensor: A random tensor of shape (1024, 1, 7168) with bfloat16 dtype
                     and requires_grad set to True.
    """
    hidden_states = torch.randn(*(1024, 1, 7168), dtype=torch.bfloat16, device="cuda") * 100
    hidden_states.requires_grad = True

    return hidden_states


@contextmanager
def deterministic_mode():
    """
    Context manager that sets up a deterministic environment for reproducible testing.

    This function:
    1. Sets environment variables to ensure deterministic behavior in CUDA operations
    2. Configures NCCL communication to be deterministic
    3. Disables non-deterministic algorithms in NVIDIA Transformer Engine
    4. Sets a fixed random seed

    The environment is restored to its previous state when exiting the context.

    Yields:
        None: This is a context manager that yields control back to the caller.
    """
    envs = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_FUSED_ATTN": "0",
        "NCCL_ALGO": "^NVLS",
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "8",
    }
    for k, v in envs.items():
        os.environ[k] = v
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        yield
    finally:
        for k in envs:
            if k in os.environ:
                del os.environ[k]


def run_model_ref_with_capture(model, input_tensors, iterations):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False

    fp8_context = get_fp8_context(model.config) if model.config.fp8 else nullcontext()
    output_tensors = []
    for i in range(iterations):
        with get_fp8_context(model.config):
            output = model(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_model_a2a_overlap_with_capture(model, input_tensors, microbatches):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False

    event = torch.cuda.Event()
    comp_stream = torch.cuda.current_stream()
    com_stream = torch.cuda.Stream(device="cuda")
    layers = [
        TransformerLayerSchedulePlan(model, event, DummyState(), comp_stream, com_stream)
        for _ in range(microbatches)
    ]
    output_tensors = []

    # forward for 1st microbatch
    output, _, _ = schedule_layer_1f1b(layers[0], None, f_input=input_tensors[0], b_grad=None)
    output_tensors.append(output)
    torch.cuda.synchronize()
    # overlapped forward and backward
    pre_backward_dw = None
    for i in range(1, microbatches):
        pre_forward, pre_backward, pre_backward_dw = schedule_layer_1f1b(
            layers[i], layers[i - 1], f_input=input_tensors[i], b_grad=torch.ones_like(output)
        )
        output_tensors.append(pre_forward())
        pre_backward()
        pre_backward_dw()
        torch.cuda.synchronize()
    # backward for last microbatch
    schedule_layer_1f1b(None, layers[-1], f_input=None, b_grad=torch.ones_like(output))
    torch.cuda.synchronize()
    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def reset_model(model, params=None):
    """
    Resets the model's gradients and optionally its parameters.

    Args:
        model: The model to reset.
        params: Optional dictionary of parameter values to restore.
               If None, the current parameter values are saved.

    Returns:
        dict: If params is None, returns a dictionary of the current parameter values.
              Otherwise, returns None.
    """
    model.zero_grad()
    if params is None:
        params = {}
        for name, param in model.named_parameters():
            params[name] = param.data.clone()
        return params
    else:
        for name, param in model.named_parameters():
            param.data.copy_(params[name])


def compare_captures(capture_ref, capture_a2a_overlap, verbose=False):
    """
    Compares two capture dictionaries to check if they contain the same values.

    Args:
        capture_ref: Reference capture dictionary.
        capture_a2a_overlap: All-to-all overlap capture dictionary to compare against.
        verbose: Whether to print detailed information about mismatches.

    Returns:
        tuple: (bool, str) - Whether the captures match and a message describing any mismatch.
    """

    def bit_same(a, b):
        if not a.dtype == b.dtype:
            return False, "dtype mismatch"
        if not a.shape == b.shape:
            return False, "shape mismatch"
        if a.dtype in [torch.bfloat16, torch.half]:
            res = torch.all(a.view(torch.int16) == b.view(torch.int16))
        else:
            res = torch.all(a.view(torch.int32) == b.view(torch.int32))
        if not res and verbose:
            max_diff = torch.abs(a - b)
            max_diff_value = torch.max(max_diff)
            max_diff_index = torch.argmax(max_diff.view(-1))
            flat_original = a.view(-1)
            flat_a2a = b.view(-1)
            print(
                f"max diff: {max_diff_value} at index {max_diff_index}, original/a2a_overlap value at max diff: {flat_original[max_diff_index]}/{flat_a2a[max_diff_index]}"
            )
        return res, "value mismatch"

    msg = ""
    for name, value in capture_ref.items():
        if name not in capture_a2a_overlap:
            msg = f"gradient name mismatch, '{name}' not in capture_a2a_overlap.keys()"
            return False, msg
        if type(value) != type(capture_a2a_overlap[name]):
            msg = f"value type mismatch"
            return False, msg
        if value is None:
            continue
        elif isinstance(value, list):
            if len(value) != len(capture_a2a_overlap[name]):
                msg = "outputs length mismatch"
                return False, msg
            for i in range(len(value)):
                comp_res = bit_same(value[i], capture_a2a_overlap[name][i])
                if not comp_res[0]:
                    msg = f"{comp_res[1]} at index {i}."
                    return False, msg
        elif isinstance(value, torch.Tensor):
            comp_res = bit_same(value, capture_a2a_overlap[name])
            if not comp_res[0]:
                msg = f"{comp_res[1]}"
                return False, msg
        else:
            msg = f"unsupported value type: {type(value)}"
            return False, msg

    return True, "pass"


class TestA2AOverlap:
    """
    Test class for all-to-all overlap optimization in transformer models.

    This class contains tests to verify that the all-to-all overlap optimization
    produces the same results as the reference implementation.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    # TODO: Add flex dispatcher test back in when CI image installs DeepEP.
    @pytest.mark.parametrize("dispatcher_type", ["alltoall"])
    @pytest.mark.parametrize("fp8", [False, True])
    @pytest.mark.parametrize("fp8_recipe", ['tensorwise', 'blockwise'])
    def test_1f1b_overlap(self, dispatcher_type, fp8, fp8_recipe):
        """
        Tests the 1-forward-1-backward overlap optimization.

        This test verifies that the all-to-all overlap optimization produces
        the same results as the reference implementation.
        """

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
        )
        extra_kwargs = {}
        if dispatcher_type == "flex":
            extra_kwargs["moe_enable_deepep"] = True
            extra_kwargs["moe_router_dtype"] = "fp32"
        if fp8:
            extra_kwargs["fp8"] = 'hybrid'
            extra_kwargs["fp8_recipe"] = fp8_recipe
        config = MLATransformerConfig(
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=2,
            deterministic_mode=True,
            combined_1f1b=True,
            combined_1f1b_recipe='ep_a2a',
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            num_layers=16,
            hidden_size=7168,
            add_bias_linear=False,
            num_attention_heads=128,
            ffn_hidden_size=18432,
            kv_channels=128,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            multi_latent_attention=True,
            num_moe_experts=16,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type=dispatcher_type,
            moe_shared_expert_intermediate_size=2048,
            **extra_kwargs,
        )
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=16,
                moe_grouped_gemm=True,
                qk_layernorm=True,
                multi_latent_attention=True,
            )
            model = TransformerLayer(config, transformer_layer_spec.submodules)

            params = reset_model(model)
            input_tensors = [build_data() for _ in range(microbatches)]

            capture_ref = run_model_ref_with_capture(model, input_tensors, microbatches)
            reset_model(model, params)
            capture_a2a_overlap = run_model_a2a_overlap_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
            Utils.destroy_model_parallel()
