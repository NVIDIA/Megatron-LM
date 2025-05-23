# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import os
from contextlib import contextmanager

import pytest
import torch

from megatron.core.models.gpt.fine_grained_callables import build_layer_callables
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class DummyState:
    """
    A dummy state class that holds intermediate results of the transformer layer.

    This class is used to simulate the state object that would normally be passed
    to transformer callables in a real model.
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

    output_tensors = []
    for i in range(iterations):
        output = model(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_model_submodules_with_capture(model, input_tensors, microbatches):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """

    def dummy_detach(t):
        return t

    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False

    output_tensors = []
    # get callables
    callables, dw = build_layer_callables(model)
    attn, post_attn, dispatch, moe, combine, post_process = callables
    assert post_process is None
    for i in range(microbatches):
        # build mock func/state
        state = DummyState()
        node = DummyState()
        node.common_state = DummyState()
        node.chunk_state = DummyState()
        node.detach = dummy_detach

        # attn fwd
        hidden_states = attn(node, input_tensors[i])

        # post attn fwd
        local_tokens, probs = post_attn(node, hidden_states)

        # dispatch fwd
        dispatched_tokens, probs = dispatch(node, local_tokens, probs)

        # moe fwd
        expert_outputs = moe(node, dispatched_tokens, probs)
        if model.mlp.use_shared_expert:
            expert_output, shared_expert_output = expert_outputs
        else:
            expert_output = expert_outputs
            shared_expert_output = None

        # combine fwd
        hidden_states = combine(node, expert_output, shared_expert_output)

        # loss
        output_tensors.append(hidden_states)
        hidden_states.backward(torch.ones_like(hidden_states))

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


def compare_captures(capture_ref, capture_callables, verbose=False):
    """
    Compares two capture dictionaries to check if they contain the same values.

    Args:
        capture_ref: Reference capture dictionary.
        capture_callables: Transformer layer submodule callables dictionary to compare against.
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
        if name not in capture_callables:
            msg = f"gradient name mismatch, '{name}' not in capture_callables.keys()"
            return False, msg
        if type(value) != type(capture_callables[name]):
            msg = f"value type mismatch"
            return False, msg
        if value is None:
            continue
        elif isinstance(value, list):
            if len(value) != len(capture_callables[name]):
                msg = "outputs length mismatch"
                return False, msg
            for i in range(len(value)):
                comp_res = bit_same(value[i], capture_callables[name][i])
                if not comp_res[0]:
                    msg = f"{comp_res[1]} at index {i}."
                    return False, msg
        elif isinstance(value, torch.Tensor):
            comp_res = bit_same(value, capture_callables[name])
            if not comp_res[0]:
                msg = f"{comp_res[1]}"
                return False, msg
        else:
            msg = f"unsupported value type: {type(value)}"
            return False, msg

    return True, "pass"


class TestTransformerLayerSubmoduleCallables:
    """
    Test class for transformer layer submodule callables.

    This class contains tests to verify that the transformer layer submodule callables
    provide the same results as the reference implementation.
    """

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    # TODO: Add flex dispatcher test back in when CI image installs DeepEP.
    @pytest.mark.parametrize("dispatcher_type", ["alltoall"])
    def test_1f1b_overlap(self, dispatcher_type):
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
        config = MLATransformerConfig(
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=2,
            deterministic_mode=True,
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
            capture_callables = run_model_submodules_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_callables, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
            Utils.destroy_model_parallel()
