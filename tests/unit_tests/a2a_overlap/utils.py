import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from megatron.core import config
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed


@dataclass
class DummyState:
    """
    A dummy state class that holds various attention-related parameters.

    This class is used to simulate the state object that would normally be passed
    to transformer layers in a real model.
    """

    def __getattr__(self, name):
        """
        Get an attribute from the state. If the attribute is not found, set it to None and return None.
        """
        setattr(self, name, None)
        return None


class DummyNode:
    """
    A dummy node class that holds various attention-related parameters.

    This class is used to simulate the node object that would normally be passed
    to transformer layers in a real model.
    """

    common_state = DummyState()
    chunk_state = DummyState()

    def detach(self, x):
        return x


def build_data(seq_len=1024):
    """
    Creates a random tensor for testing purposes.

    Returns:
        torch.Tensor: A random tensor of shape (1024, 1, 1024) with bfloat16 dtype
                     and requires_grad set to True.
    """
    hidden_states = torch.randn(*(seq_len, 1, 512), dtype=torch.bfloat16, device="cuda") * 100
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
    config.ENABLE_EXPERIMENTAL = True
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
    origin_envs = {}
    for k, v in envs.items():
        origin_envs[k] = os.environ.get(k)
        os.environ[k] = v
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        yield
    finally:
        for k in envs:
            if origin_envs[k] is not None:
                os.environ[k] = origin_envs[k]
            elif k in os.environ:
                del os.environ[k]


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


def compare_captures(capture_ref, capture_a2a_overlap, verbose=False, skip_embedding=False):
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
                f"total diff({max_diff.shape}): {torch.abs(max_diff).sum()}, max diff: {max_diff_value} at index {max_diff_index}, original/a2a_overlap value at max diff: {flat_original[max_diff_index]}/{flat_a2a[max_diff_index]}"
            )
        return res, "value mismatch"

    msg = ""
    for name, value in capture_ref.items():
        if skip_embedding and ("embedding" in name or "output_layer" in name):
            capture_ref[name] = None
            capture_a2a_overlap[name] = None
            continue
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
                value[i] = None
                capture_a2a_overlap[name][i] = None
                if not comp_res[0]:
                    msg = f"{comp_res[1]} at index {i}."
                    return False, msg
        elif isinstance(value, torch.Tensor):
            comp_res = bit_same(value, capture_a2a_overlap[name])
            capture_ref[name] = None
            capture_a2a_overlap[name] = None
            if not comp_res[0]:
                msg = f"{name}: {comp_res[1]}"
                return False, msg
        else:
            msg = f"unsupported value type: {type(value)}"
            return False, msg

    return True, "pass"


def get_test_config(num_layers=1, num_moe_experts=8, extra_kwargs={}, moe_grouped_gemm=True):
    config = MLATransformerConfig(
        attention_backend="unfused",
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=4 if num_moe_experts is not None else 1,
        deterministic_mode=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_layers=num_layers,
        hidden_size=512,
        add_bias_linear=False,
        num_attention_heads=128,
        ffn_hidden_size=512,
        kv_channels=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        multi_latent_attention=True,
        num_moe_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        **extra_kwargs,
    )
    return config


def get_valid_token_dispatcher_types():
    try:
        from deep_ep import Buffer
        from deep_ep.utils import EventHandle, EventOverlap

        return ["alltoall", "flex"]
    except ImportError:
        return ["alltoall"]


def get_valid_fp8_flags():
    from megatron.core.enums import Fp8Recipe

    fp8_types = ["e4m3", "hybrid"]
    recipes = []
    valid_flags = []
    if is_te_min_version("2.3.0.dev0"):
        recipes.append(Fp8Recipe.blockwise)
        recipes.append(Fp8Recipe.tensorwise)

    for fp8_type in fp8_types:
        for recipe in recipes:
            valid_flags.append((fp8_type, recipe))
    valid_flags.append(None)

    return valid_flags
