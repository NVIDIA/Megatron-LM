# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native forward/backward parity for the PR 7054 GLM two-stage KDA port.

The independent reference follows Hugging Face Transformers v5.16.1 exactly:
repository ``https://github.com/huggingface/transformers``, commit
``93c8b7b485963a10800c91f55304db6be211c2bd``, path
``src/transformers/models/glm5_next/modeling_glm5_next.py``.

Relevant source definitions are ``Glm5NextTextForgetGate``,
``Glm5NextTextRMSNormGated``, ``Glm5NextTextLinearAttention``, and
``recurrent_kimi_delta_attention``. The reference intentionally uses only
native PyTorch tensor operations; no FLA or MCore operation is reused.
The pinned model's ``_keep_in_fp32_modules_strict`` keeps ``conv1d`` FP32.
Its fallback casts projected activations to convolution-weight dtype and casts
back only after the convolution and SiLU, including BF16 checkpoint loading.
"""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, KimiDeltaAttention
from megatron.core.ssm.gated_delta_net import common as gdn_common
from megatron.core.ssm.gated_delta_net import kda as kda_module
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils

_HIDDEN_SIZE = 4096
_NUM_HEADS = 64
_HEAD_DIM = 128
_QKV_DIM = _NUM_HEADS * _HEAD_DIM
_LORA_RANK = 128
_CONV_KERNEL_SIZE = 4
_LOWER_BOUND = -5.0
_NORM_EPS = 1e-5

# FLA's native KDA tests allow up to 2% relative-L2 error for the most
# reduction-sensitive gate/beta gradients. For equal-norm tensors that floor
# is about 2e-4 in tensor similarity, so 1e-3 leaves BF16 GEMM/convolution
# headroom while still requiring >0.999 agreement in both direction and energy.
_SIMILARITY_EPS = 1e-3
_SIMILARITY_CHUNK_SIZE = 4 * 1024 * 1024


def test_beta_sigmoid_switch_defaults_off_and_parses_through_real_cli():
    """The GLM beta order is opt-in; dev's FP32-first sigmoid stays the default."""
    from megatron.training.arguments import add_megatron_arguments

    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    assert config.kda_beta_sigmoid_in_projection_dtype is False
    assert KDALayerConfig.from_config(config).kda_beta_sigmoid_in_projection_dtype is False
    config.kda_beta_sigmoid_in_projection_dtype = True
    assert KDALayerConfig.from_config(config).kda_beta_sigmoid_in_projection_dtype is True
    parser = add_megatron_arguments(ArgumentParser(allow_abbrev=False))
    assert parser.parse_args([]).kda_beta_sigmoid_in_projection_dtype is False
    args = parser.parse_args(["--kda-beta-sigmoid-in-projection-dtype"])
    assert args.kda_beta_sigmoid_in_projection_dtype is True


@pytest.mark.parametrize("sigmoid_in_projection_dtype", [False, True])
def test_beta_sigmoid_dtype_order(sigmoid_in_projection_dtype):
    """GLM rounds the sigmoid in BF16 when opted in; the default keeps PR 7054's FP32 order."""
    beta = torch.tensor([-2.25, -0.75, 0.5, 1.25], dtype=torch.bfloat16)
    official = torch.sigmoid(beta).to(torch.float32)
    legacy = torch.sigmoid(beta.to(torch.float32))
    assert not torch.equal(official, legacy), "Fixture must distinguish the rounding orders."
    actual = kda_module._activate_beta(
        beta, sigmoid_in_projection_dtype=sigmoid_in_projection_dtype
    )
    assert actual.dtype == torch.float32
    expected = official if sigmoid_in_projection_dtype else legacy
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _skip_unless_inductor_builds(device: str) -> None:
    """Skip where Inductor cannot build kernels; its CPU backend needs a C++20 compiler."""
    if device == "cuda":
        return
    torch._dynamo.reset()
    try:
        # The unit-test package turns Dynamo errors into eager fallbacks (suppress_errors).
        with torch._dynamo.config.patch(suppress_errors=False):
            torch.compile(lambda tensor: tensor.sigmoid() * 2)(torch.ones(4))
    except Exception as error:  # pylint: disable=broad-except
        pytest.skip(f"Inductor cannot build CPU kernels on this host: {type(error).__name__}")
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize(
    "sigmoid_in_projection_dtype, stand_in_self", [(False, False), (True, False), (False, True)]
)
def test_compiled_direct_gates_follow_beta_switch(
    device, sigmoid_in_projection_dtype, stand_in_self
):
    """The compiled direct-gate _compute_gates follows the beta switch.

    By default it applies dev's FP32 sigmoid. With the switch on, pre_gated_delta_rule has
    already applied the sigmoid in the projection dtype, and the compiled function must
    only promote it, exactly. test_direct_gate_beta_through_compiled_input_preparation
    checks that the rounding survives compilation on the real path. stand_in_self calls
    the function as the determinism tests do, with a self that has only key_head_dim, and
    must get the default order.
    """
    _skip_unless_inductor_builds(device)
    batch, seq_len, heads, head_dim = 2, 64, 4, 8
    generator = torch.Generator().manual_seed(0)
    raw_g = torch.randn(batch, seq_len, heads * head_dim, generator=generator)
    beta = 3 * torch.randn(batch, seq_len, heads, generator=generator)
    raw_g, beta = (tensor.to(device=device, dtype=torch.bfloat16) for tensor in (raw_g, beta))
    official = beta.sigmoid().float()
    legacy = beta.float().sigmoid()
    assert (official - legacy).abs().max() > 1e-4, "Fixture must distinguish the orders."
    if stand_in_self:
        layer = SimpleNamespace(key_head_dim=head_dim)
    else:
        layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
        nn.Module.__init__(layer)
        layer.key_head_dim = head_dim
        layer.beta_sigmoid_in_projection_dtype = sigmoid_in_projection_dtype
    A_log = torch.zeros(heads, device=device)
    dt_bias = torch.zeros(heads * head_dim, device=device)
    torch._dynamo.reset()
    try:
        # Fail on a Dynamo error instead of silently testing eager code.
        with torch._dynamo.config.patch(suppress_errors=False):
            _, gates = KimiDeltaAttention._compute_gates(
                layer,
                A_log,
                dt_bias,
                batch,
                seq_len,
                raw_g,
                beta.sigmoid() if sigmoid_in_projection_dtype else beta,
            )
    finally:
        torch._dynamo.reset()
    actual = gates["beta"]
    assert actual.dtype == torch.float32
    if sigmoid_in_projection_dtype:
        # The compiled promotion of the eager projection-dtype sigmoid is exact.
        assert torch.equal(actual, official)
    else:
        # Compiled FP32 sigmoid may differ from eager by FP32 ulps, far below the BF16
        # rounding (about 1e-3 here) that separates the two orders.
        torch.testing.assert_close(actual, legacy, rtol=0, atol=1e-6)


def _direct_gate_layer(device: str, sigmoid_in_projection_dtype: bool) -> KimiDeltaAttention:
    """A direct-gate KDA stand-in with just what pre_gated_delta_rule reads.

    deterministic_mode selects the PyTorch causal convolution and L2 norm, so the input
    preparation runs without FLA kernels on CPU and GPU.
    """
    heads, head_dim = 4, 8
    qk_dim = heads * head_dim
    layer = KimiDeltaAttention.__new__(KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(deterministic_mode=True, gdn_conv_pad_alignment=None)
    layer.two_stage_gates = False
    layer.cp_size = 1
    layer.qk_dim_local_tp = layer.v_dim_local_tp = qk_dim
    layer.key_head_dim = layer.value_head_dim = head_dim
    layer.use_qk_l2norm = True
    layer.num_key_heads = layer.num_value_heads = heads
    layer.conv_dim_local_tp = 3 * qk_dim
    layer.conv_bias = False
    layer.conv_in_fp32 = False
    layer.act_fn = F.silu
    layer.conv1d = nn.Conv1d(
        3 * qk_dim, 3 * qk_dim, 4, groups=3 * qk_dim, padding=3, bias=False, dtype=torch.bfloat16
    )
    layer.A_log = nn.Parameter(torch.zeros(heads))
    layer.dt_bias = nn.Parameter(torch.zeros(qk_dim))
    layer.beta_sigmoid_in_projection_dtype = sigmoid_in_projection_dtype
    return layer.to(device)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("sigmoid_in_projection_dtype", [False, True])
def test_direct_gate_beta_through_compiled_input_preparation(device, sigmoid_in_projection_dtype):
    """The real direct-gate path gives the kernel the selected beta order, forward and backward.

    The compiled _GDNBase._prepare_input_for_gated_delta_rule inlines the compiled
    _compute_gates. A graph break inside the inner function would make Dynamo call it
    without self, and compiling the projection-dtype sigmoid would drop its rounding;
    calling _compute_gates on its own catches neither.
    """
    _skip_unless_inductor_builds(device)
    batch, seq_len = 2, 64
    layer = _direct_gate_layer(device, sigmoid_in_projection_dtype)
    heads, qk_dim = layer.num_key_heads, layer.qk_dim_local_tp
    generator = torch.Generator().manual_seed(0)
    qkvfg = torch.randn(seq_len, batch, 5 * qk_dim, generator=generator)
    beta = 3 * torch.randn(seq_len, batch, heads, generator=generator)
    qkvfg, beta = (tensor.to(device=device, dtype=torch.bfloat16) for tensor in (qkvfg, beta))
    beta.requires_grad_()
    reference_beta = beta.detach().clone().requires_grad_()
    torch._dynamo.reset()
    try:
        # An eager fallback would hide a rounding dropped by compilation; a dropped self
        # fails either way.
        with torch._dynamo.config.patch(suppress_errors=False):
            kernel_beta = layer.pre_gated_delta_rule(qkvfg, beta, batch, seq_len, 1, None)[4]
    finally:
        torch._dynamo.reset()
    raw = reference_beta.transpose(0, 1)
    official, legacy = raw.sigmoid().float(), raw.float().sigmoid()
    assert (official - legacy).abs().max() > 1e-4, "Fixture must distinguish the orders."
    expected = official if sigmoid_in_projection_dtype else legacy
    assert kernel_beta.dtype == torch.float32
    cotangent = torch.randn(kernel_beta.shape, generator=generator).to(device)
    kernel_beta.backward(cotangent)
    expected.backward(cotangent)
    if sigmoid_in_projection_dtype:
        # Eager BF16 sigmoid plus an exact compiled promotion: bitwise, gradients included.
        assert torch.equal(kernel_beta, expected)
        assert torch.equal(beta.grad, reference_beta.grad)
    else:
        torch.testing.assert_close(kernel_beta, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(beta.grad, reference_beta.grad)


class _NativeForgetGate(nn.Module):
    """PyTorch transcription of ``Glm5NextTextForgetGate``."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.f_a_proj = nn.Linear(_HIDDEN_SIZE, _LORA_RANK, bias=False, device=device, dtype=dtype)
        self.f_b_proj = nn.Linear(_LORA_RANK, _QKV_DIM, bias=False, device=device, dtype=dtype)
        self.dt_bias = nn.Parameter(torch.empty(_QKV_DIM, device=device, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(_NUM_HEADS, device=device, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw_gate = F.linear(F.linear(hidden_states, self.f_a_proj.weight), self.f_b_proj.weight)
        gate = (raw_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(
            hidden_states.shape[0], hidden_states.shape[1], _NUM_HEADS, _HEAD_DIM
        )
        decay_rate = self.A_log.float().view(1, 1, _NUM_HEADS, 1).exp()
        return _LOWER_BOUND * torch.sigmoid(decay_rate * gate)


class _NativeRMSNormGated(nn.Module):
    """Strict-FP32 RMSNorm and sigmoid gate with one final dtype cast."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(_HEAD_DIM, device=device, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + _NORM_EPS)
        hidden_states = hidden_states * self.weight.float()
        hidden_states = hidden_states * torch.sigmoid(gate.float())
        return hidden_states.to(input_dtype)


class _NativeGlm5NextKDA(nn.Module):
    """Official GLM-5.3-Flash KDA expressed with native PyTorch operations."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.q_proj = nn.Linear(_HIDDEN_SIZE, _QKV_DIM, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(_HIDDEN_SIZE, _QKV_DIM, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(_HIDDEN_SIZE, _QKV_DIM, bias=False, device=device, dtype=dtype)
        self.conv1d = nn.Conv1d(
            3 * _QKV_DIM,
            3 * _QKV_DIM,
            kernel_size=_CONV_KERNEL_SIZE,
            groups=3 * _QKV_DIM,
            padding=_CONV_KERNEL_SIZE - 1,
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        self.forget_gate = _NativeForgetGate(device=device, dtype=dtype)
        self.b_proj = nn.Linear(_HIDDEN_SIZE, _NUM_HEADS, bias=False, device=device, dtype=dtype)
        self.g_a_proj = nn.Linear(_HIDDEN_SIZE, _LORA_RANK, bias=False, device=device, dtype=dtype)
        self.g_b_proj = nn.Linear(_LORA_RANK, _QKV_DIM, bias=False, device=device, dtype=dtype)
        self.o_norm = _NativeRMSNormGated(device=device, dtype=dtype)
        self.o_proj = nn.Linear(_QKV_DIM, _HIDDEN_SIZE, bias=False, device=device, dtype=dtype)

    @staticmethod
    def _l2norm(x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) + 1e-6)

    @classmethod
    def _recurrent_kda(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = query.dtype
        query, key, value, gate, beta = (
            tensor.float() for tensor in (query, key, value, gate, beta)
        )
        query = cls._l2norm(query) * (_HEAD_DIM**-0.5)
        key = cls._l2norm(key)

        batch_size, sequence_length = query.shape[:2]
        state = torch.zeros(
            batch_size, _NUM_HEADS, _HEAD_DIM, _HEAD_DIM, device=query.device, dtype=torch.float32
        )
        outputs = []
        for token_idx in range(sequence_length):
            query_i = query[:, token_idx]
            key_i = key[:, token_idx]
            value_i = value[:, token_idx]
            decay_i = gate[:, token_idx].exp().unsqueeze(-1)
            beta_i = beta[:, token_idx].unsqueeze(-1)

            state = state * decay_i
            value_memory = (state * key_i.unsqueeze(-1)).sum(dim=-2)
            delta = (value_i - value_memory) * beta_i
            state = state + key_i.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * query_i.unsqueeze(-1)).sum(dim=-2))

        return torch.stack(outputs, dim=1).to(input_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MCore uses SBHD while the official implementation uses BSHD.
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        batch_size, sequence_length = hidden_states.shape[:2]
        head_shape = (batch_size, sequence_length, _NUM_HEADS, _HEAD_DIM)

        # These are deliberately three independent official F.linear calls. MCore's
        # single in_proj parameter is split explicitly when parameters are copied.
        mixed_qkv = torch.cat(
            [
                F.linear(hidden_states, self.q_proj.weight),
                F.linear(hidden_states, self.k_proj.weight),
                F.linear(hidden_states, self.v_proj.weight),
            ],
            dim=-1,
        ).transpose(1, 2)
        mixed_qkv = F.conv1d(
            mixed_qkv.to(self.conv1d.weight.dtype),
            self.conv1d.weight,
            bias=None,
            padding=_CONV_KERNEL_SIZE - 1,
            groups=3 * _QKV_DIM,
        )[..., :sequence_length]
        mixed_qkv = F.silu(mixed_qkv).to(hidden_states.dtype).transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, _QKV_DIM, dim=-1)
        query = query.view(head_shape)
        key = key.view(head_shape)
        value = value.view(head_shape)

        forget_gate = self.forget_gate(hidden_states)
        beta = torch.sigmoid(F.linear(hidden_states, self.b_proj.weight))
        core_output = self._recurrent_kda(query, key, value, forget_gate, beta)

        output_gate = F.linear(
            F.linear(hidden_states, self.g_a_proj.weight), self.g_b_proj.weight
        ).view(head_shape)
        output = self.o_norm(core_output, output_gate).reshape(
            batch_size, sequence_length, _QKV_DIM
        )
        output = F.linear(output, self.o_proj.weight)
        return output.transpose(0, 1).contiguous()


def _make_config(*, deterministic_mode: bool) -> TransformerConfig:
    return TransformerConfig(
        hidden_size=_HIDDEN_SIZE,
        num_layers=1,
        num_attention_heads=_NUM_HEADS,
        num_query_groups=_NUM_HEADS,
        kv_channels=_HIDDEN_SIZE // _NUM_HEADS,
        linear_conv_kernel_dim=_CONV_KERNEL_SIZE,
        linear_key_head_dim=_HEAD_DIM,
        linear_value_head_dim=_HEAD_DIM,
        linear_num_key_heads=_NUM_HEADS,
        linear_num_value_heads=_NUM_HEADS,
        kda_two_stage_gates=True,
        kda_disable_fp8=True,
        kda_beta_sigmoid_in_projection_dtype=True,
        kda_conv_in_fp32=True,
        kda_safe_gate=True,
        kda_lower_bound=_LOWER_BOUND,
        normalization="RMSNorm",
        layernorm_epsilon=_NORM_EPS,
        layernorm_zero_centered_gamma=False,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        linear_cp_mode="headwise",
        experimental_attention_variant="kda",
        is_hybrid_model=True,
        transformer_impl="transformer_engine",
        deterministic_mode=deterministic_mode,
        gradient_accumulation_fusion=False,
        fp8=None,
    )


def _build_real_kda(config: TransformerConfig) -> KimiDeltaAttention:
    kda_spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    return build_module(
        kda_spec,
        config=config,
        layer_number=1,
        conv_init=0.02,
        pg_collection=ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        ),
    ).cuda()


def _parameter_mapping() -> dict[str, tuple[str, slice | None]]:
    """Map every independent official parameter to its exact MCore storage."""

    return {
        "q_proj.weight": ("in_proj.weight", slice(0, _QKV_DIM)),
        "k_proj.weight": ("in_proj.weight", slice(_QKV_DIM, 2 * _QKV_DIM)),
        "v_proj.weight": ("in_proj.weight", slice(2 * _QKV_DIM, 3 * _QKV_DIM)),
        "conv1d.weight": ("conv1d.weight", None),
        "forget_gate.f_a_proj.weight": ("f_a_proj.weight", None),
        "forget_gate.f_b_proj.weight": ("f_b_proj.weight", None),
        "forget_gate.dt_bias": ("dt_bias", None),
        "forget_gate.A_log": ("A_log", None),
        "b_proj.weight": ("beta_proj.weight", None),
        "g_a_proj.weight": ("g_a_proj.weight", None),
        "g_b_proj.weight": ("g_b_proj.weight", None),
        "o_norm.weight": ("out_norm.weight", None),
        "o_proj.weight": ("out_proj.weight", None),
    }


def _mapped_tensor(tensor: torch.Tensor, row_slice: slice | None) -> torch.Tensor:
    return tensor if row_slice is None else tensor[row_slice]


def _copy_and_validate_parameter_map(
    real: KimiDeltaAttention, reference: _NativeGlm5NextKDA
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    mapping = _parameter_mapping()
    real_parameters = dict(real.named_parameters())
    reference_parameters = dict(reference.named_parameters())

    assert set(reference_parameters) == set(mapping)
    assert set(real_parameters) == {real_name for real_name, _ in mapping.values()}
    assert real_parameters["in_proj.weight"].shape == (3 * _QKV_DIM, _HIDDEN_SIZE)

    with torch.no_grad():
        for reference_name, (real_name, row_slice) in mapping.items():
            source = _mapped_tensor(real_parameters[real_name], row_slice)
            target = reference_parameters[reference_name]
            assert source.shape == target.shape, (
                f"parameter shape mismatch for {reference_name} <- {real_name}: "
                f"{target.shape} != {source.shape}"
            )
            assert source.dtype == target.dtype, (
                f"parameter dtype mismatch for {reference_name} <- {real_name}: "
                f"{target.dtype} != {source.dtype}"
            )
            target.copy_(source)

    return real_parameters, reference_parameters


def _similarity_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Compute cosine and tensor similarity without duplicating production weights."""

    assert actual.shape == expected.shape
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    dot = torch.zeros((), device=actual.device, dtype=torch.float64)
    actual_norm = torch.zeros_like(dot)
    expected_norm = torch.zeros_like(dot)

    for start in range(0, actual_flat.numel(), _SIMILARITY_CHUNK_SIZE):
        end = min(start + _SIMILARITY_CHUNK_SIZE, actual_flat.numel())
        actual_chunk = actual_flat[start:end].double()
        expected_chunk = expected_flat[start:end].double()
        dot = dot + torch.dot(actual_chunk, expected_chunk)
        actual_norm = actual_norm + torch.dot(actual_chunk, actual_chunk)
        expected_norm = expected_norm + torch.dot(expected_chunk, expected_chunk)

    cosine_denom = torch.sqrt(actual_norm * expected_norm)
    tensor_denom = actual_norm + expected_norm
    cosine = 1.0 if cosine_denom.item() == 0.0 else (dot / cosine_denom).item()
    tensor_similarity = 1.0 if tensor_denom.item() == 0.0 else (2.0 * dot / tensor_denom).item()
    return cosine, tensor_similarity


def _assert_similarity(
    actual: torch.Tensor, expected: torch.Tensor, *, label: str, eps: float = _SIMILARITY_EPS
) -> None:
    assert torch.isfinite(actual).all(), f"non-finite values in MCore {label}"
    assert torch.isfinite(expected).all(), f"non-finite values in native {label}"
    assert actual.numel() > 0, f"empty MCore tensor for {label}"
    assert expected.numel() > 0, f"empty native tensor for {label}"
    assert torch.count_nonzero(actual.detach()).item() > 0, f"all-zero MCore {label}"
    assert torch.count_nonzero(expected.detach()).item() > 0, f"all-zero native {label}"
    cosine, tensor_similarity = _similarity_metrics(actual, expected)
    assert cosine > 1.0 - eps, f"{label}: cosine_sim={cosine:.6f}"
    assert tensor_similarity > 1.0 - eps, f"{label}: tensor_sim={tensor_similarity:.6f}"


def _assert_all_parameter_gradients(
    real_parameters: dict[str, nn.Parameter], reference_parameters: dict[str, nn.Parameter]
) -> None:
    mapping = _parameter_mapping()
    covered_real_parameters = set()
    for reference_name, (real_name, row_slice) in mapping.items():
        real_gradient = real_parameters[real_name].grad
        reference_gradient = reference_parameters[reference_name].grad
        assert real_gradient is not None, f"missing MCore gradient for {real_name}"
        assert reference_gradient is not None, f"missing native gradient for {reference_name}"
        _assert_similarity(
            _mapped_tensor(real_gradient, row_slice),
            reference_gradient,
            label=f"parameter gradient {reference_name} <- {real_name}",
        )
        covered_real_parameters.add(real_name)

    assert covered_real_parameters == set(real_parameters)


@pytest.mark.parametrize(
    ("actual_values", "expected_values", "message"),
    [
        ([], [], "empty MCore tensor"),
        ([0.0], [0.0], "all-zero MCore"),
        ([1.0], [0.0], "all-zero native"),
        ([0.0], [1.0], "all-zero MCore"),
    ],
)
def test_similarity_guard_rejects_empty_or_zero_tensors(
    actual_values: list[float], expected_values: list[float], message: str
) -> None:
    """Zero-energy tensors must never satisfy a parity assertion."""

    actual = torch.tensor(actual_values, dtype=torch.float32)
    expected = torch.tensor(expected_values, dtype=torch.float32)
    with pytest.raises(AssertionError, match=message):
        _assert_similarity(actual, expected, label="guard fixture")


def test_similarity_guard_accepts_identical_nonzero_tensors() -> None:
    """The nonzero guard must preserve the existing numerical threshold."""

    tensor = torch.tensor([1.0, -2.0], dtype=torch.float32)
    _assert_similarity(tensor, tensor.clone(), label="guard fixture")


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GLM5-Next KDA parity requires CUDA.")
@pytest.mark.skipif(
    not HAVE_FLA_KDA or kda_module.rms_norm_gated is None,
    reason="FLA with chunk KDA and fused RMSNorm-gate support is required.",
)
@pytest.mark.parametrize(
    "deterministic_mode", [False, True], ids=["fla-causal-conv", "torch-deterministic-causal-conv"]
)
def test_glm5_next_kda_matches_native_forward_backward(deterministic_mode: bool) -> None:
    """Compare the production-shape MCore module with the official recurrent math."""

    if not deterministic_mode and gdn_common.causal_conv1d is None:
        pytest.skip("The fused FLA causal-convolution path is unavailable.")

    Utils.initialize_distributed()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(20260914)
        torch.manual_seed(20260914)
        torch.cuda.manual_seed(20260914)

        config = _make_config(deterministic_mode=deterministic_mode)
        real = _build_real_kda(config)
        device = torch.device("cuda", torch.cuda.current_device())
        reference = _NativeGlm5NextKDA(device=device, dtype=torch.bfloat16)
        real_parameters, reference_parameters = _copy_and_validate_parameter_map(real, reference)

        sequence_length = 8
        batch_size = 1
        hidden_states = torch.randn(
            sequence_length, batch_size, _HIDDEN_SIZE, device=device, dtype=torch.bfloat16
        )
        real_input = hidden_states.detach().clone().requires_grad_(True)
        reference_input = hidden_states.detach().clone().requires_grad_(True)
        output_gradient = torch.randn_like(hidden_states)

        real_output, real_bias = real(real_input, attention_mask=None)
        reference_output = reference(reference_input)
        assert real_bias is None
        _assert_similarity(real_output, reference_output, label="forward output")

        real_output.backward(output_gradient)
        reference_output.backward(output_gradient)
        assert real_input.grad is not None
        assert reference_input.grad is not None
        _assert_similarity(real_input.grad, reference_input.grad, label="input gradient")
        _assert_all_parameter_gradients(real_parameters, reference_parameters)
    finally:
        Utils.destroy_model_parallel()
