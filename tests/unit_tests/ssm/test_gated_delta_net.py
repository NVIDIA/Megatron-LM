# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.ssm.gated_delta_net.gdn as gdn_module
from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    HAVE_FLA,
    HAVE_TE_GDN,
    GatedDeltaNet,
    torch_chunk_gated_delta_rule,
)
from megatron.core.ssm.gated_delta_net.common import (
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    chunk_gated_delta_rule,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.gated_delta_net_test_cases import (
    HAVE_FUSED_PRE_GDR,
    GatedDeltaNetModuleTestCases,
)
from tests.unit_tests.ssm.gated_delta_net_test_utils import _unpack_sequence
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(autouse=True)
def fresh_gdn_compile_cache(request, monkeypatch):
    """Isolate compiler state and the reduction algorithm used in layout comparisons."""
    # Parameter-vs-Tensor CP views and shape variants can exhaust Dynamo's default
    # recompile budget across cases. A later SBHD/THD pair would then compare eager
    # with compiled BF16 arithmetic. Keep the default limit and share the cache
    # within each comparison, without disabling compilation or relaxing tolerances.
    torch.compiler.reset()
    if request.node.originalname in (
        "test_gpu_forward_thd_correctness",
        "test_gpu_forward_thd_padding_correctness",
    ):
        from fla.modules.l2norm import l2norm_fwd_kernel

        # Inductor specializes the FLA call for each layout. Independent tuning
        # choices can change BF16 tie rounding on Blackwell. Compare layouts with
        # one supported reduction configuration; leave production tuning alone.
        config = next(
            config
            for config in l2norm_fwd_kernel.configs
            if config.kwargs.get("BT") == 32 and config.num_warps == 4
        )
        monkeypatch.setattr(l2norm_fwd_kernel, "configs", [config])
        monkeypatch.setattr(l2norm_fwd_kernel, "cache", {})
        monkeypatch.setattr(
            l2norm_fwd_kernel,
            "best_config",
            getattr(l2norm_fwd_kernel, "best_config", None),
            raising=False,
        )
    try:
        yield
    finally:
        torch.compiler.reset()


def _assert_relative_rms_close(
    actual: torch.Tensor, expected: torch.Tensor, tolerance: float, name: str
) -> None:
    actual = actual.detach().double()
    expected = expected.detach().double()
    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
    assert torch.isfinite(expected).all(), f"reference {name} contains non-finite values"
    error_rms = (actual - expected).square().mean().sqrt()
    expected_rms = expected.square().mean().sqrt().clamp_min(1e-12)
    relative_rms = (error_rms / expected_rms).item()
    assert (
        relative_rms < tolerance
    ), f"{name} relative RMS error {relative_rms:.4g} exceeds tolerance {tolerance}"


def _make_gdn_variant_stub(backend: str, *, deterministic_mode: bool = False) -> SimpleNamespace:
    """Build the minimal state needed to exercise GDN backend selection."""
    stub = SimpleNamespace(
        config=SimpleNamespace(
            deterministic_mode=deterministic_mode,
            gdn_kernel_backend=backend,
            gdn_pre_gated_delta_rule_fusion=False,
            linear_cp_mode="headwise",
            dynamic_context_parallel=False,
        ),
        num_value_heads=2,
        qk_dim_local_tp=64,
        v_dim_local_tp=128,
        tp_size=1,
        cp_size=1,
        num_v_heads_local_tp=2,
        key_head_dim=64,
        value_head_dim=64,
        layer_number=1,
    )
    stub._get_feat_dim_split = partial(GatedDeltaNet._get_feat_dim_split, stub)
    return stub


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_gdn_selects_fla_backend(monkeypatch):
    """An explicit FLA selection bypasses TE."""

    class UnexpectedTEGatedDeltaNetAttention:
        def __init__(self, **kwargs):
            raise AssertionError(f"TE should not be constructed: {kwargs}")

    monkeypatch.setattr(gdn_module, "HAVE_TE_GDN", True)
    monkeypatch.setattr(gdn_module, "TEGatedDeltaNetAttention", UnexpectedTEGatedDeltaNetAttention)
    gdn = _make_gdn_variant_stub("fla")

    GatedDeltaNet._setup_variant_attrs(gdn)

    assert gdn.gdn_backend == "fla"
    assert gdn.gated_delta_rule is chunk_gated_delta_rule


def test_gdn_rejects_unavailable_explicit_te_backend(monkeypatch):
    """An explicit TE selection fails instead of silently using FLA."""
    monkeypatch.setattr(gdn_module, "HAVE_TE_GDN", False)
    gdn = _make_gdn_variant_stub("transformer_engine")

    with pytest.raises(ImportError, match="requires TransformerEngine's GatedDeltaNetAttention"):
        GatedDeltaNet._setup_variant_attrs(gdn)


def test_gdn_selects_te_backend(monkeypatch):
    """An explicit TE selection constructs the GatedDeltaNetAttention (GDN) adapter."""

    class FakeTEGatedDeltaNetAttention:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(gdn_module, "HAVE_TE_GDN", True)
    monkeypatch.setattr(gdn_module, "TEGatedDeltaNetAttention", FakeTEGatedDeltaNetAttention)
    gdn = _make_gdn_variant_stub("transformer_engine")

    GatedDeltaNet._setup_variant_attrs(gdn)

    assert gdn.gdn_backend == "transformer_engine"
    assert isinstance(gdn.core_attention, FakeTEGatedDeltaNetAttention)
    assert gdn.gated_delta_rule is gdn.core_attention


def test_gdn_selects_explicit_torch_backend():
    """The PyTorch reference kernel is selected only when explicitly configured."""
    gdn = _make_gdn_variant_stub("torch", deterministic_mode=True)

    GatedDeltaNet._setup_variant_attrs(gdn)

    assert gdn.gdn_backend == "torch"
    assert gdn.gated_delta_rule is torch_chunk_gated_delta_rule


@pytest.mark.parametrize("backend", ["transformer_engine", "fla"])
def test_gdn_rejects_nondeterministic_backend_in_deterministic_mode(backend):
    """Deterministic mode must not silently replace the configured backend."""
    gdn = _make_gdn_variant_stub(backend, deterministic_mode=True)

    with pytest.raises(ValueError, match="requires gdn_kernel_backend='torch'"):
        GatedDeltaNet._setup_variant_attrs(gdn)


@pytest.mark.parametrize("backend", ["transformer_engine", "fla"])
def test_gdn_config_rejects_nondeterministic_backend_in_deterministic_mode(backend):
    """Invalid deterministic GDN backend combinations fail during config construction."""
    with pytest.raises(ValueError, match="requires gdn_kernel_backend='torch'"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            experimental_attention_variant="gated_delta_net",
            deterministic_mode=True,
            gdn_kernel_backend=backend,
        )


@pytest.mark.parametrize("backend", ["auto", "unsupported"])
def test_gdn_kernel_backend_validation(backend):
    """Direct config construction rejects unsupported GDN backend names."""
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=2)
    assert config.gdn_kernel_backend == "fla"
    torch_config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=2, gdn_kernel_backend="torch"
    )
    assert torch_config.gdn_kernel_backend == "torch"

    with pytest.raises(ValueError, match="gdn_kernel_backend must be one of"):
        TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=2, gdn_kernel_backend=backend
        )


def _make_gdn_config(**overrides):
    config_kwargs = {
        "hidden_size": 128,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "num_layers": 1,
        "normalization": "RMSNorm",
        "use_cpu_initialization": True,
        "layernorm_zero_centered_gamma": True,
        "num_attention_heads": 8,
        "activation_func": F.silu,
        "bf16": True,
        "experimental_attention_variant": "gated_delta_net",
        "linear_cp_mode": "headwise",
        "linear_attention_freq": [1],
        "transformer_impl": "transformer_engine",
    }
    config_kwargs.update(overrides)
    return TransformerConfig(**config_kwargs)


def test_gdn_pre_gated_delta_rule_fusion_defaults_to_disabled():
    config = _make_gdn_config()
    assert not config.gdn_pre_gated_delta_rule_fusion


def test_gdn_pre_gated_delta_rule_fusion_accepts_gdn_variant():
    config = _make_gdn_config(gdn_pre_gated_delta_rule_fusion=True)
    assert config.gdn_pre_gated_delta_rule_fusion


def test_gdn_pre_gated_delta_rule_fusion_requires_gdn_variant():
    with pytest.raises(ValueError, match="experimental_attention_variant='gdn'"):
        _make_gdn_config(
            experimental_attention_variant=None,
            linear_attention_freq=None,
            gdn_pre_gated_delta_rule_fusion=True,
        )


@pytest.mark.parametrize("use_gdn2", [False], ids=["gdn"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet(GatedDeltaNetModuleTestCases):
    """GDN assertions shared with the separately scheduled GDN2 wrapper."""


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not HAVE_FUSED_PRE_GDR, reason="causal-conv1d fused backward is not installed.")
@pytest.mark.internal
class TestFusedPreGatedDeltaRule:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        self.pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.unfused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=False)
        self.fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False
        )
        self.fused_gdn.load_state_dict(self.unfused_gdn.state_dict())

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _build_gdn(
        self,
        gdn_pre_gated_delta_rule_fusion: bool,
        *,
        deterministic_mode: bool = True,
        conv_kernel_dim: int = 2,
    ):
        transformer_config = TransformerConfig(
            hidden_size=256,
            linear_conv_kernel_dim=conv_kernel_dim,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
            deterministic_mode=deterministic_mode,
            # deterministic_mode no longer implies the torch kernel; the backend must be
            # named explicitly, so keep this helper on the kernel each mode used before.
            gdn_kernel_backend="torch" if deterministic_mode else "fla",
            gdn_pre_gated_delta_rule_fusion=gdn_pre_gated_delta_rule_fusion,
        )
        gdn_submodules = get_experimental_attention_variant_module_spec(
            config=transformer_config
        ).submodules
        gdn = GatedDeltaNet(
            transformer_config,
            submodules=gdn_submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=self.pg_collection,
        )
        return gdn.cuda().bfloat16()

    def _packed_pre_gated_delta_rule_reference(self, gdn, qkvzba, cu_seqlens):
        """Run the unfused pre-GDR path independently on each packed sequence."""

        segment_outputs = [[] for _ in range(6)]
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            outputs = gdn.pre_gated_delta_rule(
                qkvzba[start:end],
                batch=1,
                seq_len=end - start,
                cp_size_headwise=gdn.cp_size,
                cp_group_headwise=gdn.pg_collection.cp,
            )
            for output_list, output in zip(segment_outputs, outputs):
                output_list.append(output)
        return tuple(torch.cat(outputs, dim=1) for outputs in segment_outputs)

    def _assert_pre_gated_delta_rule_outputs_close(
        self, fused_outputs, unfused_outputs, *, atol: float, rtol: float, output_tolerances=None
    ):
        """Compare named pre-GDR outputs with optional per-output tolerances."""

        output_names = ("query", "key", "value", "gate", "beta", "g")
        output_tolerances = output_tolerances or {}
        for name, fused, unfused in zip(output_names, fused_outputs, unfused_outputs):
            output_atol, output_rtol = output_tolerances.get(name, (atol, rtol))
            torch.testing.assert_close(
                fused,
                unfused,
                atol=output_atol,
                rtol=output_rtol,
                msg=lambda msg, output_name=name: f"{output_name} mismatch: {msg}",
            )

    def test_fused_and_unfused_forward_match(self):
        hidden_states = torch.randn(
            (32, 2, self.unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            unfused_output, unfused_bias = self.unfused_gdn(hidden_states, None)
            fused_output, fused_bias = self.fused_gdn(hidden_states, None)

        torch.testing.assert_close(fused_output, unfused_output, atol=1e-3, rtol=1e-3)
        assert fused_bias == unfused_bias

    def test_fused_and_unfused_forward_thd_match(self):
        unfused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(unfused_gdn.state_dict())

        hidden_states = torch.randn(
            (32, 1, unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        cu_seqlens = torch.tensor(
            [0, 1, 4, 11, 32], device=torch.cuda.current_device(), dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=21,
            max_seqlen_kv=21,
            total_tokens=hidden_states.shape[0],
        )
        assert packed_seq_params.seq_idx is not None

        with torch.no_grad():
            unfused_output, unfused_bias = unfused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )
            fused_output, fused_bias = fused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )

        torch.testing.assert_close(fused_output, unfused_output, atol=2e-3, rtol=2e-3)
        assert fused_bias == unfused_bias

    def test_fused_and_unfused_forward_thd_padding_match(self):
        unfused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(unfused_gdn.state_dict())

        hidden_states = torch.randn(
            (12, 1, unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        cu_seqlens = torch.tensor(
            [0, 1, 4, 9], device=torch.cuda.current_device(), dtype=torch.int32
        )
        cu_seqlens_padded = torch.tensor(
            [0, 2, 6, 12], device=torch.cuda.current_device(), dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=6,
            max_seqlen_kv=6,
            total_tokens=hidden_states.shape[0],
        )
        assert packed_seq_params.seq_idx is not None

        with torch.no_grad():
            unfused_output, unfused_bias = unfused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )
            fused_output, fused_bias = fused_gdn(
                hidden_states, None, packed_seq_params=packed_seq_params
            )

        torch.testing.assert_close(fused_output, unfused_output, atol=2e-3, rtol=2e-3)
        assert fused_bias == unfused_bias

    def test_fused_and_unfused_pre_gated_delta_rule_match(self):
        batch = 2
        seq_len = 32
        hidden_states = torch.randn(
            (seq_len, batch, self.unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            qkvzba, _ = self.unfused_gdn.in_proj(hidden_states)
            unfused_outputs = self.unfused_gdn.pre_gated_delta_rule(
                qkvzba, batch, seq_len, self.unfused_gdn.cp_size, self.unfused_gdn.pg_collection.cp
            )
            fused_outputs = self.fused_gdn._fused_streamed_pre_gated_delta_rule(qkvzba)

        self._assert_pre_gated_delta_rule_outputs_close(
            fused_outputs,
            unfused_outputs,
            atol=1e-3,
            rtol=1e-3,
            output_tolerances={"g": (1e-3, 3e-3)},
        )

    def test_fused_and_unfused_beta_use_fp32_sigmoid(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=True, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 2
        seq_len = 32
        device = torch.cuda.current_device()
        num_value_heads = reference_gdn.num_v_heads_local_tp
        beta_channel_offset = 2 * reference_gdn.qk_dim_local_tp + 2 * reference_gdn.v_dim_local_tp
        qkvzba = torch.randn(
            (seq_len, batch, reference_gdn.in_proj_dim), device=device, dtype=torch.bfloat16
        )
        beta_raw = torch.linspace(
            -6.0, 6.0, seq_len * batch * num_value_heads, device=device, dtype=torch.float32
        ).reshape(seq_len, batch, num_value_heads)
        qkvzba[:, :, beta_channel_offset : beta_channel_offset + num_value_heads] = beta_raw.to(
            torch.bfloat16
        )
        expected_beta = (
            qkvzba[:, :, beta_channel_offset : beta_channel_offset + num_value_heads]
            .transpose(0, 1)
            .float()
            .sigmoid()
        )

        qkvzba_unfused = qkvzba.detach().clone().requires_grad_(True)
        qkvzba_fused = qkvzba.detach().clone().requires_grad_(True)
        reference_gdn.zero_grad(set_to_none=True)
        fused_gdn.zero_grad(set_to_none=True)

        unfused_outputs = reference_gdn.pre_gated_delta_rule(
            qkvzba_unfused, batch, seq_len, reference_gdn.cp_size, reference_gdn.pg_collection.cp
        )
        fused_outputs = fused_gdn._fused_streamed_pre_gated_delta_rule(qkvzba_fused)
        unfused_beta = unfused_outputs[4]
        fused_beta = fused_outputs[4]

        assert unfused_beta.dtype == torch.float32
        assert fused_beta.dtype == torch.float32
        torch.testing.assert_close(unfused_beta, expected_beta, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(fused_beta, expected_beta, atol=1e-5, rtol=1e-5)

        beta_grad = torch.linspace(
            -0.25, 0.25, unfused_beta.numel(), device=device, dtype=torch.float32
        ).reshape_as(unfused_beta)

        def _beta_only_loss(outputs):
            loss = outputs[4].float().mul(beta_grad).sum()
            return loss + sum(
                output.float().sum() * 0.0 for idx, output in enumerate(outputs) if idx != 4
            )

        _beta_only_loss(unfused_outputs).backward()
        _beta_only_loss(fused_outputs).backward()
        beta_slice = slice(beta_channel_offset, beta_channel_offset + num_value_heads)
        torch.testing.assert_close(
            qkvzba_fused.grad[:, :, beta_slice].float(),
            qkvzba_unfused.grad[:, :, beta_slice].float(),
            atol=2e-3,
            rtol=2e-3,
        )

    def test_fused_and_unfused_pre_gated_delta_rule_backward_match(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=True, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 2
        seq_len = 32
        qkvzba = torch.randn(
            (seq_len, batch, reference_gdn.in_proj_dim),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        qkvzba_unfused = qkvzba.detach().clone().requires_grad_(True)
        qkvzba_fused = qkvzba.detach().clone().requires_grad_(True)

        reference_gdn.zero_grad(set_to_none=True)
        fused_gdn.zero_grad(set_to_none=True)

        unfused_outputs = reference_gdn.pre_gated_delta_rule(
            qkvzba_unfused, batch, seq_len, reference_gdn.cp_size, reference_gdn.pg_collection.cp
        )
        fused_outputs = fused_gdn._fused_streamed_pre_gated_delta_rule(qkvzba_fused)
        grad_outputs = [torch.randn_like(output.float()) for output in unfused_outputs]

        unfused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(unfused_outputs, grad_outputs)
        )
        fused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(fused_outputs, grad_outputs)
        )
        unfused_loss.backward()
        fused_loss.backward()

        torch.testing.assert_close(qkvzba_fused.grad, qkvzba_unfused.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            fused_gdn.conv1d.weight.grad, reference_gdn.conv1d.weight.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.A_log.grad, reference_gdn.A_log.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.dt_bias.grad, reference_gdn.dt_bias.grad, atol=3e-2, rtol=3e-2
        )

    def test_fused_and_unfused_packed_pre_gated_delta_rule_forward_match(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=True, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 1
        cu_seqlens = torch.tensor(
            [0, 1, 4, 6, 11], device=torch.cuda.current_device(), dtype=torch.int32
        )
        seq_len = cu_seqlens[-1].item()
        qkvzba = torch.randn(
            (seq_len, batch, reference_gdn.in_proj_dim),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        with torch.no_grad():
            unfused_outputs = self._packed_pre_gated_delta_rule_reference(
                reference_gdn, qkvzba, cu_seqlens
            )
            fused_outputs = fused_gdn._fused_streamed_pre_gated_delta_rule(
                qkvzba, cu_seqlens_q=cu_seqlens
            )

        self._assert_pre_gated_delta_rule_outputs_close(
            fused_outputs, unfused_outputs, atol=2e-3, rtol=2e-3
        )

    def test_fused_packed_g_softplus_matches_torch_for_large_alpha(self):
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        device = torch.cuda.current_device()
        batch = 1
        cu_seqlens = torch.tensor([0, 3, 8], device=device, dtype=torch.int32)
        seq_len = cu_seqlens[-1].item()
        num_value_heads = fused_gdn.num_v_heads_local_tp
        beta_channel_offset = 2 * fused_gdn.qk_dim_local_tp + 2 * fused_gdn.v_dim_local_tp
        alpha_channel_offset = beta_channel_offset + num_value_heads

        qkvzba = torch.zeros(
            (seq_len, batch, fused_gdn.in_proj_dim),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with torch.no_grad():
            qkvzba[:, :, alpha_channel_offset : alpha_channel_offset + num_value_heads] = 100.0
            fused_gdn.A_log.zero_()
            fused_gdn.dt_bias.zero_()
        fused_gdn.zero_grad(set_to_none=True)

        *_, g = fused_gdn._fused_streamed_pre_gated_delta_rule(qkvzba, cu_seqlens_q=cu_seqlens)

        alpha = qkvzba[
            :, :, alpha_channel_offset : alpha_channel_offset + num_value_heads
        ].transpose(0, 1)
        expected_g = -torch.exp(fused_gdn.A_log.float()).view(1, 1, -1) * F.softplus(
            alpha.float() + fused_gdn.dt_bias.float().view(1, 1, -1)
        )
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g.float(), expected_g, atol=0.0, rtol=0.0)

        g.float().sum().backward()
        expected_alpha_grad = -torch.ones_like(alpha.float())
        expected_A_log_grad = expected_g.sum(dim=(0, 1))
        expected_dt_bias_grad = -torch.full_like(fused_gdn.dt_bias.float(), seq_len * batch)
        alpha_grad = qkvzba.grad[
            :, :, alpha_channel_offset : alpha_channel_offset + num_value_heads
        ].transpose(0, 1)

        assert torch.isfinite(alpha_grad).all()
        assert torch.isfinite(fused_gdn.A_log.grad).all()
        assert torch.isfinite(fused_gdn.dt_bias.grad).all()
        torch.testing.assert_close(alpha_grad.float(), expected_alpha_grad, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            fused_gdn.A_log.grad.float(), expected_A_log_grad, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            fused_gdn.dt_bias.grad.float(), expected_dt_bias_grad, atol=0.0, rtol=0.0
        )

    def test_fused_and_unfused_packed_pre_gated_delta_rule_backward_match(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=True, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 1
        cu_seqlens = torch.tensor(
            [0, 1, 4, 6, 11], device=torch.cuda.current_device(), dtype=torch.int32
        )
        seq_len = cu_seqlens[-1].item()
        qkvzba = torch.randn(
            (seq_len, batch, reference_gdn.in_proj_dim),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        qkvzba_unfused = qkvzba.detach().clone().requires_grad_(True)
        qkvzba_fused = qkvzba.detach().clone().requires_grad_(True)

        reference_gdn.zero_grad(set_to_none=True)
        fused_gdn.zero_grad(set_to_none=True)

        unfused_outputs = self._packed_pre_gated_delta_rule_reference(
            reference_gdn, qkvzba_unfused, cu_seqlens
        )
        fused_outputs = fused_gdn._fused_streamed_pre_gated_delta_rule(
            qkvzba_fused, cu_seqlens_q=cu_seqlens
        )
        grad_outputs = [torch.randn_like(output.float()) for output in unfused_outputs]

        unfused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(unfused_outputs, grad_outputs)
        )
        fused_loss = sum(
            (output.float() * grad).sum() for output, grad in zip(fused_outputs, grad_outputs)
        )
        unfused_loss.backward()
        fused_loss.backward()

        torch.testing.assert_close(qkvzba_fused.grad, qkvzba_unfused.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            fused_gdn.conv1d.weight.grad, reference_gdn.conv1d.weight.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.A_log.grad, reference_gdn.A_log.grad, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            fused_gdn.dt_bias.grad, reference_gdn.dt_bias.grad, atol=3e-2, rtol=3e-2
        )

    def test_fused_packed_conv_forward_boundary_isolation(self):
        from megatron.core.fusions.fused_pre_gated_delta_rule import (
            fused_streamed_pre_gated_delta_rule,
        )

        seq_len = 5
        boundary = 3
        num_key_heads = 1
        num_value_heads = 4
        key_head_dim = 32
        value_head_dim = 32
        conv_width = 4
        qk_channels = num_key_heads * key_head_dim
        v_channels = num_value_heads * value_head_dim
        k_offset = qk_channels
        v_offset = 2 * qk_channels
        total_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
        device = torch.cuda.current_device()

        qkvzba = torch.zeros((seq_len, 1, total_channels), device=device, dtype=torch.bfloat16)
        qkvzba[boundary - 1, 0, :qk_channels] = 10.0
        qkvzba[boundary - 1, 0, k_offset : k_offset + qk_channels] = 10.0
        qkvzba[boundary - 1, 0, v_offset : v_offset + v_channels] = 10.0
        conv_weight = torch.zeros((2 * qk_channels + v_channels, 1, conv_width), device=device)
        conv_weight[:qk_channels, 0, conv_width - 2] = 1.0
        conv_weight[k_offset : k_offset + qk_channels, 0, conv_width - 2] = 1.0
        conv_weight[v_offset : v_offset + v_channels, 0, conv_width - 2] = 1.0
        A_log = torch.zeros((num_value_heads,), device=device, dtype=torch.bfloat16)
        dt_bias = torch.zeros((num_value_heads,), device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, boundary, seq_len], device=device, dtype=torch.int32)

        query, key, value, _, _, _ = fused_streamed_pre_gated_delta_rule(
            qkvzba,
            conv_weight.to(torch.bfloat16),
            None,
            A_log,
            dt_bias,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            cu_seqlens=cu_seqlens,
        )

        torch.testing.assert_close(
            query[0, boundary], torch.zeros_like(query[0, boundary]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            key[0, boundary], torch.zeros_like(key[0, boundary]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            value[0, boundary], torch.zeros_like(value[0, boundary]), atol=0.0, rtol=0.0
        )

    def test_fused_packed_conv_backward_boundary_isolation(self):
        from megatron.core.fusions.fused_pre_gated_delta_rule import (
            fused_streamed_pre_gated_delta_rule,
        )

        seq_len = 5
        boundary = 3
        num_key_heads = 1
        num_value_heads = 4
        key_head_dim = 32
        value_head_dim = 32
        conv_width = 4
        qk_channels = num_key_heads * key_head_dim
        v_channels = num_value_heads * value_head_dim
        k_offset = qk_channels
        v_offset = 2 * qk_channels
        total_channels = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
        device = torch.cuda.current_device()

        qkvzba = torch.zeros(
            (seq_len, 1, total_channels), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        conv_weight = torch.zeros(
            (2 * qk_channels + v_channels, 1, conv_width),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with torch.no_grad():
            qkvzba[boundary - 1, 0, :qk_channels] = 10.0
            qkvzba[boundary - 1, 0, k_offset : k_offset + qk_channels] = 10.0
            qkvzba[boundary - 1, 0, v_offset : v_offset + v_channels] = 10.0
            conv_weight[:qk_channels, 0, conv_width - 2] = 1.0
            conv_weight[k_offset : k_offset + qk_channels, 0, conv_width - 2] = 1.0
            conv_weight[v_offset : v_offset + v_channels, 0, conv_width - 2] = 1.0
        A_log = torch.zeros(
            (num_value_heads,), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        dt_bias = torch.zeros(
            (num_value_heads,), device=device, dtype=torch.bfloat16, requires_grad=True
        )
        cu_seqlens = torch.tensor([0, boundary, seq_len], device=device, dtype=torch.int32)

        query, key, value, gate, beta, g = fused_streamed_pre_gated_delta_rule(
            qkvzba,
            conv_weight,
            None,
            A_log,
            dt_bias,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            cu_seqlens=cu_seqlens,
        )

        loss = (
            query[0, boundary].float().sum()
            + key[0, boundary].float().sum()
            + value[0, boundary].float().sum()
        )
        loss = loss + 0.0 * (gate.float().sum() + beta.float().sum() + g.float().sum())
        loss.backward()

        leaked_q_grad = qkvzba.grad[boundary - 1, 0, :qk_channels]
        leaked_k_grad = qkvzba.grad[boundary - 1, 0, k_offset : k_offset + qk_channels]
        leaked_v_grad = qkvzba.grad[boundary - 1, 0, v_offset : v_offset + v_channels]
        torch.testing.assert_close(
            leaked_q_grad, torch.zeros_like(leaked_q_grad), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            leaked_k_grad, torch.zeros_like(leaked_k_grad), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            leaked_v_grad, torch.zeros_like(leaked_v_grad), atol=0.0, rtol=0.0
        )


@pytest.mark.skipif(not HAVE_TE_GDN, reason="TransformerEngine GDN is not available.")
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
def test_te_gdn_matches_previous_fla_path():
    """TE's fused GDN core matches the previously used FLA rule."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        config = TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=2,
            num_query_groups=2,
            activation_func=F.silu,
            bf16=True,
            experimental_attention_variant="gated_delta_net",
            gdn_kernel_backend="transformer_engine",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        submodules = get_experimental_attention_variant_module_spec(config=config).submodules
        gdn = (
            GatedDeltaNet(
                config,
                submodules=submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                conv_init=1.0,
                use_qk_l2norm=True,
                A_init_range=(1, 16),
                pg_collection=pg_collection,
            )
            .cuda()
            .bfloat16()
        )

        batch, sequence = 2, 64
        heads = config.linear_num_value_heads
        q = F.normalize(
            torch.randn(batch, sequence, heads, config.linear_key_head_dim, device="cuda").float(),
            dim=-1,
        ).bfloat16()
        k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).bfloat16()
        v = (
            torch.randn(batch, sequence, heads, config.linear_value_head_dim, device="cuda")
            .mul_(0.1)
            .bfloat16()
        )
        g = torch.empty(batch, sequence, heads, device="cuda").uniform_(0.1, 1.0).log()
        beta = torch.rand(batch, sequence, heads, device="cuda").bfloat16()
        output_grad = torch.randn_like(v, dtype=torch.float32)

        def run(gated_delta_rule):
            inputs = {
                name: tensor.detach().clone().requires_grad_(True)
                for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
            }
            output, _ = gated_delta_rule(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                g=inputs["g"],
                beta=inputs["beta"],
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=None,
            )
            (output.float() * output_grad).sum().backward()
            return output.detach(), {name: tensor.grad.detach() for name, tensor in inputs.items()}

        te_gated_delta_rule = gdn.gated_delta_rule
        te_output, te_grads = run(te_gated_delta_rule)
        fla_output, fla_grads = run(chunk_gated_delta_rule)

        _assert_relative_rms_close(te_output, fla_output, 2e-2, "output")
        for name in te_grads:
            _assert_relative_rms_close(te_grads[name], fla_grads[name], 5e-2, f"gradient {name}")
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGDNCuSeqlensResolve:

    @pytest.fixture
    def mock_gdn(self):
        class MockGDN:
            _resolve_cu_seqlens = GatedDeltaNet._resolve_cu_seqlens

        return MockGDN()

    def test_padded_preferred_when_available(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        padded = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(padded, actual, 1008, "cu_seqlens_q", cp_size=2)
        assert torch.equal(result, padded)

    def test_actual_used_when_no_padding(self, mock_gdn):
        actual = torch.tensor([0, 504, 1008], dtype=torch.int32)
        result = mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=2)
        assert torch.equal(result, actual)

    def test_raises_when_padding_mismatch(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=2)

    def test_raises_when_padded_mismatches_total(self, mock_gdn):
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        padded = torch.tensor([0, 504, 1004], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(padded, actual, 1008, "cu_seqlens_q", cp_size=2)

    def test_raises_when_not_divisible_by_cp_size(self, mock_gdn):
        actual = torch.tensor([0, 505, 1008], dtype=torch.int32)
        with pytest.raises(ValueError, match="must be divisible by cp_size"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=2)

    def test_cp1_still_validates_total(self, mock_gdn):
        mock_gdn.cp_size = 1
        actual = torch.tensor([0, 500, 1000], dtype=torch.int32)
        with pytest.raises(ValueError, match="does not match"):
            mock_gdn._resolve_cu_seqlens(None, actual, 1008, "cu_seqlens_q", cp_size=1)


def _have_cudnn_frontend_min_version(min_version: str = "1.29.0") -> bool:
    """Check whether the installed nvidia-cudnn-frontend package meets a minimum version."""
    try:
        from importlib.metadata import version as pkg_version

        from packaging.version import Version

        return Version(pkg_version("nvidia-cudnn-frontend")) >= Version(min_version)
    except Exception:
        return False


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 2),  # CP
        (2, False, 2),  # TP w/o SP + CP
        (2, True, 2),  # TP w/ SP + CP
    ],
)
@pytest.mark.parametrize(
    "gdn_kernel_backend", ["fla", pytest.param("transformer_engine", marks=pytest.mark.internal)]
)
@pytest.mark.parametrize("head_dim", [32, 128])
def test_parallel_gated_delta_net_correctness(
    tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, gdn_kernel_backend, head_dim, run_config
):
    if gdn_kernel_backend == "fla" and not HAVE_FLA:
        pytest.skip("FLA is not installed.")
    if gdn_kernel_backend == "transformer_engine" and not HAVE_TE_GDN:
        pytest.skip("TransformerEngine GDN is not available.")
    if (
        head_dim == 32
        and gdn_kernel_backend == "transformer_engine"
        and not _have_cudnn_frontend_min_version("1.29.0")
    ):
        pytest.skip("nvidia-cudnn-frontend >= 1.29.0 is required for head dim 32.")

    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant="gated_delta_net",
        gdn_kernel_backend=gdn_kernel_backend,
        linear_cp_mode="headwise",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    if cp:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 5e-4, 5e-4

    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
        micro_batch_size=4,
        sequence_packing=sequence_packing,
    )


@pytest.mark.parametrize("cp_size", [2, 4], scope="class")
@pytest.mark.internal
@pytest.mark.skip(
    "Used to verify the correctness of the fused THD AllToAll implementation, locally validated thus no need to run on CI."
)
class TestFusedThdAllToAll:
    """Verify fused 1 AllToAll + permute matches the per-sequence, per-channel loop in GDN."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        # Attach on the class so every test method can read self.cp_*.
        request.cls.cp_size = cp_size
        request.cls.cp_group = parallel_state.get_context_parallel_group()
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _per_seq_a2a_cp2hp(local_t, cu_seqlens, cp_group, split_sections=None):
        cp_size = cp_group.size()
        unpacked = _unpack_sequence(local_t, cu_seqlens // cp_size, dim=0)
        outputs = []
        for x in unpacked:
            outputs.append(
                tensor_a2a_cp2hp(
                    x,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=cp_group,
                    split_sections=split_sections,
                    undo_attention_load_balancing=True,
                )
            )
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _per_seq_a2a_hp2cp(global_t, cu_seqlens, cp_group, split_sections=None):
        unpacked = _unpack_sequence(global_t, cu_seqlens, dim=0)
        outputs = []
        for x in unpacked:
            outputs.append(
                tensor_a2a_hp2cp(
                    x,
                    seq_dim=0,
                    head_dim=-1,
                    cp_group=cp_group,
                    split_sections=split_sections,
                    redo_attention_load_balancing=True,
                )
            )
        return torch.cat(outputs, dim=0)

    # ---- Optimized: single a2a + production permutation helper ----

    @staticmethod
    def _batched_a2a_cp2hp(local_t, cu_seqlens, cp_group, split_sections=None):
        cp_size = cp_group.size()
        t_global = int(cu_seqlens[-1].item())
        if split_sections is not None and cp_size > 1:
            head_perm = _build_head_perm_for_split_sections(split_sections, cp_size, local_t.device)
            local_t = local_t.index_select(-1, head_perm)
        naive = tensor_a2a_cp2hp(
            local_t,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            split_sections=None,  # always single fused a2a
            undo_attention_load_balancing=False,
        )
        idx, _ = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, t_global)
        return naive.index_select(0, idx)

    @staticmethod
    def _batched_a2a_hp2cp(global_t, cu_seqlens, cp_group, split_sections=None):
        cp_size = cp_group.size()
        t_global = int(cu_seqlens[-1].item())
        _, inv = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, t_global)
        permuted = global_t.index_select(0, inv)
        return tensor_a2a_hp2cp(
            permuted,
            seq_dim=0,
            head_dim=-1,
            cp_group=cp_group,
            split_sections=split_sections,
            redo_attention_load_balancing=False,
        )

    @pytest.mark.parametrize(
        "cu_seqlens",
        [
            (0, 32, 64),  # 2 equal sequences
            (0, 32, 64, 96, 128),  # 4 equal sequences (matches existing THD test)
            (0, 16, 48, 80),  # 3 unequal sequences
        ],
    )
    @pytest.mark.parametrize("split_sections", [(8, 8, 4, 16, 32, 4)])
    def test_cp2hp_batched_matches_per_seq(self, cu_seqlens, split_sections):
        cu = torch.tensor(cu_seqlens, dtype=torch.long, device=torch.cuda.current_device())
        if (torch.diff(cu) % self.cp_size != 0).any():
            pytest.skip(f"cu_seqlens {cu_seqlens} not divisible by cp_size {self.cp_size}")

        T_global = cu_seqlens[-1]
        T_local = T_global // self.cp_size
        hidden = sum(split_sections)
        torch.manual_seed(42)
        local_t = (
            torch.rand(T_local, 1, hidden, device=torch.cuda.current_device())
            .bfloat16()
            .contiguous()
        )

        out_ref = self._per_seq_a2a_cp2hp(local_t, cu, self.cp_group, split_sections=split_sections)
        out_fused = self._batched_a2a_cp2hp(
            local_t, cu, self.cp_group, split_sections=split_sections
        )

        rank = torch.distributed.get_rank()
        assert torch.equal(out_fused, out_ref), (
            f"Batched CP->HP mismatch on rank={rank} " f"(split_sections={split_sections})"
        )

    @pytest.mark.parametrize("cu_seqlens", [(0, 32, 64), (0, 32, 64, 96, 128), (0, 16, 48, 80)])
    def test_hp2cp_batched_matches_per_seq(self, cu_seqlens):
        cu = torch.tensor(cu_seqlens, dtype=torch.long, device=torch.cuda.current_device())
        if ((cu[1:] - cu[:-1]) % self.cp_size != 0).any():
            pytest.skip(f"cu_seqlens {cu_seqlens} not divisible by cp_size {self.cp_size}")

        T_global = cu_seqlens[-1]
        hidden = 32
        # Hidden must be divisible by cp_size for the HP-sharded input layout.
        assert hidden % self.cp_size == 0
        h_local = hidden // self.cp_size
        torch.manual_seed(42)
        global_t = (
            torch.rand(T_global, 1, h_local, device=torch.cuda.current_device())
            .bfloat16()
            .contiguous()
        )

        out_ref = self._per_seq_a2a_hp2cp(global_t, cu, self.cp_group)
        out_fused = self._batched_a2a_hp2cp(global_t, cu, self.cp_group)

        rank = torch.distributed.get_rank()
        assert torch.equal(out_fused, out_ref), f"Batched HP->CP mismatch on rank={rank}"

    @pytest.mark.parametrize("cu_seqlens", [(0, 32, 64, 96, 128)])
    def test_cp2hp_hp2cp_round_trip(self, cu_seqlens):
        """cp2hp followed by hp2cp on the batched path should be the identity."""
        cu = torch.tensor(cu_seqlens, dtype=torch.long, device=torch.cuda.current_device())
        if ((cu[1:] - cu[:-1]) % self.cp_size != 0).any():
            pytest.skip(f"cu_seqlens {cu_seqlens} not divisible by cp_size {self.cp_size}")

        T_global = cu_seqlens[-1]
        T_local = T_global // self.cp_size
        hidden = 32
        torch.manual_seed(7)
        local_t = (
            torch.rand(T_local, 1, hidden, device=torch.cuda.current_device())
            .bfloat16()
            .contiguous()
        )

        mid = self._batched_a2a_cp2hp(local_t, cu, self.cp_group)
        back = self._batched_a2a_hp2cp(mid, cu, self.cp_group)

        assert torch.equal(back, local_t), "Batched cp2hp -> hp2cp not identity"
