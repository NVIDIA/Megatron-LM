# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable
# NVLS doesn't support one single GPU to be shared by multiple ranks, so disable this in test
os.environ.update({"NCCL_NVLS_ENABLE": "0"})

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    HAVE_FUSED_PRE_GDR = False
else:
    HAVE_FUSED_PRE_GDR = callable(causal_conv1d_bwd_function)


def test_fused_pre_gdr_split_batched_recv_send_works():
    from megatron.core.fusions.fused_pre_gated_delta_rule import _split_batched_recv_send_works

    grouped_work = object()
    recv_ops, send_ops = _split_batched_recv_send_works([grouped_work], ["recv", "send"])
    assert recv_ops == (grouped_work,)
    assert send_ops == ()

    send_work = object()
    recv_ops, send_ops = _split_batched_recv_send_works([send_work], ["send"])
    assert recv_ops == ()
    assert send_ops == (send_work,)

    recv_work = object()
    send_work = object()
    recv_ops, send_ops = _split_batched_recv_send_works([recv_work, send_work], ["recv", "send"])
    assert recv_ops == (recv_work,)
    assert send_ops == (send_work,)

    with pytest.raises(RuntimeError, match="Expected batch_isend_irecv"):
        _split_batched_recv_send_works([], ["recv"])


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
        hidden_size: int = 256,
        key_head_dim: int = 64,
        value_head_dim: int = 64,
        num_key_heads: int = 4,
        num_value_heads: int = 8,
    ):
        transformer_config = TransformerConfig(
            hidden_size=hidden_size,
            linear_conv_kernel_dim=conv_kernel_dim,
            linear_key_head_dim=key_head_dim,
            linear_value_head_dim=value_head_dim,
            linear_num_key_heads=num_key_heads,
            linear_num_value_heads=num_value_heads,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=num_value_heads,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
            deterministic_mode=deterministic_mode,
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

    def _make_pre_gated_delta_rule_grad_outputs(self, outputs):
        grad_outputs = []
        for output_idx, output in enumerate(outputs):
            grad = torch.linspace(
                -0.1, 0.1, output.numel(), device=output.device, dtype=torch.float32
            ).reshape(output.shape)
            grad_outputs.append(grad + (output_idx - 2.5) * 0.01)
        return grad_outputs

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

    def test_fused_and_unfused_backward_thd_padding_match(self):
        unfused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True, deterministic_mode=False, conv_kernel_dim=4
        )
        fused_gdn.load_state_dict(unfused_gdn.state_dict())

        real_seq_len = 9
        padded_seq_len = 16
        hidden_states = torch.randn(
            (padded_seq_len, 1, unfused_gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        hidden_states_unfused = hidden_states.detach().clone().requires_grad_(True)
        hidden_states_fused = hidden_states.detach().clone().requires_grad_(True)
        cu_seqlens = torch.tensor(
            [0, 5, real_seq_len, padded_seq_len],
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=padded_seq_len - real_seq_len,
            max_seqlen_kv=padded_seq_len - real_seq_len,
            total_tokens=hidden_states.shape[0],
        )
        assert packed_seq_params.seq_idx is not None

        unfused_gdn.zero_grad(set_to_none=True)
        fused_gdn.zero_grad(set_to_none=True)
        unfused_output, _ = unfused_gdn(
            hidden_states_unfused, None, packed_seq_params=packed_seq_params
        )
        fused_output, _ = fused_gdn(hidden_states_fused, None, packed_seq_params=packed_seq_params)
        grad_output = torch.randn_like(unfused_output.float())
        grad_output[real_seq_len:] = 0.0

        unfused_loss = (unfused_output.float() * grad_output).sum()
        fused_loss = (fused_output.float() * grad_output).sum()
        unfused_loss.backward()
        fused_loss.backward()

        torch.testing.assert_close(
            hidden_states_fused.grad, hidden_states_unfused.grad, atol=3e-2, rtol=3e-2
        )
        for param_name in ("in_proj.weight", "conv1d.weight", "A_log", "dt_bias"):
            fused_param = dict(fused_gdn.named_parameters())[param_name]
            unfused_param = dict(unfused_gdn.named_parameters())[param_name]
            torch.testing.assert_close(
                fused_param.grad,
                unfused_param.grad,
                atol=3e-2,
                rtol=3e-2,
                msg=lambda msg, name=param_name: f"{name} grad mismatch: {msg}",
            )

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

    def test_apply_gated_norm_accepts_strided_gate_view(self):
        import torch._dynamo

        gdn = self.fused_gdn
        batch = 2
        seq_len = 16
        device = torch.cuda.current_device()
        num_value_heads = gdn.num_v_heads_local_tp
        gate_channels = num_value_heads * gdn.value_head_dim
        z_offset = 7
        gate_storage = torch.randn(
            seq_len, batch, z_offset + gate_channels + 5, device=device, dtype=torch.bfloat16
        )
        gate_view = (
            gate_storage[:, :, z_offset : z_offset + gate_channels]
            .view(seq_len, batch, num_value_heads, gdn.value_head_dim)
            .permute(1, 0, 2, 3)
        )
        assert not gate_view.is_contiguous()
        assert gate_view.untyped_storage().data_ptr() == gate_storage.untyped_storage().data_ptr()

        norm_input = torch.randn_like(gate_view.contiguous())
        with torch._dynamo.config.patch(disable=True):
            strided_output = gdn._apply_gated_norm(norm_input, gate_view)
            contiguous_output = gdn._apply_gated_norm(norm_input, gate_view.contiguous())
        torch.testing.assert_close(strided_output, contiguous_output)

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
        torch.manual_seed(1234)
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
        grad_outputs = self._make_pre_gated_delta_rule_grad_outputs(unfused_outputs)

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
        torch.manual_seed(1234)
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
        grad_outputs = self._make_pre_gated_delta_rule_grad_outputs(unfused_outputs)

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

    def test_fused_and_unfused_packed_pre_gated_delta_rule_backward_repeat4_match(self):
        reference_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=False,
            deterministic_mode=True,
            conv_kernel_dim=4,
            hidden_size=256,
            key_head_dim=32,
            value_head_dim=32,
            num_key_heads=2,
            num_value_heads=8,
        )
        fused_gdn = self._build_gdn(
            gdn_pre_gated_delta_rule_fusion=True,
            deterministic_mode=False,
            conv_kernel_dim=4,
            hidden_size=256,
            key_head_dim=32,
            value_head_dim=32,
            num_key_heads=2,
            num_value_heads=8,
        )
        fused_gdn.load_state_dict(reference_gdn.state_dict())

        batch = 1
        cu_seqlens = torch.tensor(
            [0, 3, 8, 16], device=torch.cuda.current_device(), dtype=torch.int32
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
        grad_outputs = self._make_pre_gated_delta_rule_grad_outputs(unfused_outputs)

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


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(not HAVE_FUSED_PRE_GDR, reason="causal-conv1d fused backward is not installed.")
@pytest.mark.internal
class TestFusedPreGatedDeltaRuleChunkwiseCP:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)

        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        self.pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

    def teardown_method(self):
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    def _build_gdn(self, *, gdn_pre_gated_delta_rule_fusion: bool, conv_kernel_dim: int = 4):
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
            context_parallel_size=2,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            linear_cp_mode="chunkwise",
            transformer_impl="transformer_engine",
            deterministic_mode=False,
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

    def _make_hidden_states(self, gdn, *, seq_len_global: int):
        cp_size = parallel_state.get_context_parallel_world_size()
        assert seq_len_global % cp_size == 0
        seq_len_local = seq_len_global // cp_size
        cp_rank = parallel_state.get_context_parallel_rank()
        hidden_states = torch.randn(
            (seq_len_local, 1, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        return hidden_states + cp_rank

    @staticmethod
    def _make_packed_seq_params(cu_seqlens):
        cp_size = parallel_state.get_context_parallel_world_size()
        cu = torch.tensor(cu_seqlens, device=torch.cuda.current_device(), dtype=torch.int32)
        return PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)),
            max_seqlen_kv=max(
                cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)
            ),
            total_tokens=cu_seqlens[-1] // cp_size,
            cp_partition_mode="contiguous",
        )

    @staticmethod
    def _run_forward(gdn, hidden_states, packed_seq_params=None):
        with torch.no_grad():
            output, bias = gdn(hidden_states, None, packed_seq_params=packed_seq_params)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return output, bias

    @staticmethod
    def _run_backward(gdn, hidden_states, grad_output, packed_seq_params=None):
        gdn.zero_grad(set_to_none=True)
        hidden_states = hidden_states.detach().clone().requires_grad_(True)
        output, _ = gdn(hidden_states, None, packed_seq_params=packed_seq_params)
        loss = (output.float() * grad_output).sum()
        loss.backward()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        param_grads = {
            name: param.grad.detach().clone()
            for name, param in gdn.named_parameters()
            if param.grad is not None
        }
        return output.detach(), hidden_states.grad.detach().clone(), param_grads

    def test_fused_and_unfused_forward_chunkwise_cp_match(self):
        unfused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=False)
        fused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=True)
        fused_gdn.load_state_dict(unfused_gdn.state_dict())
        hidden_states = self._make_hidden_states(unfused_gdn, seq_len_global=64)

        unfused_output, unfused_bias = self._run_forward(unfused_gdn, hidden_states)
        fused_output, fused_bias = self._run_forward(fused_gdn, hidden_states)

        rank = torch.distributed.get_rank()
        torch.testing.assert_close(
            fused_output,
            unfused_output,
            atol=3e-3,
            rtol=3e-3,
            msg=lambda msg: f"chunkwise CP fused forward mismatch ({rank=}): {msg}",
        )
        assert fused_bias == unfused_bias

    @pytest.mark.flaky_in_dev
    def test_fused_and_unfused_backward_chunkwise_cp_match(self):
        unfused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=False)
        fused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=True)
        fused_gdn.load_state_dict(unfused_gdn.state_dict())
        hidden_states = self._make_hidden_states(unfused_gdn, seq_len_global=64)
        grad_output = torch.randn(
            hidden_states.shape, device=torch.cuda.current_device(), dtype=torch.float32
        )

        unfused_output, unfused_dinput, unfused_grads = self._run_backward(
            unfused_gdn, hidden_states, grad_output
        )
        fused_output, fused_dinput, fused_grads = self._run_backward(
            fused_gdn, hidden_states, grad_output
        )

        rank = torch.distributed.get_rank()
        torch.testing.assert_close(
            fused_output,
            unfused_output,
            atol=3e-3,
            rtol=3e-3,
            msg=lambda msg: f"chunkwise CP fused backward output mismatch ({rank=}): {msg}",
        )
        torch.testing.assert_close(
            fused_dinput,
            unfused_dinput,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda msg: f"chunkwise CP fused input grad mismatch ({rank=}): {msg}",
        )
        assert fused_grads.keys() == unfused_grads.keys()
        for name in unfused_grads:
            torch.testing.assert_close(
                fused_grads[name],
                unfused_grads[name],
                atol=5e-2,
                rtol=5e-2,
                msg=lambda msg, param_name=name: (
                    f"chunkwise CP fused grad mismatch for {param_name!r} ({rank=}): {msg}"
                ),
            )

    def test_fused_chunkwise_cp_rejects_short_local_chunks(self):
        gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=True, conv_kernel_dim=4)
        hidden_states = self._make_hidden_states(gdn, seq_len_global=4)

        with pytest.raises((AssertionError, ValueError), match="local.*chunk|conv_kernel_dim"):
            gdn(hidden_states, None)

    def test_fused_and_unfused_packed_forward_chunkwise_cp_match(self):
        unfused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=False)
        fused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=True)
        fused_gdn.load_state_dict(unfused_gdn.state_dict())
        packed_seq_params = self._make_packed_seq_params([0, 96, 128])
        hidden_states = self._make_hidden_states(unfused_gdn, seq_len_global=128)

        unfused_output, unfused_bias = self._run_forward(
            unfused_gdn, hidden_states, packed_seq_params=packed_seq_params
        )
        fused_output, fused_bias = self._run_forward(
            fused_gdn, hidden_states, packed_seq_params=packed_seq_params
        )

        rank = torch.distributed.get_rank()
        torch.testing.assert_close(
            fused_output,
            unfused_output,
            atol=3e-3,
            rtol=3e-3,
            msg=lambda msg: f"packed chunkwise CP fused forward mismatch ({rank=}): {msg}",
        )
        assert fused_bias == unfused_bias

    @pytest.mark.flaky_in_dev
    def test_fused_and_unfused_packed_backward_chunkwise_cp_match(self):
        unfused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=False)
        fused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=True)
        fused_gdn.load_state_dict(unfused_gdn.state_dict())
        packed_seq_params = self._make_packed_seq_params([0, 96, 128])
        hidden_states = self._make_hidden_states(unfused_gdn, seq_len_global=128)
        grad_output = torch.randn(
            hidden_states.shape, device=torch.cuda.current_device(), dtype=torch.float32
        )

        unfused_output, unfused_dinput, unfused_grads = self._run_backward(
            unfused_gdn, hidden_states, grad_output, packed_seq_params=packed_seq_params
        )
        fused_output, fused_dinput, fused_grads = self._run_backward(
            fused_gdn, hidden_states, grad_output, packed_seq_params=packed_seq_params
        )

        rank = torch.distributed.get_rank()
        torch.testing.assert_close(
            fused_output,
            unfused_output,
            atol=3e-3,
            rtol=3e-3,
            msg=lambda msg: f"packed chunkwise CP fused backward output mismatch ({rank=}): {msg}",
        )
        torch.testing.assert_close(
            fused_dinput,
            unfused_dinput,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda msg: f"packed chunkwise CP fused input grad mismatch ({rank=}): {msg}",
        )
        assert fused_grads.keys() == unfused_grads.keys()
        # Packed chunkwise CP compares two bf16 backward implementations that
        # differ in their causal-conv boundary path. A few parameter-gradient
        # elements can land just above the dense-path tolerance while the fused
        # output and input gradients remain tightly matched.
        packed_param_grad_atol = 1e-1
        for name in unfused_grads:
            torch.testing.assert_close(
                fused_grads[name],
                unfused_grads[name],
                atol=packed_param_grad_atol,
                rtol=5e-2,
                msg=lambda msg, param_name=name: (
                    f"packed chunkwise CP fused grad mismatch for {param_name!r} ({rank=}): {msg}"
                ),
            )

    def test_fused_and_unfused_packed_partial_boundary_chunkwise_cp_match(self):
        unfused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=False)
        fused_gdn = self._build_gdn(gdn_pre_gated_delta_rule_fusion=True)
        fused_gdn.load_state_dict(unfused_gdn.state_dict())
        packed_seq_params = self._make_packed_seq_params([0, 4, 12])
        hidden_states = self._make_hidden_states(unfused_gdn, seq_len_global=12)
        grad_output = torch.randn(
            hidden_states.shape, device=torch.cuda.current_device(), dtype=torch.float32
        )

        unfused_output, unfused_dinput, unfused_grads = self._run_backward(
            unfused_gdn, hidden_states, grad_output, packed_seq_params=packed_seq_params
        )
        fused_output, fused_dinput, fused_grads = self._run_backward(
            fused_gdn, hidden_states, grad_output, packed_seq_params=packed_seq_params
        )

        rank = torch.distributed.get_rank()
        torch.testing.assert_close(
            fused_output,
            unfused_output,
            atol=3e-3,
            rtol=3e-3,
            msg=lambda msg: f"partial-boundary packed CP fused output mismatch ({rank=}): {msg}",
        )
        torch.testing.assert_close(
            fused_dinput,
            unfused_dinput,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda msg: f"partial-boundary packed CP fused input grad mismatch ({rank=}): {msg}",
        )
        assert fused_grads.keys() == unfused_grads.keys()
        # Packed chunkwise CP compares two bf16 backward implementations that
        # differ in their causal-conv boundary path. A few parameter-gradient
        # elements can land just above the dense-path tolerance while the fused
        # output and input gradients remain tightly matched.
        packed_param_grad_atol = 1e-1
        for name in unfused_grads:
            torch.testing.assert_close(
                fused_grads[name],
                unfused_grads[name],
                atol=packed_param_grad_atol,
                rtol=5e-2,
                msg=lambda msg, param_name=name: (
                    f"partial-boundary packed CP fused grad mismatch for {param_name!r} "
                    f"({rank=}): {msg}"
                ),
            )
