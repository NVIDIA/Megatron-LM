# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared test bodies for independently scheduled GDN correctness buckets."""

import copy
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNet2,
    chunk_gdn2,
    torch_chunk_gated_delta_rule,
    torch_chunk_gdn2,
)
from megatron.core.ssm.gated_delta_net.common import chunk_gated_delta_rule
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.gated_delta_net_test_utils import GatedDeltaNetTestBase
from tests.unit_tests.transformer.test_multi_latent_attention import (
    make_test_packed_seq_params,
    make_test_packed_seq_params_with_padding,
)

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    HAVE_FUSED_PRE_GDR = False
else:
    HAVE_FUSED_PRE_GDR = callable(causal_conv1d_bwd_function)


class GatedDeltaNetModuleTestCases(GatedDeltaNetTestBase):
    """Shared module assertions collected only through variant-specific wrappers."""

    def test_gpu_forward(self):
        gdn = self.gdn

        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        attention_mask = None

        output, bias = gdn(hidden_states, attention_mask)

        assert output.dim() == 3, f"Output too many dimensions ({output.shape=})"
        assert output.shape[0] == seq_length // self.sp_size // self.cp_size, (
            f"Output shape {output.shape[0]=} mismatch with "
            f" {seq_length=} // {self.sp_size=} // {self.cp_size=}."
        )
        assert (
            output.shape[1] == micro_batch_size
        ), f"Output shape {output.shape[1]=} mismatch with {micro_batch_size=}"
        assert (
            output.shape[2] == gdn.config.hidden_size
        ), f"Output shape {output.shape[2]=} mismatch with {gdn.config.hidden_size=}"
        assert (
            output.dtype == hidden_states.dtype
        ), f"Output dtype {output.dtype=} mismatch with {hidden_states.dtype=}"

    def test_gpu_forward_thd_correctness(self):
        if self.sp_size > 1:
            pytest.skip("Sequence parallel is not supported for this test case.")

        if self.use_gdn2:
            # FLA uses different kernels for SBHD and THD:
            # https://github.com/fla-org/flash-linear-attention/blob/ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0/fla/ops/gdn2/chunk_intra.py#L40-L53
            # so we relax the error bound here
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 3e-4, 3e-4

        # Input shape
        sequence_length = 32
        micro_batch_size = 4
        cu_seqlens = [0, 32, 64, 96, 128]
        # sbhd input shape: [sequence length, batch size, hidden size]
        sub_sequence_length = sequence_length // self.cp_size
        hidden_states_sbhd = torch.rand(
            (sub_sequence_length, micro_batch_size, self.gdn.config.hidden_size)
        )
        attention_mask_sbhd = None
        hidden_states_sbhd = hidden_states_sbhd.cuda().bfloat16()
        # thd input shape: [sequence length * batch size, 1, hidden size]
        hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
        hidden_states_thd = hidden_states_thd.view(-1, 1, self.gdn.config.hidden_size)
        attention_mask_thd = None
        packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

        # THD format
        output_thd, _ = self.gdn(
            hidden_states_thd, attention_mask_thd, packed_seq_params=packed_seq_params
        )
        # SBHD format
        output_sbhd, _ = self.gdn(hidden_states_sbhd, attention_mask_sbhd)
        output_sbhd_T = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)

        rank = torch.distributed.get_rank()
        assert output_thd.shape[0] == sub_sequence_length * micro_batch_size
        assert output_thd.shape[1] == 1
        assert output_thd.shape[2] == self.gdn.config.hidden_size
        torch.testing.assert_close(
            output_sbhd_T,
            output_thd,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Output mismatch ({rank=}): {msg}",
        )

    def test_gpu_forward_thd_padding_correctness(self):
        if self.sp_size > 1:
            pytest.skip("Sequence parallel is not supported for this test case.")

        if self.use_gdn2:
            # See test_gpu_forward_thd_correctness: varlen vs batched kernel paths only
            # match up to bf16 ULP-level differences for GDN2.
            atol, rtol = 1e-2, 1e-2
        else:
            # SBHD and padded THD use distinct FLA bf16 kernel paths, which can
            # differ by one ULP. Match test_gated_delta_net_padding.py and #6851:
            # https://github.com/NVIDIA/Megatron-LM/pull/6851#issuecomment-5455230207
            atol, rtol = 5e-3, 5e-3
        sequence_length = 32
        micro_batch_size = 4

        # sbhd input shape: [sequence length, batch size, hidden size]
        sub_sequence_length = sequence_length // self.cp_size
        hidden_states_sbhd = torch.rand(
            (sub_sequence_length, micro_batch_size, self.gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        output_sbhd, _ = self.gdn(hidden_states_sbhd, None)

        # thd input shape: [sequence length * batch size, 1, hidden size]
        hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
        hidden_states_thd = hidden_states_thd.view(-1, 1, self.gdn.config.hidden_size)
        output_bshd = output_sbhd.transpose(0, 1).contiguous()

        rank = torch.distributed.get_rank()

        # A) padded branch: prefer *_padded when available.
        padded_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 128]
        )
        output_thd_padded, _ = self.gdn(hidden_states_thd, None, packed_seq_params=padded_params)
        output_thd2bshd = output_thd_padded.view(*output_bshd.shape)
        torch.testing.assert_close(
            output_bshd[:, :30, :],
            output_thd2bshd[:, :30, :],
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"THD padded output mismatch ({rank=}): {msg}",
        )

        # B) no-padded branch: use actual cu_seqlens when it matches total_sequence_length.
        no_padding_params = make_test_packed_seq_params(cu_seqlens=[0, 32, 64, 96, 128])
        output_thd_no_padding, _ = self.gdn(
            hidden_states_thd, None, packed_seq_params=no_padding_params
        )
        assert output_thd_no_padding.shape == output_thd_padded.shape

        # C) padded mismatch branch: if *_padded[-1] mismatches total_sequence_length, should raise.
        padded_mismatch_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 126]
        )
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=padded_mismatch_params)

        # D) actual mismatch branch without *_padded: should raise.
        actual_mismatch_params = make_test_packed_seq_params(cu_seqlens=[0, 32, 64, 96, 129])
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=actual_mismatch_params)

    def test_selective_recompute_norm_out(self):
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        def build_gdn(config):
            gdn_spec = get_experimental_attention_variant_module_spec(config=config)
            gdn = gdn_spec.module(
                config,
                submodules=gdn_spec.submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                conv_init=1.0,
                use_qk_l2norm=True,
                A_init_range=(1, 16),
                pg_collection=pg_collection,
            )
            return gdn.cuda().bfloat16()

        def run(gdn, hidden_states):
            output, _ = gdn(hidden_states, None)
            output.float().sum().backward()
            grads = {
                name: param.grad.detach()
                for name, param in gdn.named_parameters()
                if param.grad is not None
            }
            input_grad = hidden_states.grad.detach().clone()
            return output.detach(), grads, input_grad

        micro_batch_size = 2
        seq_length = 64
        base_config = copy.deepcopy(self.transformer_config)
        rec_config = copy.deepcopy(self.transformer_config)
        rec_config.recompute_granularity = "selective"
        rec_config.recompute_modules = ["gdn_norm_out"]

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        hidden_states = torch.randn(
            (
                seq_length // self.sp_size // self.cp_size,
                micro_batch_size,
                self.gdn.config.hidden_size,
            ),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # --- Baseline (no recompute) ---
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        base_gdn = build_gdn(base_config)
        assert base_gdn.recompute_norm_out is False
        base_output, base_grads, base_input_grad = run(base_gdn, hidden_states)
        hidden_states.grad = None
        assert base_gdn.norm_out_checkpoint is None
        del base_gdn
        torch.cuda.empty_cache()

        # --- Recompute ---
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        rec_gdn = build_gdn(rec_config)
        assert rec_gdn.recompute_norm_out is True
        rec_output, rec_grads, rec_input_grad = run(rec_gdn, hidden_states)
        assert rec_gdn.norm_out_checkpoint is not None

        rank = torch.distributed.get_rank()
        assert torch.equal(rec_output, base_output), f"Output not identical ({rank=})"
        assert torch.equal(rec_input_grad, base_input_grad), f"Input grad not identical ({rank=})"
        assert set(rec_grads.keys()) == set(base_grads.keys())
        for name in base_grads:
            assert torch.equal(
                rec_grads[name], base_grads[name]
            ), f"Grad not identical for {name} ({rank=})"

    def test_deterministic_mode(self, monkeypatch):
        from megatron.core.ssm.gated_delta_net import common

        def reject_autotuned_l2norm(*args, **kwargs):
            raise AssertionError("Deterministic GDN must not dispatch FLA's autotuned L2 norm")

        monkeypatch.setattr(common, "l2norm", reject_autotuned_l2norm)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        det_config = copy.deepcopy(self.transformer_config)
        det_config.deterministic_mode = True
        det_config.gdn_kernel_backend = "torch"

        gdn_spec = get_experimental_attention_variant_module_spec(config=det_config)

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        gdn = (
            gdn_spec.module(
                det_config,
                submodules=gdn_spec.submodules,
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

        # The deterministic configuration explicitly selects the variant's torch-native kernel.
        assert gdn.gdn_backend == "torch"
        if self.use_gdn2:
            assert isinstance(gdn, GatedDeltaNet2)
            assert gdn.gated_delta_rule is torch_chunk_gdn2
        else:
            assert isinstance(gdn, GatedDeltaNet)
            assert gdn.gated_delta_rule is torch_chunk_gated_delta_rule

        micro_batch_size = 2
        seq_length = 64
        torch.manual_seed(0)
        base_input = torch.randn(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        def run(module):
            hidden_states = base_input.clone().requires_grad_(True)
            output, _ = module(hidden_states, None)
            output.float().sum().backward()
            grads = {
                name: param.grad.detach().clone()
                for name, param in module.named_parameters()
                if param.grad is not None
            }
            module.zero_grad(set_to_none=True)
            return output.detach().clone(), grads, hidden_states.grad.detach().clone()

        out1, grads1, input_grad1 = run(gdn)
        out2, grads2, input_grad2 = run(gdn)

        rank = torch.distributed.get_rank()
        assert torch.equal(out1, out2), f"Output not reproducible ({rank=})"
        assert torch.equal(input_grad1, input_grad2), f"Input grad not reproducible ({rank=})"
        assert set(grads1.keys()) == set(grads2.keys())
        for name in grads1:
            assert torch.equal(
                grads1[name], grads2[name]
            ), f"Grad not reproducible for {name} ({rank=})"

    def test_module_construction(self):
        gdn = self.gdn
        if self.use_gdn2:
            assert isinstance(gdn, GatedDeltaNet2)
            assert gdn.gated_delta_rule is chunk_gdn2
            assert gdn.in_proj_dim == 4 * gdn.qk_dim + 3 * gdn.v_dim
            assert gdn.A_log.shape == (gdn.num_key_heads // self.tp_size,)
            assert gdn.dt_bias.shape == (gdn.qk_dim // self.tp_size,)
        else:
            assert isinstance(gdn, GatedDeltaNet)
            assert gdn.gdn_backend == "fla"
            assert gdn.gated_delta_rule is chunk_gated_delta_rule
            assert gdn.in_proj_dim == 2 * gdn.qk_dim + 2 * gdn.v_dim + 2 * gdn.num_value_heads
            assert gdn.A_log.shape == (gdn.num_value_heads // self.tp_size,)
            assert gdn.dt_bias.shape == (gdn.num_value_heads // self.tp_size,)

    def test_inference_state_shapes(self):
        if self.use_gdn2:
            pytest.skip("GDN2 inference is not supported.")
        assert self.gdn.mamba_state_shapes_per_request() == (
            (self.gdn.conv_dim_local_tp, self.gdn.conv_kernel_dim),
            (self.gdn.num_v_heads_local_tp, self.gdn.key_head_dim, self.gdn.value_head_dim),
        )

    def test_jit_compiled_helpers(self):
        import torch._dynamo

        gdn = self.gdn
        batch = 2
        seq_len = 16

        device = torch.cuda.current_device()
        num_v_heads_local = gdn.num_value_heads // gdn.tp_size // gdn.cp_size
        num_k_heads_local = gdn.num_key_heads // gdn.tp_size // gdn.cp_size
        qk_dim_local = gdn.qk_dim_local_tp // gdn.cp_size
        v_dim_local = gdn.v_dim_local_tp // gdn.cp_size

        qkv = torch.randn(
            batch, seq_len, 2 * qk_dim_local + v_dim_local, device=device, dtype=torch.bfloat16
        )
        gate = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            gdn.value_head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        if self.use_gdn2:
            gate_feats = (
                torch.randn(batch, seq_len, qk_dim_local, device=device, dtype=torch.bfloat16),
                torch.randn(batch, seq_len, qk_dim_local, device=device, dtype=torch.bfloat16),
                torch.randn(batch, seq_len, v_dim_local, device=device, dtype=torch.bfloat16),
            )  # f, b, w
            A_log_mock = torch.randn(num_k_heads_local, device=device, dtype=torch.bfloat16)
            dt_bias_mock = torch.randn(qk_dim_local, device=device, dtype=torch.bfloat16)
            expected_keys = {"q", "k", "v", "g", "b", "w"}
        else:
            gate_feats = (
                torch.randn(batch, seq_len, num_v_heads_local, device=device, dtype=torch.bfloat16),
                torch.randn(batch, seq_len, num_v_heads_local, device=device, dtype=torch.bfloat16),
            )  # beta, alpha
            A_log_mock = torch.randn(num_v_heads_local, device=device, dtype=torch.bfloat16)
            dt_bias_mock = torch.randn(num_v_heads_local, device=device, dtype=torch.bfloat16)
            expected_keys = {"q", "k", "v", "g", "beta"}

        # Disable dynamo so coverage.py can trace through the method bodies,
        # which are normally wrapped by @jit_fuser (torch.compile).
        with torch._dynamo.config.patch(disable=True):
            kernel_inputs = gdn._prepare_input_for_gated_delta_rule(
                qkv, gate, A_log_mock, dt_bias_mock, batch, seq_len, gdn.cp_size, *gate_feats
            )

        # The output gate (z) rides along under "gate" and is popped by forward before
        # the kernel call; everything else is passed straight through as kernel kwargs.
        gate_out = kernel_inputs.pop("gate")
        assert set(kernel_inputs) == expected_keys

        query, key, value, g = (kernel_inputs[k] for k in ("q", "k", "v", "g"))
        assert query.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert key.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert value.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        assert gate_out.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        for t in (query, key, value, gate_out, *kernel_inputs.values()):
            assert t.is_contiguous()

        if self.use_gdn2:
            # Per-channel decay and erase/write gates squashed to [0, 1]
            b, w = kernel_inputs["b"], kernel_inputs["w"]
            assert g.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
            assert b.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
            assert w.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
            assert (g <= 0).all()
            assert (b >= 0).all() and (b <= 1).all()
            assert (w >= 0).all() and (w <= 1).all()
        else:
            # Per-head decay and write strength beta
            beta = kernel_inputs["beta"]
            assert g.shape == (batch, seq_len, num_v_heads_local)
            assert g.dtype == torch.float32
            assert beta.shape == (batch, seq_len, num_v_heads_local)
            assert beta.dtype == torch.float32
            assert (g <= 0).all()
            assert (beta >= 0).all() and (beta <= 1).all()
            torch.testing.assert_close(beta, gate_feats[0].float().sigmoid())

            # The fused pre-GDR path exposes Z as a strided view into the combined
            # qkvzba projection. Verify gated norm consumes that view directly and
            # remains numerically identical to a contiguous gate tensor.
            gate_channels = num_v_heads_local * gdn.value_head_dim
            z_offset = 7
            gate_storage = torch.randn(
                seq_len,
                batch,
                z_offset + gate_channels + 5,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
            gate_view = (
                gate_storage[:, :, z_offset : z_offset + gate_channels]
                .view(seq_len, batch, num_v_heads_local, gdn.value_head_dim)
                .permute(1, 0, 2, 3)
            )
            assert not gate_view.is_contiguous()
            assert (
                gate_view.untyped_storage().data_ptr() == gate_storage.untyped_storage().data_ptr()
            )

            norm_input = torch.randn_like(gate)
            with torch._dynamo.config.patch(disable=True):
                strided_output = gdn._apply_gated_norm(norm_input, gate_view)
                contiguous_output = gdn._apply_gated_norm(norm_input, gate_view.contiguous())
            torch.testing.assert_close(strided_output, contiguous_output)

    def test_fused_pre_gated_delta_rule_headwise_cp_uses_cp_local_parameters(self):
        if not HAVE_FUSED_PRE_GDR:
            pytest.skip("causal-conv1d fused backward is not installed.")
        if self.use_gdn2:
            pytest.skip("Pre-GDR fusion is GDN1-specific.")
        if self.cp_size == 1:
            pytest.skip("Only CP>1 needs CP-local fused pre-GDR params.")

        cp_size_headwise = self.cp_size
        gdn = self.gdn
        batch = 2
        seq_len = 16
        qk_channels = gdn.qk_dim_local_tp // cp_size_headwise
        v_channels = gdn.v_dim_local_tp // cp_size_headwise
        num_key_heads = qk_channels // gdn.key_head_dim
        num_value_heads = v_channels // gdn.value_head_dim
        qkvzba_dim = 2 * qk_channels + 2 * v_channels + 2 * num_value_heads
        qkvzba = torch.randn(
            seq_len, batch, qkvzba_dim, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        captured = {}

        def fake_fused_streamed_pre_gated_delta_rule(
            qkvzba_arg,
            conv1d_weight,
            conv1d_bias,
            A_log,
            dt_bias,
            *,
            num_key_heads,
            num_value_heads,
            **kwargs,
        ):
            captured.update(
                {
                    "qkvzba": qkvzba_arg,
                    "conv1d_weight": conv1d_weight,
                    "conv1d_bias": conv1d_bias,
                    "A_log": A_log,
                    "dt_bias": dt_bias,
                    "num_key_heads": num_key_heads,
                    "num_value_heads": num_value_heads,
                    "cp_group": kwargs.get("cp_group"),
                }
            )
            return tuple(torch.empty(0, device=qkvzba_arg.device) for _ in range(6))

        with mock.patch(
            "megatron.core.fusions.fused_pre_gated_delta_rule."
            "fused_streamed_pre_gated_delta_rule",
            side_effect=fake_fused_streamed_pre_gated_delta_rule,
        ):
            gdn._fused_streamed_pre_gated_delta_rule(
                qkvzba, cp_size_headwise=cp_size_headwise, cp_group_headwise=gdn.pg_collection.cp
            )

        assert captured["qkvzba"] is qkvzba
        assert captured["conv1d_weight"].shape == (
            2 * qk_channels + v_channels,
            1,
            gdn.conv_kernel_dim,
        )
        assert captured["conv1d_bias"] is None
        assert captured["A_log"].shape == (num_value_heads,)
        assert captured["dt_bias"].shape == (num_value_heads,)
        assert captured["num_key_heads"] == num_key_heads
        assert captured["num_value_heads"] == num_value_heads
        assert captured["cp_group"] is None
