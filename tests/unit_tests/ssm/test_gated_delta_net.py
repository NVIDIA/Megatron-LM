# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    HAVE_FLA,
    HAVE_FLA_GDN2,
    GatedDeltaNet,
    GatedDeltaNet2,
    chunk_gdn2,
    torch_chunk_gated_delta_rule,
    torch_chunk_gdn2,
)
from megatron.core.ssm.gated_delta_net.common import (
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness
from tests.unit_tests.transformer.test_multi_latent_attention import (
    make_test_packed_seq_params,
    make_test_packed_seq_params_with_padding,
)

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable
# NVLS doesn't support one single GPU to be shared by multiple ranks, so disable this in test
os.environ.update({"NCCL_NVLS_ENABLE": "0"})


def _unpack_sequence(x: torch.Tensor, cu_seqlens: torch.Tensor, dim=1) -> list[torch.Tensor]:
    unpacked_x = []
    cu_seqlens_list = cu_seqlens.tolist()
    num_seqs = len(cu_seqlens_list) - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens_list[i]
        idx_end = cu_seqlens_list[i + 1]
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    HAVE_FUSED_PRE_GDR = False
else:
    HAVE_FUSED_PRE_GDR = callable(causal_conv1d_bwd_function)


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
    with pytest.raises(ValueError, match="experimental_attention_variant='gated_delta_net'"):
        _make_gdn_config(
            experimental_attention_variant=None,
            linear_attention_freq=None,
            gdn_pre_gated_delta_rule_fusion=True,
        )


@pytest.mark.parametrize("use_gdn2", [False, True], ids=["gdn", "gdn2"])
@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, sp, cp_size, use_gdn2):
        if use_gdn2 and not HAVE_FLA_GDN2:
            pytest.skip("FLA with GDN2 support is not installed.")

        # Initialize parallel and random seed
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.sp_size = tp_size if sp else 1
        self.use_gdn2 = use_gdn2

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        # Initialize model, with the same config as Qwen Next except `num_layers`
        self.transformer_config = TransformerConfig(
            hidden_size=2048,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=16,
            num_query_groups=2,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sp,
            context_parallel_size=cp_size,
            experimental_attention_variant="gdn2" if use_gdn2 else "gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        gdn_spec = get_experimental_attention_variant_module_spec(config=self.transformer_config)

        self.gdn = gdn_spec.module(
            self.transformer_config,
            submodules=gdn_spec.submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        )
        self.gdn = self.gdn.cuda().bfloat16()

    def teardown_method(self):
        Utils.destroy_model_parallel()

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

    def test_deterministic_mode(self):
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        det_config = copy.deepcopy(self.transformer_config)
        det_config.deterministic_mode = True

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

        # deterministic_mode must select the variant's torch-native kernel, not FLA.
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
                qkv, gate, A_log_mock, dt_bias_mock, batch, seq_len, *gate_feats
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
            assert beta.shape == (batch, seq_len, num_v_heads_local)
            assert (g <= 0).all()
            assert (beta >= 0).all() and (beta <= 1).all()

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
                qkvzba,
                cp_size_headwise=cp_size_headwise,
                cp_group_headwise=gdn.pg_collection.cp,
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
            atol, rtol = 3e-4, 3e-4
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
                cp_size=gdn.cp_size,
                cp_group=gdn.pg_collection.cp,
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
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_parallel_gated_delta_net_correctness(tmp_path_dist_ckpt, sequence_packing, tp, sp, cp):
    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
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


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [(4, True, 1), (1, False, 2), (2, True, 2)],  # TP w/ SP  # CP  # TP w/ SP + CP
)
@pytest.mark.skipif(not HAVE_FLA_GDN2, reason="FLA with GDN2 support is not installed.")
def test_parallel_gated_delta_net2_correctness(tmp_path_dist_ckpt, sequence_packing, tp, sp, cp):
    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant="gdn2",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    atol = rtol = 3e-2 if cp > 1 else 2e-2
    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=42,
        sequence_length=512,
        micro_batch_size=2,
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
