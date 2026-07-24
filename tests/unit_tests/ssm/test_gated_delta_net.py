# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from functools import partial
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_experimental_attention_variant_stage_input_cp_partition_mode,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import (
    GatedDeltaNet,
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import unwrap_model
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness
from tests.unit_tests.transformer.test_multi_latent_attention import (
    make_test_packed_seq_params,
    make_test_packed_seq_params_with_padding,
)

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

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


def test_gdn_conv_pad_alignment_rejects_chunkwise_cp():
    with pytest.raises(AssertionError, match="gdn_conv_pad_alignment is incompatible"):
        _make_gdn_config(
            context_parallel_size=2, linear_cp_mode="chunkwise", gdn_conv_pad_alignment=4096
        )


def test_gdn_chunkwise_cp_head_divisibility_ignores_cp_size():
    config = _make_gdn_config(
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        linear_cp_mode="chunkwise",
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )
    assert config.linear_cp_mode == "chunkwise"


def test_gdn_headwise_cp_head_divisibility_includes_cp_size():
    with pytest.raises(AssertionError, match="linear_head_parallel_size"):
        _make_gdn_config(
            tensor_model_parallel_size=2,
            context_parallel_size=4,
            linear_cp_mode="headwise",
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        )


@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size", "linear_cp_mode"),
    [
        # cp_size=1: the CP path is inactive, so linear_cp_mode choice is irrelevant.
        # Cover the "chunkwise" default and skip the "headwise" variants for brevity.
        (1, False, 1, None),
        (2, False, 1, None),
        (2, True, 1, None),
        # cp_size=2: exercise both CP paths.
        (1, False, 2, "headwise"),
        (2, False, 2, "headwise"),
        (2, True, 2, "headwise"),
        (1, False, 2, "chunkwise"),
        (2, False, 2, "chunkwise"),
        (2, True, 2, "chunkwise"),
    ],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, sp, cp_size, linear_cp_mode):
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
        self.linear_cp_mode = linear_cp_mode
        if self.linear_cp_mode == "headwise":
            self.cp_size_chunkwise = 1
            self.cp_size_headwise = self.cp_size
        elif self.linear_cp_mode == "chunkwise":
            self.cp_size_chunkwise = self.cp_size
            self.cp_size_headwise = 1
        elif self.cp_size == 1:
            self.cp_size_chunkwise = 1
            self.cp_size_headwise = 1
        else:
            raise ValueError(f"Invalid linear CP mode: {self.linear_cp_mode}")

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
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=[1],
            linear_cp_mode=self.linear_cp_mode,
            transformer_impl="transformer_engine",
        )
        gdn_submodules = get_experimental_attention_variant_module_spec(
            config=self.transformer_config
        ).submodules

        self.gdn = GatedDeltaNet(
            self.transformer_config,
            submodules=gdn_submodules,
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

        micro_batch_size = 1 if self.linear_cp_mode == "chunkwise" and self.cp_size > 1 else 2
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

    @pytest.mark.flaky_in_dev  # Issue #5473
    def test_selective_recompute_gdn(self):
        """Whole-module 'gdn' recompute must match the non-recompute forward and gradients.

        The same module/input is run twice (recompute off, then on); the forward output and
        all parameter / input gradients must agree within a tight tolerance (rtol/atol=1e-4).
        The recompute path is run-to-run deterministic on these kernels (empirically bitwise),
        so a tolerance well below the bf16 floor is expected to hold.
        """
        gdn = self.gdn
        gdn.train()

        micro_batch_size = 1 if self.linear_cp_mode == "chunkwise" and self.cp_size > 1 else 2
        seq_length = 64
        torch.manual_seed(1234)
        base_input = torch.randn(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        def run(recompute):
            gdn.recompute_gdn = recompute
            gdn.zero_grad(set_to_none=True)
            hidden_states = base_input.clone().detach().requires_grad_(True)
            output, _ = gdn(hidden_states, None)
            output.float().square().mean().backward()
            param_grads = {
                name: param.grad.detach().clone()
                for name, param in gdn.named_parameters()
                if param.grad is not None
            }
            return output.detach().clone(), hidden_states.grad.detach().clone(), param_grads

        try:
            out_ref, dinput_ref, pgrad_ref = run(recompute=False)
            out_rc, dinput_rc, pgrad_rc = run(recompute=True)
        finally:
            gdn.recompute_gdn = False

        torch.testing.assert_close(out_rc, out_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dinput_rc, dinput_ref, rtol=1e-4, atol=1e-4)
        assert pgrad_ref.keys() == pgrad_rc.keys(), "recompute changed the set of grad params"
        assert len(pgrad_ref) > 0, "expected at least one parameter gradient"
        for name in pgrad_ref:
            torch.testing.assert_close(
                pgrad_rc[name],
                pgrad_ref[name],
                rtol=1e-4,
                atol=1e-4,
                msg=lambda m, n=name: f"gradient mismatch for parameter '{n}': {m}",
            )

    def test_gpu_forward_rejects_sbhd_chunkwise_cp_batch_gt_one(self):
        if not (self.linear_cp_mode == "chunkwise" and self.cp_size > 1):
            pytest.skip("Only chunkwise CP with CP>1 uses the FLA CP batch guard.")

        gdn = self.gdn

        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        with pytest.raises(ValueError, match="requires micro_batch_size == 1"):
            gdn(hidden_states, None)

    def test_gpu_forward_rejects_sbhd_conv_padding(self):
        gdn = self.gdn
        gdn.config.gdn_conv_pad_alignment = 4096

        micro_batch_size = 1 if self.linear_cp_mode == "chunkwise" and self.cp_size > 1 else 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        expected_error = (
            "incompatible with GDN chunkwise CP"
            if self.linear_cp_mode == "chunkwise" and self.cp_size > 1
            else "only supported with packed sequence"
        )
        with pytest.raises(ValueError, match=expected_error):
            gdn(hidden_states, None)

    def test_jit_compiled_helpers(self):
        import torch._dynamo

        gdn = self.gdn
        batch = 2
        seq_len = 16

        num_v_heads_local = gdn.num_value_heads // gdn.tp_size // self.cp_size_headwise

        qkv_last_dim = (2 * gdn.qk_dim_local_tp + gdn.v_dim_local_tp) // self.cp_size_headwise
        qkv = torch.randn(
            batch, seq_len, qkv_last_dim, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        gate = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            gdn.value_head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        beta = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        alpha = torch.randn(
            batch,
            seq_len,
            num_v_heads_local,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        # Disable dynamo so coverage.py can trace through the method bodies,
        # which are normally wrapped by @jit_fuser (torch.compile).
        with torch._dynamo.config.patch(disable=True):
            query, key, value, gate_out, beta_out, alpha_out = (
                gdn._prepare_qkv_for_gated_delta_rule(
                    qkv, gate, beta, alpha, batch, seq_len, cp_size_headwise=self.cp_size_headwise
                )
            )

        assert query.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert key.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert value.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        assert query.is_contiguous()
        assert key.is_contiguous()
        assert value.is_contiguous()

        A_log_mock = torch.randn(
            num_v_heads_local, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        dt_bias_mock = torch.randn(
            num_v_heads_local, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )

        with torch._dynamo.config.patch(disable=True):
            g, beta_sig = gdn._compute_g_and_beta(A_log_mock, dt_bias_mock, alpha, beta)

        assert g.dtype == torch.float32
        assert g.shape == alpha.shape
        assert beta_sig.shape == beta.shape

    def test_gpu_forward_thd_correctness(self):
        if self.sp_size > 1:
            pytest.skip("Sequence parallel is not supported for this test case.")
        if self.cp_size > 1 and self.linear_cp_mode == "chunkwise":
            pytest.skip("Chunkwise CP is not supported for this test case.")

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
        if self.cp_size > 1 and self.linear_cp_mode == "chunkwise":
            pytest.skip("Chunkwise CP is not supported for this test case.")

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

        # C) explicit causal-conv padding is only applied to packed inputs and
        # should not affect the original unpadded token outputs.
        self.gdn.config.gdn_conv_pad_alignment = 48
        output_thd_conv_pad, _ = self.gdn(
            hidden_states_thd, None, packed_seq_params=no_padding_params
        )
        self.gdn.config.gdn_conv_pad_alignment = None
        assert output_thd_conv_pad.shape == output_thd_no_padding.shape
        torch.testing.assert_close(
            output_thd_conv_pad,
            output_thd_no_padding,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"THD conv-padded output mismatch ({rank=}): {msg}",
        )

        # D) padded mismatch branch: if *_padded[-1] mismatches total_sequence_length, should raise.
        padded_mismatch_params = make_test_packed_seq_params_with_padding(
            cu_seqlens=[0, 30, 60, 90, 120], cu_seqlens_padded=[0, 32, 64, 96, 126]
        )
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=padded_mismatch_params)

        # E) actual mismatch branch without *_padded: should raise.
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
    ("tp", "sp", "cp", "linear_cp_mode"),
    [
        (4, False, 1, None),  # TP w/o SP
        (4, True, 1, None),  # TP w/ SP
        (1, False, 2, "headwise"),  # Headwise CP
        (2, False, 2, "headwise"),  # TP w/o SP + Headwise CP
        (2, True, 2, "headwise"),  # TP w/ SP + Headwise CP
        (1, False, 2, "chunkwise"),  # Chunkwise CP
        (2, False, 2, "chunkwise"),  # TP w/o SP + chunkwise CP
        (2, True, 2, "chunkwise"),  # TP w/ SP + chunkwise CP
    ],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_parallel_gated_delta_net_correctness(
    tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, linear_cp_mode
):
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
        linear_cp_mode=linear_cp_mode,
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    cosine_similarity_threshold = None
    if cp > 1:
        atol, rtol = 2e-3, 1e-2
        cosine_similarity_threshold = 0.9999
    else:
        atol, rtol = 2e-4, 2e-3
        cosine_similarity_threshold = 0.99999

    is_chunkwise_cp = linear_cp_mode == "chunkwise" and cp > 1
    micro_batch_size = 1 if is_chunkwise_cp and not sequence_packing else 4

    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        cosine_similarity_threshold=cosine_similarity_threshold,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
        micro_batch_size=micro_batch_size,
        sequence_packing=sequence_packing,
        cp_partition_mode="contiguous" if is_chunkwise_cp else "zigzag",
        cp_stage_entry_partition_mode="contiguous" if is_chunkwise_cp else "zigzag",
        compare_param_grads=is_chunkwise_cp and tp == 1 and not sequence_packing,
    )


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
@pytest.mark.parametrize("cp", [2, 4])
def test_mixed_gdn_sdpa_gpt_model_cp_boundary_forward_backward_correctness(
    tmp_path_dist_ckpt, cp
):
    if not torch.cuda.is_available() or Utils.world_size < cp:
        pytest.skip(f"Mixed GDN/SDPA CP parity needs at least {cp} CUDA ranks.")

    sequence_length = 64
    micro_batch_size = 1
    vocab_size = 128
    seed = 123

    def make_config(context_parallel_size):
        return TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=4,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            activation_func=F.silu,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=4,
            linear_cp_mode="chunkwise",
            transformer_impl="transformer_engine",
            context_parallel_size=context_parallel_size,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

    def initialize_gpt_model(
        config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
    ):
        transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
            config=config, vp_stage=vp_stage, pp_rank=0
        )
        cp_stage_entry_partition_mode = (
            get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            position_embedding_type="rope",
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            cp_stage_entry_partition_mode=cp_stage_entry_partition_mode,
        )

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(micro_batch_size, sequence_length),
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
    )
    position_ids = torch.arange(
        sequence_length, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)
    labels = (input_ids + 1) % vocab_size

    def _get_param_grad(param):
        grad = param.grad
        if grad is None:
            grad = getattr(param, "main_grad", None)
        if grad is None:
            param.grad = torch.zeros_like(param)
            grad = param.grad
        return grad

    def _scale_grads(model, scale):
        for param in model.parameters():
            if param.requires_grad:
                _get_param_grad(param).data.mul_(scale)

    def _all_reduce_grads(model, group):
        for param in model.parameters():
            if param.requires_grad:
                torch.distributed.all_reduce(_get_param_grad(param), group=group)

    def _collect_grads(model):
        return {
            name: _get_param_grad(param).detach().float().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _zero_grads(model):
        for param in model.parameters():
            param.grad = None
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                main_grad.zero_()

    with TempNamedDir(tmp_path_dist_ckpt / 'test_mixed_gdn_sdpa_gpt_cp', sync=True) as ckpt_dir:
        mock_args = parse_args(ignore_unknown_args=True)
        set_args(mock_args)

        baseline_config = make_config(context_parallel_size=1)
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = 1
        baseline_model = unwrap_model(get_model(initialize_gpt_model, config=baseline_config))
        baseline_model[0].eval()

        init_checkpointing_mock_args(mock_args, ckpt_dir, False)
        mock_args.no_save_optim = True
        mock_args.no_save_rng = True
        mock_args.no_load_optim = True
        mock_args.no_load_rng = True
        save_checkpoint(10, baseline_model, None, None, 0)

        _zero_grads(baseline_model[0])
        baseline_loss = baseline_model[0](
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
        )
        baseline_loss.float().sum().backward()
        _scale_grads(baseline_model[0], 1.0 / baseline_loss.numel())
        baseline_grads = _collect_grads(baseline_model[0])
        baseline_loss = baseline_loss.detach()

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp
        )
        model_parallel_cuda_manual_seed(seed)
        parallel_config = make_config(context_parallel_size=cp)
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = cp
        parallel_model = unwrap_model(get_model(initialize_gpt_model, config=parallel_config))
        parallel_model[0].eval()
        with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
            with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
                load_checkpoint(parallel_model, None, None)

        cp_group = parallel_state.get_context_parallel_group()
        input_partition_mode = parallel_model[0].get_input_cp_partition_mode()
        assert input_partition_mode == "contiguous"

        local_input_ids = get_tensor_on_this_cp_rank(
            input_ids, 1, cp_group, cp_partition_mode=input_partition_mode
        )
        local_position_ids = get_tensor_on_this_cp_rank(
            position_ids, 1, cp_group, cp_partition_mode=input_partition_mode
        )
        local_labels = get_tensor_on_this_cp_rank(
            labels, 1, cp_group, cp_partition_mode=input_partition_mode
        )
        _zero_grads(parallel_model[0])
        parallel_loss = parallel_model[0](
            input_ids=local_input_ids,
            position_ids=local_position_ids,
            attention_mask=None,
            labels=local_labels,
        )

        expected_loss = get_tensor_on_this_cp_rank(
            baseline_loss, 1, cp_group, cp_partition_mode=input_partition_mode
        )
        torch.testing.assert_close(
            parallel_loss.float(), expected_loss.float(), atol=2e-3, rtol=2e-3
        )

        parallel_loss.float().sum().backward()
        _all_reduce_grads(parallel_model[0], cp_group)
        _scale_grads(parallel_model[0], 1.0 / baseline_loss.numel())
        parallel_grads = _collect_grads(parallel_model[0])

        assert baseline_grads.keys() == parallel_grads.keys()
        for name, baseline_grad in baseline_grads.items():
            torch.testing.assert_close(
                parallel_grads[name],
                baseline_grad,
                atol=2e-3,
                rtol=2e-3,
                msg=lambda msg, param_name=name: f"gradient mismatch for {param_name}: {msg}",
            )

        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
@pytest.mark.parametrize("cp", [2, 4])
def test_mixed_gdn_sdpa_gpt_model_cp_boundary_accumulated_backward_correctness(
    tmp_path_dist_ckpt, cp
):
    if not torch.cuda.is_available() or Utils.world_size < cp:
        pytest.skip(f"Mixed GDN/SDPA CP parity needs at least {cp} CUDA ranks.")

    sequence_length = 8192
    micro_batch_size = 1
    num_microbatches = 8
    vocab_size = 32768
    seed = 123

    def make_config(context_parallel_size):
        return TransformerConfig(
            hidden_size=4096,
            ffn_hidden_size=10240,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=64,
            num_layers=4,
            normalization="RMSNorm",
            layernorm_epsilon=1e-6,
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=32,
            kv_channels=256,
            num_query_groups=2,
            qk_layernorm=True,
            attention_output_gate=True,
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=4,
            linear_cp_mode="chunkwise",
            transformer_impl="transformer_engine",
            context_parallel_size=context_parallel_size,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            calculate_per_token_loss=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=cp,
            expert_tensor_parallel_size=1,
            num_moe_experts=64,
            moe_ffn_hidden_size=1024,
            moe_shared_expert_intermediate_size=1024,
            moe_shared_expert_gate=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=10,
            moe_grouped_gemm=True,
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="alltoall",
            moe_router_dtype="fp32",
        )

    def initialize_gpt_model(
        config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
    ):
        transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
            config=config, vp_stage=vp_stage, pp_rank=0
        )
        cp_stage_entry_partition_mode = (
            get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            position_embedding_type="rope",
            rotary_percent=0.25,
            rotary_base=10000000,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            cp_stage_entry_partition_mode=cp_stage_entry_partition_mode,
        )

    def _get_param_grad(param):
        grad = param.grad
        if grad is None:
            grad = getattr(param, "main_grad", None)
        if grad is None:
            param.grad = torch.zeros_like(param)
            grad = param.grad
        return grad

    def _scale_grads(model, scale):
        for param in model.parameters():
            if param.requires_grad:
                _get_param_grad(param).data.mul_(scale)

    def _is_routed_expert_param(name):
        return ".mlp.experts." in name

    def _param_category(name):
        if ".mlp.experts." in name:
            return "routed_expert"
        if ".mlp.shared_experts." in name:
            return "shared_expert"
        if ".mlp.router." in name:
            return "router"
        if ".self_attention." in name:
            return "self_attention"
        if ".input_layernorm." in name or ".pre_mlp_layernorm." in name:
            return "layernorm"
        return "other"

    def _all_reduce_grads(model, group):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if _is_routed_expert_param(name):
                    continue
                torch.distributed.all_reduce(_get_param_grad(param), group=group)

    def _collect_grads(model):
        return {
            name: _get_param_grad(param).detach().float().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _grad_norm_from_grads(grads):
        total = torch.zeros([], device=input_ids_list[0].device, dtype=torch.float64)
        for grad in grads.values():
            total += grad.double().pow(2).sum()
        return total.sqrt()

    def _print_grad_diagnostics(baseline_grads, parallel_grads, group):
        worst_name = None
        worst_abs = -1.0
        categories = {}
        for name, baseline_grad in baseline_grads.items():
            diff = (parallel_grads[name] - baseline_grad).float().abs()
            category = _param_category(name)
            stats = categories.setdefault(
                category,
                {
                    "max_abs": 0.0,
                    "sum_abs": 0.0,
                    "numel": 0,
                    "baseline_norm_sq": 0.0,
                    "parallel_norm_sq": 0.0,
                    "worst_name": None,
                },
            )
            max_abs = diff.max().item()
            stats["max_abs"] = max(stats["max_abs"], max_abs)
            stats["sum_abs"] += diff.sum().item()
            stats["numel"] += diff.numel()
            stats["baseline_norm_sq"] += baseline_grad.float().pow(2).sum().item()
            stats["parallel_norm_sq"] += parallel_grads[name].float().pow(2).sum().item()
            if max_abs >= stats["max_abs"]:
                stats["worst_name"] = name
            if max_abs > worst_abs:
                worst_abs = max_abs
                worst_name = name

        cp_rank = torch.distributed.get_rank(group=group)
        for category, stats in sorted(categories.items()):
            mean_abs = stats["sum_abs"] / max(stats["numel"], 1)
            print(
                "CP_GRAD_CATEGORY "
                f"cp_rank={cp_rank} "
                f"category={category} "
                f"max_abs={stats['max_abs']:.8f} "
                f"mean_abs={mean_abs:.8f} "
                f"baseline_norm={stats['baseline_norm_sq'] ** 0.5:.8f} "
                f"parallel_norm={stats['parallel_norm_sq'] ** 0.5:.8f} "
                f"worst_name={stats['worst_name']}",
                flush=True,
            )
        print(
            "CP_GRAD_WORST "
            f"cp_rank={cp_rank} "
            f"max_abs={worst_abs:.8f} "
            f"name={worst_name}",
            flush=True,
        )
        return worst_abs

    def _install_route_capture(model, captures):
        for name, module in model.named_modules():
            if not hasattr(module, "route") or not hasattr(module, "token_dispatcher"):
                continue
            original_route = module.route

            def wrapped_route(
                hidden_states,
                padding_mask=None,
                input_ids=None,
                packed_seq_params=None,
                *,
                _name=name,
                _original_route=original_route,
            ):
                probs, routing_map = _original_route(
                    hidden_states,
                    padding_mask=padding_mask,
                    input_ids=input_ids,
                    packed_seq_params=packed_seq_params,
                )
                captures.setdefault(_name, []).append(routing_map.detach().clone())
                return probs, routing_map

            module.route = wrapped_route

    def _print_route_diagnostics(baseline_routes, parallel_routes, group, get_partition_mode):
        cp_rank = torch.distributed.get_rank(group=group)
        worst_token_mismatch = 0.0
        worst_name = None
        worst_index = -1
        assert baseline_routes.keys() == parallel_routes.keys()
        for name, baseline_maps in sorted(baseline_routes.items()):
            partition_mode = get_partition_mode(name)
            parallel_maps = parallel_routes[name]
            assert len(baseline_maps) == len(parallel_maps)
            for index, (baseline_map, parallel_map) in enumerate(zip(baseline_maps, parallel_maps)):
                expected_map = get_tensor_on_this_cp_rank(
                    baseline_map, 0, group, cp_partition_mode=partition_mode
                )
                if expected_map.shape != parallel_map.shape:
                    print(
                        "CP_ROUTE_DIAG "
                        f"cp_rank={cp_rank} "
                        f"partition_mode={partition_mode} "
                        f"name={name} "
                        f"index={index} "
                        f"shape_mismatch expected={tuple(expected_map.shape)} "
                        f"actual={tuple(parallel_map.shape)}",
                        flush=True,
                    )
                    token_mismatch = 1.0
                    entry_mismatch = 1.0
                else:
                    mismatch = expected_map != parallel_map
                    token_mismatch = mismatch.any(dim=-1).float().mean().item()
                    entry_mismatch = mismatch.float().mean().item()
                    print(
                        "CP_ROUTE_DIAG "
                        f"cp_rank={cp_rank} "
                        f"partition_mode={partition_mode} "
                        f"name={name} "
                        f"index={index} "
                        f"token_mismatch={token_mismatch:.8f} "
                        f"entry_mismatch={entry_mismatch:.8f}",
                        flush=True,
                    )
                if token_mismatch > worst_token_mismatch:
                    worst_token_mismatch = token_mismatch
                    worst_name = name
                    worst_index = index

        print(
            "CP_ROUTE_WORST "
            f"cp_rank={cp_rank} "
            f"token_mismatch={worst_token_mismatch:.8f} "
            f"name={worst_name} "
            f"index={worst_index}",
            flush=True,
        )
        return worst_token_mismatch

    def _zero_grads(model):
        for param in model.parameters():
            param.grad = None
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                main_grad.zero_()

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=cp,
        expert_tensor_parallel_size=1,
    )
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    input_ids_list = [
        torch.randint(
            low=0,
            high=vocab_size,
            size=(micro_batch_size, sequence_length),
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        for _ in range(num_microbatches)
    ]
    valid_lengths = [
        sequence_length,
        7013,
        6144,
        4097,
        4096,
        3073,
        2048,
        1025,
    ]
    prompt_lengths = [
        0,
        123,
        2345,
        2049,
        3071,
        17,
        1023,
        1000,
    ]
    loss_mask_list = []
    for input_ids, valid_length, prompt_length in zip(
        input_ids_list, valid_lengths, prompt_lengths
    ):
        assert 0 <= prompt_length < valid_length
        input_ids[:, valid_length:] = 0
        loss_mask = torch.zeros(
            (micro_batch_size, sequence_length),
            device=input_ids.device,
            dtype=torch.float32,
        )
        loss_mask[:, prompt_length:valid_length] = 1.0
        loss_mask_list.append(loss_mask)
    position_ids = torch.arange(
        sequence_length, device=input_ids_list[0].device, dtype=torch.long
    ).unsqueeze(0)
    labels_list = []
    for input_ids, valid_length, prompt_length in zip(
        input_ids_list, valid_lengths, prompt_lengths
    ):
        labels = (input_ids + 1) % vocab_size
        labels[:, :prompt_length] = -100
        labels[:, valid_length:] = 0
        labels_list.append(labels)

    with TempNamedDir(
        tmp_path_dist_ckpt / 'test_mixed_gdn_sdpa_gpt_cp_accumulated', sync=True
    ) as ckpt_dir:
        mock_args = parse_args(ignore_unknown_args=True)
        set_args(mock_args)

        baseline_config = make_config(context_parallel_size=1)
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = 1
        mock_args.expert_model_parallel_size = cp
        mock_args.expert_tensor_parallel_size = 1
        baseline_model = unwrap_model(get_model(initialize_gpt_model, config=baseline_config))
        baseline_model[0].train()
        baseline_routes = {}
        _install_route_capture(baseline_model[0], baseline_routes)

        init_checkpointing_mock_args(mock_args, ckpt_dir, False)
        mock_args.no_save_optim = True
        mock_args.no_save_rng = True
        mock_args.no_load_optim = True
        mock_args.no_load_rng = True
        save_checkpoint(10, baseline_model, None, None, 0)

        _zero_grads(baseline_model[0])
        baseline_losses = []
        total_tokens = 0.0
        for input_ids, labels, loss_mask in zip(input_ids_list, labels_list, loss_mask_list):
            baseline_loss = baseline_model[0](
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
            )
            baseline_losses.append(baseline_loss.detach())
            total_tokens += loss_mask.sum().item()
            (baseline_loss.float() * loss_mask).sum().backward()
        _scale_grads(baseline_model[0], 1.0 / total_tokens)
        baseline_grads = _collect_grads(baseline_model[0])

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp,
            expert_model_parallel_size=cp,
            expert_tensor_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(seed)
        parallel_config = make_config(context_parallel_size=cp)
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = cp
        mock_args.expert_model_parallel_size = cp
        mock_args.expert_tensor_parallel_size = 1
        parallel_model = unwrap_model(get_model(initialize_gpt_model, config=parallel_config))
        parallel_model[0].train()
        parallel_routes = {}
        _install_route_capture(parallel_model[0], parallel_routes)
        with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
            with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
                load_checkpoint(parallel_model, None, None)

        cp_group = parallel_state.get_context_parallel_group()
        input_partition_mode = parallel_model[0].get_input_cp_partition_mode()
        assert input_partition_mode in ("contiguous", "zigzag")

        local_position_ids = get_tensor_on_this_cp_rank(
            position_ids, 1, cp_group, cp_partition_mode=input_partition_mode
        )
        _zero_grads(parallel_model[0])
        max_loss_avg_diff = 0.0
        max_loss_token_diff = 0.0
        for input_ids, labels, loss_mask, baseline_loss in zip(
            input_ids_list, labels_list, loss_mask_list, baseline_losses
        ):
            local_input_ids = get_tensor_on_this_cp_rank(
                input_ids, 1, cp_group, cp_partition_mode=input_partition_mode
            )
            local_labels = get_tensor_on_this_cp_rank(
                labels, 1, cp_group, cp_partition_mode=input_partition_mode
            )
            local_loss_mask = get_tensor_on_this_cp_rank(
                loss_mask, 1, cp_group, cp_partition_mode=input_partition_mode
            )
            parallel_loss = parallel_model[0](
                input_ids=local_input_ids,
                position_ids=local_position_ids,
                attention_mask=None,
                labels=local_labels,
            )

            expected_loss = get_tensor_on_this_cp_rank(
                baseline_loss, 1, cp_group, cp_partition_mode=input_partition_mode
            )
            loss_diff = (parallel_loss.float() - expected_loss.float()).abs()
            masked_loss_diff = loss_diff * local_loss_mask
            local_valid_tokens = local_loss_mask.sum()
            if local_valid_tokens.item() > 0:
                local_loss_max = loss_diff[local_loss_mask.bool()].max()
            else:
                local_loss_max = torch.zeros([], device=loss_diff.device)
            local_loss_stats = torch.tensor(
                [
                    (parallel_loss.float() * local_loss_mask).sum().item(),
                    (expected_loss.float() * local_loss_mask).sum().item(),
                    masked_loss_diff.sum().item(),
                    local_valid_tokens.item(),
                ],
                device=parallel_loss.device,
            )
            torch.distributed.all_reduce(local_loss_stats, group=cp_group)
            torch.distributed.all_reduce(
                local_loss_max, op=torch.distributed.ReduceOp.MAX, group=cp_group
            )
            parallel_avg = local_loss_stats[0] / local_loss_stats[3].clamp(min=1)
            expected_avg = local_loss_stats[1] / local_loss_stats[3].clamp(min=1)
            mean_abs = local_loss_stats[2] / local_loss_stats[3].clamp(min=1)
            loss_avg_diff = (parallel_avg - expected_avg).abs().item()
            max_loss_avg_diff = max(max_loss_avg_diff, loss_avg_diff)
            max_loss_token_diff = max(max_loss_token_diff, local_loss_max.item())
            if torch.distributed.get_rank(group=cp_group) == 0:
                print(
                    "CP_LOSS_DIAG "
                    f"input_partition_mode={input_partition_mode} "
                    f"parallel_avg={parallel_avg.item():.8f} "
                    f"expected_avg={expected_avg.item():.8f} "
                    f"avg_diff={loss_avg_diff:.8f} "
                    f"mean_abs={mean_abs.item():.8f} "
                    f"max_abs={local_loss_max.item():.8f}",
                    flush=True,
                )
            (parallel_loss.float() * local_loss_mask).sum().backward()

        _all_reduce_grads(parallel_model[0], cp_group)
        _scale_grads(parallel_model[0], 1.0 / total_tokens)
        parallel_grads = _collect_grads(parallel_model[0])
        baseline_grad_norm = _grad_norm_from_grads(baseline_grads)
        parallel_grad_norm = _grad_norm_from_grads(parallel_grads)
        if torch.distributed.get_rank(group=cp_group) == 0:
            print(
                "CP_GRAD_DIAG "
                f"baseline_norm={baseline_grad_norm.item():.8f} "
                f"parallel_norm={parallel_grad_norm.item():.8f} "
                f"abs_diff={(baseline_grad_norm - parallel_grad_norm).abs().item():.8f}",
                flush=True,
            )

        assert baseline_grads.keys() == parallel_grads.keys()
        def _get_route_partition_mode(name):
            parts = name.split(".")
            if "layers" in parts:
                layer_index = int(parts[parts.index("layers") + 1])
                required_partition_mode = parallel_model[0].decoder.cp_partition_mode_plan[
                    layer_index
                ]
                if required_partition_mode is not None:
                    return required_partition_mode
                return parallel_model[0].decoder.get_cp_partition_mode_before_local_index(
                    layer_index
                )
            return input_partition_mode

        worst_route_mismatch = _print_route_diagnostics(
            baseline_routes, parallel_routes, cp_group, _get_route_partition_mode
        )
        worst_grad_abs = _print_grad_diagnostics(baseline_grads, parallel_grads, cp_group)
        for name, baseline_grad in baseline_grads.items():
            torch.testing.assert_close(
                parallel_grads[name],
                baseline_grad,
                atol=1e-3,
                rtol=0.0,
                msg=lambda msg, param_name=name: f"gradient mismatch for {param_name}: {msg}",
            )
        assert max_loss_avg_diff <= 1e-3, (
            f"max averaged loss diff too large: {max_loss_avg_diff}; "
            f"max token loss diff: {max_loss_token_diff}; "
            f"worst route mismatch: {worst_route_mismatch}; "
            f"worst grad abs: {worst_grad_abs}"
        )

        Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_size", [2, 4], scope="class")
@pytest.mark.internal
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
