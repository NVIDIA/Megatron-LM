# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import inspect
import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, torch_chunk_gated_delta_rule
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils
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
        "experimental_attention_variant": "gdn",
        "linear_attention_freq": [1],
        "transformer_impl": "transformer_engine",
    }
    config_kwargs.update(overrides)
    return TransformerConfig(**config_kwargs)


def _set_gdn_test_cp_partition_mode(packed_seq_params, cp_size, linear_cp_mode):
    if cp_size <= 1:
        return packed_seq_params
    if linear_cp_mode == "headwise":
        packed_seq_params.cp_partition_mode = "zigzag"
    elif linear_cp_mode == "chunkwise":
        packed_seq_params.cp_partition_mode = "contiguous"
    else:
        raise ValueError(f"Invalid linear CP mode: {linear_cp_mode}")
    return packed_seq_params


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


def test_gdn_norm_out_recompute_accepts_gdn_variant():
    config = _make_gdn_config(recompute_granularity="selective", recompute_modules=["gdn_norm_out"])
    assert "gdn_norm_out" in config.recompute_modules


def test_gdn_norm_out_recompute_rejects_non_hybrid_non_gdn_config():
    with pytest.raises(ValueError, match="gdn_norm_out in recompute_modules"):
        _make_gdn_config(
            experimental_attention_variant=None,
            linear_attention_freq=None,
            recompute_granularity="selective",
            recompute_modules=["gdn_norm_out"],
        )


def test_gdn_and_norm_out_recompute_are_mutually_exclusive():
    with pytest.raises(ValueError, match="'gdn' and 'gdn_norm_out'"):
        _make_gdn_config(
            recompute_granularity="selective", recompute_modules=["gdn", "gdn_norm_out"]
        )


def test_gdn_norm_out_recompute_accepts_non_experimental_hybrid_config():
    # Hybrid specs select GDN/KDA per layer without setting a global
    # experimental_attention_variant, so the selector must remain valid here.
    config = _make_gdn_config(
        experimental_attention_variant=None,
        is_hybrid_model=True,
        linear_attention_freq=None,
        recompute_granularity="selective",
        recompute_modules=["gdn_norm_out"],
    )

    assert config.recompute_modules == ["gdn_norm_out"]


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


def test_torch_chunk_gated_delta_rule_preserves_public_signature():
    signature = inspect.signature(torch_chunk_gated_delta_rule)
    assert tuple(signature.parameters) == (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "chunk_size",
        "initial_state",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "cu_seqlens",
        "cp_context",
        "scale",
    )


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
        tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group, tp_cp=tp_cp_group)

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

    @pytest.mark.flaky_in_dev  # Issue #5473
    def test_selective_recompute_gdn_norm_out(self):
        """Output-discarding 'gdn_norm_out' recompute must preserve outputs and gradients."""
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

        def run(recompute_norm_out):
            gdn.recompute_gdn = False
            gdn.recompute_norm_out = recompute_norm_out
            gdn.norm_out_checkpoint = None
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
            out_ref, dinput_ref, pgrad_ref = run(recompute_norm_out=False)
            out_rc, dinput_rc, pgrad_rc = run(recompute_norm_out=True)
        finally:
            gdn.recompute_norm_out = False
            gdn.norm_out_checkpoint = None

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

    def test_deterministic_mode(self):
        if self.cp_size > 1:
            pytest.skip(
                "deterministic_mode uses torch_chunk_gated_delta_rule, which does not support CP."
            )

        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        det_config = copy.deepcopy(self.transformer_config)
        det_config.deterministic_mode = True

        gdn_submodules = get_experimental_attention_variant_module_spec(
            config=det_config
        ).submodules

        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        gdn = (
            GatedDeltaNet(
                det_config,
                submodules=gdn_submodules,
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

        # deterministic_mode must select the torch-native kernel, not FLA.
        assert gdn.gated_delta_rule is torch_chunk_gated_delta_rule

        micro_batch_size = 2
        seq_length = 64
        torch.manual_seed(0)
        base_input = torch.randn(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        def run():
            hidden_states = base_input.clone().requires_grad_(True)
            output, _ = gdn(hidden_states, None)
            output.float().sum().backward()
            grads = {
                name: param.grad.detach().clone()
                for name, param in gdn.named_parameters()
                if param.grad is not None
            }
            gdn.zero_grad(set_to_none=True)
            return output.detach().clone(), grads, hidden_states.grad.detach().clone()

        out1, grads1, input_grad1 = run()
        out2, grads2, input_grad2 = run()

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
        assert gdn.in_proj_dim == 2 * gdn.qk_dim + 2 * gdn.v_dim + 2 * gdn.num_value_heads
        assert gdn.A_log.shape == (gdn.num_value_heads // self.tp_size,)
        assert gdn.dt_bias.shape == (gdn.num_value_heads // self.tp_size,)

    def test_sharded_state_dict_splits_gdn_parameters(self):
        sharded_sd = self.gdn.sharded_state_dict(prefix="gdn.")

        in_proj_weight = sharded_sd["gdn.in_proj.weight"]
        conv1d_weight = sharded_sd["gdn.conv1d.weight"]
        assert isinstance(in_proj_weight, ShardedTensorFactory)
        assert isinstance(conv1d_weight, ShardedTensorFactory)

        in_proj_chunks = in_proj_weight.build()
        assert tuple(chunk.key for chunk in in_proj_chunks) == tuple(
            f"gdn.in_proj.weight.{name}" for name in self.gdn.in_proj_split_names
        )
        assert sum(chunk.data.numel() for chunk in in_proj_chunks) == in_proj_weight.data.numel()

        conv1d_chunks = conv1d_weight.build()
        assert tuple(chunk.key for chunk in conv1d_chunks) == (
            "gdn.conv1d.weight.query",
            "gdn.conv1d.weight.key",
            "gdn.conv1d.weight.value",
        )
        assert sum(chunk.data.numel() for chunk in conv1d_chunks) == conv1d_weight.data.numel()

    def test_jit_compiled_helpers(self):
        import torch._dynamo

        gdn = self.gdn
        batch = 2
        seq_len = 16

        device = torch.cuda.current_device()
        num_v_heads_local = gdn.num_value_heads // gdn.tp_size // self.cp_size_headwise
        qk_dim_local = gdn.qk_dim_local_tp // self.cp_size_headwise
        v_dim_local = gdn.v_dim_local_tp // self.cp_size_headwise

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
        gate_feats = (
            torch.randn(batch, seq_len, num_v_heads_local, device=device, dtype=torch.bfloat16),
            torch.randn(batch, seq_len, num_v_heads_local, device=device, dtype=torch.bfloat16),
        )  # beta, alpha

        # Disable dynamo so coverage.py can trace through the method bodies,
        # which are normally wrapped by @jit_fuser (torch.compile).
        A_log_mock = torch.randn(num_v_heads_local, device=device, dtype=torch.bfloat16)
        dt_bias_mock = torch.randn(num_v_heads_local, device=device, dtype=torch.bfloat16)

        with torch._dynamo.config.patch(disable=True):
            kernel_inputs = gdn._prepare_input_for_gated_delta_rule(
                qkv,
                gate,
                A_log_mock,
                dt_bias_mock,
                batch,
                seq_len,
                *gate_feats,
                cp_size_headwise=self.cp_size_headwise,
            )

        query = kernel_inputs["q"]
        key = kernel_inputs["k"]
        value = kernel_inputs["v"]
        g = kernel_inputs["g"]
        gate_out = kernel_inputs["gate"]
        beta_out = kernel_inputs["beta"]

        assert query.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert key.shape == (batch, seq_len, num_v_heads_local, gdn.key_head_dim)
        assert value.shape == (batch, seq_len, num_v_heads_local, gdn.value_head_dim)
        for t in (query, key, value, gate_out, beta_out):
            assert t.is_contiguous()

        assert g.dtype == torch.float32
        assert g.shape == (batch, seq_len, num_v_heads_local)
        assert beta_out.shape == (batch, seq_len, num_v_heads_local)

    def test_fused_pre_gated_delta_rule_headwise_cp_uses_cp_local_parameters(self):
        if not HAVE_FUSED_PRE_GDR:
            pytest.skip("causal-conv1d fused backward is not installed.")
        if not (self.linear_cp_mode == "headwise" and self.cp_size > 1):
            pytest.skip("Only headwise CP with CP>1 needs CP-local fused pre-GDR params.")

        gdn = self.gdn
        batch = 2
        seq_len = 16
        qk_channels = gdn.qk_dim_local_tp // self.cp_size_headwise
        v_channels = gdn.v_dim_local_tp // self.cp_size_headwise
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
                    "cp_group": kwargs["cp_group"],
                }
            )
            return tuple(torch.empty(0, device=qkvzba_arg.device) for _ in range(6))

        with mock.patch(
            "megatron.core.fusions.fused_pre_gated_delta_rule."
            "fused_streamed_pre_gated_delta_rule",
            side_effect=fake_fused_streamed_pre_gated_delta_rule,
        ):
            gdn._fused_streamed_pre_gated_delta_rule(qkvzba, cp_group_headwise=gdn.pg_collection.cp)

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
        _set_gdn_test_cp_partition_mode(packed_seq_params, self.cp_size, self.linear_cp_mode)

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
        _set_gdn_test_cp_partition_mode(padded_params, self.cp_size, self.linear_cp_mode)
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
        _set_gdn_test_cp_partition_mode(no_padding_params, self.cp_size, self.linear_cp_mode)
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
        _set_gdn_test_cp_partition_mode(padded_mismatch_params, self.cp_size, self.linear_cp_mode)
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=padded_mismatch_params)

        # E) actual mismatch branch without *_padded: should raise.
        actual_mismatch_params = make_test_packed_seq_params(cu_seqlens=[0, 32, 64, 96, 129])
        _set_gdn_test_cp_partition_mode(actual_mismatch_params, self.cp_size, self.linear_cp_mode)
        with pytest.raises(ValueError, match="does not match"):
            self.gdn(hidden_states_thd, None, packed_seq_params=actual_mismatch_params)


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
