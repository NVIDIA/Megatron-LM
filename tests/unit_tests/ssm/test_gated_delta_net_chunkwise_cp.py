# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunkwise context parallelism for Gated DeltaNet (FLA CP kernels).

The CP=2 tests need a two-rank world, e.g.
``torchrun --nproc_per_node=2 -m pytest tests/unit_tests/ssm/test_gated_delta_net_chunkwise_cp.py``.
"""

import os

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
from megatron.core.ssm.context_parallel.chunkwise import build_packed_sequence_cp_metadata
from megatron.core.ssm.gated_delta_net import HAVE_FLA, GatedDeltaNet
from megatron.core.ssm.gated_delta_net.common import build_cp_context
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable
# NVLS doesn't support one single GPU to be shared by multiple ranks, so disable this in test
os.environ.update({"NCCL_NVLS_ENABLE": "0"})

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    HAVE_FUSED_PRE_GDR = False
else:
    HAVE_FUSED_PRE_GDR = callable(causal_conv1d_bwd_function)

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_requires_cp2 = [
    pytest.mark.internal,
    pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed."),
    pytest.mark.skipif(
        build_cp_context is None, reason="FLA build without fla.ops.cp.build_cp_context."
    ),
    pytest.mark.skipif(
        _WORLD_SIZE < 2,
        reason="CP=2 tests require a multi-rank world; run via torchrun --nproc_per_node=2.",
    ),
]


def _gdn_config_kwargs(**overrides):
    kwargs = dict(
        hidden_size=256,
        linear_conv_kernel_dim=4,
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
        linear_cp_layout="contiguous",
        transformer_impl="transformer_engine",
        deterministic_mode=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_gdn_chunkwise_cp_head_divisibility_ignores_cp_size():
    """Chunkwise CP keeps heads local, so head counts only need to divide by TP."""
    config = TransformerConfig(
        **_gdn_config_kwargs(
            context_parallel_size=8, linear_num_key_heads=2, linear_num_value_heads=4
        )
    )
    assert config.linear_cp_mode == "chunkwise"


def test_gdn_headwise_cp_head_divisibility_includes_cp_size():
    """Headwise CP shards heads over CP, so head counts must divide by TP x CP."""
    with pytest.raises(AssertionError, match="linear_head_parallel_size"):
        TransformerConfig(
            **_gdn_config_kwargs(
                context_parallel_size=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_cp_mode="headwise",
                linear_cp_layout="zigzag",
            )
        )


def test_gdn2_rejects_chunkwise_cp():
    """GDN2 has no chunkwise kernel path."""
    with pytest.raises(ValueError, match="supports only headwise context parallelism"):
        TransformerConfig(**_gdn_config_kwargs(experimental_attention_variant="gdn2"))


def test_gpt_block_spec_rejects_gdn_chunkwise_cp():
    """Only HybridStack converts GDN layers to the contiguous layout chunkwise CP needs."""
    config = TransformerConfig(**_gdn_config_kwargs())
    with pytest.raises(ValueError, match="supported only in HybridModel"):
        get_transformer_block_with_experimental_attention_variant_spec(
            config=config, vp_stage=None, pp_rank=0
        )


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


def _build_gdn(cp_group, **config_overrides):
    """Build a GDN wired to ``cp_group`` (chunkwise CP when its size is > 1)."""
    config = TransformerConfig(
        **_gdn_config_kwargs(context_parallel_size=cp_group.size(), **config_overrides)
    )
    pg_collection = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(), cp=cp_group
    )
    gdn = GatedDeltaNet(
        config,
        submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    )
    return gdn.cuda().bfloat16()


def _make_packed_seq_params(seq_lens):
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(seq_lens), 0).tolist(),
        dtype=torch.int32,
        device=torch.cuda.current_device(),
    )
    total = int(cu[-1].item())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(seq_lens),
        max_seqlen_kv=max(seq_lens),
        total_tokens=total,
    )


@pytest.mark.parametrize(
    "seq_lens,fusion",
    (
        pytest.param(None, False, id="sbhd"),
        pytest.param([96, 32], False, id="thd"),
        pytest.param([4, 60], False, id="thd-short-boundary"),
        pytest.param(None, True, id="sbhd-fused-pre-gdr"),
        pytest.param([96, 32], True, id="thd-fused-pre-gdr"),
    ),
)
class TestGatedDeltaNetChunkwiseCP:
    """GDN with chunkwise CP=2 matches a CP=1 reference on the full sequence."""

    pytestmark = _requires_cp2

    @pytest.fixture(autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        yield
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    def test_forward_and_backward_match_reference(self, seq_lens, fusion):
        if fusion and not HAVE_FUSED_PRE_GDR:
            pytest.skip("causal-conv1d fused backward is not installed.")
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = parallel_state.get_context_parallel_rank()
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        cp1_group = [torch.distributed.new_group(ranks=[r]) for r in range(world_size)][global_rank]

        gdn_cp = _build_gdn(cp_group, gdn_pre_gated_delta_rule_fusion=fusion)
        for parameter in gdn_cp.parameters():
            torch.distributed.broadcast(parameter.data, src=0)
        gdn_reference = _build_gdn(cp1_group, linear_cp_mode="headwise", linear_cp_layout="zigzag")
        gdn_reference.load_state_dict(gdn_cp.state_dict())

        packed_seq_params = None
        if seq_lens is None:
            total_tokens = 128
        else:
            packed_seq_params = _make_packed_seq_params(seq_lens)
            total_tokens = sum(seq_lens)
        torch.manual_seed(0)
        hidden_full = torch.randn(
            total_tokens, 1, gdn_cp.config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        torch.distributed.broadcast(hidden_full, src=0)

        reference_input = hidden_full.detach().clone().requires_grad_(True)
        reference_output, _ = gdn_reference(
            reference_input, None, packed_seq_params=packed_seq_params
        )
        reference_output.float().square().sum().backward()

        local_len = total_tokens // cp_group.size()
        local_slice = slice(cp_rank * local_len, (cp_rank + 1) * local_len)
        local_input = hidden_full[local_slice].detach().clone().requires_grad_(True)
        kwargs = {}
        if packed_seq_params is not None:
            seq_idx = torch.repeat_interleave(
                torch.arange(len(seq_lens), dtype=torch.int32, device="cuda"),
                torch.tensor(seq_lens, device="cuda"),
            ).unsqueeze(0)
            kwargs["packed_sequence_cp_metadata"] = build_packed_sequence_cp_metadata(
                seq_idx, cp_rank=cp_rank, cp_size=cp_group.size()
            )
        local_output, _ = gdn_cp(local_input, None, packed_seq_params=packed_seq_params, **kwargs)
        local_output.float().square().sum().backward()

        torch.testing.assert_close(
            local_output, reference_output[local_slice], atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            local_input.grad, reference_input.grad[local_slice], atol=5e-2, rtol=5e-2
        )
        reference_parameters = dict(gdn_reference.named_parameters())
        for name, parameter in gdn_cp.named_parameters():
            local_grad = (
                torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.clone()
            )
            torch.distributed.all_reduce(local_grad, group=cp_group)
            reference_grad = reference_parameters[name].grad
            assert reference_grad is not None, f"reference has no gradient for {name}"
            torch.testing.assert_close(
                local_grad,
                reference_grad,
                atol=1e-1,
                rtol=5e-2,
                msg=lambda msg, param_name=name: f"grad mismatch for {param_name!r}: {msg}",
            )


@pytest.mark.skipif(not HAVE_FUSED_PRE_GDR, reason="causal-conv1d fused backward is not installed.")
class TestFusedPreGatedDeltaRuleChunkwiseCP:
    """Fused and unfused pre-GDR agree under chunkwise CP=2 (ported from dev #5638)."""

    pytestmark = _requires_cp2

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        yield
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_pair(conv_kernel_dim: int = 4):
        cp_group = parallel_state.get_context_parallel_group()
        unfused = _build_gdn(
            cp_group, gdn_pre_gated_delta_rule_fusion=False, linear_conv_kernel_dim=conv_kernel_dim
        )
        fused = _build_gdn(
            cp_group, gdn_pre_gated_delta_rule_fusion=True, linear_conv_kernel_dim=conv_kernel_dim
        )
        fused.load_state_dict(unfused.state_dict())
        return unfused, fused

    @staticmethod
    def _make_hidden_states(gdn, *, seq_len_global: int):
        cp_size = parallel_state.get_context_parallel_world_size()
        assert seq_len_global % cp_size == 0
        cp_rank = parallel_state.get_context_parallel_rank()
        hidden_states = torch.randn(
            (seq_len_global // cp_size, 1, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        return hidden_states + cp_rank

    @staticmethod
    def _run_forward(gdn, hidden_states, packed_seq_params=None):
        with torch.no_grad():
            output, bias = gdn(hidden_states, None, packed_seq_params=packed_seq_params)
        torch.distributed.barrier()
        return output, bias

    @staticmethod
    def _run_backward(gdn, hidden_states, grad_output, packed_seq_params=None):
        gdn.zero_grad(set_to_none=True)
        hidden_states = hidden_states.detach().clone().requires_grad_(True)
        output, _ = gdn(hidden_states, None, packed_seq_params=packed_seq_params)
        (output.float() * grad_output).sum().backward()
        torch.distributed.barrier()
        param_grads = {
            name: param.grad.detach().clone()
            for name, param in gdn.named_parameters()
            if param.grad is not None
        }
        return output.detach(), hidden_states.grad.detach().clone(), param_grads

    def _assert_backward_match(self, packed_seq_params, seq_len_global, param_grad_atol, label):
        unfused_gdn, fused_gdn = self._build_pair()
        hidden_states = self._make_hidden_states(unfused_gdn, seq_len_global=seq_len_global)
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
            msg=lambda msg: f"{label} fused backward output mismatch ({rank=}): {msg}",
        )
        torch.testing.assert_close(
            fused_dinput,
            unfused_dinput,
            atol=5e-2,
            rtol=5e-2,
            msg=lambda msg: f"{label} fused input grad mismatch ({rank=}): {msg}",
        )
        assert fused_grads.keys() == unfused_grads.keys()
        for name in unfused_grads:
            torch.testing.assert_close(
                fused_grads[name],
                unfused_grads[name],
                atol=param_grad_atol,
                rtol=5e-2,
                msg=lambda msg, param_name=name: (
                    f"{label} fused grad mismatch for {param_name!r} ({rank=}): {msg}"
                ),
            )

    def test_fused_and_unfused_forward_chunkwise_cp_match(self):
        unfused_gdn, fused_gdn = self._build_pair()
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
        self._assert_backward_match(None, 64, 5e-2, "chunkwise CP")

    def test_fused_chunkwise_cp_rejects_short_local_chunks(self):
        _, fused_gdn = self._build_pair(conv_kernel_dim=4)
        hidden_states = self._make_hidden_states(fused_gdn, seq_len_global=4)

        with pytest.raises((AssertionError, ValueError), match="local.*chunk|conv_kernel_dim"):
            fused_gdn(hidden_states, None)

    def test_fused_and_unfused_packed_forward_chunkwise_cp_match(self):
        unfused_gdn, fused_gdn = self._build_pair()
        packed_seq_params = _make_packed_seq_params([96, 32])
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
        # Packed chunkwise CP compares two bf16 backward implementations that differ in
        # their causal-conv boundary path. A few parameter-gradient elements can land just
        # above the dense-path tolerance while outputs and input gradients match tightly.
        self._assert_backward_match(_make_packed_seq_params([96, 32]), 128, 1e-1, "packed CP")

    def test_fused_and_unfused_packed_partial_boundary_chunkwise_cp_match(self):
        self._assert_backward_match(
            _make_packed_seq_params([4, 8]), 12, 1e-1, "partial-boundary packed CP"
        )
