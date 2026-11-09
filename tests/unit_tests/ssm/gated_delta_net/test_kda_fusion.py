# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import replace

import pytest
import torch

from megatron.core.fusions import fused_pre_kda as fused_pre_kda_module
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.gated_delta_net.test_kda import _build_kda as build_kda
from tests.unit_tests.ssm.gated_delta_net.test_kda import _make_config
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    HAVE_FUSED_PRE_KDA = False
else:
    HAVE_FUSED_PRE_KDA = callable(causal_conv1d_bwd_function)


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
@pytest.mark.skipif(not HAVE_FUSED_PRE_KDA, reason="causal-conv1d backward is not installed.")
class TestFusedPreKDA:
    """Validate KDA's fused preprocessing against the unfused module path."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        """Initialize the single-rank model-parallel test environment."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _build_kda(
        self,
        *,
        fusion: bool,
        f_lora_rank: int | None,
        gate_lora_rank: int | None,
        deterministic_mode: bool | None = None,
    ) -> KimiDeltaAttention:
        if deterministic_mode is None:
            deterministic_mode = not fusion
        config = replace(
            _make_config(f_lora_rank=f_lora_rank, gate_lora_rank=gate_lora_rank),
            deterministic_mode=deterministic_mode,
            gdn_pre_gated_delta_rule_fusion=fusion,
        )
        return build_kda(config)

    @staticmethod
    def _compare_forward_backward(reference, fused, hidden_states, packed_seq_params=None):
        reference_input = hidden_states.detach().clone().requires_grad_(True)
        fused_input = hidden_states.detach().clone().requires_grad_(True)
        reference.zero_grad(set_to_none=True)
        fused.zero_grad(set_to_none=True)

        reference_output, reference_bias = reference(
            reference_input, None, packed_seq_params=packed_seq_params
        )
        fused_output, fused_bias = fused(fused_input, None, packed_seq_params=packed_seq_params)
        torch.testing.assert_close(fused_output, reference_output, atol=4e-3, rtol=4e-3)
        assert fused_bias == reference_bias

        grad_output = torch.linspace(
            -0.1, 0.1, reference_output.numel(), device=reference_output.device, dtype=torch.float32
        ).reshape(reference_output.shape)
        (reference_output.float() * grad_output).sum().backward()
        (fused_output.float() * grad_output).sum().backward()

        torch.testing.assert_close(fused_input.grad, reference_input.grad, atol=4e-2, rtol=4e-2)
        reference_params = dict(reference.named_parameters())
        fused_params = dict(fused.named_parameters())
        assert fused_params.keys() == reference_params.keys()
        for name in reference_params:
            reference_grad = reference_params[name].grad
            fused_grad = fused_params[name].grad
            assert (fused_grad is None) == (reference_grad is None), name
            if reference_grad is not None:
                torch.testing.assert_close(
                    fused_grad,
                    reference_grad,
                    atol=4e-2,
                    rtol=4e-2,
                    msg=lambda msg, param_name=name: f"{param_name} grad mismatch: {msg}",
                )

    @pytest.mark.parametrize(
        ("f_lora_rank", "gate_lora_rank"),
        [(None, None), (16, None), (None, 12), (8, 12)],
        ids=[
            "legacy-fused",
            "low-rank-f-full-rank-gate",
            "full-rank-f-low-rank-gate",
            "low-rank-f-low-rank-gate",
        ],
    )
    def test_fused_and_unfused_forward_backward_match(self, f_lora_rank, gate_lora_rank):
        """Match all legacy, full-rank, and low-rank projection combinations."""

        reference = self._build_kda(
            fusion=False, f_lora_rank=f_lora_rank, gate_lora_rank=gate_lora_rank
        )
        fused = self._build_kda(fusion=True, f_lora_rank=f_lora_rank, gate_lora_rank=gate_lora_rank)
        fused.load_state_dict(reference.state_dict())
        hidden_states = torch.randn(
            (32, 2, reference.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        self._compare_forward_backward(reference, fused, hidden_states)

    def test_activation_recompute_initial_forward_skips_backward_state(self, monkeypatch):
        """Allocate Q/K SiLU state only in the backward-triggered recomputation."""

        config = replace(
            _make_config(f_lora_rank=8, gate_lora_rank=12),
            deterministic_mode=False,
            gdn_pre_gated_delta_rule_fusion=True,
            recompute_granularity="selective",
            recompute_modules=["gdn"],
        )
        kda = build_kda(config)
        hidden_states = torch.randn(
            (32, 2, config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        save_silu_calls = []
        original_forward = fused_pre_kda_module._triton_pre_kda_forward

        def record_save_silu(*args, **kwargs):
            save_silu = kwargs["save_silu"]
            save_silu_calls.append(save_silu)
            result = original_forward(*args, **kwargs)
            if not save_silu:
                assert result[6] is None
            return result

        monkeypatch.setattr(fused_pre_kda_module, "_triton_pre_kda_forward", record_save_silu)

        output, _ = kda(hidden_states, None)
        assert save_silu_calls == [False]

        output.float().sum().backward()

        assert save_silu_calls == [False, True]
        assert hidden_states.grad is not None
        assert kda.conv1d.weight.grad is not None

    def test_fused_and_unfused_packed_low_rank_forward_backward_match(self):
        """Match packed THD forward and backward with an independent F projection."""

        reference = self._build_kda(
            fusion=False, f_lora_rank=16, gate_lora_rank=None, deterministic_mode=False
        )
        fused = self._build_kda(
            fusion=True, f_lora_rank=16, gate_lora_rank=None, deterministic_mode=False
        )
        fused.load_state_dict(reference.state_dict())
        hidden_states = torch.randn(
            (32, 1, reference.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        cu_seqlens = torch.tensor(
            [0, 1, 5, 13, 32], device=torch.cuda.current_device(), dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=19,
            max_seqlen_kv=19,
            total_tokens=hidden_states.shape[0],
        )

        self._compare_forward_backward(
            reference, fused, hidden_states, packed_seq_params=packed_seq_params
        )

    def test_fusion_rejects_deterministic_mode(self):
        """Reject the nondeterministic fusion when deterministic execution is required."""

        with pytest.raises(ValueError, match="Pre-GDR fusion is non-deterministic"):
            self._build_kda(
                fusion=True, f_lora_rank=16, gate_lora_rank=None, deterministic_mode=True
            )


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
@pytest.mark.skipif(not HAVE_FUSED_PRE_KDA, reason="causal-conv1d backward is not installed.")
@pytest.mark.parametrize("linear_cp_mode", ["headwise", "chunkwise"])
def test_fused_low_rank_kda_parallel_correctness(tmp_path_dist_ckpt, linear_cp_mode, run_config):
    """Cover independent low-rank tensors across both KDA CP communication modes."""

    # dev's chunkwise linear CP now requires a contiguous layout (GDP CP support, #6815); the
    # default (zigzag) is rejected. headwise is unconstrained, so only override it for chunkwise.
    layout_override = {"linear_cp_layout": "contiguous"} if linear_cp_mode == "chunkwise" else {}
    config = replace(
        _make_config(linear_cp_mode=linear_cp_mode, f_lora_rank=8, gate_lora_rank=12),
        deterministic_mode=False,
        gdn_pre_gated_delta_rule_fusion=True,
        **layout_override,
    )
    _test_parallel_attention_correctness(
        transformer_config=config,
        transformer_layer_spec=hybrid_stack_spec.submodules.kda_layer,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=1e-2,
        rtol=1e-2,
        input_grad_atol=3.2e-2,
        input_grad_rtol=1e-2,
        cosine_similarity_threshold=0.9999,
        compare_param_grads=True,
        tp=1,
        cp=2,
        seed=42,
        sequence_length=128,
        micro_batch_size=2,
        sequence_packing=True,
    )
