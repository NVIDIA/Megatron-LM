# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The Hybrid stack's shared-prefix forward routes every Mamba layer through the selected backend.

Mamba-only, so it needs mamba-ssm and causal-conv1d but not flash-attn.
"""

import pytest
import torch

from megatron.core.shared_prefix_layout import SharedPrefixLayout
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixStackMambaDispatch:
    """The Hybrid stack routes every Mamba layer through the selected backend."""

    def setup_method(self, method):
        pytest.importorskip("mamba_ssm")
        pytest.importorskip("causal_conv1d")
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ("impl", "expected"),
        [
            (None, {"replay_prefix": False, "ragged_state_fork": True}),
            ("state_fork", {"replay_prefix": False, "ragged_state_fork": False}),
            ("replay_prefix", {"replay_prefix": True, "ragged_state_fork": False}),
        ],
    )
    def test_tp1_cp1_single_star_honors_mamba_impl(self, monkeypatch, impl, expected):
        # Regression guard: a TP1/CP1 single star once took a dedicated padded path that
        # bypassed NRL_SP_MAMBA_IMPL.
        from megatron.core.models.hybrid import shared_prefix
        from megatron.core.models.hybrid.hybrid_block import HybridStack
        from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.ssm import mamba_shared_prefix
        from megatron.core.transformer import TransformerConfig

        config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=4,
            mamba_state_dim=32,
            mamba_head_dim=32,
            mamba_num_groups=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            params_dtype=torch.bfloat16,
            bf16=True,
            use_cpu_initialization=True,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'pp', 'cp']
        )
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers("M", config),
            pp_layer_offset=0,
            pg_collection=pg_collection,
        ).cuda()
        calls = []
        original = mamba_shared_prefix._forward_mamba_layer_shared_prefix_cp_impl

        def spy(layer, hidden_states, layout, **kwargs):
            calls.append(kwargs)
            return original(layer, hidden_states, layout, **kwargs)

        monkeypatch.setattr(mamba_shared_prefix, "_forward_mamba_layer_shared_prefix_cp_impl", spy)
        if impl is None:
            monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
        else:
            monkeypatch.setenv("NRL_SP_MAMBA_IMPL", impl)
        layout = SharedPrefixLayout(prefix_len=40, completion_lens=(17, 5, 30))
        hidden_states = torch.randn(
            layout.total_len, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        output = shared_prefix.forward_hybrid_stack_shared_prefix(stack, hidden_states, layout)
        output.sum().backward()

        assert output.shape == hidden_states.shape
        assert len(calls) == 1
        assert {key: calls[0].get(key, False) for key in expected} == expected
