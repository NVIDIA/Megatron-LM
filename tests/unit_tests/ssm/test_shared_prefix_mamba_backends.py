# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix Mamba backend selection, convolution layout and ragged Triton kernels."""

import copy
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid import shared_prefix
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.shared_prefix_layout import SharedPrefixLayout
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import causal_conv1d_fn, mamba_chunk_scan_combined
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

requires_mamba_kernels = pytest.mark.skipif(
    causal_conv1d_fn is None or mamba_chunk_scan_combined is None or not torch.cuda.is_available(),
    reason="requires CUDA, causal-conv1d and mamba-ssm",
)


def _rel(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


def _fake_mixer(*, chunk_size=128, state_dtype=torch.bfloat16):
    return SimpleNamespace(
        pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: 1)),
        config=SimpleNamespace(sequence_parallel=False, params_dtype=torch.bfloat16),
        rmsnorm=True,
        chunk_size=chunk_size,
        mamba_training_ssm_states_dtype=state_dtype,
    )


@pytest.fixture
def kernels_present(monkeypatch):
    """Let the validation tests run without the optional Mamba kernels."""
    monkeypatch.setattr(shared_prefix, "causal_conv1d_fn", object())
    monkeypatch.setattr(shared_prefix, "mamba_chunk_scan_combined", object())


def test_ragged_state_fork_is_the_default(monkeypatch):
    monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
    assert shared_prefix._shared_prefix_mamba_impl() == "ragged_state_fork"


@pytest.mark.parametrize("value", ["ragged", "state-fork", "RAGGED_STATE_FORK", ""])
def test_unknown_mamba_impl_fails_validation(monkeypatch, kernels_present, value):
    monkeypatch.setenv("NRL_SP_MAMBA_IMPL", value)
    with pytest.raises(ValueError, match="NRL_SP_MAMBA_IMPL must be one of"):
        shared_prefix._validate_mamba_fork(_fake_mixer())


@pytest.mark.parametrize("impl", [None, "ragged_state_fork", "ragged_state_fork_training"])
def test_ragged_requires_power_of_two_chunk_size(monkeypatch, kernels_present, impl):
    if impl is None:
        monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
    else:
        monkeypatch.setenv("NRL_SP_MAMBA_IMPL", impl)
    with pytest.raises(NotImplementedError, match="power-of-two chunk_size"):
        shared_prefix._validate_mamba_fork(_fake_mixer(chunk_size=96))
    monkeypatch.setenv("NRL_SP_MAMBA_IMPL", "state_fork")
    shared_prefix._validate_mamba_fork(_fake_mixer(chunk_size=96))


def test_ragged_rejects_non_default_ssm_state_dtype(monkeypatch, kernels_present):
    monkeypatch.setattr(shared_prefix, "MAMBA_HAS_STATE_DTYPE", True)
    monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
    fp32_states = _fake_mixer(state_dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="mamba_training_ssm_states_dtype"):
        shared_prefix._validate_mamba_fork(fp32_states)
    shared_prefix._validate_mamba_fork(_fake_mixer(state_dtype=torch.bfloat16))
    monkeypatch.setenv("NRL_SP_MAMBA_IMPL", "state_fork")
    shared_prefix._validate_mamba_fork(fp32_states)


def _dense_rows(layer, hidden_states, layout):
    """Stock MambaLayer on each expanded [prompt, completion] row, folded back to the star."""
    outputs = []
    for offset, root in layout.iter_roots():
        start = offset + root.prefix_len
        for index, length in enumerate(root.completion_lens):
            row = torch.cat(
                [
                    hidden_states[offset : offset + root.prefix_len],
                    hidden_states[start : start + length],
                ]
            )
            output = layer(hidden_states=row, attention_mask=None)
            output = output[0] if isinstance(output, tuple) else output
            if index == 0:
                outputs.append(output[: root.prefix_len])
            outputs.append(output[root.prefix_len :])
            start += length
    return torch.cat(outputs)


def _forward_backward(layer, forward, hidden_states, cotangent):
    layer.zero_grad(set_to_none=True)
    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    output = forward(hidden_states)
    (output.float() * cotangent).sum().backward()
    grads = {name: param.grad.detach().clone() for name, param in layer.named_parameters()}
    return output.detach(), hidden_states.grad.detach(), grads


@requires_mamba_kernels
class TestSharedPrefixMambaBackends:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'pp', 'cp']
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _config(dtype):
        return TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=4,
            mamba_state_dim=32,
            mamba_head_dim=32,
            mamba_num_groups=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            params_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            use_cpu_initialization=True,
        )

    @pytest.mark.parametrize(
        ("impl", "prefix_len", "completion_lens"),
        [
            # state_fork branch convolution length 3 + 44 + 466 = 513 (L % 512 == 1).
            ("state_fork", 300, (466, 64, 257, 33)),
            # 3 + 44 + 978 = 1025 (L % 1024 == 1, also the bf16 class).
            ("state_fork", 300, (978, 64, 257)),
            # replay_prefix rectangle 256 + 257 = 513.
            ("replay_prefix", 256, (257, 64, 33)),
            (None, 300, (466, 64, 257, 33)),
        ],
    )
    def test_conv_gradients_match_dense_at_channel_first_bug_lengths(
        self, monkeypatch, impl, prefix_len, completion_lens
    ):
        """causal_conv1d's channel-first backward is wrong at these lengths; channel-last is not."""
        torch.manual_seed(1234)
        spec = copy.deepcopy(hybrid_stack_spec.submodules.mamba_layer)
        layer = MambaLayer(
            self._config(torch.float32), spec.submodules, pg_collection=self.pg_collection
        ).cuda()
        layer.mixer.chunk_size = 128
        layout = SharedPrefixLayout(prefix_len=prefix_len, completion_lens=completion_lens)
        generator = torch.Generator(device="cuda").manual_seed(7)
        hidden_states = torch.randn(layout.total_len, 1, 256, device="cuda", generator=generator)
        cotangent = torch.randn(layout.total_len, 1, 256, device="cuda", generator=generator)
        if impl is None:
            monkeypatch.delenv("NRL_SP_MAMBA_IMPL", raising=False)
        else:
            monkeypatch.setenv("NRL_SP_MAMBA_IMPL", impl)

        out, dh, grads = _forward_backward(
            layer,
            lambda h: shared_prefix._forward_mamba_layer_shared_prefix_cp(layer, h, layout),
            hidden_states,
            cotangent,
        )
        ref_out, ref_dh, ref_grads = _forward_backward(
            layer, lambda h: _dense_rows(layer, h, layout), hidden_states, cotangent
        )

        # The channel-first backward bug is about 5e-4 in dh and 3e-3 in the conv gradients.
        assert _rel(out, ref_out) < 1e-5
        assert _rel(dh, ref_dh) < 1e-4
        conv_names = [name for name in ref_grads if "conv1d" in name]
        assert conv_names
        for name in conv_names:
            assert _rel(grads[name], ref_grads[name]) < 2e-5, name

    @pytest.mark.parametrize(
        ("impl", "expected"),
        [
            (None, {"replay_prefix": False, "ragged_state_fork": True}),
            ("state_fork", {"replay_prefix": False, "ragged_state_fork": False}),
            ("replay_prefix", {"replay_prefix": True, "ragged_state_fork": False}),
        ],
    )
    def test_tp1_cp1_single_star_honors_mamba_impl(self, monkeypatch, impl, expected):
        config = self._config(torch.bfloat16)
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers("M", config),
            pp_layer_offset=0,
            pg_collection=self.pg_collection,
        ).cuda()
        calls = []
        original = shared_prefix._forward_mamba_layer_shared_prefix_cp_impl

        def spy(layer, hidden_states, layout, **kwargs):
            calls.append(kwargs)
            return original(layer, hidden_states, layout, **kwargs)

        monkeypatch.setattr(shared_prefix, "_forward_mamba_layer_shared_prefix_cp_impl", spy)
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

    def test_unknown_mamba_impl_fails_before_any_layer_runs(self, monkeypatch):
        config = self._config(torch.bfloat16)
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers("M", config),
            pp_layer_offset=0,
            pg_collection=self.pg_collection,
        ).cuda()

        def fail(*args, **kwargs):
            pytest.fail("a Mamba layer ran before NRL_SP_MAMBA_IMPL was validated")

        monkeypatch.setattr(shared_prefix, "_forward_mamba_layer_shared_prefix_cp", fail)
        monkeypatch.setenv("NRL_SP_MAMBA_IMPL", "ragged")
        layout = SharedPrefixLayout(prefix_len=40, completion_lens=(17, 5, 30))
        hidden_states = torch.randn(layout.total_len, 1, 256, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="NRL_SP_MAMBA_IMPL must be one of"):
            shared_prefix.forward_hybrid_stack_shared_prefix(stack, hidden_states, layout)
