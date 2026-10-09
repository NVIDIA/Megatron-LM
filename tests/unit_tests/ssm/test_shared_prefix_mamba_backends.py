# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix Mamba backend selection, convolution layout and ragged Triton kernels."""

import copy

import pytest
import torch

from megatron.core.models.hybrid import shared_prefix
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
