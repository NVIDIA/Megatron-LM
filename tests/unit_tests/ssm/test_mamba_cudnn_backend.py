# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""mamba_kernel_backend='cudnn' against the Triton training path of MambaMixer."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.ssm.ops.mamba2.cudnn_ssd import cudnn_mamba2_unsupported_reason
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 256


def _config(**kwargs):
    return TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_layers=1,
        num_attention_heads=1,
        use_cpu_initialization=True,
        **kwargs,
    )


def _mixer(config, **kwargs):
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    return MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=pg_collection,
        **kwargs,
    )


def _fwd_bwd(mixer, hidden, dout, packed_seq_params):
    mixer.zero_grad(set_to_none=True)
    x = hidden.detach().requires_grad_()
    out, _ = mixer(x, packed_seq_params=packed_seq_params)
    out.backward(dout)
    grads = {"input": x.grad}
    grads.update({name: p.grad for name, p in mixer.named_parameters()})
    return out.detach(), grads


def _rel(actual, expected):
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def test_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="mamba_kernel_backend must be one of"):
        _config(mamba_kernel_backend="cutlass")


class TestMambaCudnnBackendErrorChecks:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("head_dim, state_dim", [(32, 128), (64, 64)])
    def test_rejects_unsupported_shape(self, head_dim, state_dim):
        config = _config(
            mamba_kernel_backend="cudnn", mamba_head_dim=head_dim, mamba_state_dim=state_dim
        )
        with pytest.raises(ValueError, match="mamba_head_dim=64 and mamba_state_dim=128 only"):
            _mixer(config)

    def test_rejects_non_mem_eff_path(self):
        config = _config(mamba_kernel_backend="cudnn", use_mamba_mem_eff_path=False)
        with pytest.raises(ValueError, match="use_mamba_mem_eff_path=False"):
            _mixer(config)


class TestMambaCudnnBackendParity:

    def setup_method(self, method):
        reason = cudnn_mamba2_unsupported_reason(64, 128)
        if reason is not None:
            pytest.skip(reason)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "ngroups, D_has_hdim, packing",
        [
            (8, False, None),
            (2, True, None),
            (8, False, "seq_idx"),
            (2, False, "cu_seqlens"),
        ],
    )
    def test_matches_triton(self, ngroups, D_has_hdim, packing):
        mixer = _mixer(
            _config(mamba_kernel_backend="cudnn", mamba_num_groups=ngroups), D_has_hdim=D_has_hdim
        )
        mixer = mixer.cuda().bfloat16()
        assert mixer.use_cudnn_kernels

        seqlen, batch = 512, 2
        packed_seq_params = None
        if packing is not None:
            # The tokens after the last boundary form one more sequence.
            batch = 1
            cu_seqlens = torch.tensor([0, 100, 300, 450], dtype=torch.int32, device="cuda")
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                total_tokens=seqlen if packing == "seq_idx" else None,
            )
            assert (packed_seq_params.seq_idx is not None) == (packing == "seq_idx")

        hidden = torch.randn(seqlen, batch, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn_like(hidden)

        mixer.use_cudnn_kernels = False
        expected, expected_grads = _fwd_bwd(mixer, hidden, dout, packed_seq_params)
        mixer.use_cudnn_kernels = True
        actual, grads = _fwd_bwd(mixer, hidden, dout, packed_seq_params)

        assert _rel(actual, expected) < 2e-2
        assert grads.keys() == expected_grads.keys()
        for name, expected_grad in expected_grads.items():
            # The per-head gradients sum heavily cancelling per-token terms. Against an fp32 run,
            # Triton's bf16 dD is up to ~10% off here, and cuDNN's ddt_bias up to ~15%.
            tol = 0.5 if name in ("A_log", "dt_bias", "D") else 2e-2
            assert grads[name] is not None, name
            assert _rel(grads[name], expected_grad) < tol, name
