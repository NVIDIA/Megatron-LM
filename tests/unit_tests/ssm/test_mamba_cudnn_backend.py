# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""mamba_kernel_backend='cudnn' against the Triton training path of MambaMixer."""

import sys
import types
from dataclasses import replace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import (
    CudnnMambaSplitConv1dScan,
    MambaMixer,
    TritonMambaSplitConv1dScan,
)
from megatron.core.ssm.ops.mamba2.cudnn_ssd import (
    _cudnn_mamba2_import_error,
    _ensure_stride,
    cudnn_mamba2_unsupported_reason,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 256
MIXER_SUBMODULES = hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules


def _config(**kwargs):
    defaults = dict(
        hidden_size=HIDDEN_SIZE, num_layers=1, num_attention_heads=1, use_cpu_initialization=True
    )
    return TransformerConfig(**(defaults | kwargs))


def _mixer(config, submodules=MIXER_SUBMODULES, **kwargs):
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    return MambaMixer(
        config,
        submodules,
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


def test_ensure_stride_aligns_rows():
    # A [1, 1, d] slice is contiguous to PyTorch whatever its row stride.
    one_token = torch.randn(1, 1, 385)[..., :320]
    assert one_token.is_contiguous() and one_token.stride(1) == 385
    aligned = _ensure_stride(one_token)
    assert aligned.stride(1) % 8 == 0 and torch.equal(aligned, one_token)

    multi_token = torch.randn(2, 3, 385)[..., :320]
    aligned = _ensure_stride(multi_token)
    assert aligned.stride(1) % 8 == 0 and torch.equal(aligned, multi_token)

    already_aligned = torch.randn(1, 4, 392)[..., :320]
    assert _ensure_stride(already_aligned) is already_aligned


def test_rejects_old_cudnn_frontend(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudnn", types.SimpleNamespace(__version__="1.32.0"))
    _cudnn_mamba2_import_error.cache_clear()
    try:
        assert _cudnn_mamba2_import_error() == "found 1.32.0"
    finally:
        _cudnn_mamba2_import_error.cache_clear()


class TestMambaKernelSelection:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_spec_kernel_overrides_config(self):
        # mamba_head_dim=32 would make the cuDNN kernel module raise.
        config = _config(mamba_kernel_backend="cudnn", mamba_head_dim=32)
        submodules = replace(MIXER_SUBMODULES, split_conv1d_scan=TritonMambaSplitConv1dScan)
        mixer = _mixer(config, submodules)
        assert isinstance(mixer.split_conv1d_scan, TritonMambaSplitConv1dScan)

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


@pytest.mark.launch_on_gb200
class TestMambaCudnnBackendParity:

    def setup_method(self, method):
        reason = cudnn_mamba2_unsupported_reason(64, 128)
        if reason is not None:
            pytest.skip(reason)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("batch_first", [True, False], ids=["batch_first", "sequence_first"])
    @pytest.mark.parametrize(
        "nheads, ngroups, D_has_hdim, packing",
        [
            (8, 8, False, None),
            (8, 2, True, None),
            (8, 8, False, "seq_idx"),
            (8, 2, False, "cu_seqlens"),
            # zxBCdt rows that are not 16-byte aligned, which go through packed copies.
            (4, 2, False, None),
            (12, 4, True, "cu_seqlens"),
            (20, 4, False, "seq_idx"),
        ],
    )
    def test_matches_triton(self, monkeypatch, nheads, ngroups, D_has_hdim, packing, batch_first):
        hidden_size = 32 * nheads
        mixers = {}
        for backend in ("triton", "cudnn"):
            config = _config(
                mamba_kernel_backend=backend, hidden_size=hidden_size, mamba_num_groups=ngroups
            )
            mixer = _mixer(config, D_has_hdim=D_has_hdim).cuda().bfloat16()
            # This config passes the batch-first gate, so only sequence-first needs pinning.
            assert mixer._use_batch_first_ssm(inference_mode=False)
            if not batch_first:
                monkeypatch.setattr(mixer, "_use_batch_first_ssm", lambda inference_mode: False)
            mixers[backend] = mixer
        # The kernel modules hold no state, so the mixers swap state dicts.
        mixers["cudnn"].load_state_dict(mixers["triton"].state_dict())
        assert isinstance(mixers["cudnn"].split_conv1d_scan, CudnnMambaSplitConv1dScan)
        assert mixers["cudnn"].nheads == nheads

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

        hidden = torch.randn(seqlen, batch, hidden_size, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn_like(hidden)

        expected, expected_grads = _fwd_bwd(mixers["triton"], hidden, dout, packed_seq_params)
        actual, grads = _fwd_bwd(mixers["cudnn"], hidden, dout, packed_seq_params)

        assert _rel(actual, expected) < 2e-2
        assert grads.keys() == expected_grads.keys()
        for name, expected_grad in expected_grads.items():
            # The per-head gradients sum heavily cancelling per-token terms. Against an fp32 run,
            # Triton's bf16 dD is up to ~10% off here, and cuDNN's ddt_bias up to ~15%.
            tol = 0.5 if name in ("A_log", "dt_bias", "D") else 2e-2
            assert grads[name] is not None, name
            assert _rel(grads[name], expected_grad) < tol, name

    @pytest.mark.parametrize("ngroups", [1, 2])
    def test_single_token(self, ngroups):
        """One token with four heads, whose zxBCdt rows are not 16-byte aligned.

        Compares the single-token output with token 0 of an 8-token run, both on the same cuDNN
        mixer, which is causal. The gradients are only checked for being finite.
        """
        hidden_size = 128
        mixer = _mixer(
            _config(mamba_kernel_backend="cudnn", hidden_size=hidden_size, mamba_num_groups=ngroups)
        )
        mixer = mixer.cuda().bfloat16()
        assert mixer.nheads == 4 and mixer.in_proj.weight.shape[0] % 8 != 0

        hidden = torch.randn(8, 1, hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            expected, _ = mixer(hidden)
        actual, grads = _fwd_bwd(mixer, hidden[:1], torch.randn_like(hidden[:1]), None)

        assert _rel(actual, expected[:1]) < 1e-2
        for name, grad in grads.items():
            assert grad is not None and torch.isfinite(grad).all(), name
