# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical and replay coverage for deterministic non-CP CSA attention and indexer gradients."""

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as dk,
)


def _indexer_case(batch, dtype, device):
    generator = torch.Generator(device=device).manual_seed(193)
    q = torch.randn(batch, 4, 2, 3, generator=generator, device=device, dtype=dtype)
    w = torch.randn(batch, 4, 2, generator=generator, device=device, dtype=dtype)
    k = torch.randn(batch, 3, 3, generator=generator, device=device, dtype=dtype)
    indices = torch.tensor(
        [[0, 2, -1], [1, 2, -1], [-1, -1, -1], [0, 1, 2]], device=device, dtype=torch.int32
    ).expand(batch, -1, -1)
    # Row 1's target sums to 0.75: its score gradient is predict * 0.75 - target, not
    # predict - target.
    target = torch.tensor(
        [[0.25, 0.75, 0.0], [0.5, 0.25, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.5]], device=device
    ).expand(batch, -1, -1)
    return q, w, k, indices, target


def _reference_indexer_grads(q, w, k, indices, target, loss_coeff, grad_loss, sm_scale):
    """Differentiate per-sample KL directly; no flattened key addressing or manual dK."""
    ref_k = k.detach().float().requires_grad_()
    ref_w = w.detach().float().requires_grad_()
    predict = torch.zeros_like(target)
    loss = torch.zeros((), device=q.device)
    for batch in range(q.shape[0]):
        for row in range(q.shape[1]):
            valid = indices[batch, row] >= 0
            if not valid.any():
                continue
            selected_keys = ref_k[batch, indices[batch, row, valid].long()]
            dot = q[batch, row].float() @ selected_keys.T
            score = (dot.relu() * ref_w[batch, row, :, None]).sum(0) * sm_scale
            log_predict = torch.log_softmax(score, dim=0)
            predict[batch, row, valid] = log_predict.detach().exp()
            loss = loss - (target[batch, row, valid] * log_predict).sum()
    loss = loss * (loss_coeff * grad_loss / (q.shape[0] * q.shape[1]))
    grad_w, grad_k = torch.autograd.grad(loss, (ref_w, ref_k))
    return predict, grad_w.to(w.dtype), grad_k.to(k.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_grad_w", [False, True])
@pytest.mark.parametrize("chunk_rows", [None, 3])
@pytest.mark.parametrize("torch_deterministic", [True, False])
def test_deterministic_indexer_grads_match_reference_and_replay(
    batch, dtype, compute_grad_w, chunk_rows, torch_deterministic, monkeypatch
):
    # B=1 is also the synthetic-batch contract used by packed THD and CP callers.
    # Three query rows per chunk forces the B=2 case to cross a sample boundary.
    # The torch flag selects index_add_ (on) or the stable sort + prefix sums (off).
    # Tiny test heads otherwise allocate a large unused padding-id array under the 1 GiB budget.
    monkeypatch.setattr(dk, "_DETERMINISTIC_INDEXER_DK_CHUNK_MAX_BYTES", 4096)
    # NGC PyTorch containers allow TF32 for cuBLAS by default, and whether cuBLAS then runs these
    # tiny FP32 GEMMs on TF32 tensor cores depends on the GPU. A TF32 GEMM in the kernel or in the
    # reference alone moves results by ~1e-3, so pin IEEE FP32 for the FP32 tolerance below.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    if chunk_rows is not None:
        monkeypatch.setattr(dk, "_deterministic_indexer_dk_chunk_rows", lambda *_: chunk_rows)
    device = torch.device("cuda", torch.cuda.current_device())
    q, w, k, indices, target = _indexer_case(batch, dtype, device)
    loss_coeff, grad_loss, sm_scale = 0.7, 1.3, 0.5
    predict, expected_w, expected_k = _reference_indexer_grads(
        q, w, k, indices, target, loss_coeff, grad_loss, sm_scale
    )
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(torch_deterministic)
        results = [
            dk._deterministic_sparse_indexer_grads_wk(
                q.clone(),
                w.clone(),
                k.clone(),
                target.clone(),
                predict.clone(),
                indices.clone(),
                loss_coeff=loss_coeff,
                grad_loss=torch.tensor(grad_loss, device=device),
                sm_scale=sm_scale,
                compute_grad_w=compute_grad_w,
            )
            for _ in range(2)
        ]
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
    tolerance = (
        dict(rtol=2e-2, atol=2e-3) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=2e-6)
    )
    for grad_w, grad_k in results:
        torch.testing.assert_close(grad_k, expected_k, **tolerance)
        if compute_grad_w:
            torch.testing.assert_close(grad_w, expected_w, **tolerance)
        else:
            assert grad_w is None
    for first, second in zip(results[0], results[1]):
        if first is not None:
            assert torch.equal(
                first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8)
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_indexer_grads_match_cudnn_kernel():
    """The replacement must follow cuDNN's clipped-log KL, not only well-normalized targets."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("compares against the SM100 cuDNN sparse-indexer backward")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    dk._ensure_dsa_namespace()
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(11)
    rows, keys, heads, dim, topk = 256, 256, 64, 128, 128
    q = torch.randn(1, rows, heads, dim, generator=generator, device=device, dtype=torch.bfloat16)
    w = torch.randn(1, rows, heads, generator=generator, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, keys, dim, generator=generator, device=device, dtype=torch.bfloat16)
    order = torch.rand(rows, keys, generator=generator, device=device).argsort(dim=-1)
    indices = order[:, :topk].int()
    indices[::7, topk // 2 :] = -1
    valid = indices >= 0
    logits = torch.randn(rows, topk, generator=generator, device=device).masked_fill(
        ~valid, float("-inf")
    )
    predict = torch.softmax(logits, dim=-1)
    # Probabilities below exp(-100), including exact zeros, drop out of cuDNN's KL gradient.
    predict[::5, 0] = 0.0
    predict[1::5, 1] = 1e-45
    target = torch.rand(rows, topk, generator=generator, device=device).masked_fill(~valid, 0.0)
    target = target / target.sum(dim=-1, keepdim=True)
    target[::3] *= 0.5
    grad_loss = torch.ones((), device=device)
    det_w, det_k = dk._deterministic_sparse_indexer_grads_wk(
        q,
        w,
        k,
        target.view(1, rows, topk),
        predict.view(1, rows, topk),
        indices.view(1, rows, topk),
        loss_coeff=0.3,
        grad_loss=grad_loss,
        sm_scale=0.1,
        compute_grad_w=True,
    )
    expected = dk._DSA.indexer_backward_wrapper(
        q,
        w,
        k,
        target.view(1, rows, topk).clone(),
        predict.view(1, rows, topk).clone(),
        indices.view(1, rows, topk),
        sm_scale=0.1,
        loss_coeff=0.3,
        grad_loss=grad_loss,
        block_I=128,
    )
    for actual, reference in ((det_k, expected["d_index_k"]), (det_w, expected["d_weights"])):
        error = (actual.float() - reference.float()).norm() / reference.float().norm()
        assert error < 1e-2, f"relative L2 error {error.item():.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [16, 64])
def test_non_cp_sparse_attention_backward_reference_and_replay(heads):
    """Real FlashMLA/cuDNN forward/backward, including head padding and SBHD batches."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("deterministic cuDNN sparse attention requires SM10x")
    pytest.importorskip("flash_mla")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    dk._ensure_dsa_namespace()
    assert (
        "deterministic" in inspect.signature(dk._DSA.sparse_attention_backward_wrapper).parameters
    ), "This regression requires the real deterministic backward API from cudnn-frontend >= 1.29"
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(193)
    sequence, batch, dim = 64, 2, 512
    scale = dim**-0.5
    query = (
        torch.randn(
            sequence, batch, heads, dim, generator=generator, device=device, dtype=torch.bfloat16
        )
        * 0.5
    )
    kv = (
        torch.randn(sequence, batch, dim, generator=generator, device=device, dtype=torch.bfloat16)
        * 0.5
    )
    sink = torch.randn(heads, generator=generator, device=device) * 0.3
    grad_output = (
        torch.randn(
            sequence, batch, heads * dim, generator=generator, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    # Flattening SBHD interleaves samples. Repeated key destinations across query
    # rows exercise dKV accumulation; -1 entries cover partially padded TopK rows.
    query_rows = torch.arange(sequence, device=device).repeat_interleave(batch)
    sample_ids = torch.arange(batch, device=device).repeat(sequence)
    key_rows = torch.arange(sequence, device=device)
    indices = (key_rows[None, :] * batch + sample_ids[:, None]).int()
    indices.masked_fill_(key_rows[None, :] > query_rows[:, None], -1)
    assert (indices == -1).any()

    # Independently differentiate dense causal attention per sample on the CPU.
    # The sink contributes to the softmax denominator and has a zero value vector.
    ref_query, ref_kv, ref_sink = (
        value.detach().cpu().double().requires_grad_() for value in (query, kv, sink)
    )
    causal_mask = torch.arange(sequence)[None, :] > torch.arange(sequence)[:, None]
    outputs = []
    for sample in range(batch):
        logits = torch.einsum("qhd,kd->qhk", ref_query[:, sample], ref_kv[:, sample]) * scale
        logits = logits.masked_fill(causal_mask[:, None, :], float("-inf"))
        logits_with_sink = torch.cat(
            (logits, ref_sink.view(1, heads, 1).expand(sequence, -1, -1)), dim=-1
        )
        probability = logits_with_sink.softmax(dim=-1)[..., :-1]
        outputs.append(torch.einsum("qhk,kd->qhd", probability, ref_kv[:, sample]))
    ref_output = torch.stack(outputs, dim=1).reshape_as(grad_output)
    ref_output.backward(grad_output.cpu().double())
    references = (ref_output, ref_query.grad, ref_kv.grad, ref_sink.grad)

    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_fill = torch.utils.deterministic.fill_uninitialized_memory
    results = []
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        for _ in range(2):
            values = tuple(value.detach().clone().requires_grad_() for value in (query, kv, sink))
            output = dk.csa_sparse_attn(*values, indices, scale, deterministic=True)
            output.backward(grad_output)
            results.append((output.detach(), *(value.grad for value in values)))
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)
        torch.utils.deterministic.fill_uninitialized_memory = prior_fill

    for name, first, second, reference in zip(
        ("output", "query gradient", "KV gradient", "sink gradient"), *results, references
    ):
        assert first is not None and second is not None, name
        assert torch.isfinite(first).all() and torch.isfinite(second).all(), name
        assert torch.equal(
            first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8)
        ), name
        actual = first.cpu().double()
        assert reference.norm() > 0, name
        relative_error = (actual - reference).norm() / reference.norm()
        assert relative_error < 1e-2, f"{name}: relative L2 error {relative_error.item():.3e}"
        atol = 2e-3
        if name == "KV gradient":
            # Shared-key reductions accumulate BF16 roundoff. Scale the absolute
            # allowance by reference RMS so near-zero cancellation is covered.
            reference_rms = reference.square().mean().sqrt().item()
            atol = 4 * torch.finfo(kv.dtype).eps * reference_rms
        torch.testing.assert_close(
            actual,
            reference,
            rtol=2e-2,
            atol=atol,
            msg=lambda message, name=name: f"{name}: {message}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("loss_coeff", [0.0, 0.3])
def test_real_ratio4_no_grad_and_training_forward_parity(loss_coeff, monkeypatch):
    """Real learned-indexer CSA forwards preserve compact key order and output bits."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("deterministic cuDNN sparse attention requires SM10x")
    pytest.importorskip("flash_mla")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    pytest.importorskip("fast_hadamard_transform")
    from megatron.core.models.common.embeddings import RotaryEmbedding
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.experimental_attention_variant import csa as csa_module
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_csa import (
        _make_csa_submodules,
        _make_mla_config,
    )

    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_fill = torch.utils.deterministic.fill_uninitialized_memory
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        torch.manual_seed(193)
        model_parallel_cuda_manual_seed(193)
        config = _make_mla_config(
            num_layers=1,
            num_attention_heads=64,
            v_head_dim=512,
            csa_compress_ratios=[4],
            csa_window_size=32,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=64,
            dsa_indexer_loss_coeff=loss_coeff,
            dsa_indexer_use_sparse_loss=True,
        )
        config.dsa_kernel_backend = "cudnn"
        config.deterministic_mode = True
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        rotary = RotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            cp_group=groups.cp,
        )
        module = (
            csa_module.CompressedSparseAttention(
                config=config,
                submodules=_make_csa_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                pg_collection=groups,
                rotary_pos_emb=rotary,
                compress_ratio=4,
            )
            .cuda()
            .train()
        )
        assert module.use_fused_kernels
        assert isinstance(module.indexer, csa_module.CSAIndexer)
        assert any(parameter.requires_grad for parameter in module.indexer.parameters())
        monkeypatch.setattr(csa_module.DSAIndexerLossLoggingHelper, "tracker", {})
        sequence, batch = 128, 2
        device = torch.device("cuda", torch.cuda.current_device())
        query = torch.randn(sequence, batch, 64, 512, device=device, dtype=torch.bfloat16) * 0.5
        key = torch.randn(sequence, batch, 1, 512, device=device, dtype=torch.bfloat16) * 0.5
        x = torch.randn(sequence, batch, config.hidden_size, device=device, dtype=torch.bfloat16)
        qr = torch.randn(sequence, batch, config.q_lora_rank, device=device, dtype=torch.bfloat16)
        captured = []
        real_flash = dk._csa_fwd_flash_mla

        def recording_flash(q, kv, indices, scale, **kwargs):
            captured.append((indices.detach().clone(), kwargs["topk_length"].detach().clone()))
            return real_flash(q, kv, indices, scale, **kwargs)

        monkeypatch.setattr(dk, "_csa_fwd_flash_mla", recording_flash)

        def forward(grad_enabled):
            with torch.set_grad_enabled(grad_enabled):
                output = module(query, key, key, None, x=x, qr=qr)
            assert output.requires_grad == grad_enabled
            assert torch.isfinite(output).all()
            return output.detach()

        # Compile/autotune both paths before comparing repeated real kernel calls.
        forward(False)
        forward(True)
        captured.clear()
        outputs = [forward(enabled) for enabled in (False, True, False, True)]
        assert len(captured) == 4
        indices, lengths = captured[0]
        assert (indices == -1).any() and (lengths < indices.shape[-1]).any()
        assert torch.equal(lengths, (indices >= 0).sum(-1).int())
        sample_ids = torch.arange(batch, device=device).repeat(sequence)[:, None]
        query_rows = torch.arange(sequence, device=device).repeat_interleave(batch)[:, None]
        assert torch.all((indices < 0) | (indices % batch == sample_ids))
        local_ids = indices // batch
        compressed = local_ids >= sequence
        assert compressed.any()
        assert torch.all(~compressed | ((local_ids - sequence) < (query_rows + 1) // 4))
        window = (indices >= 0) & ~compressed
        assert torch.all(~window | ((local_ids <= query_rows) & (local_ids > query_rows - 32)))
        for (other_indices, other_lengths), output in zip(captured[1:], outputs[1:]):
            assert torch.equal(indices, other_indices), "no-grad and training key order differ"
            assert torch.equal(lengths, other_lengths), "no-grad and training key lengths differ"
            assert torch.equal(
                outputs[0].contiguous().view(torch.uint8), output.contiguous().view(torch.uint8)
            ), "no-grad and training forward bytes differ"
    finally:
        Utils.destroy_model_parallel()
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)
        torch.utils.deterministic.fill_uninitialized_memory = prior_fill


@pytest.mark.parametrize("global_mode", [False, True])
def test_stable_topk_ties_causal_padding_and_short_keys(monkeypatch, global_mode):
    # Force one row per slab and tied zero logits: selection must prefer smaller IDs.
    monkeypatch.setattr(dk, "_STABLE_TOPK_SORT_BYTES", 1)

    class FakeDSA:
        @staticmethod
        def indexer_forward_wrapper(q, k, w, ratio):
            return {"scores": torch.zeros(2, 4, 3)}

        @staticmethod
        def indexer_top_k_wrapper(*args, **kwargs):
            raise AssertionError("Deterministic TopK must not use radix selection")

    monkeypatch.setattr(dk, "_DSA", FakeDSA)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(global_mode)
        actual, lengths, _ = dk._indexer_topk_bshd(
            torch.zeros(2, 4, 1, 2),
            torch.zeros(2, 3, 2),
            torch.ones(2, 4, 1),
            topk=5,
            ratio=2,
            deterministic=True,
        )
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
    expected = torch.tensor(
        [[-1, -1, -1, -1, -1], [0, -1, -1, -1, -1], [0, -1, -1, -1, -1], [0, 1, -1, -1, -1]],
        dtype=torch.int32,
    ).expand(2, -1, -1)
    assert torch.equal(actual, expected)
    assert torch.equal(lengths, torch.tensor([[0, 1, 1, 2]], dtype=torch.int32).expand(2, -1))


def test_deterministic_backward_rejects_missing_cudnn_api(monkeypatch):
    monkeypatch.setattr(dk, "_DSA", SimpleNamespace(sparse_attention_backward_wrapper=lambda: None))
    q = torch.zeros(1, 16, 8)
    with pytest.raises(RuntimeError, match="no deterministic DSA.*>= 1.29"):
        dk._deterministic_sparse_bwd_kwargs(q, requested=True)


def test_deterministic_backward_requires_sm100_and_supported_heads(monkeypatch):
    def backward(q, kv, out, dout, lse, sink, indices, *, deterministic=False):
        raise AssertionError("the support checks must not launch the kernel")

    monkeypatch.setattr(dk, "_DSA", SimpleNamespace(sparse_attention_backward_wrapper=backward))
    with pytest.raises(RuntimeError, match="needs an SM100 GPU"):
        dk._deterministic_sparse_bwd_kwargs(torch.zeros(1, 16, 8), requested=True)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
        q = torch.zeros(1, 16, 8, device="cuda")
        assert dk._deterministic_sparse_bwd_kwargs(q, requested=True) == {"deterministic": True}
        with pytest.raises(RuntimeError, match="query-head count"):
            dk._deterministic_sparse_bwd_kwargs(q[:, :8], requested=True)


def test_central_sparse_backward_forwards_deterministic_flag(monkeypatch):
    seen = {}

    def deterministic_kwargs(q, requested):
        assert requested is True
        return {"deterministic": True}

    def backward(q, kv, out, dout, lse, sink, indices, **kwargs):
        seen.update(kwargs)
        return {"dq": dout.clone(), "dkv": torch.zeros_like(kv), "d_sink": torch.zeros_like(sink)}

    monkeypatch.setattr(dk, "_deterministic_sparse_bwd_kwargs", deterministic_kwargs)
    monkeypatch.setattr(dk, "_DSA", SimpleNamespace(sparse_attention_backward_wrapper=backward))
    q, kv, sink = torch.zeros(2, 16, 8), torch.zeros(3, 8), torch.zeros(16)
    dq, dkv, dsink = dk._csa_sparse_attention_backward(
        q,
        kv,
        q,
        torch.ones_like(q),
        torch.zeros(2, 16),
        sink,
        torch.zeros(2, 1, dtype=torch.int32),
        1.0,
        None,
        deterministic=True,
    )
    # cuDNN's wrapper allocates its scratch workspace per call when none is passed.
    assert seen["deterministic"] is True and "workspace" not in seen
    assert torch.equal(dq, torch.ones_like(q))
    assert dkv.shape == kv.shape and dsink.shape == sink.shape


def test_deterministic_dense_indexer_loss_fails_before_launch(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(dk, "_DSA", SimpleNamespace())
    empty = torch.empty(0)
    with pytest.raises(RuntimeError, match="requires dsa_indexer_use_sparse_loss=True"):
        dk.fused_csa_indexer_sparse_attn(
            *([empty] * 7),
            indexer_topk=2,
            ratio=4,
            softmax_scale=1.0,
            loss_coeff=0.01,
            sparse_loss=False,
            deterministic=True,
        )


def test_torch_flag_alone_preserves_external_kernel_dispatch(monkeypatch):
    """The global torch flag must not opt H100 callers into the SM100-only path."""
    seen = []

    def radix_topk(scores, lengths, **kwargs):
        seen.append(kwargs)
        return {"indices": torch.zeros(scores.shape[0], 1, dtype=torch.int32)}

    monkeypatch.setattr(
        dk,
        "_DSA",
        SimpleNamespace(
            indexer_forward_wrapper=lambda *args, **kwargs: {"scores": torch.zeros(1, 4, 2)},
            indexer_top_k_wrapper=radix_topk,
        ),
    )
    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        q = torch.zeros(1, 4, 1, 2)
        dk._indexer_topk_bshd(q, torch.zeros(1, 2, 2), torch.ones(1, 4, 1), 1, 2)
        assert dk._deterministic_sparse_bwd_kwargs(q, requested=False) == {}
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)
    assert len(seen) == 1
