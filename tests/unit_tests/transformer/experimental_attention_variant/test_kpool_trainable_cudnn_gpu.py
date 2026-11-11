# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU tests of the trainable KPool indexer loss on cuDNN frontend kernels (phase 2).

At TP size 1, with ``dsa_kernel_backend='cudnn'``, the cuDNN KPool scorer, HF scoring, a
forward the scorer accepts and cuDNN frontend >= 1.31, a trainable KPool layer computes its
sparse pool-level KL after the sparse attention, on cuDNN frontend kernels: the teacher from
FE ``sparse_attn_score_recompute`` fed the attention's own LSE, the student from the cuDNN
selector's pool scores and the gradients from the FE sparse indexer backward on the pooled
keys. Everywhere else the PyTorch ``KPoolSparseIndexerLoss`` runs. These tests run real
``DSAttention`` layers (Transformer Engine indexer projections, FlashMLA sparse attention):

- the cuDNN path against the PyTorch path on one layer and input (the frontend gate forces
  the PyTorch path): the same selection, the loss to BF16 tolerance and every indexer
  gradient, the pool gate and the slot bias included, to BF16 tolerance; the attention
  output and the trunk gradients bitwise when the selections are bitwise equal;
- the cuDNN path against the independent reference (``kpool_trainable_reference.py``):
  loss and the gradients of ``q``, ``weights``, ``k``, the gate score, the gate and the APE,
  with two samples per batch;
- the dispatch matrix scorer x scoring x layout at TP1 (against the reference) and at TP2
  (against the TP1 layer), with the one-time log line naming the cuDNN path only for TP1 +
  cuDNN scorer + HF scoring + an accepted forward (+ cuDNN frontend >= 1.31).

The layers use the cuDNN DSA geometry of ``test_kpool_trainable_gpu.py`` (64 heads, latent
512 + RoPE 64, a 32 x 128 indexer). TP2 cases skip below two ranks. Launch from the
Megatron-LM root, e.g.::

    torchrun --nproc-per-node 2 -m pytest -q tests/unit_tests/transformer/\
experimental_attention_variant/test_kpool_trainable_cudnn_gpu.py
"""

import logging

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels
from tests.unit_tests.test_utilities import Utils

# exact_cuda_math and unit_indexer_loss_scale are autouse fixtures: importing them applies them.
from tests.unit_tests.transformer.experimental_attention_variant.test_kpool_trainable_gpu import (  # noqa: E501,F401
    _APE,
    _COEFF,
    _CUDNN_GEOMETRY,
    _CUDNN_TOPK,
    _GATE,
    _REFERENCE_PROFILE,
    _TRUNK_INPUTS,
    HAVE_TE,
    _all_gather,
    _assert_indexer_trained,
    _assert_similar,
    _build_layer,
    _config,
    _forbid_token_level_loss_hooks,
    _install_recorders,
    _layer_losses,
    _make_inputs,
    _oracle,
    _packed_seq_params,
    _reference_case,
    _require_cudnn_dsa,
    _require_ranks,
    _run_layer,
    _single_rank_groups,
    _slice_heads,
    _spy_indexer,
    exact_cuda_math,
    unit_indexer_loss_scale,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available"),
]

_ROWS = 512
_PACKED_CU = (0, 141, 350, _ROWS)  # documents off the pool grid
_DSA_LOGGER = dsa_module.__name__
_CUDNN_PATH = "runs on the cuDNN frontend kernels"
_PYTORCH_PATH = "runs in PyTorch (KPoolSparseIndexerLoss)"


@pytest.fixture(autouse=True)
def fresh_path_log(monkeypatch):
    """Every test sees the first forward of each reason, so each logs its path line."""
    monkeypatch.setattr(dsa_module, "_KPOOL_LOSS_PATHS_LOGGED", set())


def _frontend_runs_the_cudnn_loss():
    return dsa_cudnn_kernels._cudnn_frontend_at_least(
        dsa_cudnn_kernels._KPOOL_SPARSE_LOSS_MIN_FE_VERSION
    )


def _require_cudnn_loss():
    _require_cudnn_dsa()
    if not _frontend_runs_the_cudnn_loss():
        pytest.skip(
            "the cuDNN KPool loss needs cuDNN frontend >= 1.31, found "
            f"{dsa_cudnn_kernels._cudnn_frontend_version()}"
        )


def _count_cudnn_loss(monkeypatch):
    counts = {"cudnn": 0}
    function = dsa_cudnn_kernels.KPoolSparseIndexerLossFromScoresFunc
    real_apply = function.apply

    def apply(*args, **kwargs):
        counts["cudnn"] += 1
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(function, "apply", apply)
    return counts


def _path_lines(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _DSA_LOGGER and "trainable KPool indexer loss runs" in record.getMessage()
    ]


def _cudnn_layer_config(*, scorer="cudnn", hf_scoring=True, per_token=False, tp=1):
    return _config(
        loss_coeff=_COEFF,
        hf_scoring=hf_scoring,
        topk=_CUDNN_TOPK,
        kernel_backend="cudnn",
        scorer=scorer,
        geometry=_CUDNN_GEOMETRY,
        per_token=per_token,
        tp=tp,
    )


def _row_sets(published):
    """Per row, the sorted selected token ids: the selection without its slot order."""
    return torch.sort(published.long(), dim=-1).values


def _check_reference(layer, inputs, run, indexer_call, record, *, cu_seqlens, eps):
    """The logged loss and every indexer gradient against the independent reference."""
    config = layer.config
    batch = inputs["query"].size(1)
    rows = inputs["query"].size(0)
    row_count = 1.0 if config.calculate_per_token_loss else float(batch * rows)
    case = _reference_case(layer, inputs, run, indexer_call, cu_seqlens=cu_seqlens)
    common = dict(case.fixed, loss_coeff=_COEFF, row_count=row_count)
    profile = _REFERENCE_PROFILE[config.dsa_indexer_kpool_key_dtype_compression]
    key_dtype_leaves = ("k", "gate_score") if config.dsa_indexer_kpool_key_dtype_compression else ()
    leaves = {
        name: (tensor.clone() if name in key_dtype_leaves else tensor.double()).requires_grad_()
        for name, tensor in case.tensors.items()
    }
    oracle = _oracle(**leaves, **common, profile=profile)
    torch.testing.assert_close(
        record["loss"].double().cpu(), oracle.loss.detach().double(), rtol=1e-4, atol=1e-8
    )
    oracle.loss.backward()
    for name in ("q", "weights", "k", "gate_score"):
        assert name in indexer_call.grads, f"no gradient reached the indexer's {name}"
        _assert_similar(indexer_call.grads[name], leaves[name].grad, label=f"d{name}", eps=eps)
    expected_gate = torch.einsum("sbd,sbh->dh", leaves["gate_score"].grad.double(), case.x.double())
    _assert_similar(run.param_grads[_GATE], expected_gate, label="dgate", eps=eps)
    _assert_similar(run.param_grads[_APE], leaves["ape"].grad, label="dape", eps=eps)


# ---------------------------------------------------------------------------------------
# The cuDNN path against the PyTorch path, and against the reference
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("per_token", [False, True], ids=["mean", "per-token"])
def test_cudnn_path_matches_the_pytorch_path_on_one_layer(monkeypatch, caplog, per_token):
    """One layer, one input: the cuDNN path, then the PyTorch path forced by the frontend gate.

    Both select with the cuDNN KPool scorer, whose top-k launch is the same with or without
    the pool scores. The selections must hold the same pools; when they are bitwise equal
    (slot order included) so are the attention output and the main query, key and value
    gradients, since asking the attention for its LSE leaves its output unchanged. The loss
    agrees to BF16 tolerance, and so does every indexer gradient: ``q``, ``weights``, ``k``
    and the gate score at the indexer, and every indexer parameter (gate and APE included).
    """
    _require_cudnn_loss()
    seen = _install_recorders(monkeypatch)
    counts = _count_cudnn_loss(monkeypatch)
    config = _cudnn_layer_config(per_token=per_token)
    layer = _build_layer(config, _single_rank_groups())
    _forbid_token_level_loss_hooks(monkeypatch)
    indexer_calls = _spy_indexer(layer)
    inputs = _make_inputs(config, _ROWS, batch=2, seed=31)

    with caplog.at_level(logging.INFO, logger=_DSA_LOGGER):
        cudnn = _run_layer(layer, inputs)
    assert (counts["cudnn"], seen.loss_function_calls) == (1, 0)
    assert any(_CUDNN_PATH in line for line in _path_lines(caplog)), _path_lines(caplog)
    (cudnn_loss,) = _layer_losses(seen)

    caplog.clear()
    seen.losses.clear()
    monkeypatch.setattr(dsa_cudnn_kernels, "_KPOOL_SPARSE_LOSS_MIN_FE_VERSION", (99, 0))
    with caplog.at_level(logging.INFO, logger=_DSA_LOGGER):
        pytorch = _run_layer(layer, inputs)
    assert (counts["cudnn"], seen.loss_function_calls) == (1, 1)
    lines = _path_lines(caplog)
    assert any(_PYTORCH_PATH in line and "older than 99.0" in line for line in lines), lines
    (pytorch_loss,) = _layer_losses(seen)

    # The fused sparse-attention backward accumulates dQ/dKV with atomics, so two runs of the
    # same path need not agree bitwise. A second PyTorch-path run sets the bar per gradient:
    # bitwise where the path repeats itself bitwise, otherwise within BF16 similarity.
    pytorch_again = _run_layer(layer, inputs)

    assert torch.equal(_row_sets(cudnn.published), _row_sets(pytorch.published))
    if torch.equal(cudnn.published, pytorch.published):
        assert torch.equal(cudnn.output, pytorch.output), "the attention output changed"
        for name in _TRUNK_INPUTS:
            if torch.equal(pytorch.input_grads[name], pytorch_again.input_grads[name]):
                assert torch.equal(cudnn.input_grads[name], pytorch.input_grads[name]), name
            else:
                _assert_similar(
                    cudnn.input_grads[name], pytorch.input_grads[name], label=name, eps=1e-4
                )
    else:
        _assert_similar(cudnn.output, pytorch.output, label="output", eps=1e-3)
    assert cudnn.input_grads["x"] is None and cudnn.input_grads["qr"] is None
    torch.testing.assert_close(
        cudnn_loss["loss"].double(), pytorch_loss["loss"].double(), rtol=1e-3, atol=1e-7
    )
    _assert_indexer_trained(layer, cudnn)
    cudnn_call, pytorch_call, _pytorch_again_call = indexer_calls
    for name in ("q", "weights", "k", "gate_score"):
        _assert_similar(
            cudnn_call.grads[name], pytorch_call.grads[name], label=f"d{name}", eps=1e-2
        )
    assert set(cudnn.param_grads) == set(pytorch.param_grads)
    for name, grad in cudnn.param_grads.items():
        _assert_similar(grad, pytorch.param_grads[name], label=name, eps=1e-2)


@pytest.mark.parametrize("per_token", [False, True], ids=["mean", "per-token"])
def test_cudnn_path_matches_the_reference(monkeypatch, per_token):
    """The cuDNN path's loss and indexer gradients against the independent reference, with
    two samples per batch (the backward addresses pooled keys by batch-global ids)."""
    _require_cudnn_loss()
    seen = _install_recorders(monkeypatch)
    counts = _count_cudnn_loss(monkeypatch)
    config = _cudnn_layer_config(per_token=per_token)
    layer = _build_layer(config, _single_rank_groups())
    _forbid_token_level_loss_hooks(monkeypatch)
    indexer_calls = _spy_indexer(layer)
    inputs = _make_inputs(config, _ROWS, batch=2, seed=37)

    run = _run_layer(layer, inputs)
    assert (counts["cudnn"], seen.loss_function_calls) == (1, 0)
    (record,) = _layer_losses(seen)
    (indexer_call,) = indexer_calls
    _assert_indexer_trained(layer, run)
    _check_reference(layer, inputs, run, indexer_call, record, cu_seqlens=None, eps=1e-2)


# ---------------------------------------------------------------------------------------
# The dispatch matrix
# ---------------------------------------------------------------------------------------


def _expected_reason(*, scorer, hf_scoring, packed, tp_size):
    """Words of why the PyTorch loss runs, or None where the cuDNN path runs."""
    if tp_size > 1:
        return f"tensor parallelism (TP size {tp_size})"
    if scorer != "cudnn":
        return "dsa_indexer_kpool_scorer_backend is 'torch', not 'cudnn'"
    if not hf_scoring:
        return "legacy (non-HF) KPool scoring"
    if not _frontend_runs_the_cudnn_loss():
        return "is older than 1.31"
    if packed:
        return "the cuDNN KPool scorer declined this forward"
    return None


def _layout(layout):
    packed = layout == "packed"
    return packed, (_PACKED_CU if packed else None), (1 if packed else 2)


@pytest.mark.parametrize("layout", ["plain", "packed"])
@pytest.mark.parametrize("hf_scoring", [True, False], ids=["hf", "legacy"])
@pytest.mark.parametrize("scorer", ["torch", "cudnn"])
def test_tp1_dispatch_matches_the_reference_and_logs_the_path(
    monkeypatch, caplog, scorer, hf_scoring, layout
):
    """At TP1 every combination trains to the reference; the log line names the cuDNN path
    only for the cuDNN scorer with HF scoring on a layout it accepts."""
    _require_cudnn_dsa()
    packed, cu_seqlens, batch = _layout(layout)
    seen = _install_recorders(monkeypatch)
    counts = _count_cudnn_loss(monkeypatch)
    config = _cudnn_layer_config(scorer=scorer, hf_scoring=hf_scoring)
    layer = _build_layer(config, _single_rank_groups())
    _forbid_token_level_loss_hooks(monkeypatch)
    indexer_calls = _spy_indexer(layer)
    inputs = _make_inputs(config, _ROWS, batch=batch, seed=41)
    reason = _expected_reason(scorer=scorer, hf_scoring=hf_scoring, packed=packed, tp_size=1)

    with caplog.at_level(logging.INFO, logger=_DSA_LOGGER):
        run = _run_layer(
            layer, inputs, packed_seq_params=_packed_seq_params(cu_seqlens) if packed else None
        )
    assert (counts["cudnn"], seen.loss_function_calls) == ((1, 0) if reason is None else (0, 1))
    (line,) = _path_lines(caplog)
    if reason is None:
        assert _CUDNN_PATH in line
    else:
        assert _PYTORCH_PATH in line and reason in line, line
    (record,) = _layer_losses(seen)
    (indexer_call,) = indexer_calls
    _assert_indexer_trained(layer, run)
    _check_reference(
        layer,
        inputs,
        run,
        indexer_call,
        record,
        cu_seqlens=cu_seqlens,
        eps=1e-2 if reason is None else 1e-3,
    )


@pytest.mark.parametrize("layout", ["plain", "packed"])
@pytest.mark.parametrize("hf_scoring", [True, False], ids=["hf", "legacy"])
@pytest.mark.parametrize("scorer", ["torch", "cudnn"])
def test_tp2_dispatch_runs_the_pytorch_loss_and_matches_tp1(
    monkeypatch, caplog, scorer, hf_scoring, layout
):
    """At TP2 the PyTorch loss runs for every combination, and says so: a rank's cuDNN top-k
    values belong to its own selection, not to the TP rank 0 selection every rank attends.
    The TP2 loss is the TP1 layer's (which runs the cuDNN path where it applies) and every
    rank's indexer gradients are the TP1 gradients."""
    tp_size = 2
    _require_ranks(tp_size)
    _require_cudnn_dsa()
    packed, cu_seqlens, batch = _layout(layout)
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        model_parallel_cuda_manual_seed(1234)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        tp_group = pg_collection.tp
        seen = _install_recorders(monkeypatch)
        counts = _count_cudnn_loss(monkeypatch)
        _forbid_token_level_loss_hooks(monkeypatch)
        packed_seq_params = _packed_seq_params(cu_seqlens) if packed else None

        tp1_config = _cudnn_layer_config(scorer=scorer, hf_scoring=hf_scoring)
        tp1_layer = _build_layer(tp1_config, _single_rank_groups())
        inputs = _make_inputs(tp1_config, _ROWS, batch=batch, seed=43)
        tp1 = _run_layer(tp1_layer, inputs, packed_seq_params=packed_seq_params)
        (tp1_loss,) = _layer_losses(seen)
        tp1_reason = _expected_reason(
            scorer=scorer, hf_scoring=hf_scoring, packed=packed, tp_size=1
        )
        assert counts["cudnn"] == (1 if tp1_reason is None else 0)

        seen.losses.clear()
        counts["cudnn"] = 0
        phase1_calls = seen.loss_function_calls
        caplog.clear()
        layer = _build_layer(
            _cudnn_layer_config(scorer=scorer, hf_scoring=hf_scoring, tp=tp_size), pg_collection
        )
        with caplog.at_level(logging.INFO, logger=_DSA_LOGGER):
            run = _run_layer(
                layer,
                _slice_heads(inputs, tp_group.rank(), tp_size),
                packed_seq_params=packed_seq_params,
            )
        (loss,) = _layer_losses(seen)
        lines = _path_lines(caplog)
        all_losses = _all_gather(loss["loss"].float().reshape(1), tp_group)
    finally:
        Utils.destroy_model_parallel()

    assert counts["cudnn"] == 0 and seen.loss_function_calls == phase1_calls + 1
    assert len(lines) == 1 and _PYTORCH_PATH in lines[0], lines
    assert f"tensor parallelism (TP size {tp_size})" in lines[0], lines
    assert torch.equal(all_losses[0], all_losses[1]), f"TP ranks log {all_losses}"
    assert torch.equal(run.published, tp1.published), "TP rank 0 does not select as TP1"
    torch.testing.assert_close(
        loss["loss"].double(), tp1_loss["loss"].double(), rtol=1e-3, atol=1e-7
    )
    _assert_indexer_trained(layer, run)
    eps = 1e-2 if tp1_reason is None else 1e-3
    for name, grad in run.param_grads.items():
        _assert_similar(grad, tp1.param_grads[name], label=f"TP{tp_size} {name}", eps=eps)
