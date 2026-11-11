# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KPool selection inside ``DSAttention.forward``, on CPU.

The CP reorder. Under all-gather CP, KPool selection sees the indexer K and the KPool gate in
one row order: DSAttention gathers the indexer K and, for KPool, the gate score over the CP
group, then reorders both into full-sequence order, the key through ``_permute_rows``
(inverse-permutation backward) and the gate through ``index_select``. The KPool selector
compresses keys with the gate row by row, so both must arrive with the global length and the
same order. These tests fake the CP gather, the indexer projections and the KPool kernel.

The cuDNN KPool scorer through the real dispatch. ``_real_kpool_layers`` runs the forward
with the real ``DSAIndexer``, ``fused_qk_topk_kpool``, ``dsa_kernels.run_kpool_pool_topk`` and
``dsa_cudnn_kernels.run_kpool_pool_topk`` on a stand-in cuDNN frontend
(``dsa_cudnn_fe_fakes``), so a keyword, a forwarded argument or an input dtype that the
backend does not take fails here; the real kernel's arithmetic is the GPU test's
(``test_kpool_cudnn_selection_gpu.py``).
"""

import functools
import inspect
import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.core import utils as core_utils
from megatron.core.models.common.embeddings import rotary_pos_embedding
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_kernels,
    dsa_layout,
)
from megatron.core.transformer.forward_sharing import forward_sharing_lifetime
from tests.unit_tests.transformer.experimental_attention_variant import dsa_cudnn_fe_fakes as fakes
from tests.unit_tests.transformer.experimental_attention_variant import kpool_selection_utils as ks
from tests.unit_tests.transformer.experimental_attention_variant.test_mtp_dsa_cross_depth_sharing import (
    _make_config as make_small_dsa_config,
)

LOCAL_ROWS = 8
# The real-dispatch tests: local query rows, and a top-k of 16 pools per row, above the
# cuDNN top-k's small-K guard (K < 14 on BF16 scores), which has its own tests.
REAL_ROWS = 128
REAL_TOPK = 64
_FUSED_DSA_HOOKS = (
    "run_fused_dsa_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_loss",
    "run_fused_qk_topk_with_scores",
    "run_fused_absorbed_sparse_attention",
)


class _FakeCPGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _kpool_config(monkeypatch, **overrides):
    # Config validation only; neither a CUDA device nor the kernel dependencies are needed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: [])
    kwargs = dict(
        dsa_kernel_backend="cudnn",
        attention_backend=AttnBackend.auto,
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=8,
        dsa_indexer_kpool=4,
        dsa_indexer_topk=4,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=True,
        use_cpu_initialization=True,
    )
    kwargs.update(overrides)
    return make_small_dsa_config(**kwargs)


def _patch_cpu_forward(monkeypatch, cp_size, cp_rank):
    """Let DSAttention.forward run on CPU as one rank of a fake CP group; returns the group.

    The fused attention kernels decline, so the PyTorch reference attention runs.
    """
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    for name in _FUSED_DSA_HOOKS:
        if hasattr(dsa_kernels, name):
            monkeypatch.setattr(dsa_kernels, name, lambda *a, **k: None)
    monkeypatch.setattr(dsa_module, "_validate_nonpacked_cp_uniform_length", lambda **_k: None)

    def pg_size(g=None):
        return g.size() if g is not None else 1

    monkeypatch.setattr(dsa_module, "get_pg_size", pg_size)
    monkeypatch.setattr(dsa_layout, "get_pg_size", pg_size)
    return _FakeCPGroup(cp_size, cp_rank)


def _run_dsattention(monkeypatch, config, cp_size, cp_rank, training, fake_kpool):
    """Run DSAttention.forward on CPU with faked CP gather, indexer projections and KPool."""
    real_indexer_cls = dsa_module.DSAIndexer

    def fake_build(spec, *args, **kwargs):
        if spec is real_indexer_cls:
            kwargs.setdefault("submodules", dsa_module.DSAIndexerSubmodules())
            return real_indexer_cls(*args, **kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(dsa_module, "build_module", fake_build)
    group = _patch_cpu_forward(monkeypatch, cp_size, cp_rank)

    def fake_gather(tensor, group):
        # Rows of CP rank j carry the id j * 1000 + local_row in feature 0.
        pieces = []
        for j in range(group.size()):
            if j == group.rank():
                pieces.append(tensor)
            else:
                piece = torch.zeros_like(tensor)
                ids = j * 1000 + torch.arange(tensor.size(0), dtype=tensor.dtype)
                piece[..., 0] = ids.view(-1, *([1] * (tensor.dim() - 2))).expand_as(piece[..., 0])
                pieces.append(piece)
        return torch.cat(pieces, dim=0)

    monkeypatch.setattr(dsa_module, "gather_from_sequence_parallel_region", fake_gather)
    monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", fake_kpool)

    attention = dsa_module.DSAttention(
        config=config,
        submodules=dsa_module.DSAttentionSubmodules(indexer=real_indexer_cls),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        pg_collection=SimpleNamespace(tp=None, cp=group),
    )
    attention.cp_comm_type = "allgather"

    def fake_before_topk(x, _qr, _packed_seq_params):
        rows, batch = x.shape[:2]
        ids = (cp_rank * 1000 + torch.arange(rows, dtype=torch.float32)).view(rows, 1)
        q = torch.randn(rows, batch, 4, 8)
        k = torch.zeros(rows, batch, 8)
        k[..., 0] = ids
        gate = torch.zeros(rows, batch, 8)
        gate[..., 0] = ids
        attention.indexer._kpool_gate_score = gate
        return q, k, torch.ones(rows, batch, 4)

    attention.indexer.forward_before_topk = fake_before_topk
    attention.train(training)
    heads = config.num_attention_heads
    absorbed = config.kv_lora_rank + config.qk_pos_emb_head_dim
    with forward_sharing_lifetime(None, None, config):
        attention(
            query=torch.randn(LOCAL_ROWS, 1, heads, absorbed),
            key=torch.randn(LOCAL_ROWS, 1, 1, absorbed),
            value=None,
            attention_mask=None,
            x=torch.randn(LOCAL_ROWS, 1, config.hidden_size),
            qr=torch.randn(LOCAL_ROWS, 1, config.q_lora_rank),
            up_v_weight=torch.randn(heads, config.v_head_dim, config.kv_lora_rank),
            attn_mask_type=AttnMaskType.causal,
        )


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(2, 1), (4, 2)])
@pytest.mark.parametrize("training", [True, False])
def test_kpool_gate_and_indexer_k_share_the_cp_reorder(monkeypatch, cp_size, cp_rank, training):
    config = _kpool_config(monkeypatch)
    seen = {}

    def fake_kpool(q, k, weights, topk, kpool, gate, ape, **kwargs):
        seen.update(k=k.clone(), gate=gate.clone(), grad=torch.is_grad_enabled())
        batch, sq = q.size(1), q.size(0)
        return None, torch.arange(topk).expand(batch, sq, topk).contiguous()

    _run_dsattention(monkeypatch, config, cp_size, cp_rank, training, fake_kpool)

    k_ids, gate_ids = seen["k"][:, 0, 0], seen["gate"][:, 0, 0]
    # Gathered to the global length and reordered: a real permutation of the gather order.
    assert k_ids.numel() == LOCAL_ROWS * cp_size
    assert not torch.equal(k_ids, torch.sort(k_ids).values)
    assert torch.equal(torch.sort(k_ids).values, torch.sort(gate_ids).values)
    # The KPool gate follows the indexer K row for row.
    assert torch.equal(k_ids, gate_ids)
    # KPool selection is discrete and runs without an autograd graph.
    assert seen["grad"] is False


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(2, 1), (4, 0), (4, 2)])
def test_cudnn_kpool_scorer_receives_the_zigzag_query_runs(monkeypatch, cp_size, cp_rank):
    """DSAttention hands the cuDNN KPool selector this CP rank's zigzag positions."""
    config = _kpool_config(
        monkeypatch, dsa_indexer_kpool_scorer_backend="cudnn", dsa_indexer_scoring_relu=True
    )
    hook_calls = []

    def spy_hook(cfg, q, k_pooled, weights, pool_topk, pool_size, query_runs, hf_scoring, **kw):
        hook_calls.append(dict(config=cfg, query_runs=query_runs, pool_topk=pool_topk, **kw))
        return None  # decline: the PyTorch scorer would run

    monkeypatch.setattr(dsa_kernels, "run_kpool_pool_topk", spy_hook)

    def fake_kpool(q, k, weights, topk, kpool, gate, ape, **kwargs):
        selector = kwargs["pool_topk_fn"]
        assert selector is not None
        k_pooled = torch.zeros(k.size(0) // kpool, k.size(1), k.size(2), dtype=torch.bfloat16)
        assert selector(q, k_pooled, weights, topk // kpool, kpool, kwargs["hf_scoring"]) is None
        batch, sq = q.size(1), q.size(0)
        return None, torch.arange(topk).expand(batch, sq, topk).contiguous()

    _run_dsattention(monkeypatch, config, cp_size, cp_rank, False, fake_kpool)
    chunk = LOCAL_ROWS // 2  # the sequence is LOCAL_ROWS * cp_size tokens in 2 * cp_size chunks
    assert len(hook_calls) == 1
    assert hook_calls[0]["config"] is config
    assert hook_calls[0]["pool_topk"] == config.dsa_indexer_topk // config.dsa_indexer_kpool
    assert hook_calls[0]["query_runs"] == (
        (0, chunk, cp_rank * chunk),
        (chunk, LOCAL_ROWS, (2 * cp_size - cp_rank - 1) * chunk),
    )


def test_torch_kpool_scorer_passes_no_selector(monkeypatch):
    config = _kpool_config(monkeypatch)
    seen = {}

    def fake_kpool(q, k, weights, topk, kpool, gate, ape, **kwargs):
        seen["selector"] = kwargs["pool_topk_fn"]
        batch, sq = q.size(1), q.size(0)
        return None, torch.arange(topk).expand(batch, sq, topk).contiguous()

    _run_dsattention(monkeypatch, config, 2, 1, False, fake_kpool)
    assert seen["selector"] is None


class _BF16Linear(torch.nn.Linear):
    """Stands in for TELinear: parameters in ``params_dtype`` (BF16), ``(output, bias)``."""

    def __init__(self, input_size, output_size, *, config, **_unused):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)

    def forward(self, x):
        return super().forward(x), None


class _BF16LayerNorm(torch.nn.LayerNorm):
    """Stands in for TENorm as the indexer's key norm, in ``params_dtype`` (BF16)."""

    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps, dtype=config.params_dtype)


def _host_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_group):
    """``get_pos_emb_on_this_cp_rank`` without its pinned-memory, CUDA index tensor."""
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    chunks = pos_emb.view(*pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[seq_dim + 1 :])
    chunks = chunks.index_select(seq_dim, torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1]))
    return chunks.view(*chunks.shape[:seq_dim], -1, *chunks.shape[seq_dim + 2 :])


def _real_kpool_config(
    monkeypatch, scorer, cp_size, *, deterministic=False, hf_scoring=False, **overrides
):
    # The HF-scoring cases also keep the key dtype in the pooled keys (the other cases re-round
    # them to BF16), so a key the forward hands over in another dtype reaches the backend.
    return _kpool_config(
        monkeypatch,
        dsa_indexer_kpool_scorer_backend=scorer,
        dsa_indexer_scoring_relu=True,
        dsa_indexer_n_heads=8,
        dsa_indexer_topk=REAL_TOPK,
        dsa_indexer_hf_scoring=hf_scoring,
        dsa_indexer_kpool_key_dtype_compression=hf_scoring,
        deterministic_mode=deterministic,
        context_parallel_size=cp_size,
        cp_comm_type="allgather",
        **overrides,
    )


def _real_kpool_layers(monkeypatch, cp_size, cp_rank, tp_group=None):
    """Prepare CPU runs of the real ``DSAttention.forward`` with the real KPool selection.

    Returns ``(run, seen)``. ``run(config, position_ids=None)`` builds a layer with seeded
    parameters, runs its forward on seeded BF16 inputs and returns the attention output and
    the top-k selection the layer published; two calls differ only by ``config``. ``seen``
    records the calls of ``fused_qk_topk_kpool`` (arguments), of
    ``dsa_kernels.run_kpool_pool_topk`` (results), of
    ``dsa_cudnn_kernels.run_kpool_pool_topk`` (bound arguments) and of the PyTorch scorer
    ``_kpool_topk_from_score_chunks`` (one entry per call). The four are wrapped, not
    replaced. ``tp_group`` is the layer's TP group, for a TP sequence-parallel local query.

    The indexer is the real ``DSAIndexer``: only its Transformer Engine projections and key
    norm are BF16 torch modules, so q, the keys, the gate and the head weights reach the
    selector in the dtypes and shapes production hands it. What else is replaced needs CUDA
    or a process group: the cuDNN frontend (``fakes.Fe131DSA``), the CP slice of the rotary
    embedding (its index goes through pinned memory to CUDA) and the CP and TP gathers.
    Under CP, only this rank's rows of the gathered keys and gate are the indexer's; the
    other ranks' rows are seeded noise in the same dtype. The same holds for the rows
    gathered over a TP group.
    """
    group = _patch_cpu_forward(monkeypatch, cp_size, cp_rank)
    real_indexer_cls = dsa_module.DSAIndexer
    indexer_submodules = dsa_module.DSAIndexerSubmodules(
        linear_wq_b=_BF16Linear,
        linear_wk=_BF16Linear,
        k_norm=_BF16LayerNorm,
        linear_weights_proj=_BF16Linear,
    )

    def build(spec, *args, **kwargs):
        if spec is real_indexer_cls:
            kwargs.setdefault("submodules", indexer_submodules)
            return real_indexer_cls(*args, **kwargs)
        if spec in (_BF16Linear, _BF16LayerNorm):
            return spec(*args, **kwargs)
        return torch.nn.Identity()

    gathers = []

    def gather(tensor, group):
        gathers.append(tensor.shape)
        pieces = []
        for rank in range(group.size()):
            if rank == group.rank():
                pieces.append(tensor)
            else:
                generator = torch.Generator().manual_seed(1000 * rank + len(gathers))
                pieces.append(torch.randn(tensor.shape, generator=generator).to(tensor.dtype))
        return torch.cat(pieces, dim=0)

    monkeypatch.setattr(dsa_module, "build_module", build)
    monkeypatch.setattr(dsa_module, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(
        rotary_pos_embedding, "get_pos_emb_on_this_cp_rank", _host_pos_emb_on_this_cp_rank
    )
    fakes.Fe131DSA.reset()
    monkeypatch.setattr(dsa_cudnn_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", fakes.Fe131DSA)
    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_frontend_version", lambda: (1, 31))
    # The backend's once-only log states: start clean, restored on teardown.
    monkeypatch.setattr(dsa_cudnn_kernels, "_KPOOL_DECLINES_LOGGED", set())
    monkeypatch.setattr(dsa_cudnn_kernels, "_KPOOL_FUSED_SELECTION_LOGGED", False)
    monkeypatch.setattr(dsa_cudnn_kernels, "_DETERMINISTIC_TOPK_PATHS_LOGGED", set())

    seen = SimpleNamespace(kpool=[], dispatch=[], backend=[], torch_scorer=[])
    real_kpool = dsa_module.fused_qk_topk_kpool
    real_dispatch = dsa_kernels.run_kpool_pool_topk
    real_backend = dsa_cudnn_kernels.run_kpool_pool_topk
    real_torch_scorer = dsa_module._kpool_topk_from_score_chunks

    def kpool(*args, **kwargs):
        seen.kpool.append((args, kwargs))
        return real_kpool(*args, **kwargs)

    def dispatch(*args, **kwargs):
        result = real_dispatch(*args, **kwargs)
        seen.dispatch.append(result)
        return result

    # The dispatch reads the backend's signature to decide whether to forward ``deterministic``
    # and ``config``: functools.wraps keeps it. Binding the call fails on a keyword the
    # backend does not have and shows a keyword the dispatch dropped as its default.
    @functools.wraps(real_backend)
    def backend(*args, **kwargs):
        bound = inspect.signature(real_backend).bind(*args, **kwargs)
        bound.apply_defaults()
        seen.backend.append(dict(bound.arguments))
        return real_backend(*args, **kwargs)

    def torch_scorer(*args, **kwargs):
        seen.torch_scorer.append(kwargs["index_topk"])
        return real_torch_scorer(*args, **kwargs)

    monkeypatch.setattr(dsa_module, "fused_qk_topk_kpool", kpool)
    monkeypatch.setattr(dsa_module, "_kpool_topk_from_score_chunks", torch_scorer)
    monkeypatch.setattr(dsa_kernels, "run_kpool_pool_topk", dispatch)
    monkeypatch.setattr(dsa_cudnn_kernels, "run_kpool_pool_topk", backend)
    # The signature cache keeps every callable it has seen; do not leave the spy in it.
    monkeypatch.setattr(core_utils, "_CALLABLE_KWARG_SIGNATURE_CACHE", {})

    def run(config, position_ids=None):
        attention = dsa_module.DSAttention(
            config=config,
            submodules=dsa_module.DSAttentionSubmodules(indexer=real_indexer_cls),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=SimpleNamespace(tp=tp_group, cp=group),
        )
        attention.cp_comm_type = "allgather"
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.copy_(0.1 * torch.randn(parameter.shape, generator=generator))
        attention.eval()
        heads = config.num_attention_heads
        absorbed = config.kv_lora_rank + config.qk_pos_emb_head_dim
        generator = torch.Generator().manual_seed(1)

        def randn(*shape):
            return torch.randn(shape, generator=generator).to(torch.bfloat16)

        gathers.clear()
        with forward_sharing_lifetime(None, None, config):
            output = attention(
                query=randn(REAL_ROWS, 1, heads, absorbed),
                key=randn(REAL_ROWS, 1, 1, absorbed),
                value=None,
                attention_mask=None,
                x=randn(REAL_ROWS, 1, config.hidden_size),
                qr=randn(REAL_ROWS, 1, config.q_lora_rank),
                up_v_weight=randn(heads, config.v_head_dim, config.kv_lora_rank),
                attn_mask_type=AttnMaskType.causal,
                position_ids=position_ids,
            )
            published = attention._get_dsa_index_sharing_payload(None, None).topk_by_layer[1]
        return output, published

    return run, seen


def _describe(call):
    return {
        name: (str(value.dtype), tuple(value.shape))
        for name, value in call.items()
        if isinstance(value, torch.Tensor)
    }


@pytest.mark.parametrize("hf_scoring", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(("cp_size", "cp_rank"), [(1, 0), (2, 1), (4, 0), (4, 2)])
def test_cudnn_kpool_scorer_takes_the_real_forward_call(
    monkeypatch, cp_size, cp_rank, deterministic, hf_scoring
):
    """The real forward reaches the cuDNN KPool backend through the real dispatch.

    The backend takes the call for the tensors the forward and the real indexer produce,
    with the layer's config and its deterministic flag, and the layer publishes the selection
    of the same layer with the ``'torch'`` scorer up to pools tying at the boundary.
    """
    # Deterministic selection follows the config alone.
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    run, seen = _real_kpool_layers(monkeypatch, cp_size, cp_rank)
    configs = {
        scorer: _real_kpool_config(
            monkeypatch, scorer, cp_size, deterministic=deterministic, hf_scoring=hf_scoring
        )
        for scorer in ("torch", "cudnn")
    }

    reference_output, reference = run(configs["torch"])
    # The 'torch' scorer builds no selector: the dispatch is not reached.
    assert len(seen.kpool) == 1 and not seen.dispatch and not seen.backend
    assert len(seen.torch_scorer) == 1
    candidate_output, candidate = run(configs["cudnn"])

    # One selection, and the backend took it: the dispatch returned pool ids, not None.
    assert len(seen.kpool) == 2 and len(seen.dispatch) == 1 and len(seen.backend) == 1
    call = seen.backend[0]
    pools = seen.dispatch[0]
    assert pools is not None, f"the cuDNN KPool backend declined {_describe(call)}"
    # The layer kept them: the PyTorch scorer did not run again to select in their place.
    assert len(seen.torch_scorer) == 1, "the PyTorch scorer selected after the backend did"
    assert tuple(pools.shape) == (1, REAL_ROWS, REAL_TOPK // ks.POOL)
    assert fakes.Fe131DSA.calls, "the stand-in indexer kernel did not score"
    # The dispatch forwards the layer's config and the deterministic flag of that config.
    assert call["config"] is configs["cudnn"]
    assert call["deterministic"] is deterministic
    # The indexer's own projections, as production hands them over.
    assert call["q"].dtype == torch.bfloat16 and call["k_pooled"].dtype == torch.bfloat16
    assert call["weights"].dtype == (torch.float32 if hf_scoring else torch.bfloat16)
    assert tuple(call["q"].shape) == (REAL_ROWS, 1, 8, 8)
    assert tuple(call["k_pooled"].shape) == (REAL_ROWS * cp_size // ks.POOL, 1, 8)
    assert call["hf_scoring"] is hf_scoring and call["use_relu"] is True
    chunk = REAL_ROWS // 2
    assert call["query_runs"] == (
        ((0, REAL_ROWS, 0),)
        if cp_size == 1
        else ((0, chunk, cp_rank * chunk), (chunk, REAL_ROWS, (2 * cp_size - cp_rank - 1) * chunk))
    )

    # Both layers selected from the same indexer tensors.
    (reference_args, _), (args, kwargs) = seen.kpool
    q, k, weights, _topk, _pool, gate, ape = args
    for mine, theirs in zip((q, k, weights, gate), reference_args[:3] + reference_args[5:6]):
        assert torch.equal(mine, theirs)
    # The published selections agree up to pools whose FP64 scores tie at a row's boundary.
    # Without HF scoring the weights carry head_dim**-0.5, which the HF formula of the check
    # applies to the logits; ReLU is positively homogeneous, so the scores are the same.
    weights_hf = weights.double() * (1.0 if hf_scoring else q.size(-1) ** 0.5)
    scoring = SimpleNamespace(
        q=q, k=k, gate=gate, ape=ape, weights_hf=weights_hf, ends=kwargs["varlen_ends"]
    )
    differing = ks.assert_same_selection_up_to_boundary_ties(
        scoring,
        reference,
        candidate,
        topk=REAL_TOPK,
        rel_tol=1e-5,
        kpool_key_dtype_compression=kwargs["kpool_key_dtype_compression"],
    )
    assert differing <= REAL_ROWS // 16
    # The slot order may differ (canonical order in deterministic mode, tied pools
    # otherwise); the reference attention scatters the selection into a mask, so rows with
    # the same pools must give bitwise-equal output.
    same = (reference.sort(dim=-1).values == candidate.sort(dim=-1).values).all(dim=-1)[0]
    assert int(same.sum()) == REAL_ROWS - differing
    assert torch.equal(reference_output[same], candidate_output[same])


@pytest.mark.parametrize(
    ("cp_size", "cp_rank", "tp_rank", "declines"),
    [(1, 0, None, False), (2, 1, None, True), (1, 0, 1, True)],
)
def test_position_ids_keep_the_torch_kpool_scorer_only_as_query_positions(
    monkeypatch, caplog, cp_size, cp_rank, tp_rank, declines
):
    """``position_ids`` keep the PyTorch scorer only where the forward reads positions from them.

    That is under context parallelism and for a TP sequence-parallel local query (the case
    with a ``tp_rank``: this rank's rows of a sequence split over two TP ranks). Without
    either they are not read: the selector is built and the backend takes the call.
    """
    monkeypatch.setattr(dsa_module, "_KPOOL_SCORER_FALLBACK_REASONS_WARNED", set())
    if tp_rank is None:
        tp_group, overrides = None, {}
        positions, _ = dsa_layout.get_cp_positions_from_layout(
            sq=REAL_ROWS,
            skv=REAL_ROWS * cp_size,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_comm_type="allgather",
            device="cpu",
        )
    else:
        tp_group = _FakeCPGroup(2, tp_rank)
        overrides = dict(tensor_model_parallel_size=2, sequence_parallel=True)
        positions = torch.arange(tp_rank * REAL_ROWS, (tp_rank + 1) * REAL_ROWS)
    run, seen = _real_kpool_layers(monkeypatch, cp_size, cp_rank, tp_group=tp_group)
    config = _real_kpool_config(monkeypatch, "cudnn", cp_size, **overrides)
    with caplog.at_level(logging.WARNING):
        run(config, position_ids=positions.unsqueeze(0))
    assert len(seen.kpool) == 1
    if declines:
        assert not seen.dispatch and not seen.backend
        assert len(seen.torch_scorer) == 1
        assert "the query positions come from position_ids" in caplog.text
    else:
        assert len(seen.dispatch) == 1 and seen.dispatch[0] is not None
        assert not seen.torch_scorer
        assert "position_ids" not in caplog.text
