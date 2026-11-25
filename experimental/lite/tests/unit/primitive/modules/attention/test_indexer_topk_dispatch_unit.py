# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dispatch of the DSA indexer top-k to optional indexer top-k bindings (CPU).

The modules run their real forward code on the CPU; the CUDA kernels around the selection are
replaced by recorders. A module without a binding, in training mode (also a training forward
without autograd, as Lite's reentrant activation recompute runs it) or with autograd enabled,
must call the upstream selector exactly as before: the same statement (compared as an AST with
the upstream text) with the same arguments. A bound module in eval mode with gradients disabled
must hand the binding the upstream selector's operands and the layout of its query rows, and
feed the binding's result into the unchanged code that follows.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.mlite

LITE_ROOT = Path(__file__).resolve().parents[5]
REPO_ROOT = LITE_ROOT.parents[1]

# The upstream selection statements, verbatim.
_UPSTREAM_DSA_FULL = """
topk_indices, _ = _dsa_kernels.indexer_topk(
    q_indexer,
    k_indexer,
    weights_indexer,
    effective_indexer_topk,
    1,
    indexer_softmax_scale=self.indexer_softmax_scale,
)
"""


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


def _assignments_calling(function, callee: str) -> list[str]:
    """``ast.dump`` of every assignment in ``function`` whose value calls ``callee``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return [
        ast.dump(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == callee
    ]


def _dump(statement: str) -> str:
    return ast.dump(ast.parse(textwrap.dedent(statement)).body[0])


class FakeBinding:
    """Records what a module asks of its binding; ``select`` returns ``result(call)``."""

    def __init__(self, result=None, *, required=False):
        self.result = result
        self.required = required
        self.selects: list[dict] = []
        self.declines: list[str] = []

    def active(self) -> bool:
        return not torch.is_grad_enabled()

    def decline(self, reason: str) -> None:
        self.declines.append(reason)
        if self.required:
            raise RuntimeError(f"required binding declined: {reason}")

    def select(self, q, k, weights, *, layout, topk, softmax_scale):
        call = dict(
            q=q, k=k, weights=weights, layout=layout, topk=topk, softmax_scale=softmax_scale
        )
        self.selects.append(call)
        return None if self.result is None else self.result(call)


def _ids(call: dict) -> torch.Tensor:
    """A distinguishable selection: row r holds r, r + 1, ... and -1 in the last column."""
    rows, topk = call["layout"].rows, call["topk"]
    ids = torch.arange(rows, dtype=torch.int32)[:, None] + torch.arange(topk, dtype=torch.int32)
    ids[:, -1] = -1
    return ids


def _real_binding(geometry):
    from megatron.lite.primitive.kernels.indexer_topk import IndexerTopKConfig
    from megatron.lite.primitive.modules.attention.indexer_topk import IndexerTopKBinding

    return IndexerTopKBinding(
        name="layer",
        geometry=geometry,
        config=IndexerTopKConfig(backend="reference", precision="fast"),
        fmt="fp8",
        tuning=None,
        kernel=None,
        plugin=None,
        route=None,
    )


# ---------------------------------------------------------------------------
# DSA: _forward_dense_full (non-CP prompts, packed prompts, legacy gather-all CP)
# ---------------------------------------------------------------------------

DSA_KWARGS = dict(
    hidden_size=32,
    num_attention_heads=2,
    q_lora_rank=16,
    kv_lora_rank=8,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=8,
    index_n_heads=4,
    index_head_dim=16,
    index_topk=4,
    rms_norm_eps=1e-5,
)
# IndexShare with every second layer shared: layer 1 selects, layer 2 reuses its top-k.
INDEX_SHARE = {
    "full": dict(layer_number=1, index_topk_freq=2, index_skip_topk_offset=1),
    "shared": dict(layer_number=2, index_topk_freq=2, index_skip_topk_offset=1),
}


def _dsa(monkeypatch, **overrides):
    """A real DynamicSparseAttention on the CPU (torch RMSNorm) and its module."""
    from megatron.lite.primitive.modules.attention import dsa as dsa_module

    monkeypatch.setattr(dsa_module, "RMSNorm", lambda size, eps: nn.RMSNorm(size, eps=eps))
    torch.manual_seed(0)
    attention = dsa_module.DynamicSparseAttention(**{**DSA_KWARGS, **overrides}).eval()
    return dsa_module, attention


class DsaRecorder:
    """Replaces the DSA kernels around the selection and records their inputs."""

    def __init__(self, monkeypatch, dsa_module, attention):
        self.indexer_topk: list[tuple] = []
        self.flat: list[torch.Tensor] = []
        self.projections: list[tuple] = []
        kernels = dsa_module._dsa_kernels
        build_flat = kernels.build_flat_topk_idxs

        def indexer_topk(*args, **kwargs):
            self.indexer_topk.append((args, kwargs))
            q = args[0]
            ids = torch.arange(args[3], dtype=torch.int32).expand(q.shape[1], q.shape[0], -1)
            return ids.contiguous(), torch.full(q.shape[:2], args[3], dtype=torch.int32)

        def flat_topk(topk_indices, **kwargs):
            self.flat.append(topk_indices)
            return build_flat(topk_indices, **kwargs)

        def sparse_attn(query, kv, sink, idxs, scale, topk_length=None, value_dim=None):
            return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

        project = attention.indexer.forward_before_topk

        def forward_before_topk(*args, **kwargs):
            self.projections.append(project(*args, **kwargs))
            return self.projections[-1]

        monkeypatch.setattr(kernels, "indexer_topk", indexer_topk)
        monkeypatch.setattr(kernels, "build_flat_topk_idxs", flat_topk)
        monkeypatch.setattr(kernels, "dsa_sparse_attn", sparse_attn)
        monkeypatch.setattr(attention.indexer, "forward_before_topk", forward_before_topk)


def _dsa_inputs(attention, *, batch: int = 1, seq: int = 6):
    from megatron.lite.primitive.modules.attention.dsa import build_rope_cache

    generator = torch.Generator().manual_seed(1)
    x = torch.randn(batch, seq, DSA_KWARGS["hidden_size"], generator=generator)
    cos, sin = build_rope_cache(
        dim=attention.qk_rope_head_dim, max_position_embeddings=seq, rope_theta=10000.0
    )
    return x, cos, sin, torch.arange(seq).unsqueeze(0)


def test_default_dsa_calls_upstream_indexer_topk_verbatim(monkeypatch):
    dsa_module, attention = _dsa(monkeypatch)
    assert _assignments_calling(
        dsa_module.DynamicSparseAttention._forward_dense_full, "_dsa_kernels.indexer_topk"
    ) == [_dump(_UPSTREAM_DSA_FULL)]
    assert attention._indexer_topk is None
    recorder = DsaRecorder(monkeypatch, dsa_module, attention)
    for seq in (6, 3):  # min(index_topk, seq) is the upstream top-k
        with torch.no_grad():
            attention._forward_dense_full(*_dsa_inputs(attention, seq=seq))
        q, k, weights = recorder.projections[-1]
        args, kwargs = recorder.indexer_topk[-1]
        assert args[0] is q and args[1] is k and args[2] is weights
        assert args[3:] == (min(4, seq), 1)
        assert kwargs == {"indexer_softmax_scale": attention.indexer_softmax_scale}
        assert recorder.flat[-1].shape == (1, seq, min(4, seq))


def test_dsa_binding_receives_full_layout(monkeypatch):
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout
    from megatron.lite.primitive.modules.attention.dsa import DSAIndexShareState

    dsa_module, attention = _dsa(monkeypatch, **INDEX_SHARE["full"])
    assert attention.index_share_enabled and not attention.skip_topk
    recorder = DsaRecorder(monkeypatch, dsa_module, attention)
    binding = FakeBinding(_ids)
    attention.set_indexer_topk(binding)
    state = DSAIndexShareState(retain_for_recompute=False)
    for seq in (6, 3):
        with torch.no_grad():
            attention._forward_dense_full(*_dsa_inputs(attention, seq=seq), index_share_state=state)
        q, k, weights = recorder.projections[-1]
        call = binding.selects[-1]
        assert call["layout"] == QueryLayout.full(seq, keys=seq)
        assert (call["topk"], call["softmax_scale"]) == (
            min(4, seq),
            attention.indexer_softmax_scale,
        )
        # The batch-1 slices of the upstream operands, not copies.
        for given, operand in ((call["q"], q), (call["k"], k), (call["weights"], weights)):
            assert given.data_ptr() == operand.data_ptr() and torch.equal(given, operand[:, 0])
        # The selection feeds IndexShare and the sparse attention as [1, sq, topk].
        expected = _ids(call).unsqueeze(0)
        assert torch.equal(recorder.flat[-1], expected)
        assert torch.equal(state.get_topk(2, 1), expected)
    assert recorder.indexer_topk == [] and binding.declines == []


def test_dsa_packed_prompts_select_each_sequence(monkeypatch):
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    dsa_module, attention = _dsa(monkeypatch)
    recorder = DsaRecorder(monkeypatch, dsa_module, attention)
    binding = FakeBinding(_ids)
    attention.set_indexer_topk(binding)
    x, cos, sin, _ = _dsa_inputs(attention, seq=8)
    positions = torch.cat((torch.arange(3), torch.arange(5))).unsqueeze(0)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 8], dtype=torch.int32))
    with torch.no_grad():
        attention._forward_packed_full(x, cos, sin, positions, packed, index_share_state=None)
    assert [(call["layout"], call["topk"]) for call in binding.selects] == [
        (QueryLayout.full(3, keys=3), 3),
        (QueryLayout.full(5, keys=5), 4),
    ]
    assert recorder.indexer_topk == []


def test_dsa_shared_layer_has_no_geometry(monkeypatch):
    from megatron.lite.primitive.kernels.indexer_topk import IndexerGeometry

    _, full = _dsa(monkeypatch, **INDEX_SHARE["full"])
    _, shared = _dsa(monkeypatch, **INDEX_SHARE["shared"])
    assert shared.skip_topk and shared.indexer is None and not full.skip_topk
    assert full.indexer_geometry() == IndexerGeometry(num_heads=4, head_dim=16, topk=4)
    assert shared.indexer_geometry() is None
    with pytest.raises(ValueError, match="top-k of layer 1 .* takes no binding"):
        shared.set_indexer_topk(FakeBinding())
    shared.set_indexer_topk(None)
    state = list(full.state_dict())
    binding = _real_binding(full.indexer_geometry())
    full.set_indexer_topk(binding)
    assert full._indexer_topk is binding
    # The binding is not module state: no submodule, parameter or buffer.
    assert list(full.state_dict()) == state
    assert all(module is not binding for module in full.modules())
    full.set_indexer_topk(None)
    assert full._indexer_topk is None


def test_dsa_batch_above_one_declines_to_upstream(monkeypatch):
    dsa_module, attention = _dsa(monkeypatch)
    recorder = DsaRecorder(monkeypatch, dsa_module, attention)
    binding = FakeBinding(_ids)
    attention.set_indexer_topk(binding)
    with torch.no_grad():
        attention._forward_dense_full(*_dsa_inputs(attention, batch=2))
    q, k, weights = recorder.projections[-1]
    ((args, kwargs),) = recorder.indexer_topk
    assert args[0] is q and args[1] is k and args[2] is weights and q.shape[1] == 2
    assert args[3:] == (4, 1)
    assert kwargs == {"indexer_softmax_scale": attention.indexer_softmax_scale}
    assert binding.declines == ["batch>1"] and binding.selects == []
    # A required binding raises instead.
    attention.set_indexer_topk(FakeBinding(_ids, required=True))
    with torch.no_grad(), pytest.raises(RuntimeError, match="batch>1"):
        attention._forward_dense_full(*_dsa_inputs(attention, batch=2))


def test_dsa_grad_enabled_uses_upstream(monkeypatch):
    dsa_module, attention = _dsa(monkeypatch)
    recorder = DsaRecorder(monkeypatch, dsa_module, attention)
    binding = _real_binding(attention.indexer_geometry())
    calls = []
    monkeypatch.setattr(binding, "select", lambda *args, **kwargs: calls.append(kwargs) or None)
    attention.set_indexer_topk(binding)
    attention._forward_dense_full(*_dsa_inputs(attention))  # autograd enabled
    assert calls == [] and len(recorder.indexer_topk) == 1
    with torch.no_grad():
        attention._forward_dense_full(*_dsa_inputs(attention))
    # Active without autograd; select returned None (as an inactive binding does): upstream.
    assert len(calls) == 1 and len(recorder.indexer_topk) == 2


def test_configure_binds_only_layers_with_an_indexer(monkeypatch):
    from megatron.lite.primitive.modules.attention import indexer_topk as bindings

    layers = [_dsa(monkeypatch, **INDEX_SHARE[kind])[1] for kind in ("full", "shared")]
    native_format = "fp8"
    # The build-time probes need a GPU (the reference score kernel, cuDNN).
    monkeypatch.setattr(bindings, "_module_device", lambda module: torch.device("cpu"))
    monkeypatch.setattr(bindings, "score_kernel_heads", lambda heads, **kwargs: heads)
    monkeypatch.setattr(bindings, "topk_kernel", lambda exact_topk: None)
    chunk = nn.ModuleList(layers)
    installation = bindings.configure_indexer_topk(
        [chunk], {"backend": "reference", "precision": "fast"}, native_format=native_format
    )
    (binding,) = installation.bindings
    assert layers[0]._indexer_topk is binding and layers[1]._indexer_topk is None
    assert binding.geometry == layers[0].indexer_geometry()
    assert bindings.configure_indexer_topk([chunk], None, native_format=native_format) is None
    assert layers[0]._indexer_topk is None and layers[1]._indexer_topk is None


def test_default_path_does_not_import_indexer_topk(tmp_path):
    """Unbound modules run their upstream selectors without importing the bindings."""
    script = textwrap.dedent("""
        import json, sys, types
        import torch
        from torch import nn

        try:
            import transformer_engine.pytorch  # noqa: F401
        except ImportError:
            for name in ("transformer_engine", "transformer_engine.pytorch"):
                sys.modules[name] = types.ModuleType(name)
            sys.modules["transformer_engine"].pytorch = sys.modules["transformer_engine.pytorch"]

        from megatron.lite.primitive.modules.attention import dsa

        def indexer_topk(q, k, w, topk, ratio, indexer_softmax_scale=1.0):
            ids = torch.zeros((q.shape[1], q.shape[0], topk), dtype=torch.int32)
            return ids, torch.full(q.shape[:2], topk, dtype=torch.int32)

        def sparse_attn(query, kv, sink, idxs, scale, topk_length=None, value_dim=None):
            width = query.shape[2] * (value_dim or query.shape[3])
            return query.new_zeros(query.shape[0], query.shape[1], width)

        dsa._dsa_kernels.indexer_topk = indexer_topk
        dsa._dsa_kernels.dsa_sparse_attn = sparse_attn
        dsa.RMSNorm = lambda size, eps: nn.RMSNorm(size, eps=eps)
        kwargs = json.loads(sys.argv[1])
        attention = dsa.DynamicSparseAttention(**kwargs).eval()
        attention.set_indexer_topk(None)
        cos, sin = dsa.build_rope_cache(dim=8, max_position_embeddings=6, rope_theta=1e4)
        x = torch.randn(1, 6, kwargs["hidden_size"])
        with torch.no_grad():
            attention._forward_dense_full(x, cos, sin, torch.arange(6).unsqueeze(0))
        ran = ["dsa"]
        print(json.dumps({"ran": ran, "loaded": sorted(
            name for name in sys.modules
            if name.startswith("megatron.lite.primitive.kernels.indexer_topk")
            or name == "megatron.lite.primitive.modules.attention.indexer_topk"
        )}))
        """)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(LITE_ROOT), str(REPO_ROOT), environment.get("PYTHONPATH")))
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(DSA_KWARGS)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded"] == []
    assert report["ran"] == ["dsa"]


# ---------------------------------------------------------------------------
# Bindings select only in eval mode with gradients disabled
# ---------------------------------------------------------------------------


def test_binding_selects_only_in_eval_mode_without_grad(monkeypatch):
    module_code, layer = _dsa(monkeypatch)
    binding = FakeBinding(_ids)
    layer.set_indexer_topk(binding)
    for training in (False, True):
        for grad in (False, True):
            layer.train(training)
            with torch.set_grad_enabled(grad):
                selected = module_code._bound_indexer_topk(layer)
            assert (selected is binding) == (not training and not grad), (training, grad)
    # Test doubles: no binding, no training flag (treated as eval mode), training flag set.
    with torch.no_grad():
        assert module_code._bound_indexer_topk(SimpleNamespace()) is None
        assert module_code._bound_indexer_topk(SimpleNamespace(_indexer_topk=binding)) is binding
        double = SimpleNamespace(_indexer_topk=binding, training=True)
        assert module_code._bound_indexer_topk(double) is None


def _requiring_grad(tensors):
    return tuple(tensor.detach().requires_grad_(True) for tensor in tensors)


def test_dsa_training_recompute_never_selects(monkeypatch):
    """Lite's reentrant recompute runs a training forward without autograd and recomputes it
    with gradients in the backward pass: neither run may select through the binding."""
    from megatron.lite.primitive.recompute import wrap_checkpoint

    dsa_module, attention = _dsa(monkeypatch)
    recorder = DsaRecorder(monkeypatch, dsa_module, attention)
    fused: list[bool] = []

    def fused_indexer_sparse_attn(query, *args, value_dim=None, **kwargs):
        fused.append(torch.is_grad_enabled())
        out = query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)
        return out + query.sum() * 0, query.sum() * 0

    monkeypatch.setattr(dsa_module, "_fused_indexer_sparse_attn", fused_indexer_sparse_attn)
    binding = FakeBinding(_ids)
    attention.set_indexer_topk(binding)
    attention.train()
    wrap_checkpoint(attention, preserve_rng_state=False)
    x, cos, sin, positions = _dsa_inputs(attention)
    (x,) = _requiring_grad((x,))
    out = attention(x, cos=cos, sin=sin, position_ids=positions)
    # The checkpointed forward ran without autograd: the upstream inference selector.
    assert len(recorder.indexer_topk) == 1 and fused == []
    out.sum().backward()
    # The backward pass recomputed the training path with gradients.
    assert fused == [True] and len(recorder.indexer_topk) == 1 and x.grad is not None
    assert binding.selects == [] and binding.declines == []
    # The same layer in eval mode without autograd selects through the binding.
    attention.eval()
    with torch.no_grad():
        attention(x, cos=cos, sin=sin, position_ids=positions)
    assert len(binding.selects) == 1 and len(recorder.indexer_topk) == 1
