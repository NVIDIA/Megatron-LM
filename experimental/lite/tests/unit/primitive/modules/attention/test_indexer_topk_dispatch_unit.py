# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dispatch of the DSA and CSA indexer top-k to optional indexer top-k bindings (CPU).

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
_UPSTREAM_CSA_BSHD = """
topk_indices, _topk_length = dsa_kernels.indexer_topk(
    q_indexer,
    index_k,
    weights_indexer,
    indexer_topk,
    self.compress_ratio,
    indexer_softmax_scale=self.indexer.softmax_scale,
)
"""
_UPSTREAM_CSA_THD = """
compressed_topk, indexer_layout, _ = cp_utils.compute_cp_indexer_topk(
    q_indexer_cp,
    weights_indexer_cp,
    k_indexer_seq_major,
    cu_seqlens,
    cu_seqlens_compressed,
    global_start,
    ratio,
    indexer.index_topk,
    indexer.softmax_scale,
    max_seqlen_q=max_seqlen_q,
    use_fused=self.apply_dsa_kernel_fusion,
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
    assert full.indexer_geometry() == IndexerGeometry(num_heads=4, head_dim=16, topk=4, key_ratio=1)
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


@pytest.mark.parametrize("model", ["dsa", "csa"])
def test_configure_binds_only_layers_with_an_indexer(monkeypatch, model):
    from megatron.lite.primitive.modules.attention import indexer_topk as bindings

    if model == "dsa":
        layers = [_dsa(monkeypatch, **INDEX_SHARE[kind])[1] for kind in ("full", "shared")]
        native_format = "fp8"
    else:
        layers = [_csa_module(monkeypatch, ratio=ratio) for ratio in (4, 128)]
        native_format = "mxfp4"
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
        try:
            from megatron.lite.primitive.modules.attention import csa
        except ImportError:  # Megatron Core CSA needs Transformer Engine
            csa = None
        if csa is not None:
            from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

            csa.te = types.SimpleNamespace(RMSNorm=lambda n, eps: nn.RMSNorm(n, eps=eps))
            config = DeepseekV4Config(**json.loads(sys.argv[2]))
            ps = types.SimpleNamespace(cp_size=1, cp_rank=0, cp_group=None)
            layer = csa.CompressedSparseAttention(config, layer_idx=0, ps=ps).eval()
            layer.set_indexer_topk(None)
            x = torch.randn(1, 16, config.hidden_size)
            positions = torch.arange(16).unsqueeze(0)
            cos, sin = csa.build_compressed_rope_cos_sin(
                positions, 4, config.compress_rope_theta, config=config, use_yarn=True,
                device=x.device, dtype=x.dtype,
            )
            with torch.no_grad():
                layer._forward_fused_dsa_cp1(
                    x, torch.randn(1, 4, 16, 8), torch.randn(1, 16, 16),
                    torch.randn(1, 1, 16, 8), position_ids=positions, cos=cos, sin=sin,
                    attention_mask=None,
                )
            ran.append("csa")
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
    csa_config = (
        {
            name: getattr(_csa_config(), name)
            for name in (
                "hidden_size",
                "num_attention_heads",
                "head_dim",
                "qk_rope_head_dim",
                "q_lora_rank",
                "o_lora_rank",
                "o_groups",
                "compress_ratios",
                "sliding_window",
                "index_head_dim",
                "index_n_heads",
                "index_topk",
                "num_hidden_layers",
            )
        }
        if _csa() is not None
        else {}
    )
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(DSA_KWARGS), json.dumps(csa_config)],
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
    assert report["ran"] == (["dsa", "csa"] if _csa() is not None else ["dsa"])


# ---------------------------------------------------------------------------
# CSA: THD (_forward_thd_cp, CP=1 and CP>1) and BSHD CP=1 (_forward_fused_dsa_cp1)
# ---------------------------------------------------------------------------


def _csa():
    """The Lite CSA module, or None where Megatron Core CSA cannot be imported (no TE)."""
    try:
        from megatron.lite.primitive.modules.attention import csa
    except ImportError:
        return None
    return csa


def _require_csa():
    csa = _csa()
    if csa is None:
        pytest.skip("Megatron Core CSA needs Transformer Engine")
    return csa


def _csa_config(ratio: int = 4):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        compress_ratios=[ratio],
        sliding_window=4,
        index_head_dim=8,
        index_n_heads=4,
        index_topk=4,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
    )


def _ps(cp_size: int = 1, cp_rank: int = 0):
    group = SimpleNamespace(size=lambda: cp_size, rank=lambda: cp_rank)
    return SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank, cp_group=group)


def _csa_module(monkeypatch, *, ratio: int = 4, cp_size: int = 1, cp_rank: int = 0):
    """A real CompressedSparseAttention on the CPU (torch RMSNorm)."""
    csa = _require_csa()
    monkeypatch.setattr(csa, "te", SimpleNamespace(RMSNorm=lambda n, eps: nn.RMSNorm(n, eps=eps)))
    torch.manual_seed(0)
    module = csa.CompressedSparseAttention(
        _csa_config(ratio), layer_idx=0, ps=_ps(cp_size, cp_rank)
    )
    return module.eval()


def test_csa_geometry_only_for_ratio4(monkeypatch):
    from megatron.lite.primitive.kernels.indexer_topk import IndexerGeometry

    c4 = _csa_module(monkeypatch, ratio=4)
    assert c4.indexer_geometry() == IndexerGeometry(num_heads=4, head_dim=8, topk=4, key_ratio=4)
    assert c4._indexer_topk is None
    for ratio in (128, 0):
        layer = _csa_module(monkeypatch, ratio=ratio)
        assert layer.indexer is None and layer.indexer_geometry() is None
        with pytest.raises(ValueError, match="no indexer"):
            layer.set_indexer_topk(FakeBinding())
        layer.set_indexer_topk(None)
    binding = FakeBinding()
    c4.set_indexer_topk(binding)
    assert c4._indexer_topk is binding
    assert not any("indexer_topk" in name for name in c4.state_dict())


L_LOCAL, D_WINDOW, C_CAP = 16, 8, 8


class ThdRecorder:
    """Replaces the Core CP kernels of ``_forward_thd_cp`` and records their inputs.

    ``compressed`` holds the global sequence-major compressed offsets of the packed
    sequences, as the compaction kernel computes them (``length // 4`` keys each).
    """

    def __init__(self, monkeypatch, csa, *, cp_size: int, compressed: list[int]):
        self.compute: list[tuple] = []
        self.indices: list[dict] = []
        self.cu_seqlens_compressed = torch.tensor(compressed, dtype=torch.int32)
        rank_rows = C_CAP * cp_size
        self.seq_to_rank_row = torch.arange(L_LOCAL * cp_size // 4, dtype=torch.int32) % rank_rows

        def prepare(hidden, boundary, cu_seqlens, global_start, cp_size_arg, ratio):
            generator = torch.Generator().manual_seed(2)
            groups = torch.arange(C_CAP, dtype=torch.int32)
            compact = torch.randn(C_CAP * ratio, 1, hidden.shape[-1], generator=generator)
            return (
                compact,
                groups,
                groups * ratio,
                None,
                None,
                self.cu_seqlens_compressed,
                self.seq_to_rank_row,
            )

        def compute(*args, **kwargs):
            self.compute.append((args, kwargs))
            rows = args[0].shape[0]
            return torch.full((rows, args[7]), 7, dtype=torch.int32), "logical-layout", None

        def gather(tensor, *, group):
            return torch.cat([tensor] * cp_size)

        def build_indices(cu, start, rows, d_window, window, ratio, width, topk, **kwargs):
            self.indices.append(dict(width=width, compressed_topk=topk, **kwargs))
            full = window + width
            return (
                torch.zeros(rows, full, dtype=torch.int32),
                torch.full((rows,), full, dtype=torch.int32),
                None,
                None,
            )

        monkeypatch.setattr(csa.cp_utils, "prepare_cp_compressor_input", prepare)
        monkeypatch.setattr(csa.cp_utils, "compute_cp_indexer_topk", compute)
        monkeypatch.setattr(csa, "gather_from_sequence_parallel_region", gather)
        monkeypatch.setattr(csa.thd_layout_kernels, "build_attention_indices", build_indices)
        monkeypatch.setattr(csa, "csa_sparse_attn", lambda query, *args, **kwargs: query)
        monkeypatch.setattr(csa, "unfused_compressed_sparse_attn", lambda query, *args: query)


def _thd_inputs(module, cu_seqlens: list[int], *, max_seqlen_q: int | None = None):
    """The tensor inputs of ``_forward_thd_cp`` and its packed sequence parameters."""
    generator = torch.Generator().manual_seed(3)
    hidden, head_dim = module.config.hidden_size, module.head_dim
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    lengths = [end - start for start, end in zip(cu_seqlens, cu_seqlens[1:])]
    packed = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_q_padded=None,
        max_seqlen_q=max(lengths) if max_seqlen_q is None else max_seqlen_q,
    )
    tensors = (
        torch.randn(L_LOCAL, module.num_heads, head_dim, generator=generator),
        torch.randn(L_LOCAL, 1, 1, head_dim, generator=generator),
        torch.randn(L_LOCAL, 1, hidden, generator=generator),
        torch.randn(L_LOCAL, 1, module.config.q_lora_rank, generator=generator),
        torch.randn(D_WINDOW, 1, hidden, generator=generator),
        torch.randn(D_WINDOW, 1, 1, head_dim, generator=generator),
    )
    return tensors, packed


def _run_thd(module, cu_seqlens: list[int], *, max_seqlen_q: int | None = None):
    tensors, packed = _thd_inputs(module, cu_seqlens, max_seqlen_q=max_seqlen_q)
    return module._forward_thd_cp(*tensors, packed)


def test_default_thd_calls_compute_cp_indexer_topk_verbatim(monkeypatch):
    csa = _require_csa()
    assert _assignments_calling(
        csa.CompressedSparseAttention._forward_thd_cp, "cp_utils.compute_cp_indexer_topk"
    ) == [_dump(_UPSTREAM_CSA_THD)]
    for cp_size, cu_seqlens, compressed in (
        (1, [0, 16], [0, 4]),
        (2, [0, 10, 26, 32], [0, 2, 6, 7]),
    ):
        module = _csa_module(monkeypatch, cp_size=cp_size, cp_rank=cp_size - 1)
        recorder = ThdRecorder(monkeypatch, csa, cp_size=cp_size, compressed=compressed)
        with torch.no_grad():
            _run_thd(module, cu_seqlens)
        ((args, kwargs),) = recorder.compute
        q, weights, k = args[:3]
        assert q.shape == (L_LOCAL, 4, 8) and weights.shape == (L_LOCAL, 4)
        assert k.shape == (L_LOCAL * cp_size // 4, 8)
        assert args[3].tolist() == cu_seqlens and args[4] is recorder.cu_seqlens_compressed
        assert args[5:] == ((cp_size - 1) * L_LOCAL, 4, 4, module.indexer.softmax_scale)
        assert kwargs == {"max_seqlen_q": 16, "use_fused": module.apply_dsa_kernel_fusion}
        (indices,) = recorder.indices
        assert indices["width"] == 4 and indices["for_indexer_loss"] is False
        assert torch.equal(indices["compressed_topk"], torch.full((L_LOCAL, 4), 7))


def _thd_call(monkeypatch, cp_size: int, cu_seqlens: list[int], compressed: list[int], **kwargs):
    """Run the default and the bound THD path of one module; return (recorder, binding)."""
    csa = _require_csa()
    module = _csa_module(monkeypatch, cp_size=cp_size, cp_rank=cp_size - 1)
    recorder = ThdRecorder(monkeypatch, csa, cp_size=cp_size, compressed=compressed)
    with torch.no_grad():
        _run_thd(module, cu_seqlens, **kwargs)
    binding = FakeBinding(_ids)
    module.set_indexer_topk(binding)
    with torch.no_grad():
        _run_thd(module, cu_seqlens, **kwargs)
    return module, recorder, binding


@pytest.mark.parametrize(
    "cp_size, cu_seqlens, compressed",
    [(1, [0, 16], [0, 4]), (2, [0, 32], [0, 8]), (2, [0, 10, 26, 32], [0, 2, 6, 7])],
)
def test_thd_binding_receives_upstream_operands_and_packed_layout(
    monkeypatch, cp_size, cu_seqlens, compressed
):
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    module, recorder, binding = _thd_call(monkeypatch, cp_size, cu_seqlens, compressed)
    # The first run (unbound) called the upstream selector, the second run only the binding.
    ((args, kwargs),) = recorder.compute
    (call,) = binding.selects
    assert torch.equal(call["q"], args[0]) and torch.equal(call["weights"], args[1])
    assert torch.equal(call["k"], args[2])
    global_start = (cp_size - 1) * L_LOCAL
    assert call["layout"] == QueryLayout.packed(
        cu_seqlens, row_start=global_start, rows=L_LOCAL, key_ratio=4, absolute_ids=False
    )
    assert (call["topk"], call["softmax_scale"]) == (4, module.indexer.softmax_scale)
    # The result goes to the unchanged attention indices; no logical indexer layout.
    selected = recorder.indices[-1]
    assert torch.equal(selected["compressed_topk"], _ids(call))
    assert selected["width"] == 4 and selected["for_indexer_loss"] is False
    assert selected["cu_seqlens_compressed"] is recorder.cu_seqlens_compressed
    assert selected["seq_to_rank_row"] is recorder.seq_to_rank_row
    assert binding.declines == []


def test_thd_single_sequence_layout_no_host_sync(monkeypatch):
    """One sequence over every packed row: the layout comes from host metadata only."""
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    csa = _require_csa()
    module = _csa_module(monkeypatch, cp_size=2, cp_rank=1)
    binding = FakeBinding(_ids)
    operands = (
        torch.zeros(L_LOCAL, 4, 8),
        torch.zeros(L_LOCAL, 4),
        torch.zeros(L_LOCAL * 2 // 4, 8),
    )
    select = csa.CompressedSparseAttention._select_thd_indexer_topk
    # Meta tensors cannot be read on the host: any read of the offsets would raise.
    meta = torch.empty(2, dtype=torch.int32, device="meta")
    selected = select(
        module, binding, *operands, meta, meta, global_start=16, cp_size=2, max_seqlen_q=32
    )
    assert torch.equal(selected, _ids(binding.selects[-1]))
    assert binding.selects[-1]["layout"] == QueryLayout.packed(
        [0, 32], row_start=16, rows=16, key_ratio=4, absolute_ids=False
    )
    # Several sequences, or one that the metadata does not prove to cover every row (padding
    # rows after it): the offsets are read.
    for offsets, longest in (([0, 12, 32], 20), ([0, 30], 30)):
        with pytest.raises((NotImplementedError, RuntimeError)):
            select(
                module,
                binding,
                *operands,
                torch.empty(len(offsets), dtype=torch.int32, device="meta"),
                meta,
                global_start=16,
                cp_size=2,
                max_seqlen_q=longest,
            )
        cu = torch.tensor(offsets, dtype=torch.int32)
        select(module, binding, *operands, cu, cu, global_start=16, cp_size=2, max_seqlen_q=longest)
        assert binding.selects[-1]["layout"] == QueryLayout.packed(
            offsets, row_start=16, rows=16, key_ratio=4, absolute_ids=False
        )
    # The padding rows past the last sequence select nothing (rows 14 and 15 here).
    assert binding.selects[-1]["layout"].segments[-1].row_end == 14


def test_thd_layout_is_read_on_every_call(monkeypatch):
    """A reused offsets tensor (same address and version) never yields a stale layout."""
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    csa = _require_csa()
    module = _csa_module(monkeypatch, cp_size=2, cp_rank=1)
    binding = FakeBinding(_ids)
    operands = (torch.zeros(16, 4, 8), torch.zeros(16, 4), torch.zeros(8, 8))
    cu = torch.tensor([0, 10, 26, 32], dtype=torch.int32)
    address, version = cu.data_ptr(), cu._version
    layouts = []
    for offsets in ([0, 10, 26, 32], [0, 20, 24, 32]):
        cu.data.copy_(torch.tensor(offsets, dtype=torch.int32))  # the next batch, same memory
        assert (cu.data_ptr(), cu._version) == (address, version)
        csa.CompressedSparseAttention._select_thd_indexer_topk(
            module, binding, *operands, cu, cu, global_start=16, cp_size=2, max_seqlen_q=16
        )
        layouts.append(binding.selects[-1]["layout"])
    assert layouts == [
        QueryLayout.packed(offsets, row_start=16, rows=16, key_ratio=4, absolute_ids=False)
        for offsets in ([0, 10, 26, 32], [0, 20, 24, 32])
    ]


def test_thd_without_compressed_keys_runs_upstream(monkeypatch):
    csa = _require_csa()
    module = _csa_module(monkeypatch)
    binding = FakeBinding(_ids)
    select = csa.CompressedSparseAttention._select_thd_indexer_topk
    q, weights, cu = torch.zeros(16, 4, 8), torch.zeros(16, 4), torch.tensor([0, 16])
    # No compressed key rows, or sequences shorter than the ratio: upstream returns no top-k.
    assert (
        select(
            module,
            binding,
            q,
            weights,
            torch.zeros(0, 8),
            cu,
            cu,
            global_start=0,
            cp_size=1,
            max_seqlen_q=16,
        )
        is None
    )
    assert (
        select(
            module,
            binding,
            q,
            weights,
            torch.zeros(4, 8),
            cu,
            cu,
            global_start=0,
            cp_size=1,
            max_seqlen_q=3,
        )
        is None
    )
    assert binding.selects == [] and binding.declines == []


# BSHD CP=1: _forward_fused_dsa_cp1


class BshdRecorder:
    """Replaces the kernels of ``_forward_fused_dsa_cp1`` and records their inputs."""

    def __init__(self, monkeypatch, csa, module):
        self.indexer_topk: list[tuple] = []
        self.upstream_ids: list[torch.Tensor] = []
        self.flat: list[tuple] = []
        kernels = csa._load_dsa_kernels()
        build_flat = kernels.build_flat_topk_idxs

        def indexer_topk(*args, **kwargs):
            self.indexer_topk.append((args, kwargs))
            q, k = args[0], args[1]
            ids = torch.arange(args[3], dtype=torch.int32) % k.shape[0]
            ids = ids.expand(q.shape[1], q.shape[0], -1).clone()
            ids[:, 0] = -1  # the first query row sees no compressed key
            self.upstream_ids.append(ids)
            return ids, torch.full(q.shape[:2], args[3], dtype=torch.int32)

        def flat_topk(*groups, **kwargs):
            self.flat.append(groups)
            return build_flat(*groups, **kwargs)

        def sparse_attn(query, kv, sink, idxs, scale, topk_length=None):
            return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * query.shape[3])

        monkeypatch.setattr(kernels, "indexer_topk", indexer_topk)
        monkeypatch.setattr(kernels, "build_flat_topk_idxs", flat_topk)
        monkeypatch.setattr(kernels, "dsa_sparse_attn", sparse_attn)


def _bshd_inputs(csa, module, *, batch: int = 1, seq: int = 14):
    """The tensor inputs of ``_forward_fused_dsa_cp1`` and its keyword arguments."""
    generator = torch.Generator().manual_seed(4)
    config = module.config
    x = torch.randn(batch, seq, config.hidden_size, generator=generator)
    q = torch.randn(batch, module.num_heads, seq, module.head_dim, generator=generator)
    q_low = torch.randn(batch, seq, config.q_lora_rank, generator=generator)
    kv = torch.randn(batch, 1, seq, module.head_dim, generator=generator)
    positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    cos, sin = csa.build_compressed_rope_cos_sin(
        positions,
        module.rope_head_dim,
        config.compress_rope_theta,
        config=config,
        use_yarn=True,
        device=x.device,
        dtype=x.dtype,
    )
    keywords = dict(position_ids=positions, cos=cos, sin=sin, attention_mask=None)
    return (x, q, q_low, kv), keywords


def _run_bshd(csa, module, *, batch: int = 1, seq: int = 14):
    tensors, keywords = _bshd_inputs(csa, module, batch=batch, seq=seq)
    return module._forward_fused_dsa_cp1(*tensors, **keywords)


def test_bshd_offsets_unchanged(monkeypatch):
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    csa = _require_csa()
    assert _assignments_calling(
        csa.CompressedSparseAttention._forward_fused_dsa_cp1, "dsa_kernels.indexer_topk"
    ) == [_dump(_UPSTREAM_CSA_BSHD)]
    module = _csa_module(monkeypatch)
    recorder = BshdRecorder(monkeypatch, csa, module)
    with torch.no_grad():
        _run_bshd(csa, module)  # 14 tokens, right-padded to 16
    ((args, kwargs),) = recorder.indexer_topk
    assert args[0].shape == (16, 1, 4, 8) and args[1].shape == (4, 1, 8)
    assert args[2].shape == (16, 1, 4) and args[3:] == (4, 4)
    assert kwargs == {"indexer_softmax_scale": module.indexer.softmax_scale}
    # Compressed keys follow the 16 (padded) tokens in kv_full; -1 stays -1.
    (upstream_ids,) = recorder.upstream_ids
    expected = torch.where(upstream_ids >= 0, upstream_ids + 16, upstream_ids)
    assert torch.equal(recorder.flat[-1][1], expected)

    binding = FakeBinding(_ids)
    module.set_indexer_topk(binding)
    with torch.no_grad():
        _run_bshd(csa, module)
    assert len(recorder.indexer_topk) == 1
    (call,) = binding.selects
    assert call["layout"] == QueryLayout.full(16, keys=4, key_ratio=4)
    assert (call["topk"], call["softmax_scale"]) == (4, module.indexer.softmax_scale)
    for given, operand in zip((call["q"], call["k"], call["weights"]), args[:3]):
        assert torch.equal(given, operand[:, 0])
    # The binding's ids take the same offset.
    ids = _ids(call).unsqueeze(0)
    offsets = recorder.flat[-1][1]
    assert torch.equal(offsets, torch.where(ids >= 0, ids + 16, ids))
    assert offsets.dtype == torch.int32 and binding.declines == []


def test_bshd_batch_above_one_declines_to_upstream(monkeypatch):
    csa = _require_csa()
    module = _csa_module(monkeypatch)
    recorder = BshdRecorder(monkeypatch, csa, module)
    binding = FakeBinding(_ids)
    module.set_indexer_topk(binding)
    with torch.no_grad():
        _run_bshd(csa, module, batch=2, seq=16)
    assert binding.declines == ["batch>1"] and binding.selects == []
    ((args, kwargs),) = recorder.indexer_topk
    assert args[0].shape == (16, 2, 4, 8) and args[3:] == (4, 4)


def test_csa_grad_enabled_uses_upstream(monkeypatch):
    csa = _require_csa()
    module = _csa_module(monkeypatch)
    recorder = ThdRecorder(monkeypatch, csa, cp_size=1, compressed=[0, 4])
    bshd = BshdRecorder(monkeypatch, csa, module)
    binding = _real_binding(module.indexer_geometry())
    calls = []
    monkeypatch.setattr(binding, "select", lambda *args, **kwargs: calls.append(kwargs))
    module.set_indexer_topk(binding)
    _run_thd(module, [0, 16])  # autograd enabled
    _run_bshd(csa, module)
    assert calls == []
    assert len(recorder.compute) == 1 and len(bshd.indexer_topk) == 1


# ---------------------------------------------------------------------------
# Bindings select only in eval mode with gradients disabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["dsa", "csa"])
def test_binding_selects_only_in_eval_mode_without_grad(monkeypatch, model):
    if model == "dsa":
        module_code, layer = _dsa(monkeypatch)
    else:
        module_code, layer = _require_csa(), _csa_module(monkeypatch)
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


def test_csa_training_recompute_never_selects(monkeypatch):
    """THD and BSHD training forwards under Lite's reentrant recompute select upstream."""
    from megatron.lite.primitive.recompute import CheckpointFunction

    csa = _require_csa()
    module = _csa_module(monkeypatch)
    thd = ThdRecorder(monkeypatch, csa, cp_size=1, compressed=[0, 4])
    bshd = BshdRecorder(monkeypatch, csa, module)
    fused: list[bool] = []

    def fused_indexer_sparse_attn(query, *args, **kwargs):
        fused.append(torch.is_grad_enabled())
        out = query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * query.shape[3])
        return out + query.sum() * 0, query.sum() * 0

    monkeypatch.setattr(
        csa._load_dsa_kernels(), "fused_indexer_sparse_attn", fused_indexer_sparse_attn
    )
    # A new tensor (not a view of an input) keeps the checkpointed output on the graph.
    monkeypatch.setattr(csa, "csa_sparse_attn", lambda query, *args, **kwargs: query * 1)
    binding = FakeBinding(_ids)
    module.set_indexer_topk(binding)
    thd_tensors, packed = _thd_inputs(module, [0, 16])
    bshd_tensors, keywords = _bshd_inputs(csa, module)

    def thd_forward(*tensors):
        return module._forward_thd_cp(*tensors, packed)

    def bshd_forward(*tensors):
        return module._forward_fused_dsa_cp1(*tensors, **keywords)

    module.train()
    for run, tensors in ((thd_forward, thd_tensors), (bshd_forward, bshd_tensors)):
        inputs = _requiring_grad(tensors)
        CheckpointFunction.apply(run, False, *inputs).sum().backward()
        assert any(tensor.grad is not None for tensor in inputs)  # the recompute ran
    # THD: the forward (without autograd) and the recompute (with it) both selected upstream.
    assert len(thd.compute) == 2
    # BSHD: the forward ran the upstream inference selector, the recompute the training path.
    assert len(bshd.indexer_topk) == 1 and fused == [True]
    assert binding.selects == [] and binding.declines == []
    # The same layer in eval mode without autograd selects through the binding.
    module.eval()
    with torch.no_grad():
        thd_forward(*thd_tensors)
        bshd_forward(*bshd_tensors)
    assert len(binding.selects) == 2
    assert len(thd.compute) == 2 and len(bshd.indexer_topk) == 1
