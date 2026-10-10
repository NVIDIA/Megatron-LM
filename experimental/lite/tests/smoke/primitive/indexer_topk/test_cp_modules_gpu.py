# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lite DSA native context parallelism with a LiteTopK binding on 2 and 4 GPUs (optional).

Every rank builds the same DSA layer twice, as a CP=1 layer and as rank r of a contiguous CP group
of all ranks, and binds both to a LiteTopK FP8 route with exact selection (the first FP8 entry of
``LITETOPK_TEST_SELECTORS`` with precision ``exact``; ``LITETOPK_TEST_EXACT_TOPK``; as
``test_selector_gpu.py``). Every rank runs the CP=1 layer over the whole 262144-token prompt, then
the CP layer over its contiguous shard with the indexer queries, keys and weights of the CP=1 run
injected (a shard's projections differ from the whole prompt's in the last bits). On every rank:

* the CP selection equals rows ``[r L, (r + 1) L)`` of the CP=1 selection bit for bit, and no
  dense causal mask is built;
* the layer output matches those rows of the CP=1 output (the attention projections of a shard
  may differ in the last bits);
* every rank issues the same collectives;
* when the last rank's plugin declines every tile, or reports every row failed, every rank still
  finishes with the same selection and the same collectives; with a required binding that rank
  raises and the other ranks finish.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.optional

_SELECTORS_VARIABLE = "LITETOPK_TEST_SELECTORS"
_EXACT_VARIABLE = "LITETOPK_TEST_EXACT_TOPK"
_TOKENS = 262144
_LAYER = dict(
    hidden_size=1024,
    num_attention_heads=64,
    q_lora_rank=512,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    rms_norm_eps=1e-5,
)


def _read_json(variable: str):
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return json.loads(raw if raw[0] in "[{" else Path(raw).read_text(encoding="utf-8"))


def _plugin_or_skip():
    selectors = _read_json(_SELECTORS_VARIABLE) or []
    exact = _read_json(_EXACT_VARIABLE)
    spec = next(
        (s for s in selectors if s["native_format"] == "fp8" and s["precision"] == "exact"), None
    )
    if spec is None or exact is None:
        pytest.skip(f"{_SELECTORS_VARIABLE} (FP8 exact entry) and {_EXACT_VARIABLE} are not set")
    exact = dict(exact)
    for entry in reversed(exact.pop("pythonpath", [])):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return spec, exact


def _world_or_skip(expected: int):
    import torch.distributed as dist

    if not torch.cuda.is_available() or "RANK" not in os.environ:
        pytest.skip("run with torchrun on CUDA devices")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        # A hung rank fails the test within minutes instead of the NCCL default of ten.
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
    if dist.get_world_size() != expected:
        pytest.skip(f"needs {expected} ranks, got {dist.get_world_size()}")
    return dist


class _Recorder:
    """Selections of a binding (its ``select`` wrapped)."""

    def __init__(self, binding):
        self.binding, self.selections = binding, []
        select = binding.select

        def recording_select(*args, **kwargs):
            self.selections.append(select(*args, **kwargs))
            return self.selections[-1]

        binding.select = recording_select


def _configure(layers, spec, exact, *, required=False):
    from megatron.lite.primitive.kernels.indexer_topk import (
        ExactTopKConfig,
        IndexerTopKConfig,
        IndexerTopKTuning,
        LiteTopKPluginConfig,
    )
    from megatron.lite.primitive.modules.attention.indexer_topk import configure_indexer_topk

    installation = configure_indexer_topk(
        layers,
        IndexerTopKConfig(
            backend="litetopk",
            precision="exact",
            litetopk=LiteTopKPluginConfig(**spec["litetopk"]),
            exact_topk=ExactTopKConfig(**exact),
        ),
        native_format="fp8",
        tuning=IndexerTopKTuning(required=required),
    )
    return [_Recorder(binding) for binding in installation.bindings]


def _run(expected_world: int, monkeypatch) -> None:
    spec, exact = _plugin_or_skip()
    dist = _world_or_skip(expected_world)
    import torch.distributed.nn.functional as dist_functional

    from megatron.lite.primitive.kernels.indexer_topk import IndexerTopKRuntimeError
    from megatron.lite.primitive.modules.attention import dsa

    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    local = _TOKENS // world
    rows = slice(rank * local, (rank + 1) * local)

    torch.manual_seed(0)
    whole = dsa.DynamicSparseAttention(**_LAYER).to(device, torch.bfloat16).eval()
    shard = dsa.DynamicSparseAttention(
        **_LAYER, cp_size=world, cp_rank=rank, cp_group=dist.group.WORLD
    )
    shard = shard.to(device, torch.bfloat16).eval()
    shard.load_state_dict(whole.state_dict())
    whole_recorder, shard_recorder = _configure([whole, shard], spec, exact)

    generator = torch.Generator(device=device).manual_seed(20260930)
    x = torch.randn(1, _TOKENS, _LAYER["hidden_size"], generator=generator, device=device)
    x = x.to(torch.bfloat16)
    cos, sin = dsa.build_rope_cache(
        dim=_LAYER["qk_rope_head_dim"],
        max_position_embeddings=_TOKENS,
        rope_theta=10000.0,
        device=device,
    )
    positions = torch.arange(_TOKENS, device=device).unsqueeze(0)

    # CP=1 over the whole prompt; keep its indexer operands.
    operands = []
    project = whole.indexer.forward_before_topk

    def keep_operands(*args, **kwargs):
        operands.append(project(*args, **kwargs))
        return operands[-1]

    monkeypatch.setattr(whole.indexer, "forward_before_topk", keep_operands)
    with torch.no_grad():
        whole_out = whole(x, cos=cos, sin=sin, position_ids=positions)[:, rows].clone()
    (whole_selection,) = whole_recorder.selections
    expected = whole_selection[rows].clone()
    q, k, weights = (tensor[rows].contiguous() for tensor in operands[0])
    del whole_selection, operands[:]
    torch.cuda.empty_cache()

    # The CP layer: CP=1 indexer operands injected, collectives and masks counted.
    monkeypatch.setattr(shard.indexer, "forward_before_topk", lambda *a, **kw: (q, k, weights))
    collectives = []
    all_gather = dist_functional.all_gather

    def counted_all_gather(tensor, group=None):
        collectives.append(tuple(tensor.shape))
        return all_gather(tensor, group=group)

    monkeypatch.setattr(dist_functional, "all_gather", counted_all_gather)
    masks = []
    build_mask = dsa._build_cp_causal_mask
    monkeypatch.setattr(
        dsa, "_build_cp_causal_mask", lambda *a, **kw: masks.append(a) or build_mask(*a, **kw)
    )

    def shard_forward():
        with torch.no_grad():
            return shard(x[:, rows], cos=cos, sin=sin, position_ids=positions[:, rows])

    def check_ranks(counts, label):
        gathered = [None] * world
        dist.all_gather_object(gathered, counts)
        assert all(entry == gathered[0] for entry in gathered), (label, gathered)

    shard_out = shard_forward()
    stats = shard_recorder.binding.stats
    (selection,) = shard_recorder.selections
    assert torch.equal(selection, expected), f"rank {rank}: CP selection differs from CP=1"
    assert masks == []
    difference = (shard_out.float() - whole_out.float()).norm() / whole_out.float().norm()
    assert float(difference) < 1e-2, float(difference)
    check_ranks(collectives, "collectives")
    # LiteTopK tiles start at the plan's start position: only ranks with rows past it have tiles.
    startup = shard_recorder.binding.resolved_tuning(device).startup_position
    assert (stats.tiles > 0) == ((rank + 1) * local > startup), (rank, stats.tiles, startup)
    last = world - 1
    report = {
        "rank": rank,
        "world": world,
        "counters": _counters(stats),
        "output_rel_l2": float(difference),
        "collectives": list(collectives),
    }

    # Fault injection on the last rank: every tile declined, then every row failed.
    plugin = shard_recorder.binding.plugin.module
    dispatch = plugin.try_large_exact_once_chunk

    def failing_dispatch(*args, **kwargs):
        dispatched = dispatch(*args, **kwargs)
        if dispatched and kwargs.get("status_out") is not None:
            kwargs["status_out"].fill_(3)  # FAILED
        return dispatched

    for label, replacement in (
        ("declined", lambda *args, **kwargs: False),
        ("failed", failing_dispatch),
    ):
        collectives.clear()
        shard_recorder.selections.clear()
        stats.reset()
        with monkeypatch.context() as patch:
            if rank == last:
                patch.setattr(plugin, "try_large_exact_once_chunk", replacement)
            shard_forward()
        (selection,) = shard_recorder.selections
        assert torch.equal(selection, expected), (label, rank)
        if rank == last and label == "declined":
            assert stats.tiles == 0 and sum(stats.declined_tiles.values()) > 0
        if rank == last and label == "failed":
            assert stats.status_rows.get(3, 0) > 0 and stats.recomputed_tiles > 0
        assert masks == []
        check_ranks(collectives, label)
        report[label] = _counters(stats)

    # A required binding raises on the failing rank; the other ranks finish.
    (shard_recorder,) = _configure([shard], spec, exact, required=True)
    collectives.clear()
    with monkeypatch.context() as patch:
        if rank == last:
            patch.setattr(plugin, "try_large_exact_once_chunk", lambda *args, **kwargs: False)
            with pytest.raises(IndexerTopKRuntimeError, match="required LiteTopK"):
                shard_forward()
        else:
            shard_forward()
            assert torch.equal(shard_recorder.selections[-1], expected)
    check_ranks(collectives, "required")
    dist.barrier()
    report["required_raised"] = rank == last
    print("CP-MODULE-REPORT", json.dumps(report, sort_keys=True), flush=True)


def _counters(stats) -> dict:
    return {
        name: value
        for name, value in stats.as_dict().items()
        if name
        in (
            "rows",
            "litetopk_rows",
            "reference_rows",
            "bootstrap_rows",
            "tiles",
            "plans",
            "declined_tiles",
            "status_rows",
            "recomputed_tiles",
            "recomputed_rows",
            "rerun_groups",
        )
    }


@pytest.mark.gpus(2, min_architecture="blackwell")
def test_dsa_native_cp2_selects_like_cp1(monkeypatch):
    _run(2, monkeypatch)


@pytest.mark.gpus(4, min_architecture="blackwell")
def test_dsa_native_cp4_selects_like_cp1(monkeypatch):
    _run(4, monkeypatch)
