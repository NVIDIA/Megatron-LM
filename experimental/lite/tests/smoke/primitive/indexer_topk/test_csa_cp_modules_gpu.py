# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Lite CSA THD context parallelism with indexer top-k bindings on 4 GPUs (optional).

Every rank builds the same DeepSeek-V4-sized C4 CSA layer (random weights) as rank r of a
contiguous CP group of four and runs a 262144-token THD prompt (65536 local rows: Megatron Core's
fused THD RoPE handles that many rows of 64 x 512 queries) through ``_forward_thd_packed``, as one
sequence and as two packed sequences, once with a reference binding and once with a LiteTopK
binding (the first MXFP4 entry of ``LITETOPK_TEST_SELECTORS``; the exact-tie top-k from
``LITETOPK_TEST_EXACT_TOPK``; as ``test_selector_gpu.py``). On every rank:

* both bindings receive the same operands, and the layout of the rank's rows that the THD
  dispatch point builds (sequence starts, positions and compressed-key ranges equal to Megatron
  Core's compaction metadata);
* the reference binding's selection equals ``reference_topk`` on those operands and that layout;
* the LiteTopK binding selects tiles on the ranks whose rows reach the plan's start position, and
  every row it selects differently from the reference binding is a near tie (DESIGN-REVISIONS M7:
  float64 distance of the swapped keys from the row's cutoff score within the MXFP4 near-tie
  limit of ``test_cp_sim_gpu.py``);
* both outputs are finite, and every rank issues the same collectives in every pass.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.optional

_SELECTORS_VARIABLE = "LITETOPK_TEST_SELECTORS"
_EXACT_VARIABLE = "LITETOPK_TEST_EXACT_TOPK"
_TOKENS = 262144
# Two packed sequences with the boundary inside rank 1. Neither has the 65536 compressed keys the
# MXFP4 route needs, so both bindings select them with the reference selector: this layout checks
# the rows and keys of sequences that cross ranks.
_PACKED = [0, 100004, _TOKENS]
# As test_cp_sim_gpu.py: the farthest the fast MXFP4 slab route may swap a key from a row's cutoff
# score, relative to that score (DESIGN-REVISIONS M7).
_NEAR_TIE_LIMIT = 1.1e-3
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _read_json(variable: str):
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return json.loads(raw if raw[0] in "[{" else Path(raw).read_text(encoding="utf-8"))


def _plugin_or_skip():
    selectors = _read_json(_SELECTORS_VARIABLE) or []
    exact = _read_json(_EXACT_VARIABLE)
    spec = next((s for s in selectors if s["native_format"] == "mxfp4"), None)
    if spec is None or exact is None:
        pytest.skip(f"{_SELECTORS_VARIABLE} (MXFP4 entry) and {_EXACT_VARIABLE} are not set")
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
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=300))
    if dist.get_world_size() != expected:
        pytest.skip(f"needs {expected} ranks, got {dist.get_world_size()}")
    return dist


def _layer(device, world: int, rank: int, group):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention

    config = DeepseekV4Config(
        hidden_size=4096,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        compress_ratios=[4],
        sliding_window=128,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        rotary_scaling_factor=16.0,
        original_max_position_embeddings=65536,
        num_hidden_layers=1,
        num_nextn_predict_layers=1,
    )
    torch.manual_seed(20260930)
    ps = SimpleNamespace(cp_size=world, cp_rank=rank, cp_group=group)
    module = CompressedSparseAttention(config, layer_idx=0, ps=ps)
    return module.to(device, torch.bfloat16).eval()


def _dequantize_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Exact float64 values of packed E2M1 rows ``[..., 64]`` with UE8M0 group scales ``[...]``."""
    nibbles = codes.view(torch.uint8).to(torch.int64)
    code = torch.stack((nibbles & 0xF, nibbles >> 4), dim=-1).flatten(-2)
    grid = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    value = torch.where((code & 0x8) != 0, -grid[code & 0x7], grid[code & 0x7])
    exponents = torch.stack(
        [(scales.to(torch.int64) >> shift) & 0xFF for shift in (0, 8, 16, 24)], dim=-1
    )
    group_scale = torch.exp2(exponents.to(torch.float64) - 127.0)
    return (value.unflatten(-1, (4, 32)) * group_scale[..., None]).flatten(-2)


def _max_near_tie_distance(call: dict, candidate, reference, chunk: int = 32) -> tuple[int, float]:
    """Rows where ``candidate`` and ``reference`` differ, and the largest relative distance from a
    row's cutoff (the smallest reference score) of a swapped key, in float64 scores of the
    quantized operands of the call."""
    from megatron.lite.primitive.kernels.indexer_topk.reference import (
        quantize_keys,
        quantize_queries,
    )

    keys = quantize_keys(call["k"], "mxfp4")
    keys = _dequantize_mxfp4(keys.data, keys.scale)
    layout = call["layout"]
    shift = torch.zeros(candidate.shape[0], dtype=torch.int64, device=candidate.device)
    for segment in layout.segments:
        shift[segment.row_start : segment.row_end] = segment.key_start - segment.index_base
    rows = torch.nonzero((candidate != reference).any(dim=1)).flatten()
    topk, largest = reference.shape[1], 0.0
    for first in range(0, rows.numel(), chunk):
        block = rows[first : first + chunk]
        data, scales, folded = quantize_queries(
            call["q"][block],
            call["weights"][block],
            "mxfp4",
            softmax_scale=call["softmax_scale"],
            kernel_heads=call["q"].shape[1],
        )
        ids = torch.cat((reference[block], candidate[block]), dim=1).to(torch.int64)
        valid = ids >= 0
        key_rows = torch.where(valid, ids + shift[block, None], torch.zeros_like(ids))
        dots = torch.einsum("rhd,rkd->rkh", _dequantize_mxfp4(data, scales), keys[key_rows])
        scores = (torch.relu(dots) * folded.to(torch.float64)[:, None, :]).sum(dim=-1)
        ref_ids, cand_ids = ids[:, :topk], ids[:, topk:]
        in_ref = (cand_ids[:, :, None] == ref_ids[:, None, :]).any(-1) & valid[:, topk:]
        in_cand = (ref_ids[:, :, None] == cand_ids[:, None, :]).any(-1) & valid[:, :topk]
        cutoff = scores[:, :topk].masked_fill(~valid[:, :topk], float("inf")).min(dim=1).values
        swapped = torch.cat((valid[:, :topk] & ~in_cand, valid[:, topk:] & ~in_ref), dim=1)
        relative = (scores - cutoff[:, None]).abs() / cutoff.abs().clamp_min(1e-30)[:, None]
        largest = max(largest, float(relative.masked_fill(~swapped, 0.0).max()))
    return int(rows.numel()), largest


@pytest.mark.gpus(4, min_architecture="blackwell")
def test_csa_thd_cp4_litetopk_matches_reference(monkeypatch):
    spec, exact = _plugin_or_skip()
    dist = _world_or_skip(4)
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
    from megatron.lite.primitive.kernels.indexer_topk import (
        ExactTopKConfig,
        IndexerTopKConfig,
        LiteTopKPluginConfig,
        QueryLayout,
        reference_topk,
    )
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.modules.attention.indexer_topk import configure_indexer_topk

    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    local = _TOKENS // world
    module = _layer(device, world, rank, dist.group.WORLD)
    generator = torch.Generator(device=device).manual_seed(5)
    x = torch.randn(1, _TOKENS, module.config.hidden_size, generator=generator, device=device)
    x = x[:, rank * local : (rank + 1) * local].to(torch.bfloat16).contiguous()
    positions = torch.arange(local, device=device).unsqueeze(0)

    collectives, metadata = [], []
    gather = csa.gather_from_sequence_parallel_region
    exchange = cp_utils.exchange_cp_boundary_hidden
    prepare = cp_utils.prepare_cp_compressor_input

    def counted_gather(tensor, *args, **kwargs):
        collectives.append(("all_gather", tuple(tensor.shape)))
        return gather(tensor, *args, **kwargs)

    def counted_exchange(tensor, *args, **kwargs):
        collectives.append(("boundary", tuple(tensor.shape)))
        return exchange(tensor, *args, **kwargs)

    def recorded_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        metadata.append(result[5].tolist())  # cu_seqlens_compressed of the whole prompt
        return result

    monkeypatch.setattr(csa, "gather_from_sequence_parallel_region", counted_gather)
    monkeypatch.setattr(cp_utils, "exchange_cp_boundary_hidden", counted_exchange)
    monkeypatch.setattr(cp_utils, "prepare_cp_compressor_input", recorded_prepare)

    def check_ranks(entry, label):
        gathered = [None] * world
        dist.all_gather_object(gathered, entry)
        assert all(value == gathered[0] for value in gathered), (label, gathered)

    plugin = LiteTopKPluginConfig(**spec["litetopk"])
    exact_topk = ExactTopKConfig(**exact)
    report = {"rank": rank, "layouts": {}}
    for name, offsets in (("single", [0, _TOKENS]), ("packed", _PACKED)):
        cu = torch.tensor(offsets, dtype=torch.int32, device=device)
        lengths = [end - start for start, end in zip(offsets, offsets[1:])]
        packed = SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv=cu,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        )
        arms = {}
        for backend in ("reference", "litetopk"):
            installation = configure_indexer_topk(
                [module],
                IndexerTopKConfig(
                    backend=backend,
                    precision=spec["precision"],
                    litetopk=plugin,
                    exact_topk=exact_topk,
                ),
                native_format="mxfp4",
            )
            (binding,) = installation.bindings
            calls = []
            select = binding.select

            def recording_select(*args, _select=select, _calls=calls, **kwargs):
                result = _select(*args, **kwargs)
                _calls.append(dict(q=args[0], k=args[1], weights=args[2], result=result, **kwargs))
                return result

            binding.select = recording_select
            collectives.clear()
            metadata.clear()
            with torch.no_grad():
                output = module(x, position_ids=positions, packed_seq_params=packed)
            (call,) = calls
            check_ranks(collectives, f"{name}/{backend}")
            arms[backend] = dict(call=call, output=output, stats=binding.stats.as_dict())
            module.set_indexer_topk(None)

        reference_call = arms["reference"]["call"]
        lite_call = arms["litetopk"]["call"]
        # The same operands for both bindings, and the rank's rows as Core's metadata lays them out.
        for operand in ("q", "k", "weights"):
            assert torch.equal(reference_call[operand], lite_call[operand]), (name, operand)
        layout = reference_call["layout"]
        assert lite_call["layout"] == layout
        assert layout == QueryLayout.packed(
            offsets, row_start=rank * local, rows=local, key_ratio=4, absolute_ids=False
        )
        (compressed,) = set(map(tuple, metadata))
        for segment in layout.segments:
            sequence = max(
                i for i in range(len(offsets) - 1) if offsets[i] <= rank * local + segment.row_start
            )
            assert segment.key_start == compressed[sequence]
            assert segment.key_count == compressed[sequence + 1] - compressed[sequence]
        # The reference binding is the matched-precision reference selector on those operands.
        expected = reference_topk(
            reference_call["q"],
            reference_call["k"],
            reference_call["weights"],
            layout=layout,
            topk=reference_call["topk"],
            softmax_scale=reference_call["softmax_scale"],
            fmt="mxfp4",
            exact_topk=exact_topk,
        )
        assert torch.equal(reference_call["result"], expected), name
        # LiteTopK: tiles on the ranks past the plan's start, near ties only.
        stats = arms["litetopk"]["stats"]
        rows_differ, distance = _max_near_tie_distance(
            lite_call, lite_call["result"], reference_call["result"]
        )
        if spec["precision"] == "exact":
            assert rows_differ == 0, (name, rows_differ)
        assert distance <= _NEAR_TIE_LIMIT, (name, rows_differ, distance)
        outputs = [arms[arm]["output"].float() for arm in ("reference", "litetopk")]
        assert all(bool(torch.isfinite(output).all()) for output in outputs)
        report["layouts"][name] = {
            "litetopk_tiles": stats["tiles"],
            "litetopk_rows": stats["litetopk_rows"],
            "reference_rows": stats["reference_rows"],
            "status_rows": stats["status_rows"],
            "rows_differ": rows_differ,
            "max_near_tie_distance": distance,
            "output_rel_l2": float((outputs[1] - outputs[0]).norm() / outputs[0].norm()),
            "collectives": len(collectives),
        }
    tiles = [None] * world
    dist.all_gather_object(tiles, report["layouts"]["single"]["litetopk_tiles"])
    # One sequence: LiteTopK tiles start at position 180224, inside rank 2.
    assert [count > 0 for count in tiles] == [False, False, True, True], tiles
    dist.barrier()
    print("CSA-CP-MODULE-REPORT", json.dumps(report, sort_keys=True), flush=True)
