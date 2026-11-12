# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Attention-only CP2 vs CP1 (A configuration).

Every layer's attention (ratios 0/2/1, candidate pool, indexer top-k and the
cross-layer KV/index/candidate reuse) runs inside one source microbatch on the
same global random inputs and output gradients, without optimizer/DDP. CP2
ranks take contiguous row slices. Gate: ratio-0/1 outputs equal CP1 bitwise and
the no-grad forward; input, cached KV-source and every trainable attention
parameter gradient exist, are finite and nonzero. Reported only: ratio-2 output
rows (Core lays out CP1 and CP2 indices in different orders, so FlashMLA sums
the same KV in another order) and backward differences (reduction order).
"""

import json
from dataclasses import asdict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _ds41_fixture import tiny_config
from test_training import _build, _initialize
from torch.utils._pytree import tree_leaves

BOUNDS = (0, 8, 10, 96)  # documents of 8, 2, 86 rows; CP2 splits the last at row 48


def _worker(rank, root, cp):
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", f"file://{root}/store-{cp}", rank=rank, world_size=cp)
    torch.use_deterministic_algorithms(True, warn_only=True)
    _, _, model, ps = _build(root, cp=cp)
    _initialize(model, ps, root)
    bounds = torch.tensor(BOUNDS, dtype=torch.int32, device="cuda")
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=bounds,
        cu_seqlens_kv=bounds,
        cu_seqlens_q_padded=bounds,
        cu_seqlens_kv_padded=bounds,
        max_seqlen_q=86,
        max_seqlen_kv=86,
        cp_partition_mode="contiguous",
        cp_group=ps.cp_group,
        local_cp_size=cp,
    )
    total, width = BOUNDS[-1], model.config.hidden_size
    rows = slice(ps.cp_rank * total // cp, (ps.cp_rank + 1) * total // cp)
    layers = [model.layers[str(i)] for i in range(len(model.layers))]
    index_calls = {i: [] for i, ratio in enumerate(model.config.csa_compress_ratios) if ratio == 2}
    hooks = [
        layers[i].attention.core_attention.native.indexer.linear_wq_b.register_forward_hook(
            lambda module, args, output, i=i: index_calls[i].append(1)
        )
        for i in index_calls
    ]
    generator = torch.Generator(device="cuda").manual_seed(11)
    draw = lambda: torch.randn(total, 1, width, device="cuda", generator=generator)
    inputs = [draw().bfloat16()[rows].clone().requires_grad_() for _ in layers]
    grads = [draw().bfloat16()[rows] for _ in layers]

    def forward():
        return [
            layer.attention(x, attention_mask=None, packed_seq_params=params)[0]
            for layer, x in zip(layers, inputs)
        ]

    with torch.no_grad(), model.source_state.microbatch():
        plain = forward()
    with model.source_state.microbatch():
        outputs = forward()
        cached = [
            (k, v) for k, v in model.source_state.values.items() if k[0] != "attention_inputs"
        ]
        sources = [
            (k, t)
            for k, v in cached
            for t in tree_leaves(v)
            if torch.is_tensor(t) and t.grad_fn is not None
        ]
        for _, t in sources:
            t.retain_grad()
        torch.autograd.backward(outputs, grads)
    assert sources and all(torch.equal(a, b) for a, b in zip(plain, outputs)), "grad mode"
    for key, t in sources:
        geometry = key[1] if isinstance(key[1], tuple) else ()
        assert t.grad is not None or geometry[:1] in (("index_k",), ("ratio1_index_k",)), key
    for t in [x.grad for x in inputs] + [t.grad for _, t in sources if t.grad is not None]:
        assert torch.isfinite(t).all() and t.count_nonzero(), "input or source gradient"
    assert index_calls and all(len(v) == 2 for v in index_calls.values()), index_calls
    [hook.remove() for hook in hooks]
    weights = {}
    for index, layer in enumerate(layers):
        for name, value in layer.attention.named_parameters():
            if value.requires_grad:  # every trainable parameter must receive a gradient
                assert value.grad is not None, f"{index}.{name} has no gradient"
                grad = value.grad.detach().float().clone()
                dist.all_reduce(grad, group=ps.cp_group)  # CP partial sums
                assert torch.isfinite(grad).all() and grad.count_nonzero(), f"{index}.{name}"
                weights[f"{index}.{name}"] = grad.cpu()
    flat = lambda t: t.detach().reshape(t.shape[0], -1).cpu()
    torch.save(
        dict(out=[flat(o) for o in outputs], dx=[flat(x.grad) for x in inputs], dw=weights),
        f"{root}/cp{cp}-rank{ps.cp_rank}.pt",
    )
    dist.destroy_process_group()


def _run(root, cp):
    mp.start_processes(_worker, (str(root), cp), nprocs=cp, start_method="spawn")
    return [torch.load(root / f"cp{cp}-rank{r}.pt") for r in range(cp)]


@pytest.mark.gpus(2)
def test_attention_cp2_equals_cp1_bitwise(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    (tmp_path / "config.json").write_text(json.dumps(asdict(tiny_config())))
    (one,), two = _run(tmp_path, 1), _run(tmp_path, 2)
    ratios, gate, report = tiny_config().compress_ratios, [], {}
    for kind in ("out", "dx"):
        for layer, full in enumerate(one[kind]):
            joined = torch.cat([rank[kind][layer] for rank in two])
            rows = (joined != full).any(-1).nonzero().flatten().tolist()
            error = ((joined.float() - full.float()).norm() / full.float().norm()).item()
            report[f"{kind} {layer} ratio {ratios[layer]}"] = (rows[:8], len(rows), error)
            gate += [layer] if kind == "out" and rows and ratios[layer] != 2 else []
    for rank in two:
        assert one["dw"] and rank["dw"].keys() == one["dw"].keys()
    for name, full in one["dw"].items():
        report[f"dw {name}"] = ((two[0]["dw"][name] - full).norm() / full.norm()).item()
    print("ATTENTION_CP " + json.dumps(report))
    assert not gate, report
