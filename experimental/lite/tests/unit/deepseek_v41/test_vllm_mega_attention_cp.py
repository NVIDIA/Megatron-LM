# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Mega attention core CP2 vs CP1 (gates fixed before the first run).

CP ranks hold contiguous equal row slices; each all-gathers the pre-RoPE KV and
latent, attends its rows to the gathered KV and reduce-scatters dKV. Gates:
output bytes and dq equal CP1 bitwise (a row's arithmetic does not depend on
the partition and dq has no cross-row reduction); dswa / dlatent within FP32
atomic and reduce-scatter order (relative L2 <= 1e-5); the summed dsink
partials within FP32 summation order (relative L2 <= 1e-6).
"""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 GPUs"),
]

LAYOUTS = {"single": [0, 8320], "packed": [0, 1000, 1100, 1101, 3072]}


def _run(rank, root, ratio, docs, cp):
    import os

    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from types import SimpleNamespace

    from test_vllm_mega_attention import _inputs, _ours, _upstream

    torch.cuda.set_device(rank)
    group = None
    if cp > 1:
        dist.init_process_group("nccl", f"file://{root}/store", rank=rank, world_size=cp)
        group = dist.group.WORLD
    total = docs[-1]
    x = _inputs(total, ratio, seed=21, docs=docs)
    grad = _upstream(total, torch.Generator(device="cuda").manual_seed(8), "dense")
    rows = slice(rank * total // cp, (rank + 1) * total // cp)
    local = SimpleNamespace(**vars(x))
    for name in ("q", "swa_kv", "latent", "topk", "positions"):
        value = getattr(x, name)
        setattr(local, name, None if value is None else value[rows].contiguous())
    # FP32 KV leaves (bf16-representable values) keep dKV in FP32, so the bar
    # measures reduction order, not bf16 rounding flips of the gradient.
    leaves = {
        name: getattr(local, name).detach().float().clone().requires_grad_()
        for name in ("swa_kv", "sink") + (("latent",) if ratio else ())
    }
    leaves["q"] = local.q.detach().clone().requires_grad_()
    for name, value in leaves.items():
        setattr(local, name, value)
    local.docs = docs
    out = _ours(local, cp_group=group)
    torch.autograd.backward(out.grad, grad[rows])
    torch.save(
        dict(
            data=out.data.view(torch.uint8).cpu(),
            scale=out.scale.contiguous().cpu(),
            **{f"d{name}": value.grad.cpu() for name, value in leaves.items()},
        ),
        f"{root}/cp{cp}-rank{rank}.pt",
    )
    if group is not None:
        dist.destroy_process_group()


def _metric(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("layout", ["single", "packed"])
@pytest.mark.parametrize("ratio", [0, 2, 1])
def test_cp2_matches_cp1(tmp_path, ratio, layout):
    docs = LAYOUTS[layout]
    mp.start_processes(_run, (str(tmp_path), ratio, docs, 1), nprocs=1, start_method="spawn")
    mp.start_processes(_run, (str(tmp_path), ratio, docs, 2), nprocs=2, start_method="spawn")
    one = torch.load(tmp_path / "cp1-rank0.pt")
    two = [torch.load(tmp_path / f"cp2-rank{r}.pt") for r in range(2)]
    joined = lambda key: torch.cat([rank[key] for rank in two])
    assert torch.equal(joined("data"), one["data"])
    assert torch.equal(joined("scale"), one["scale"])
    assert torch.equal(joined("dq"), one["dq"])
    report = {"dsink": _metric(two[0]["dsink"] + two[1]["dsink"], one["dsink"])}
    for name in ("dswa_kv",) + (("dlatent",) if ratio else ()):
        report[name] = _metric(joined(name), one[name])
    print(f"MEGA_CP ratio={ratio} layout={layout} {report}")
    assert report["dsink"] <= 1e-6, report
    assert all(report[k] <= 1e-5 for k in report if k != "dsink"), report
