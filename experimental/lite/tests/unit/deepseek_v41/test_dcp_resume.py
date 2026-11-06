# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small native Adam DCP: exact restored state, three continued steps and negative controls."""
import copy
import json
import pickle
import random
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._pytree import keystr, tree_flatten_with_path

from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.primitive.ckpt import dcp as checkpoint
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig
from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context

TOKENS, ATOL, RTOL = 80, 0.05, 0.005
from _ds41_fixture import tiny_config


def _build(ep, device, root):
    config = tiny_config()
    adam = dict(lr=1e-5, min_lr=1e-5, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8)
    kwargs = dict(device=device, parallel=ParallelConfig(ep=ep), attention_backend_override="cudnn")
    kwargs.update(token_map=list(range(config.vocab_size)), optimizer_config=adam)
    bundle = protocol.build_model(config, impl_cfg=protocol.ImplConfig(optimizer="adam", **kwargs))
    from test_training import _initialize

    _initialize(bundle.chunks[0].module, bundle.parallel_state, root)
    bundle.optimizer.reload_model_params()
    schedulers = [torch.optim.lr_scheduler.StepLR(o, 1, 0.8) for o in bundle.optimizer.optimizers]
    return bundle, schedulers, kwargs["parallel"]


def _step(bundle, schedulers, drop_head_grad=False):
    offset = random.randrange(30) + int(np.random.randint(30)) + int(torch.randint(30, (1,)))
    ids = (torch.randint(1, 32, (TOKENS,)) + offset % 30).cuda()
    batch = PackedBatch(ids, ids, torch.tensor([TOKENS]).cuda())
    bundle.chunks[0].zero_grad_buffer(), bundle.optimizer.zero_grad()  # as the runtime's zero_grad
    with use_loss_context(LossContext(normalization_denominator=float(TOKENS))):
        loss = bundle.forward_step(bundle.chunks[0], batch)["loss"]
    assert torch.isfinite(loss), "RESUME_LOSS"
    loss.backward()
    grad = bundle.chunks[0].module.head.weight.main_grad
    grad.zero_() if drop_head_grad else None  # control: fused wgrad accumulation bypasses hooks
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0, "RESUME_GRADIENT"
    bundle.finalize_grads()
    accepted, norm = bundle.optimizer.step()[:2]
    assert accepted and torch.isfinite(torch.as_tensor(norm)), "RESUME_UPDATE"
    for scheduler in schedulers:
        scheduler.step()
    return ids.cpu(), float(loss)


def _state(bundle, schedulers):
    chunk, inner = bundle.chunks[0], bundle.optimizer.optimizers
    tensors = {k: v.detach().cpu() for k, v in chunk.state_dict().items() if torch.is_tensor(v)}
    shards = [p.detach().cpu() for o in inner for g in o.param_groups for p in g["params"]]
    optimizer = [bundle.optimizer.state_dict(), [o.state_dict() for o in inner], shards]
    rng = [pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state())]
    rng += [torch.get_rng_state(), torch.cuda.get_rng_state()]
    schedules = [s.state_dict() for s in schedulers]
    return copy.deepcopy(dict(model=tensors, optimizer=optimizer, schedulers=schedules, rng=rng))


def _difference(actual, expected):  # the first differing leaf; restore is lossless (bitwise)
    def same(a, e):
        if not (torch.is_tensor(a) and torch.is_tensor(e)):
            return not (torch.is_tensor(a) or torch.is_tensor(e)) and a == e
        bits = lambda t: t.contiguous().reshape(-1).view(torch.uint8)
        return (a.dtype, a.shape) == (e.dtype, e.shape) and torch.equal(bits(a), bits(e))

    a, e = tree_flatten_with_path(actual)[0], tree_flatten_with_path(expected)[0]
    if [k for k, _ in a] != [k for k, _ in e]:
        return "keys"
    return next((keystr(k) for (k, x), (_, y) in zip(a, e) if not same(x, y)), None)


def _replicas_differ(chunk):  # dense parameters whose data-parallel replicas disagree
    dense = [(n, p) for n, p in chunk.named_parameters() if "experts" not in n]
    x = torch.stack([torch.stack([p.double().sum(), p.double().abs().max()]) for _, p in dense])
    every = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(every, x)
    return [n for i, (n, _) in enumerate(dense) if any(not torch.equal(g[i], x[i]) for g in every)]


def _worker(rank, root, ep, phase):
    torch.cuda.set_device(rank)
    root, record, timeout = Path(root), {}, timedelta(seconds=600)
    dist.init_process_group("nccl", f"file://{root}/store-{phase}", timeout, ep, rank)
    try:
        random.seed(915), np.random.seed(915), torch.manual_seed(915)
        bundle, schedulers, parallel = _build(ep, f"cuda:{rank}", root)
        chunk, path, control = bundle.chunks[0], str(root / "ckpt"), root / f"ctl-{rank}.pt"
        options = dict(config=parallel, ps=bundle.parallel_state)
        load = lambda: checkpoint.load_training_checkpoint(chunk, bundle.optimizer, path, **options)
        if phase == "continuous":
            _step(bundle, schedulers)
            checkpoint.save_training_checkpoint(chunk, bundle.optimizer, 1, path, **options)
            record["replicas"] = _replicas_differ(chunk)  # every replica saved the same values
            saved = _state(bundle, schedulers)
            steps = [_step(bundle, schedulers) for _ in range(3)]
            torch.save(dict(saved=saved, steps=steps), control)
        else:
            control, record["step"] = torch.load(control, weights_only=False), int(load())
            [s.load_state_dict(x) for s, x in zip(schedulers, control["saved"]["schedulers"])]
            record["restore"] = _difference(_state(bundle, schedulers), control["saved"])
            steps = [_step(bundle, schedulers) for _ in range(3)]
            pairs = list(zip(steps, control["steps"]))
            record["batches"] = all(torch.equal(s[0], c[0]) for s, c in pairs)
            record["losses"] = [[s[1], c[1]] for s, c in pairs]
            load()  # controls: a corrupted load and a vanished head gradient are caught
            chunk.module.embed.weight.data.view(-1)[0].add_(1)
            record["corrupt"] = _difference(_state(bundle, schedulers), control["saved"])
            load()
            try:
                _step(bundle, schedulers, drop_head_grad=True)
            except AssertionError as error:
                record["zero_gradient"] = str(error)
    except Exception:
        record["error"] = traceback.format_exc()[-6000:]
    finally:
        (root / f"{phase}-{rank}.json").write_text(json.dumps(record))
        dist.destroy_process_group()


@pytest.mark.parametrize("ep", [pytest.param(e, marks=pytest.mark.gpus(e)) for e in (1, 2)])
def test_dcp_resume_restores_state_and_follows_the_control(tmp_path, ep):
    records = {}
    for ph in ("continuous", "resume"):
        mp.start_processes(_worker, (str(tmp_path), ep, ph), nprocs=ep, start_method="spawn")
        records[ph] = [json.loads((tmp_path / f"{ph}-{r}.json").read_text()) for r in range(ep)]
        assert not any("error" in r for r in records[ph]), records[ph]
    print("RESUME_CONTRACT", json.dumps(dict(ep=ep, **records)), flush=True)
    assert not any(r["replicas"] for r in records["continuous"]), records["continuous"]
    for r in records["resume"]:
        assert r["step"] == 1 and r["restore"] is None and r["batches"], r
        assert all(abs(a - c) <= ATOL + RTOL * abs(c) for a, c in r["losses"]), r
        assert "embed.weight" in str(r["corrupt"]) and "RESUME_GRADIENT" in r["zero_gradient"], r
