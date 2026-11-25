# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-token-loss mode of the microbatch loop (token sums, global token divisor)."""

from __future__ import annotations

import torch

from megatron.lite.primitive.train_step import run_microbatch_loop


def _setup():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1, bias=False, dtype=torch.float64)
    batches = []
    for tokens in (3, 7):
        x = torch.randn(tokens, 4, dtype=torch.float64)
        mask = torch.ones(tokens, dtype=torch.float64)
        mask[0] = 0
        batches.append({"x": x, "y": torch.randn(tokens, 1, dtype=torch.float64), "mask": mask})
    return model, batches


def _forward(per_token):
    def forward_fn(module, batch):
        err = (module(batch["x"]) - batch["y"]).square().squeeze(-1) * batch["mask"]
        num_tokens = batch["mask"].sum()
        loss = err.sum() if per_token else err.sum() / num_tokens
        return {"loss": loss, "num_tokens": num_tokens}

    return forward_fn


def test_single_microbatch_matches_default_mode_after_token_division():
    model, batches = _setup()
    run_microbatch_loop(model, iter(batches[:1]), 1, _forward(False))
    default_grad = model.weight.grad.clone()

    model.weight.grad = None
    out = run_microbatch_loop(model, iter(batches[:1]), 1, _forward(True), per_token_loss=True)
    torch.testing.assert_close(model.weight.grad / out["_num_tokens"], default_grad)


def test_multi_microbatch_gradient_is_global_token_mean():
    model, batches = _setup()
    out = run_microbatch_loop(model, iter(batches), 2, _forward(True), per_token_loss=True)
    assert float(out["_num_tokens"]) == 2 + 6
    grad = model.weight.grad / out["_num_tokens"]

    model.weight.grad = None
    total = sum(_forward(True)(model, b)["loss"] for b in batches) / 8
    total.backward()
    torch.testing.assert_close(grad, model.weight.grad)


def test_microbatch_without_token_count_is_skipped():
    model, batches = _setup()

    def forward_fn(module, batch):
        return {"loss": (module(batch["x"]) - batch["y"]).square().sum()}

    out = run_microbatch_loop(model, iter(batches), 2, forward_fn, per_token_loss=True)
    assert "_num_tokens" not in out


def test_pipeline_divisor_and_compact_token_count():
    from types import SimpleNamespace

    from megatron.lite.primitive.parallel.pipeline import _compact_pipeline_output, _loss_divisor

    assert _loss_divisor(SimpleNamespace(num_microbatches=4), 4) == 4
    assert _loss_divisor(SimpleNamespace(num_microbatches=4, per_token_loss=True), 4) == 1
    compact = _compact_pipeline_output({"loss": torch.tensor(3.0), "num_tokens": 7})
    assert compact["loss"] == 3.0 and int(compact["num_tokens"]) == 7
    assert "num_tokens" not in _compact_pipeline_output({"loss": torch.tensor(3.0)})


def test_single_microbatch_token_total_does_not_alias_the_model_output():
    # finalize_grads all-reduces the returned total in place; with one microbatch it must not be
    # the forward output's own tensor (a float tensor's .float() is not a copy).
    model, batches = _setup()
    seen = []

    def forward_fn(module, batch):
        out = _forward(True)(module, batch)
        seen.append(out["num_tokens"])
        return out

    out = run_microbatch_loop(model, iter(batches[:1]), 1, forward_fn, per_token_loss=True)
    out["_num_tokens"] += 100.0
    assert float(seen[0]) == 2.0
