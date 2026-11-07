# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Weights must be used through their own module's forward.

Megatron-Core DDP with ``overlap_param_gather`` waits for a bucket's parameter all-gather in a
forward pre-hook that only covers the parameters of the module being called. A weight read
functionally from a parent module is used before (or without) that wait, i.e. stale on the data-
parallel ranks that do not own its shard. The router gate and the embedding table therefore run
their computation in their own forward.
"""

from types import SimpleNamespace

import torch


def _uncalled_parameter_owners(model: torch.nn.Module, run) -> list[str]:
    called = set()
    handles = [
        m.register_forward_pre_hook(lambda module, *_: called.add(id(module)))
        for m in model.modules()
    ]
    try:
        run()
    finally:
        for handle in handles:
            handle.remove()
    return [
        name
        for name, module in model.named_modules()
        if any(True for _ in module.parameters(recurse=False)) and id(module) not in called
    ]


def test_sigmoid_router_gate_runs_as_its_own_module(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    config = SimpleNamespace(
        hidden_size=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        aux_loss_alpha=0.0,
    )
    ps = SimpleNamespace(tp_size=1, tp_group=None)
    x = torch.arange(32, dtype=torch.float32).view(4, 8) / 11
    # A float64 router GEMM takes the torch.mm path, so no Transformer Engine GEMM is needed.
    for router_dtype in (None, torch.float64):
        router = SigmoidTopKRouter(config, ps, compute_aux_loss=False, router_dtype=router_dtype)
        assert _uncalled_parameter_owners(router, lambda: router(x)) == []


def test_vocab_parallel_embedding_table_runs_as_its_own_module():
    from megatron.lite.primitive.parallel.linear import VocabParallelEmbedding

    ps = SimpleNamespace(tp_size=1, tp_rank=0, tp_group=None)
    embedding = VocabParallelEmbedding(16, 4, ps, deterministic=True)
    ids = torch.tensor([[1, 5, 15]])
    assert _uncalled_parameter_owners(embedding, lambda: embedding(ids)) == []
    out = embedding(ids)
    assert torch.equal(out[:, 0], embedding.embedding.weight[ids[0]])
