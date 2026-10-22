# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Duplicated expert batching, singleton fallbacks, and optimizer-state compatibility."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("emerging_optimizers")

from megatron.core.optimizer import emerging_optimizers as eo
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def _parameters(count=7, shape=(4, 6), *, dtype=torch.float32):
    """Create independently initialized expert matrices."""
    params = [torch.nn.Parameter(torch.randn(shape, dtype=dtype)) for _ in range(count)]
    for param in params:
        param.expert_tp = True
    return params


def _optimizer(params, **kwargs):
    """Use full-precision matmuls to compare the two NS execution paths."""
    options = dict(lr=0.01, momentum=0.9, fp32_matmul_prec="highest")
    options.update(kwargs)
    return eo.TensorParallelMuon(params, **options)


def test_batch_size_one_delegates_to_existing_step(monkeypatch):
    """The default retains the exact inherited implementation, including closure handling."""
    optimizer = _optimizer(_parameters(), ns_batch_size=1)
    sentinel = object()
    closure = lambda: 1.0

    def original_step(self, provided_closure):
        assert self is optimizer
        assert provided_closure is closure
        return sentinel

    monkeypatch.setattr(eo.OrthogonalizedOptimizer, "step", original_step)
    assert optimizer.step(closure) is sentinel


@pytest.mark.parametrize("mode,syrk", [("distributed", False), ("duplicated", True)])
def test_layer_sharded_fallback_keeps_the_inherited_step(monkeypatch, mode, syrk):
    """A child optimizer can retain its own batch limit without enabling this NS path."""
    optimizer = _optimizer(_parameters(), ns_batch_size=1, tp_mode=mode)
    # LayerShardedMuon calls the parent constructor with batch size one, then stores
    # its own batch limit and can delegate to the parent before homes are assigned.
    optimizer.ns_batch_size = 3
    optimizer.use_syrk = syrk
    sentinel = object()
    monkeypatch.setattr(eo.OrthogonalizedOptimizer, "step", lambda *_args: sentinel)
    assert optimizer.step() is sentinel


@pytest.mark.parametrize("mode", ["auto", "distributed", "blockwise"])
def test_unsupported_modes_are_rejected(mode):
    """Duplicated batching cannot bypass a requested distributed or blockwise update."""
    with pytest.raises(ValueError, match="requires tp_mode='duplicated'"):
        _optimizer(_parameters(), ns_batch_size=2, tp_mode=mode)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_nonpositive_batch_size_is_rejected(batch_size):
    """Invalid limits fail before optimizer state is allocated."""
    with pytest.raises(ValueError, match="ns_batch_size must be at least 1"):
        _optimizer(_parameters(), ns_batch_size=batch_size)


def test_batched_ns_requires_supported_dependency(monkeypatch):
    """Older Emerging Optimizers versions cannot accept three-dimensional NS inputs."""
    monkeypatch.setattr(eo, "is_emerging_optimizers_min_version", lambda _: False)
    with pytest.raises(ImportError, match="requires emerging_optimizers >= 0.3.0"):
        _optimizer(_parameters(), ns_batch_size=2)
    assert _optimizer(_parameters(), ns_batch_size=1).ns_batch_size == 1


def test_duplicated_batching_rejects_syrk():
    """The duplicated batch path uses batched GEMMs rather than SYRK kernels."""
    with pytest.raises(ValueError, match="does not support use_syrk"):
        _optimizer(_parameters(), ns_batch_size=2, use_syrk=True)


def test_batched_step_rejects_closure_before_updating():
    """An unsupported closure leaves weights and momentum untouched."""
    params = _parameters()
    before = [p.detach().clone() for p in params]
    optimizer = _optimizer(params, ns_batch_size=3)
    with pytest.raises(ValueError, match="closure is not supported"):
        optimizer.step(lambda: 1.0)
    for param, original in zip(params, before):
        torch.testing.assert_close(param, original, rtol=0, atol=0)
    assert not optimizer.state


def test_batches_are_bounded_and_grouped_by_shape_dtype_and_optimizer_group(monkeypatch):
    """Batch limits and group boundaries keep mathematically independent updates separate."""
    params = _parameters()
    other_shape = _parameters(2, (3, 5))
    other_dtype = _parameters(2, dtype=torch.float64)
    separate_group = _parameters(2)
    params[-1].grad = None
    for index, param in enumerate(params[:-1] + other_shape + other_dtype + separate_group):
        param.grad = torch.full_like(param, index + 1.0)
    calls = []

    def capture_batch(values, **_kwargs):
        """Record batched inputs without imposing the real NS float32 restriction."""
        calls.append((tuple(values.shape), values.dtype))
        return values

    monkeypatch.setattr(eo, "newton_schulz", capture_batch)
    optimizer = _optimizer(
        [{"params": params + other_shape + other_dtype}, {"params": separate_group, "lr": 0.02}],
        ns_batch_size=3,
    )
    optimizer.step()
    assert calls == [
        ((3, 4, 6), torch.float32),
        ((3, 4, 6), torch.float32),
        ((2, 3, 5), torch.float32),
        ((2, 4, 6), torch.float64),
        ((2, 4, 6), torch.float32),
    ]
    assert params[-1] not in optimizer.state


@pytest.mark.parametrize("kind", ["dense", "qkv", "tp", "gtp", "unknown_tp"])
def test_incompatible_matrices_use_per_parameter_path(monkeypatch, kind):
    """Weights that need splitting or reconstruction retain their existing dispatch."""
    params = _parameters(2)
    group = SimpleNamespace()
    groups = SimpleNamespace(expt_tp=group, expt_gtp_remat=group)
    monkeypatch.setattr(eo, "get_pg_size", lambda _: 2)
    monkeypatch.setattr(eo, "resolve_gtp_remat_group", lambda *_args, **_kwargs: group)
    for param in params:
        param.grad = torch.ones_like(param)
        if kind == "dense":
            param.expert_tp = False
        elif kind == "tp" or kind == "unknown_tp":
            param.partition_dim = 0
        elif kind == "gtp":
            param.is_gtp_weight_remat = True
    optimizer = _optimizer(
        params,
        ns_batch_size=2,
        split_qkv=kind == "qkv",
        is_qkv_fn=lambda _: True,
        pg_collection=None if kind == "unknown_tp" else groups,
    )
    singletons = []

    def singleton(param, grad, **_kwargs):
        """Record the parameters routed to the existing orthogonalization method."""
        singletons.append(param)
        return grad

    def unexpected_batch(_grads):
        """Fail if a collective or QKV split would be skipped."""
        pytest.fail("Incompatible matrices must not bypass the per-parameter path")

    monkeypatch.setattr(optimizer, "orthogonalize", singleton)
    monkeypatch.setattr(optimizer, "_orthogonalize_expert_batch", unexpected_batch)
    optimizer.step()
    assert [id(p) for p in singletons] == [id(p) for p in params]


def test_gtp_eligibility_uses_the_same_group_resolver_as_orthogonalization(monkeypatch):
    """A partial collection cannot hide an active GTP axis from batch eligibility."""
    param = _parameters(1)[0]
    param.is_gtp_weight_remat = True
    # A partial collection must resolve its missing GTP field through the existing helper.
    collection = SimpleNamespace(expt_tp=None)
    resolved_group = object()
    calls = []

    def resolve(collection_arg, is_expert):
        """Model the existing resolver's fallback for a partial collection."""
        calls.append((collection_arg, is_expert))
        return resolved_group

    monkeypatch.setattr(eo, "resolve_gtp_remat_group", resolve)
    monkeypatch.setattr(eo, "get_pg_size", lambda _: 2)
    optimizer = _optimizer([param], ns_batch_size=2, pg_collection=collection)
    assert optimizer._expert_batch_key(param, torch.ones_like(param)) is None
    assert calls == [(collection, True)]


@pytest.mark.parametrize("partition_dim", [None, -1, 0, 1])
def test_single_rank_sharding_metadata_still_allows_batching(monkeypatch, partition_dim):
    """TP and GTP metadata do not disqualify a matrix when both axes have one rank."""
    param = _parameters(1)[0]
    param.partition_dim = partition_dim
    param.is_gtp_weight_remat = True
    group = object()
    collection = SimpleNamespace(expt_tp=group, expt_gtp_remat=group)
    monkeypatch.setattr(eo, "get_pg_size", lambda _: 1)
    monkeypatch.setattr(eo, "resolve_gtp_remat_group", lambda *_args, **_kwargs: group)
    optimizer = _optimizer([param], ns_batch_size=2, pg_collection=collection)
    assert optimizer._expert_batch_key(param, torch.ones_like(param)) is not None


def test_batching_preserves_weight_update_hooks_and_their_matmul_context():
    """Subclass hooks retain their pre/post ordering and caller precision context."""
    calls = []

    class HookedMuon(eo.TensorParallelMuon):
        """Use nontrivial hooks so skipping either hook changes the numerical result."""

        def pre_weight_update_fn_inplace(self, param, update):
            """Rescale the orthogonalized update before applying it."""
            calls.append(("pre", id(param), torch.get_float32_matmul_precision()))
            update.mul_(0.5)

        def post_weight_update_fn_inplace(self, param):
            """Apply a weight transform after the update."""
            calls.append(("post", id(param), torch.get_float32_matmul_precision()))
            param.mul_(0.99)

    reference = _parameters(3)
    batched = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    for original, candidate in zip(reference, batched):
        candidate.expert_tp = True
        original.grad = torch.randn_like(original)
        candidate.grad = original.grad.clone()
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        for params, limit in [(reference, 1), (batched, 3)]:
            optimizer = HookedMuon(params, ns_batch_size=limit, fp32_matmul_prec="highest")
            optimizer.step()
    finally:
        torch.set_float32_matmul_precision(precision)
    for original, candidate in zip(reference, batched):
        torch.testing.assert_close(candidate, original, rtol=1e-5, atol=1e-6)
    assert calls == [
        (kind, id(param), "high")
        for params in (reference, batched)
        for param in params
        for kind in ("pre", "post")
    ]


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("shape", [(4, 6), (6, 4)])
def test_multiple_steps_match_per_parameter_updates_and_reload(nesterov, decoupled, shape):
    """Compare repeated updates across decay modes and a batch-size-changing reload."""
    torch.manual_seed(123)
    reference = _parameters(shape=shape)
    batched = [torch.nn.Parameter(p.detach().clone()) for p in reference]
    for param in batched:
        param.expert_tp = True
    options = dict(
        nesterov=nesterov,
        use_decoupled_weight_decay=decoupled,
        weight_decay=0.1,
        scale_mode="unit_rms_norm",
        extra_scale_factor=0.7,
    )
    original_optimizer = _optimizer(reference, ns_batch_size=1, **options)
    batched_optimizer = _optimizer(batched, ns_batch_size=3, **options)
    for step in range(4):
        for original, candidate in zip(reference, batched):
            grad = torch.randn_like(original)
            original.grad = grad.clone()
            candidate.grad = grad.clone()
        original_optimizer.step()
        batched_optimizer.step()
        for original, candidate in zip(reference, batched):
            torch.testing.assert_close(candidate, original, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                batched_optimizer.state[candidate]["momentum_buffer"],
                original_optimizer.state[original]["momentum_buffer"],
                # Coupled decay feeds prior weight rounding back into subsequent momentum.
                rtol=0 if decoupled else 1e-5,
                atol=0 if decoupled else 1e-6,
            )
        if step == 1:
            saved = copy.deepcopy(batched_optimizer.state_dict())
            assert saved["param_groups"] == original_optimizer.state_dict()["param_groups"]
            assert "ns_batch_size" not in saved["param_groups"][0]
            assert all(set(state) == {"momentum_buffer"} for state in saved["state"].values())
            # Batch size is a launch choice, not part of the checkpoint state contract.
            reloaded = _optimizer(batched, ns_batch_size=2, **options)
            reloaded.load_state_dict(saved)
            batched_optimizer = reloaded


def test_config_dispatch_keeps_batching_out_of_adaptive_muon():
    """Reflective constructor kwargs expose batching only to ordinary Muon."""
    model = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=4, num_query_groups=2, kv_channels=4)
    )
    muon = OptimizerConfig(optimizer="muon", muon_ns_batch_size=3)
    kwargs = eo._muon_config_to_kwargs(muon, [model], pg_collection=None)
    assert kwargs["ns_batch_size"] == 3
    adaptive = OptimizerConfig(optimizer="adaptive_muon")
    kwargs = eo._adaptive_muon_config_to_kwargs(adaptive, [model], pg_collection=None)
    assert "ns_batch_size" not in kwargs
    assert set(kwargs) <= set(inspect.signature(eo.TensorParallelAdaptiveMuon.__init__).parameters)
    legacy_kwargs = eo._adaptive_muon_config_to_kwargs(SimpleNamespace(), [model], None)
    assert "ns_batch_size" not in legacy_kwargs


@pytest.mark.parametrize("optimizer", ["adam", "adaptive_muon"])
def test_optimizer_config_rejects_batching_for_other_optimizers(optimizer):
    """Unsupported optimizers reject the option instead of silently ignoring it."""
    with pytest.raises(ValueError, match="requires optimizer='muon'"):
        OptimizerConfig(optimizer=optimizer, muon_ns_batch_size=2)


@pytest.mark.parametrize("mode", ["auto", "distributed", "blockwise"])
def test_optimizer_config_rejects_unsupported_modes(mode):
    """Library configuration applies the same mode guard as optimizer construction."""
    with pytest.raises(ValueError, match="requires duplicated or layer_sharded"):
        OptimizerConfig(optimizer="muon", muon_ns_batch_size=2, muon_tp_mode=mode)


@pytest.mark.parametrize("mode", ["duplicated", "layer_sharded"])
def test_optimizer_config_accepts_supported_batching_modes(mode):
    """Duplicated batching extends the existing layer-sharded configuration option."""
    config = OptimizerConfig(
        optimizer="muon",
        muon_tp_mode=mode,
        muon_ns_batch_size=32,
        use_layer_wise_distributed_optimizer=True,
        muon_split_qkv=False,
    )
    assert config.muon_ns_batch_size == 32
