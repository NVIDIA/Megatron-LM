# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Source-selection and routing contracts against the installed Core version."""
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa import Compressor
from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router_replay import RouterReplay
from megatron.lite.model.deepseek_v41.lite.compressor import DS41Compressor
from megatron.lite.model.deepseek_v41.lite.moe import SortedTopKPolicy, bind_sorted_selection
from megatron.lite.model.deepseek_v41.lite.state import KVSourceState


def _compressor(state, layer, owner, group):
    # Core's constructor allocates CUDA parameters. This fixture tests dispatch,
    # not native compression numerics; the delegation spy is explicit below.
    result = DS41Compressor.__new__(DS41Compressor)
    torch.nn.Module.__init__(result)
    result.compress_ratio, result.head_dim = 4, 4
    result.pg_collection = SimpleNamespace(cp=group)
    result.bind_source(state, layer, owner)
    return result


@pytest.mark.parametrize("name", ["forward", "_forward_thd_cp"])
def test_core_override_signature(name):
    def signature(fn):
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    assert signature(getattr(DS41Compressor, name)) == signature(getattr(Compressor, name))


@pytest.mark.parametrize("cp", [False, True])
def test_owner_reader_dispatch_graph_and_lifetime(cp):
    state, group = KVSourceState(), object()
    owner = _compressor(state, 0, 0, group)
    reader = _compressor(state, 1, 0, group)
    x = torch.arange(12.0).reshape(3, 1, 4).requires_grad_()
    prefix = torch.tensor([0, 3], dtype=torch.int32)
    params = SimpleNamespace(cu_seqlens_q=prefix, cu_seqlens_q_padded=prefix, max_seqlen_q=3)
    name = "_forward_thd_cp" if cp else "forward"

    def produce(self, tensor, *args, **kwargs):
        assert self is owner
        return tensor.square(), prefix

    def run(module):
        return (
            module._forward_thd_cp(x, x, prefix, SimpleNamespace(global_start=0), 3)
            if cp
            else module(x, params)
        )

    with patch.object(Compressor, name, produce):
        with state.microbatch():
            value = run(owner)
            assert run(reader) is value
            value[0].sum().backward()
            assert torch.equal(x.grad, 2 * x.detach())
        with state.microbatch(), pytest.raises(RuntimeError, match="owner"):
            run(reader)
    assert not state.values and not state.active
    with pytest.raises(RuntimeError, match="active microbatch"):
        run(reader)


def test_owner_mode_mismatch_and_failure_cleanup():
    state = KVSourceState()
    with pytest.raises(ValueError, match="abort"):
        with state.microbatch():
            state.resolve(0, 0, ("no-grad",), lambda: torch.ones(1))
            with pytest.raises(RuntimeError, match="owner"):
                state.resolve(1, 0, ("grad",), lambda: None)
            with pytest.raises(RuntimeError, match="already active"):
                with state.microbatch():
                    pass
            raise ValueError("abort")
    assert not state.active and state.values == {}


@pytest.mark.parametrize("score", ["sigmoid", "sqrtsoftplus", "softmax"])
@pytest.mark.parametrize("bias", [False, True])
def test_sorted_router_grad_and_no_grad_contract(score, bias):
    logits = torch.tensor([[0.1, 1.3, -0.7, 2.1], [1.7, -0.3, 0.4, 0.9]], requires_grad=True)
    expert_bias = torch.tensor([0.0, 0.2, -0.1, 0.3]) if bias else None
    original = torch.topk

    def unordered(tensor, *args, **kwargs):
        values, indices = original(tensor, *args, **{**kwargs, "sorted": True})
        if not kwargs.get("sorted", True):
            return values.flip(-1), indices.flip(-1)
        return values, indices

    def route(grad, policy):
        with torch.set_grad_enabled(grad):
            return topk_routing_with_score_function(
                logits,
                3,
                score_function=score,
                expert_bias=expert_bias,
                fused=False,
                router_replay=policy,
                dense_output=True,
            )

    with patch("torch.topk", unordered):
        train = route(True, SortedTopKPolicy())
        evaluate = route(False, SortedTopKPolicy())
        assert torch.equal(train[0], evaluate[0])
        assert torch.equal(train[1], evaluate[1])
        unprotected = route(False, None)
        assert not torch.equal(train[1], unprotected[1])
    train[0].sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("unsupported", ["fusion", "quantile", "replay"])
def test_sorted_router_rejects_bypass(unsupported):
    router = SimpleNamespace(
        is_hash_layer=False,
        config=SimpleNamespace(moe_router_fusion=unsupported == "fusion"),
        use_quantile_balancing=unsupported == "quantile",
        router_replay=object() if unsupported == "replay" else None,
    )
    with pytest.raises(ValueError):
        bind_sorted_selection(router)


def test_hash_router_preserved_and_grouped_selection_delegates():
    router = SimpleNamespace(is_hash_layer=True, router_replay=None)
    bind_sorted_selection(router)
    assert router.router_replay is None
    scores = torch.arange(8.0).reshape(2, 4)

    def grouped(values, k, groups, limit):
        assert values is scores and (k, groups, limit) == (2, 2, 1)
        return torch.topk(values, k, dim=1)

    value = SortedTopKPolicy().get_replay_topk(scores, 2, 2, 1, grouped)
    assert torch.equal(value[1], torch.topk(scores, 2, dim=1).indices)


def test_core_selection_extension_signature():
    expected = inspect.signature(RouterReplay.get_replay_topk).parameters
    actual = inspect.signature(SortedTopKPolicy.get_replay_topk).parameters
    assert list(expected) == list(actual)
    assert [p.kind for p in expected.values()] == [p.kind for p in actual.values()]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("loss_coeff", [0.0, 0.1])
def test_index_relay_native_topk_order_and_ties(tied, grad, loss_coeff):
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
    from megatron.lite.model.deepseek_v41.lite.index import IndexProjection

    class Projection(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    state = KVSourceState()
    q_owner = IndexProjection(Projection(4, 4, bias=False), state, 0, 0, "index_q")
    w_owner = IndexProjection(Projection(4, 1, bias=False), state, 0, 0, "index_weights")
    q_reader = IndexProjection(None, state, 1, 0, "index_q")
    w_reader = IndexProjection(None, state, 1, 0, "index_weights")
    x = torch.arange(32.0).reshape(8, 1, 4) / 32
    cu = torch.tensor([0, 4, 8, 16], dtype=torch.int32)
    ck = torch.tensor([0, 2, 4, 8], dtype=torch.int32)
    k = torch.zeros(8, 4) if tied else torch.arange(32.0).reshape(8, 4) / 31
    with state.microbatch(), torch.set_grad_enabled(grad):
        q = q_owner(x)
        weights = w_owner(x)
        assert q_reader(-x) is q and w_reader(-x) is weights

        def select(q_value, w_value):
            return cp_utils.compute_cp_indexer_topk(
                q_value[0].reshape(8, 1, 4),
                w_value[0].squeeze(1),
                k,
                cu,
                ck,
                8,
                2,
                4,
                0.5,
                max_seqlen_q=8,
                use_fused=False,
            )[0]

        selected = select(q, weights)
        assert torch.equal(selected, select(q_reader(-x), w_reader(-x)))
        # A real native indexer-loss call checks that a cached projection keeps
        # its graph; no high-level CUDA CSA execution is claimed here.
        from megatron.core.transformer.experimental_attention_variant.csa import (
            _unfused_indexer_sparse_attn_from_topk,
        )

        single = torch.tensor([0, 8], dtype=torch.int32)
        offsets = torch.tensor([0], dtype=torch.int32)
        ids = torch.arange(8, dtype=torch.int32)[:, None]
        query = x.reshape(8, 1, 4)

        def auxiliary(q_value, w_value):
            return _unfused_indexer_sparse_attn_from_topk(
                query,
                k,
                torch.zeros(1),
                ids,
                q_value[0].reshape(8, 1, 4),
                k,
                w_value[0].squeeze(1),
                ids,
                k,
                0.5,
                0.5,
                loss_coeff,
                8,
                False,
                1,
                8,
                (single, single, offsets),
            )[1]

        owner_loss = auxiliary(q, weights)
        reader_loss = auxiliary(q_reader(-x), w_reader(-x))
        assert torch.equal(owner_loss, reader_loss)
        # One loss belongs to each attention layer, even though projections
        # are shared. The relay creates no additional loss term of its own.
        if grad:
            params = (q_owner.linear.weight, w_owner.linear.weight)
            once = torch.autograd.grad(owner_loss, params, retain_graph=True)
            (owner_loss + reader_loss).backward()
            for parameter, expected in zip(params, once):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(parameter.grad, 2 * expected)
                if loss_coeff == 0:
                    assert torch.count_nonzero(parameter.grad) == 0
        else:
            assert not owner_loss.requires_grad and not reader_loss.requires_grad
    assert list(q_reader.parameters()) == [] and list(w_reader.parameters()) == []


def test_native_indexer_interface_contract():
    from megatron.core.transformer.experimental_attention_variant.csa import CSAIndexer
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    parameters = inspect.signature(cp_utils.compute_cp_indexer_topk).parameters
    assert list(parameters) == [
        "q_indexer_local",
        "weights_indexer_local",
        "k_indexer_seq_major",
        "cu_seqlens_q",
        "cu_seqlens_compressed",
        "global_start",
        "ratio",
        "topk_width",
        "indexer_softmax_scale",
        "max_seqlen_q",
        "use_fused",
        "deterministic",
        "precision",
        "compact_workspace",
        "return_softmax",
        "indexer_layout",
        "logical_indexer_layout",
        "max_seqlen_kv",
        "prebuilt_layout",
        "synthetic_layout",
    ]
    assert parameters["use_fused"].default is inspect.Parameter.empty
    assert parameters["deterministic"].default is False
    assert parameters["precision"].default == "bf16"
    # Core creates these attributes in __init__ rather than on the class. Lock
    # the constructor contract without pretending a CPU fixture allocates CUDA.
    import ast

    tree = ast.parse(inspect.getsource(inspect.getmodule(CSAIndexer)))
    native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CSAIndexer")
    constructor = next(
        n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assigned = {
        target.attr
        for n in ast.walk(constructor)
        if isinstance(n, ast.Assign)
        for target in n.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert {"linear_wq_b", "linear_weights_proj", "compressor"} <= assigned


def test_ds41_query_projection_contract():
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridSelfAttention,
    )
    from megatron.lite.model.deepseek_v41.lite.attention import DS41SelfAttention

    def signature(cls):
        return [
            (p.name, p.kind, p.default)
            for p in inspect.signature(cls.get_query_key_value_tensors).parameters.values()
        ]

    assert signature(DS41SelfAttention) == signature(DSv4HybridSelfAttention)

    class Projection(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    # Explicit projection fixture; native CPU RoPE really executes. This does
    # not claim construction or execution of the CUDA-only parent attention.
    module = DS41SelfAttention.__new__(DS41SelfAttention)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        apply_rope_fusion=False, qk_pos_emb_head_dim=2, rotary_interleaved=False
    )
    module.recompute_up_proj = False
    module.pg_collection = SimpleNamespace(cp=SimpleNamespace(rank=lambda: 0))
    module.num_attention_heads_per_partition, module.q_head_dim = 1, 4
    module.linear_q_down_proj = Projection(4, 4, bias=False)
    module.linear_q_up_proj = Projection(4, 4, bias=False)
    module.linear_kv_proj = Projection(4, 4, bias=False)
    module.q_layernorm = torch.nn.Identity()
    module.kv_layernorm = torch.nn.Identity()
    module.rotary_pos_emb = lambda maximum, packed_seq: torch.zeros(maximum, 1, 1, 2)
    x = (torch.arange(32.0).reshape(8, 1, 4) + 1).requires_grad_()
    prefix = torch.tensor([0, 4, 8], dtype=torch.int32)
    params = SimpleNamespace(qkv_format="thd", cu_seqlens_q_padded=prefix, max_seqlen_q=4)
    q, k, v, low_q, original = module.get_query_key_value_tensors(x, packed_seq_params=params)
    raw, _ = module.linear_q_up_proj(low_q)
    # Zero frequencies still convert interleaved rotary coordinates to the
    # native non-interleaved layout. Core owns that conversion.
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    expected = cp_utils.apply_thd_cp_local_rope_unfused(
        raw.reshape(8, 1, 4), module.rotary_pos_emb(4, True), 2, 2, prefix, 0, module.config
    )
    assert torch.equal(q, expected)
    assert not torch.allclose(q.square().mean(-1), torch.ones(8, 1))
    assert original is x and torch.equal(k, v)
    (q.square().mean() + k.square().mean()).backward()
    assert torch.isfinite(x.grad).all()


@pytest.fixture
def cpu_core_checkpoint(monkeypatch):
    """CPU-only CUDA RNG snapshot adapter; restore it before leaving the fixture."""
    from megatron.core.tensor_parallel import random as core_random

    original = (
        core_random.CheckpointFunction,
        core_random.CheckpointFunction.forward,
        core_random.CheckpointFunction.backward,
        torch.autograd.backward,
    )
    get_rng, set_rng = core_random._get_all_rng_states, core_random._set_all_rng_states
    with monkeypatch.context() as fixture:
        fixture.setattr(core_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),))
        fixture.setattr(
            core_random, "_set_all_rng_states", lambda state: torch.set_rng_state(state)
        )
        assert original == (
            core_random.CheckpointFunction,
            core_random.CheckpointFunction.forward,
            core_random.CheckpointFunction.backward,
            torch.autograd.backward,
        )
        yield core_random.checkpoint
    assert core_random._get_all_rng_states is get_rng and core_random._set_all_rng_states is set_rng
    assert original == (
        core_random.CheckpointFunction,
        core_random.CheckpointFunction.forward,
        core_random.CheckpointFunction.backward,
        torch.autograd.backward,
    )


@pytest.mark.parametrize("segments", [1, 2, 3])
def test_cross_segment_sources_native_core_checkpoint(segments, cpu_core_checkpoint):
    """CPU adapts only RNG snapshots; native Core checkpoint/autograd execute."""
    from megatron.lite.model.deepseek_v41.lite.index import IndexProjection

    class Projection(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    torch.manual_seed(7)
    state = KVSourceState()
    owner = IndexProjection(Projection(4, 4, bias=False), state, 0, 0, "index_q")
    readers = [IndexProjection(None, state, i + 1, 0, "index_q") for i in range(segments)]
    x = torch.randn(3, 1, 4, requires_grad=True)
    initial_weight = owner.linear.weight.detach().clone()
    calls = []

    def owner_segment(value):
        calls.append(("owner", torch.is_grad_enabled()))
        return owner(value)[0].sin()

    def reader_segment(reader, value):
        calls.append(("reader", torch.is_grad_enabled()))
        return value.cos() + reader(value)[0].square()

    with state.microbatch():
        actual = state.checkpoint(owner_segment, x, checkpoint_fn=cpu_core_checkpoint)
        for reader in readers:
            actual = state.checkpoint(
                lambda value, reader=reader: reader_segment(reader, value),
                actual,
                checkpoint_fn=cpu_core_checkpoint,
            )
    # Forward may finish before pipeline backward. Explicit checkpoint inputs
    # must keep the graph alive after the temporary source cache has cleared.
    assert not state.active and not state.values
    actual.sum().backward()
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = initial_weight.requires_grad_()
    shared = torch.nn.functional.linear(reference_x, reference_weight)
    reference = shared.sin()
    for _ in readers:
        reference = reference.cos() + shared.square()
    reference.sum().backward()
    assert torch.equal(actual, reference)
    assert torch.equal(x.grad, reference_x.grad)
    assert torch.equal(owner.linear.weight.grad, reference_weight.grad)
    assert torch.isfinite(x.grad).all() and torch.isfinite(owner.linear.weight.grad).all()
    # Reverse graph traversal really recomputes readers before their owner.
    reverse = [name for name, grad in calls if grad]
    assert reverse == ["reader"] * segments + ["owner"]
    assert not state.active and not state.values


@pytest.mark.parametrize("halo", [False, True])
def test_query_projection_parent_without_per_head_rms_is_bitwise(halo):
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as native,
    )
    from megatron.lite.model.deepseek_v41.lite.attention import DS41SelfAttention

    class Projection(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    class Frequencies:
        def get_rotary_seq_len(self, *args):
            return 4

        def __call__(self, length, packed_seq):
            return (torch.arange(length).float() / 7).view(-1, 1, 1, 1).expand(-1, 1, 1, 2)

    module = DS41SelfAttention.__new__(DS41SelfAttention)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        apply_rope_fusion=False,
        qk_pos_emb_head_dim=2,
        qk_head_dim=2,
        q_lora_rank=4,
        layernorm_epsilon=1e-6,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mrope_section=None,
    )
    module.recompute_up_proj = module._dsv4_uses_yarn_rope = False
    module.pg_collection = SimpleNamespace(
        cp=SimpleNamespace(rank=lambda: int(halo), size=lambda: 2 if halo else 1)
    )
    module.num_attention_heads_per_partition, module.q_head_dim = 1, 4
    module.linear_q_down_proj = Projection(4, 4, bias=False)
    module.linear_q_up_proj = Projection(4, 4, bias=False)
    module.linear_kv_proj = Projection(4, 4, bias=False)
    module.q_layernorm = torch.nn.RMSNorm(4, eps=1e-6)
    module.kv_layernorm = torch.nn.RMSNorm(4, eps=1e-6)
    module.rotary_pos_emb = Frequencies()
    rows = 4 if halo else 8
    x = torch.arange(rows * 4).float().reshape(rows, 1, 4).div(11).requires_grad_()
    boundary = torch.arange(8).float().reshape(2, 1, 4).div(13).requires_grad_() if halo else None
    prefix = torch.tensor([0, 4, 8], dtype=torch.int32)
    params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q_padded=prefix,
        cu_seqlens_kv_padded=prefix,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    actual = module.get_query_key_value_tensors(
        x, packed_seq_params=params, boundary_hidden=boundary
    )
    with patch.object(native, "_q_rms_norm", lambda query, eps: query):
        reference = native.DSv4HybridSelfAttention.get_query_key_value_tensors(
            module, x, packed_seq_params=params, boundary_hidden=boundary
        )
    assert len(actual) == len(reference) == (6 if halo else 5)
    for ours, core in zip(actual, reference):
        assert torch.equal(ours, core)
