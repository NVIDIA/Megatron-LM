# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the MoE kernels in ``megatron/core/transformer/moe/``.

Operator level: token permute / unpermute (torch and TE-fused, both branches of the
``torch.are_deterministic_algorithms_enabled()`` switch), chunk sorting, top-k routing (torch
and TE-fused), group-limited routing, the load-balancing aux loss (torch and TE-fused), and
the router gating GEMM. Module level: ``TopKRouter``, ``TEGroupedMLP`` / ``SequentialMLP`` on
deliberately uneven expert loads (including an empty expert), and a full ``MoELayer`` through
the all-gather and all-to-all dispatchers (plus the flex/DeepEP dispatcher from
``fused_a2a.py`` when the dependency and the GPUs are available).
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    count_differing_replays,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HAVE_TE = moe_utils.HAVE_TE
HAVE_TE_PERMUTE = HAVE_TE and moe_utils.fused_permute is not None
HAVE_TE_ROUTER = HAVE_TE and is_te_min_version("2.7.0")
from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

NUM_TOKENS, HIDDEN, NUM_EXPERTS, TOPK = 16384, 2048, 64, 8


def _routing(num_tokens=NUM_TOKENS, num_experts=NUM_EXPERTS, topk=TOPK):
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    probs_full = torch.softmax(logits, dim=-1)
    idx = probs_full.topk(topk, dim=-1).indices
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, idx, True)
    probs = torch.zeros_like(probs_full).scatter(1, idx, probs_full.gather(1, idx))
    return routing_map, probs


# --- permute / unpermute --------------------------------------------------------------------


def _permute_roundtrip(fused, with_probs):
    def fn(tokens, routing_map, probs):
        permuted, permuted_probs, sorted_indices, *_ = moe_utils.permute(
            tokens,
            routing_map,
            probs=probs if with_probs else None,
            num_out_tokens=NUM_TOKENS * TOPK,
            fused=fused,
        )
        # The experts apply the routed probabilities before the combine; every production
        # caller then unpermutes with ``probs=None``.
        if with_probs:
            permuted = permuted * permuted_probs.unsqueeze(-1).to(permuted.dtype)
        else:
            permuted = permuted * 1.001
        return moe_utils.unpermute(
            permuted, sorted_indices, tokens.shape, routing_map=routing_map, fused=fused
        )

    return fn


@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not HAVE_TE_PERMUTE, reason="TE permute missing")
        ),
    ],
)
@pytest.mark.parametrize("with_probs", [False, True])
def test_permute_unpermute_replay_under_deterministic_algorithms(fused, with_probs):
    """Deterministic branch: ``index_add_`` combine, ``index_put_`` gathers -- bit-exact."""
    seeded()
    routing_map, probs = _routing()
    tokens = torch.randn(
        NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    probs.requires_grad_(with_probs)
    with deterministic_algorithms(True):
        assert_replays_bit_exact(
            _permute_roundtrip(fused, with_probs),
            (tokens, routing_map, probs),
            replays=3,
            what=f"permute/unpermute[fused={fused}, probs={with_probs}]",
        )


@pytest.mark.xfail(
    strict=False,
    reason="Negative control. On GB300 the bf16 scatter_add_ combine replayed bit-exactly 8 "
    "times (2026-09-04), so the race is hardware/shape dependent; recorded, not gated.",
)
def test_default_unpermute_is_the_racy_path():
    """Negative control: without deterministic algorithms the torch combine uses
    ``scatter_add_`` (atomic bf16 accumulation of 8 rows per token over 16k tokens). When it
    races visibly, the deterministic assertion above is known to be sensitive."""
    seeded()
    routing_map, probs = _routing()
    tokens = torch.randn(
        NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with deterministic_algorithms(False):
        differing = count_differing_replays(
            _permute_roundtrip(False, False), (tokens, routing_map, probs), replays=8
        )
    assert differing > 0, (
        "the scatter_add_ combine replayed bit-exactly 7 times; either torch made it deterministic "
        "(drop this control) or the shape no longer contends"
    )


@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(not HAVE_TE_PERMUTE, reason="TE sort missing")),
    ],
)
@pytest.mark.parametrize("with_probs", [False, True])
def test_sort_chunks_by_idxs_replays(fused, with_probs):
    seeded()
    num_chunks = 32
    sizes = torch.randint(100, 2000, (num_chunks,))
    sizes[3] = 0
    rows = int(sizes.sum())
    split_sizes = sizes.to("cuda") if fused else sizes
    sorted_idxs = torch.randperm(num_chunks).to("cuda") if fused else torch.randperm(num_chunks)
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(rows, device="cuda", requires_grad=with_probs)

    def fn(x, probs):
        out, out_probs = moe_utils.sort_chunks_by_idxs(
            x, split_sizes, sorted_idxs, probs=probs if with_probs else None, fused=fused
        )
        return (out, out_probs) if with_probs else out

    assert_replays_bit_exact(fn, (x, probs), replays=3, what=f"sort_chunks_by_idxs[fused={fused}]")


# --- routing --------------------------------------------------------------------------------

ROUTING_CASES = {
    "softmax_topk8": dict(topk=8, score_function="softmax"),
    "softmax_pre_softmax": dict(topk=8, use_pre_softmax=True, score_function="softmax"),
    "sigmoid_scaled": dict(topk=8, score_function="sigmoid", scaling_factor=2.5),
    "group_limited": dict(topk=8, score_function="sigmoid", num_groups=8, group_topk=4),
    "topk2": dict(topk=2, score_function="softmax"),
}


@pytest.mark.parametrize("case", sorted(ROUTING_CASES))
@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(not HAVE_TE_ROUTER, reason="TE>=2.7 needed")),
    ],
)
@pytest.mark.parametrize("det_algos", [True, False], ids=["det-branch", "default-branch"])
def test_topk_routing_replays(case, fused, det_algos):
    seeded()
    kwargs = dict(ROUTING_CASES[case])
    logits = torch.randn(8192, 256, device="cuda", dtype=torch.float32, requires_grad=True)
    expert_bias = (
        torch.randn(256, device="cuda") * 0.01
        if kwargs.get("score_function") == "sigmoid"
        else None
    )

    def fn(logits):
        probs, routing_map = moe_utils.topk_routing_with_score_function(
            logits, fused=fused, expert_bias=expert_bias, **kwargs
        )
        return probs, routing_map

    with deterministic_algorithms(det_algos):
        assert_replays_bit_exact(
            fn, (logits,), replays=3, what=f"topk routing[{case}, fused={fused}]"
        )


def test_group_limited_topk_replays():
    seeded()
    scores = torch.rand(8192, 256, device="cuda")
    assert_replays_bit_exact(
        lambda s: moe_utils.group_limited_topk(s, 8, 8192, 256, 8, 4),
        (scores,),
        backward=False,
        what="group_limited_topk",
    )


@pytest.mark.parametrize(
    "fused",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.skipif(not HAVE_TE_ROUTER, reason="TE>=2.7 needed"),
                pytest.mark.xfail(
                    strict=False,
                    reason="TE fused_moe_aux_loss reduces with atomicAdd (open gap, recorded not gated)",
                ),
            ],
        ),
    ],
)
def test_switch_load_balancing_loss_replays(fused):
    seeded()
    routing_map, probs = _routing(num_tokens=65536, num_experts=256, topk=8)
    probs = probs.detach().requires_grad_(True)
    tokens_per_expert = routing_map.sum(dim=0)

    def fn(probs):
        return moe_utils.switch_load_balancing_loss_func(
            probs, tokens_per_expert, 65536, 8, 256, 1e-2, fused=fused
        )

    assert_replays_bit_exact(fn, (probs,), replays=4, what=f"aux loss[fused={fused}]")


@pytest.mark.parametrize("router_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("with_bias", [False, True])
def test_router_gating_linear_replays(router_dtype, with_bias):
    seeded()
    inp = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True) * 0.02
    weight = weight.detach().requires_grad_(True)
    bias = (
        torch.randn(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        if with_bias
        else None
    )
    assert_replays_bit_exact(
        lambda i, w, b: moe_utils.router_gating_linear(i, w, b, router_dtype),
        (inp, weight, bias),
        replays=3,
        contention=True,
        what="router_gating_linear",
    )


# --- modules --------------------------------------------------------------------------------


def _moe_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=1024,
        ffn_hidden_size=2048,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        deterministic_mode=True,
        sequence_parallel=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestMoEModules:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _init(self, ep=1):
        Utils.initialize_model_parallel(expert_model_parallel_size=ep)
        model_parallel_cuda_manual_seed(123)

    @pytest.mark.parametrize(
        "balancing,score,expert_bias",
        [
            ("aux_loss", "softmax", False),
            ("seq_aux_loss", "sigmoid", True),
            ("sinkhorn", "softmax", False),
        ],
        ids=["aux_loss", "seq_aux_loss+sigmoid+bias", "sinkhorn"],
    )
    def test_topk_router_replays(self, balancing, score, expert_bias):
        self._init()
        seeded()
        config = _moe_config(
            num_moe_experts=64,
            moe_router_topk=8 if balancing != "sinkhorn" else 1,
            moe_router_load_balancing_type=balancing,
            moe_router_score_function=score,
            moe_router_enable_expert_bias=expert_bias,
            moe_router_pre_softmax=balancing == "sinkhorn",
            moe_aux_loss_coeff=0.0 if balancing == "sinkhorn" else 0.01,
        )
        router = TopKRouter(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        router.set_layer_number(0)
        hidden = torch.randn(2048, 4, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        assert_module_replays_bit_exact(
            router, (hidden,), replays=3, what=f"TopKRouter[{balancing}]"
        )

    @pytest.mark.skipif(not HAVE_TE, reason="TE grouped MLP needs Transformer Engine")
    def test_te_grouped_mlp_replays_on_uneven_experts(self):
        self._init()
        seeded()
        config = _moe_config(hidden_size=2048, ffn_hidden_size=4096)
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=8, moe_grouped_gemm=True)
        experts = get_submodules(spec.submodules.mlp).experts(
            num_local_experts=8,
            config=config,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert isinstance(experts, TEGroupedMLP)
        experts = experts.cuda()
        tokens_per_expert = torch.tensor([4096, 17, 0, 2048, 1, 8191, 33, 1998], dtype=torch.int64)
        rows = int(tokens_per_expert.sum())
        hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(rows, device="cuda", requires_grad=True)
        assert_module_replays_bit_exact(
            experts,
            (hidden, tokens_per_expert, probs),
            replays=3,
            contention=True,
            what="TEGroupedMLP",
        )

    def test_sequential_mlp_replays_on_uneven_experts(self):
        self._init()
        seeded()
        config = _moe_config(hidden_size=2048, ffn_hidden_size=4096, moe_grouped_gemm=False)
        submodules = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False)
        experts = get_submodules(submodules.mlp).experts(
            num_local_experts=8,
            config=config,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert isinstance(experts, SequentialMLP)
        experts = experts.cuda()
        tokens_per_expert = torch.tensor([4096, 17, 0, 2048, 1, 8191, 33, 1998], dtype=torch.int64)
        rows = int(tokens_per_expert.sum())
        hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        probs = torch.rand(rows, device="cuda", requires_grad=True)
        assert_module_replays_bit_exact(
            experts,
            (hidden, tokens_per_expert, probs),
            replays=3,
            contention=True,
            what="SequentialMLP",
        )

    @pytest.mark.parametrize(
        "dispatcher,ep,extra",
        [
            ("allgather", 1, {}),
            ("alltoall", 1, {}),
            (
                "alltoall",
                1,
                {"moe_expert_capacity_factor": 0.5, "moe_pad_expert_input_to_capacity": True},
            ),
            pytest.param("alltoall", 2, {"moe_permute_fusion": HAVE_TE_PERMUTE}, id="alltoall-ep2"),
            pytest.param(
                "flex",
                2,
                {"moe_flex_dispatcher_backend": "deepep"},
                id="flex-deepep-ep2",
                marks=pytest.mark.skipif(not HAVE_DEEP_EP, reason="DeepEP not installed"),
            ),
        ],
    )
    def test_moe_layer_replays(self, dispatcher, ep, extra):
        if Utils.world_size % ep != 0 or (ep > 1 and Utils.world_size < ep):
            pytest.skip(f"needs a world size divisible by EP={ep}")
        self._init(ep=ep)
        seeded()
        config = _moe_config(
            moe_token_dispatcher_type=dispatcher, expert_model_parallel_size=ep, **extra
        )
        if not HAVE_TE:
            config.moe_grouped_gemm = False
            mlp_spec = get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
        else:
            mlp_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=8, moe_grouped_gemm=True
            ).submodules.mlp
        layer = MoELayer(config, get_submodules(mlp_spec)).cuda()
        layer.set_layer_number(0)
        hidden = torch.randn(2048, 2, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with deterministic_algorithms(True):
            assert_module_replays_bit_exact(
                layer,
                (hidden,),
                replays=3,
                contention=True,
                what=f"MoELayer[{dispatcher}, ep={ep}]",
            )
