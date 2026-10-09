# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Behavior of the shared-prefix MoE and MTP capabilities.

``hybrid_star_moe_expert_bias_v1``: a shared prompt row stands for G dense rows, so a real
``TopKRouter`` must count it G times in ``local_tokens_per_expert`` (and every branch token
once), including when the MoE layer recomputes its forward in backward.

``hybrid_star_mtp_dense_heads_v1``: the MTP heads run on the dense branches reconstructed from
the star, with ``loss_group_lengths`` preserving the per-forward token-count normalization.
"""

import pytest
import torch

from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.models.hybrid.shared_prefix_test_utils import (
    ReplayedRouting,
    SharedPrefixProblem,
    TokenProblem,
    build_hybrid_model,
    clear_attention_env,
    compare_model_runs,
    copy_params,
    round_params_to,
    run_dense_rows,
    run_shared,
)
from tests.unit_tests.test_utilities import Utils

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# See test_shared_prefix_model_parity.py: shared BF16 must be as close to the FP32 dense
# reference as dense BF16 is.
REFERENCE_RATIO = 1.25
SLACK = 1e-5


def _local_moe_layer(**overrides):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = dict(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=16,
        moe_router_topk=4,
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        # FP64 routing makes the dense and shared router logits of a token identical.
        moe_router_dtype="fp64",
        moe_token_dispatcher_type="alltoall",
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        use_cpu_initialization=True,
    )
    config.update(overrides)
    config = TransformerConfig(**config)
    submodules = get_submodules(
        get_gpt_layer_local_submodules(
            num_experts=config.num_moe_experts, moe_grouped_gemm=False
        ).mlp
    )
    return MoELayer(config, submodules).cuda().train()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixExpertBiasCounts:
    """Logical token multiplicities through a real (compiled) TopKRouter."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(7)
        torch.manual_seed(7)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("exclude_sequence_padding", [False, True])
    def test_router_counts_match_dense_expansion(self, exclude_sequence_padding):
        layer = _local_moe_layer()
        router = layer.router
        problem = SharedPrefixProblem(((7, (4, 3, 2)), (9, (6, 1))), padding_multiple=4)
        layout = problem.layout(forest=True)
        star = torch.randn(problem.physical_len, 1, 64, device="cuda").bfloat16()
        dense_index = torch.cat([row.star_indices() for row in problem.rows]).cuda()
        dense = star.index_select(0, dense_index)
        # Ordinary per-branch padding of the dense rows.
        padding = torch.cat(
            [
                torch.arange(row.dense_len) >= row.prefix_len + row.logical_len
                for row in problem.rows
            ]
        ).cuda()
        multiplicities = layout.padded_token_multiplicities(
            problem.physical_len, "cuda", exclude_sequence_padding=exclude_sequence_padding
        )

        def counts(hidden, **kwargs):
            router.local_tokens_per_expert.zero_()
            with torch.enable_grad():
                _, routing_map = router(hidden, **kwargs)
            return router.local_tokens_per_expert.clone(), routing_map

        shared_counts, shared_map = counts(star, token_multiplicities=multiplicities)
        dense_kwargs = {"padding_mask": padding.unsqueeze(1)} if exclude_sequence_padding else {}
        dense_counts, dense_map = counts(dense, **dense_kwargs)
        assert torch.equal(shared_map.index_select(0, dense_index), dense_map)
        torch.testing.assert_close(shared_counts, dense_counts, rtol=0, atol=0)
        expected_total = (
            sum(row.dense_len for row in problem.rows)
            - (int(padding.sum()) if exclude_sequence_padding else 0)
        ) * router.topk
        assert int(shared_counts.sum()) == expected_total
        # Without multiplicities the router keeps counting physical tokens.
        plain_counts, plain_map = counts(star)
        assert torch.equal(plain_counts, plain_map.sum(0).to(plain_counts.dtype))

    def test_dense_index_counts_use_multiplicities(self):
        """Dense ``[tokens, topk]`` indices (flex backends), including invalid ``-1`` routes."""
        layer = _local_moe_layer()
        router = layer.router
        indices = torch.tensor([[0, 3], [1, -1], [2, 15], [0, 0]], device="cuda", dtype=torch.int16)
        multiplicities = torch.tensor([3.0, 1.0, 0.0, 2.0], device="cuda")
        router.local_tokens_per_expert.zero_()
        with torch.enable_grad():
            router._apply_expert_bias(
                indices, padding_mask=None, token_multiplicities=multiplicities
            )
        expected = torch.zeros_like(router.local_tokens_per_expert)
        expected[0] = 3 + 2 + 2
        expected[3] = 3
        expected[1] = 1
        torch.testing.assert_close(router.local_tokens_per_expert, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("recompute", [False, True], ids=["plain", "moe-recompute"])
    def test_moe_layer_counts_logical_tokens(self, recompute):
        """With MoE selective recompute, counting happens in backward from the closure capture."""
        overrides = {"num_moe_experts": 8, "moe_router_topk": 2, "ffn_hidden_size": 128}
        if recompute:
            overrides.update(recompute_granularity="selective", recompute_modules=["moe"])
        layer = _local_moe_layer(**overrides)
        router = layer.router
        assert layer.moe_layer_recompute == recompute
        hidden = torch.randn(40, 1, 64, device="cuda").bfloat16().requires_grad_()
        multiplicities = torch.randint(0, 4, (40,), device="cuda").float()
        router.local_tokens_per_expert.zero_()
        layer._shared_prefix_token_multiplicities = multiplicities
        try:
            output, _ = layer(hidden)
        finally:
            # The shared-prefix stack removes the scope before backward runs.
            del layer._shared_prefix_token_multiplicities
        output.float().pow(2).sum().backward()
        counted = router.local_tokens_per_expert.clone()
        router.local_tokens_per_expert.zero_()
        with torch.enable_grad():
            router(hidden.detach(), token_multiplicities=multiplicities)
        torch.testing.assert_close(counted, router.local_tokens_per_expert, rtol=0, atol=0)


# ----------------------------------------------------------------------------- MTP normalization
_VOCAB, _HIDDEN, _DEPTHS, _FACTOR = 11, 8, 2, 0.3


def _segment_shift(mask: torch.Tensor, lengths) -> torch.Tensor:
    """``roll(-1)`` inside every packed segment, zeroing each segment's last position."""
    pieces = []
    for piece in mask.split(list(lengths), dim=-1):
        pieces.append(torch.cat([piece[..., 1:], torch.zeros_like(piece[..., :1])], dim=-1))
    return torch.cat(pieces, dim=-1)


def _process_mtp_loss(hidden_depths, input_ids, loss_mask, lengths, groups, input_mask=None):
    from types import SimpleNamespace

    from megatron.core.transformer.multi_token_prediction import process_mtp_loss

    projection = torch.linspace(-1, 1, _VOCAB * _HIDDEN, dtype=torch.float64, device=_DEVICE)
    projection = projection.reshape(_VOCAB, _HIDDEN)

    def output_layer(hidden, weight=None, runtime_gather_output=None):
        return hidden @ projection.t(), None

    def language_model_loss(labels, logits):
        sequence, batch, vocab = logits.shape
        return torch.nn.functional.cross_entropy(
            logits.transpose(0, 1).reshape(-1, vocab), labels.reshape(-1), reduction="none"
        ).view(batch, sequence)

    cumulative = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()))
    cumulative = cumulative.to(torch.int32).to(_DEVICE)
    config = SimpleNamespace(
        mtp_num_layers=_DEPTHS,
        mtp_loss_scaling_factor=_FACTOR,
        calculate_per_token_loss=True,
        mtp_detach_heads=False,
    )
    return process_mtp_loss(
        hidden_states=torch.cat(hidden_depths, dim=0),
        labels=None,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=None,
        is_training=False,
        compute_language_model_loss=language_model_loss,
        config=config,
        packed_seq_params=PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cumulative,
            cu_seqlens_kv=cumulative,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        ),
        input_ids=input_ids,
        mtp_input_mask=input_mask,
        loss_group_lengths=groups,
    )


class _MTPPack:
    """Packed rows ``[prompt, completion]`` in two loss groups; some masks touch row starts."""

    # (prompt length, completion length) per packed row; prompt length 0 puts the loss mask at
    # the segment start, where every MTP roll drops one counted token.
    ROWS = (((0, 6), (2, 3)), ((1, 6), (0, 4), (3, 3)))

    def __init__(self, seed=3):
        generator = torch.Generator().manual_seed(seed)
        self.group_rows = [list(group) for group in self.ROWS]
        self.lengths = [p + c for group in self.ROWS for p, c in group]
        self.groups = tuple(sum(p + c for p, c in group) for group in self.ROWS)
        total = sum(self.lengths)
        self.input_ids = torch.randint(0, _VOCAB, (1, total), generator=generator).to(_DEVICE)
        mask = []
        for group in self.ROWS:
            for prompt, completion in group:
                mask += [0.0] * prompt + [1.0] * completion
        self.loss_mask = torch.tensor([mask], dtype=torch.float64, device=_DEVICE)
        self.input_mask = (torch.rand(1, total, generator=generator) > 0.3).to(_DEVICE)
        self.hidden = [
            torch.randn(total, 1, _HIDDEN, generator=generator, dtype=torch.float64).to(_DEVICE)
            for _ in range(_DEPTHS + 1)
        ]

    def group_slices(self):
        start = 0
        for group, length in zip(self.group_rows, self.groups):
            yield slice(start, start + length), [p + c for p, c in group]
            start += length


@pytest.fixture
def _mtp_scale():
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    previous = MTPLossAutoScaler.main_loss_backward_scale
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, dtype=torch.float64, device=_DEVICE))
    yield
    MTPLossAutoScaler.set_loss_scale(previous)


@pytest.mark.usefixtures("_mtp_scale")
@pytest.mark.parametrize("with_input_mask", [False, True], ids=["no-input-mask", "input-mask"])
def test_grouped_mtp_loss_matches_independent_forwards(with_input_mask):
    """One grouped call must inject the MTP gradient of one independent call per group.

    In the input-mask case the grouped denominator must use the cumulative ``mtp_input_mask``
    like the ungrouped path.
    """
    pack = _MTPPack()
    input_mask = pack.input_mask if with_input_mask else None
    hidden = [t.clone().requires_grad_() for t in pack.hidden]
    _process_mtp_loss(
        hidden, pack.input_ids, pack.loss_mask, pack.lengths, pack.groups, input_mask
    ).sum().backward()
    grouped = [t.grad[:, 0] for t in hidden[1:]]

    independent = [torch.zeros_like(g) for g in grouped]
    for group_slice, lengths in pack.group_slices():
        hidden = [t[group_slice].clone().requires_grad_() for t in pack.hidden]
        _process_mtp_loss(
            hidden,
            pack.input_ids[:, group_slice],
            pack.loss_mask[:, group_slice],
            lengths,
            None,
            None if input_mask is None else input_mask[:, group_slice],
        ).sum().backward()
        for depth in range(_DEPTHS):
            independent[depth][group_slice] = hidden[depth + 1].grad[:, 0]
    for depth in range(_DEPTHS):
        torch.testing.assert_close(grouped[depth], independent[depth], rtol=1e-12, atol=1e-12)


@pytest.mark.usefixtures("_mtp_scale")
def test_grouped_mtp_normalization_matches_analytic_weights(monkeypatch):
    """Per-token MTP weights equal ``factor / D * original_count / rolled_count`` per group."""
    from megatron.core.transformer import multi_token_prediction

    recorded = []
    original_apply = multi_token_prediction.MTPLossAutoScaler.apply

    def record(hidden, loss):
        recorded.append(loss.detach().clone())
        return original_apply(hidden, loss)

    monkeypatch.setattr(multi_token_prediction.MTPLossAutoScaler, "apply", record)
    pack = _MTPPack()
    # Cross entropy of a zero hidden state is log(V) for every token: weights are then visible.
    zeros = [torch.zeros_like(t) for t in pack.hidden]
    _process_mtp_loss(zeros, pack.input_ids, pack.loss_mask, pack.lengths, pack.groups)
    assert len(recorded) == _DEPTHS

    mask = _segment_shift(pack.loss_mask, pack.lengths)  # labels are derived from input_ids
    originals = [part.sum() for part in mask.split(pack.groups, dim=-1)]
    cross_entropy = torch.log(torch.tensor(float(_VOCAB), dtype=torch.float64))
    corrections = set()
    for depth in range(_DEPTHS):
        mask = _segment_shift(mask, pack.lengths)
        expected = []
        for part, original in zip(mask.split(pack.groups, dim=-1), originals):
            correction = original / part.sum().clamp(min=1)
            corrections.add(round(float(correction), 9))
            expected.append(part * (_FACTOR / _DEPTHS) * correction)
        expected = torch.cat(expected, dim=-1) * cross_entropy.to(mask.device)
        torch.testing.assert_close(recorded[depth], expected, rtol=1e-12, atol=1e-12)
    # The pack is built so that the per-group corrections differ from one another and from 1.
    assert len(corrections) == len(pack.groups) * _DEPTHS and 1.0 not in corrections


@pytest.mark.internal
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_mtp_branch_packing_reassembles_dense_branches(cp_size):
    """Every CP rank packs its zigzag share of each dense branch ``[prompt, completion_g]``."""
    from megatron.core.models.hybrid.hybrid_model import _pack_shared_prefix_mtp_branches

    problem = SharedPrefixProblem(((5, (3, 6)), (7, (1, 9, 4))), padding_multiple=8)
    layout = problem.layout(forest=True)
    physical_len = problem.physical_len
    ids = torch.arange(physical_len).unsqueeze(0) + 1000
    hidden = torch.arange(physical_len, dtype=torch.float64).reshape(-1, 1, 1)
    loss_mask = torch.zeros(1, physical_len)
    for row in problem.rows:
        loss_mask[0, row.completion_offset : row.completion_offset + row.logical_len] = 1
    covered = [torch.zeros(row.dense_len, dtype=torch.bool) for row in problem.rows]
    for cp_rank in range(cp_size):
        packed_hidden, packed_ids, packed_mask, positions = _pack_shared_prefix_mtp_branches(
            hidden, ids, loss_mask, layout, cp_size=cp_size, cp_rank=cp_rank
        )
        offset = 0
        for row, seen in zip(problem.rows, covered):
            local = SharedPrefixLayout.cp_local_indices(row.dense_len, cp_size, cp_rank, "cpu")
            span = slice(offset, offset + local.numel())
            expected_ids = ids[0, row.star_indices()][local]
            assert torch.equal(packed_ids[0, span], expected_ids)
            assert torch.equal(packed_hidden[span, 0, 0], hidden[row.star_indices(), 0, 0][local])
            assert torch.equal(packed_mask[0, span], loss_mask[0, row.star_indices()][local])
            assert torch.equal(positions[0, span], local)
            seen[local] = True
            offset += local.numel()
        assert offset == packed_ids.shape[1]
    assert all(seen.all() for seen in covered)
    # One loss group per root by default: dense lengths of that root's branches.
    assert isinstance(layout, SharedPrefixForestLayout)
    assert layout.mtp_loss_group_lengths == (
        sum(row.dense_len for row in problem.rows if row.root == 0),
        sum(row.dense_len for row in problem.rows if row.root == 1),
    )


# ----------------------------------------------------------------------------- MTP model parity
# build_hybrid_model uses an init std large enough for RoPE to matter (the review's fu-mtp-parity
# check found the default 0.02 cannot tell RoPE from no RoPE); the test verifies it explicitly.
MTP_PATTERN = "M*EM*E/*E/*E"

# A forest whose MTP count ratios (original / rolled loss-mask count) differ from 1, so every
# loss grouping gives different weights. Each MTP roll drops the counted token that reaches a
# branch start: the 1-token prompt loses one token per branch at depth 0 and two at depth 1, the
# 2-token prompt one at depth 1, the 37-token prompt none. At CP2 the zigzag split also moves
# rolled tokens between ranks, which changes the CP-local counts of every root.
MTP_GROUPED_FOREST = ((1, (13, 30, 22)), (2, (9, 41)), (37, (20, 45)))


def _row_groups(problem: SharedPrefixProblem, root_counts) -> list[list[int]]:
    """Dense-row indices of each MTP loss group of ``root_counts`` consecutive roots."""
    groups, first_root = [], 0
    for count in root_counts:
        roots = range(first_root, first_root + count)
        groups.append([index for index, row in enumerate(problem.rows) if row.root in roots])
        first_root += count
    return groups


def _expected_mtp_weights(problem, groups, depths, factor, cp_size, cp_rank) -> list[torch.Tensor]:
    """Expected ``normalized MTP loss / CE`` per depth on one CP rank's packed dense branches.

    Port of the review's fu-mtp-parity ``wsim.ratios``. ``process_mtp_loss`` derives the labels
    by rolling the loss mask once (the original count), then rolls it once more per depth (the
    rolled count); both counts cover the tokens this CP rank owns. A group's counted tokens get
    ``factor / depths * original / rolled``, with both counts summed over the group's branches.
    The pack is branch-major and holds each branch's CP-local zigzag share.
    """

    def owned(length):
        if cp_size == 1:
            return torch.arange(length)
        return SharedPrefixLayout.cp_local_indices(length, cp_size, cp_rank, "cpu")

    # rolled[row][k]: this rank's share of the row's loss mask rolled k + 1 times.
    rolled = []
    for row in problem.rows:
        mask = torch.zeros(row.dense_len, dtype=torch.float64)
        mask[row.prefix_len : row.prefix_len + row.logical_len] = 1
        local = owned(row.dense_len)
        rolled.append(
            [torch.cat([mask[k:], mask.new_zeros(k)])[local] for k in range(1, depths + 2)]
        )
    offsets = [0]
    for masks in rolled:
        offsets.append(offsets[-1] + masks[0].numel())
    weights = [torch.zeros(offsets[-1], dtype=torch.float64) for _ in range(depths)]
    for group in groups:
        original = sum(float(rolled[index][0].sum()) for index in group)
        for depth in range(depths):
            count = sum(float(rolled[index][depth + 1].sum()) for index in group)
            for index in group:
                weights[depth][offsets[index] : offsets[index + 1]] = (
                    rolled[index][depth + 1] * factor / depths * original / max(count, 1.0)
                )
    return weights


def _check_grouped_mtp_normalization(monkeypatch, root_counts):
    """Shared MTP weights must follow the layout's loss groups on every CP rank and depth."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import multi_token_prediction

    model_parallel_cuda_manual_seed(123)
    clear_attention_env(monkeypatch)
    torch.manual_seed(0)
    model = build_hybrid_model(MTP_PATTERN, torch.bfloat16, calculate_per_token_loss=True)
    problem = SharedPrefixProblem(MTP_GROUPED_FOREST, padding_multiple=8, topology_multiple=8)
    tokens = TokenProblem(problem, vocab_size=2048, seed=1)
    layout = SharedPrefixForestLayout(
        problem.layout(forest=True).roots, mtp_loss_group_root_counts=root_counts
    )

    # Record the per-token normalized MTP loss and the per-token cross entropy of every depth.
    normalized, cross_entropy = [], []
    original_apply = multi_token_prediction.MTPLossAutoScaler.apply
    original_loss = model.compute_language_model_loss

    def record_normalized(hidden, loss):
        normalized.append(loss.detach().double()[0].cpu())
        return original_apply(hidden, loss)

    def record_cross_entropy(labels, logits):
        loss = original_loss(labels, logits)
        cross_entropy.append(loss.detach().double()[0].cpu())
        return loss

    monkeypatch.setattr(multi_token_prediction.MTPLossAutoScaler, "apply", record_normalized)
    model.compute_language_model_loss = record_cross_entropy
    routing = ReplayedRouting(model, tokens.num_keys, seed=2)
    try:
        run_shared(model, tokens, layout, routing)
    finally:
        routing.close()
    depths = model.config.mtp_num_layers
    assert len(normalized) == len(cross_entropy) == depths

    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    factor = model.config.mtp_loss_scaling_factor
    num_roots = len(problem.roots)

    def expected(groups, rank=cp_rank):
        return _expected_mtp_weights(problem, groups, depths, factor, cp_size, rank)

    groups = _row_groups(problem, root_counts or (1,) * num_roots)
    for depth, weights in enumerate(expected(groups)):
        counted = weights > 0
        assert torch.equal(normalized[depth][~counted], torch.zeros(int((~counted).sum())))
        # Both sides are FP32 products of the same counts: a few ulp apart.
        torch.testing.assert_close(
            normalized[depth][counted] / cross_entropy[depth][counted],
            weights[counted],
            rtol=1e-5,
            atol=0,
        )

    # Sensitivity: on some CP rank and depth, every other grouping changes the weights by far
    # more than the tolerance. Dropping loss_group_lengths (one group for the whole pack),
    # ignoring explicit root counts (one group per root) or normalizing per dense row would fail.
    alternatives = [
        _row_groups(problem, (num_roots,)),
        _row_groups(problem, (1,) * num_roots),
        [[index] for index in range(len(problem.rows))],
    ]
    for alternative in alternatives:
        if alternative == groups:
            continue
        difference = max(
            float(((other - weights).abs() / weights.clamp(min=1e-30))[weights > 0].max())
            for rank in range(cp_size)
            for other, weights in zip(expected(alternative, rank), expected(groups, rank))
        )
        assert difference > 1e-3, (alternative, difference)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixMTPParity:
    """BF16 HybridModel with two MoE MTP depths: shared star vs dense rows at TP1/CP1."""

    def setup_method(self, method):
        pytest.importorskip("mamba_ssm")
        pytest.importorskip("flash_attn")
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.usefixtures("_mtp_scale")
    @pytest.mark.parametrize("main_loss", [True, False], ids=["main+mtp", "mtp-only"])
    def test_shared_mtp_matches_dense_rows(self, main_loss, monkeypatch):
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        overrides = dict(calculate_per_token_loss=True)
        model = build_hybrid_model(MTP_PATTERN, torch.bfloat16, **overrides)
        round_params_to(model, torch.bfloat16)
        reference_model = build_hybrid_model(MTP_PATTERN, torch.float32, **overrides)
        copy_params(model, reference_model)
        assert model.mtp_process and model.config.mtp_num_layers == 2
        problem = SharedPrefixProblem(
            ((64, (203, 260, 333, 190)),), padding_multiple=8, topology_multiple=8
        )
        tokens = TokenProblem(problem, vocab_size=2048, seed=1)
        if not main_loss:
            # Only the MTP loss drives the gradients.
            for cotangent in tokens.row_cotangents:
                cotangent.zero_()
            tokens.star_cotangent.zero_()
        layout = problem.layout(forest=False)
        routing = ReplayedRouting(model, tokens.num_keys, seed=2)
        reference_routing = ReplayedRouting(reference_model, tokens.num_keys, seed=2)
        try:
            reference = run_dense_rows(reference_model, tokens, reference_routing)
            dense = run_dense_rows(model, tokens, routing)
            shared = run_shared(model, tokens, layout, routing)

            def errors(run):
                return compare_model_runs(run, reference, model)

            dense_error, shared_error = errors(dense), errors(shared)
            label = "main+mtp" if main_loss else "mtp-only"
            print(f"\n[{label}] dense={dense_error} shared={shared_error}")
            metrics = ("logits", "grads", "mtp_grads") if main_loss else ("grads", "mtp_grads")
            for metric in metrics:
                assert shared_error[metric] <= REFERENCE_RATIO * dense_error[metric] + SLACK, (
                    f"shared {metric} error {shared_error[metric]:.3e} vs dense "
                    f"{dense_error[metric]:.3e} against the FP32 reference"
                )
            for shared_count, dense_count in zip(shared.counts, dense.counts):
                torch.testing.assert_close(shared_count, dense_count, rtol=0, atol=0)
            if not main_loss:
                # Sensitivity guard: without RoPE the MTP gradients must visibly change.
                monkeypatch.setattr(model, "position_embedding_type", "none")
                positionless = errors(run_shared(model, tokens, layout, routing))
                print(f"  without RoPE: {positionless}")
                assert positionless["mtp_grads"] > 5 * REFERENCE_RATIO * dense_error["mtp_grads"]
        finally:
            routing.close()
            reference_routing.close()

    @pytest.mark.usefixtures("_mtp_scale")
    @pytest.mark.parametrize("root_counts", [(), (2, 1)], ids=["per-root", "explicit-2-1"])
    def test_grouped_mtp_normalization_follows_layout(self, root_counts, monkeypatch):
        """``_forward_shared_prefix_mtp`` normalizes each layout loss group on its own."""
        _check_grouped_mtp_normalization(monkeypatch, root_counts)


# MTP loss normalization uses CP-local token counts, so grouped (shared) and per-row (dense)
# normalization agree at CP>1 only when every row's local count ratio is 1. These lengths satisfy
# that at CP2 with padding multiple 8 (review fu-mtp-parity, wsim.py), making a direct
# shared-vs-dense comparison valid. Non-unit ratios are checked against the analytic grouped
# weights instead (test_grouped_mtp_normalization_*).
MTP_CP2_STAR = ((64, (203, 260, 333, 190)),)
MTP_CP2_FOREST = ((48, (150, 177)), (40, (131, 160)))
# See test_shared_prefix_model_parity.py: the TP/SP/CP gap must stay within 1.5x of the TP1/CP1 gap.
TOPOLOGY_RATIO = 1.5


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixMTPDistributedParity:
    """MTP branch repacking across TP/SP/CP: shared star vs dense rows."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _gap(self, monkeypatch, roots, forest):
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)
        clear_attention_env(monkeypatch)
        torch.manual_seed(0)
        model = build_hybrid_model(MTP_PATTERN, torch.bfloat16, calculate_per_token_loss=True)
        problem = SharedPrefixProblem(roots, padding_multiple=8, topology_multiple=8)
        tokens = TokenProblem(problem, vocab_size=2048, seed=1)
        routing = ReplayedRouting(model, tokens.num_keys, seed=2)
        try:
            dense = run_dense_rows(model, tokens, routing)
            shared = run_shared(model, tokens, problem.layout(forest), routing)
        finally:
            routing.close()
        return dense, shared, compare_model_runs(shared, dense, model)

    @pytest.mark.usefixtures("_mtp_scale")
    @pytest.mark.parametrize(
        "roots,forest", [(MTP_CP2_STAR, False), (MTP_CP2_FOREST, True)], ids=["star", "forest"]
    )
    def test_mtp_matches_dense_rows_tp2_sp_cp2(self, roots, forest, monkeypatch):
        if Utils.world_size < 4 or Utils.world_size % 4:
            pytest.skip("requires a world size divisible by 4")
        Utils.initialize_model_parallel(1, 1)
        _, _, baseline = self._gap(monkeypatch, roots, forest)
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(2, 1, context_parallel_size=2)
        dense, shared, gap = self._gap(monkeypatch, roots, forest)
        print(f"\n[mtp tp2 cp2] gap={gap} tp1/cp1 gap={baseline}")
        for shared_count, dense_count in zip(shared.counts, dense.counts):
            torch.testing.assert_close(shared_count, dense_count, rtol=0, atol=0)
        for metric in ("logits", "grads", "mtp_grads"):
            assert gap[metric] <= TOPOLOGY_RATIO * baseline[metric] + SLACK, (
                f"TP2/CP2 shared-vs-dense {metric} gap {gap[metric]:.3e} exceeds "
                f"{TOPOLOGY_RATIO}x the TP1/CP1 gap {baseline[metric]:.3e}"
            )

    @pytest.mark.usefixtures("_mtp_scale")
    def test_shared_mtp_rerun_is_bitwise_tp2_sp_cp2(self, monkeypatch):
        """With deterministic kernels, a shared forest with MoE MTP heads reruns bit for bit.

        Routing is natural (not replayed): the fixed router row blocks and the fixed-order
        prompt-copy gradient sum must leave nothing to scheduling at TP2/SP/CP2.
        """
        if Utils.world_size < 4 or Utils.world_size % 4:
            pytest.skip("requires a world size divisible by 4")
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            Utils.initialize_model_parallel(2, 1, context_parallel_size=2)
            model_parallel_cuda_manual_seed(123)
            clear_attention_env(monkeypatch)
            torch.manual_seed(0)
            model = build_hybrid_model(MTP_PATTERN, torch.bfloat16, calculate_per_token_loss=True)
            problem = SharedPrefixProblem(MTP_CP2_FOREST, padding_multiple=8, topology_multiple=8)
            tokens = TokenProblem(problem, vocab_size=2048, seed=1)
            layout = problem.layout(forest=True)
            first = run_shared(model, tokens, layout)
            second = run_shared(model, tokens, layout)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
        for first_logits, second_logits in zip(first.logits, second.logits):
            assert torch.equal(first_logits, second_logits)
        assert any(name.startswith("mtp.") for name in first.grads)
        for name, grad in first.grads.items():
            assert torch.equal(grad, second.grads[name]), name
        assert len(first.counts) == MTP_PATTERN.count("E")
        for first_count, second_count in zip(first.counts, second.counts):
            assert torch.equal(first_count, second_count)

    @pytest.mark.usefixtures("_mtp_scale")
    @pytest.mark.parametrize("root_counts", [(), (2, 1)], ids=["per-root", "explicit-2-1"])
    def test_grouped_mtp_normalization_tp2_sp_cp2(self, root_counts, monkeypatch):
        """CP-local loss groups (``length // cp_size``) under zigzag ownership and SP."""
        if Utils.world_size < 4 or Utils.world_size % 4:
            pytest.skip("requires a world size divisible by 4")
        Utils.initialize_model_parallel(2, 1, context_parallel_size=2)
        _check_grouped_mtp_normalization(monkeypatch, root_counts)
