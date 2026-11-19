# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CLI state isolation, target accumulation and summed-KL reference tests."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.ops.attention.dsa.cli_state import (
    DSACrossLayerIndexerState,
    cli_group_size,
    get_dsa_cross_layer_state_holder,
)


@pytest.mark.parametrize(
    "layer,total,offset,frequency,expected",
    [
        (1, 9, 1, 4, 4),
        (4, 9, 1, 4, 4),
        (9, 9, 1, 4, 1),
        (1, 8, 3, 4, 1),
        (3, 8, 3, 4, 4),
        (7, 8, 3, 4, 2),
    ],
)
def test_cli_group_sizes(layer, total, offset, frequency, expected):
    assert cli_group_size(layer, total, offset, frequency) == expected


def test_cli_flag_requires_dsa():
    from megatron.core.transformer.transformer_config import TransformerConfig

    with pytest.raises(ValueError, match="requires the DSA variant"):
        TransformerConfig(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=4,
            dsa_indexer_cross_layer_distillation=True,
        )


def _make_cli_config(monkeypatch, **overrides):
    from megatron.core.transformer import transformer_config

    monkeypatch.setattr(
        transformer_config, "_validate_dsa_kernel_backend_dependencies", lambda *_: None
    )
    kwargs = dict(
        num_layers=8,
        hidden_size=128,
        num_attention_heads=4,
        add_bias_linear=False,
        experimental_attention_variant="dsa",
        dsa_kernel_backend="cudnn",
        dsa_indexer_cross_layer_distillation=True,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_topk_freq=4,
        dsa_indexer_topk=2048,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )
    return transformer_config.MLATransformerConfig(**(kwargs | overrides))


def test_cli_rejects_gpt_mtp(monkeypatch):
    with pytest.raises(ValueError, match="does not support MTP"):
        _make_cli_config(monkeypatch, mtp_num_layers=1)


def test_cli_rejects_hybrid_dsa_mlp_layout(monkeypatch):
    from megatron.core.models.hybrid.hybrid_block import HybridStack

    config = _make_cli_config(monkeypatch)
    with pytest.raises(ValueError, match="does not support hybrid layer layouts"):
        HybridStack(config, submodules=None, layer_type_list=list("D-D-D-D-"))


@pytest.mark.parametrize("topk", [0, 2048, 2304])
def test_cli_topk_bounds(monkeypatch, topk):
    if topk == 2048:
        assert _make_cli_config(monkeypatch, dsa_indexer_topk=topk).dsa_indexer_topk == topk
    else:
        with pytest.raises(ValueError, match="between 256 and 2048"):
            _make_cli_config(monkeypatch, dsa_indexer_topk=topk)


@pytest.mark.parametrize(
    "freq,offset,cli,owners",
    [(2, 1, True, (1, 3)), (2, 1, False, (1, 3)), (2, 4, False, (1, 2, 3, 4))],
)
def test_dsa_index_share_checkpoint_shards_are_complete(monkeypatch, freq, offset, cli, owners):
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.validation import validate_sharding_integrity
    from megatron.core.transformer.transformer_block import TransformerBlock

    config = _make_cli_config(
        monkeypatch,
        num_layers=4,
        dsa_indexer_topk_freq=freq,
        dsa_indexer_skip_topk_offset=offset,
        dsa_indexer_cross_layer_distillation=cli,
    )

    class Layer:
        def __init__(self, layer_number):
            self.layer_number = layer_number

        def sharded_state_dict(self, prefix, offsets, metadata):
            if self.layer_number not in owners:
                return {}
            weights = prefix + "indexer.weight"
            extra = prefix + "indexer._extra_state"
            return {
                weights: ShardedTensor.from_rank_offsets(
                    weights, torch.ones(2, 2), *offsets, prepend_axis_num=len(offsets)
                ),
                extra: ShardedObject(
                    extra,
                    b"extra",
                    tuple(count for _, _, count in offsets) or (1,),
                    tuple(rank for _, rank, _ in offsets) or (0,),
                ),
            }

    block = SimpleNamespace(
        config=config,
        layers=[Layer(i) for i in range(1, 5)],
        vp_stage=None,
        pg_collection=SimpleNamespace(pp=None),
        named_children=lambda: (),
    )
    values = TransformerBlock.sharded_state_dict(block, prefix="decoder.").values()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    validate_sharding_integrity([[item.without_data() for item in values]])
    keys = {item.key for item in values if isinstance(item, ShardedTensor)}
    if len(owners) < 4:
        assert keys == {"decoder.layers.0.indexer.weight", "decoder.layers.2.indexer.weight"}
    else:
        assert keys == {"decoder.layers.indexer.weight"}


def test_cli_state_is_per_forward_and_releases_tensors():
    first, second = SimpleNamespace(), SimpleNamespace()
    state = DSACrossLayerIndexerState(topk_indices=torch.ones(1), topk_length=torch.ones(1))
    get_dsa_cross_layer_state_holder(first)[1] = state
    assert get_dsa_cross_layer_state_holder(first)[1] is state
    assert not get_dsa_cross_layer_state_holder(second)
    state.release_forward_values()
    assert state.topk_indices is None
    assert state.topk_length is None
    assert state.deferred_cli_target is None


def test_cli_packed_forwards_allow_reverse_backward(monkeypatch):
    """Both checkpoint chunks must recover their own forward's state."""
    from torch.utils.checkpoint import checkpoint

    from megatron.core import recompute
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention

    monkeypatch.setattr(
        recompute,
        "precision_aware_checkpoint",
        lambda fn, _config, _tp, *args, **kwargs: checkpoint(fn, *args, use_reentrant=True),
    )
    monkeypatch.setattr(recompute, "observe_layer_residuals", lambda *args: None)
    config = SimpleNamespace(
        experimental_attention_variant="dsa",
        dsa_indexer_topk_freq=2,
        dsa_indexer_cross_layer_distillation=True,
        recompute_method="uniform",
        recompute_num_layers=1,
        enable_mhc_connections=False,
        distribute_saved_activations=False,
    )
    attention = SimpleNamespace(config=config)
    replayed = []

    class Layer:
        def __init__(self, number):
            self.layer_number = number

        def __call__(self, x, packed_seq_params, attention_mask, **kwargs):
            carrier = DSAttention._get_index_share_carrier(
                attention, packed_seq_params, attention_mask
            )
            state = get_dsa_cross_layer_state_holder(carrier)
            if self.layer_number == 1:
                if not torch.is_grad_enabled():
                    state[1] = DSACrossLayerIndexerState(topk_indices=x.detach().to(torch.int32))
                    return x.clone()
                owner = state.pop(1)
                return x * owner.topk_indices * owner.topk_length
            if torch.is_grad_enabled():
                owner = state[1]
                replayed.append(owner.topk_indices.item())
                output = x * owner.topk_indices

                def collect_teacher(gradient):
                    owner.topk_length = owner.topk_indices + 1
                    return gradient

                output.register_hook(collect_teacher)
                return output
            return x * state[1].topk_indices

    block = SimpleNamespace(
        config=config,
        layers=[Layer(1), Layer(2)],
        num_layers_per_pipeline_rank=2,
        pg_collection=SimpleNamespace(tp=None),
    )
    packed = PackedSeqParams(qkv_format="thd")

    def forward(value):
        x = torch.tensor([value], requires_grad=True)
        out = recompute.checkpointed_forward(block, x, None, None, None, None, None, packed, False)
        return x, out

    isolated = []
    for value in (2.0, 3.0):
        x, out = forward(value)
        out.sum().backward()
        isolated.append(x.grad)
    replayed.clear()
    first, second = forward(2.0), forward(3.0)
    for _, out in (second, first):
        out.sum().backward()
    assert replayed == [3, 2]
    for (x, _), expected in zip((first, second), isolated):
        torch.testing.assert_close(x.grad, expected)


def test_cli_owner_recompute_uses_original_indices(monkeypatch):
    """Tied scores must never cause another selection during owner recomputation."""
    from megatron.core.ops.attention.dsa import cli_scores
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_cudnn_kernels as kernels,
    )

    original = torch.arange(256, dtype=torch.int32).expand(1, 2, -1).contiguous()
    seen = {}

    def fail_selection(*args, **kwargs):
        raise AssertionError("CLI must not select another Top-K during owner recomputation")

    logits = torch.zeros(original.shape)
    logits[..., 0] = 120.0

    def selected_scores(q, k, weights, indices, lengths):
        del q, k, weights, lengths
        assert indices is original
        return logits

    def loss(**kwargs):
        seen["indices"] = kwargs["topk_indices_cmp"]
        assert kwargs["indexer_score_payload"] is logits
        return torch.zeros(()), torch.zeros(1), torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(kernels, "_indexer_topk_bshd", fail_selection)
    monkeypatch.setattr(cli_scores, "cli_selected_indexer_logits", selected_scores)
    monkeypatch.setattr(kernels, "_compute_sparse_indexer_loss_and_grads", loss)
    monkeypatch.setattr(
        kernels,
        "_run_sparse_attention_forward",
        lambda *args, **kw: (torch.zeros(2, 32, 512), torch.zeros(2, 32)),
    )
    context = SimpleNamespace(
        save_for_backward=lambda *a: None, mark_non_differentiable=lambda *a: None
    )
    kernels.FusedQKTopKWithSparseLossFunc.forward(
        context,
        torch.zeros(2, 1, 32, 128),
        torch.zeros(256, 1, 128),
        torch.zeros(2, 1, 32),
        torch.zeros(2, 1, 32, 576),
        torch.zeros(256, 1, 576),
        256,
        0.1,
        0.001,
        False,
        512,
        torch.zeros(2, dtype=torch.int32),
        torch.full((2,), 256, dtype=torch.int32),
        None,
        False,
        False,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        1,
        None,
        SimpleNamespace(topk_indices=original),
    )
    assert seen["indices"] is original


@pytest.mark.parametrize("chunk_rows", [1, 2, 4])
def test_cli_target_matches_reference(chunk_rows):
    pytest.importorskip("tilelang")
    from megatron.core.ops.attention.dsa.cli_target import accumulate_cli_attention_target

    torch.manual_seed(2468)
    indices = torch.arange(256, device="cuda", dtype=torch.int32).expand(1, 3, -1).clone()
    indices[:, 1, 240:] = -1
    lengths = torch.tensor([[256, 240, 0]], device="cuda", dtype=torch.int32)
    query_valid = torch.tensor([[True, True, False]], device="cuda")
    targets = [torch.rand(1, 3, 256, device="cuda") for _ in range(3)]
    expected_sum = torch.zeros_like(targets[0])
    expected_entropy = torch.zeros((), device="cuda")
    accumulator = None
    for target in targets:
        valid = (indices >= 0) & (torch.arange(256, device="cuda") < lengths[..., None])
        valid &= query_valid[..., None]
        probability = target.masked_fill(~valid, 0)
        probability /= probability.sum(-1, keepdim=True).clamp_min(1e-10)
        expected_sum += probability
        expected_entropy += (probability * probability.clamp_min(1e-10).log()).sum()
        accumulator = accumulate_cli_attention_target(
            target, indices, lengths, query_valid, chunk_rows, accumulator
        ).detach()
    torch.testing.assert_close(accumulator.target_sum, expected_sum)
    torch.testing.assert_close(accumulator.teacher_entropy_sum, expected_entropy)
    assert accumulator.reuse_count == len(targets)
    assert accumulator.topk_indices.data_ptr() == indices.data_ptr()


@pytest.mark.parametrize("group_size,large_gap", [(2, False), (4, False), (2, True)])
def test_cli_merged_loss_and_score_gradient_match_sum_of_kl(monkeypatch, group_size, large_gap):
    """Validate the production merge and scaling independently of the cuDNN derivative."""
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_cudnn_kernels as kernels,
    )

    torch.manual_seed(14)
    rows, topk = 2, 256
    logits = torch.randn(1, rows, topk)
    teachers = torch.rand(group_size, 1, rows, topk)
    teachers /= teachers.sum(-1, keepdim=True)
    lengths = torch.full((1, rows), topk, dtype=torch.int32)
    if large_gap:
        logits.fill_(1000)
        logits[..., :2] = torch.tensor([120.0, 0.0])
        teachers.zero_()
        teachers[0, ..., 0] = 1.0
        teachers[1, ..., 1] = 1.0
        lengths.fill_(2)
    logits.requires_grad_()
    valid = torch.arange(topk) < lengths[..., None]
    log_probs = logits.masked_fill(~valid, float("-inf")).log_softmax(-1).masked_fill(~valid, 0)
    coefficient = 0.003
    reference = coefficient * sum(
        (torch.xlogy(p, p) - p * log_probs).sum(-1).mean() for p in teachers
    )
    (reference_gradient,) = torch.autograd.grad(reference, logits)
    captured = {}

    monkeypatch.setattr(kernels, "_compute_attn_target", lambda *a, **kw: teachers[0].clone())
    monkeypatch.setattr(kernels, "_pad_indexer_heads_for_backward", lambda q, w: (q, w, q.size(2)))
    monkeypatch.setattr(kernels, "_slice_indexer_backward_head_grads", lambda q, w, _: (q, w))

    def backward(q, w, k, target, predict, indices, *, loss_coeff, grad_loss, block_i):
        del indices, block_i
        captured["gradient"] = (predict - target) * loss_coeff * grad_loss / rows
        return torch.zeros_like(q), torch.zeros_like(w), torch.zeros_like(k)

    monkeypatch.setattr(kernels, "_run_sparse_indexer_backward", backward)
    accumulator = SimpleNamespace(
        target_sum=teachers[1:].sum(0),
        teacher_entropy_sum=torch.xlogy(teachers[1:], teachers[1:]).sum(),
        reuse_count=group_size - 1,
    )
    loss, *_ = kernels._compute_sparse_indexer_loss_and_grads(
        q_idx_bshd=torch.zeros(1, rows, 32, 128),
        k_idx_bsd=torch.zeros(1, topk, 128),
        w_bsh=torch.zeros(1, rows, 32),
        topk_indices_cmp=torch.arange(topk, dtype=torch.int32).expand(1, rows, -1),
        topk_length_cmp=lengths,
        indexer_score_payload=logits.detach(),
        query=torch.zeros(rows, 1, 32, 576),
        kv_full=torch.zeros(topk, 1, 576),
        lse=torch.zeros(rows, 32),
        softmax_scale=0.1,
        loss_coeff=coefficient,
        query_valid_rows=None,
        calculate_per_token_loss=False,
        tp_group=None,
        cli_target_accumulator=accumulator,
    )
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(captured["gradient"], reference_gradient)
    if large_gap:
        torch.testing.assert_close(loss / coefficient, torch.tensor(120.0))
