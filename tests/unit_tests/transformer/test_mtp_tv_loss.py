# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.fusions import fused_mtp_tv as fused_tv_module
from megatron.core.fusions.fused_mtp_tv import vocab_parallel_tv_distance
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    prepare_mtp_sequence_roll_context,
    process_mtp_loss,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_BEBOP_PAPER = "https://arxiv.org/abs/2606.12370"
pytestmark = pytest.mark.launch_on_gb200


def _native_tv_distance(draft_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Independent PyTorch reference for Bebop Eq. (10)."""
    draft_prob = F.softmax(draft_logits.float(), dim=-1)
    target_prob = F.softmax(target_logits.detach().float(), dim=-1)
    return 1.0 - torch.minimum(draft_prob, target_prob).sum(dim=-1)


def _native_e2e_tv_loss(
    draft_logits: list[torch.Tensor], target_logits: list[torch.Tensor]
) -> torch.Tensor:
    """Independent PyTorch reference for Bebop Eq. (13)."""
    per_step_acceptance = torch.stack(
        [
            1.0 - _native_tv_distance(draft_step, target_step)
            for draft_step, target_step in zip(draft_logits, target_logits)
        ],
        dim=0,
    )
    return 1.0 - torch.cumprod(per_step_acceptance, dim=0).mean(dim=0)


def _native_roll_packed_sequence_first(
    tensor: torch.Tensor, cu_seqlens: tuple[int, ...]
) -> torch.Tensor:
    """Independent packed left roll for tensors whose first dimension is sequence."""
    rolled = torch.zeros_like(tensor)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        rolled[start : end - 1] = tensor[start + 1 : end]
    return rolled


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.flatten().double().unsqueeze(0), b.flatten().double().unsqueeze(0)
    ).item()


def _tensor_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    denom = (a * a + b * b).sum()
    return (2.0 * (a * b).sum() / denom).item() if denom else 1.0


def _assert_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5):
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    assert _cosine_sim(a, b) > 1 - eps
    assert _tensor_sim(a, b) > 1 - eps


class _OutputLayer(torch.nn.Module):
    """Small output projection with the same call contract as MCore's output layer."""

    gather_output = True

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, hidden, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        weight = self.weight if weight is None else weight
        return torch.matmul(hidden, weight.t()), None


def _make_tv_config(mtp_num_layers: int, hidden_size: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=1,
        mtp_num_layers=mtp_num_layers,
        mtp_loss_type="e2e_tv",
        mtp_loss_scaling_factor=1.0,
        mtp_detach_heads=True,
        use_cpu_initialization=True,
    )


def test_mtp_loss_type_validation():
    default_config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    assert default_config.mtp_loss_type == "cross_entropy"

    with pytest.raises(ValueError, match="mtp_loss_type must be one of"):
        TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=1, mtp_loss_type="unknown"
        )
    inactive_e2e_tv_config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        mtp_loss_type="e2e_tv",
        mtp_detach_heads=True,
    )
    assert inactive_e2e_tv_config.mtp_num_layers is None
    with pytest.raises(ValueError, match="requires mtp_detach_heads=True"):
        TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            mtp_num_layers=2,
            mtp_loss_type="e2e_tv",
        )


def test_process_mtp_e2e_tv_requires_tp_group_for_sharded_logits(monkeypatch):
    """A missing explicit TP group cannot silently normalize each vocab shard."""
    monkeypatch.setattr(parallel_state, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    draft_logits = torch.randn(2, 1, 7, requires_grad=True)
    target_logits = torch.randn_like(draft_logits)

    with pytest.raises(ValueError, match="tp_group must be provided"):
        vocab_parallel_tv_distance(
            draft_logits, target_logits, tp_group=None, logits_are_vocab_sharded=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_vocab_tv_forward_and_gradient_match_native_reference(dtype):
    """Use GLM-5/5.2's production vocabulary size with paper-native math.

    The reference is derived independently from Eq. (10) in ``_BEBOP_PAPER``;
    it does not invoke MCore's analytical backward.
    """
    torch.manual_seed(1234)
    shape = (2, 1, 128256)
    draft_data = torch.randn(shape, device="cuda", dtype=dtype)
    target_logits = torch.randn(shape, device="cuda", dtype=dtype)
    grad_weight = torch.tensor([[[0.25]], [[1.5]]], device="cuda")

    draft_actual = draft_data.detach().clone().requires_grad_(True)
    actual = vocab_parallel_tv_distance(draft_actual, target_logits, logits_are_vocab_sharded=False)
    (actual * grad_weight.squeeze(-1)).sum().backward()

    draft_reference = draft_data.detach().clone().requires_grad_(True)
    reference = _native_tv_distance(draft_reference, target_logits)
    (reference * grad_weight.squeeze(-1)).sum().backward()

    tolerance = 3e-3 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    assert draft_actual.grad is not None
    assert draft_reference.grad is not None
    _assert_similarity(draft_actual.grad, draft_reference.grad, eps=tolerance)
    assert target_logits.grad is None


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_process_mtp_e2e_tv_matches_native_alignment_and_gradient(calculate_per_token_loss):
    """Check shifted target alignment, Eq. (13), and draft-only gradients on CPU."""
    torch.manual_seed(7)
    mtp_num_layers = 2
    seq_len = 6
    batch_size = 1
    hidden_size = 7
    config = _make_tv_config(mtp_num_layers, hidden_size)
    config.calculate_per_token_loss = calculate_per_token_loss
    output_layer = _OutputLayer(torch.eye(hidden_size))

    hidden_data = torch.randn(
        (1 + mtp_num_layers) * seq_len, batch_size, hidden_size, dtype=torch.float32
    )
    hidden_actual = hidden_data.detach().clone().requires_grad_(True)
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    loss_mask = torch.ones(batch_size, seq_len)

    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    result = process_mtp_loss(
        hidden_states=hidden_actual,
        labels=labels,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=True,
        is_training=False,
        compute_language_model_loss=lambda *_: pytest.fail("cross entropy must not run for e2e TV"),
        config=config,
    )
    result.sum().backward()

    base_hidden, *draft_hidden = torch.chunk(hidden_data, 1 + mtp_num_layers, dim=0)
    target_hidden = base_hidden.detach().permute(1, 2, 0)
    target_logits = []
    draft_logits = []
    draft_references = []
    reference_mask = loss_mask.clone()
    chain_valid = torch.ones_like(loss_mask, dtype=torch.bool)
    for draft_hidden_step in draft_hidden:
        target_hidden = torch.roll(target_hidden, shifts=-1, dims=-1)
        target_hidden[..., -1] = 0
        target_logits.append(target_hidden.permute(2, 0, 1))

        draft_reference = draft_hidden_step.detach().clone().requires_grad_(True)
        draft_references.append(draft_reference)
        draft_logits.append(draft_reference)

        reference_mask = torch.roll(reference_mask, shifts=-1, dims=-1)
        reference_mask[..., -1] = 0
        chain_valid &= reference_mask.bool()

    chain_mask = chain_valid.transpose(0, 1).float()
    reference_loss = _native_e2e_tv_loss(draft_logits, target_logits)
    if calculate_per_token_loss:
        original_num_tokens = loss_mask.sum()
        num_tokens = chain_mask.sum()
        (reference_loss * chain_mask * (original_num_tokens / num_tokens)).sum().backward()
    else:
        (reference_loss * chain_mask).sum().div(chain_mask.sum()).backward()

    assert hidden_actual.grad is not None
    actual_chunks = torch.chunk(hidden_actual.grad, 1 + mtp_num_layers, dim=0)
    torch.testing.assert_close(actual_chunks[0], torch.ones_like(actual_chunks[0]))
    for actual_grad, draft_reference in zip(actual_chunks[1:], draft_references):
        assert draft_reference.grad is not None
        _assert_similarity(actual_grad, draft_reference.grad)
        torch.testing.assert_close(actual_grad, draft_reference.grad, rtol=1e-5, atol=1e-6)
    assert output_layer.weight.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_process_mtp_e2e_tv_uses_fused_tv_prefix_and_logs_each_depth(monkeypatch):
    """The integrated materialized-roll path dispatches both fused objectives."""
    torch.manual_seed(26)
    mtp_num_layers = 2
    sequence_length = 7
    hidden_size = 257
    config = _make_tv_config(mtp_num_layers, hidden_size)
    output_layer = _OutputLayer(torch.eye(hidden_size, device="cuda"))
    hidden_states = torch.randn(
        (1 + mtp_num_layers) * sequence_length, 1, hidden_size, device="cuda", requires_grad=True
    )
    loss_mask = torch.ones(1, sequence_length, device="cuda")
    dp_cp_group = object()
    fused_tv_distances = []
    fused_prefix_losses = []
    logged = []

    original_fused_tv = fused_tv_module._fused_vocab_parallel_tv_distance

    def record_fused_tv(draft_logits, target_logits, tp_group, logits_are_vocab_sharded):
        assert draft_logits.is_contiguous()
        assert target_logits.is_contiguous()
        result = original_fused_tv(draft_logits, target_logits, tp_group, logits_are_vocab_sharded)
        fused_tv_distances.append(result.detach().clone())
        return result

    original_prefix_objective = mtp_module.mtp_e2e_prefix_objective

    def record_prefix_objective(acceptances):
        objective, prefix_losses = original_prefix_objective(acceptances)
        fused_prefix_losses.append(prefix_losses.detach().clone())
        return objective, prefix_losses

    def record_loss(loss_sum, num_tokens, layer_number, num_layers, **kwargs):
        assert num_layers == mtp_num_layers
        assert kwargs["avg_group"] is dp_cp_group
        logged.append((loss_sum.detach().clone(), num_tokens.detach().clone(), layer_number))

    monkeypatch.setattr(fused_tv_module, "_fused_vocab_parallel_tv_distance", record_fused_tv)
    monkeypatch.setattr(mtp_module, "mtp_e2e_prefix_objective", record_prefix_objective)
    monkeypatch.setattr(mtp_module.MTPLossLoggingHelper, "save_loss_to_tracker", record_loss)

    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device="cuda"))
    result = process_mtp_loss(
        hidden_states=hidden_states,
        labels=torch.zeros(1, sequence_length, dtype=torch.long, device="cuda"),
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=True,
        is_training=True,
        compute_language_model_loss=lambda *_: pytest.fail("cross entropy must not run for e2e TV"),
        config=config,
        dp_cp_group=dp_cp_group,
    )
    result.sum().backward()

    assert len(fused_tv_distances) == mtp_num_layers
    assert len(fused_prefix_losses) == 1
    actual_tv_distances = torch.stack(fused_tv_distances)
    expected_prefix_losses = 1.0 - torch.cumprod(1.0 - actual_tv_distances, dim=0)
    torch.testing.assert_close(fused_prefix_losses[0], expected_prefix_losses, rtol=1e-5, atol=1e-6)

    chain_mask = torch.zeros(sequence_length, 1, device="cuda")
    chain_mask[: sequence_length - mtp_num_layers] = 1
    assert len(logged) == mtp_num_layers
    for layer_number, (loss_sum, num_tokens, logged_layer_number) in enumerate(logged):
        assert logged_layer_number == layer_number
        torch.testing.assert_close(num_tokens, chain_mask.sum(), rtol=0, atol=0)
        torch.testing.assert_close(
            loss_sum, torch.sum(fused_prefix_losses[0][layer_number] * chain_mask), rtol=0, atol=0
        )


def test_process_mtp_e2e_tv_masks_incomplete_packed_chains():
    """A fixed-gamma objective must neither wrap nor train across THD segments."""
    torch.manual_seed(17)
    mtp_num_layers = 2
    seq_len = 7
    hidden_size = 5
    config = _make_tv_config(mtp_num_layers, hidden_size)
    output_layer = _OutputLayer(torch.eye(hidden_size))
    hidden_states = torch.randn((1 + mtp_num_layers) * seq_len, 1, hidden_size, requires_grad=True)
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 3, 7], dtype=torch.int32),
        max_seqlen_q=4,
        max_seqlen_kv=4,
        qkv_format="thd",
    )

    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    result = process_mtp_loss(
        hidden_states=hidden_states,
        labels=torch.zeros(1, seq_len, dtype=torch.long),
        loss_mask=torch.ones(1, seq_len),
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=True,
        is_training=False,
        compute_language_model_loss=lambda *_: pytest.fail("cross entropy must not run for e2e TV"),
        config=config,
        packed_seq_params=packed_seq_params,
    )
    result.sum().backward()

    assert hidden_states.grad is not None
    _, draft_0_grad, draft_1_grad = torch.chunk(hidden_states.grad, 3, dim=0)
    expected_valid = torch.tensor([True, False, False, True, True, False, False])
    assert torch.count_nonzero(draft_0_grad[~expected_valid]) == 0
    assert torch.count_nonzero(draft_1_grad[~expected_valid]) == 0
    assert torch.count_nonzero(draft_0_grad[expected_valid]) > 0
    assert torch.count_nonzero(draft_1_grad[expected_valid]) > 0


def test_process_mtp_e2e_tv_contiguous_packed_cp2_matches_global_reference(monkeypatch):
    """Contiguous packed CP rolls target distributions without crossing segments."""
    if Utils.world_size < 2:
        pytest.skip("A distributed run with at least two ranks is required")

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = torch.distributed.get_rank(group=cp_group)
        mtp_num_layers = 2
        global_seq_len = 8
        local_seq_len = global_seq_len // 2
        hidden_size = 5
        cu_seqlens = (0, 6, 8)

        torch.manual_seed(23)
        full_hidden = torch.randn(
            (1 + mtp_num_layers) * global_seq_len, 1, hidden_size, device="cuda"
        )
        full_chunks = torch.chunk(full_hidden, 1 + mtp_num_layers, dim=0)
        local_start = cp_rank * local_seq_len
        local_chunks = [
            chunk.narrow(0, local_start, local_seq_len).clone() for chunk in full_chunks
        ]
        local_hidden = torch.cat(local_chunks, dim=0).requires_grad_(True)
        output_layer = _OutputLayer(torch.eye(hidden_size, device="cuda"))
        config = _make_tv_config(mtp_num_layers, hidden_size)

        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_tensor,
            cu_seqlens_kv=cu_seqlens_tensor,
            cu_seqlens_q_padded=cu_seqlens_tensor,
            cu_seqlens_kv_padded=cu_seqlens_tensor,
            max_seqlen_q=6,
            max_seqlen_kv=6,
            qkv_format="thd",
            cp_partition_mode="contiguous",
        )
        local_loss_mask = torch.ones(1, local_seq_len, device="cuda")
        sequence_roll_context = prepare_mtp_sequence_roll_context(
            tensor=local_loss_mask, cp_group=cp_group, packed_seq_params=packed_seq_params
        )

        captured_target_steps = []
        tv_distance = mtp_module.vocab_parallel_tv_distance

        def capture_target(draft_logits, target_logits, **kwargs):
            captured_target_steps.append(target_logits.detach().clone())
            return tv_distance(draft_logits, target_logits, **kwargs)

        monkeypatch.setattr(mtp_module, "vocab_parallel_tv_distance", capture_target)

        MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device="cuda"))
        result = process_mtp_loss(
            hidden_states=local_hidden,
            labels=torch.zeros(1, local_seq_len, dtype=torch.long, device="cuda"),
            loss_mask=local_loss_mask,
            output_layer=output_layer,
            output_weight=None,
            runtime_gather_output=True,
            is_training=False,
            compute_language_model_loss=lambda *_: pytest.fail(
                "cross entropy must not run for e2e TV"
            ),
            config=config,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            sequence_roll_context=sequence_roll_context,
        )
        result.sum().backward()

        reference_weight = output_layer.weight.detach()
        target_logits = torch.matmul(full_chunks[0].detach(), reference_weight.t())
        full_mask = torch.ones(global_seq_len, 1, device="cuda")
        full_chain_valid = torch.ones_like(full_mask, dtype=torch.bool)
        target_steps = []
        draft_references = []
        draft_steps = []
        for full_draft in full_chunks[1:]:
            target_logits = _native_roll_packed_sequence_first(target_logits, cu_seqlens)
            target_steps.append(target_logits.narrow(0, local_start, local_seq_len))
            full_mask = _native_roll_packed_sequence_first(full_mask, cu_seqlens)
            full_chain_valid &= full_mask.bool()

            draft_reference = (
                full_draft.narrow(0, local_start, local_seq_len)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            draft_references.append(draft_reference)
            draft_steps.append(torch.matmul(draft_reference, reference_weight.t()))

        local_chain_mask = full_chain_valid.narrow(0, local_start, local_seq_len).float()
        assert len(captured_target_steps) == mtp_num_layers
        for captured_target, target_step in zip(captured_target_steps, target_steps):
            torch.testing.assert_close(captured_target, target_step, rtol=0, atol=0)
        reference_loss = _native_e2e_tv_loss(draft_steps, target_steps)
        (reference_loss * local_chain_mask).sum().div(
            local_chain_mask.sum().clamp(min=1)
        ).backward()

        assert local_hidden.grad is not None
        actual_chunks = torch.chunk(local_hidden.grad, 1 + mtp_num_layers, dim=0)
        torch.testing.assert_close(actual_chunks[0], torch.ones_like(actual_chunks[0]))
        for actual_grad, draft_reference in zip(actual_chunks[1:], draft_references):
            assert draft_reference.grad is not None
            torch.testing.assert_close(actual_grad, draft_reference.grad, rtol=1e-5, atol=1e-6)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("logits_are_vocab_sharded", [False, True], ids=["gathered", "sharded"])
def test_vocab_parallel_tv_tp2_matches_full_vocab_reference(logits_are_vocab_sharded):
    """TP gathered/sharded forward and local gradients match a full-vocabulary reference."""
    if Utils.world_size < 2:
        pytest.skip("A distributed run with at least two ranks is required")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = torch.distributed.get_rank(group=tp_group)
        torch.manual_seed(2026)
        full_shape = (2, 1, 128256)
        full_draft_data = torch.randn(full_shape, dtype=torch.float32).cuda()
        full_target = torch.randn(full_shape, dtype=torch.float32).cuda()
        if logits_are_vocab_sharded:
            local_draft_data = full_draft_data.chunk(2, dim=-1)[tp_rank].contiguous()
            local_target = full_target.chunk(2, dim=-1)[tp_rank].contiguous()
        else:
            local_draft_data = full_draft_data
            local_target = full_target

        local_draft = local_draft_data.detach().clone().requires_grad_(True)
        actual = vocab_parallel_tv_distance(
            local_draft,
            local_target,
            tp_group=tp_group,
            logits_are_vocab_sharded=logits_are_vocab_sharded,
        )
        actual.sum().backward()

        full_draft = full_draft_data.detach().clone().requires_grad_(True)
        reference = _native_tv_distance(full_draft, full_target)
        reference.sum().backward()

        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        assert local_draft.grad is not None
        assert full_draft.grad is not None
        expected_local_grad = (
            full_draft.grad.chunk(2, dim=-1)[tp_rank]
            if logits_are_vocab_sharded
            else full_draft.grad
        )
        _assert_similarity(local_draft.grad, expected_local_grad)
        torch.testing.assert_close(local_draft.grad, expected_local_grad, rtol=1e-5, atol=1e-6)
    finally:
        Utils.destroy_model_parallel()
