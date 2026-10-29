# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical and MTP integration coverage for weighted self-distillation."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.ops.mtp_kl_loss import vocab_parallel_kl_div, vocab_parallel_log_normalizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    _roll_mtp_teacher,
    process_mtp_loss,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def reference_kl(student, teacher, temperature):
    log_p = F.log_softmax(teacher.float() / temperature, dim=-1)
    log_q = F.log_softmax(student.float() / temperature, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1) * temperature**2


@pytest.fixture(autouse=True)
def reset_mtp_state():
    Utils.initialize_model_parallel()
    MTPLossLoggingHelper.tracker = {}
    MTPLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device="cuda")
    yield
    MTPLossLoggingHelper.tracker = {}
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cached", [False, True])
def test_kl_loss_and_student_gradient(temperature, dtype, cached):
    torch.manual_seed(125)
    student = torch.randn(7, 2, 32, device="cuda", dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    weights = torch.rand(7, 2, device="cuda")
    with torch.no_grad():
        normalizer = vocab_parallel_log_normalizer(teacher, temperature) if cached else None
    actual = vocab_parallel_kl_div(student, teacher, temperature, teacher_log_normalizer=normalizer)
    (actual * weights).sum().backward()
    ref = student.detach().clone().requires_grad_()
    expected = reference_kl(ref, teacher.detach(), temperature)
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(student.grad, ref.grad, rtol=0.02, atol=2e-6)
    assert teacher.grad is None


@pytest.mark.parametrize("tp", [2, 4])
def test_vocab_shards_match_full_vocabulary(tp):
    if torch.distributed.get_world_size() < tp:
        pytest.skip(f"Requires {tp} ranks")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
    group = parallel_state.get_tensor_model_parallel_group()
    torch.manual_seed(198)
    full = torch.randn(5, 2, 32, device="cuda", requires_grad=True)
    teacher = torch.randn_like(full)
    local = full.detach().chunk(tp, dim=-1)[group.rank()].contiguous().requires_grad_()
    actual = vocab_parallel_kl_div(local, teacher.chunk(tp, dim=-1)[group.rank()], 2.0, group)
    expected = reference_kl(full, teacher, 2.0)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(
        local.grad, full.grad.chunk(tp, dim=-1)[group.rank()], atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("ce_weight,kl_weight", [(1.0, 0.0), (0.0, 1.0), (0.3, 0.7), (0.0, 0.0)])
@pytest.mark.parametrize("early_backward", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_weighted_mtp_gradients(ce_weight, kl_weight, early_backward, per_token, training):
    torch.manual_seed(178)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        mtp_num_layers=2,
        mtp_ce_loss_weight=ce_weight,
        mtp_kl_loss_weight=kl_weight,
        mtp_kd_logit_temperature=2.0,
        mtp_loss_early_backward=early_backward,
        calculate_per_token_loss=per_token,
    )
    seq, batch, vocab = 7, 2, 16
    hidden = torch.randn(3 * seq, batch, 8, device="cuda", requires_grad=True)
    main = torch.randn(seq, batch, 8, device="cuda", requires_grad=True)
    weight = torch.randn(vocab, 8, device="cuda", requires_grad=True)
    labels = torch.randint(vocab, (batch, seq), device="cuda")
    mask = torch.tensor([[1, 0, 0.5, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0]], device="cuda")
    valid = torch.tensor([[1, 1, 1, 0, 1, 1, 1], [1] * seq], device="cuda").bool()
    calls = []
    sentinel = torch.zeros(1, device="cuda")

    def project(h, weight, **kwargs):
        calls.append(torch.is_grad_enabled())
        project.embedding_activation_buffer.append(h)
        return h @ weight.t(), None

    project.embedding_activation_buffer = [sentinel]

    def ce(y, logits):
        assert ce_weight > 0
        # Exercise native CE, which can mutate its FP32 logits in place.
        return tensor_parallel.vocab_parallel_cross_entropy(
            logits, y.t().contiguous(), tp_group=parallel_state.get_tensor_model_parallel_group()
        ).t()

    result = process_mtp_loss(
        hidden,
        labels,
        mask,
        project,
        weight,
        training,
        ce,
        config,
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        metric_avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
        main_hidden_states=main,
        mtp_input_mask=valid,
    )
    torch.testing.assert_close(result, main)
    (result * 0).sum().backward()
    kd_enabled = training and kl_weight > 0
    enabled = ce_weight > 0 or kd_enabled
    assert calls == (([False] if kd_enabled else []) + [True] * 2 if enabled else [])
    assert len(project.embedding_activation_buffer) == (3 if enabled else 1)
    assert project.embedding_activation_buffer[0] is sentinel
    rh = hidden.detach().clone().requires_grad_()
    rw = weight.detach().clone().requires_grad_()
    teacher = (rh[:seq] @ rw.t()).detach()
    expected = rh.sum() * 0 + rw.sum() * 0
    for depth in (1, 2):
        student = rh[depth * seq : (depth + 1) * seq - depth] @ rw.t()
        path_valid = valid[:, 1 : seq - depth + 1].clone()
        for j in range(2, depth + 1):
            path_valid &= valid[:, j : seq - depth + j]
        depth_mask = mask[:, depth:] * path_valid
        loss = student.sum(-1).t() * 0
        if ce_weight > 0:
            loss = (
                loss
                + ce_weight
                * F.cross_entropy(
                    student.reshape(-1, vocab), labels[:, depth:].t().reshape(-1), reduction="none"
                )
                .reshape(seq - depth, batch)
                .t()
            )
        if kd_enabled:
            loss = loss + kl_weight * reference_kl(student, teacher[depth:], 2.0).t()
        if training and enabled:
            torch.testing.assert_close(
                MTPLossLoggingHelper.tracker["loss_values"][depth - 1],
                (loss * depth_mask).sum() / depth_mask.sum().clamp(min=1),
                rtol=1e-4,
                atol=4e-6,
            )
        factor = mask.sum() if per_token else 1
        expected = expected + config.mtp_loss_scaling_factor / 2 * factor * (
            loss * depth_mask
        ).sum() / depth_mask.sum().clamp(min=1)
    expected.backward()
    if enabled:
        torch.testing.assert_close(hidden.grad, rh.grad, rtol=1e-4, atol=4e-6)
        torch.testing.assert_close(weight.grad, rw.grad, rtol=1e-4, atol=4e-6)
        assert torch.count_nonzero(hidden.grad[:seq]) == 0
    else:
        assert hidden.grad is None and weight.grad is None
    assert torch.count_nonzero(main.grad) == 0
    assert ("loss_values" in MTPLossLoggingHelper.tracker) == (training and enabled)
    assert ("kd_loss_values" in MTPLossLoggingHelper.tracker) == kd_enabled


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_teacher_alignment_with_packed_padding(layout):
    tensor = torch.arange(24, device="cuda").reshape(12, 1, 2).float()
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 12], device="cuda", dtype=torch.int32),
    )
    shifted = tensor
    for depth in (1, 2, 3):
        shifted = _roll_mtp_teacher(shifted, None, params, layout)
        expected = torch.zeros_like(tensor)
        for start, end in [(0, 3), (4, 9)]:
            if end - start > depth:
                expected[start : end - depth] = tensor[start + depth : end]
        torch.testing.assert_close(shifted, expected)


def test_kd_logging_without_ce_and_after_cleanup():
    group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    for iteration in (1, 2):
        for numerator, count in [(6.0, 2.0), (8.0, 4.0), (0.0, 0.0)]:
            MTPLossLoggingHelper.save_kd_loss_to_tracker(
                torch.tensor(numerator, device="cuda"),
                torch.tensor(count, device="cuda"),
                0,
                1,
                group,
            )
        MTPLossLoggingHelper.save_metrics_to_tracker(
            torch.tensor(2.0, device="cuda"),
            torch.tensor(3.0, device="cuda"),
            torch.tensor(4.0, device="cuda"),
            0,
            1,
            avg_group=group,
        )
        metrics = {}
        MTPLossLoggingHelper.track_mtp_metrics(0.125, iteration, None, total_loss_dict=metrics)
        assert set(metrics) == {"mtp_1 loss", "mtp_1 kd_logit_loss"}
        assert metrics["mtp_1 loss"].item() == 0.25
        assert metrics["mtp_1 kd_logit_loss"].item() == 2.0
        assert MTPLossLoggingHelper.tracker["kd_avg_group"] is group
        assert MTPLossLoggingHelper.tracker["kd_loss_counts"].sum() == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"mtp_kl_loss_weight": -1},
        {"mtp_ce_loss_weight": float("nan")},
        {"mtp_kd_logit_temperature": 0},
        {"mtp_kd_logit_temperature": float("inf")},
        {"mtp_kl_loss_weight": 1, "mtp_num_layers": 0},
    ],
)
def test_invalid_configuration(changes):
    kwargs = dict(num_layers=2, hidden_size=8, num_attention_heads=2, mtp_num_layers=1)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        TransformerConfig(**kwargs)


def test_cli_defaults_and_weights(monkeypatch):
    from megatron.training.arguments import parse_args

    monkeypatch.setattr("sys.argv", ["test"])
    args = parse_args()
    assert (args.mtp_ce_loss_weight, args.mtp_kl_loss_weight, args.mtp_kd_logit_temperature) == (
        1,
        0,
        1,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "test",
            "--mtp-ce-loss-weight",
            "0.3",
            "--mtp-kl-loss-weight",
            "0.7",
            "--mtp-kd-logit-temperature",
            "2",
        ],
    )
    args = parse_args()
    assert (args.mtp_ce_loss_weight, args.mtp_kl_loss_weight, args.mtp_kd_logit_temperature) == (
        0.3,
        0.7,
        2,
    )


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("packed", [False, True])
def test_teacher_alignment_across_cp_ranks(layout, packed):
    if torch.distributed.get_world_size() < 2:
        pytest.skip("Requires two context-parallel ranks")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(context_parallel_size=2)
    group = parallel_state.get_context_parallel_group()
    full = torch.arange(32 * 3, device="cuda").reshape(32, 1, 3).float()
    lengths = [16, 16] if packed else [32]

    def partition(tensor):
        if layout == "contiguous":
            return tensor.chunk(2)[group.rank()]
        parts = []
        for doc in tensor.split(lengths):
            chunks = doc.chunk(4)
            parts.extend([chunks[group.rank()], chunks[3 - group.rank()]])
        return torch.cat(parts)

    params = None
    if packed:
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 16, 32], device="cuda", dtype=torch.int32),
        )
    actual = partition(full)
    for depth in (1, 2, 3):
        actual = _roll_mtp_teacher(actual, group, params, layout)
        expected = torch.cat(
            [torch.cat([doc[depth:], torch.zeros_like(doc[:depth])]) for doc in full.split(lengths)]
        )
        torch.testing.assert_close(actual, partition(expected))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("detach", [False, True])
def test_kd_zero_mask_and_detached_head(empty, detach):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        mtp_num_layers=1,
        mtp_ce_loss_weight=0,
        mtp_kl_loss_weight=1,
        mtp_detach_heads=detach,
    )
    hidden = torch.randn(12, 1, 8, device="cuda", requires_grad=True)
    weight = torch.randn(16, 8, device="cuda", requires_grad=True)
    result = process_mtp_loss(
        hidden,
        torch.arange(6, device="cuda").unsqueeze(0),
        torch.full((1, 6), 0.0 if empty else 1.0, device="cuda"),
        lambda h, weight, **kwargs: (h @ weight.t(), None),
        weight,
        True,
        lambda *args: pytest.fail("KD-only must not compute CE"),
        config,
        metric_avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    (result * 0).sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad[:6]) == 0
    assert (torch.count_nonzero(hidden.grad[6:]) == 0) == empty
    if detach:
        assert weight.grad is None
    else:
        assert (torch.count_nonzero(weight.grad) == 0) == empty


def test_sequence_parallel_projection():
    if torch.distributed.get_world_size() < 2:
        pytest.skip("Requires two tensor-parallel ranks")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    group = parallel_state.get_tensor_model_parallel_group()
    config = TransformerConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        mtp_num_layers=2,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        gradient_accumulation_fusion=False,
        perform_initialization=False,
        mtp_ce_loss_weight=0,
        mtp_kl_loss_weight=1,
    )
    head = tensor_parallel.ColumnParallelLinear(
        8,
        16,
        config=config,
        init_method=lambda t: t,
        bias=False,
        gather_output=False,
        tp_group=group,
    )
    torch.manual_seed(77)
    full_hidden = torch.randn(3, 8, 1, 8, device="cuda")
    full_weight = torch.randn(16, 8, device="cuda") * 0.2
    with torch.no_grad():
        head.weight.copy_(full_weight.chunk(2)[group.rank()])
    local = torch.cat([h.chunk(2)[group.rank()] for h in full_hidden]).detach().requires_grad_()
    result = process_mtp_loss(
        local,
        torch.arange(8, device="cuda").unsqueeze(0),
        None,
        head,
        None,
        True,
        lambda *args: pytest.fail("KD-only must not compute CE"),
        config,
        tp_group=group,
        metric_avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    (result * 0).sum().backward()
    rh = full_hidden.clone().requires_grad_()
    rw = full_weight.clone().requires_grad_()
    teacher = (rh[0] @ rw.t()).detach()
    expected = sum(
        config.mtp_loss_scaling_factor
        / 2
        * reference_kl(rh[depth, :-depth] @ rw.t(), teacher[depth:], 1.0).mean()
        for depth in (1, 2)
    )
    expected.backward()
    torch.testing.assert_close(
        local.grad, torch.cat([g.chunk(2)[group.rank()] for g in rh.grad]), rtol=1e-4, atol=4e-6
    )
    torch.testing.assert_close(
        head.weight.grad, rw.grad.chunk(2)[group.rank()], rtol=1e-4, atol=4e-6
    )


@pytest.mark.parametrize("ce_weight,kd_weight", [(1.0, 0.0), (0.0, 1.0), (0.3, 0.7)])
def test_combined_and_component_logging(ce_weight, kd_weight):
    class Writer:
        def __init__(self):
            self.values = {}

        def add_scalar(self, name, value, iteration):
            self.values[name] = value.item()

        def log(self, values, iteration):
            self.values.update({name: value.item() for name, value in values.items()})

    writer, wandb_writer = Writer(), Writer()
    scalar = lambda value: torch.tensor(value, device="cuda")
    for iteration, multiplier in enumerate((1.0, 2.0), start=1):
        # Unequal token counts exercise the distinct aggregation rules:
        # combined/CE average microbatch means; KD divides total sums by counts.
        for ce, kd, count in ((2.0, 4.0, 2.0), (6.0, 2.0, 4.0)):
            MTPLossLoggingHelper.save_metrics_to_tracker(
                scalar(multiplier * (ce_weight * ce + kd_weight * kd)),
                scalar(1.0),
                scalar(2.0),
                0,
                1,
                ce_loss=scalar(multiplier * ce) if ce_weight else None,
            )
            if kd_weight:
                MTPLossLoggingHelper.save_kd_loss_to_tracker(
                    scalar(multiplier * kd * count), scalar(count), 0, 1, None
                )
        metrics = {}
        MTPLossLoggingHelper.track_mtp_metrics(
            0.5, iteration, writer, wandb_writer, total_loss_dict=metrics
        )
        expected = {"mtp_1 loss": multiplier * (ce_weight * 4.0 + kd_weight * 3.0)}
        if ce_weight:
            expected["mtp_1 ce_loss"] = multiplier * 4.0
        if kd_weight:
            expected["mtp_1 kd_logit_loss"] = multiplier * 16.0 / 6.0
        assert set(metrics) == set(expected)
        for name, value in expected.items():
            assert metrics[name].item() == pytest.approx(value)
            assert writer.values[name] == pytest.approx(value)
            assert wandb_writer.values[name] == pytest.approx(value)
        assert writer.values["mtp_1_acceptance_rate"] == 50.0
        assert writer.values["mtp_1_cumulative_acceptance_rate"] == 50.0
        for key in ("loss_values", "ce_loss_values", "kd_loss_values", "kd_loss_counts"):
            if key in MTPLossLoggingHelper.tracker:
                assert MTPLossLoggingHelper.tracker[key].count_nonzero() == 0


def test_combined_logging_without_acceptance_counters():
    MTPLossLoggingHelper.tracker = {"loss_values": torch.tensor([3.0], device="cuda")}
    metrics = {}
    MTPLossLoggingHelper.track_mtp_metrics(0.5, 1, None, total_loss_dict=metrics)
    assert metrics["mtp_1 loss"].item() == 1.5
