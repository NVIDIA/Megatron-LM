# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Soft-MTP gradient routing in process_mtp_loss.

Gradients are compared in FP64 with an explicit objective using fixed NTP teacher targets
and independent indexing, with no production roll and no router.
"""

import copy
import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp
from megatron.core.transformer.soft_mtp import SoftMTPGradientRouter, validate_soft_mtp_config

# These tests patch and call private MTP helpers.
pytestmark = pytest.mark.internal

GRADIENT_SCALE = 2.75  # Stands in for the pipeline's loss scale.
LOGIT_SCALE = 0.8  # Stands in for MuP output scaling.


class TinySequentialMTP(nn.Module):
    """A backbone and D sequential MTP steps that share one output projection."""

    def __init__(self, depth, repeated=False, tied=False):
        super().__init__()
        self.embedding = nn.Embedding(9, 4)
        self.backbone = nn.Linear(4, 4)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(1 if repeated else depth))
        self.norms = nn.ModuleList(nn.LayerNorm(4) for _ in range(depth + 1))
        self.weight = self.embedding.weight if tied else nn.Parameter(torch.randn(9, 4) / 3)
        self.depth, self.repeated = depth, repeated

    def states(self, tokens):
        """Return the main hidden states followed by each MTP step's hidden states."""
        hidden = self.backbone(self.embedding(tokens)).tanh()
        states = [self.norms[0](hidden)]
        for step in range(self.depth):
            hidden = self.layers[0 if self.repeated else step](hidden).tanh()
            states.append(self.norms[step + 1](hidden))
        return states

    def project(self, hidden, **kwargs):
        """Output layer with the production call signature."""
        return F.linear(hidden, self.weight), None


def ce(labels, logits):
    """Per-token cross-entropy: labels [b, s], logits [s, b, v] -> [b, s]."""
    flat = F.cross_entropy(logits.flatten(0, 1), labels.t().flatten(), reduction="none")
    return flat.view(logits.shape[:2]).t()


def linear_loss(labels, logits):
    """A loss whose gradient is one per position, for closed-form checks."""
    return logits.sum(dim=-1).transpose(0, 1)


def make_config(depth=3, weight=0.35, reduction="per_token", layout="zigzag", **overrides):
    """Every config field that Soft-MTP and process_mtp_loss read."""
    config = SimpleNamespace(
        mtp_num_layers=depth,
        mtp_loss_scaling_factor=0.7,
        soft_mtp_weight=weight,
        calculate_per_token_loss=reduction != "mean",
        mtp_normalize_loss_by_main_tokens=reduction == "summed",
        mtp_detach_heads=False,
        mtp_loss_early_backward=False,
        freeze_base_model_for_mtp=False,
        overlap_moe_expert_parallel_comm=False,
        fp8_output_proj=False,
        cuda_graph_impl="none",
        quant_recipe=None,
        mtp_grad_scale_func=None,
        sequence_parallel=False,
        attention_cp_layout=layout,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def make_batch(case, sequence=12, batch=2):
    """Labels, a loss mask and an MTP input mask for one case."""
    labels = torch.randint(0, 9, (batch, sequence))
    mask = torch.ones(batch, sequence, dtype=torch.float64)
    input_mask = torch.ones(batch, sequence, dtype=torch.bool)
    if case in ("holes", "fractional"):
        mask[0, [0, 4, 7]] = 0
        mask[1, 6:] = 0
        input_mask[:, [3, 8]] = False
    if case == "fractional":
        mask[0, [1, 5]] = 0.5
        mask[1, [2, 3]] = 0.25
    elif case == "empty_head":
        # Two NTP targets; deeper MTP heads have no supported position at all.
        mask.zero_()
        mask[:, 1] = 1
    return labels, mask, input_mask


def reference_loss(states, weight, labels, mask, input_mask, config, segments, scale, alpha=None):
    """Explicit objective whose gradients are the expected Soft-MTP gradients.

    Representation terms hold W fixed; the output projection receives hard-label CE only.
    ``segments`` lists each document's logical [start, end) positions.
    """
    alpha = config.soft_mtp_weight if alpha is None else alpha
    logits = F.linear(states[0], weight) * scale
    loss = (ce(labels, logits) * mask).sum()
    if not config.calculate_per_token_loss:
        loss = loss / mask.sum().clamp(min=1)
    for depth, hidden in enumerate(states[1:], 1):
        positions, sources = [], []
        support = torch.zeros_like(mask)
        for start, end in segments:
            for position in range(start, max(start, end - depth)):
                positions.append(position)
                sources.append(position + depth)
                conditioning = input_mask[:, position + 1 : position + depth + 1].all(dim=-1)
                support[:, position] = mask[:, position + depth] * conditioning
        target = torch.zeros_like(labels)
        target[:, positions] = labels[:, sources]
        teacher = torch.zeros_like(logits)
        teacher[positions] = logits.detach()[sources].softmax(dim=-1)
        student = F.linear(hidden, weight.detach()) * scale
        hard = ce(target, student)
        soft = -(teacher * student.log_softmax(dim=-1)).sum(dim=-1).t()
        projection = ce(target, F.linear(hidden.detach(), weight) * scale)
        coefficient = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if not config.mtp_normalize_loss_by_main_tokens:
            coefficient = coefficient / support.sum().clamp(min=1)
            if config.calculate_per_token_loss:
                coefficient = coefficient * mask.sum()
        objective = (1 - alpha) * hard + alpha * soft + projection
        loss = loss + (objective * support).sum() * coefficient
    return loss


def production_loss(
    model,
    states,
    labels,
    mask,
    input_mask,
    config,
    is_training=True,
    soft_mtp_compatible_loss=True,
    tp_group=None,
):
    """Run process_mtp_loss and return its output and the caller's main CE loss."""
    output = mtp.process_mtp_loss(
        torch.cat(states),
        labels,
        mask,
        model.project,
        None,
        is_training,
        ce,
        config,
        tp_group=tp_group,
        mtp_input_mask=input_mask,
        metric_avg_group=object(),
        main_hidden_states=states[0],
        scale_logits_fn=lambda logits: logits * LOGIT_SCALE,
        soft_mtp_compatible_loss=soft_mtp_compatible_loss,
    )
    loss = (ce(labels, model.project(output)[0] * LOGIT_SCALE) * mask).sum()
    if not config.calculate_per_token_loss:
        loss = loss / mask.sum().clamp(min=1)
    return output, loss


def assert_same_gradients(model, reference, tolerance):
    """Compare parameter gradients by name."""
    for (name, actual), (_, expected) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(
            actual.grad, expected.grad, rtol=tolerance, atol=tolerance, msg=name
        )


@pytest.fixture(autouse=True)
def quiet_metrics(monkeypatch):
    """Skip metric logging and acceptance counts; fix the pipeline's MTP backward scale."""
    monkeypatch.setattr(mtp.MTPLossLoggingHelper, "save_metrics_to_tracker", lambda *a, **k: None)
    monkeypatch.setattr(mtp, "_compute_mtp_acceptance_counts", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(GRADIENT_SCALE)
    )


@pytest.mark.parametrize(
    "reduction,data,alpha,repeated,tied",
    [
        ("mean", "holes", 0.35, False, False),
        ("mean", "empty_head", 0.35, False, False),
        ("per_token", "fractional", 0.35, False, False),
        ("per_token", "empty_head", 0.35, False, False),
        ("summed", "fractional", 0.35, False, False),
        ("per_token", "holes", 1.0, True, True),
    ],
)
def test_gradients_match_stopped_teacher_objective(reduction, data, alpha, repeated, tied):
    """Mask holes, fractional weights, empty heads, and one repeated, tied pure-teacher case."""
    torch.manual_seed(123)
    model = TinySequentialMTP(3, repeated, tied).double()
    reference = copy.deepcopy(model)
    tokens = torch.randint(0, 9, (12, 2))
    labels, mask, input_mask = make_batch(data)
    config = make_config(3, alpha, reduction)

    states = model.states(tokens)
    before = states[0].detach().clone()
    output, loss = production_loss(model, states, labels, mask, input_mask, config)
    torch.testing.assert_close(output, before, rtol=0, atol=0)
    (loss * GRADIENT_SCALE).backward()

    expected = reference_loss(
        reference.states(tokens),
        reference.weight,
        labels,
        mask,
        input_mask,
        config,
        [(0, 12)],
        LOGIT_SCALE,
    )
    (expected * GRADIENT_SCALE).backward()
    assert_same_gradients(model, reference, 2e-10)


@pytest.mark.parametrize(
    "reduction,coefficient", [("mean", 1 / 3), ("per_token", 4 / 3), ("summed", 1.0)]
)
def test_closed_form_head_gradient(reduction, coefficient, monkeypatch):
    """Identity projection and a linear loss: the head gradient is (1 - α) * c_k.

    Four tokens and one head give N0 = 4 and N1 = 3, so c_1 is 1/3 for mean reduction,
    N0 / N1 = 4/3 per token, and 1 for losses summed over main-loss tokens.
    """
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    alpha = 0.5
    config = make_config(1, alpha, reduction, mtp_loss_scaling_factor=1.0)
    hidden = torch.ones(2 * 4, 1, 1, dtype=torch.float64, requires_grad=True)
    output = mtp.process_mtp_loss(
        hidden,
        torch.zeros(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.float64),
        lambda value, **kwargs: (value, None),
        None,
        True,
        linear_loss,
        config,
        metric_avg_group=object(),
        soft_mtp_compatible_loss=True,
    )
    main = linear_loss(None, output).sum()
    if reduction == "mean":
        main = main / 4
    main.backward()
    head = torch.tensor([coefficient * (1 - alpha)] * 3 + [0.0], dtype=torch.float64)
    torch.testing.assert_close(hidden.grad[4:, 0, 0], head)
    main_gradient = 0.25 if reduction == "mean" else 1.0
    torch.testing.assert_close(hidden.grad[:4, 0, 0], torch.full((4,), main_gradient).double())


@pytest.mark.parametrize("reduction", ["mean", "per_token", "summed"])
def test_pure_teacher_cancels_hard_gradient_with_bf16_mask(reduction, monkeypatch):
    """At α = 1 a linear loss leaves the head no gradient, if both sides use one c_k.

    A BF16 mask gives N0 = 857 and N1 = 856, and BF16 rounds N0 to 856. The correction must
    reuse the head's own rounded counts, and mean reduction must undo the caller's exact N0.
    """
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    sequence = 1000
    mask = torch.ones(1, sequence, dtype=torch.bfloat16)
    mask[:, 3::7] = 0
    hidden = torch.ones(2 * sequence, 1, 1, dtype=torch.float64, requires_grad=True)
    output = mtp.process_mtp_loss(
        hidden,
        torch.zeros(1, sequence, dtype=torch.long),
        mask,
        lambda value, **kwargs: (value, None),
        None,
        True,
        linear_loss,
        make_config(1, 1.0, reduction),
        metric_avg_group=object(),
        soft_mtp_compatible_loss=True,
    )
    main = (linear_loss(None, output) * mask.double()).sum()
    if reduction == "mean":
        main = main / 857  # The pipeline divides by the caller's integer token count.
    main.backward()
    torch.testing.assert_close(
        hidden.grad[sequence:, 0, 0], torch.zeros(sequence, dtype=torch.float64), rtol=0, atol=1e-12
    )


def test_router_scales_low_precision_gradients_in_fp32():
    """BF16 gradients are scaled in FP32 and cast back to each head's dtype, bit for bit."""
    torch.manual_seed(4)
    main_grad = torch.randn(6, 2, 3).to(torch.bfloat16)
    corrections = torch.rand(2, 2, 6)
    count = torch.tensor(7.0)
    main = torch.zeros(6, 2, 3, dtype=torch.bfloat16, requires_grad=True)
    heads = [torch.zeros_like(main, requires_grad=True) for _ in range(2)]
    roll = partial(
        mtp._roll_soft_mtp_gradient, cp_group=None, packed_seq_params=None, cp_layout="zigzag"
    )
    SoftMTPGradientRouter.apply(main, corrections, count, roll, None, *heads).backward(main_grad)
    assert torch.equal(main.grad, main_grad)
    routed = (main_grad.float() * count).movedim(0, -1)
    for correction, head in zip(corrections, heads):
        routed = torch.roll(routed, -1, dims=-1)
        routed[..., -1] = 0
        expected = (routed.movedim(-1, 0) * -correction.t().unsqueeze(-1)).to(torch.bfloat16)
        assert head.grad.dtype == torch.bfloat16
        assert torch.equal(head.grad, expected)


def test_each_microbatch_keeps_its_own_weight():
    """Two microbatches with different α; the config changes before either backward."""
    torch.manual_seed(27)
    model = TinySequentialMTP(3, repeated=True).double()
    reference = copy.deepcopy(model)
    config = make_config(3, 0.3, "mean")
    losses, expected = [], []
    for alpha in (0.3, 0.7):
        config.soft_mtp_weight = alpha
        tokens = torch.randint(0, 9, (12, 2))
        labels, mask, input_mask = make_batch("holes")
        losses.append(
            production_loss(model, model.states(tokens), labels, mask, input_mask, config)[1]
        )
        expected.append(
            reference_loss(
                reference.states(tokens),
                reference.weight,
                labels,
                mask,
                input_mask,
                config,
                [(0, 12)],
                LOGIT_SCALE,
                alpha,
            )
        )
    config.soft_mtp_weight = 0.9
    for actual, wanted in zip(losses, expected):
        (actual * GRADIENT_SCALE).backward()
        (wanted * GRADIENT_SCALE).backward()
    assert_same_gradients(model, reference, 1e-10)


def test_routing_is_off_unless_training_with_positive_weight(monkeypatch):
    """At α = 0, in evaluation, and without grad, nothing routes, even with SP enabled."""

    def unexpected_gather(*args, **kwargs):
        raise AssertionError("the Soft-MTP router must not run")

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", unexpected_gather)
    tp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    torch.manual_seed(2)
    tokens = torch.randint(0, 9, (12, 2))
    labels, mask, input_mask = make_batch("holes")

    def run(alpha, is_training, soft_mtp_compatible_loss, sequence_parallel):
        torch.manual_seed(3)
        model = TinySequentialMTP(2).double()
        config = make_config(2, alpha, sequence_parallel=sequence_parallel)
        output, loss = production_loss(
            model,
            model.states(tokens),
            labels,
            mask,
            input_mask,
            config,
            is_training=is_training,
            soft_mtp_compatible_loss=soft_mtp_compatible_loss,
            tp_group=tp_group,
        )
        if output.requires_grad:
            loss.backward()
        return output, [parameter.grad for parameter in model.parameters()]

    _, baseline = run(0.0, True, False, False)
    for alpha, is_training, soft_mtp_compatible_loss in ((0.0, True, True), (1.0, False, False)):
        _, gradients = run(alpha, is_training, soft_mtp_compatible_loss, True)
        for expected, actual in zip(baseline, gradients):
            assert torch.equal(expected, actual)
    with torch.no_grad():
        output, _ = run(1.0, True, False, True)
    assert not output.requires_grad


UNSUPPORTED_SETTINGS = {
    "early_backward": ("mtp_loss_early_backward", True),
    "detached_heads": ("mtp_detach_heads", True),
    "frozen_base": ("freeze_base_model_for_mtp", True),
    "expert_overlap": ("overlap_moe_expert_parallel_comm", True),
    "fp8_output_projection": ("fp8_output_proj", True),
    "cuda_graphs": ("cuda_graph_impl", "local"),
    "quantization_recipe": ("quant_recipe", object()),
    "separate_mtp_scale": ("mtp_grad_scale_func", lambda scale: scale),
    "no_mtp_layers": ("mtp_num_layers", 0),
    "nan_loss_weight": ("mtp_loss_scaling_factor", float("nan")),
    "negative_loss_weight": ("mtp_loss_scaling_factor", -0.1),
    "missing_loss_weight": ("mtp_loss_scaling_factor", None),
}


@pytest.mark.parametrize("case", list(UNSUPPORTED_SETTINGS))
def test_validator_rejects_unsupported_settings(case):
    name, value = UNSUPPORTED_SETTINGS[case]
    with pytest.raises(ValueError, match=f"soft_mtp_weight > 0 .*{name}"):
        validate_soft_mtp_config(make_config(**{name: value}))


def soft_mtp_call(config, labels, loss_mask, soft_mtp_compatible_loss=True, **kwargs):
    """Call process_mtp_loss in a training forward that should raise before any compute."""
    return mtp.process_mtp_loss(
        torch.ones(8, 1, 4, requires_grad=True),
        labels,
        loss_mask,
        None,
        None,
        True,
        ce,
        config,
        soft_mtp_compatible_loss=soft_mtp_compatible_loss,
        **kwargs,
    )


def test_process_mtp_loss_validates_settings():
    config = make_config(1, mtp_loss_early_backward=True)
    with pytest.raises(ValueError, match="mtp_loss_early_backward"):
        soft_mtp_call(config, torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4))


@pytest.mark.parametrize(
    "missing,message",
    [
        ("soft_mtp_compatible_loss", "soft_mtp_compatible_loss=True"),
        ("labels", "requires labels"),
        ("loss_mask", "requires a loss mask"),
    ],
)
def test_requires_a_compatible_main_loss(missing, message):
    with pytest.raises(ValueError, match=message):
        soft_mtp_call(
            make_config(1),
            None if missing == "labels" else torch.zeros(1, 4, dtype=torch.long),
            None if missing == "loss_mask" else torch.ones(1, 4),
            soft_mtp_compatible_loss=missing != "soft_mtp_compatible_loss",
        )


@pytest.mark.parametrize(
    "config_kwargs,call_kwargs,loss_mask,message",
    [
        ({"reduction": "mean"}, {}, torch.full((1, 4), 0.5), "binary loss mask"),
        ({"sequence_parallel": True}, {}, torch.ones(1, 4), "tp_group"),
        (
            {"reduction": "mean"},
            {"cp_group": SimpleNamespace(size=lambda: 2)},
            torch.ones(1, 4),
            "calculate_per_token_loss=True",
        ),
    ],
    ids=[
        "mean_with_fractional_mask",
        "sequence_parallel_without_tp_group",
        "mean_with_context_parallel",
    ],
)
def test_rejects_unsupported_inputs(config_kwargs, call_kwargs, loss_mask, message):
    with pytest.raises(ValueError, match=message):
        soft_mtp_call(
            make_config(1, **config_kwargs),
            torch.zeros(1, 4, dtype=torch.long),
            loss_mask,
            **call_kwargs,
        )


def test_real_config_rejects_weight_outside_unit_interval():
    from megatron.core.transformer.transformer_config import TransformerConfig

    for weight in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match=r"soft_mtp_weight must be in \[0, 1\]"):
            TransformerConfig(
                num_layers=2,
                hidden_size=4,
                num_attention_heads=1,
                use_cpu_initialization=True,
                mtp_num_layers=1,
                soft_mtp_weight=weight,
            )


def test_real_config_passes_validation():
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        use_cpu_initialization=True,
        mtp_num_layers=1,
        soft_mtp_weight=1.0,
    )
    validate_soft_mtp_config(config)


@pytest.fixture(scope="module")
def rank_groups():
    """Rank pairs and 2x2 blocks; the unit-test runner uses NCCL on GPUs, Gloo otherwise."""
    if int(os.environ.get("WORLD_SIZE", "1")) < 4:
        pytest.skip("Soft-MTP routing across ranks needs at least four distributed ranks")
    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            from tests.unit_tests.test_utilities import Utils

            Utils.initialize_distributed()
        else:
            torch.distributed.init_process_group("gloo")
    world_size = torch.distributed.get_world_size()
    assert world_size % 4 == 0
    rank = torch.distributed.get_rank()
    owned = {}
    # Every rank creates every group in the same order.
    for first in range(0, world_size, 2):
        ranks = [first, first + 1]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            owned["pair"] = group
    for first in range(0, world_size, 4):
        for offset in range(2):
            ranks = [first + offset, first + offset + 2]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                owned["stripe"] = group
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    yield SimpleNamespace(rank=rank, device=device, **owned)
    torch.distributed.barrier()
    for group in owned.values():
        torch.distributed.destroy_process_group(group)


def zigzag_partitions(sequence, documents, cp_size=2):
    """Token indices each CP rank holds: chunks r and 2 * cp_size - 1 - r of each document."""
    chunks = torch.arange(sequence).view(documents, 2 * cp_size, -1)
    return torch.stack(
        [chunks[:, [rank, 2 * cp_size - 1 - rank]].reshape(-1) for rank in range(cp_size)]
    )


@pytest.mark.parametrize(
    "layout,packing,reduction",
    [
        ("zigzag", None, "per_token"),
        ("zigzag", "packed", "per_token"),
        ("zigzag", "padded", "summed"),
        ("contiguous", "packed", "per_token"),
    ],
)
def test_context_parallel_gradients(rank_groups, layout, packing, reduction, monkeypatch):
    """CP2 ranks with uneven NTP counts and live cross-rank MTP targets."""
    device, cp_group = rank_groups.device, rank_groups.pair
    cp_rank = rank_groups.rank % 2
    monkeypatch.setattr(
        mtp.MTPLossAutoScaler,
        "main_loss_backward_scale",
        torch.tensor(GRADIENT_SCALE, device=device),
    )
    torch.manual_seed(13)
    sequence, depth = 16, 3
    config = make_config(depth, 0.25, reduction, layout)
    if layout == "contiguous":
        partitions = torch.arange(sequence).view(2, -1)
        boundaries = [0, 6, 16]
    else:
        partitions = zigzag_partitions(sequence, 1 if packing is None else 2)
        boundaries = [0, 8, 16]
    labels = torch.arange(sequence, device=device).remainder(7).view(1, -1)
    mask = torch.ones(1, sequence, dtype=torch.float64, device=device)
    mask[:, partitions[0][:2]] = 0
    mask[:, partitions[1][-1]] = 0
    input_mask = torch.ones_like(mask, dtype=torch.bool)
    input_mask[:, [5, 11]] = False
    segments, packed = [(0, sequence)], None
    if packing is not None:
        logical = boundaries
        if packing == "padded":
            # Logical documents of 6 and 7 tokens in two 8-slot physical documents.
            logical = [0, 6, 13]
            mask[:, [6, 7, 15]] = 0
        lengths = [end - start for start, end in zip(logical[:-1], logical[1:])]
        segments = [(start, start + length) for start, length in zip(boundaries[:-1], lengths)]
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(logical, dtype=torch.int32, device=device),
            cu_seqlens_q_padded=(
                torch.tensor(boundaries, dtype=torch.int32, device=device)
                if packing == "padded"
                else None
            ),
            local_cp_size=2,
            cp_group=cp_group,
        )
    owner = torch.empty(sequence, dtype=torch.long)
    for rank, indices in enumerate(partitions):
        owner[indices] = rank
    crossing = [
        (position, step)
        for step in range(1, depth + 1)
        for start, end in segments
        for position in range(start, end - step)
        if owner[position] != owner[position + step]
        and bool(mask[0, position + step])
        and bool(input_mask[0, position + 1 : position + step + 1].all())
    ]
    assert crossing, "the data must route some gradient across the CP boundary"

    hidden = torch.randn(
        depth + 1, sequence, 1, 4, dtype=torch.float64, device=device, requires_grad=True
    )
    weight = torch.randn(7, 4, dtype=torch.float64, device=device, requires_grad=True)
    local = partitions[cp_rank].to(device)
    output = mtp.process_mtp_loss(
        hidden[:, local].reshape(-1, 1, 4),
        labels[:, local],
        mask[:, local],
        lambda value, **kwargs: (F.linear(value, weight), None),
        None,
        True,
        ce,
        config,
        cp_group=cp_group,
        packed_seq_params=packed,
        mtp_input_mask=input_mask[:, local],
        metric_avg_group=object(),
        soft_mtp_compatible_loss=True,
    )
    loss = (ce(labels[:, local], F.linear(output, weight)) * mask[:, local]).sum()
    (loss * GRADIENT_SCALE).backward()
    torch.distributed.all_reduce(hidden.grad, group=cp_group)
    torch.distributed.all_reduce(weight.grad, group=cp_group)

    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    expected = reference_loss(
        list(reference_hidden), reference_weight, labels, mask, input_mask, config, segments, 1.0
    )
    (expected * GRADIENT_SCALE).backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("with_context_parallel", [False, True])
def test_router_alignment_across_ranks(rank_groups, with_context_parallel):
    """Sequence-parallel gather and slice, alone and with a zigzag CP2 roll.

    Without CP the routed gradient may be a mean, so the router undoes its count first. With
    CP only per-token and summed losses are supported, so the count is one.
    """
    device = rank_groups.device
    cp_rank, tp_rank = divmod(rank_groups.rank % 4, 2)
    sequence = 16
    if with_context_parallel:
        indices = torch.arange(sequence).view(4, 4)[[cp_rank, 3 - cp_rank]].reshape(-1)
        cp_group, count = rank_groups.stripe, 1.0
    else:
        indices, cp_group, count = torch.arange(sequence), None, 5.0
    local = indices.chunk(2)[tp_rank]
    values = torch.arange(48, dtype=torch.float64, device=device).view(sequence, 1, 3) / 13
    donor = values[local.to(device)] / count
    main = torch.zeros_like(donor, requires_grad=True)
    heads = [torch.zeros_like(main, requires_grad=True) for _ in range(3)]
    corrections = torch.stack(
        [
            torch.full((1, len(indices)), value, dtype=torch.float64, device=device)
            for value in (0.2, 0.4, 0.1)
        ]
    )
    corrections[:, :, 2] = 0
    roll = partial(
        mtp._roll_soft_mtp_gradient, cp_group=cp_group, packed_seq_params=None, cp_layout="zigzag"
    )
    result = SoftMTPGradientRouter.apply(
        main,
        corrections,
        torch.tensor(count, dtype=torch.float64, device=device),
        roll,
        rank_groups.pair,
        *heads,
    )
    result.backward(donor)
    torch.testing.assert_close(main.grad, donor, atol=0, rtol=0)
    for step, head in enumerate(heads, 1):
        expected = torch.zeros_like(values)
        expected[:-step] = -values[step:]
        expected = expected[indices.to(device)] * corrections[step - 1].t().unsqueeze(-1)
        expected = expected.chunk(2)[tp_rank]
        torch.testing.assert_close(head.grad, expected, atol=1e-14, rtol=1e-14)
