# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``process_mtp_loss`` with ``loss_group_lengths``: one call over several packed loss groups.

Each group must keep the token-count normalization it would have in its own call, so combining
forwards (as shared-prefix execution does) does not change any MTP gradient.
"""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
        attention_cp_layout="zigzag",
    )
    return process_mtp_loss(
        hidden_states=torch.cat(hidden_depths, dim=0),
        labels=None,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=None,
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


def test_grouped_mtp_normalization_rejects_contiguous_cp():
    """A contiguous CP layout splits groups by pack position, so grouping would change counts."""
    from types import SimpleNamespace

    from megatron.core.transformer.multi_token_prediction import process_mtp_loss

    config = SimpleNamespace(
        mtp_num_layers=1,
        mtp_detach_heads=False,
        calculate_per_token_loss=True,
        attention_cp_layout="contiguous",
    )
    with pytest.raises(NotImplementedError, match="zigzag CP layout"):
        process_mtp_loss(
            hidden_states=torch.zeros(8, 1, _HIDDEN),
            labels=torch.zeros(1, 4, dtype=torch.long),
            loss_mask=torch.ones(1, 4),
            output_layer=None,
            output_weight=None,
            is_training=True,
            compute_language_model_loss=None,
            config=config,
            cp_group=SimpleNamespace(size=lambda: 2),
            loss_group_lengths=(2, 2),
        )


def _grouped_call(packed_seq_params, groups):
    from types import SimpleNamespace

    from megatron.core.transformer.multi_token_prediction import process_mtp_loss

    config = SimpleNamespace(
        mtp_num_layers=1,
        mtp_detach_heads=False,
        calculate_per_token_loss=True,
        attention_cp_layout="zigzag",
    )
    process_mtp_loss(
        hidden_states=torch.zeros(16, 1, _HIDDEN),
        labels=torch.zeros(1, 8, dtype=torch.long),
        loss_mask=torch.ones(1, 8),
        output_layer=None,
        output_weight=None,
        is_training=True,
        compute_language_model_loss=None,
        config=config,
        packed_seq_params=packed_seq_params,
        loss_group_lengths=groups,
    )


def test_grouped_mtp_normalization_requires_packed_sequences():
    """Without packed sequences the label roll would cross the group boundary."""
    with pytest.raises(ValueError, match="requires packed_seq_params"):
        _grouped_call(None, (4, 4))


def test_grouped_mtp_normalization_rejects_groups_splitting_a_sequence():
    """Every group boundary must be a packed-sequence boundary."""
    cumulative = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cumulative, cu_seqlens_kv=cumulative)
    with pytest.raises(ValueError, match="whole packed sequences"):
        _grouped_call(packed, (3, 5))
