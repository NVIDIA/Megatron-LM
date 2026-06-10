from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.primitive.data import _resolve_thd_padding, fixed_batches
from megatron.lite.primitive.deterministic import deterministic_requested
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.gated_delta_rule import l2norm, torch_chunk_gated_delta_rule
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_log_probs_from_logits
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.primitive.train_step import compute_and_clip_grad_norm, run_microbatch_loop
from megatron.lite.primitive.utils import ensure_divisible

pytestmark = pytest.mark.mlite


def test_vocab_parallel_cross_entropy_matches_torch_cross_entropy():
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0, 4, 2], [3, 1, 0]])

    loss = vocab_parallel_cross_entropy(logits, labels)
    expected = F.cross_entropy(
        logits.float().reshape(-1, 5), labels.reshape(-1), reduction="none"
    ).view_as(labels)

    torch.testing.assert_close(loss, expected)
    loss.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_logprob_selects_labels_in_batch_sequence_or_sequence_batch_layout():
    logits = torch.randn(2, 3, 4)
    labels_bsh = torch.tensor([[0, 1], [2, 3], [1, 0]])

    selected = vocab_parallel_log_probs_from_logits(logits, labels_bsh)
    expected = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, labels_bsh.transpose(0, 1).unsqueeze(-1))
        .squeeze(-1)
        .transpose(0, 1)
        .contiguous()
    )

    torch.testing.assert_close(selected, expected)
    with pytest.raises(ValueError, match="Could not align"):
        vocab_parallel_log_probs_from_logits(logits, torch.zeros(4, 2, dtype=torch.long))


def test_linear_cross_entropy_fallback_matches_explicit_matmul():
    hidden = torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -0.5]])
    weight = torch.tensor(
        [[0.5, -1.0, 0.25], [1.0, 0.0, -0.5], [-0.5, 0.75, 1.0], [0.25, 0.5, -1.5]]
    )
    labels = torch.tensor([0, 3])

    log_probs, entropy = linear_cross_entropy(hidden, weight, labels, temperature=2.0)
    logits = hidden.matmul(weight.t()) / 2.0
    expected_loss = F.cross_entropy(logits, labels, reduction="none")
    expected_entropy = torch.distributions.Categorical(logits=logits).entropy()

    torch.testing.assert_close(log_probs, -expected_loss)
    torch.testing.assert_close(entropy, expected_entropy)


def test_gated_delta_rule_math_helpers_return_finite_stateful_outputs():
    x = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    normalized = l2norm(x, dim=-1, eps=0.0)
    torch.testing.assert_close(normalized.norm(dim=-1), torch.ones(2))

    query = torch.randn(1, 3, 1, 2)
    key = torch.randn(1, 3, 1, 2)
    value = torch.randn(1, 3, 1, 2)
    g = -torch.rand(1, 3, 1)
    beta = torch.rand(1, 3, 1)

    output, final_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=2,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    assert output.shape == value.shape
    assert final_state is not None
    assert final_state.shape == (1, 1, 2, 2)
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


def test_data_padding_env_and_fixed_batches_are_deterministic(monkeypatch):
    monkeypatch.delenv("MEGATRON_LITE_THD_PAD_TO_ALIGNMENT", raising=False)
    monkeypatch.delenv("MEGATRON_LITE_THD_PAD_MULTIPLE", raising=False)
    assert _resolve_thd_padding(seq_len=7, cp_size=2) == (8, 4, True)
    assert _resolve_thd_padding(seq_len=7, cp_size=1) == (7, 1, False)

    monkeypatch.setenv("MEGATRON_LITE_THD_PAD_TO_ALIGNMENT", "0")
    monkeypatch.setenv("MEGATRON_LITE_THD_PAD_MULTIPLE", "8")
    assert _resolve_thd_padding(seq_len=7, cp_size=2) == (7, 8, False)
    monkeypatch.setenv("MEGATRON_LITE_THD_PAD_MULTIPLE", "bad")
    with pytest.raises(ValueError, match="PAD_MULTIPLE"):
        _resolve_thd_padding(seq_len=7, cp_size=2)

    batches_a = fixed_batches(11, seq_len=4, num_steps=2, batch_size=2, device="cpu", seed=123)
    batches_b = fixed_batches(11, seq_len=4, num_steps=2, batch_size=2, device="cpu", seed=123)
    for (ids_a, labels_a), (ids_b, labels_b) in zip(batches_a, batches_b, strict=True):
        torch.testing.assert_close(ids_a, ids_b)
        torch.testing.assert_close(labels_a, labels_b)


def test_deterministic_env_request_parser(monkeypatch):
    monkeypatch.setenv("MEGATRON_LITE_DETERMINISTIC", "yes")
    assert deterministic_requested()
    monkeypatch.setenv("MEGATRON_LITE_DETERMINISTIC", "0")
    assert not deterministic_requested()


def test_recompute_parser_and_wrapper_replays_forward_on_backward():
    assert parse_recompute_spec(None) == []
    assert parse_recompute_spec("none") == []
    assert parse_recompute_spec("full") == ["full"]
    assert parse_recompute_spec("attn,mlp") == ["attn", "mlp"]
    assert parse_recompute_spec(["attn"]) == ["attn"]

    class CountingModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x * x

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = CountingModule()

    layer = Layer()
    apply_recompute(
        nn.ModuleList([layer]),
        ["inner"],
        {"inner": lambda module: module.inner},
        no_rng_modules={"inner"},
    )

    x = torch.tensor([2.0, -3.0], requires_grad=True)
    layer.inner(x).sum().backward()

    assert layer.inner.calls == 2
    torch.testing.assert_close(x.grad, torch.tensor([4.0, -6.0]))


def test_train_step_microbatch_loop_and_grad_clip_cpu_contract():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    data = iter(
        [
            {"x": torch.tensor([[1.0, 2.0]]), "y": torch.tensor([[1.0]])},
            {"x": torch.tensor([[3.0, 4.0]]), "y": torch.tensor([[2.0]])},
        ]
    )

    def forward_fn(module, batch):
        return {"loss": F.mse_loss(module(batch["x"]), batch["y"])}

    output = run_microbatch_loop(model, data, 2, forward_fn)

    assert output is not None
    assert output["loss"].ndim == 0
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()

    grad_norm = compute_and_clip_grad_norm(model, optimizer=None, max_norm=0.25, use_dist_opt=False)

    assert torch.isfinite(grad_norm)
    assert model.weight.grad.norm() <= 0.25 + 1.0e-6


def test_utils_ensure_divisible_returns_quotient_and_reports_context():
    assert ensure_divisible(12, 3, "tp") == 4
    with pytest.raises(ValueError, match=r"10 is not divisible by 4 \(tp\)"):
        ensure_divisible(10, 4, "tp")
