from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from megatron.lite.model.deepseek_v4.vllm import model as model_module
from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Model
from megatron.lite.model.deepseek_v4.vllm.primitive import logprob as logprob_module

pytestmark = pytest.mark.gpus(1)


def test_model_constructor_owns_logprob_chunk_size() -> None:
    parameters = inspect.signature(DeepseekV4Model.__init__).parameters
    assert parameters["logprob_chunk_size"].default == 8192


class _Layer(nn.Module):
    def __init__(self, hidden: int, mult: int):
        super().__init__()
        self.projection = nn.Linear(hidden, hidden, bias=False)
        self.self_attn = SimpleNamespace(_projection_streams=None)
        self.mult = mult

    def forward(self, hidden_states, **_kwargs):
        value = self.projection(hidden_states)
        return value.unsqueeze(-2).expand(-1, self.mult, -1)


class _Norm(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, value, eps):
        return F.rms_norm(value, (value.shape[-1],), self.weight, eps)


def _head_graph(x, fn, scale, base, eps):
    flat = x.flatten(-2).float()
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(flat, fn.float()) * rstd
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + eps
    return torch.sum(pre.unsqueeze(-1) * x.float(), dim=-2).to(x.dtype)


def test_model_loss_log_probs_and_entropy_cover_embedding_and_head(monkeypatch) -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    tokens, hidden, mult, vocab = 5, 128, 2, 17
    config = SimpleNamespace(
        num_hidden_layers=1,
        vocab_size=vocab,
        rms_norm_eps=1e-5,
        hc_eps=1e-6,
    )
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.config = config
    model.pre_process = True
    model.post_process = True
    model.layer_indices = [0]
    model.embed_tokens = nn.Module()
    model.embed_tokens.embedding = nn.Embedding(vocab, hidden)
    model.layers = nn.ModuleDict({"0": _Layer(hidden, mult)})
    model.norm = _Norm(hidden)
    model.hc_head = nn.Module()
    model.hc_head.hc_fn = nn.Parameter(torch.randn(mult, mult * hidden))
    model.hc_head.hc_scale = nn.Parameter(torch.randn(mult))
    model.hc_head.hc_base = nn.Parameter(torch.randn(mult))
    model.lm_head = nn.Linear(hidden, vocab, bias=False)
    model._logprob_chunk_size = 2
    model.ps = SimpleNamespace(tp_group=None)
    model._input_tensor = None
    model._shared_projection_streams = None
    model.to(device=device, dtype=torch.bfloat16)

    monkeypatch.setattr(
        model_module,
        "mhc_head",
        lambda _visible, x, fn, scale, base, *, eps: _head_graph(
            x, fn, scale, base, eps
        ),
    )

    input_ids = torch.tensor([1, 3, 5, 7, 9], device=device)
    labels = torch.tensor([2, 4, 6, 8, 10], device=device)
    loss_mask = torch.tensor([1, 1, 0, 1, 1], dtype=torch.float32, device=device)
    result = model(
        input_ids=input_ids,
        labels=labels,
        loss_mask=loss_mask,
        temperature=0.7,
        calculate_entropy=True,
    )
    assert result["log_probs"].shape == (tokens,)
    assert result["entropy"].shape == (tokens,)
    objective = (
        result["loss"]
        - 0.01 * result["log_probs"].mean()
        + 0.01 * result["entropy"].mean()
    )
    objective.backward()

    expected_parameters = (
        model.embed_tokens.embedding.weight,
        model.layers["0"].projection.weight,
        model.hc_head.hc_fn,
        model.hc_head.hc_scale,
        model.hc_head.hc_base,
        model.norm.weight,
        model.lm_head.weight,
    )
    for parameter in expected_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    rollout_value = -labels.float()
    monkeypatch.setattr(
        logprob_module,
        "_rollout_selected_log_probs",
        lambda _logits, chunk_labels, _temperature: -chunk_labels.float(),
    )
    with torch.no_grad():
        forward_only = model(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            temperature=1.0,
            calculate_entropy=False,
        )
    torch.testing.assert_close(
        forward_only["log_probs"], rollout_value, rtol=0, atol=0
    )


def test_aligned_chunked_logprob_uses_rollout_value_and_same_head_vjp(
    monkeypatch,
) -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    tokens, hidden, vocab = 5, 64, 19
    temperature = 0.7
    labels = torch.tensor([1, 3, 5, 7, 9], device=device)

    head = nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.bfloat16)
    value = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_()
    reference_weight = head.weight.detach().clone().requires_grad_()
    calls = []

    def rollout_value(logits, chunk_labels, chunk_temperature):
        calls.append(logits.shape[0])
        assert chunk_temperature == temperature
        return 1000.0 + chunk_labels.float()

    monkeypatch.setattr(logprob_module, "_rollout_selected_log_probs", rollout_value)
    actual, entropy = logprob_module.aligned_selected_log_probs(
        value,
        head,
        labels,
        temperature,
        2,
        calculate_entropy=False,
    )
    assert entropy is None
    torch.testing.assert_close(actual, 1000.0 + labels.float(), rtol=0, atol=0)
    actual.sum().backward()

    reference_chunks = []
    for start in range(0, tokens, 2):
        stop = min(start + 2, tokens)
        reference_logits = F.linear(
            reference_value[start:stop], reference_weight
        )
        reference = F.log_softmax(
            reference_logits.float() / temperature, dim=-1
        )
        reference_chunks.append(
            reference.gather(
                -1, labels[start:stop].unsqueeze(-1)
            ).squeeze(-1)
        )
    reference = torch.cat(reference_chunks, dim=0)
    reference.sum().backward()

    assert calls[:3] == [2, 2, 1]
    torch.testing.assert_close(value.grad, reference_value.grad, rtol=0, atol=0)
    torch.testing.assert_close(head.weight.grad, reference_weight.grad, rtol=0, atol=0)

    calls.clear()
    with torch.no_grad():
        forward_only, _ = logprob_module.aligned_selected_log_probs(
            value,
            head,
            labels,
            temperature,
            2,
            calculate_entropy=False,
        )
    torch.testing.assert_close(
        forward_only, 1000.0 + labels.float(), rtol=0, atol=0
    )
    assert calls == [2, 2, 1]


def test_supported_local_head_chunk_matches_unchunked_vllm_logprob() -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    tokens, hidden, vocab = 4096, 128, 257
    labels = torch.randint(0, vocab, (tokens,), device=device)
    value = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    head = nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        chunked, _ = logprob_module.aligned_selected_log_probs(
            value,
            head,
            labels,
            1.0,
            8192,
            calculate_entropy=False,
        )
        unchunked, _ = logprob_module.aligned_selected_log_probs(
            value,
            head,
            labels,
            1.0,
            tokens,
            calculate_entropy=False,
        )

    torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)
