# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.rl import rl_utils
from megatron.training.arguments import _add_rl_args


class _SizedGroup:
    def __init__(self, size=1):
        self._size = size

    def size(self):
        return self._size


class _CapturingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(flash_decode=True, mtp_num_layers=None, cuda_graph_impl=None)
        self.pg_collection = SimpleNamespace(cp=_SizedGroup(), pp=_SizedGroup(), tp=_SizedGroup())
        self.output_layer = object()
        self.forward_kwargs = None
        self.output_processor = None
        self.output_layer_kwargs = None

    def forward(
        self,
        tokens,
        position_ids,
        attention_mask,
        *,
        output_processor=None,
        output_processor_context=None,
        **kwargs,
    ):
        del position_ids, attention_mask
        self.forward_kwargs = kwargs
        self.output_processor = output_processor
        if output_processor is None:
            return torch.ones(tokens.shape[0], tokens.shape[1], 8)

        def output_layer(hidden_states, **output_layer_kwargs):
            self.output_layer_kwargs = output_layer_kwargs
            sequence, batch, _ = hidden_states.shape
            return torch.ones(sequence, batch, 8), None

        hidden_states = torch.ones(tokens.shape[1], tokens.shape[0], 4)
        return output_processor(
            context=output_processor_context,
            output_layer=output_layer,
            hidden_states=hidden_states,
            output_weight=None,
            scale_logits=lambda value: value,
            input_ids=tokens,
        )


class _NoOutputProcessorModel(_CapturingModel):
    def forward(self, tokens, position_ids, attention_mask, **kwargs):
        del position_ids, attention_mask
        self.forward_kwargs = kwargs
        return torch.ones(tokens.shape[0], tokens.shape[1], 8)


@pytest.fixture
def logprobs_globals(monkeypatch):
    monkeypatch.setattr(
        rl_utils,
        "get_args",
        lambda: SimpleNamespace(
            fp16=False,
            bf16=False,
            cuda_graph_impl="none",
            rl_sequence_packing_max_sequences_per_bin=8,
        ),
    )
    monkeypatch.setattr(rl_utils, "get_nvtx_range", lambda: (lambda *args, **kwargs: nullcontext()))
    monkeypatch.setattr(rl_utils, "is_pp_last_stage", lambda _group: True)
    monkeypatch.setattr(rl_utils, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(rl_utils, "is_batch_invariant_mode_enabled", lambda: False)
    rl_utils._log_vocab_parallel_fallback_once.cache_clear()
    yield
    rl_utils._log_vocab_parallel_fallback_once.cache_clear()


def test_rl_selected_logprobs_flag_defaults_off():
    parser = argparse.ArgumentParser()
    _add_rl_args(parser)
    assert parser.parse_args([]).rl_use_vocab_parallel_selected_logprobs is False


def test_get_logprobs_default_remains_full_gather(logprobs_globals):
    model = _CapturingModel()
    output = rl_utils.get_logprobs(
        model, torch.tensor([[1, 2, 3]]), position_ids=None, packed_seq_params=object()
    )

    assert model.forward_kwargs["runtime_gather_output"] is True
    assert model.output_processor is None
    assert output.shape == (1, 2)
    assert output.dtype == torch.float32


def test_eligible_request_uses_local_vocab_output_processor(monkeypatch, logprobs_globals):
    monkeypatch.setattr(
        rl_utils,
        "vocab_parallel_cross_entropy",
        lambda local_logits, target, **kwargs: local_logits.sum(dim=-1),
    )
    model = _CapturingModel()
    output = rl_utils.get_logprobs(
        model,
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
        position_ids=None,
        packed_seq_params=object(),
        loss_mask=torch.ones(2, 3),
        use_vocab_parallel_selected_logprobs=True,
    )

    assert model.forward_kwargs["runtime_gather_output"] is False
    assert model.output_processor is rl_utils._vocab_parallel_logprobs_output_processor
    assert model.output_layer_kwargs["runtime_gather_output"] is False
    assert output.shape == (2, 3)
    assert output.dtype == torch.float32


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_get_logprobs_rejects_invalid_temperature(logprobs_globals, temperature):
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        rl_utils.get_logprobs(
            _CapturingModel(),
            torch.tensor([[1, 2, 3]]),
            position_ids=None,
            packed_seq_params=object(),
            logprob_temperature=temperature,
        )


def _fallback_reason(model, **overrides):
    options = {
        "cuda_graph_impl": "none",
        "label_smoothing": 0.0,
        "consumer_requires_full_logits": False,
        "consumer_requires_entropy": False,
        "consumer_top_n_logprobs": 0,
        "output_processor_in_use": False,
    }
    options.update(overrides)
    return rl_utils._vocab_parallel_logprobs_fallback_reason(model, model.pg_collection, **options)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"consumer_requires_full_logits": True}, "FULL_LOGITS_REQUESTED"),
        ({"consumer_requires_entropy": True}, "FULL_ENTROPY_REQUESTED"),
        ({"consumer_top_n_logprobs": 4}, "TOP_N_LOGPROBS_REQUESTED"),
        ({"output_processor_in_use": True}, "OUTPUT_PROCESSOR_CONFLICT"),
        ({"label_smoothing": 0.1}, "NONZERO_LABEL_SMOOTHING"),
    ],
)
def test_consumer_fallback_reason_codes(options, reason, logprobs_globals):
    assert _fallback_reason(_CapturingModel(), **options) == reason


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda model: setattr(model.pg_collection, "cp", _SizedGroup(2)), "UNVERIFIED_CP_GT_1"),
        (lambda model: setattr(model.pg_collection, "pp", _SizedGroup(2)), "UNVERIFIED_PP_GT_1"),
        (lambda model: setattr(model.config, "mtp_num_layers", 1), "MTP_ENABLED"),
        (lambda model: setattr(model.config, "cuda_graph_impl", "local"), "CUDA_GRAPH_UNVERIFIED"),
    ],
)
def test_execution_mode_fallback_reason_codes(mutate, reason, logprobs_globals):
    model = _CapturingModel()
    mutate(model)
    assert _fallback_reason(model) == reason


def test_full_iteration_cuda_graph_fallback(logprobs_globals):
    assert (
        _fallback_reason(_CapturingModel(), cuda_graph_impl="full_iteration")
        == "CUDA_GRAPH_UNVERIFIED"
    )


def test_batch_invariant_fallback(monkeypatch, logprobs_globals):
    monkeypatch.setattr(rl_utils, "is_batch_invariant_mode_enabled", lambda: True)
    assert _fallback_reason(_CapturingModel()) == "BATCH_INVARIANT_MODE"


def test_output_processor_unavailable_fallback(logprobs_globals):
    assert _fallback_reason(_NoOutputProcessorModel()) == "OUTPUT_PROCESSOR_UNAVAILABLE"


def test_fallback_log_is_once_per_reason(logprobs_globals, caplog):
    model = _CapturingModel()
    for _ in range(2):
        output = rl_utils.get_logprobs(
            model,
            torch.tensor([[1, 2, 3]]),
            position_ids=None,
            packed_seq_params=object(),
            use_vocab_parallel_selected_logprobs=True,
            consumer_requires_entropy=True,
        )
        assert output.shape == (1, 2)

    expected_log_count = 1 if int(os.environ.get("RANK", "0")) == 0 else 0
    assert caplog.text.count("FULL_ENTROPY_REQUESTED") == expected_log_count
    assert model.forward_kwargs["runtime_gather_output"] is True


def test_masked_target_is_sanitized_before_cross_entropy(monkeypatch):
    captured = {}

    def fake_cross_entropy(local_logits, target, **kwargs):
        del kwargs
        captured["target"] = target.detach().clone()
        return local_logits.sum(dim=-1) + target.to(local_logits.dtype)

    monkeypatch.setattr(rl_utils, "vocab_parallel_cross_entropy", fake_cross_entropy)
    local_logits = torch.randn(2, 5, requires_grad=True)
    target = torch.tensor([3, -100])
    loss_mask = torch.tensor([1.0, 0.0])

    selected = rl_utils._vocab_parallel_selected_logprobs(
        local_logits, target, temperature=1.0, tp_group=object(), loss_mask=loss_mask
    )
    selected.sum().backward()

    torch.testing.assert_close(captured["target"], torch.tensor([3, 0]))
    assert selected[1].item() == 0.0
    assert torch.count_nonzero(local_logits.grad[1]).item() == 0
    assert torch.count_nonzero(local_logits.grad[0]).item() == local_logits.shape[1]


def test_selected_logprobs_reject_shape_mismatches(monkeypatch):
    monkeypatch.setattr(
        rl_utils,
        "vocab_parallel_cross_entropy",
        lambda local_logits, target, **kwargs: local_logits.sum(dim=-1),
    )
    with pytest.raises(ValueError, match="local logits/target shape mismatch"):
        rl_utils._vocab_parallel_selected_logprobs(
            torch.randn(2, 5), torch.tensor([[1, 2]]), temperature=1.0, tp_group=object()
        )
    with pytest.raises(ValueError, match="loss mask/target shape mismatch"):
        rl_utils._vocab_parallel_selected_logprobs(
            torch.randn(2, 5),
            torch.tensor([1, 2]),
            temperature=1.0,
            tp_group=object(),
            loss_mask=torch.ones(1),
        )


def test_output_processor_transposes_native_batch_major_mask(monkeypatch, logprobs_globals):
    captured = {}

    def fake_cross_entropy(local_logits, target, **kwargs):
        del kwargs
        captured["target"] = target.detach().clone()
        return local_logits.sum(dim=-1)

    def output_layer(hidden_states, **kwargs):
        captured["runtime_gather_output"] = kwargs["runtime_gather_output"]
        sequence, batch, _ = hidden_states.shape
        return torch.ones(sequence, batch, 8, dtype=torch.bfloat16), None

    monkeypatch.setattr(rl_utils, "vocab_parallel_cross_entropy", fake_cross_entropy)
    result = rl_utils._vocab_parallel_logprobs_output_processor(
        context={"temperature": 1.0, "tp_group": object(), "loss_mask": torch.ones(2, 3)},
        output_layer=output_layer,
        hidden_states=torch.ones(4, 2, 5),
        output_weight=None,
        scale_logits=lambda value: value,
        input_ids=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
    )

    assert captured["target"].shape == (3, 2)
    assert captured["runtime_gather_output"] is False
    assert result.shape == (2, 3)
    assert result.dtype == torch.bfloat16
