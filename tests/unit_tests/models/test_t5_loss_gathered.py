# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""T5 must tell the loss whether the output layer already gathered the vocabulary.

``T5LMHead`` builds its ``ColumnParallelLinear`` with ``gather_output=not
parallel_output`` and never passes ``runtime_gather_output`` at call time, so the
layout of ``lm_logits`` is fixed at construction. ``T5Model.forward`` therefore has
to forward that fact to ``compute_language_model_loss``: without it the
vocab-parallel cross entropy treats the full vocabulary as one shard, counts the
softmax denominator once per TP rank, and the loss is off by ``log(TP)``.

Only the call-site plumbing is pinned here — the gathered-logits branch of the loss
itself is covered by ``tests/unit_tests/tensor_parallel/test_cross_entropy.py``.
The network is stubbed out so the test stays CPU-only and does not need a
distributed launch.
"""

import torch

from megatron.core.models.T5.t5_model import T5Model


class _FakeOutputLayer:
    def __init__(self, gather_output: bool):
        self.gather_output = gather_output


class _FakeLMHead:
    def __init__(self, gather_output: bool):
        self.output_layer = _FakeOutputLayer(gather_output)

    def __call__(self, hidden_states, word_embeddings_weight=None):
        return torch.zeros(3, 2, 8)


def _fake_t5(gather_output: bool) -> T5Model:
    model = object.__new__(T5Model)
    model.add_decoder = True
    model.pre_process = False
    model.post_process = True
    model.position_embedding_type = "absolute"
    model.share_embeddings_and_output_weights = False
    model.lm_head = _FakeLMHead(gather_output)
    model.decoder = lambda **kwargs: torch.zeros(2, 3, 4)
    return model


def _capture_loss_kwargs(gather_output: bool) -> dict:
    model = _fake_t5(gather_output)
    captured = {}

    def fake_loss(labels, logits, **kwargs):
        captured.update(kwargs)
        return torch.zeros(3, 2)

    model.compute_language_model_loss = fake_loss
    T5Model.forward(
        model,
        encoder_input_ids=torch.zeros(2, 3, dtype=torch.long),
        decoder_input_ids=torch.zeros(2, 3, dtype=torch.long),
        encoder_attn_mask=None,
        decoder_attn_mask=None,
        encoder_decoder_attn_mask=None,
        lm_labels=torch.zeros(3, 2, dtype=torch.long),
        encoder_hidden_states=torch.zeros(2, 3, 4),
    )
    return captured


def test_t5_forwards_gather_output_to_loss():
    """A gathered output layer must reach the loss as ``logits_are_gathered=True``."""
    assert _capture_loss_kwargs(True)["logits_are_gathered"] is True


def test_t5_forwards_sharded_layout_to_loss():
    """A sharded (vocab-parallel) output layer must not claim the logits are gathered."""
    assert _capture_loss_kwargs(False)["logits_are_gathered"] is False
