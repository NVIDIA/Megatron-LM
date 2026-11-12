# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The deployment tail delegates packing/PP and owns only LP arithmetic."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model as Native
from megatron.lite.model.deepseek_v41.vllm.model import DeepseekV41Model as Deployment


def test_native_output_delegation_and_masked_lp():
    model = object.__new__(Deployment)
    torch.nn.Module.__init__(model)
    model.layout, model.ps = NS(has_head=True), NS(tp_size=1)
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 2.0, 0.0]], requires_grad=True)
    rows, labels, mask = torch.arange(2), torch.tensor([1, 0]), torch.tensor([1.0, 0.0])
    with patch.object(Native, 'forward', return_value={'rows': rows, 'logits': logits}) as call:
        output = model(
            torch.tensor([0, 1]),
            labels=labels,
            loss_mask=mask,
            temperature=0.5,
            calculate_entropy=True,
        )
        assert call.call_args.kwargs['labels'] is None
        assert call.call_args.kwargs['temperature'] == 0.5
        assert output['rows'] is rows and torch.equal(output['valid_mask'], mask.bool())
        expected = logits.log_softmax(-1)
        assert torch.equal(output['log_probs'], expected[rows, labels])
        assert torch.equal(output['loss'], -expected[0, 1])
        assert torch.equal(output['entropy'], -(expected.exp() * expected).sum(-1))
        output['loss'].backward()
        assert torch.isfinite(logits.grad).all() and logits.grad[1].count_nonzero() == 0
        model.layout.has_head = False
        call.return_value = {'hidden_states': logits}
        assert model(torch.tensor([0, 1]), labels=labels) is call.return_value
        assert call.call_args.kwargs['labels'] is labels
