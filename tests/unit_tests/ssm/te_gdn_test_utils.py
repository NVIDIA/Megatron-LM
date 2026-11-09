# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reference recurrence and TE-version adaptation shared by GDN test buckets."""

import importlib

import pytest
import torch

from megatron.core.ssm.gated_delta_net.gdn import torch_chunk_gated_delta_rule


def torch_te_gdn_op(q, k, v, g, beta, cu_seqlens, **kwargs):
    """Use MCore's reference recurrence behind TE's actual layout and shape validation."""
    outputs, states = [], []
    initial_state = kwargs.pop("initial_state", None)
    bounds = cu_seqlens.tolist()
    for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = (
            None if initial_state is None else initial_state[index : index + 1].transpose(-1, -2)
        )
        output, final_state = torch_chunk_gated_delta_rule(
            *(tensor[start:end].unsqueeze(0) for tensor in (q, k, v, g, beta)),
            initial_state=state,
            chunk_size=4,
            **kwargs,
        )
        outputs.append(output.squeeze(0))
        if final_state is not None:
            states.append(final_state.squeeze(0).transpose(-1, -2))
    return torch.cat(outputs), torch.stack(states) if states else None


def patch_te_gdn_op(monkeypatch):
    """Replace the external recurrence while retaining the installed TE frontend."""
    for module_name, hook in (
        ("base", "_import_linear_attention_op"),
        ("gdn", "_import_gated_delta_net"),
    ):
        try:
            module = importlib.import_module(
                f"transformer_engine.pytorch.attention.linear_attention.{module_name}"
            )
        except ModuleNotFoundError:
            continue
        if hasattr(module, hook):
            monkeypatch.setattr(module, hook, lambda *args: torch_te_gdn_op)
            return
    pytest.skip("installed TE GDN exposes no supported recurrence import hook")
