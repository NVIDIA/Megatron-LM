# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock

import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe import fused_a2a, token_dispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


def test_destroy_model_parallel_releases_nccl_ep_zero_copy_buffers(monkeypatch):
    """A later non-zero-copy dispatcher must not inherit the previous context's buffers."""
    manager_type = token_dispatcher._NCCLEPManager
    buffer_names = ('_zc_fwd_token_buf', '_zc_bwd_token_buf', '_zc_recv_topk_weights_buf')
    for name in buffer_names:
        monkeypatch.setattr(manager_type, name, torch.empty(4))

    buffers_at_finalize = []

    def finalize():
        buffers_at_finalize.append(tuple(getattr(manager_type, name) for name in buffer_names))

    # Observe either teardown path so the regression also fails with the old direct finalizer.
    monkeypatch.setattr(fused_a2a, 'nccl_ep_finalize', finalize)
    monkeypatch.setattr(token_dispatcher, 'nccl_ep_finalize', finalize)
    parallel_state.destroy_model_parallel()

    assert len(buffers_at_finalize) == 1
    assert all(buffer is None for buffer in buffers_at_finalize[0])

    # Exercise the next manager's actual constructor, bootstrap, dispatch and combine on CPU;
    # only the external NCCL EP operations are replaced.
    hidden_states = torch.empty(2, 4)
    dispatch = Mock(return_value=(hidden_states, torch.tensor([2]), torch.ones(2)))
    combine = Mock(return_value=hidden_states)
    bootstrap = Mock()
    monkeypatch.setattr(token_dispatcher, 'ensure_nccl_ep_bootstrapped', bootstrap)
    monkeypatch.setattr(token_dispatcher, 'new_nccl_ep_buffer', Mock(return_value=object()))
    monkeypatch.setattr(token_dispatcher, 'nccl_ep_dispatch', dispatch)
    monkeypatch.setattr(token_dispatcher, 'nccl_ep_combine', combine)

    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        num_moe_experts=1,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_ncclep_zero_copy=False,
    )
    manager = manager_type(
        group=object(), num_local_experts=1, router_topk=1, num_experts=1, config=config
    )
    manager.setup_metadata(torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1))
    received = manager.dispatch(hidden_states)
    manager.combine(received)

    bootstrap.assert_called_once()
    assert bootstrap.call_args.kwargs['zero_copy'] is False
    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs['recv_tokens'] is None
    assert dispatch.call_args.kwargs['recv_topk_weights'] is None
    combine.assert_called_once()
    assert combine.call_args.kwargs['grad_out'] is None
