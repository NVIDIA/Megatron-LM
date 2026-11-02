# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA graphs of the bounded FlashInfer MoE rows in the dynamic inference engine.

A captured graph fixes whether its MoE kernels process only the bounded rows. When the replicas
sharing the MoE collectives agree on each step's phase, a decode-only graph is captured with and
without the bound; the bounded one once the engine's tp_ep communicator makes agreement possible.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.transformer.enums import InferenceCudaGraphScope

DECODE = InferenceBatchDimensions(token_count=8, prefill_req_count=0, decode_req_count=8)
MIXED = InferenceBatchDimensions(token_count=64, prefill_req_count=1, decode_req_count=7)


def _engine(variants):
    """An engine whose context reports `variants` per CUDA graph, recording each capture."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.inference_cuda_graph_scope = InferenceCudaGraphScope.block
    engine.cuda_graph_impl = "local"
    engine.context = SimpleNamespace(
        cuda_graph_batch_dimensions_list=list(variants),
        bounded_moe_row_graph_variants=variants.__getitem__,
        reset=mock.Mock(),
    )
    engine.captured = []

    def context_init(**kwargs):
        engine.captured.append(
            (kwargs["construct_graph_dimensions"], kwargs["construct_graph_bounded_moe_rows"])
        )
        return None, None, None

    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(sequence_parallel=False, moe_enable_routing_replay=False)
            )
        ),
        _language_model=SimpleNamespace(),
        num_mtp_depths=0,
        num_speculative_tokens=0,
        _sampling_backend="torch",
        _dynamic_step_context_init=context_init,
        _dynamic_step_forward_logits=mock.Mock(),
    )
    return engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graph capture reads CUDA memory stats")
def test_create_cuda_graphs_captures_each_bounded_moe_row_variant():
    """Every variant a step can replay is captured, and the communicator adds the bounded ones."""
    engine = _engine({MIXED: (False,), DECODE: (False, True)})

    engine.create_cuda_graphs()
    assert engine.captured == [(MIXED, False), (DECODE, False), (DECODE, True)]

    engine.captured.clear()
    engine.create_cuda_graphs(bounded_moe_rows_only=True)
    assert engine.captured == [(DECODE, True)]


@pytest.mark.parametrize("agreed", [False, True])
def test_attaching_the_tp_ep_communicator_captures_the_bounded_graphs(agreed):
    """Once the replicas can agree that a step is decode-only, its graphs may bound the rows."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(set_ep_zmq_communicator=mock.Mock())
    engine.context.set_ep_zmq_communicator.side_effect = lambda communicator: setattr(
        engine.context, "moe_step_phase_is_agreed", agreed
    )
    engine.create_cuda_graphs = mock.Mock()
    communicator = object()

    engine._attach_tp_ep_communicator(communicator)

    assert engine.expert_parallel_zmq_communicator is communicator
    engine.context.set_ep_zmq_communicator.assert_called_once_with(communicator)
    if agreed:
        engine.create_cuda_graphs.assert_called_once_with(bounded_moe_rows_only=True)
    else:
        engine.create_cuda_graphs.assert_not_called()
