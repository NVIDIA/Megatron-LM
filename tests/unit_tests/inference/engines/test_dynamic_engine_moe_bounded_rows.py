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
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.enums import InferenceCudaGraphScope

DECODE = InferenceBatchDimensions(token_count=8, prefill_req_count=0, decode_req_count=8)
MIXED = InferenceBatchDimensions(token_count=64, prefill_req_count=1, decode_req_count=7)


def _engine(variants, draft_bounded=False):
    """An engine whose context reports `variants` per CUDA graph and one MTP depth, recording
    each base capture and each MTP graph key."""
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.inference_cuda_graph_scope = InferenceCudaGraphScope.block
    engine.cuda_graph_impl = "local"
    engine.capture_stats = None
    engine.context = SimpleNamespace(
        cuda_graph_batch_dimensions_list=list(variants),
        bounded_moe_row_graph_variants=variants.__getitem__,
        mtp_draft_moe_rows_bounded=draft_bounded,
        enable_mtp_kv_cache=False,
        reset=mock.Mock(),
    )
    engine.captured = []
    engine.mtp_keys = []

    def context_init(**kwargs):
        engine.captured.append(
            (kwargs["construct_graph_dimensions"], kwargs["construct_graph_bounded_moe_rows"])
        )
        return None, None, None

    def compute_mtp_single_step(**kwargs):
        # A captured graph fixes the bound the forward ran with; its key must say so.
        assert kwargs["cache_key"][-1] == InferenceMode.use_bounded_mxfp8_rows()
        engine.mtp_keys.append(kwargs["cache_key"])

    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    sequence_parallel=False,
                    moe_enable_routing_replay=False,
                    hidden_size=8,
                    params_dtype=torch.float32,
                )
            ),
            tp_group=None,
        ),
        _language_model=SimpleNamespace(
            mtp=SimpleNamespace(mtp_use_repeated_layer=True),
            compute_mtp_single_step=compute_mtp_single_step,
        ),
        num_mtp_depths=1,
        num_speculative_tokens=1,
        _sampling_backend="torch",
        _dynamic_step_context_init=context_init,
        _dynamic_step_forward_logits=mock.Mock(),
    )
    return engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graph capture reads CUDA memory stats")
@pytest.mark.parametrize("draft_bounded", [False, True])
def test_create_cuda_graphs_captures_each_bounded_moe_row_variant(draft_bounded):
    """Every base variant a step can replay is captured: at construction the unbounded ones,
    and once the communicator makes the agreement possible the bounded ones. MTP drafts run with
    one bound in every step, so each draft size is captured once, at construction, under a key
    that carries that bound. The capture statistics cover both passes."""
    mixed = InferenceBatchDimensions(token_count=64, prefill_req_count=2, decode_req_count=2)
    variants = {mixed: (False,), DECODE: (False,)}
    engine = _engine(variants, draft_bounded=draft_bounded)

    with InferenceMode.active():
        engine.create_cuda_graphs()
        first_pass = dict(engine.capture_stats)
        assert engine.captured == [(mixed, False), (DECODE, False)]
        assert engine.mtp_keys == [("mtp", 4, None, draft_bounded), ("mtp", 8, None, draft_bounded)]

        # The tp_ep communicator is attached.
        variants[DECODE] = (False, True)
        engine.captured.clear()
        engine.mtp_keys.clear()
        engine.create_cuda_graphs(bounded_moe_rows_only=True)

    assert engine.captured == [(DECODE, True)]
    assert engine.mtp_keys == []
    assert engine.capture_stats["time"] >= first_pass["time"]


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
