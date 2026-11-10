# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The declaration rules of megatron.core.inference.tp_buffer_order, on stand-ins (CPU).

test_inference_all_gather_barriers.py checks the declarations of each layer type on GPUs.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference import tp_buffer_order
from megatron.core.transformer.identity_op import IdentityOp


class _StandIn:
    """A linear or embedding reduced to the attributes the declaration rules read."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class _ColumnParallel(_StandIn):
    pass


class _LayerNormColumnParallel(_StandIn):
    pass


class _RowParallel(_StandIn):
    pass


class _Embedding(_StandIn):
    pass


class TestDeclarationRules:

    tp_group = object()

    @pytest.fixture(autouse=True)
    def stand_ins(self, monkeypatch):
        inference_layers = tp_buffer_order.inference_layers
        monkeypatch.setattr(inference_layers, "HAVE_TE", True)
        monkeypatch.setattr(inference_layers, "InferenceColumnParallelLinear", _ColumnParallel)
        monkeypatch.setattr(
            inference_layers, "InferenceLayerNormColumnParallelLinear", _LayerNormColumnParallel
        )
        monkeypatch.setattr(inference_layers, "InferenceRowParallelLinear", _RowParallel)
        monkeypatch.setattr(
            tp_buffer_order.tensor_parallel_layers, "VocabParallelEmbedding", _Embedding
        )

    def _all_gather(self, cls=_LayerNormColumnParallel, in_features=256, fp32_residual=False):
        return cls(
            tp_group=self.tp_group,
            in_features=in_features,
            config=SimpleNamespace(fp32_residual_connection=fp32_residual),
            barrier_before_all_gather=True,
        )

    def _reduce_scatter(self, width=256, tp_group=None, embedding=False):
        tp_group = tp_group or self.tp_group
        if embedding:
            return _Embedding(tp_group=tp_group, embedding_dim=width)
        return _RowParallel(tp_group=tp_group, out_features=width)

    @pytest.mark.parametrize(
        "all_gather_kwargs, reduce_scatter_kwargs, barrier",
        [
            ({}, {}, False),
            ({"cls": _ColumnParallel}, {}, False),
            ({}, {"embedding": True}, False),
            # A different width writes into the segments that peers may still reduce.
            ({"in_features": 128}, {}, True),
            ({"in_features": 128}, {"embedding": True}, True),
            ({}, {"tp_group": object()}, True),
            # An FP32 residual stream doubles the bytes each rank all-gathers.
            ({"fp32_residual": True}, {}, True),
        ],
    )
    def test_after_reduce_scatter(self, all_gather_kwargs, reduce_scatter_kwargs, barrier):
        all_gather = self._all_gather(**all_gather_kwargs)
        tp_buffer_order.skip_all_gather_barrier_after_reduce_scatter(
            self._reduce_scatter(**reduce_scatter_kwargs), all_gather
        )
        assert all_gather.barrier_before_all_gather is barrier

    def test_after_unknown_or_other_reduce_scatter(self):
        all_gather = self._all_gather()
        tp_buffer_order.skip_all_gather_barrier_after_reduce_scatter(None, all_gather)
        tp_buffer_order.skip_all_gather_barrier_after_reduce_scatter(
            _StandIn(tp_group=self.tp_group, out_features=256), all_gather
        )
        assert all_gather.barrier_before_all_gather is True
        other = _StandIn(barrier_before_all_gather=True)
        tp_buffer_order.skip_all_gather_barrier_after_reduce_scatter(self._reduce_scatter(), other)
        assert other.barrier_before_all_gather is True

    def test_after_synchronizing_collective(self):
        all_gather = self._all_gather(_ColumnParallel)
        tp_buffer_order.skip_all_gather_barrier_after_synchronizing_collective(all_gather)
        assert all_gather.barrier_before_all_gather is False
        other = _StandIn(barrier_before_all_gather=True)
        tp_buffer_order.skip_all_gather_barrier_after_synchronizing_collective(other)
        assert other.barrier_before_all_gather is True

    def test_chain(self):
        reduce_scatter = self._reduce_scatter()
        # Modules that do nothing on the buffer pass the preceding reduce-scatter on.
        assert tp_buffer_order.chain_tp_buffer_order(None, reduce_scatter) is reduce_scatter
        assert tp_buffer_order.chain_tp_buffer_order(IdentityOp(), reduce_scatter) is reduce_scatter
        # A module that does not declare its order may do anything there.
        assert tp_buffer_order.chain_tp_buffer_order(torch.nn.Linear(2, 2), reduce_scatter) is None
        declaring = mock.Mock(spec=["declare_tp_buffer_order"])
        declaring.declare_tp_buffer_order.return_value = "last"
        assert tp_buffer_order.chain_tp_buffer_order(declaring, reduce_scatter) == "last"
        declaring.declare_tp_buffer_order.assert_called_once_with(reduce_scatter)
