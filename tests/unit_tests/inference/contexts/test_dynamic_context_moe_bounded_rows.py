# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded FlashInfer MoE rows in the dynamic inference context.

With ``inference_flashinfer_mxfp8_token_capacity`` set, the experts of a decode-only step process
only the first ``capacity`` rows of the tokens gathered over the expert tensor-and-model-parallel
(tp_ep) group. The context sizes that bound for the decode tokens of every data-parallel replica
in the group, uses it only in steps that every replica agrees are decode-only, and keys CUDA
graphs on it, since a captured graph fixes it.
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

CAPACITY = 128


class _ScriptedCommunicator:
    """Stands in for the engine's tp_ep ZMQ communicator, with the peers' votes scripted."""

    def __init__(self, peers_not_decode_only: bool):
        self.peers = int(peers_not_decode_only)
        self.sent = []

    def sync_all_reduce_max(self, value):
        self.sent.append(value)
        return max(value, self.peers)


class TestBoundedMoeRows:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _context(*, tp=1, etp=1, ep=2, max_requests=64, num_cuda_graphs=8, communicator=None):
        """A context for an NVLS FlashInfer MoE model with a 128-row token capacity."""
        model_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_ffn_hidden_size=128,
            moe_grouped_gemm=True,
            moe_router_dtype="fp32",
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            add_bias_linear=False,
            activation_func=squared_relu,
            inference_grouped_gemm_backend="flashinfer",
            inference_moe_nvfp4_recipe="per_token",
            inference_moe_token_dispatcher_type="nvls",
            inference_flashinfer_mxfp8_token_capacity=CAPACITY,
            tensor_model_parallel_size=tp,
            expert_tensor_parallel_size=etp,
            expert_model_parallel_size=ep,
        )
        ctx = DynamicInferenceContext(
            model_config=model_config,
            inference_config=InferenceConfig(
                max_sequence_length=256,
                buffer_size_gb=0.05,
                block_size_tokens=128,
                max_tokens=256,
                max_requests=max_requests,
                num_cuda_graphs=num_cuda_graphs,
                unified_memory_level=0,
            ),
        )
        ctx.set_ep_zmq_communicator(communicator)
        return ctx

    @staticmethod
    def _decode_graph(ctx):
        return min(
            dims for dims in ctx.cuda_graph_batch_dimensions_list if dims.prefill_req_count == 0
        )

    @staticmethod
    def _dummy_decode_step(ctx):
        """An idle rank's dummy decode step: agree, then initialize the attention state.
        Returns whether it bounds its rows, whether it replays a graph, and that graph's bound."""
        ctx.agree_on_moe_decode_only(True)
        ctx.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=True)
        result = (
            ctx.use_bounded_mxfp8_rows,
            ctx.using_cuda_graph_this_step(),
            ctx.cuda_graph_key[1],
        )
        ctx.reset()
        return result

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "etp, ep, max_requests, bounded",
        [
            # 2 replicas x 64 decode rows fit the 128-row capacity.
            (1, 2, 64, True),
            # ETP > TP: the NVLS all-gather holds 4 replicas x 64 decode rows.
            (2, 2, 64, False),
            (2, 2, 32, True),
        ],
    )
    def test_bound_holds_every_replicas_decode_tokens(self, etp, ep, max_requests, bounded):
        """Bounded MoE rows must hold the decode tokens of every replica in the tp_ep group."""
        ctx = self._context(
            etp=etp,
            ep=ep,
            max_requests=max_requests,
            num_cuda_graphs=None,
            communicator=_ScriptedCommunicator(peers_not_decode_only=False),
        )
        assert self._dummy_decode_step(ctx) == (bounded, False, bounded)

    @pytest.mark.internal
    def test_decode_step_bounds_rows_only_when_every_replica_is_decode_only(self):
        """A decode-only rank bounds its MoE rows only when every replica agrees that it is
        decode-only too; a peer's prefill tokens sit past the capacity in the gathered rows.
        Either way it replays a decode-only CUDA graph, captured with or without the bound."""
        peers_decode = _ScriptedCommunicator(peers_not_decode_only=False)
        ctx = self._context(communicator=peers_decode)
        assert self._dummy_decode_step(ctx) == (True, True, True)
        assert peers_decode.sent == [0]

        ctx.set_ep_zmq_communicator(_ScriptedCommunicator(peers_not_decode_only=True))
        assert self._dummy_decode_step(ctx) == (False, True, False)

        # A prefilling rank tells its peers, and runs unbounded itself.
        ctx.set_ep_zmq_communicator(peers_decode)
        ctx.add_request(
            DynamicInferenceRequest(
                request_id=1,
                prompt_tokens=torch.arange(0, 8, device='cpu'),
                sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
            )
        )
        ctx.agree_on_moe_decode_only(ctx.num_prefill_requests == 0)
        ctx.initialize_attention_state()
        assert not ctx.use_bounded_mxfp8_rows
        assert peers_decode.sent == [0, 1]
        ctx.reset()

        # Without the engine's tp_ep communicator the replicas cannot agree.
        ctx.set_ep_zmq_communicator(None)
        assert self._dummy_decode_step(ctx) == (False, True, False)

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "tp, etp, ep, max_requests, communicator, decode_variants",
        [
            # tp_ep inside one attention TP group: a single replica, so every decode-only step
            # is bounded and needs no agreement.
            (2, 1, 2, 64, False, (True,)),
            # Two replicas: bounded only in agreed steps, which need the communicator.
            (1, 1, 2, 64, False, (False,)),
            (1, 1, 2, 64, True, (False, True)),
            # The replicas' decode rows do not fit: never bounded.
            (1, 2, 2, 64, True, (False,)),
        ],
    )
    def test_cuda_graphs_are_captured_for_every_bound_steps_can_use(
        self, tp, etp, ep, max_requests, communicator, decode_variants
    ):
        """A decode-only CUDA graph is captured once per bound its steps can use, and a mixed
        graph only unbounded."""
        ctx = self._context(
            tp=tp,
            etp=etp,
            ep=ep,
            max_requests=max_requests,
            communicator=_ScriptedCommunicator(False) if communicator else None,
        )
        graphs = ctx.cuda_graph_batch_dimensions_list
        assert any(dims.prefill_req_count > 0 for dims in graphs)
        for dims in graphs:
            expected = decode_variants if dims.prefill_req_count == 0 else (False,)
            assert ctx.bounded_moe_row_graph_variants(dims) == expected, dims

    @pytest.mark.internal
    def test_graph_capture_builds_each_variant_without_a_vote(self):
        """Graph capture runs the same batch on every rank: it captures the requested variant
        under its own key and consumes no agreement."""
        communicator = _ScriptedCommunicator(peers_not_decode_only=True)
        ctx = self._context(communicator=communicator)
        decode = self._decode_graph(ctx)
        for bounded in (False, True, None):
            ctx.initialize_attention_state(
                construct_graph_dimensions=decode, construct_graph_bounded_moe_rows=bounded
            )
            expected = True if bounded is None else bounded
            assert ctx.use_bounded_mxfp8_rows is expected
            assert ctx.using_cuda_graph_this_step()
            assert ctx.cuda_graph_key == (decode, expected)
            ctx.reset()
        assert communicator.sent == []

        # Without the communicator no step ever replays a bounded graph.
        ctx.set_ep_zmq_communicator(None)
        with pytest.raises(AssertionError, match="is ever replayed"):
            ctx.initialize_attention_state(
                construct_graph_dimensions=decode, construct_graph_bounded_moe_rows=True
            )
        ctx.reset()
