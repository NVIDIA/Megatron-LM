# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inference contexts take their process groups from an explicit collection.

The collections here are built for a grid whose tensor-parallel size differs from the global one
(as for RL inference shards), and every global group, rank and size accessor raises while the
code under test runs, so a read of the global grid fails instead of passing by coincidence.
"""

import contextlib
import math
import sys
import warnings
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext, StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.shards import build_inference_pg_collection
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils

# Same accessor selection as tools/check_process_group_usage.py.
_ACCESSOR_SUFFIXES = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size")
_NOT_ACCESSORS = {
    "get_nccl_options",
    "get_all_ranks",
    "get_global_memory_buffer",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
}


class GlobalProcessGroupRead(Exception):
    """Raised by a global process-group accessor inside `forbid_global_process_groups`.

    Not an AssertionError, so code that treats AssertionError as "not initialized" cannot hide it.
    """


@contextlib.contextmanager
def forbid_global_process_groups():
    """Make every read of the global parallel grid raise inside the block.

    Patches the `parallel_state` group, rank and size accessors, the copies that Megatron modules
    imported by name (under any alias), and `ProcessGroupCollection.use_mpu_process_groups`.
    """

    def failing(name):
        def fail(*args, **kwargs):
            raise GlobalProcessGroupRead(f"read of the global parallel grid: {name}")

        return fail

    # Keyed by id() because module attributes need not be hashable; the values keep the
    # functions alive, so an id cannot be reused while the block runs.
    accessors = {
        id(function): (name, function)
        for name, function in vars(parallel_state).items()
        if name.startswith("get_")
        and name.endswith(_ACCESSOR_SUFFIXES)
        and name not in _NOT_ACCESSORS
    }
    with pytest.MonkeyPatch.context() as patch:
        for module in list(sys.modules.values()):
            if not getattr(module, "__name__", "").startswith("megatron."):
                continue
            for attribute, value in list(getattr(module, "__dict__", {}).items()):
                if id(value) in accessors:
                    patch.setattr(module, attribute, failing(accessors[id(value)][0]))
        patch.setattr(
            ProcessGroupCollection,
            "use_mpu_process_groups",
            classmethod(failing("ProcessGroupCollection.use_mpu_process_groups")),
        )
        yield


def _round_up(value, multiple):
    return math.ceil(value / multiple) * multiple


def _require_world_size_multiple_of(*sizes):
    for size in sizes:
        if Utils.world_size < size or Utils.world_size % size != 0:
            pytest.skip(f"needs a world size that is a multiple of {size}")


def _build_collection(tp_size, ep_size=1):
    """A collection for an inference grid with its own TP size, built without global reads."""
    return build_inference_pg_collection(
        Utils.world_size, tp_size=tp_size, pp_size=1, cp_size=1, ep_size=ep_size, expt_tp_size=1
    )


@pytest.fixture
def unit_rounders(monkeypatch):
    """Pad to the next multiple of the TP size only.

    The default rounders are multiples of every power-of-two TP size, which would hide padding
    to the wrong TP size.
    """
    monkeypatch.setattr(DynamicInferenceContext, "TOKEN_ROUNDER", 1)
    monkeypatch.setattr(DynamicInferenceContext, "REQUEST_ROUNDER", 1)


class TestDynamicContextTensorParallelSize:
    """`DynamicInferenceContext` pads to the TP size of its collection."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("global_tp, context_tp", [(1, 2), (1, 4), (2, 1)])
    @pytest.mark.parametrize("num_speculative_tokens", [0, 2])
    def test_step_padding_uses_collection_tp_size(
        self, unit_rounders, global_tp, context_tp, num_speculative_tokens
    ):
        _require_world_size_multiple_of(global_tp, context_tp)
        Utils.initialize_model_parallel(tensor_model_parallel_size=global_tp)
        pg_collection = _build_collection(context_tp)
        model_config = TransformerConfig(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=8,
            num_attention_heads=4,
            tensor_model_parallel_size=context_tp,
        )
        inference_config = InferenceConfig(
            max_sequence_length=64,
            buffer_size_gb=0.01,
            block_size_tokens=16,
            max_tokens=64,
            max_requests=8,
            num_speculative_tokens=num_speculative_tokens,
            unified_memory_level=0,
            pg_collection=pg_collection,
        )
        prompt_lengths = (5, 4, 4)
        num_requests = len(prompt_lengths)

        with forbid_global_process_groups():
            context = DynamicInferenceContext(model_config, inference_config)

            # Prefill step.
            for request_id, prompt_length in enumerate(prompt_lengths):
                context.add_request(
                    DynamicInferenceRequest(
                        request_id=request_id,
                        prompt_tokens=torch.arange(prompt_length, device="cuda"),
                        sampling_params=SamplingParams(num_tokens_to_generate=8),
                    )
                )
            context.initialize_attention_state()
            assert context.padded_active_token_count == _round_up(sum(prompt_lengths), context_tp)
            assert context.padded_batch_dimensions.req_count == _round_up(num_requests, context_tp)

            # Decode step for the same requests.
            context.update_requests(
                active_requests_mask=torch.ones(num_requests, dtype=torch.int32, device="cuda"),
                new_tokens=torch.arange(num_requests, device="cuda"),
                new_speculative_tokens=(
                    torch.zeros(
                        num_speculative_tokens, num_requests, dtype=torch.long, device="cuda"
                    )
                    if num_speculative_tokens > 0
                    else None
                ),
            )
            assert context.num_decode_requests == num_requests
            context.initialize_attention_state()

        padded_decode_requests = _round_up(num_requests, context_tp)
        assert context.padded_batch_dimensions.decode_req_count == padded_decode_requests
        assert context.padded_active_token_count == padded_decode_requests * (
            num_speculative_tokens + 1
        )

    @pytest.mark.parametrize("global_tp, context_tp", [(1, 2), (2, 1)])
    def test_moe_buffers_use_collection_tp_size(self, unit_rounders, global_tp, context_tp):
        ep_size = 2
        _require_world_size_multiple_of(global_tp, context_tp, ep_size)
        Utils.initialize_model_parallel(tensor_model_parallel_size=global_tp)
        pg_collection = _build_collection(context_tp, ep_size=ep_size)
        model_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=2,
            hidden_size=64,
            kv_channels=16,
            num_attention_heads=4,
            tensor_model_parallel_size=context_tp,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=1,
            num_moe_experts=4,
            moe_ffn_hidden_size=64,
            moe_router_dtype="fp32",
            sequence_parallel=context_tp > 1,
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            add_bias_linear=False,
        )
        max_tokens = 63
        inference_config = InferenceConfig(
            max_sequence_length=64,
            buffer_size_gb=0.01,
            block_size_tokens=16,
            max_tokens=max_tokens,
            max_requests=8,
            unified_memory_level=0,
            pg_collection=pg_collection,
        )

        with (
            mock.patch.object(NVLSAllGatherVDispatcher, "allocate_buffers") as nvls_buffers,
            mock.patch.object(VllmFusedMoeBuffers, "allocate_buffers") as vllm_buffers,
            forbid_global_process_groups(),
        ):
            DynamicInferenceContext(model_config, inference_config)

        # Each TP rank holds its share of a step's padded tokens.
        per_rank_token_count = _round_up(max_tokens, context_tp) // context_tp
        nvls_buffers.assert_called_once()
        assert nvls_buffers.call_args.kwargs["per_rank_worst_case_token_count"] == (
            per_rank_token_count
        )
        assert nvls_buffers.call_args.kwargs["ep_group"] is pg_collection.ep
        vllm_buffers.assert_called_once()
        assert vllm_buffers.call_args.kwargs["max_tokens"] == max(
            max_tokens, per_rank_token_count * ep_size
        )


class TestStaticContextProcessGroups:
    """`StaticInferenceContext` carries a collection to everything built on it."""

    def teardown_method(self, method):
        InferenceMode.unset_active()
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    def test_collection_is_stored_in_config(self):
        assert StaticInferenceContext(4, 64).config.pg_collection is None
        pg_collection = ProcessGroupCollection()
        context = StaticInferenceContext(4, 64, pg_collection=pg_collection)
        assert context.config.pg_collection is pg_collection

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_static_engine_uses_collection_of_static_context(self):
        context_tp = 2
        _require_world_size_multiple_of(context_tp)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        pg_collection = _build_collection(context_tp)
        model_parallel_cuda_manual_seed(123)
        vocab_size = 100
        model = GPTModel(
            config=TransformerConfig(
                num_layers=2,
                hidden_size=32,
                num_attention_heads=4,
                use_cpu_initialization=True,
                tensor_model_parallel_size=context_tp,
                params_dtype=torch.bfloat16,
            ),
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=vocab_size,
            max_sequence_length=64,
            pg_collection=pg_collection,
        ).cuda()
        model.to(torch.bfloat16)
        tokenizer = mock.Mock(vocab_size=vocab_size, eod=vocab_size - 1)

        with warnings.catch_warnings(record=True) as caught, forbid_global_process_groups():
            warnings.simplefilter("always")
            context = StaticInferenceContext(4, 64, pg_collection=pg_collection)
            controller = TextGenerationController(
                inference_wrapped_model=GPTInferenceWrapper(model, context), tokenizer=tokenizer
            )
            engine = StaticInferenceEngine(controller, max_batch_size=4, buffer_size_gb=0.1)

        # The engine catches construction errors and falls back to the legacy engine.
        fallbacks = [str(w.message) for w in caught if "legacy static engine" in str(w.message)]
        assert not fallbacks, fallbacks
        assert not engine.legacy
        assert controller.inference_wrapped_model.tp_group is pg_collection.tp
        assert controller.pp_group is pg_collection.pp
        dynamic_context = engine.dynamic_engine.context
        assert dynamic_context.config.pg_collection is pg_collection
        assert dynamic_context.tp_size == context_tp
        assert dynamic_context.expert_model_parallel_group is pg_collection.ep
        assert engine.dynamic_engine.pg_collection is pg_collection
