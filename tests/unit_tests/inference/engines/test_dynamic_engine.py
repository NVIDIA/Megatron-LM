# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import gc
import math
import os
import random
import types
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple
from unittest import mock

import msgpack
import pytest
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.inference.config import (
    AsyncScheduleMode,
    CudaGraphSizingDistribution,
    InferenceConfig,
    KVCacheManagementMode,
    MambaInferenceStateConfig,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import (
    ActiveRequestCountOverflowError,
    BlockOverflowError,
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    DynamicVLMInferenceRequest,
    Status,
    compute_block_hashes_batched,
    compute_media_cache_key,
)
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.inference.engines.ssm_test_helpers import (
    hybrid_mixer_kwargs,
    hybrid_stack_spec_for,
    skip_if_sequence_packing_not_available,
)
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

try:
    from torch_memory_saver import torch_memory_saver  # noqa: F401

    HAVE_TORCH_MEMORY_SAVER = True
except ImportError:
    HAVE_TORCH_MEMORY_SAVER = False


class _ImageOnlyCapabilityWrapper:
    supports_text = True
    supports_image = True
    supports_video = False
    supports_audio = False
    validate_input_modalities = GPTInferenceWrapper.validate_input_modalities


@pytest.mark.parametrize(
    ("kwargs", "modality"),
    [
        ({"imgs": torch.ones(1)}, "image"),
        ({"imgs": torch.ones(1), "num_frames": torch.ones(1, dtype=torch.int64)}, "video"),
    ],
)
def test_add_request_rejects_unsupported_media_before_tokenization(kwargs, modality):
    engine = object.__new__(DynamicInferenceEngine)
    wrapper = _ImageOnlyCapabilityWrapper()
    if modality == "image":
        wrapper.supports_image = False
    engine.controller = mock.Mock(inference_wrapped_model=wrapper)

    with pytest.raises(
        ValueError, match=rf"_ImageOnlyCapabilityWrapper does not support {modality} inputs"
    ):
        engine.add_request(request_id=1, prompt=[1], **kwargs)

    engine.controller.tokenize_prompt.assert_not_called()


def test_validate_input_modalities_rejects_unsupported_audio():
    wrapper = _ImageOnlyCapabilityWrapper()

    with pytest.raises(ValueError, match="does not support audio inputs"):
        wrapper.validate_input_modalities("audio")


def _build_mock_vlm_engine(image_embeddings):
    engine = object.__new__(DynamicInferenceEngine)
    wrapper = mock.Mock()
    wrapper._forward_vision_encoder.return_value = image_embeddings
    wrapper.resolve_media_token_id.return_value = 99
    controller = mock.Mock()
    controller.pp_group = None
    controller.inference_wrapped_model = wrapper
    engine.controller = controller
    engine.context = mock.Mock(block_size_tokens=256, enable_prefix_caching=False)
    engine._get_cached_vision_embedding = mock.Mock(return_value=None)
    engine._cache_vision_embedding = mock.Mock()
    return engine, wrapper


def _call_build_vlm_request(engine, tokens, *, media_tokens_preexpanded):
    with mock.patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")):
        return engine._build_vlm_request(
            request_id=1,
            prompt_str=None,
            tokens=tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=0),
            imgs=torch.ones(1, 2, 4),
            num_tiles=None,
            num_img_embeddings_per_tile=0,
            imgs_sizes=torch.tensor([[2, 2]]),
            media_tokens_preexpanded=media_tokens_preexpanded,
        )


@pytest.mark.parametrize(
    ("media_kwargs", "error"),
    [
        (
            {"num_frames": torch.tensor([2])},
            "Video input requires imgs, imgs_sizes, and num_frames",
        ),
        (
            {"imgs_sizes": torch.tensor([[2, 2]])},
            "Dynamic-resolution image input requires imgs and imgs_sizes",
        ),
        (
            {"imgs": torch.ones(1, 2, 4), "num_tiles": torch.tensor([1])},
            "Static-tiling image input requires imgs, num_tiles",
        ),
    ],
)
def test_build_vlm_request_rejects_incomplete_media(media_kwargs, error):
    engine, _ = _build_mock_vlm_engine(torch.ones(2, 4))
    kwargs = {
        "imgs": None,
        "num_tiles": None,
        "num_img_embeddings_per_tile": 0,
        "imgs_sizes": None,
        "num_frames": None,
    }
    kwargs.update(media_kwargs)

    with pytest.raises(ValueError, match=error):
        engine._build_vlm_request(
            request_id=1,
            prompt_str=None,
            tokens=torch.tensor([10, 20], dtype=torch.int64),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=0),
            **kwargs,
        )


def test_build_vlm_request_preserves_preexpanded_tokens_and_derives_mask():
    engine, wrapper = _build_mock_vlm_engine(torch.ones(2, 4))
    tokens = torch.tensor([10, 99, 99, 20], dtype=torch.int64)
    wrapper.build_preexpanded_media_token_mask.return_value = torch.tensor(
        [-1, 0, 1, -1], dtype=torch.int64
    )

    request = _call_build_vlm_request(engine, tokens, media_tokens_preexpanded=True)

    assert torch.equal(request.prompt_tokens, tokens)
    assert request.compact_prompt_tokens is None
    assert request.image_token_mask.tolist() == [-1, 0, 1, -1]
    wrapper.expand_image_tokens.assert_not_called()
    wrapper.resolve_media_token_id.assert_not_called()
    wrapper.build_preexpanded_media_token_mask.assert_called_once_with(tokens, "image")


def test_build_vlm_request_rejects_preexpanded_embedding_count_mismatch():
    engine, wrapper = _build_mock_vlm_engine(torch.ones(1, 4))
    wrapper.build_preexpanded_media_token_mask.return_value = torch.tensor(
        [-1, 0, 1, -1], dtype=torch.int64
    )

    with pytest.raises(ValueError, match="2 media-token position.*1 embedding"):
        _call_build_vlm_request(
            engine, torch.tensor([10, 99, 99, 20], dtype=torch.int64), media_tokens_preexpanded=True
        )


def test_build_vlm_request_validates_cached_embedding_count_once():
    engine, wrapper = _build_mock_vlm_engine(torch.ones(2, 4))
    engine._get_cached_vision_embedding.return_value = torch.ones(1, 4)
    wrapper.build_preexpanded_media_token_mask.return_value = torch.tensor(
        [-1, 0, 1, -1], dtype=torch.int64
    )

    with pytest.raises(ValueError, match="2 media-token position.*1 embedding"):
        _call_build_vlm_request(
            engine, torch.tensor([10, 99, 99, 20], dtype=torch.int64), media_tokens_preexpanded=True
        )

    wrapper._forward_vision_encoder.assert_not_called()


def test_build_vlm_request_keeps_compact_expansion_path():
    engine, wrapper = _build_mock_vlm_engine(torch.ones(2, 4))
    compact_tokens = torch.tensor([10, 42, 20], dtype=torch.int64)
    wrapper.expand_image_tokens.return_value = ([[10, -1, -1, 20]], [[None, 0, 1, None]])

    request = _call_build_vlm_request(engine, compact_tokens, media_tokens_preexpanded=False)

    wrapper.expand_image_tokens.assert_called_once()
    encoder_args, encoder_kwargs = wrapper._forward_vision_encoder.call_args
    assert torch.equal(encoder_args[0], torch.ones(1, 2, 4))
    assert encoder_kwargs["num_image_tiles"] is None
    assert torch.equal(encoder_kwargs["imgs_sizes"], torch.tensor([[2, 2]]))
    assert request.prompt_tokens.tolist() == [10, 99, 99, 20]
    assert torch.equal(request.compact_prompt_tokens, compact_tokens)
    assert request.image_token_mask.tolist() == [-1, 0, 1, -1]


def test_build_vlm_request_preserves_adjacent_compact_media_placeholders():
    engine, wrapper = _build_mock_vlm_engine(torch.ones(2, 4))
    compact_tokens = torch.tensor([10, 42, 42, 20], dtype=torch.int64)
    # The expanded sequence is structurally ambiguous: it could also represent
    # one placeholder expanded to two positions. The saved compact prompt is
    # therefore required for lossless multi-turn reconstruction.
    wrapper.expand_image_tokens.return_value = ([[10, -1, -1, 20]], [[None, 0, 1, None]])

    request = _call_build_vlm_request(engine, compact_tokens, media_tokens_preexpanded=False)

    assert request.prompt_tokens.tolist() == [10, 99, 99, 20]
    assert torch.equal(request.compact_prompt_tokens, compact_tokens)


def test_build_vlm_request_enables_media_salted_prefix_caching():
    engine, wrapper = _build_mock_vlm_engine(torch.ones(2, 4))
    engine.context.enable_prefix_caching = True
    engine.context.block_size_tokens = 2
    wrapper.expand_image_tokens.return_value = ([[10, -1, -1, 20]], [[None, 0, 1, None]])

    request = _call_build_vlm_request(
        engine, torch.tensor([10, 42, 20], dtype=torch.int64), media_tokens_preexpanded=False
    )

    media_cache_key = compute_media_cache_key(
        "image", {"imgs": torch.ones(1, 2, 4), "imgs_sizes": torch.tensor([[2, 2]])}
    )
    assert request.enable_prefix_caching
    assert request.block_hash_salt == media_cache_key
    assert request.precomputed_block_hashes == compute_block_hashes_batched(
        request.prompt_tokens, block_size=2, cache_salt=media_cache_key
    )


def teardown_module(module):
    # inference_optimized MoE parametrizations use the default 'vllm' grouped-GEMM
    # backend, which allocates class-level persistent intermediate buffers at
    # context init. Release them so no GPU memory or state leaks across modules.
    VllmFusedMoeBuffers._delete_buffers()


def skip_if_mamba_sequence_packing_not_available(model_provider: str, ssm_mixer: str = "mamba"):
    if model_provider != "hybrid":
        return
    if ssm_mixer == "gdn":
        # GDN packing rides on FLA, which every GDN test already gates on
        # separately via HAVE_FLA.
        return
    skip_if_sequence_packing_not_available(ssm_mixer)


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value  # For backwards compatibility
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


def mock_forward(input_ids, position_ids, attention_mask, *args, **kwargs):
    """Mock forward function to avoid numerics issues with random inputs."""
    return torch.randn(
        input_ids.size(0),
        input_ids.size(1),
        kwargs["vocab_size"],
        device=input_ids.device,
        dtype=torch.bfloat16,
    )


@dataclass
class DynamicEngineTestConfig:
    """Test configuration args."""

    random_seed = 123
    vocab_size = 100

    set_rounder(4)
    num_requests: int = 2 * DynamicInferenceContext.round_up_requests(1, 1)
    min_prompt_length: int = 4
    max_prompt_length: int = 16
    num_tokens_to_generate: Optional[int] = 4
    num_tokens_total: Optional[int] = None
    max_sequence_length: Optional[int] = None

    num_gap_steps: int = 2

    context_buffer_size_gb: float = 0.1  # enough room for all tokens.
    context_paused_buffer_size_gb: float | None = None
    context_block_size_tokens: int = 256
    context_max_requests: Optional[int] = None
    context_max_tokens: Optional[int] = None
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False

    use_fixed_output_lengths: bool = False
    num_cuda_graphs: int = None
    use_cuda_graphs_for_non_decode_steps: bool = True
    cuda_graph_all_prefills: bool = False
    # Defaults to the production default (HYBRID: exponential prefill/mixed graphs,
    # linear decode-only graphs). Tests that assert on exact token counts can pin a
    # single distribution here.
    cuda_graph_sizing_distribution: CudaGraphSizingDistribution = CudaGraphSizingDistribution.HYBRID
    cuda_graph_mixed_prefill_count: Optional[int] = 16
    cuda_graph_max_tokens: int = 512
    fp8: bool = False
    hidden_size: int = 32
    model_provider: str = "gpt"
    # Which linear-attention mixer a hybrid stack uses ("mamba", "gdp", or
    # "gdn"). Ignored unless model_provider == "hybrid": all three build a
    # HybridModel, and only the layer pattern and stack spec (and so the
    # mixer) differ.
    ssm_mixer: str = "mamba"
    return_log_probs: bool = False
    logprobs_mode: str = "raw_logprobs"
    materialize_only_last_token_logits: bool = True
    skip_prompt_log_probs: bool = False
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
        PrefixCachingEvictionPolicy.REF_ZERO
    )
    cuda_graph_modules: List[CudaGraphModule] = field(default_factory=list)
    inference_cuda_graph_scope: InferenceCudaGraphScope = InferenceCudaGraphScope.block
    cuda_graph_impl: Optional[str] = None
    force_build_cuda_graphs: bool = False
    transformer_impl: str = "local"
    inference_moe_token_dispatcher_type: str = "nccl"
    # If False, do not build cuda graphs in the tests, even if
    # num_cuda_graphs is set.
    # For tests concerning cuda-graph warmups, we set this to False
    # to avoid the overhead of building the graphs, which is not
    # relevant to the test. The tests only check if the required
    # context attributes are set correctly.
    suspend_resume_interval: Optional[int] = None
    kv_cache_management_mode: str = "persist"
    static_kv_memory_pointers: bool = True
    track_generated_token_events: bool = False
    num_speculative_tokens: int = 0
    position_embedding_type: str = "learned_absolute"
    use_flashinfer_fused_rope: Optional[bool] = None
    sampling_backend: str = 'torch'
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    offset_sampling_seed_by_dp_rank: bool = True
    async_sched_mode: AsyncScheduleMode = AsyncScheduleMode.ASYNC
    # Sliding-window attention config. When `window_size` is None, SWA is
    # disabled and all layers do full causal attention. When set to a
    # `(left, right)` tuple, layers selected by `window_attn_skip_freq` use a
    # local window of `left` past tokens and `right` future tokens.
    window_size: Optional[Tuple[int, int]] = None
    window_attn_skip_freq: Optional[int] = None
    # Sink (off-by-one / learnable) softmax — exercises the post-hoc LSE
    # rescale path inside Attention.flash_decode_and_prefill. Default keeps
    # behavior unchanged for existing tests.
    softmax_type: str = "vanilla"

    def __post_init__(self):

        # Compute max_sequence_length.
        if self.max_sequence_length is None:
            assert self.num_tokens_to_generate is None or self.num_tokens_total is None
            if self.num_tokens_to_generate is not None:
                self.max_sequence_length = (
                    self.max_prompt_length
                    + self.num_tokens_to_generate
                    + self.num_speculative_tokens
                )
            else:
                assert self.num_tokens_total is not None
                self.max_sequence_length = self.num_tokens_total + self.num_speculative_tokens

        # Default paused buffer size.
        if self.context_paused_buffer_size_gb is None:
            self.context_paused_buffer_size_gb = 0.2 * self.context_buffer_size_gb


@dataclass
class DynamicEngineTestEnv:
    """Test environment, including requests and engine."""

    config: DynamicEngineTestConfig
    requests: List[DynamicInferenceRequest]
    engine: DynamicInferenceEngine
    mem_usage: dict = field(
        default_factory=lambda: {"start": None, "end": None, "suspend_resume": {}}
    )


class DynamicInferenceEngineTestBase:

    @staticmethod
    def _assert_inference_cuda_graphs_disabled(env) -> None:
        model = env.engine.controller.inference_wrapped_model.model
        assert env.engine.cuda_graph_impl == "full_iteration"
        assert env.engine.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert env.engine.capture_stats is None
        assert not hasattr(model, 'cudagraph_manager')
        assert not hasattr(model.decoder, 'cudagraph_manager')
        for layer in model.decoder.layers:
            assert not hasattr(layer, 'cudagraph_manager')

    @staticmethod
    def _cuda_graph_batch_dimension_signature(env) -> List[Tuple[int, int, int]]:
        return [
            (dim.token_count, dim.prefill_req_count, dim.decode_req_count)
            for dim in env.engine.context.cuda_graph_batch_dimensions_list
        ]

    @classmethod
    def _build_requests(cls, test_config: DynamicEngineTestConfig) -> List[DynamicInferenceRequest]:

        requests = []
        for request_id in range(test_config.num_requests):

            # Prompt length.
            if test_config.min_prompt_length == test_config.max_prompt_length:
                prompt_length = test_config.min_prompt_length
            else:
                prompt_length = random.randint(
                    test_config.min_prompt_length, test_config.max_prompt_length
                )

            # Num tokens to generate.
            num_tokens_to_generate = test_config.num_tokens_to_generate
            num_tokens_total = test_config.num_tokens_total

            if test_config.use_fixed_output_lengths:
                if num_tokens_to_generate is not None:
                    num_tokens_to_generate = random.randint(
                        1, test_config.max_sequence_length - prompt_length
                    )
                else:
                    num_tokens_total = random.randint(
                        prompt_length + 1, test_config.max_sequence_length
                    )

            # Sampling params.
            sampling_params = SamplingParams(
                num_tokens_to_generate=num_tokens_to_generate,
                termination_id=(
                    -1 if test_config.use_fixed_output_lengths else test_config.vocab_size - 1
                ),
                return_log_probs=test_config.return_log_probs,
                skip_prompt_log_probs=test_config.skip_prompt_log_probs,
                temperature=test_config.temperature,
                top_k=test_config.top_k,
                top_p=test_config.top_p,
            )
            if not hasattr(sampling_params, "num_tokens_total"):
                # Remove this if statement branch in megatron-core 0.16
                sampling_params.add_attributes({"num_tokens_total": num_tokens_total})
            else:
                sampling_params.num_tokens_total = num_tokens_total

            # Request.
            prompt_tokens = torch.randint(
                0,
                test_config.vocab_size - 1,
                (prompt_length,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            request = DynamicInferenceRequest(
                request_id=request_id, prompt_tokens=prompt_tokens, sampling_params=sampling_params
            )
            requests.append(request)

        return requests

    @classmethod
    def _build_inference_context(
        cls,
        test_config: DynamicEngineTestConfig,
        transformer_config: TransformerConfig,
        requests: List[DynamicInferenceRequest],
        mamba_inference_state_config: Optional[MambaInferenceStateConfig] = None,
    ):
        """The inference context manages the KV cache and other inference state."""

        # Inference context.
        context = DynamicInferenceContext(
            model_config=transformer_config,
            inference_config=InferenceConfig(
                max_sequence_length=test_config.max_sequence_length,
                num_cuda_graphs=test_config.num_cuda_graphs,
                cuda_graph_mixed_prefill_count=test_config.cuda_graph_mixed_prefill_count,
                cuda_graph_sizing_distribution=test_config.cuda_graph_sizing_distribution,
                cuda_graph_max_tokens=test_config.cuda_graph_max_tokens,
                use_cuda_graphs_for_non_decode_steps=(
                    test_config.use_cuda_graphs_for_non_decode_steps
                ),
                cuda_graph_all_prefills=test_config.cuda_graph_all_prefills,
                buffer_size_gb=test_config.context_buffer_size_gb,
                paused_buffer_size_gb=test_config.context_paused_buffer_size_gb,
                block_size_tokens=test_config.context_block_size_tokens,
                max_requests=test_config.context_max_requests,
                max_tokens=test_config.context_max_tokens,
                mamba_inference_state_config=mamba_inference_state_config,
                materialize_only_last_token_logits=test_config.materialize_only_last_token_logits,
                kv_cache_management_mode=KVCacheManagementMode(
                    test_config.kv_cache_management_mode
                ),
                static_kv_memory_pointers=test_config.static_kv_memory_pointers,
                enable_chunked_prefill=test_config.enable_chunked_prefill,
                enable_prefix_caching=test_config.enable_prefix_caching,
                prefix_caching_eviction_policy=test_config.prefix_caching_eviction_policy,
                use_flashinfer_fused_rope=test_config.use_flashinfer_fused_rope,
                # this is for compatibility with the LTS environment
                unified_memory_level=0,  # unit tests currently broken with UVM
                track_generated_token_events=test_config.track_generated_token_events,
                num_speculative_tokens=test_config.num_speculative_tokens,
                sampling_backend=test_config.sampling_backend,
                offset_sampling_seed_by_dp_rank=test_config.offset_sampling_seed_by_dp_rank,
                async_sched_mode=test_config.async_sched_mode,
                logprobs_mode=test_config.logprobs_mode,
            ),
        )

        return context

    @classmethod
    @torch.inference_mode()
    def _build_test_env(cls, test_config):
        clear_nvte_env_vars()
        set_rounder(4)

        # Random state.
        random.seed(test_config.random_seed)
        torch.manual_seed(test_config.random_seed)
        model_parallel_cuda_manual_seed(
            seed=test_config.random_seed,
            inference_rng_tracker=True,
            use_cudagraphable_rng=False,
            force_reset_rng=True,
        )

        # Requests.
        requests = cls._build_requests(test_config)
        effective_cuda_graph_impl = test_config.cuda_graph_impl
        if effective_cuda_graph_impl is None:
            effective_cuda_graph_impl = (
                "local"
                if test_config.num_cuda_graphs is not None and test_config.force_build_cuda_graphs
                else "none"
            )

        if test_config.model_provider == "gpt":
            # Transformer config.
            transformer_config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=4,
                mtp_num_layers=test_config.num_speculative_tokens,
                hidden_size=test_config.hidden_size,
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl=effective_cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=test_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=test_config.pipeline_model_parallel_size,
                expert_model_parallel_size=test_config.expert_model_parallel_size,
                num_moe_experts=(
                    None
                    if test_config.expert_model_parallel_size == 1
                    else test_config.expert_model_parallel_size
                ),
                sequence_parallel=test_config.sequence_parallel,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=test_config.expert_model_parallel_size == 1
                and not (test_config.transformer_impl == "inference_optimized"),
                fp8="hybrid" if test_config.fp8 else None,
                fp8_recipe="tensorwise" if test_config.fp8 else None,
                inference_sampling_seed=test_config.random_seed,
                cuda_graph_modules=test_config.cuda_graph_modules,
                inference_cuda_graph_scope=(
                    test_config.inference_cuda_graph_scope
                    if test_config.num_cuda_graphs is not None
                    and test_config.force_build_cuda_graphs
                    else InferenceCudaGraphScope.none
                ),
                transformer_impl=test_config.transformer_impl,
                activation_func=(
                    squared_relu
                    if test_config.transformer_impl == "inference_optimized"
                    and test_config.expert_model_parallel_size > 1
                    else torch.nn.functional.gelu
                ),
                moe_router_dtype=(
                    "fp32"
                    if test_config.transformer_impl == "inference_optimized"
                    and test_config.expert_model_parallel_size > 1
                    else None
                ),
                inference_moe_token_dispatcher_type=(
                    test_config.inference_moe_token_dispatcher_type
                ),
                normalization=(
                    "RMSNorm"
                    if test_config.transformer_impl == "inference_optimized"
                    else "LayerNorm"
                ),
                softmax_type=test_config.softmax_type,
                # inference optimized currently only supports RMS Norm
                window_size=test_config.window_size,
                window_attn_skip_freq=test_config.window_attn_skip_freq,
            )
            # Layer-spec factories do not receive TransformerConfig. Forward num_moe_experts
            # explicitly so MoE test cases build MoE layers instead of a dense MLP.
            if test_config.fp8 or test_config.transformer_impl == "transformer_engine":
                layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    num_experts=transformer_config.num_moe_experts
                )
            elif test_config.transformer_impl == "local":
                layer_spec = get_gpt_layer_local_spec(
                    num_experts=transformer_config.num_moe_experts
                )
            elif test_config.transformer_impl == "inference_optimized":
                layer_spec = get_gpt_layer_with_inference_spec(
                    num_experts=transformer_config.num_moe_experts
                )

            # MTP block spec (needed for speculative decoding).
            mtp_block_spec = None
            if test_config.num_speculative_tokens > 0:
                use_te = test_config.fp8 or test_config.transformer_impl == "transformer_engine"
                mtp_block_spec = get_gpt_mtp_block_spec(
                    config=transformer_config, spec=layer_spec, use_transformer_engine=use_te
                )

            # GPT model.
            model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=test_config.vocab_size,
                max_sequence_length=test_config.max_sequence_length,
                parallel_output=True,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
                mtp_block_spec=mtp_block_spec,
                position_embedding_type=test_config.position_embedding_type,
            ).cuda()
        elif test_config.model_provider == "hybrid":
            is_gdn = test_config.ssm_mixer == "gdn"
            pp_size = test_config.pipeline_model_parallel_size
            # Transformer config.
            transformer_config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=(
                    3 if pp_size == 1 else 6
                ),  # 1 Mamba layer, 1 attention layer, 1 MLP layer
                mtp_num_layers=test_config.num_speculative_tokens,
                hidden_size=256,  # The Mamba layer places several constraints on this
                **hybrid_mixer_kwargs(test_config.ssm_mixer),
                num_attention_heads=16,
                linear_conv_kernel_dim=4,
                linear_key_head_dim=32,
                linear_value_head_dim=64,
                linear_num_key_heads=4,
                linear_num_value_heads=8,
                use_cpu_initialization=True,
                cuda_graph_impl=effective_cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=test_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=pp_size,
                expert_model_parallel_size=test_config.expert_model_parallel_size,
                num_moe_experts=(
                    None
                    if test_config.expert_model_parallel_size == 1
                    else test_config.expert_model_parallel_size
                ),
                sequence_parallel=test_config.sequence_parallel,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=test_config.expert_model_parallel_size == 1
                and not (test_config.transformer_impl == "inference_optimized"),
                fp8="hybrid" if test_config.fp8 else None,
                fp8_recipe="tensorwise" if test_config.fp8 else None,
                inference_sampling_seed=test_config.random_seed,
                cuda_graph_modules=test_config.cuda_graph_modules,
                inference_cuda_graph_scope=(
                    test_config.inference_cuda_graph_scope
                    if test_config.num_cuda_graphs is not None
                    and test_config.force_build_cuda_graphs
                    else InferenceCudaGraphScope.none
                ),
                transformer_impl=test_config.transformer_impl,
                inference_moe_token_dispatcher_type=(
                    test_config.inference_moe_token_dispatcher_type
                ),
                normalization=(
                    "RMSNorm"
                    if is_gdn or test_config.transformer_impl == "inference_optimized"
                    else "LayerNorm"
                ),
                layernorm_zero_centered_gamma=is_gdn,
                activation_func=F.silu if is_gdn else F.gelu,
                is_hybrid_model=True,  # Needs to be set for correct out_proj init
            )

            # Hybrid model.
            # When speculative tokens are configured, append MTP depth sections
            # to the hybrid layer pattern so the model creates MTP blocks.
            mtp_suffix = "/M" * test_config.num_speculative_tokens
            recurrent_symbol = "G" if is_gdn else "M"
            if pp_size == 1:
                mamba_pattern = recurrent_symbol + "*-" + mtp_suffix
            else:
                mamba_pattern = recurrent_symbol + "*-|" + recurrent_symbol + "*-" + mtp_suffix
            model = HybridModel(
                config=transformer_config,
                hybrid_stack_spec=hybrid_stack_spec_for(test_config.ssm_mixer),
                vocab_size=test_config.vocab_size,
                max_sequence_length=test_config.max_sequence_length,
                parallel_output=True,
                hybrid_layer_pattern=mamba_pattern,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
        else:
            raise ValueError(f"Invalid model provider {test_config.model_provider}")

        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)

        model.eval()

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(model)

        # Inference context.
        inference_context = cls._build_inference_context(
            test_config=test_config,
            transformer_config=transformer_config,
            requests=requests,
            mamba_inference_state_config=mamba_inference_state_config,
        )

        # Inference model wrapper.
        inference_wrapped_model = GPTInferenceWrapper(model, inference_context)

        # Note: the following is taken from AbstractModelInferenceWrapper.prep_model_for_inference().
        inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        # Text generation controller.
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=types.SimpleNamespace(
                vocab_size=test_config.vocab_size, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )

        delete_cuda_graphs()

        # Inference engine.
        engine = DynamicInferenceEngine(text_generation_controller, inference_context)

        # Test env.
        env = DynamicEngineTestEnv(config=test_config, requests=requests, engine=engine)

        return env

    @classmethod
    @torch.inference_mode()
    def _run_step(cls, env):
        set_rounder(4)
        # Step inference engine (i.e., generate one token per request).
        # It's safe to use request 0's sampling params here because
        # the only thing that differs between requests is num_tokens_to_generate,
        # and engine.async_step() doesn't use this sampling param's
        # num_tokens_to_generate.
        result = env.engine.step_modern()

        # Suspend + resume.
        if (
            env.config.suspend_resume_interval is not None
            and env.engine.context.step_count % env.config.suspend_resume_interval == 0
        ):
            suspend_resume_mems = {}
            suspend_resume_mems["start"] = torch.cuda.memory_stats()
            env.engine.suspend()  # suspend.
            suspend_resume_mems["mid"] = torch.cuda.memory_stats()
            env.engine.resume()  # resume.
            suspend_resume_mems["end"] = torch.cuda.memory_stats()
            env.mem_usage["suspend_resume"][env.engine.context.step_count] = suspend_resume_mems

        # Nothing done?
        finished_request_records = result["finished_request_records"]
        if len(finished_request_records) == 0:
            return

        # Append output tokens.
        for finished_request_record in finished_request_records:
            finished_request = finished_request_record.merge()
            request = env.requests[finished_request.request_id]
            request.output = finished_request.generated_tokens
            request.status = finished_request.status

    @classmethod
    @torch.inference_mode()
    def _run_test(cls, **test_config_kwargs):
        # Test environment.
        test_config = DynamicEngineTestConfig(**test_config_kwargs)
        env = cls._build_test_env(test_config)

        # Add requests to engine.
        env.mem_usage["start"] = torch.cuda.memory_stats()
        for request in tqdm(env.requests, "add requests"):

            # Add request.
            env.engine._add_request(request)
            request.state = "pending"

            # Insert gap steps between adding requests.
            for _ in range(test_config.num_gap_steps):
                cls._run_step(env)

        # Step engine until finished.
        while True:
            # Run at least one step to collect failed requests.
            cls._run_step(env)
            if not env.engine.has_unfinished_requests():
                break

        # Validate all requests finished.
        for request in env.requests:
            assert request.status in (
                Status.COMPLETED,
                Status.FAILED,
            ), f"request.status == '{request.status}'."

            num_tokens_to_generate = request.sampling_params.num_tokens_to_generate
            num_tokens_total = request.sampling_params.num_tokens_total
            num_tokens_expected = (
                num_tokens_to_generate
                if num_tokens_total is None
                else num_tokens_total - len(request.prompt_tokens)
            )

            # Validate the output length only if suspend_resume_interval is None.
            # If it is not None, then the output length could be anything in the
            # range [1, num_tokens_to_generate].
            if test_config.suspend_resume_interval is None:
                assert (
                    (num_tokens_to_generate is None and num_tokens_total is None)
                    or len(request.generated_tokens) <= num_tokens_expected
                    or request.status == Status.FAILED
                ), (
                    f"Request {request.request_id} expected to generate {num_tokens_to_generate} "
                    f"tokens but generated {len(request.generated_tokens)}"
                )
        env.mem_usage["end"] = torch.cuda.memory_stats()

        return env


def _make_prefix_cached_request_for_checkpoint(request_id: int) -> DynamicInferenceRequest:
    """Build a request whose generated tokens complete one additional cache block."""
    return DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=6, termination_id=-1),
        generated_tokens=[5, 6],
        block_size_tokens=2,
        enable_prefix_caching=True,
    )


def _assert_prefix_cache_checkpoint(
    original: DynamicInferenceRequest, checkpointed: DynamicInferenceRequest
) -> None:
    """Verify a checkpoint retained config and rehashed its expanded prompt."""
    expected_prompt = torch.cat(
        (
            original.prompt_tokens,
            torch.tensor(
                original.generated_tokens,
                dtype=original.prompt_tokens.dtype,
                device=original.prompt_tokens.device,
            ),
        )
    )
    expected_hashes = compute_block_hashes_batched(
        expected_prompt, block_size=original.block_size_tokens
    )

    assert checkpointed.enable_prefix_caching is True
    assert checkpointed.block_size_tokens == original.block_size_tokens
    assert torch.equal(checkpointed.prompt_tokens, expected_prompt)
    assert torch.equal(checkpointed.remaining_prompt_tokens, expected_prompt)
    assert checkpointed.precomputed_block_hashes == expected_hashes
    assert len(checkpointed.precomputed_block_hashes) == len(original.precomputed_block_hashes) + 1


def test_post_process_eviction_requeues_prefix_cached_request_with_fresh_hashes():
    """Eviction must checkpoint and requeue a prefix-enabled request without losing its config."""
    request = _make_prefix_cached_request_for_checkpoint(request_id=17)
    record = DynamicInferenceRequestRecord.from_request(request)
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = types.SimpleNamespace(
        chunked_prefill_request_id=-1, kv_block_allocator=types.SimpleNamespace()
    )
    engine.requests = {request.request_id: types.SimpleNamespace(record=record)}
    engine.waiting_request_ids = deque()
    engine.finished_request_count = 0
    engine.evicted_request_count = 0
    engine.track_generated_token_events = False
    engine.num_speculative_tokens = 0
    engine.stop_word_being_finished_ids = set()

    active_request_ids, finished_records = engine.post_process_requests(
        request_ids=torch.empty(0, dtype=torch.int64),
        finished_request_ids=torch.empty(0, dtype=torch.int64),
        evict_request_ids=torch.tensor([request.request_id], dtype=torch.int64),
        step_time=0.0,
        sample=torch.empty(0, dtype=torch.int64),
        accepted_tokens=None,
        log_probs=[],
        consumed_chunked_prefill_request_id=-1,
    )

    assert active_request_ids == []
    assert finished_records == []
    assert list(engine.waiting_request_ids) == [request.request_id]
    assert len(record.requests) == 2
    _assert_prefix_cache_checkpoint(request, engine.get_request(request.request_id))


def test_recompute_suspend_resume_readds_prefix_cached_request_with_fresh_hashes():
    """RECOMPUTE suspend/resume must re-add the prefix-enabled checkpoint tail."""
    request = _make_prefix_cached_request_for_checkpoint(request_id=23)
    record = DynamicInferenceRequestRecord.from_request(request)
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.context = types.SimpleNamespace(
        chunked_prefill_request_id=-1,
        kv_cache_management_mode=KVCacheManagementMode.RECOMPUTE,
        static_kv_memory_pointers=True,
        deallocate_inference_state_buffers=mock.Mock(),
        reinitialize_inference_state_buffers=mock.Mock(),
    )
    engine.requests = {request.request_id: types.SimpleNamespace(record=record)}
    engine.waiting_request_ids = deque()
    engine.controller = types.SimpleNamespace(_async_sched_logits=mock.Mock())
    engine.state = EngineState.RUNNING
    engine.unified_memory_level = 0
    engine.use_coordinator = False
    engine._vision_embedding_cache = {}
    engine._vision_embedding_cache_bytes = 0
    engine._add_request = mock.Mock()
    engine._notify_cond_for_new_request = mock.Mock(return_value=None)
    engine._loop = types.SimpleNamespace(call_soon_threadsafe=mock.Mock())

    with (
        mock.patch.object(
            DynamicInferenceEngine,
            "suspend_resume_ctx",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ),
        mock.patch.object(InferenceMode, "unset_active"),
        mock.patch.object(InferenceMode, "set_active"),
        mock.patch.object(torch.cuda, "synchronize"),
    ):
        engine.suspend()
        checkpointed = engine.get_request(request.request_id)
        _assert_prefix_cache_checkpoint(request, checkpointed)

        engine.resume()

    assert engine.context.deallocate_inference_state_buffers.call_count == 1
    assert engine.context.reinitialize_inference_state_buffers.call_count == 1
    engine.controller._async_sched_logits.clear.assert_called_once_with()
    assert engine.state == EngineState.RUNNING
    assert engine._add_request.call_count == 1
    assert engine._add_request.call_args.args[0] is checkpointed


def test_vision_state_invalidation_marks_request_local_embeddings_stale():
    request = DynamicVLMInferenceRequest(
        request_id=31,
        prompt_tokens=torch.tensor([99, 99, 5]),
        compact_prompt_tokens=torch.tensor([99, 5]),
        sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
        num_img_embeddings_per_tile=0,
        imgs=torch.ones(1),
        num_tiles=torch.tensor([1]),
        imgs_sizes=torch.tensor([[1, 1]]),
        decoder_seq_length=0,
        image_embeddings=torch.ones(2, 1, 4),
        image_token_mask=torch.tensor([0, 1, -1]),
    )
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.allow_stale_multimodal_embeddings = False
    engine._vision_embedding_cache = {"media": request.image_embeddings}
    engine._vision_embedding_cache_bytes = request.image_embeddings.numel() * 4
    engine.requests = {
        request.request_id: types.SimpleNamespace(
            record=DynamicInferenceRequestRecord.from_request(request)
        )
    }

    engine._invalidate_vision_state()

    assert not engine._vision_embedding_cache
    assert engine._vision_embedding_cache_bytes == 0
    assert request.image_embeddings is None
    assert request.image_token_mask is None


def test_vision_state_invalidation_can_explicitly_retain_stale_embeddings():
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.allow_stale_multimodal_embeddings = True
    engine._vision_embedding_cache = {"media": torch.ones(1)}
    engine._vision_embedding_cache_bytes = 4
    engine.requests = {}

    engine._invalidate_vision_state()

    assert "media" in engine._vision_embedding_cache
    assert engine._vision_embedding_cache_bytes == 4


def test_refresh_vlm_request_recomputes_embeddings_and_mask():
    request = DynamicVLMInferenceRequest(
        request_id=32,
        prompt_tokens=torch.tensor([99, 99, 5, 7]),
        compact_prompt_tokens=torch.tensor([99, 5]),
        sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
        block_hash_salt="media",
        num_img_embeddings_per_tile=0,
        imgs=torch.ones(1),
        num_tiles=torch.tensor([1]),
        imgs_sizes=torch.tensor([[1, 1]]),
        decoder_seq_length=0,
        image_embeddings=None,
        image_token_mask=None,
    )
    wrapper = types.SimpleNamespace(
        resolve_media_token_id=mock.Mock(return_value=99),
        expand_image_tokens=mock.Mock(return_value=([[99, 99, 5]], [[0, 1, None]])),
        _forward_vision_encoder=mock.Mock(return_value=torch.ones(2, 1, 4)),
    )
    retained_imgs = mock.Mock()
    device_imgs = torch.ones(1)
    retained_imgs.to.return_value = device_imgs
    request.imgs = retained_imgs
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.controller = types.SimpleNamespace(inference_wrapped_model=wrapper, tokenizer=object())
    engine.context = types.SimpleNamespace(add_vlm_request_data=mock.Mock())
    engine._cache_vision_embedding = mock.Mock()

    engine._refresh_vlm_request_data(request)

    retained_imgs.to.assert_called_once_with(device=request.prompt_tokens.device)
    encoder_args, encoder_kwargs = wrapper._forward_vision_encoder.call_args
    assert encoder_args[0] is device_imgs
    assert encoder_kwargs["num_image_tiles"].device == request.prompt_tokens.device
    assert encoder_kwargs["imgs_sizes"].device == request.prompt_tokens.device
    assert request.image_embeddings is wrapper._forward_vision_encoder.return_value
    assert request.image_token_mask.tolist() == [0, 1, -1, -1]
    engine.context.add_vlm_request_data.assert_called_once_with(
        request.request_id,
        image_embeddings=request.image_embeddings,
        image_token_mask=request.image_token_mask,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_checkpointed_vlm_request_refreshes_cpu_media_on_gpu():
    device = torch.device("cuda", torch.cuda.current_device())
    imgs = torch.ones(1)
    imgs_sizes = torch.tensor([[1, 1]])
    original_request = DynamicVLMInferenceRequest(
        request_id=33,
        prompt_tokens=torch.tensor([99, 99, 5], device=device),
        compact_prompt_tokens=torch.tensor([99, 5], device=device),
        sampling_params=SamplingParams(num_tokens_to_generate=4, termination_id=-1),
        block_hash_salt="media",
        num_img_embeddings_per_tile=0,
        imgs=imgs,
        num_tiles=torch.tensor([1]),
        imgs_sizes=imgs_sizes,
        decoder_seq_length=0,
        generated_tokens=[7, 8],
        image_embeddings=torch.full((2, 1, 4), -1.0),
        image_token_mask=torch.tensor([0, 1, -1]),
    )
    record = DynamicInferenceRequestRecord.from_request(original_request)
    record.checkpoint()
    checkpointed_request = record[-1]

    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine.allow_stale_multimodal_embeddings = False
    engine._vision_embedding_cache = {"media": original_request.image_embeddings}
    engine._vision_embedding_cache_bytes = original_request.image_embeddings.numel() * 4
    engine.requests = {checkpointed_request.request_id: types.SimpleNamespace(record=record)}
    refreshed_embeddings = torch.ones(2, 1, 4, device=device)
    wrapper = types.SimpleNamespace(
        resolve_media_token_id=mock.Mock(return_value=99),
        expand_image_tokens=mock.Mock(return_value=([[99, 99, 5]], [[0, 1, None]])),
        _forward_vision_encoder=mock.Mock(return_value=refreshed_embeddings),
    )
    engine.controller = types.SimpleNamespace(inference_wrapped_model=wrapper, tokenizer=object())
    engine.context = types.SimpleNamespace(add_vlm_request_data=mock.Mock())
    engine._cache_vision_embedding = mock.Mock()

    engine._invalidate_vision_state()
    engine._refresh_vlm_request_data(checkpointed_request)

    assert checkpointed_request.prompt_tokens.tolist() == [99, 99, 5, 7, 8]
    assert checkpointed_request.prompt_tokens.device == device
    assert checkpointed_request.compact_prompt_tokens is original_request.compact_prompt_tokens
    assert checkpointed_request.imgs is imgs
    assert checkpointed_request.imgs_sizes is imgs_sizes
    assert checkpointed_request.imgs.device.type == "cpu"
    assert checkpointed_request.imgs_sizes.device.type == "cpu"
    assert checkpointed_request.image_embeddings is refreshed_embeddings
    assert checkpointed_request.image_token_mask.tolist() == [0, 1, -1, -1, -1]
    encoder_args, encoder_kwargs = wrapper._forward_vision_encoder.call_args
    assert encoder_args[0].device == checkpointed_request.prompt_tokens.device
    assert encoder_kwargs["num_image_tiles"].device == checkpointed_request.prompt_tokens.device
    assert encoder_kwargs["imgs_sizes"].device == checkpointed_request.prompt_tokens.device
    engine.context.add_vlm_request_data.assert_called_once_with(
        checkpointed_request.request_id,
        image_embeddings=refreshed_embeddings,
        image_token_mask=checkpointed_request.image_token_mask,
    )


def test_streaming_partials_are_sent():
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine._partial_emit_lengths = {}
    request = types.SimpleNamespace(
        generated_tokens=[11, 12, 13],
        generated_log_probs=[-0.1, -0.2, -0.3],
        generated_top_n_logprobs=[{"eleven": -0.01}, {"twelve": -0.02}, {"thirteen": -0.03}],
        prompt_log_probs=[-0.4],
        prompt_top_n_logprobs=[{"prompt": -0.04}],
        sampling_params=types.SimpleNamespace(
            streaming=True, return_log_probs=True, skip_prompt_log_probs=False
        ),
    )
    engine.requests = {7: types.SimpleNamespace(record=[request])}
    engine.socket_for_receiving_requests = mock.Mock()

    engine._try_send_streaming_partials()

    engine.socket_for_receiving_requests.send.assert_called_once()
    assert engine._partial_emit_lengths == {7: 3}
    payload = msgpack.unpackb(
        engine.socket_for_receiving_requests.send.call_args.args[0], raw=False
    )
    partial = payload[1][0]
    assert partial["new_top_n_logprobs"] == request.generated_top_n_logprobs
    assert partial["prompt_log_probs"] == request.prompt_log_probs
    assert partial["prompt_top_n_logprobs"] == request.prompt_top_n_logprobs


def test_streaming_partials_buffer_until_token_interval():
    engine = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    engine._partial_emit_lengths = {}
    request = types.SimpleNamespace(
        generated_tokens=[11, 12],
        generated_log_probs=None,
        sampling_params=SamplingParams(
            streaming=True, streaming_interval=3, return_log_probs=False
        ),
    )
    engine.requests = {7: types.SimpleNamespace(record=[request])}
    engine.socket_for_receiving_requests = mock.Mock()

    engine._try_send_streaming_partials()

    engine.socket_for_receiving_requests.send.assert_not_called()
    assert engine._partial_emit_lengths == {}

    request.generated_tokens.append(13)
    engine._try_send_streaming_partials()

    engine.socket_for_receiving_requests.send.assert_called_once()
    assert engine._partial_emit_lengths == {7: 3}


class TestDynamicInferenceEngine(DynamicInferenceEngineTestBase):

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "hybrid"])
    @pytest.mark.parametrize("num_cuda_graphs", [None, 1, 4, -1])
    @pytest.mark.parametrize(
        "inference_cuda_graph_scope", [InferenceCudaGraphScope.layer, InferenceCudaGraphScope.block]
    )
    def test_simple(self, model_provider, num_cuda_graphs, inference_cuda_graph_scope) -> None:
        """Simple test that runs without errors, and validates output."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        num_tokens_to_generate = 16

        # Run test.
        # Force decode-only CG capture: capturing mixed graphs across the full range will OOM.
        env = self._run_test(
            num_tokens_to_generate=num_tokens_to_generate,
            model_provider=model_provider,
            num_cuda_graphs=num_cuda_graphs,
            inference_cuda_graph_scope=inference_cuda_graph_scope,
            force_build_cuda_graphs=True,
            context_max_requests=128,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        # Validate max_requests, max_tokens.
        assert env.engine.context.max_tokens == DynamicInferenceContext.DEFAULT_MAX_TOKENS

        if num_cuda_graphs is not None:
            assert env.engine.context.cuda_graph_token_counts is not None
            assert env.engine.context.cuda_graph_batch_dimensions_list
            model = env.engine.controller.inference_wrapped_model.model
            if inference_cuda_graph_scope == InferenceCudaGraphScope.block:
                # GPT and hybrid models both own the block-scope graph at model level;
                # GPT removes the decoder's fallback manager at construction.
                assert model.cudagraph_manager.cudagraph_runners
                assert not hasattr(model.decoder, 'cudagraph_manager')
            else:
                # check if cudagraph runners are created at the layer level
                for layer in model.decoder.layers:
                    assert layer.cudagraph_manager.cudagraph_runners

        # Because the TextGenerationController produces different outputs on different DP ranks,
        # only verify the accuracy of the output on DP rank 0.
        if parallel_state.get_data_parallel_rank() == 0:

            # Validate generated tokens.
            gpt_expected_generated_tokens = [
                [69, 85, 55, 74, 56, 89, 64, 59, 55, 67, 15, 58, 6, 37, 54, 47],
                [29, 54, 33, 72, 45, 76, 41, 56, 28, 25, 17, 2, 61, 6, 98, 76],
                [35, 78, 54, 16, 79, 98, 22, 5, 60, 0, 1, 76, 77, 11, 25, 7],
                [25, 75, 57, 85, 81, 37, 88, 17, 71, 15, 70, 64, 50, 0, 64, 45],
                [32, 5, 85, 75, 30, 68, 23, 33, 20, 26, 89, 20, 49, 28, 38, 81],
                [33, 69, 32, 49, 93, 24, 33, 6, 54, 89, 92, 97, 42, 80, 50, 53],
                [82, 78, 78, 65, 26, 5, 69, 36, 37, 99],
                [51, 70, 22, 1, 87, 42, 36, 26, 27, 56, 82, 32, 8, 80, 20, 43],
            ]

            mamba_expected_generated_tokens = [
                [69, 85, 55, 74, 85, 89, 64, 59, 55, 67, 15, 58, 6, 37, 34, 47],
                [29, 16, 33, 30, 45, 76, 41, 46, 82, 17, 17, 2, 61, 6, 98, 76],
                [35, 78, 54, 16, 79, 98, 22, 5, 37, 30, 1, 76, 5, 11, 25, 86],
                [25, 75, 57, 85, 81, 59, 88, 38, 71, 15, 70, 64, 50, 0, 64, 45],
                [32, 5, 85, 75, 30, 68, 23, 33, 20, 26, 35, 20, 49, 28, 34, 81],
                [87, 69, 32, 49, 93, 24, 33, 6, 54, 89, 92, 97, 42, 80, 50, 53],
                [82, 78, 78, 19, 70, 5, 97, 36, 37, 99],
                [51, 70, 22, 1, 87, 42, 36, 26, 27, 56, 82, 32, 8, 20, 20, 43],
            ]

            if model_provider == "gpt":
                expected_generated_tokens_list = gpt_expected_generated_tokens
            elif model_provider == "hybrid":
                expected_generated_tokens_list = mamba_expected_generated_tokens
            else:
                raise ValueError(f"Invalid model_provider {model_provider}")

            print(f"Validating {len(env.requests)} requests.")
            print(f"Expected generated tokens: {expected_generated_tokens_list}")
            print(
                f"Actual generated tokens: {[request.generated_tokens for request in env.requests]}"
            )

            assert len(env.requests) == len(expected_generated_tokens_list)

            for request, expected_generated_tokens in zip(
                env.requests, expected_generated_tokens_list
            ):
                assert request.generated_tokens == expected_generated_tokens, (
                    f"request {request.request_id}, "
                    f"result ({request.generated_tokens}) != "
                    f"expected ({expected_generated_tokens})."
                )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_full_iteration_impl_does_not_setup_inference_cuda_graphs(self) -> None:
        """impl=full_iteration is training-only; inference graph setup follows inference scope."""
        env = self._build_test_env(
            DynamicEngineTestConfig(
                model_provider="gpt",
                num_tokens_to_generate=4,
                num_cuda_graphs=1,
                force_build_cuda_graphs=True,
                context_max_requests=128,
                cuda_graph_impl="full_iteration",
                inference_cuda_graph_scope=InferenceCudaGraphScope.none,
            )
        )

        self._assert_inference_cuda_graphs_disabled(env)

        with mock.patch.object(
            env.engine.controller,
            "_dynamic_step_forward_logits",
            wraps=env.engine.controller._dynamic_step_forward_logits,
        ) as forward_logits:
            with torch.inference_mode():
                env.engine.create_cuda_graphs()

        assert forward_logits.call_count == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_deprecated_full_iteration_inference_scope_matches_new_flag_runtime_behavior(
        self,
    ) -> None:
        """Deprecated scope='full_iteration_inference' must still build block-level graphs."""
        with pytest.warns(
            DeprecationWarning, match="cuda_graph_modules 'full_iteration_inference' is deprecated"
        ):
            env = self._run_test(
                num_tokens_to_generate=4,
                model_provider="gpt",
                num_cuda_graphs=1,
                cuda_graph_modules='full_iteration_inference',
                force_build_cuda_graphs=True,
                context_max_requests=128,
            )

        model = env.engine.controller.inference_wrapped_model.model
        assert model.config.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert model.config.cuda_graph_modules == []
        assert model.cudagraph_manager.cudagraph_runners
        assert not hasattr(model.decoder, 'cudagraph_manager')
        for layer in model.decoder.layers:
            assert not hasattr(layer, 'cudagraph_manager')

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_token_overflow_transient(self) -> None:
        """Test token overflow."""
        test_config = DynamicEngineTestConfig(
            num_requests=2,
            min_prompt_length=512,
            max_prompt_length=512,
            num_tokens_to_generate=2,
            context_max_tokens=900,
        )
        env = self._build_test_env(test_config)
        env.engine._add_request(env.requests[0])
        env.engine._add_request(env.requests[1])
        env.engine.schedule_waiting_requests()
        assert list(env.engine.waiting_request_ids) == [1]

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skip(
        reason="activate for `megatron-core >= 0.16`, after fixing "
        "`raise TokenOverflowError(is_transient=False)` compatibility with "
        "legacy tests."
    )
    def test_token_overflow_nontransient(self) -> None:
        """Test token overflow (non-transient)."""
        test_config = DynamicEngineTestConfig(context_max_tokens=8)
        env = self._build_test_env(test_config)
        try:
            env.engine._add_request(env.requests[0])
        except TokenOverflowError as e:
            assert e.is_transient == False
        else:
            raise Exception("should have raised TokenOverflowError(is_transient=False).")

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "hybrid"])
    def test_block_overflow(self, model_provider: str) -> None:
        """Test block overflow."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        env = self._build_test_env(DynamicEngineTestConfig(model_provider=model_provider))
        context = env.engine.context
        block_size_bytes = context.block_size_bytes
        buffer_size_gb = (block_size_bytes + 1) / 1024**3
        test_config = DynamicEngineTestConfig(
            context_buffer_size_gb=buffer_size_gb, model_provider=model_provider
        )
        env = self._build_test_env(test_config)
        env.engine._add_request(env.requests[0])
        assert list(env.engine.waiting_request_ids) == [0]

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_block_overflow_insufficient_kv_cache(self) -> None:
        """Test that a request fails when KV cache blocks cannot fit the request's sequence."""
        # Use a large max_sequence_length with a small buffer so that the total
        # block count is smaller than what a single max-length request needs.
        # With num_tokens_total=8192 and prompt_length=4, the request needs
        # ceil(8192 / 256) = 32 blocks, but the small buffer only has ~8 blocks.
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=None,
            num_tokens_total=8192,
            max_sequence_length=8192,
            context_buffer_size_gb=0.001,
            context_block_size_tokens=256,
            context_max_tokens=16384,
        )
        env = self._build_test_env(test_config)
        request = env.requests[0]
        env.engine._add_request(request)
        assert request.status == Status.FAILED
        assert list(env.engine.waiting_request_ids) == []

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_max_sequence_length_clamp(self) -> None:
        """Clamp (not reject) when num_tokens_to_generate exceeds the remaining sequence budget."""
        test_config = DynamicEngineTestConfig(
            num_requests=1, min_prompt_length=8, max_prompt_length=8, num_tokens_to_generate=4
        )
        env = self._build_test_env(test_config)
        request = env.requests[0]

        remaining_tokens = env.engine.context.max_sequence_length - len(request.prompt_tokens)
        request.sampling_params.num_tokens_to_generate = remaining_tokens + 100

        env.engine._add_request(request)

        # Clamped to the remaining budget, not rejected.
        assert request.status != Status.FAILED
        assert request.sampling_params.num_tokens_to_generate == remaining_tokens

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("num_speculative_tokens,exact_fit_tokens", [(0, 249), (2, 247)])
    def test_generation_within_tight_kv_pool(
        self, num_speculative_tokens: int, exact_fit_tokens: int
    ) -> None:
        """Admission is bounded by what the shared pool can grant a running request.
        Only stored tokens need slots:
        the final sampled token is never stored, the last decode step stores its speculative drafts.
        Exact fit: 8 prompt + (exact_fit_tokens - 1) outputs + drafts = 256."""
        env = self._build_test_env(DynamicEngineTestConfig())
        block_size_bytes = env.engine.context.block_size_bytes

        # 2-block pool: 1 usable + 1 dummy.
        test_config = DynamicEngineTestConfig(
            num_requests=3,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=None,
            max_sequence_length=512,
            num_speculative_tokens=num_speculative_tokens,
            context_buffer_size_gb=2 * block_size_bytes / 1024**3,
            context_paused_buffer_size_gb=0.0,
            context_max_requests=4,
        )
        env = self._build_test_env(test_config)

        # The msl-derived default budget (8 + 504) exceeds the usable block.
        doomed_request = env.requests[0]
        env.engine._add_request(doomed_request)
        assert doomed_request.status == Status.FAILED

        # One more stored token than the usable block holds; fails at admission.
        overflow_request = env.requests[2]
        overflow_request.sampling_params.num_tokens_to_generate = exact_fit_tokens + 1
        env.engine._add_request(overflow_request)
        assert overflow_request.status == Status.FAILED

        # An exact-fit request runs to completion.
        request = env.requests[1]
        request.sampling_params.num_tokens_to_generate = exact_fit_tokens
        request.sampling_params.termination_id = -1  # never terminate early
        env.engine._add_request(request)
        assert request.status != Status.FAILED

        # Bound the loop so a scheduling regression fails instead of hanging.
        for _ in range(400):
            self._run_step(env)
            if not env.engine.has_unfinished_requests():
                break
        assert not env.engine.has_unfinished_requests()
        assert request.status == Status.COMPLETED
        assert len(request.output) == exact_fit_tokens

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_sustained_kv_cache_saturation_with_paused_retention(self) -> None:
        """Many concurrent requests against a KV pool far too small to hold them all.

        The other eviction tests pin the paused buffer to zero, so every pause turns
        into an immediate eviction and the retention budget is never consulted. Here
        the budget holds two blocks, which is smaller than the number of requests that
        pause, so `update_requests` has to retain some paused requests, resume others,
        and evict only the overflow -- repeatedly, over many steps.

        This is the shape high-concurrency RL workloads hit and standalone inference
        tests do not: the invariants worth pinning are that the retention budget is
        never exceeded, that requeued requests still generate their full output, and
        that the pool drains completely afterwards. A block leaked during a saturation
        storm silently shrinks capacity for every later request.
        """
        num_requests = 16
        prompt_length = 256
        num_tokens_to_generate = 64
        block_size_tokens = 256
        pool_block_count = 8
        paused_block_count = 2

        probe_env = self._build_test_env(DynamicEngineTestConfig())
        block_size_bytes = probe_env.engine.context.block_size_bytes

        # A prompt fills exactly one block, so every request needs a second block on
        # its first decode step -- the pause trigger. Sizing in whole blocks (+1 byte
        # to survive the float GB round trip) keeps the pool arithmetic exact.
        test_config = DynamicEngineTestConfig(
            num_requests=num_requests,
            min_prompt_length=prompt_length,
            max_prompt_length=prompt_length,
            num_tokens_to_generate=num_tokens_to_generate,
            context_block_size_tokens=block_size_tokens,
            context_buffer_size_gb=(pool_block_count * block_size_bytes + 1) / 1024**3,
            context_paused_buffer_size_gb=(paused_block_count * block_size_bytes + 1) / 1024**3,
            context_max_requests=num_requests,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)
        context = env.engine.context
        allocator = context.kv_block_allocator

        # Fail loudly if the buffer sizing did not land where the test assumes, rather
        # than silently running a differently-shaped workload.
        assert allocator.pool_size == pool_block_count
        assert allocator.paused_limit == paused_block_count

        # Deterministic logits: random weights make torch.multinomial trip over NaNs.
        vocab_size = test_config.vocab_size

        def mock_greedy_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            logits = torch.zeros(
                *tokens.shape, vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0
            if test_config.materialize_only_last_token_logits:
                logits = context.last_token_logits(logits).unsqueeze(0)
            return logits

        env.engine.controller.inference_wrapped_model.model.forward = mock_greedy_forward

        # Every request arrives up front, so the engine has to queue and reschedule
        # rather than admitting them at a comfortable rate.
        for request in env.requests:
            request.sampling_params.termination_id = -1  # run the full output length
            env.engine._add_request(request)

        finished_records = []
        max_paused_block_count = 0

        # Drive admission through step_modern so async scheduling only mutates the
        # request rows after consuming the pending forward. Bound the loop so a
        # scheduling regression fails instead of hanging.
        for _ in range(4000):
            if not env.engine.has_unfinished_requests():
                break
            finished_records.extend(env.engine.step_modern()["finished_request_records"])

            # Checked at a step boundary, i.e. after the pause/resume/evict lifecycle
            # has settled: paused requests may only retain blocks within the budget.
            paused_block_count_now = allocator.get_paused_used()
            assert paused_block_count_now <= allocator.paused_limit
            max_paused_block_count = max(max_paused_block_count, paused_block_count_now)

        assert not env.engine.has_unfinished_requests()

        # The workload has to have actually saturated the cache, otherwise the
        # assertions above are vacuous and the test rots into a no-op. A request
        # surviving a step boundary while paused is what a zero budget cannot
        # produce -- with no retention every paused request must resume or be
        # evicted immediately -- so together with an eviction this pins the
        # partial-retention path the other tests skip.
        assert max_paused_block_count > 0, "no request was ever retained paused"
        assert env.engine.evicted_request_count > 0, "no request overflowed the budget"

        # Requeued requests resume from a checkpointed prompt, so the output length is
        # what catches a token double-counted or dropped across an eviction.
        assert len(finished_records) == num_requests
        for record in finished_records:
            request = record.merge()
            assert request.status == Status.COMPLETED, f"request {request.request_id} unfinished"
            assert len(request.generated_tokens) == num_tokens_to_generate

        # No block, request row, or token slot may survive the storm.
        assert allocator.get_total_used() == 0
        assert allocator.pool_avail == allocator.pool_size - 1
        assert context.total_request_count == 0
        assert context.paused_request_count == 0
        assert context.active_token_count == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_cuda_graph_padding_uses_dummy_block(self) -> None:
        """One real request in a four-request graph bucket: the padded block-table
        rows must hold the dummy block index, not the old -1 sentinel (OOB reads)."""
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=8,
            max_prompt_length=8,
            num_cuda_graphs=1,
            context_max_requests=4,
        )
        env = self._build_test_env(test_config)
        context = env.engine.context

        env.engine._add_request(env.requests[0])
        self._run_step(env)  # prefill
        self._run_step(env)  # decode: 1 real request in the 4-request graph bucket

        assert context.using_cuda_graph_this_step()
        assert context.padded_batch_dimensions.req_count == 4
        padded_rows = context._cpu_mha_block_table[1:4]
        assert (padded_rows != -1).all()
        assert (padded_rows == context.kv_block_allocator.dummy_block_idx).all()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_active_row_block_table_tail_uses_dummy_block(self) -> None:
        """A real request's unallocated trailing block-table columns must be
        staged as the dummy block, not the -1 sentinel: graphed decode
        advertises max_seqlen_k = max_sequence_length, so the kernel's
        page-table reach can include the tail of a near-limit request."""
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=8,
            max_prompt_length=8,
            num_cuda_graphs=1,
            context_max_requests=4,
            max_sequence_length=512,
        )
        env = self._build_test_env(test_config)
        context = env.engine.context

        env.engine._add_request(env.requests[0])
        self._run_step(env)  # prefill
        self._run_step(env)  # decode: graphed, one real row

        assert context.using_cuda_graph_this_step()
        # The 512-token budget spans two 256-token pages. 8-token prompt allocates only the first.
        # So the real row genuinely has an unallocated tail to pin.
        block_count = int(context.request_kv_block_counts[0].item())
        staged_row = context._cpu_mha_block_table[0]
        assert 0 < block_count < staged_row.numel()
        assert (staged_row != -1).all()
        assert (staged_row[block_count:] == context.kv_block_allocator.dummy_block_idx).all()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "hybrid"])
    def test_multi_add(self, model_provider: str) -> None:
        """Test adding multiple requests simultaneously."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        self._run_test(num_gap_steps=0, model_provider=model_provider)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        # Cover three regimes:
        #   - SWA active on every layer (window_attn_skip_freq=None)
        #   - SWA active on a subset of layers (gpt-oss style: every other layer)
        #   - window smaller than the longest sequence we generate, so the
        #     kernel actually applies the local-attention mask.
        "window_size,window_attn_skip_freq",
        [((4, 0), None), ((4, 0), 2), ((127, 0), 2)],
    )
    def test_sliding_window_attention(
        self, window_size: Tuple[int, int], window_attn_skip_freq: Optional[int]
    ) -> None:
        """Exercise SWA on the dynamic batching (FA2/FA3/FA4) attention path.

        This mirrors the gpt-oss configuration (window 127 to the left, no
        future tokens, applied every other layer) at a much smaller scale.
        The test only checks that decoding runs end-to-end and produces the
        expected number of tokens; numerical correctness of the SWA kernels
        themselves is owned by the upstream flash-attention test suites.
        """
        self._run_test(
            model_provider="gpt",
            num_gap_steps=0,
            window_size=window_size,
            window_attn_skip_freq=window_attn_skip_freq,
            # Disable CUDA graphs: this test only validates the SWA plumbing
            # through the attention kernel, not the CG capture path.
            num_cuda_graphs=None,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "hybrid"])
    def test_fixed_output_lengths(self, model_provider: str) -> None:
        """Test generating a fixed number of output tokens."""
        skip_if_mamba_sequence_packing_not_available(model_provider)
        self._run_test(use_fixed_output_lengths=True, model_provider=model_provider)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("use_non_decode", [False, True])
    def test_cuda_graph_token_counts(self, use_non_decode: bool) -> None:
        """Test initialization of `cuda_graph_token_counts` in dynamic context."""

        # Exponential-decay graph distribution (halve from max down to tp_size). Pinned
        # explicitly below: the production default is HYBRID, which spaces the
        # decode-only family linearly instead and so yields different token counts.
        # decode-only path: cuda_graph_max_tokens = max_requests * (spec+1) = 80.
        # non-decode path: cuda_graph_max_tokens = self.max_tokens (DEFAULT 16384);
        # most large prefill sizes are filtered by is_valid because
        # token_count > prefill_req_count * (max_sequence_length - 1).
        decode_only_cases = [
            (0, [80]),
            (1, [80]),
            (2, [80, 1]),
            (4, [80, 40, 20, 1]),
            (8, [80, 40, 20, 10, 4, 2, 1]),
            (16, [80, 40, 20, 10, 4, 2, 1]),
            (32, [80, 40, 20, 10, 4, 2, 1]),
        ]
        non_decode_cases = [
            (0, [80]),
            (1, [80]),
            (2, [80, 1]),
            (4, [80, 40, 20, 1]),
            (8, [1024, 512, 256, 80, 40, 20, 10, 4, 2, 1]),
            (16, [1024, 512, 256, 128, 80, 64, 40, 32, 20, 16, 10, 8, 4, 2, 1]),
            (32, [1024, 512, 256, 128, 80, 64, 40, 32, 20, 16, 10, 8, 4, 2, 1]),
        ]
        cases = non_decode_cases if use_non_decode else decode_only_cases

        for num_cuda_graphs, expected_cuda_graph_token_counts in cases:

            # Build cuda graphs (inside dynamic engine).
            env = self._build_test_env(
                DynamicEngineTestConfig(
                    context_buffer_size_gb=0.01,
                    num_cuda_graphs=num_cuda_graphs,
                    use_cuda_graphs_for_non_decode_steps=use_non_decode,
                    cuda_graph_all_prefills=use_non_decode,
                    cuda_graph_sizing_distribution=CudaGraphSizingDistribution.EXPONENTIAL,
                )
            )
            actual_cuda_graph_token_counts = env.engine.context.cuda_graph_token_counts
            assert actual_cuda_graph_token_counts == expected_cuda_graph_token_counts, (
                "num_cuda_graphs %d use_non_decode=%s ... cuda_graph_token_counts: "
                "expected %s, found %s."
                % (
                    num_cuda_graphs,
                    use_non_decode,
                    expected_cuda_graph_token_counts,
                    actual_cuda_graph_token_counts,
                )
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("model_provider", ["gpt", "hybrid"])
    @torch.inference_mode()
    def test_generate_function(self, model_provider: str) -> None:
        """Test the generate function that processes multiple prompts at once."""
        skip_if_mamba_sequence_packing_not_available(model_provider)

        # Set up test environment
        test_config = DynamicEngineTestConfig(
            num_requests=4,
            max_prompt_length=8,
            num_tokens_to_generate=4,
            model_provider=model_provider,
        )
        env = self._build_test_env(test_config)

        # Create string prompts (just mock strings, since the test environment mocks the tokenizer)
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]

        # Mock the tokenize_prompt method to return predictable token sequences
        def mock_tokenize_prompt(tokenizer, prompt, add_BOS=False):
            # Return a token sequence based on the prompt number
            prompt_num = int(prompt[-1])
            return [10 + i for i in range(prompt_num + 2)]

        env.engine.controller.tokenize_prompt = mock_tokenize_prompt

        # Call the generate function.
        # It's safe to use request 0's sampling params here because all sampling
        # params are identical as long as use_fixed_output_lengths == False.
        finished_request_records = env.engine.generate(prompts, env.requests[0].sampling_params)
        finished_requests = [r.merge() for r in finished_request_records]

        # Verify results
        assert len(finished_requests) == len(
            prompts
        ), "Should return same number of finished requests as prompts"

        request_ids = [r.request_id for r in finished_requests]
        assert request_ids == sorted(
            request_ids
        ), f"Request ids are not in sorted order: {request_ids}"

        # Check each request was processed
        for i, request in enumerate(finished_requests):
            # Verify each request has generated tokens
            assert len(request.generated_tokens) > 0, f"Request {i} should have generated tokens"
            assert request.status == Status.COMPLETED, f"Request {i} should be completed"

    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    async def test_run_engine(self):
        """
        Test asynchronously adding and waiting for requests while the engine is
        running continuously.
        """
        # Have to wrap inference mode in-line because async functions are not supported
        with torch.inference_mode():
            # Test environment.
            test_config = DynamicEngineTestConfig(num_requests=8, use_fixed_output_lengths=True)
            env = self._build_test_env(test_config)

            engine_task = asyncio.create_task(env.engine.run_engine())

            request_completion_futures: Dict[int, asyncio.Future[DynamicInferenceRequest]] = {}

            # Add requests to engine.
            for request in tqdm(env.requests, "add requests"):
                request_completion_futures[request.request_id] = env.engine._add_request(request)

            # Wait for all requests to complete.
            await asyncio.gather(*request_completion_futures.values())

            # Verify that all request outputs were set.
            for request_id, fut in request_completion_futures.items():
                num_tokens_to_generate = env.requests[
                    request_id
                ].sampling_params.num_tokens_to_generate
                request_record = fut.result()
                request = request_record.merge()
                assert request.generated_length == num_tokens_to_generate, (
                    f"Request {request_id} expected to generate {num_tokens_to_generate} "
                    f"tokens but generated {request.generated_length}"
                )

            engine_task.cancel()

    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    async def test_async_sched_run_engine_accepts_request_during_overlap(self):
        """Verify async overlap yields so a new request can enter a running engine."""
        with torch.inference_mode():
            test_config = DynamicEngineTestConfig(
                num_requests=2,
                min_prompt_length=4,
                max_prompt_length=4,
                num_tokens_to_generate=16,
                async_sched_mode=AsyncScheduleMode.ASYNC,
            )
            env = self._build_test_env(test_config)
            long_request, short_request = env.requests
            for request in env.requests:
                request.sampling_params.top_k = 1
                request.sampling_params.top_p = 0.0
                request.sampling_params.termination_id = -1
            short_request.sampling_params.num_tokens_to_generate = 2

            engine_task = asyncio.create_task(env.engine.run_engine())
            try:
                long_request_future = env.engine._add_request(long_request)

                while len(long_request.generated_tokens) < 2:
                    await asyncio.sleep(0)

                generated_count_at_submission = len(long_request.generated_tokens)
                short_request_future = env.engine._add_request(short_request)
                await asyncio.gather(long_request_future, short_request_future)

                assert generated_count_at_submission < 16
                assert len(short_request.generated_tokens) == 2
            finally:
                engine_task.cancel()
                await engine_task

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        ("enable_chunked_prefill", "enable_prefix_caching"),
        [(True, False), (False, True), (True, True)],
    )
    @torch.inference_mode()
    def test_async_sched_prefix_caching_and_chunked_prefill_e2e(
        self, enable_chunked_prefill, enable_prefix_caching
    ):
        """Async output matches legacy for chunking, KV caching, and their combination."""

        def run(mode):
            test_config = DynamicEngineTestConfig(
                num_requests=0,
                num_tokens_to_generate=4,
                max_sequence_length=768,
                context_block_size_tokens=256,
                context_max_tokens=384 if enable_chunked_prefill else 1024,
                context_max_requests=4,
                enable_chunked_prefill=enable_chunked_prefill,
                enable_prefix_caching=enable_prefix_caching,
                async_sched_mode=mode,
            )
            env = self._build_test_env(test_config)
            prompt = torch.arange(512, dtype=torch.int64, device="cuda") % (
                test_config.vocab_size - 1
            )
            outputs = {}

            def add_request(request_id):
                env.engine.add_request(
                    request_id=request_id,
                    prompt=prompt.clone(),
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=4, termination_id=-1, top_k=1, top_p=0.0
                    ),
                )

            add_request(0)
            env.engine.step_modern()
            add_request(1)
            while env.engine.has_unfinished_requests():
                result = env.engine.step_modern()
                for record in result["finished_request_records"]:
                    request = record.merge()
                    outputs[request.request_id] = list(request.generated_tokens)
            return env.engine, outputs

        _, legacy_outputs = run(AsyncScheduleMode.LEGACY)
        async_engine, async_outputs = run(AsyncScheduleMode.ASYNC)

        assert async_outputs == legacy_outputs
        assert all(len(tokens) == 4 for tokens in async_outputs.values())
        assert async_engine.context.async_sched_step_count > 0
        if enable_prefix_caching:
            assert async_engine._prefill_tokens_skipped > 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "feature_config, sampling_backend, temperature, top_k, top_p, num_cuda_graphs",
        [
            pytest.param({}, "torch", 0.8, 10, 0.0, None, id="torch-eager-top-k"),
            pytest.param({}, "torch", 1.0, 0, 0.9, 2, id="torch-graphed-forward-top-p"),
            pytest.param({}, "flashinfer", 1.2, 0, 0.0, None, id="flashinfer-eager-unfiltered"),
            pytest.param({}, "flashinfer", 0.8, 10, 0.9, 2, id="flashinfer-graphed-forward"),
            pytest.param(
                {
                    "model_provider": "hybrid",
                    "num_speculative_tokens": 1,
                    "num_requests": 2,
                    "num_tokens_to_generate": 4,
                },
                "torch",
                0.8,
                8,
                0.0,
                None,
                id="mamba-mtp-top-k",
            ),
        ],
    )
    @torch.inference_mode()
    def test_async_sched_sampling_matches_legacy(
        self, feature_config, sampling_backend, temperature, top_k, top_p, num_cuda_graphs
    ):
        """Require seeded sampling parity across scheduling modes.

        Args:
            feature_config (dict): Additional cumulative feature configuration.
            sampling_backend (str): Sampling implementation under test.
            temperature (float): Sampling temperature used by every request.
            top_k (int): Top-k filter used by every request.
            top_p (float): Top-p filter used by every request.
            num_cuda_graphs (Optional[int]): Number of CUDA graph buckets, or
                `None` for eager execution.
        """
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")
        if feature_config.get("model_provider") == "hybrid":
            skip_if_mamba_sequence_packing_not_available("hybrid")

        common_config = dict(
            num_requests=4,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=6,
            num_gap_steps=0,
            use_fixed_output_lengths=True,
            sampling_backend=sampling_backend,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            context_max_requests=8,
            num_cuda_graphs=num_cuda_graphs,
            force_build_cuda_graphs=num_cuda_graphs is not None,
            use_cuda_graphs_for_non_decode_steps=False,
        )
        common_config.update(feature_config)
        generated_tokens = {}
        final_env = None
        for mode in (AsyncScheduleMode.LEGACY, AsyncScheduleMode.ASYNC):
            final_env = self._run_test(async_sched_mode=mode, **common_config)
            assert all(request.status == Status.COMPLETED for request in final_env.requests)
            generated_tokens[mode] = [request.generated_tokens for request in final_env.requests]

        assert (
            generated_tokens[AsyncScheduleMode.ASYNC] == generated_tokens[AsyncScheduleMode.LEGACY]
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "feature_config, sampling_backend, logprobs_mode, skip_prompt_log_probs, num_cuda_graphs",
        [
            pytest.param({}, "torch", "raw_logprobs", False, None, id="torch-raw-prompt"),
            pytest.param({}, "torch", "processed_logprobs", True, 2, id="torch-processed-graph"),
            pytest.param({}, "flashinfer", "raw_logprobs", False, None, id="flashinfer-raw-prompt"),
            pytest.param(
                {}, "flashinfer", "processed_logprobs", True, 2, id="flashinfer-processed-graph"
            ),
            pytest.param(
                {
                    "model_provider": "hybrid",
                    "num_speculative_tokens": 1,
                    "num_requests": 2,
                    "num_tokens_to_generate": 4,
                },
                "torch",
                "raw_logprobs",
                True,
                None,
                id="mamba-mtp-raw",
            ),
        ],
    )
    @torch.inference_mode()
    def test_async_sched_log_probs_match_legacy(
        self,
        feature_config,
        sampling_backend,
        logprobs_mode,
        skip_prompt_log_probs,
        num_cuda_graphs,
    ):
        """Require prompt and generated logprob parity across scheduling modes.

        Args:
            feature_config (dict): Additional cumulative feature configuration.
            sampling_backend (str): Sampling implementation under test.
            logprobs_mode (str): Raw or sampling-processed logprob mode.
            skip_prompt_log_probs (bool): Whether to omit prompt logprobs.
            num_cuda_graphs (Optional[int]): Number of CUDA graph buckets.
        """
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")
        if feature_config.get("model_provider") == "hybrid":
            skip_if_mamba_sequence_packing_not_available("hybrid")

        common_config = dict(
            num_requests=4,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=6,
            num_gap_steps=0,
            use_fixed_output_lengths=True,
            model_provider="gpt",
            sampling_backend=sampling_backend,
            temperature=0.8,
            top_k=8,
            return_log_probs=True,
            logprobs_mode=logprobs_mode,
            materialize_only_last_token_logits=skip_prompt_log_probs,
            skip_prompt_log_probs=skip_prompt_log_probs,
            context_max_requests=8,
            num_cuda_graphs=num_cuda_graphs,
            force_build_cuda_graphs=num_cuda_graphs is not None,
            use_cuda_graphs_for_non_decode_steps=False,
        )
        common_config.update(feature_config)
        outputs = {}
        for mode in AsyncScheduleMode:
            env = self._run_test(async_sched_mode=mode, **common_config)
            outputs[mode] = [
                (request.generated_tokens, request.prompt_log_probs, request.generated_log_probs)
                for request in env.requests
            ]

        legacy_outputs = outputs[AsyncScheduleMode.LEGACY]
        for legacy, actual in zip(legacy_outputs, outputs[AsyncScheduleMode.ASYNC]):
            assert actual[0] == legacy[0]
            assert (actual[1] or []) == pytest.approx(legacy[1] or [])
            assert actual[2] == pytest.approx(legacy[2])

    def _run_stop_word_schedule(
        self,
        test_config: DynamicEngineTestConfig,
        stop_word: Optional[str] = None,
        detokenize_stop_sequence: bool = False,
    ) -> DynamicEngineTestEnv:
        """Run a schedule where only the first request has a string stop word.

        Args:
            test_config (DynamicEngineTestConfig): Engine configuration for the run.
            stop_word (Optional[str]): Whitespace-delimited token IDs used as the stop word.
            detokenize_stop_sequence (bool): Whether the completed output retains the stop word.

        Returns:
            DynamicEngineTestEnv: Completed test environment.
        """
        env = self._build_test_env(test_config)
        env.engine.controller.tokenizer.bos = None
        env.engine.controller.tokenizer.tokenize = lambda text: [
            int(token_id) for token_id in text.split()
        ]

        for request_idx, request in enumerate(env.requests):
            request.sampling_params.termination_id = -1
            request.sampling_params.detokenize_stop_sequence = detokenize_stop_sequence
            if request_idx == 0 and stop_word is not None:
                request.sampling_params.stop_words = [stop_word]
            env.engine._add_request(request)

        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        return env

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "sampling_backend,num_cuda_graphs,detokenize_stop_sequence",
        [
            pytest.param("torch", None, True, id="torch-eager-keep"),
            pytest.param("torch", 2, False, id="torch-graph-strip"),
            pytest.param("flashinfer", None, False, id="flashinfer-eager-strip"),
            pytest.param("flashinfer", 2, True, id="flashinfer-graph-keep"),
        ],
    )
    @torch.inference_mode()
    def test_async_sched_stop_words_match_legacy(
        self, sampling_backend, num_cuda_graphs, detokenize_stop_sequence
    ):
        """Require string stop-word parity while survivor requests keep decoding.

        Args:
            sampling_backend (str): Sampling backend under test.
            num_cuda_graphs (Optional[int]): CUDA graph bucket count, or ``None`` for eager mode.
            detokenize_stop_sequence (bool): Whether completed output retains the stop word.
        """
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")

        common_config = dict(
            num_requests=4,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=8,
            num_gap_steps=0,
            model_provider="gpt",
            sampling_backend=sampling_backend,
            temperature=1.0,
            top_k=1,
            context_max_requests=8,
            num_cuda_graphs=num_cuda_graphs,
            force_build_cuda_graphs=num_cuda_graphs is not None,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        probe_env = self._run_stop_word_schedule(
            DynamicEngineTestConfig(async_sched_mode=AsyncScheduleMode.LEGACY, **common_config)
        )
        stop_word_ids = probe_env.requests[0].generated_tokens[2:4]
        stop_word = " ".join(str(token_id) for token_id in stop_word_ids)

        legacy_env = self._run_stop_word_schedule(
            DynamicEngineTestConfig(async_sched_mode=AsyncScheduleMode.LEGACY, **common_config),
            stop_word,
            detokenize_stop_sequence,
        )
        async_env = self._run_stop_word_schedule(
            DynamicEngineTestConfig(async_sched_mode=AsyncScheduleMode.ASYNC, **common_config),
            stop_word,
            detokenize_stop_sequence,
        )

        legacy_tokens = [request.generated_tokens for request in legacy_env.requests]
        async_tokens = [request.generated_tokens for request in async_env.requests]
        assert async_tokens == legacy_tokens
        assert len(async_tokens[0]) < common_config["num_tokens_to_generate"]
        if detokenize_stop_sequence:
            assert async_tokens[0][-len(stop_word_ids) :] == stop_word_ids
        assert async_env.engine.context.async_sched_step_count > 0
        assert async_env.engine.context.async_sched_compaction_step_count > 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("model_provider", ["gpt", "hybrid"])
    def test_fp8_inference(self, model_provider: str):
        skip_if_mamba_sequence_packing_not_available(model_provider)

        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        self._run_test(model_provider=model_provider, fp8=True, hidden_size=128)

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_return_log_probs(self):
        """Verify that log probs are returned and computed correctly."""
        # Returning log probs requires materializing the full prompt logits or
        # explicitly disabling prompt logits.
        with pytest.raises(AssertionError):
            env = self._run_test(return_log_probs=True, materialize_only_last_token_logits=True)

        # Test with full logits materialization
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=False,
            num_tokens_to_generate=5,
        )

        # Validate log probs for each completed request
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # Validate prompt log probs
            if request.prompt_log_probs is not None and len(request.prompt_log_probs) > 0:
                prompt_len = len(request.prompt_tokens)
                # Should have log probs for all tokens except the first one
                assert len(request.prompt_log_probs) == prompt_len - 1, (
                    f"Request {request.request_id}: Expected {prompt_len - 1} prompt log probs, "
                    f"got {len(request.prompt_log_probs)}"
                )

                # Validate each prompt log prob
                for i, log_prob in enumerate(request.prompt_log_probs):
                    assert not math.isnan(
                        log_prob
                    ), f"Request {request.request_id}, prompt token {i}: log_prob is NaN"
                    assert not math.isinf(
                        log_prob
                    ), f"Request {request.request_id}, prompt token {i}: log_prob is inf"
                    assert log_prob <= 0.0, (
                        f"Request {request.request_id}, prompt token {i}: "
                        f"log_prob {log_prob} should be <= 0"
                    )
                    assert log_prob >= -50.0, (
                        f"Request {request.request_id}, prompt token {i}: "
                        f"log_prob {log_prob} is unreasonably small"
                    )

            # Validate generated log probs
            assert (
                request.generated_log_probs is not None
            ), f"Request {request.request_id}: generated_log_probs should not be None"
            assert len(request.generated_log_probs) == len(request.generated_tokens), (
                f"Request {request.request_id}: Expected {len(request.generated_tokens)} "
                f"generated log probs, got {len(request.generated_log_probs)}"
            )

            # Validate each generated log prob
            for i, log_prob in enumerate(request.generated_log_probs):
                assert not math.isnan(
                    log_prob
                ), f"Request {request.request_id}, generated token {i}: log_prob is NaN"
                assert not math.isinf(
                    log_prob
                ), f"Request {request.request_id}, generated token {i}: log_prob is inf"
                assert log_prob <= 0.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} should be <= 0"
                )
                assert log_prob >= -50.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is unreasonably small"
                )

            # Validate that all generated tokens are valid
            for i, token_id in enumerate(request.generated_tokens):
                assert 0 <= token_id < env.config.vocab_size, (
                    f"Request {request.request_id}, token {i}: token_id {token_id} "
                    f"is out of valid range [0, {env.config.vocab_size})"
                )

        # Test with skipping prompt log probs
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=True,
            skip_prompt_log_probs=True,
            num_tokens_to_generate=5,
        )

        # Validate that prompt log probs are empty/None when skipped
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # When skip_prompt_log_probs is True, prompt_log_probs should be empty
            assert request.prompt_log_probs is None or len(request.prompt_log_probs) == 0, (
                f"Request {request.request_id}: prompt_log_probs should be empty when "
                f"skip_prompt_log_probs=True, but got {len(request.prompt_log_probs)} items"
            )

            # Generated log probs should still be present
            assert (
                request.generated_log_probs is not None and len(request.generated_log_probs) > 0
            ), f"Request {request.request_id}: generated_log_probs should be present"

            # Validate generated log probs are still valid
            for i, log_prob in enumerate(request.generated_log_probs):
                assert not math.isnan(log_prob) and not math.isinf(log_prob), (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is invalid"
                )
                assert -50.0 <= log_prob <= 0.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is out of expected range [-50.0, 0.0]"
                )

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("async_sched_mode", list(AsyncScheduleMode))
    @torch.inference_mode()
    def test_return_prompt_log_probs_with_zero_tokens_to_generate(
        self, async_sched_mode: AsyncScheduleMode
    ):
        """Prompt log probs must be returned when scoring only (num_tokens_to_generate=0).

        Regression test for a prefill-step trimming bug: when a request generates
        no tokens, the end-of-generation trim set ``keep=0`` and front-sliced
        ``request_log_probs[:0]``, discarding every prompt log prob (in a prefill
        step ``request_log_probs`` covers the whole prompt, with the disposable
        sampled-token log prob at the tail). The fix trims the excess *trailing*
        log probs instead. This is the path exercised by loglikelihood / echo
        evaluations (e.g. lm-eval-harness sends ``max_tokens=0``).

        Args:
            async_sched_mode (AsyncScheduleMode): Scheduling mode under test.
        """
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=False,
            skip_prompt_log_probs=False,
            num_tokens_to_generate=0,
            async_sched_mode=async_sched_mode,
        )

        validated_any = False
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # No tokens were requested, so none should be generated.
            assert len(request.generated_tokens) == 0, (
                f"Request {request.request_id}: expected 0 generated tokens, "
                f"got {len(request.generated_tokens)}"
            )
            assert request.generated_log_probs is None or len(request.generated_log_probs) == 0, (
                f"Request {request.request_id}: expected no generated log probs, got "
                f"{len(request.generated_log_probs) if request.generated_log_probs else 0}"
            )

            # The full set of prompt log probs (all tokens except the first) must
            # still be present -- before the fix this list was empty.
            prompt_len = len(request.prompt_tokens)
            assert request.prompt_log_probs is not None, (
                f"Request {request.request_id}: prompt_log_probs should not be None "
                f"when scoring with num_tokens_to_generate=0"
            )
            assert len(request.prompt_log_probs) == prompt_len - 1, (
                f"Request {request.request_id}: Expected {prompt_len - 1} prompt log probs, "
                f"got {len(request.prompt_log_probs)}"
            )
            for i, log_prob in enumerate(request.prompt_log_probs):
                assert (
                    log_prob is not None
                ), f"Request {request.request_id}, prompt token {i}: log_prob is None"
                assert not math.isnan(log_prob) and not math.isinf(
                    log_prob
                ), f"Request {request.request_id}, prompt token {i}: log_prob {log_prob} invalid"
                assert -50.0 <= log_prob <= 0.0, (
                    f"Request {request.request_id}, prompt token {i}: "
                    f"log_prob {log_prob} is out of expected range [-50.0, 0.0]"
                )
            validated_any = True

        assert validated_any, "No completed requests were validated"

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_log_probs_token_correspondence(self):
        """
        Verify that log probabilities correspond to the actual sampled tokens.
        This test checks that the log probability reported for each token actually
        corresponds to that token's probability in the distribution.
        """
        # Run test with log probs enabled
        env = self._run_test(
            return_log_probs=True,
            materialize_only_last_token_logits=False,
            num_tokens_to_generate=5,
            num_requests=4,
        )

        # For each completed request
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # Check that we have log probs for generated tokens
            assert request.generated_log_probs is not None
            assert len(request.generated_log_probs) == len(request.generated_tokens)

            # Verify log probs are valid and in reasonable range
            for i, (token_id, log_prob) in enumerate(
                zip(request.generated_tokens, request.generated_log_probs)
            ):
                # Basic validity checks
                assert not math.isnan(
                    log_prob
                ), f"Request {request.request_id}, token {i}: log_prob is NaN"
                assert not math.isinf(
                    log_prob
                ), f"Request {request.request_id}, token {i}: log_prob is inf"

                # Log probabilities should be <= 0 (since prob <= 1)
                assert log_prob <= 0.0, (
                    f"Request {request.request_id}, token {i}: "
                    f"log_prob {log_prob} should be <= 0"
                )

                # Check reasonable range (not too negative)
                # Using a more lenient threshold since actual model outputs can vary
                assert log_prob >= -100.0, (
                    f"Request {request.request_id}, token {i}: "
                    f"log_prob {log_prob} is unreasonably small"
                )

                # Token ID should be valid
                assert 0 <= token_id < env.config.vocab_size, (
                    f"Request {request.request_id}, token {i}: "
                    f"token_id {token_id} is out of range [0, {env.config.vocab_size})"
                )

            # Check prompt log probs if available
            if request.prompt_log_probs is not None and len(request.prompt_log_probs) > 0:
                expected_prompt_log_probs = len(request.prompt_tokens) - 1
                assert len(request.prompt_log_probs) == expected_prompt_log_probs, (
                    f"Request {request.request_id}: Expected {expected_prompt_log_probs} "
                    f"prompt log probs, got {len(request.prompt_log_probs)}"
                )

                for i, log_prob in enumerate(request.prompt_log_probs):
                    assert not math.isnan(log_prob) and not math.isinf(log_prob)
                    assert -100.0 <= log_prob <= 0.0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_num_tokens_total(self):
        """Simple test, but using num_tokens_total instead of num_tokens_to_generate."""
        # Run test.
        env = self._run_test(
            num_tokens_to_generate=None, num_tokens_total=20, use_fixed_output_lengths=True
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.skip(
        reason="activate for `megatron-core >= 0.16`, after fixing "
        "`raise TokenOverflowError(is_transient=False)` compatibility with "
        "legacy tests."
    )
    def test_events(self):
        """Test events."""
        env = self._run_test(
            num_requests=16,
            max_prompt_length=10,
            num_tokens_to_generate=32,
            context_buffer_size_gb=0.001,  # 0.001, # 8 blocks
            context_max_tokens=8,
            num_gap_steps=1,
        )

        expected_event_types = [
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ERROR_NONTRANSIENT', 'FAIL'],
            ['ADD_ENGINE', 'ADD_CONTEXT', 'FINISH'],
        ]
        result_event_types = [
            [e.type.name for e in r.events if e.type.name != 'GENERATED_TOKEN']
            for r in env.requests
        ]
        assert result_event_types == expected_event_types

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_event_timestamps(self):
        """Test that events are recorded with sensical timestamps.

        Verifies:
        1. Completed requests have ADD_ENGINE, ADD_CONTEXT, GENERATED_TOKEN(s), FINISH events
        2. Event timestamps are monotonically increasing
        3. TTFT (time-to-first-token) can be computed as first GENERATED_TOKEN - ADD_ENGINE
        """
        num_tokens_to_generate = 8
        env = self._run_test(
            num_requests=4,
            max_prompt_length=16,
            num_tokens_to_generate=num_tokens_to_generate,
            context_buffer_size_gb=0.1,
            num_gap_steps=0,
            track_generated_token_events=True,
        )

        # All requests should complete with this generous config (large buffer, no gap steps).
        assert all(r.status == Status.COMPLETED for r in env.requests)
        for request in env.requests:

            # Verify event types for completed requests
            event_types = [e.type.name for e in request.events]
            # Should be: ADD_ENGINE, ADD_CONTEXT, GENERATED_TOKEN (repeated), FINISH
            assert (
                event_types[0] == 'ADD_ENGINE'
            ), f"Request {request.request_id}: first event should be ADD_ENGINE, got {event_types[0]}"
            assert (
                event_types[1] == 'ADD_CONTEXT'
            ), f"Request {request.request_id}: second event should be ADD_CONTEXT, got {event_types[1]}"
            assert (
                event_types[-1] == 'FINISH'
            ), f"Request {request.request_id}: last event should be FINISH, got {event_types[-1]}"
            # Check that GENERATED_TOKEN events are in the middle
            gen_token_count = event_types.count('GENERATED_TOKEN')
            assert gen_token_count == len(request.generated_tokens), (
                f"Request {request.request_id}: GENERATED_TOKEN count ({gen_token_count}) != "
                f"generated_tokens length ({len(request.generated_tokens)})"
            )

            # Verify timestamps are monotonically increasing
            timestamps = [e.timestamp for e in request.events]
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i - 1], (
                    f"Request {request.request_id}: timestamp[{i}] ({timestamps[i]}) < "
                    f"timestamp[{i-1}] ({timestamps[i-1]})"
                )

            # Verify TTFT is positive and sensical (first GENERATED_TOKEN - ADD_ENGINE)
            add_engine_ts = request.events[0].timestamp
            first_token_ts = request.events[2].timestamp  # First GENERATED_TOKEN event
            assert (
                request.events[2].type.name == 'GENERATED_TOKEN'
            ), f"Request {request.request_id}: event[2] should be GENERATED_TOKEN"
            ttft = first_token_ts - add_engine_ts
            assert ttft >= 0, f"Request {request.request_id}: TTFT is negative ({ttft})"

            # Verify total request time is positive
            finish_ts = request.events[-1].timestamp
            total_time = finish_ts - add_engine_ts
            assert (
                total_time >= ttft
            ), f"Request {request.request_id}: total_time ({total_time}) < TTFT ({ttft})"

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("ssm_mixer", ["mamba", "gdp"])
    def test_mamba_chunked_prefill(self, ssm_mixer):
        """
        Test chunked prefill with an SSM model (Mamba2 or Gated Delta Product).
        """
        skip_if_mamba_sequence_packing_not_available("hybrid", ssm_mixer)

        # Context max tokens = 50.
        test_config = DynamicEngineTestConfig(
            model_provider="hybrid",
            ssm_mixer=ssm_mixer,
            num_requests=0,
            num_tokens_to_generate=None,
            num_tokens_total=200,
            context_max_tokens=52,
            context_max_requests=5,
            context_block_size_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        # Mock the model forward function to avoid possible numerics issues
        # caused by random inputs
        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        # Request 1: 150 tokens
        req1_tokens = torch.randint(0, test_config.vocab_size, (130,), device='cuda')
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req1_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=3),
        )

        # Request 2: 160 tokens
        req2_tokens = torch.randint(0, test_config.vocab_size, (160,), device='cuda')
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=req2_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )

        # Request 3: 24 tokens
        req3_tokens = torch.randint(0, test_config.vocab_size, (24,), device='cuda')
        req3 = DynamicInferenceRequest(
            request_id=3,
            prompt_tokens=req3_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )

        # Add requests 1-3
        env.engine._add_request(req1)
        env.engine._add_request(req2)
        env.engine._add_request(req3)

        # Step 1: schedule and launch the first prefill chunk as the async primer.
        env.engine.step_modern()
        assert req1.finished_chunk_token_count == 52

        # Step 2: resolve the first chunk and launch the second chunk.
        env.engine.step_modern()
        assert req1.finished_chunk_token_count == 104

        # Verify that requests 2 and 3 are queued because request 1 is still running
        assert ctx.num_prefill_requests == 1
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert 1 in active_ids
        assert 2 not in active_ids
        assert 3 not in active_ids
        assert list(env.engine.waiting_request_ids) == [1, 2, 3]

        # Verify that active token count == max tokens
        assert ctx.active_token_count == 52

        # Verify that request 1 is the designated chunked prefill request
        assert ctx.chunked_prefill_request_id == 1

        # Step 3: resolve request 1's second chunk and launch the batch containing
        # request 1's final prefill chunk and request 2's first chunk.
        env.engine.step_modern()
        assert req1.finished_chunk_token_count == 104

        # Verify that request 2 got partially scheduled and is now
        # the designated chunked prefill request
        req2_idx = ctx.request_ids.tolist().index(2)
        assert req2_idx == 1
        assert ctx.num_prefill_requests == 2
        assert ctx.chunked_prefill_request_id == 2
        assert ctx.get_index_of_chunked_prefill_request() == req2_idx
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert 1 in active_ids
        assert 2 in active_ids
        assert 3 not in active_ids

        # Store the Mamba state tensor idx for request 2
        req2_mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[req2_idx].item()

        # Verify that the active token count is the maximum token count
        assert ctx.active_token_count == 52

        # Step 4: resolve that mixed batch and launch request 1's decode with
        # request 2's next prefill chunk.
        env.engine.step_modern()
        assert req2.finished_chunk_token_count == 77

        # Verify that request 2 is still the first prefill request
        assert ctx.request_ids.tolist().index(2) == 1
        assert ctx.mamba_metadata.request_to_mamba_state_idx[1] == req2_mamba_idx

        # Verify that request 1 is running decode
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert ctx.num_decode_requests == 1
        assert 1 in active_ids

        # Verify that request 2 is still running prefill as the designated chunked prefill request
        assert ctx.num_prefill_requests == 1
        assert ctx.chunked_prefill_request_id == 2
        assert ctx.get_index_of_chunked_prefill_request() == 1

        # Verify that request 3 is still waiting
        assert 3 not in active_ids
        assert 3 in env.engine.waiting_request_ids

        # Verify that active token count == max tokens
        assert ctx.active_token_count == 52

        # Step 5: resolve that batch and launch the next decode/prefill batch.
        env.engine.step_modern()
        assert req2.finished_chunk_token_count == 128

        # Verify that request 2 is still the first prefill request
        assert ctx.request_ids.tolist().index(2) == 1
        assert ctx.mamba_metadata.request_to_mamba_state_idx[1] == req2_mamba_idx

        # Step 6: resolve request 1's final decode and launch request 2's final
        # prefill chunk together with request 3's first chunk.
        env.engine.step_modern()

        # Verify that request 1 has completed
        assert req1.status == Status.COMPLETED

        # Verify that request 2 is still the first prefill request
        assert ctx.request_ids.tolist().index(2) == 0
        assert ctx.mamba_metadata.request_to_mamba_state_idx[0] == req2_mamba_idx

        # Verify that request 3 is now scheduled as the chunked prefill request
        active_ids = ctx.request_ids[: ctx.total_request_count].tolist()
        assert 2 in active_ids
        assert 3 in active_ids
        assert ctx.chunked_prefill_request_id == 3
        req3_idx = active_ids.index(3)
        assert req3_idx == 1

        # Store the Mamba state tensor idx for request 3
        req3_mamba_idx = ctx.mamba_metadata.request_to_mamba_state_idx[req3_idx].item()

        assert req3.finished_chunk_token_count == 20

        # Step 7: resolve request 2's final prefill and launch request 3's final
        # prefill chunk.
        env.engine.step_modern()

        # Verify that request 2 has finished
        assert req2.status == Status.COMPLETED

        # Verify that request 3 is now the first prefill request
        req3_idx = ctx.request_ids.tolist().index(3)
        assert req3_idx == 0
        assert ctx.mamba_metadata.request_to_mamba_state_idx[0] == req3_mamba_idx

        # Step 8: resolve request 3's final prefill output.
        env.engine.step_modern()

        # Verify that request 3 has finished
        assert req3.status == Status.COMPLETED

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_chunked_prefill_avoid_single_token_chunk(self):
        """
        Test that chunked prefill scheduling avoids leaving exactly 1 token for the final chunk.
        This leads to a known bug in the Flash Attention kernel:
        https://github.com/Dao-AILab/flash-attention/issues/1537

        Scenario:
            - Max tokens per step (Chunk Size): 256
            - Request prompt length: 513

        Default scheduling would do:
            1. Chunk 256 (Remaining 257)
            2. Chunk 256 (Remaining 1) -> max_seqlen_q=1 triggers decode path in kernel
            3. Chunk 1

        Fixed scheduling should do:
            1. Chunk 256 (Remaining 257) -> 513 - 256 == 257. Schedule full 256.
            2. Chunk 255 (Remaining 2)   -> 257 tokens left. If we take 256, 1 remains.
                                            So we reduce chunk to 255.
            3. Chunk 2   (Remaining 0)
        """
        prefill_chunk_size = 256
        # Prompt length designed to trigger the edge case: Chunk + (Chunk + 1)
        # 256 + 255 + 2 = 513
        prompt_len = 513

        test_config = DynamicEngineTestConfig(
            model_provider="gpt",
            num_requests=0,
            num_tokens_to_generate=None,
            num_tokens_total=prompt_len + 1,
            context_max_tokens=prefill_chunk_size,
            context_max_requests=1,
            context_block_size_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        # Mock the model forward function to avoid possible numerics issues
        # caused by random inputs
        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        # Create a request with length 513
        req_tokens = torch.randint(0, test_config.vocab_size, (prompt_len,), device='cuda')
        req = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )

        env.engine._add_request(req)

        assert req.status == Status.ACTIVE_AND_GENERATING_TOKENS

        # --- Step 1 (async primer) ---
        # Available: 256. Remaining: 513.
        # Logic: 513 - 256 = 257. Not 1. Schedule full 256.
        env.engine.step_modern()

        assert env.engine.context.total_request_count == 1, env.engine.context.total_request_count
        assert ctx.request_query_lengths[0].item() == 256

        assert (
            req.finished_chunk_token_count == 256
        ), f"Step 1: Expected 256 tokens processed, got {req.finished_chunk_token_count}"

        # --- Step 2 ---
        # Resolve the first chunk and launch the second.
        # Available: 256. Remaining un-prefilled: 257.
        # Logic: 257 - 256 = 1. This is the edge case!
        # Fix should reduce chunk size by 1 (to 255).
        env.engine.step_modern()

        assert env.engine.context.total_request_count == 1, env.engine.context.total_request_count
        assert ctx.request_query_lengths[0].item() == 255

        # 256 (previous) + 255 (this step) = 511
        assert req.finished_chunk_token_count == 511, (
            "Step 2: Expected 511 tokens processed (256+255), "
            f"got {req.finished_chunk_token_count}. "
        )

        # --- Step 3 ---
        # Resolve the second chunk and launch the final chunk.
        # Remaining un-prefilled: 2. Available: 256.
        # Logic: 2 <= 256. Schedule 2.
        env.engine.step_modern()

        assert ctx.total_request_count == 1
        assert ctx.num_prefill_requests == 1
        assert ctx.request_query_lengths[0].item() == 2

        # --- Step 4 ---
        # Resolve the final prefill output and complete the request.
        env.engine.step_modern()

        assert ctx.num_prefill_requests == 0
        assert req.status == Status.COMPLETED

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_chunked_prefill_delay_scheduling_for_unavoidable_single_token_chunk(self):
        """
        Test that chunked prefill scheduling delays execution when the only available
        option is to schedule a chunk of size 1 that leaves exactly 1 token remaining.

        Scenario:
            - Max tokens per step: 256
            - Request A: 254 token prompt
            - Request B: 2 token prompt

        Sequence:
            1. Step 1 async primer:
               - Request A is scheduled (255 tokens).
               - Context has 1 token available (256 - 255).
               - Request B has 2 tokens remaining.
               - If we schedule 1 token for B, it leaves exactly 1 token for its final chunk,
                 crashing FA3. Since chunk_length is 1, we can't safely reduce it.
                 The engine MUST delay scheduling Request B.
            2. Step 2 resolves Request A and schedules Request B.
               - Request A completes after its prefill sample is resolved.
               - Context has all 256 tokens available.
               - Request B is now safely scheduled for its full 2 tokens.
            3. Step 3 resolves Request B's prefill sample.
        """
        test_config = DynamicEngineTestConfig(
            model_provider="gpt",
            num_requests=0,
            num_tokens_to_generate=None,
            num_tokens_total=256,
            context_max_tokens=256,
            context_max_requests=2,
            context_block_size_tokens=256,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        # Mock the model forward function to avoid possible numerics issues
        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        # Add Request A (Length 255)
        req_a_tokens = torch.randint(0, test_config.vocab_size, (255,), device='cuda')
        req_a = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req_a_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        env.engine._add_request(req_a)

        # Add Request B (Length 2)
        req_b_tokens = torch.randint(0, test_config.vocab_size, (2,), device='cuda')
        req_b = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=req_b_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        env.engine._add_request(req_b)

        # --- Step 1 (async primer) ---
        # Schedule and launch Request A fully (255), but delay Request B.
        env.engine.step_modern()

        assert ctx.total_request_count == 1
        assert ctx.active_token_count == 255

        # Request B MUST be delayed (0 tokens processed) to avoid the FA3 bug
        assert (
            req_b.finished_chunk_token_count == 0
        ), "Request B should have been delayed to avoid leaving a 1-token chunk"
        assert len(env.engine.waiting_request_ids) == 1
        assert env.engine.waiting_request_ids[0] == 2

        # --- Step 2 ---
        # Resolve Request A, then schedule and launch Request B's full 2-token prompt.
        env.engine.step_modern()

        assert req_a.status == Status.COMPLETED
        assert ctx.total_request_count == 1
        assert ctx.request_ids[0].item() == 2
        assert ctx.request_query_lengths[0].item() == 2

        # --- Step 3 ---
        # Resolve Request B's prefill output.
        env.engine.step_modern()

        assert req_b.status == Status.COMPLETED
        assert len(env.engine.waiting_request_ids) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_chunked_prefill_skips_request_when_token_budget_exhausted(self):
        """
        Test that chunked prefill scheduling leaves a waiting request untouched when the
        token batch is already full (active_token_count == max_tokens), instead of
        scheduling a zero-length chunk.

        This exercises the `token_partially_can_be_added` guard in the scheduler's entry
        condition. When a fully-admitted request fills the batch exactly to max_tokens, a
        following waiting request has zero token budget. Without the guard, the scheduler
        would compute token_budget = max_tokens - active_token_count = 0 and try to admit a
        0-token chunk, tripping `assert prefill_chunk_length > 0` in `add_request`.

        Scenario:
            - Max tokens per step: 16
            - Request A: 16-token prompt -> fully admitted, fills the batch exactly.
            - Request B: 8-token prompt  -> zero budget remains; must stay waiting.
        """
        max_tokens = 16

        test_config = DynamicEngineTestConfig(
            model_provider="gpt",
            num_requests=0,
            num_tokens_to_generate=None,
            num_tokens_total=max_tokens,
            context_max_tokens=max_tokens,
            context_max_requests=2,
            context_block_size_tokens=256,
            max_sequence_length=128,
            enable_chunked_prefill=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        # Mock the model forward function to avoid possible numerics issues
        # caused by random inputs.
        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        # Request A: prompt length == max_tokens, so it fills the batch exactly.
        req_a_tokens = torch.randint(0, test_config.vocab_size, (max_tokens,), device='cuda')
        req_a = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req_a_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        env.engine._add_request(req_a)

        # Request B: added after A, so it sits behind A in the waiting queue.
        req_b_tokens = torch.randint(0, test_config.vocab_size, (8,), device='cuda')
        req_b = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=req_b_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        env.engine._add_request(req_b)

        # Schedule without crashing: A is fully admitted, B is left waiting.
        env.engine.schedule_waiting_requests()

        # Request A fully admitted and the batch is exactly full.
        assert ctx.active_token_count == max_tokens
        assert ctx.total_request_count == 1
        assert ctx.num_prefill_requests == 1
        # A was a full (non-chunked) admit, so no chunked prefill is in flight.
        assert ctx.chunked_prefill_request_id == -1

        # Request B got zero budget: untouched and still at the head of the waiting queue.
        assert req_b.finished_chunk_token_count == 0
        assert list(env.engine.waiting_request_ids) == [2]

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_prefix_caching_avoid_single_token_effective_chunk(self):
        """
        Test that prefix caching combined with chunked prefill avoids leaving exactly
        1 token for the effective prefill chunk. A 1-token prefill chunk routes to
        the Flash Attention decode kernel, which crashes due to shape mismatches.
        """
        block_size = 16
        prompt_len = 17  # 1 full block (16) + 1 token

        test_config = DynamicEngineTestConfig(
            model_provider="gpt",
            num_requests=0,
            num_tokens_to_generate=1,
            context_max_tokens=256,
            context_max_requests=2,
            context_block_size_tokens=block_size,
            max_sequence_length=128,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            use_cuda_graphs_for_non_decode_steps=False,
        )

        env = self._build_test_env(test_config)
        ctx = env.engine.context

        model_instance = env.engine.controller.inference_wrapped_model.model
        model_instance.forward = partial(mock_forward, vocab_size=test_config.vocab_size)

        req_a_tokens = torch.randint(0, test_config.vocab_size, (prompt_len,), device='cuda')

        # Request A: Populate the prefix cache (set to generate 10 tokens so it stays active)
        req_a = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=req_a_tokens,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # 1. Add Request A and launch its async primer. Scheduling registers its
        # complete block in the prefix cache.
        env.engine._add_request(req_a)
        env.engine.step_modern()

        # Request B: Same prompt, added AFTER Req A's blocks are registered
        req_b = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=req_a_tokens.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=1),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
        )

        # 2. Add Request B. The next async step resolves Request A, then admits and
        # launches Request B using the cached block.
        env.engine._add_request(req_b)
        env.engine.step_modern()

        # Verify that `_compute_prefix_match` successfully clamped the skip.
        req_b_idx = ctx.request_ids.tolist().index(2)

        assert ctx.request_query_lengths[req_b_idx].item() == 17, (
            f"Expected effective chunk length to be backed off to 17, "
            f"but got {ctx.request_query_lengths[req_b_idx].item()}."
        )

    @pytest.mark.internal
    @torch.inference_mode()
    def test_mamba_match_is_chunk_local_when_chunked_prefill_limits_kv_match(self):
        """Mamba restore depth is bounded to KV blocks assigned for the current chunk."""
        block_size = 256
        block_hashes = [111, 222]
        req = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.arange(512, dtype=torch.int64, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=1),
            block_size_tokens=block_size,
            enable_prefix_caching=True,
            precomputed_block_hashes=block_hashes,
        )

        # This simulates the old scheduler-side full-prompt Mamba match. The
        # context must ignore it and record the chunk-local executable count.
        req._mamba_num_matched_blocks = 2

        ctx = DynamicInferenceContext.__new__(DynamicInferenceContext)
        ctx.block_size_tokens = block_size
        ctx.enable_prefix_caching = True
        ctx.is_hybrid_model = True
        ctx.kv_block_allocator = types.SimpleNamespace(
            kv_hash_to_block_id={block_hashes[0]: 7, block_hashes[1]: 8}
        )
        ctx.mamba_slot_allocator = types.SimpleNamespace(
            hash_to_block_id={block_hashes[0]: 7, block_hashes[1]: 8}
        )

        (
            matched_block_ids,
            num_blocks_from_pool,
            already_allocated_blocks,
            overall_required_blocks,
            prefix_skip_tokens,
            effective_prefill_chunk_length,
        ) = DynamicInferenceContext._compute_prefix_match(
            ctx, req, prefill_chunk_length=211, record_mamba_match=True
        )

        assert matched_block_ids == [7]
        assert num_blocks_from_pool == 0
        assert already_allocated_blocks == 0
        assert overall_required_blocks == 1
        assert req._mamba_num_matched_blocks == 1
        assert prefix_skip_tokens == 0
        assert effective_prefill_chunk_length == 211

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @torch.inference_mode()
    def test_chunked_prefill_with_log_probs(
        self, materialize_only_last_token_logits: bool, skip_prompt_log_probs: bool
    ):
        """
        Test that chunked prefill correctly handles log probs across all branches
        of the log-prob accumulation logic.
        When materialize_only_last_token_logits=True, skip_prompt_log_probs must be True.
        """
        if materialize_only_last_token_logits and not skip_prompt_log_probs:
            with pytest.raises(AssertionError, match="only last token logits are materialized"):
                self._run_test(
                    num_requests=1,
                    min_prompt_length=1200,
                    max_prompt_length=1200,
                    num_tokens_to_generate=8,
                    materialize_only_last_token_logits=True,
                    return_log_probs=True,
                    skip_prompt_log_probs=False,
                    model_provider="gpt",
                    context_block_size_tokens=256,
                    context_max_tokens=1000,
                    enable_chunked_prefill=True,
                )
            return

        prompt_length = 1200
        num_tokens_to_generate = 8

        env = self._run_test(
            num_requests=1,
            min_prompt_length=prompt_length,
            max_prompt_length=prompt_length,
            num_tokens_to_generate=num_tokens_to_generate,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            return_log_probs=True,
            skip_prompt_log_probs=skip_prompt_log_probs,
            model_provider="gpt",
            context_block_size_tokens=256,
            context_max_tokens=1000,
            enable_chunked_prefill=True,
        )

        # Validate results
        for request in env.requests:
            if request.status != Status.COMPLETED:
                continue

            # Validate generated log probs
            assert (
                request.generated_log_probs is not None
            ), f"Request {request.request_id}: generated_log_probs should not be None"
            assert len(request.generated_log_probs) == len(request.generated_tokens), (
                f"Request {request.request_id}: Expected {len(request.generated_tokens)} "
                f"generated log probs, got {len(request.generated_log_probs)}"
            )

            if skip_prompt_log_probs:
                assert request.prompt_log_probs is None or len(request.prompt_log_probs) == 0, (
                    f"Request {request.request_id}: prompt_log_probs should be empty when "
                    f"skip_prompt_log_probs=True, but got "
                    f"{len(request.prompt_log_probs) if request.prompt_log_probs else 0} items"
                )
            else:
                assert len(request.prompt_log_probs) == prompt_length - 1, (
                    f"Request {request.request_id}: Expected {prompt_length - 1} "
                    f"prompt log probs, got {len(request.prompt_log_probs)}"
                )

            # Validate each generated log prob
            for i, log_prob in enumerate(request.generated_log_probs):
                assert not math.isnan(log_prob) and not math.isinf(log_prob), (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is invalid"
                )
                assert -50.0 <= log_prob <= 0.0, (
                    f"Request {request.request_id}, generated token {i}: "
                    f"log_prob {log_prob} is out of expected range [-50.0, 0.0]"
                )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_chunked_prefill_log_probs_match_baseline(self):
        """
        Verify that chunked prefill computes the exact same prompt log probabilities
        as non-chunked prefill. This explicitly catches the bug where garbage
        sampled tokens corrupt the prompt log probabilities at chunk boundaries.
        """
        prompt_length = 512
        num_tokens_to_generate = 4

        # Create a deterministic mock forward pass that returns logits
        # dependent ONLY on position_ids. This guarantees the same logits
        # whether processed in one giant chunk or split across multiple chunks.
        def deterministic_mock_forward(input_ids, position_ids, attention_mask, *args, **kwargs):
            vocab_size = kwargs["vocab_size"]
            # Use torch.linspace to generate varying but 100% deterministic logits per position
            static_logits = torch.linspace(
                -50, 50, 4096 * vocab_size, device=input_ids.device, dtype=torch.bfloat16
            ).view(4096, vocab_size)

            return static_logits[position_ids]

        def get_log_probs(chunked: bool, max_tokens: int):
            test_config = DynamicEngineTestConfig(
                num_requests=0,  # Added manually below
                num_tokens_to_generate=num_tokens_to_generate,
                materialize_only_last_token_logits=False,
                return_log_probs=True,
                skip_prompt_log_probs=False,
                model_provider="gpt",
                context_block_size_tokens=256,
                context_max_requests=1,
                context_max_tokens=max_tokens,
                max_sequence_length=1024,
                enable_chunked_prefill=chunked,
                use_cuda_graphs_for_non_decode_steps=False,
            )
            env = self._build_test_env(test_config)

            # Patch the mock forward to be deterministic
            model_instance = env.engine.controller.inference_wrapped_model.model
            model_instance.forward = partial(
                deterministic_mock_forward, vocab_size=test_config.vocab_size
            )

            # Ensure identical prompt tokens for both runs
            torch.manual_seed(42)
            req_tokens = torch.randint(0, test_config.vocab_size, (prompt_length,), device='cuda')
            req = DynamicInferenceRequest(
                request_id=1,
                prompt_tokens=req_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    return_log_probs=True,
                    skip_prompt_log_probs=False,
                    termination_id=-1,
                ),
            )

            env.engine._add_request(req)

            # Drive the engine until the request finishes
            while env.engine.has_unfinished_requests():
                env.engine.step_modern()

            return req.prompt_log_probs

        # Run non-chunked baseline (all 512 tokens in one pass)
        baseline_log_probs = get_log_probs(chunked=False, max_tokens=1000)

        # Run chunked (512 tokens split across 256-token boundaries)
        chunked_log_probs = get_log_probs(chunked=True, max_tokens=256)

        assert baseline_log_probs is not None, "Baseline prompt_log_probs is missing"
        assert chunked_log_probs is not None, "Chunked prompt_log_probs is missing"

        assert len(baseline_log_probs) == prompt_length - 1
        assert len(chunked_log_probs) == prompt_length - 1

        # Compare element-wise using math.isclose to handle minor floating point rounding
        for i, (base_lp, chunk_lp) in enumerate(zip(baseline_log_probs, chunked_log_probs)):
            assert math.isclose(base_lp, chunk_lp, rel_tol=1e-3, abs_tol=1e-3), (
                f"Log prob mismatch at prompt token index {i}: "
                f"Baseline={base_lp:.4f}, Chunked={chunk_lp:.4f}. "
                "This indicates log prob corruption at chunk boundaries!"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("async_sched_mode", list(AsyncScheduleMode))
    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @torch.inference_mode()
    def test_top_n_logprobs_dynamic(
        self, skip_prompt_log_probs: bool, async_sched_mode: AsyncScheduleMode
    ):
        """Test that top_n_logprobs are computed correctly in dynamic batching mode.

        Verifies:
        1. top_n_logprobs are returned for generated tokens
        2. skip_prompt_log_probs controls whether prompt top-n logprobs are skipped
        3. The top-n values are consistent with the selected token's log prob

        Args:
            skip_prompt_log_probs (bool): Whether to omit prompt top-n logprobs.
            async_sched_mode (AsyncScheduleMode): Scheduling mode under test.
        """
        # Build test environment with multiple requests of varying lengths
        test_config = DynamicEngineTestConfig(
            num_requests=4,
            min_prompt_length=4,
            max_prompt_length=12,
            num_tokens_to_generate=4,
            materialize_only_last_token_logits=False,
            async_sched_mode=async_sched_mode,
        )
        env = self._build_test_env(test_config)

        # Override detokenize to produce unique strings per token ID so the
        # top-n dict doesn't collapse all entries to a single key.
        env.engine.controller.tokenizer.detokenize = lambda tokens, **kw: f"tok_{tokens[0]}"

        # Create requests with top_n_logprobs enabled.
        # top_n must be >= top_k so the sampled token is guaranteed to appear
        # in the top-n dict for the consistency check below.
        top_n = 10
        requests_to_add = []
        for request in env.requests:
            # Update sampling params to include top_n_logprobs
            request.sampling_params = SamplingParams(
                num_tokens_to_generate=test_config.num_tokens_to_generate,
                termination_id=test_config.vocab_size - 1,
                return_log_probs=True,
                top_n_logprobs=top_n,
                skip_prompt_log_probs=skip_prompt_log_probs,
                top_k=10,  # Add some sampling randomness
            )
            requests_to_add.append(request)

        # Add requests and run inference
        for request in requests_to_add:
            env.engine._add_request(request)

        # Step engine until all requests are finished
        while env.engine.has_unfinished_requests():
            result = env.engine.step_modern()

        # Validate results
        for request in requests_to_add:
            assert request.status == Status.COMPLETED, f"Request {request.request_id} not completed"

            # Validate generated top-n logprobs
            assert hasattr(
                request, 'generated_top_n_logprobs'
            ), f"Request {request.request_id} missing generated_top_n_logprobs"
            assert (
                request.generated_top_n_logprobs is not None
            ), f"Request {request.request_id} has None generated_top_n_logprobs"
            assert len(request.generated_top_n_logprobs) == len(
                request.generated_tokens
            ), f"Request {request.request_id}: generated_top_n_logprobs length mismatch"

            # Validate each top-n dict
            for i, top_n_dict in enumerate(request.generated_top_n_logprobs):
                assert isinstance(
                    top_n_dict, dict
                ), f"Request {request.request_id}, token {i}: top_n_dict is not a dict"
                assert (
                    len(top_n_dict) <= top_n
                ), f"Request {request.request_id}, token {i}: too many top-n entries"
                assert (
                    len(top_n_dict) > 0
                ), f"Request {request.request_id}, token {i}: empty top-n dict"

            # Validate prompt top-n logprobs based on skip_prompt_log_probs flag
            if not skip_prompt_log_probs:
                assert hasattr(
                    request, 'prompt_top_n_logprobs'
                ), f"Request {request.request_id} missing prompt_top_n_logprobs"
                assert (
                    request.prompt_top_n_logprobs is not None
                ), f"Request {request.request_id} has None prompt_top_n_logprobs"
                # Prompt top-n should have N-1 entries (excluding first token)
                expected_prompt_top_n_len = len(request.prompt_tokens) - 1
                assert (
                    len(request.prompt_top_n_logprobs) == expected_prompt_top_n_len
                ), f"Request {request.request_id}: prompt_top_n_logprobs length {len(request.prompt_top_n_logprobs)} != expected {expected_prompt_top_n_len}"

                # Validate each prompt top-n dict
                for i, top_n_dict in enumerate(request.prompt_top_n_logprobs):
                    assert isinstance(
                        top_n_dict, dict
                    ), f"Request {request.request_id}, prompt token {i}: top_n_dict is not a dict"
                    assert (
                        len(top_n_dict) <= top_n
                    ), f"Request {request.request_id}, prompt token {i}: too many top-n entries"
                    assert (
                        len(top_n_dict) > 0
                    ), f"Request {request.request_id}, prompt token {i}: empty top-n dict"
            else:
                # When skip_prompt_log_probs is True, prompt_top_n_logprobs should be None or empty
                if hasattr(request, 'prompt_top_n_logprobs'):
                    assert (
                        request.prompt_top_n_logprobs is None
                        or len(request.prompt_top_n_logprobs) == 0
                    ), f"Request {request.request_id}: prompt_top_n_logprobs should be None or empty when skip_prompt_log_probs is True"

            # Validate consistency between log_probs and top_n_logprobs
            if hasattr(request, 'generated_log_probs') and request.generated_log_probs is not None:
                assert len(request.generated_log_probs) == len(
                    request.generated_top_n_logprobs
                ), f"Request {request.request_id}: generated_log_probs and generated_top_n_logprobs length mismatch"

                # Check that the selected token's log prob appears in the top-n
                for i, (log_prob, top_n_dict, token_id) in enumerate(
                    zip(
                        request.generated_log_probs,
                        request.generated_top_n_logprobs,
                        request.generated_tokens,
                    )
                ):
                    # Get the token string for this token_id
                    token_str = env.engine.controller.tokenizer.detokenize([token_id])
                    # The selected token should be in the top-n
                    assert (
                        token_str in top_n_dict
                    ), f"Request {request.request_id}, token {i}: selected token '{token_str}' not in top-n"
                    # The log prob should match (with some tolerance for floating point precision)
                    # Using 0.1 tolerance to account for FP16/BF16 precision in mixed precision training
                    assert (
                        abs(log_prob - top_n_dict[token_str]) < 0.1
                    ), f"Request {request.request_id}, token {i}: log_prob mismatch {log_prob} vs {top_n_dict[token_str]}"

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("max_requests", [None, 4])
    @torch.inference_mode()
    def test_max_requests(self, max_requests: int | None):
        """Test max requests."""
        env = self._run_test(
            context_max_requests=max_requests, num_tokens_to_generate=16, num_gap_steps=1
        )
        context = env.engine.context
        if max_requests is None:
            assert context.max_requests == 816
        else:
            assert max_requests < len(env.requests), (
                f"Test is only useful if max_requests ({max_requests}) < "
                f"num_requests ({len(env.requests)})."
            )
            assert context.max_requests == 4

        assert all(request.status == Status.COMPLETED for request in env.requests)
        generated_token_count = sum(len(request.generated_tokens) for request in env.requests)
        max_concurrent_requests = min(context.max_requests, len(env.requests))
        minimum_generation_steps = (
            generated_token_count + max_concurrent_requests - 1
        ) // max_concurrent_requests
        assert context.step_count >= minimum_generation_steps
        assert context.kv_block_allocator.pool_size == 819

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("static_kv_memory_pointers", [True, False])
    @pytest.mark.parametrize("kv_cache_management_mode", ["persist", "offload", "recompute"])
    @torch.inference_mode()
    def test_suspend_resume_cycle(self, kv_cache_management_mode, static_kv_memory_pointers):
        """Full suspend -> resume cycle with memory, data, and address checks."""
        needs_tms = static_kv_memory_pointers and kv_cache_management_mode != "persist"

        test_config = DynamicEngineTestConfig(
            kv_cache_management_mode=kv_cache_management_mode,
            static_kv_memory_pointers=static_kv_memory_pointers,
        )

        # Without TMS, these combos must assert on construction.
        if needs_tms and not HAVE_TORCH_MEMORY_SAVER:
            with pytest.raises(AssertionError, match="Static KV memory pointers"):
                self._build_test_env(test_config)
            return

        env = self._build_test_env(test_config)
        engine = env.engine
        context = engine.context

        assert engine.state != EngineState.SUSPENDED
        assert context.is_tensor_state_allocated

        deallocates = kv_cache_management_mode != "persist"
        uses_tms = context._uses_torch_memory_saver
        preserves_data = kv_cache_management_mode != "recompute"

        # Write a deterministic pattern for data integrity check.
        if preserves_data:
            context.memory_buffer.copy_(torch.randn_like(context.memory_buffer))
            expected = context.memory_buffer.clone()

        addr_before = context.memory_buffer.data_ptr()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        if uses_tms:
            phys_mem_before = torch.cuda.mem_get_info()[0]

        # Suspend.
        engine.suspend()
        assert engine.state == EngineState.SUSPENDED
        assert not context.is_tensor_state_allocated

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_suspended = torch.cuda.memory_allocated()
        if uses_tms:
            phys_mem_suspended = torch.cuda.mem_get_info()[0]

        if deallocates and not uses_tms:
            assert mem_suspended < mem_before, (
                f"GPU memory should decrease after suspend "
                f"(mode={kv_cache_management_mode}). "
                f"Before: {mem_before}, After: {mem_suspended}"
            )
        else:
            assert mem_suspended == mem_before, (
                f"Memory should not change on suspend. "
                f"Before: {mem_before}, Suspended: {mem_suspended}"
            )

        if uses_tms:
            assert phys_mem_suspended > phys_mem_before, (
                f"torch_memory_saver should free physical GPU memory after suspend. "
                f"Before: {phys_mem_before}, After: {phys_mem_suspended}"
            )

        # Resume.
        engine.resume()
        assert engine.state != EngineState.SUSPENDED
        assert context.is_tensor_state_allocated

        if deallocates and not uses_tms:
            torch.cuda.synchronize()
            mem_resumed = torch.cuda.memory_allocated()
            assert mem_resumed > mem_suspended, (
                f"GPU memory should increase after resume. "
                f"Suspended: {mem_suspended}, Resumed: {mem_resumed}"
            )

        if uses_tms:
            torch.cuda.synchronize()
            phys_mem_resumed = torch.cuda.mem_get_info()[0]
            assert phys_mem_resumed < phys_mem_suspended, (
                f"torch_memory_saver should re-allocate physical GPU memory after resume. "
                f"Suspended: {phys_mem_suspended}, Resumed: {phys_mem_resumed}"
            )

        # Data integrity.
        if preserves_data:
            torch.testing.assert_close(
                context.memory_buffer,
                expected,
                msg="memory_buffer data must be identical after suspend/resume",
            )

        # Address stability when CUDA graphs persist.
        if static_kv_memory_pointers:
            addr_after = context.memory_buffer.data_ptr()
            assert addr_before == addr_after, (
                f"Tensor address must be stable when static_kv_memory_pointers is set. "
                f"Before: {addr_before:#x}, After: {addr_after:#x}"
            )

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("use_checkpoint", [False, True], ids=["persist", "recompute"])
    @torch.inference_mode()
    def test_staleness_tracking(self, use_checkpoint):
        """Test that training-iteration stamps are correctly tracked.
        The use_checkpoint parameter simulates the behavior of different kv_cache_management_mode.
        """
        PROMPT_LEN = 8
        NUM_TOKENS = 8

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=PROMPT_LEN,
            max_prompt_length=PROMPT_LEN,
            num_tokens_to_generate=NUM_TOKENS,
        )
        env = self._build_test_env(test_config)
        engine = env.engine

        for i in range(2):
            prompt_tokens = torch.randint(
                0,
                test_config.vocab_size - 1,
                (PROMPT_LEN,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            engine._add_request(
                DynamicInferenceRequest(
                    request_id=i,
                    prompt_tokens=prompt_tokens,
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=NUM_TOKENS, termination_id=-1
                    ),
                )
            )

        def set_epoch(epoch):
            """Simulate receiving a SET_GENERATION_EPOCH signal."""
            engine._generation_epoch = epoch
            for entry in engine.requests.values():
                request = entry.record[-1]
                total = len(request.prompt_tokens) + len(request.generated_tokens)
                if total > 0:
                    boundary = (total - 1, epoch)
                    if request.policy_epoch is None:
                        request.policy_epoch = [(0, epoch)]
                    else:
                        request.policy_epoch.append(boundary)
                    if request.kv_cache_epoch is None:
                        request.kv_cache_epoch = [(0, epoch)]
                    else:
                        request.kv_cache_epoch.append(boundary)

        def capture_epoch_boundaries(epoch):
            """Capture the expected boundary at the instant an epoch is changed."""
            return {
                request_id: (
                    len(entry.record[-1].prompt_tokens)
                    + len(entry.record[-1].generated_tokens)
                    - 1,
                    epoch,
                )
                for request_id, entry in engine.requests.items()
            }

        # Steps without a generation epoch set — no stamps.
        engine.step_modern()
        for entry in engine.requests.values():
            assert entry.record[-1].policy_epoch is None
            assert entry.record[-1].kv_cache_epoch is None

        # Generation epoch 0: stamps all active requests at their current length.
        set_epoch(0)
        for _ in range(2):
            engine.step_modern()

        for entry in engine.requests.values():
            ps = entry.record[-1].policy_epoch
            ks = entry.record[-1].kv_cache_epoch
            assert ps == ks == [(0, 0)]

        # Generation epoch 1: boundary at current length, before next step.
        epoch_1_boundaries = capture_epoch_boundaries(1)
        set_epoch(1)
        for _ in range(3):
            engine.step_modern()

        for request_id, entry in engine.requests.items():
            ps = entry.record[-1].policy_epoch
            ks = entry.record[-1].kv_cache_epoch
            assert ps == ks == [(0, 0), epoch_1_boundaries[request_id]]

        # Simulate RECOMPUTE — checkpoint clears kv_cache so the engine's
        # stamping logic will recreate it fresh on the next epoch signal.
        if use_checkpoint:
            for entry in engine.requests.values():
                old_req = entry.record[-1]
                event_add_engine = old_req.event_add_engine
                entry.record.checkpoint()
                # Prevent TTFT crash due to missing _add_request in test.
                entry.record[-1].event_add_engine = event_add_engine

            for entry in engine.requests.values():
                assert entry.record[-1].kv_cache_epoch is None

        # Generation epoch 2: stamp then generate remaining tokens.
        epoch_2_boundaries = capture_epoch_boundaries(2)
        set_epoch(2)

        finished_records = []
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            finished_records.extend(result["finished_request_records"])

        for record in finished_records:
            merged = record.merge()
            expected_policy_epoch = [
                (0, 0),
                epoch_1_boundaries[merged.request_id],
                epoch_2_boundaries[merged.request_id],
            ]

            assert merged.policy_epoch == expected_policy_epoch

            if use_checkpoint:
                # KV cache was cleared by checkpoint; stamping logic recreated it at epoch 2.
                assert merged.kv_cache_epoch == [(0, 2)]
            else:
                assert merged.kv_cache_epoch == expected_policy_epoch

        # Verify checkpoint clears kv_cache_epoch and preserves policy.
        record = finished_records[0]
        request_id = record[-1].request_id
        expected_policy_epoch = [
            (0, 0),
            epoch_1_boundaries[request_id],
            epoch_2_boundaries[request_id],
        ]
        record.checkpoint()
        assert record[-1].policy_epoch == expected_policy_epoch
        assert record[-1].kv_cache_epoch is None

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_local_metadata_ledger_gated_by_engine_enable(self):
        """Only ledger-enabled engines (RL training) index finished requests."""
        PROMPT_LEN = 8
        NUM_TOKENS = 4

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=PROMPT_LEN,
            max_prompt_length=PROMPT_LEN,
            num_tokens_to_generate=NUM_TOKENS,
        )
        env = self._build_test_env(test_config)
        engine = env.engine
        engine._generation_epoch = 3  # RL mode: requests are epoch-stamped
        # Take the coordinator reply path so finished requests flow through the
        # inline ledger indexing in the reply block (socket mocked out).
        engine.use_coordinator = True
        engine.is_mp_coordinator = True
        engine.socket_for_receiving_requests = mock.MagicMock()

        def run_request(request_id):
            prompt_tokens = torch.full(
                (PROMPT_LEN,), request_id + 1, dtype=torch.int64, device=torch.cuda.current_device()
            )
            engine._add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=NUM_TOKENS, termination_id=-1
                    ),
                )
            )
            finished_records = []
            while engine.has_unfinished_requests():
                result = engine.step_modern()
                finished_records.extend(result["finished_request_records"])
            return finished_records

        # Default (plain serving): nothing is indexed, nothing accumulates.
        finished_records = run_request(0)
        assert len(finished_records) == 1
        assert engine.local_metadata_ledger == {}

        # RL launch (MegatronLocal.launch) enables the ledger: every finished
        # request is indexed, no per-request tagging involved.
        engine.local_metadata_ledger_enabled = True
        finished_records = run_request(1)
        assert len(finished_records) == 1

        # The ledger keys by the request's engine-minted uid — the same string the
        # endpoints return as the OpenAI response id.
        merged = finished_records[0].merge()
        ledger = engine.local_metadata_ledger
        assert list(ledger.keys()) == [merged.uid]
        assert ledger[merged.uid].policy_epoch == [(0, 3)]

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_speculative_decoding_with_early_termination(
        self, materialize_only_last_token_logits, sampling_backend
    ):
        """Test that speculative decoding handles premature request termination safely
        (e.g. hitting max_sequence_length mid-speculative-batch)."""
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")

        # Set max_sequence_length tight so it terminates during a speculative step
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=3,  # Prompt (4) + Gen (3) = 7
            max_sequence_length=7,  # Will force termination after 3 tokens
            model_provider="gpt",
            num_speculative_tokens=3,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            sampling_backend=sampling_backend,
        )

        env = self._build_test_env(test_config)
        unwrapped_model = env.engine.controller.inference_wrapped_model.model

        # Mock forward to return deterministic data so speculative tokens are always accepted
        hidden_size = unwrapped_model.config.hidden_size

        def mock_mtp_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))

            base_logits = torch.zeros(
                tokens.size(0),
                tokens.size(1),
                test_config.vocab_size,
                device=tokens.device,
                dtype=torch.bfloat16,
            )
            base_logits[:, :, 0] = 100.0  # High probability for token 0

            # Cache hidden states for serial MTP computation
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                tokens.size(1), 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            if test_config.materialize_only_last_token_logits:
                base_logits = env.engine.context.last_token_logits(base_logits).unsqueeze(0)
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0  # High probability for token 0
            return hidden_states, logits

        unwrapped_model.forward = mock_mtp_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine._add_request(env.requests[0])

        # Step engine until finished naturally
        # This allows the bookkeeping logic to gracefully truncate the
        # speculative tokens to the max_sequence_length boundary.
        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        assert env.requests[0].status == Status.COMPLETED

        # It should trim the output to the max_sequence_length boundary
        # Prompt was 4, Max was 7, so it should have generated exactly 3 tokens.
        assert len(env.requests[0].generated_tokens) == 3

        # Validate the engine's tracking state is clean
        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

    @pytest.mark.internal
    @torch.inference_mode()
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_speculative_block_boundary_crossing(
        self, materialize_only_last_token_logits, sampling_backend
    ):
        """Test to verify KV cache block boundary crossing logic.

        When a request fills exactly one block and speculative decoding generates
        multiple tokens, the first new token shouldn't incorrectly overwrite the old block.
        """
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=256,
            max_prompt_length=256,
            num_tokens_to_generate=3,
            num_speculative_tokens=2,
            context_block_size_tokens=256,  # Exactly matches prompt length
            context_max_requests=16,
            model_provider="gpt",
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            use_fixed_output_lengths=True,
            sampling_backend=sampling_backend,
        )
        env = self._build_test_env(test_config)

        req = env.requests[0]
        req.sampling_params.num_tokens_to_generate = 3
        env.engine._add_request(req)

        # Async scheduling primes the prefill forward on the first call, then
        # consumes it and prepares the first speculative decode on the second.
        env.engine.step_modern()
        env.engine.step_modern()

        context = env.engine.context

        # The request has 2 blocks allocated now (1 for prompt, 1 for the new 3 tokens)
        assigned_blocks = context.request_to_kv_block_ids[0]
        second_block = assigned_blocks[1].item()

        # The active_token_count for the next step should be 3
        assert context.active_token_count == 3

        # Check which blocks the 3 new tokens are assigned to.
        # Because the prompt exactly filled the first block, ALL 3 new tokens
        # MUST go to the second block.
        token_blocks = context.token_to_block_idx[: context.active_token_count].tolist()

        assert token_blocks == [
            second_block,
            second_block,
            second_block,
        ], f"Expected all new tokens to go to block {second_block}, but got {token_blocks}."

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_speculative_stop_word_hit(self, materialize_only_last_token_logits, sampling_backend):
        """Test that if an accepted speculative token completes a stop word,
        the request correctly triggers the stop logic without crashing."""
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")

        test_config = DynamicEngineTestConfig(
            num_requests=0,  # We will manually add our request cleanly
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            model_provider="gpt",
            sampling_backend=sampling_backend,
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward to deterministically output an ascending sequence (1->2->3...)
        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

            # Cache hidden states for serial MTP computation
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            if test_config.materialize_only_last_token_logits:
                base_logits = env.engine.context.last_token_logits(base_logits).unsqueeze(0)
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            # Predict next_token_ids + 1 (continuing the ascending sequence)
            pred_toks = (next_token_ids + 1).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, pred_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # Add the request formally to ensure all internal state tensors align
        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=10, termination_id=99, detokenize_stop_sequence=True
            ),
        )

        # Inject the parsed stop word IDs
        tracked_req = env.engine.get_request(0)
        tracked_req.stop_word_ids = [[8, 9]]  # The sequence will generate 5, 6, 7, 8, 9, ...

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        # Retrieve the finalized request from the engine's output
        finished_req = finished_records[0].merge()

        assert finished_req.status == Status.COMPLETED
        # Since num_tokens_to_generate=10, output should stop early at ~7 tokens
        assert len(finished_req.generated_tokens) < 10
        # Verify the stop word was actually generated and caused the termination
        token_pairs = [
            finished_req.generated_tokens[i : i + 2]
            for i in range(len(finished_req.generated_tokens) - 1)
        ]
        assert [8, 9] in token_pairs

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_speculative_long_stop_word_hit(
        self, materialize_only_last_token_logits, sampling_backend
    ):
        """Test that if an accepted speculative token completes a long stop word
        (length > num_speculative_tokens), it is correctly detected."""
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            model_provider="gpt",
            sampling_backend=sampling_backend,
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward to deterministically output an ascending sequence
        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

            # Cache hidden states for serial MTP computation
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            if test_config.materialize_only_last_token_logits:
                base_logits = env.engine.context.last_token_logits(base_logits).unsqueeze(0)
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            # Predict next_token_ids + 1 (continuing the ascending sequence)
            pred_toks = (next_token_ids + 1).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, pred_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=10, termination_id=99, detokenize_stop_sequence=True
            ),
        )

        # Stop word length 3 > num_speculative_tokens (2)
        tracked_req = env.engine.get_request(0)
        tracked_req.stop_word_ids = [[7, 8, 9]]

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        finished_req = finished_records[0].merge()

        assert finished_req.status == Status.COMPLETED
        assert len(finished_req.generated_tokens) < 10
        token_triplets = [
            finished_req.generated_tokens[i : i + 3]
            for i in range(len(finished_req.generated_tokens) - 2)
        ]
        assert [7, 8, 9] in token_triplets

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_speculative_stop_word_truncates_trailing_tokens(
        self, materialize_only_last_token_logits, sampling_backend
    ):
        """Test that when a stop word lands in the middle of speculative tokens,
        the extra tokens generated after the stop word are removed.

        With num_speculative_tokens=2, each step produces up to 3 tokens
        (1 base + 2 speculative). If the stop word is [6] and the engine
        generates [5, 6, 7] in one step, token 7 must be truncated so the
        output ends with the stop word [6]."""
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            model_provider="gpt",
            sampling_backend=sampling_backend,
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward to deterministically output an ascending sequence (1->2->3...)
        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

            # Cache hidden states for serial MTP computation
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            if test_config.materialize_only_last_token_logits:
                base_logits = env.engine.context.last_token_logits(base_logits).unsqueeze(0)
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            # Predict next_token_ids + 1 (continuing the ascending sequence)
            pred_toks = (next_token_ids + 1).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, pred_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=10, termination_id=99, detokenize_stop_sequence=True
            ),
        )

        # Stop word [6] will land in the middle of a speculative batch [5, 6, 7].
        # Token 7 should be truncated from the output.
        tracked_req = env.engine.get_request(0)
        tracked_req.stop_word_ids = [[6]]

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        finished_req = finished_records[0].merge()

        assert finished_req.status == Status.COMPLETED
        # The output should end exactly at the stop word, with no trailing tokens.
        assert finished_req.generated_tokens[-1] == 6, (
            f"Expected last token to be stop word 6, "
            f"got {finished_req.generated_tokens[-1]}. "
            f"Trailing tokens after stop word were not truncated. "
            f"Full output: {finished_req.generated_tokens}"
        )
        # Verify no tokens after the stop word exist
        assert 7 not in finished_req.generated_tokens, (
            f"Token 7 should have been truncated after stop word 6. "
            f"Full output: {finished_req.generated_tokens}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "prompt_length,num_tokens_to_generate,num_speculative_tokens",
        [
            # Generates 5 tokens with spec=3 → step produces 4 tokens at a time.
            # After prefill, first decode step emits 4 tokens (seq 4→8), second step
            # would push to 12 but only 1 more is needed → must trim to exactly 5.
            (4, 5, 3),
            # Generates 2 tokens with spec=3 → step produces 4 tokens at a time.
            # After prefill, single decode step would emit 4 tokens but only 2 are
            # needed → must trim aggressively on the very first decode.
            (4, 2, 3),
            # Generates 1 token with spec=2 → step produces 3 tokens at a time.
            # Only 1 token is needed; the 2 speculative tokens must be discarded.
            (4, 1, 2),
            # Generates 7 tokens with spec=2 → step produces 3 tokens at a time.
            # 7 is not divisible by 3, so the final step must trim the excess token.
            (4, 7, 2),
        ],
        ids=[
            "overshoot_second_step",
            "overshoot_first_step",
            "single_token_generation",
            "non_divisible_boundary",
        ],
    )
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    @torch.inference_mode()
    def test_speculative_tokens_exceed_max_sequence_length(
        self,
        prompt_length,
        num_tokens_to_generate,
        num_speculative_tokens,
        materialize_only_last_token_logits,
        sampling_backend,
    ):
        """Test that speculative decoding correctly trims output when speculative
        tokens would push the sequence beyond max_sequence_length.

        Exercises the real model forward pass (attention, KV cache, MTP layers)
        but substitutes deterministic logits after the forward to ensure all
        speculative tokens are accepted and the boundary trimming logic is
        actually exercised.
        """
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")
        max_sequence_length = prompt_length + num_tokens_to_generate

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=prompt_length,
            max_prompt_length=prompt_length,
            num_tokens_to_generate=num_tokens_to_generate,
            max_sequence_length=max_sequence_length,
            num_speculative_tokens=num_speculative_tokens,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            model_provider="gpt",
            # Disable positional embeddings so speculative position IDs
            # beyond max_sequence_length don't cause out-of-bounds lookups.
            position_embedding_type="none",
            sampling_backend=sampling_backend,
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Wrap the real forward: run the actual model (attention, KV cache, etc.)
        # then replace the output logits with deterministic values so all
        # speculative tokens are accepted and sampling is predictable.
        real_forward = unwrapped_model.forward

        def deterministic_forward(*args, **kwargs):
            logits = real_forward(*args, **kwargs)
            # Overwrite with deterministic logits: always predict token 0.
            logits.zero_()
            logits[..., 0] = 100.0
            return logits

        # Wrap the real MTP step similarly.
        real_mtp = unwrapped_model.compute_mtp_single_step

        def deterministic_mtp(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            hidden_states, logits = real_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
            )
            logits.zero_()
            logits[..., 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = deterministic_forward
        unwrapped_model.compute_mtp_single_step = deterministic_mtp

        prompt = torch.zeros(prompt_length, dtype=torch.int64, device='cuda')
        env.engine.add_request(
            request_id=0,
            prompt=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=num_tokens_to_generate,
                termination_id=test_config.vocab_size - 1,  # Won't trigger naturally
            ),
        )

        finished_records = []
        step_count = 0
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])
            step_count += 1
            assert step_count < 100, "Engine did not converge"

        assert len(finished_records) == 1
        finished_req = finished_records[0].merge()

        assert (
            finished_req.status == Status.COMPLETED
        ), f"Expected COMPLETED, got {finished_req.status}"
        assert len(finished_req.generated_tokens) == num_tokens_to_generate, (
            f"Expected exactly {num_tokens_to_generate} generated tokens, "
            f"got {len(finished_req.generated_tokens)}. "
            f"Speculative tokens were not correctly trimmed at the "
            f"max_sequence_length boundary. "
            f"Output: {finished_req.generated_tokens}"
        )

        # All tokens should be 0 (the deterministic prediction).
        assert all(
            t == 0 for t in finished_req.generated_tokens
        ), f"Expected all tokens to be 0, got {finished_req.generated_tokens}"

        # Verify engine state is clean after completion.
        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

    @pytest.mark.parametrize("detokenize_stop_sequence", [True, False])
    def test_detokenize_stop_sequence_flag(self, detokenize_stop_sequence):
        """Test that _check_stop_words_for_request_post_append strips or keeps
        the stop word tokens based on detokenize_stop_sequence."""
        engine = types.SimpleNamespace(num_speculative_tokens=0)
        check = DynamicInferenceEngine._check_stop_words_for_request_post_append

        request = types.SimpleNamespace(
            generated_tokens=[1, 2, 3, 4, 5],
            stop_word_ids=[[4, 5]],
            sampling_params=SamplingParams(detokenize_stop_sequence=detokenize_stop_sequence),
        )
        hit, trimmed = check(engine, request)
        assert hit
        if detokenize_stop_sequence:
            # Stop word kept
            assert request.generated_tokens == [1, 2, 3, 4, 5]
            assert trimmed == 0
        else:
            # Stop word stripped
            assert request.generated_tokens == [1, 2, 3]
            assert trimmed == 2

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "acceptance_mode", ["all_rejected", "all_accepted"], ids=["all_rejected", "all_accepted"]
    )
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    @torch.inference_mode()
    def test_speculative_sequence_length_double_counting(
        self, acceptance_mode, materialize_only_last_token_logits, sampling_backend
    ):
        """Test to verify active_sequence_lengths is not double-counted.

        If active sequence length is double-counted during speculative decoding,
        the request will terminate prematurely before generating the requested tokens.

        The 'all_rejected' variant exercises the path where accepted_tokens=0
        (the + 1 term in the bookkeeping is correct by itself).
        The 'all_accepted' variant is the critical case: with accepted_tokens=2,
        a faulty formula that adds accepted_tokens on top of the KV length will
        over-count by 2 per step, finishing the request after only 4 of 6 tokens.
        """
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=6,
            max_sequence_length=10,  # Exactly prompt (4) + generate (6)
            context_max_requests=16,
            num_speculative_tokens=2,
            model_provider="gpt",
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            use_fixed_output_lengths=False,
            context_max_tokens=512,
            position_embedding_type="none",
            sampling_backend=sampling_backend,
        )
        env = self._build_test_env(test_config)

        model = env.engine.controller.inference_wrapped_model.model
        hidden_size = model.config.hidden_size

        if acceptance_mode == "all_rejected":
            # Mock forward pass to return deterministic base logits.
            # Speculative tokens will be wrong (predicted by MTP as tokens + 5)
            # to guarantee rejection every time.
            def mock_mtp_forward(*args, **kwargs):
                tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
                b, s = tokens.shape

                base_logits = torch.zeros(
                    b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
                )
                next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
                base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)

                env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                    s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
                )
                if test_config.materialize_only_last_token_logits:
                    base_logits = env.engine.context.last_token_logits(base_logits).unsqueeze(0)
                return base_logits

            def mock_compute_mtp(*args_mtp, **kwargs_mtp):
                hidden_states = args_mtp[0] if args_mtp else kwargs_mtp["hidden_states"]
                next_token_ids = args_mtp[1] if len(args_mtp) > 1 else kwargs_mtp["next_token_ids"]
                n = hidden_states.size(0)
                wrong_toks = (next_token_ids + 5).clamp(max=test_config.vocab_size - 1)
                logits = torch.zeros(
                    n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
                )
                logits.scatter_(2, wrong_toks.transpose(0, 1).unsqueeze(-1), 100.0)
                return hidden_states, logits

            model.forward = mock_mtp_forward
            model.compute_mtp_single_step = mock_compute_mtp
        else:
            # Wrap real forward and MTP: run the actual model, then overwrite
            # logits so both base and MTP predict token 0 → all accepted.
            real_forward = model.forward

            def deterministic_forward(*args, **kwargs):
                logits = real_forward(*args, **kwargs)
                logits.zero_()
                logits[..., 0] = 100.0
                return logits

            real_mtp = model.compute_mtp_single_step

            def deterministic_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
            ):
                hidden_states, logits = real_mtp(
                    hidden_states,
                    next_token_ids,
                    position_ids,
                    depth,
                    eager=eager,
                    cache_key=cache_key,
                )
                logits.zero_()
                logits[..., 0] = 100.0
                return hidden_states, logits

            model.forward = deterministic_forward
            model.compute_mtp_single_step = deterministic_mtp

        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=6, termination_id=99),
        )

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        finished_req = finished_records[0].merge()

        # If there is double counting, the tracked active length will outpace the actual
        # generated tokens, causing premature termination when it thinks it hit max_sequence_length.
        assert finished_req.status == Status.COMPLETED
        assert (
            len(finished_req.generated_tokens) == 6
        ), f"Expected 6 tokens, got {len(finished_req.generated_tokens)}. Double counting occurred."

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("sampling_backend", ["torch", "flashinfer"])
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_speculative_decoding_with_eviction_and_swapping(
        self, materialize_only_last_token_logits, sampling_backend
    ):
        """Test that speculative decoding works correctly when requests are paused and evicted.

        This exercises the `_swap_book_keeping_tensors` logic with the 2D `new_speculative_tokens`
        tensor, ensuring no dimensional mismatch or index errors occur during tensor swapping.
        """
        if sampling_backend == "flashinfer":
            pytest.importorskip("flashinfer")
        # Very constrained memory environment to force pausing and eviction
        test_config = DynamicEngineTestConfig(
            num_requests=3,
            min_prompt_length=256,
            max_prompt_length=256,
            num_tokens_to_generate=512,
            context_block_size_tokens=256,
            num_speculative_tokens=2,
            context_buffer_size_gb=0.00064,  # 640 KB
            context_paused_buffer_size_gb=0.0,  # 0 paused buffer forces immediate eviction
            model_provider="gpt",
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            use_fixed_output_lengths=True,
            sampling_backend=sampling_backend,
        )

        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward pass to return safe, deterministic logits to avoid NaN/Inf crashes
        # in torch.multinomial caused by randomly initialized weights.
        def mock_safe_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape

            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            base_logits[:, :, 0] = 100.0  # Force model to deterministically pick token 0

            # Cache hidden states for serial MTP computation
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            if test_config.materialize_only_last_token_logits:
                base_logits = env.engine.context.last_token_logits(base_logits).unsqueeze(0)
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0  # Force speculative heads to also pick token 0
            return hidden_states, logits

        unwrapped_model.forward = mock_safe_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # Add all requests at once. They will all start prefill, but as they generate
        # and request more blocks, the engine will run out of active blocks.
        # Since paused_buffer_size is 0, any request that pauses will immediately
        # overflow the paused buffer and trigger an eviction.
        for request in env.requests:
            request.sampling_params.num_tokens_to_generate = 512
            env.engine._add_request(request)

        eviction_occurred = False

        # Step the engine manually until all requests finish.
        while env.engine.has_unfinished_requests():
            # Record the number of evicted requests before the step
            evicted_before = env.engine.evicted_request_count

            # Step the engine
            env.engine.step_modern()

            # Check if any request was evicted during this step
            if env.engine.evicted_request_count > evicted_before:
                eviction_occurred = True

        # Assert that our constrained memory actually caused an eviction,
        # proving we exercised the evict_overflow_paused_requests path with spec tokens.
        assert (
            eviction_occurred
        ), "Test failed to trigger an eviction. The test environment memory wasn't tight enough."

        # Verify all requests successfully went back through the queue and finished cleanly.
        # We MUST check the merged records from the engine, because eviction checkpoints
        # the requests, leaving the original instances in env.requests permanently active.
        for request_id, entry in env.engine.requests.items():
            merged_req = entry.record.merge()
            assert (
                merged_req.status == Status.COMPLETED
            ), f"Request {request_id} failed to complete."
            assert (
                len(merged_req.generated_tokens) == 511
            ), f"Request {request_id} didn't generate expected tokens."

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_with_prefix_caching(self):
        """Test that speculative decoding works correctly when prefix caching is enabled.

        Two requests share the same prompt prefix. The second request should reuse
        cached KV blocks from the first and still generate correctly with spec decoding.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,  # Added manually below
            min_prompt_length=256,
            max_prompt_length=256,
            num_tokens_to_generate=4,
            num_speculative_tokens=2,
            enable_prefix_caching=True,  # Set at config level
            context_block_size_tokens=256,  # Ensure exact 1 block per prompt
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            context_max_tokens=4096,
            context_max_requests=512,
            max_sequence_length=1024,
        )
        env = self._build_test_env(test_config)

        # Create two pairs of requests with identical shared prefixes.
        shared_prompt_a = torch.randint(
            0, test_config.vocab_size - 1, (256,), dtype=torch.int64, device='cuda'
        )
        shared_prompt_b = torch.randint(
            0, test_config.vocab_size - 1, (256,), dtype=torch.int64, device='cuda'
        )

        prompts = [shared_prompt_a, shared_prompt_a, shared_prompt_b, shared_prompt_b]

        for i, prompt in enumerate(prompts):
            # Using the clean public API guarantees correct hashing and dataclass creation
            env.engine.add_request(
                request_id=i,
                prompt=prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=128, termination_id=99),
            )

        # The first async call admits Req 0 and 2 and primes their prefill forward.
        # Req 1 and 3 remain queued while the matching prefix blocks are pending.
        env.engine.step_modern()

        # The second call consumes those prefills, admits the deferred requests using
        # the now-populated prefix blocks, and launches the resulting mixed batch.
        env.engine.step_modern()

        # 4 requests. 2 unique prefixes (1 block each).
        # Without sharing, we'd need 8 blocks + 1 dummy = 9 active_used.
        # With sharing, we need 2 shared blocks + 4 generation blocks + 1 dummy = 7 active_used.
        active_used = env.engine.context.kv_block_allocator.get_active_used()
        assert (
            active_used <= 7
        ), f"Prefix caching failed, expected <= 7 active blocks but got {active_used}"

        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        # Context should be clean after all requests finish.
        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_chunked_prefill_and_prefix_caching(self):
        """End-to-end test combining speculative decoding, chunked prefill, and prefix caching.

        Verifies that all three features interact correctly:
        - Prefix caching shares KV blocks between requests with common prompts
        - Chunked prefill processes long prompts in chunks
        - Speculative decoding generates multiple tokens per step
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=512,
            max_prompt_length=512,
            num_tokens_to_generate=128,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,  # Set at config level
            context_block_size_tokens=256,
            model_provider="gpt",
            context_max_tokens=1536,  # Force chunking
            context_max_requests=48,
        )
        env = self._build_test_env(test_config)

        # Create identical prompts for all 4 requests
        shared_prompt = torch.randint(
            0, test_config.vocab_size - 1, (512,), dtype=torch.int64, device='cuda'
        )

        for i in range(4):
            env.engine.add_request(
                request_id=i,
                prompt=shared_prompt.clone(),
                sampling_params=SamplingParams(num_tokens_to_generate=128, termination_id=99),
            )

        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_stats_exclude_prefill(self):
        """Test that MTP acceptance stats are cumulative and exclude prefill requests.

        Prefill requests don't get MTP speculative proposals (MTP heads only run for
        decode requests). Verify that:
        1. Stats accumulate across the engine lifetime (no reset between logging).
        2. Prefill steps don't inflate _spec_tokens_proposed.
        3. The acceptance rate reflects only decode steps.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,  # Added manually below to stagger prefill vs decode
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)
        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward: all tokens get the same high-probability logit so every
        # speculative token is accepted (acceptance rate should be 100%).
        def mock_mtp_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            base_logits = torch.zeros(
                tokens.size(0),
                tokens.size(1),
                test_config.vocab_size,
                device=tokens.device,
                dtype=torch.bfloat16,
            )
            base_logits[:, :, 0] = 100.0
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                tokens.size(1), 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = mock_mtp_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # Verify counters start at zero.
        assert sum(env.engine._spec_tokens_proposed_per_pos) == 0
        assert sum(env.engine._spec_tokens_accepted_per_pos) == 0
        assert env.engine._spec_steps == 0

        # Add first request and run through prefill + some decode steps.
        env.engine.add_request(
            request_id=0,
            prompt=torch.randint(
                0, test_config.vocab_size - 1, (4,), dtype=torch.int64, device='cuda'
            ),
            sampling_params=SamplingParams(num_tokens_to_generate=10, termination_id=-1),
        )

        # The first async call only primes request 0's prefill forward.
        env.engine.step_modern()
        assert sum(env.engine._spec_tokens_proposed_per_pos) == 0
        assert sum(env.engine._spec_tokens_accepted_per_pos) == 0
        assert env.engine._spec_steps == 0

        # The second call consumes the prefill and launches the first decode.
        # The prefill result has accepted_tokens=None, so it must not increment stats.
        env.engine.step_modern()
        proposed_after_prefill = sum(env.engine._spec_tokens_proposed_per_pos)
        accepted_after_prefill = sum(env.engine._spec_tokens_accepted_per_pos)
        assert proposed_after_prefill == 0, "Prefill step should not propose any spec tokens"
        assert accepted_after_prefill == 0, "Prefill step should not accept any spec tokens"
        assert env.engine._spec_steps == 0, "Prefill step should not count as a spec step"

        # Consume the first decode for request 0; it should count spec tokens.
        env.engine.step_modern()
        assert (
            sum(env.engine._spec_tokens_proposed_per_pos) > proposed_after_prefill
        ), "Decode step should have incremented _spec_tokens_proposed_per_pos"
        assert (
            sum(env.engine._spec_tokens_accepted_per_pos) > accepted_after_prefill
        ), "With deterministic mock, decode step should have accepted spec tokens"

        # Now add a second request while request 0 is decoding. This makes the
        # next call launch a mixed prefill (req 1) + decode (req 0) batch.
        env.engine.add_request(
            request_id=1,
            prompt=torch.randint(
                0, test_config.vocab_size - 1, (4,), dtype=torch.int64, device='cuda'
            ),
            sampling_params=SamplingParams(num_tokens_to_generate=10, termination_id=-1),
        )

        # This call consumes the already-pending decode-only batch, then admits
        # request 1 and launches a mixed prefill/decode batch.
        env.engine.step_modern()

        proposed_before_mixed = sum(env.engine._spec_tokens_proposed_per_pos)
        # Consume the mixed batch. Only request 0 was decoding in that batch.
        env.engine.step_modern()
        proposed_after_mixed = sum(env.engine._spec_tokens_proposed_per_pos)

        # In the mixed step, only the decode request (req 0) should contribute to
        # proposed count, NOT the prefilling request (req 1). With 2 spec tokens and
        # 1 decode request, proposed should increase by exactly 2.
        proposed_delta = proposed_after_mixed - proposed_before_mixed
        assert proposed_delta == test_config.num_speculative_tokens, (
            f"Mixed prefill+decode step: expected proposed delta of "
            f"{test_config.num_speculative_tokens} (1 decode request), got {proposed_delta}"
        )

        # Run to completion.
        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        # Stats should be cumulative (non-zero after all requests finish).
        total_proposed = sum(env.engine._spec_tokens_proposed_per_pos)
        total_accepted = sum(env.engine._spec_tokens_accepted_per_pos)
        assert total_proposed > 0
        assert total_accepted > 0
        assert env.engine._spec_steps > 0

        # With deterministic mock (all tokens accepted), acceptance rate should be 100%.
        acceptance_rate = total_accepted / total_proposed
        assert (
            acceptance_rate == 1.0
        ), f"Expected 100% acceptance with deterministic mock, got {acceptance_rate * 100:.1f}%"

        # With deterministic mock, every position should have 100% acceptance.
        for pos in range(test_config.num_speculative_tokens):
            assert (
                env.engine._spec_tokens_proposed_per_pos[pos] > 0
            ), f"Position {pos} should have proposals"
            pos_rate = (
                env.engine._spec_tokens_accepted_per_pos[pos]
                / env.engine._spec_tokens_proposed_per_pos[pos]
            )
            assert pos_rate == 1.0, (
                f"Expected 100% acceptance at position {pos} with deterministic mock, "
                f"got {pos_rate * 100:.1f}%"
            )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @torch.inference_mode()
    def test_speculative_decoding_logprobs(self, skip_prompt_log_probs: bool):
        """Test that log probabilities are correctly computed with speculative decoding.

        Verifies:
        1. generated_log_probs are returned for all generated tokens (including accepted
           speculative tokens and newly sampled tokens).
        2. prompt_log_probs respect the skip_prompt_log_probs flag.
        3. The number of log probs matches the number of generated tokens.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=8,
            num_tokens_to_generate=8,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Mock forward to return deterministic logits so speculative tokens are accepted.
        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape
            base_logits = torch.randn(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            # Make token 0 very likely so speculative tokens get accepted.
            base_logits[:, :, 0] = 100.0
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            logits = torch.randn(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # Add requests with log probs enabled and varying prompt lengths.
        num_requests = 3
        prompt_lengths = [4, 6, 8]
        for i in range(num_requests):
            prompt = torch.randint(
                0, test_config.vocab_size - 1, (prompt_lengths[i],), device='cuda'
            )
            env.engine.add_request(
                request_id=i,
                prompt=prompt,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=8,
                    termination_id=test_config.vocab_size - 1,
                    return_log_probs=True,
                    skip_prompt_log_probs=skip_prompt_log_probs,
                    top_k=1,
                ),
            )

        # Run to completion.
        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        assert len(finished_records) == num_requests

        for record in finished_records:
            req = record.merge()
            assert (
                req.status == Status.COMPLETED
            ), f"Request {req.request_id} not completed: {req.status}"

            # Generated log probs should match generated token count.
            assert (
                req.generated_log_probs is not None
            ), f"Request {req.request_id}: generated_log_probs is None"
            assert len(req.generated_log_probs) == len(req.generated_tokens), (
                f"Request {req.request_id}: log probs count {len(req.generated_log_probs)} "
                f"!= token count {len(req.generated_tokens)}"
            )

            # All log probs should be valid floats (negative, since they're log probabilities).
            # With logit=100 for the chosen token, log_softmax is very close to 0.
            for j, lp in enumerate(req.generated_log_probs):
                assert isinstance(
                    lp, float
                ), f"Request {req.request_id}, token {j}: log prob is not float"
                assert -0.1 < lp <= 0.0, (
                    f"Request {req.request_id}, token {j}: "
                    f"expected log prob near 0.0 (high confidence), got {lp}"
                )

            # Prompt log probs check.
            prompt_length = prompt_lengths[req.request_id]
            if skip_prompt_log_probs:
                assert req.prompt_log_probs is None or len(req.prompt_log_probs) == 0
            else:
                assert req.prompt_log_probs is not None
                assert len(req.prompt_log_probs) == prompt_length - 1, (
                    f"Request {req.request_id}: expected {prompt_length - 1} "
                    f"prompt log probs, got {len(req.prompt_log_probs)}"
                )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("skip_prompt_log_probs", [True, False])
    @torch.inference_mode()
    def test_speculative_decoding_top_n_logprobs(self, skip_prompt_log_probs: bool):
        """Test that top-N log probabilities are correctly computed with speculative decoding.

        Verifies:
        1. generated_top_n_logprobs are returned for all generated tokens.
        2. Each top-n dict has the expected number of entries (<= top_n).
        3. The selected token appears in the top-n with a matching log prob value.
        4. prompt_top_n_logprobs respect the skip_prompt_log_probs flag.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=8,
            num_tokens_to_generate=6,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        # Override detokenize to produce unique strings per token ID so the
        # top-n dict doesn't collapse all entries to a single key.
        env.engine.controller.tokenizer.detokenize = lambda tokens, **kw: f"tok_{tokens[0]}"

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape
            base_logits = torch.randn(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            base_logits[:, :, 0] = 100.0
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            logits = torch.randn(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # top_n must be >= top_k so the sampled token is guaranteed to appear
        # in the top-n dict for the consistency check below.
        top_n = 10
        num_requests = 3
        prompt_lengths = [4, 6, 8]
        for i in range(num_requests):
            prompt = torch.randint(
                0, test_config.vocab_size - 1, (prompt_lengths[i],), device='cuda'
            )
            env.engine.add_request(
                request_id=i,
                prompt=prompt,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=6,
                    termination_id=test_config.vocab_size - 1,
                    return_log_probs=True,
                    top_n_logprobs=top_n,
                    skip_prompt_log_probs=skip_prompt_log_probs,
                    top_k=1,
                ),
            )

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        assert len(finished_records) == num_requests

        for record in finished_records:
            req = record.merge()
            assert req.status == Status.COMPLETED

            # Validate generated top-n logprobs.
            assert (
                req.generated_top_n_logprobs is not None
            ), f"Request {req.request_id}: generated_top_n_logprobs is None"
            assert len(req.generated_top_n_logprobs) == len(req.generated_tokens), (
                f"Request {req.request_id}: top-n count {len(req.generated_top_n_logprobs)} "
                f"!= token count {len(req.generated_tokens)}"
            )

            for j, top_n_dict in enumerate(req.generated_top_n_logprobs):
                assert isinstance(
                    top_n_dict, dict
                ), f"Request {req.request_id}, token {j}: top_n_dict is not a dict"
                assert 0 < len(top_n_dict) <= top_n, (
                    f"Request {req.request_id}, token {j}: "
                    f"top-n has {len(top_n_dict)} entries, expected 1..{top_n}"
                )

            # Validate consistency: selected token's log prob should appear in top-n.
            if req.generated_log_probs is not None:
                for j, (lp, top_n_dict, token_id) in enumerate(
                    zip(req.generated_log_probs, req.generated_top_n_logprobs, req.generated_tokens)
                ):
                    token_str = env.engine.controller.tokenizer.detokenize([token_id])
                    assert token_str in top_n_dict, (
                        f"Request {req.request_id}, token {j}: "
                        f"selected token '{token_str}' not in top-n keys {list(top_n_dict.keys())}"
                    )
                    assert abs(lp - top_n_dict[token_str]) < 0.01, (
                        f"Request {req.request_id}, token {j}: "
                        f"log_prob {lp} vs top-n {top_n_dict[token_str]}"
                    )

            # Validate prompt top-n logprobs.
            if not skip_prompt_log_probs:
                assert (
                    req.prompt_top_n_logprobs is not None
                ), f"Request {req.request_id}: prompt_top_n_logprobs is None"
                assert len(req.prompt_top_n_logprobs) > 0
                for j, top_n_dict in enumerate(req.prompt_top_n_logprobs):
                    assert isinstance(top_n_dict, dict)
                    assert 0 < len(top_n_dict) <= top_n
            else:
                assert req.prompt_top_n_logprobs is None or len(req.prompt_top_n_logprobs) == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_top_n_logprobs_mixed_top_n(self):
        """Test speculative decoding with different top_n_logprobs values per request.

        Verifies:
        1. Requests with top_n=0 do not receive top-n logprobs.
        2. Requests with different top_n values get the correct number of entries.
        3. Mixed top_n values in the same batch work correctly.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=4,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        env.engine.controller.tokenizer.detokenize = lambda tokens, **kw: f"tok_{tokens[0]}"

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape
            base_logits = torch.randn(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            base_logits[:, :, 0] = 100.0
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            logits = torch.randn(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits[:, :, 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        # Request 0: top_n=3, Request 1: top_n=0 (no top-n), Request 2: top_n=10.
        top_n_values = [3, 0, 10]
        for i in range(3):
            prompt = torch.randint(0, test_config.vocab_size - 1, (4,), device='cuda')
            env.engine.add_request(
                request_id=i,
                prompt=prompt,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=4,
                    termination_id=test_config.vocab_size - 1,
                    return_log_probs=True,
                    top_n_logprobs=top_n_values[i],
                    top_k=1,
                ),
            )

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        assert len(finished_records) == 3

        for record in finished_records:
            req = record.merge()
            assert req.status == Status.COMPLETED
            req_top_n = top_n_values[req.request_id]

            if req_top_n == 0:
                assert req.generated_top_n_logprobs is None or all(
                    d is None or len(d) == 0 for d in (req.generated_top_n_logprobs or [])
                ), f"Request {req.request_id}: should have no top-n logprobs"
            else:
                assert req.generated_top_n_logprobs is not None
                assert len(req.generated_top_n_logprobs) == len(req.generated_tokens)
                for j, top_n_dict in enumerate(req.generated_top_n_logprobs):
                    assert isinstance(top_n_dict, dict)
                    assert 0 < len(top_n_dict) <= req_top_n, (
                        f"Request {req.request_id}, token {j}: "
                        f"expected <= {req_top_n} entries, got {len(top_n_dict)}"
                    )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "sampling_params_kwargs",
        [
            pytest.param({"top_k": 10}, id="top_k_10"),
            pytest.param({"top_p": 0.9, "top_k": 0}, id="top_p_0.9"),
            pytest.param({"temperature": 0.8, "top_k": 10}, id="temp_0.8_top_k_10"),
        ],
    )
    @torch.inference_mode()
    def test_speculative_decoding_non_greedy_sampling(self, sampling_params_kwargs: dict):
        """Test that speculative decoding works correctly with non-greedy sampling.

        Exercises top-k, top-p, and temperature sampling through the full engine
        pipeline with speculative tokens. Verifies:
        1. All requests complete without errors.
        2. Generated tokens are within the valid vocab range.
        3. The generated token count is correct.
        4. Log probs are consistent with the generated tokens.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=8,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        num_requests = 4
        prompt_lengths = [4, 5, 6, 8]
        for i in range(num_requests):
            prompt = torch.randint(
                0, test_config.vocab_size - 1, (prompt_lengths[i],), device='cuda'
            )
            params = SamplingParams(
                num_tokens_to_generate=10,
                termination_id=test_config.vocab_size - 1,
                return_log_probs=True,
                **sampling_params_kwargs,
            )
            env.engine.add_request(request_id=i, prompt=prompt, sampling_params=params)

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        assert len(finished_records) == num_requests

        for record in finished_records:
            req = record.merge()
            assert (
                req.status == Status.COMPLETED
            ), f"Request {req.request_id} not completed: {req.status}"

            # Generated tokens should be within valid vocab range.
            for j, tok in enumerate(req.generated_tokens):
                assert 0 <= tok < test_config.vocab_size, (
                    f"Request {req.request_id}, token {j}: "
                    f"token {tok} out of vocab range [0, {test_config.vocab_size})"
                )

            # Log probs count must match token count.
            assert req.generated_log_probs is not None
            assert len(req.generated_log_probs) == len(req.generated_tokens), (
                f"Request {req.request_id}: log probs count {len(req.generated_log_probs)} "
                f"!= token count {len(req.generated_tokens)}"
            )

            # All log probs should be valid (non-positive).
            for j, lp in enumerate(req.generated_log_probs):
                assert lp <= 0.0, f"Request {req.request_id}, token {j}: log prob {lp} > 0"

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_non_greedy_with_top_n_logprobs(self):
        """Test speculative decoding with non-greedy sampling and top-n logprobs.

        Combines non-greedy sampling (top_k=10) with top_n_logprobs to verify that
        the logprobs pipeline works end-to-end when speculative tokens may be
        partially rejected due to sampling randomness.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=8,
            num_tokens_to_generate=8,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        env.engine.controller.tokenizer.detokenize = lambda tokens, **kw: f"tok_{tokens[0]}"

        # top_n must be >= top_k so the top_k-sampled token is guaranteed to
        # have a higher probability than the least probable token in top_n.
        top_n = 10
        num_requests = 3
        prompt_lengths = [4, 6, 8]
        for i in range(num_requests):
            prompt = torch.randint(
                0, test_config.vocab_size - 1, (prompt_lengths[i],), device='cuda'
            )
            env.engine.add_request(
                request_id=i,
                prompt=prompt,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=8,
                    termination_id=test_config.vocab_size - 1,
                    return_log_probs=True,
                    top_n_logprobs=top_n,
                    top_k=10,
                ),
            )

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        assert len(finished_records) == num_requests

        for record in finished_records:
            req = record.merge()
            assert req.status == Status.COMPLETED

            # Top-n logprobs must be present and match token count.
            assert req.generated_top_n_logprobs is not None
            assert len(req.generated_top_n_logprobs) == len(req.generated_tokens)

            for j, top_n_dict in enumerate(req.generated_top_n_logprobs):
                assert isinstance(top_n_dict, dict)
                assert 0 < len(top_n_dict) <= top_n

            # Consistency: selected token's log prob should appear in top-n when the
            # sampled token is among the strict top-n indices. With nearly-tied logits
            # (random models), top-k filtering keeps all ties at the cutoff, so a sampled
            # token can fall outside a separate top-n index list. In that case the
            # selected logprob must still be no worse than the weakest top-n entry.
            if req.generated_log_probs is not None:
                for j, (lp, top_n_dict, token_id) in enumerate(
                    zip(req.generated_log_probs, req.generated_top_n_logprobs, req.generated_tokens)
                ):
                    token_str = env.engine.controller.tokenizer.detokenize([token_id])
                    if token_str in top_n_dict:
                        # Sampled token is in Top N.
                        assert abs(lp - top_n_dict[token_str]) < 0.01, (
                            f"Request {req.request_id}, token {j}: "
                            f"log_prob {lp} vs top-n {top_n_dict[token_str]}"
                        )
                    else:
                        # Sampled token is not in the Top N. It must be a tie.
                        # Check that it is at least as probable as Top N tokens.
                        assert lp + 0.01 >= min(top_n_dict.values()), (
                            f"Request {req.request_id}, token {j}: "
                            f"selected token '{token_str}' log_prob {lp} is worse than "
                            f"top-n minimum {min(top_n_dict.values())}"
                        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_logprobs_with_rejection(self):
        """Test that log probabilities are correct when speculative tokens are rejected.

        MTP head predicts wrong tokens so every speculative token is rejected.
        Each step emits exactly 1 token (the base model's sample). Verifies:
        1. Log prob count matches generated token count.
        2. Log prob values are near 0.0 (base model assigns logit=100 to correct token).
        3. Prompt log probs count equals prompt_length - 1 when not skipped.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=6,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape
            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_wrong(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            wrong_toks = (next_token_ids + 5).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, wrong_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_wrong

        prompt_length = 4
        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=6,
                termination_id=test_config.vocab_size - 1,
                return_log_probs=True,
                skip_prompt_log_probs=False,
                top_k=1,
            ),
        )

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        assert len(finished_records) == 1
        req = finished_records[0].merge()
        assert req.status == Status.COMPLETED
        assert (
            len(req.generated_tokens) == 6
        ), f"Expected 6 generated tokens, got {len(req.generated_tokens)}"

        assert req.generated_log_probs is not None
        assert len(req.generated_log_probs) == len(req.generated_tokens), (
            f"Log probs count {len(req.generated_log_probs)} != "
            f"token count {len(req.generated_tokens)}"
        )

        for j, lp in enumerate(req.generated_log_probs):
            assert isinstance(lp, float)
            assert (
                -0.1 < lp <= 0.0
            ), f"Token {j}: expected log prob near 0.0 (high confidence), got {lp}"

        assert req.prompt_log_probs is not None
        assert len(req.prompt_log_probs) == prompt_length - 1, (
            f"Expected {prompt_length - 1} prompt log probs, " f"got {len(req.prompt_log_probs)}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    @pytest.mark.parametrize("async_sched_mode", list(AsyncScheduleMode))
    def test_speculative_decoding_logprobs_with_stop_word_trim(self, async_sched_mode):
        """Test that log probs are correctly trimmed when a stop word lands
        in the middle of a speculative batch.

        With num_speculative_tokens=2, each step produces up to 3 tokens
        (1 base + 2 speculative). If the stop word is [6] and the engine
        generates [5, 6, 7] in one step, token 7 is truncated. The
        corresponding log prob for token 7 must also be removed so that
        len(generated_log_probs) == len(generated_tokens).

        Args:
            async_sched_mode (AsyncScheduleMode): Scheduling mode under test.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=10,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            async_sched_mode=async_sched_mode,
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        def mock_deterministic_forward(*args, **kwargs):
            tokens = kwargs.get("tokens", args[0] if args else kwargs.get("input_ids"))
            b, s = tokens.shape
            base_logits = torch.zeros(
                b, s, test_config.vocab_size, device=tokens.device, dtype=torch.bfloat16
            )
            next_toks = (tokens + 1).clamp(max=test_config.vocab_size - 1)
            base_logits.scatter_(2, next_toks.unsqueeze(-1), 100.0)
            env.engine.context.mtp_decoder_hidden_states = torch.zeros(
                s, 1, hidden_size, device=tokens.device, dtype=torch.bfloat16
            )
            return base_logits

        def mock_compute_mtp_single_step(
            hidden_states, next_token_ids, position_ids, depth=None, eager=False, cache_key=None
        ):
            n = hidden_states.size(0)
            pred_toks = (next_token_ids + 1).clamp(max=test_config.vocab_size - 1)
            logits = torch.zeros(
                n, 1, test_config.vocab_size, device=hidden_states.device, dtype=torch.bfloat16
            )
            logits.scatter_(2, pred_toks.transpose(0, 1).unsqueeze(-1), 100.0)
            return hidden_states, logits

        unwrapped_model.forward = mock_deterministic_forward
        unwrapped_model.compute_mtp_single_step = mock_compute_mtp_single_step

        env.engine.add_request(
            request_id=0,
            prompt=torch.tensor([1, 2, 3, 4], device='cuda'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=10,
                termination_id=99,
                detokenize_stop_sequence=True,
                return_log_probs=True,
                top_k=1,
                top_n_logprobs=2,
            ),
        )

        tracked_req = env.engine.get_request(0)
        tracked_req.stop_word_ids = [[6]]

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        finished_req = finished_records[0].merge()

        assert finished_req.status == Status.COMPLETED
        assert finished_req.generated_tokens == [5, 6]
        assert finished_req.generated_tokens[-1] == 6, (
            f"Expected last token to be stop word 6, "
            f"got {finished_req.generated_tokens[-1]}. "
            f"Full output: {finished_req.generated_tokens}"
        )

        assert (
            finished_req.generated_log_probs is not None
        ), "generated_log_probs is None despite return_log_probs=True"
        assert len(finished_req.generated_log_probs) == len(finished_req.generated_tokens), (
            f"Log probs count {len(finished_req.generated_log_probs)} != "
            f"token count {len(finished_req.generated_tokens)}. "
            f"Log probs were not trimmed after stop word truncation."
        )

        for j, lp in enumerate(finished_req.generated_log_probs):
            assert isinstance(lp, float)
            assert lp <= 0.0, f"Token {j}: log prob {lp} > 0"

        assert finished_req.generated_top_n_logprobs is not None
        assert len(finished_req.generated_top_n_logprobs) == len(finished_req.generated_tokens)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "kv_cache_management_mode", ["recompute", "persist"], ids=["recompute", "persist"]
    )
    @torch.inference_mode()
    def test_speculative_decoding_suspend_resume(self, kv_cache_management_mode):
        """Test that suspend/resume preserves speculative decoding correctness.

        Runs 2 requests with speculative decoding, suspends the engine
        mid-generation (after a few decode steps), resumes, and verifies
        all requests complete with the correct token count.

        In 'recompute' mode, the KV cache is discarded on suspend and
        requests are checkpointed and re-prefilled on resume. The engine
        must correctly reconstruct MTP state after re-prefill and continue
        speculative decoding without crashes or token count mismatches.

        In 'persist' mode, the KV cache survives suspend/resume. The MTP
        internal buffers (_sampled_mtp_tokens_cuda, _accepted_tokens_per_request)
        must remain valid across the cycle since requests stay in decode.
        """
        num_tokens_to_generate = 10

        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=num_tokens_to_generate,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            position_embedding_type="none",
            kv_cache_management_mode=kv_cache_management_mode,
        )

        needs_tms = test_config.static_kv_memory_pointers and kv_cache_management_mode != "persist"
        if needs_tms and not HAVE_TORCH_MEMORY_SAVER:
            pytest.skip("torch_memory_saver required for static pointers + non-persist mode")

        env = self._build_test_env(test_config)
        engine = env.engine

        unwrapped_model = engine.controller.inference_wrapped_model.model

        # Wrap real forward with deterministic logits.
        real_forward = unwrapped_model.forward

        def deterministic_forward(*args, **kwargs):
            logits = real_forward(*args, **kwargs)
            logits.zero_()
            logits[..., 0] = 100.0
            return logits

        real_mtp = unwrapped_model.compute_mtp_single_step

        def deterministic_mtp(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            hidden_states, logits = real_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
            )
            logits.zero_()
            logits[..., 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = deterministic_forward
        unwrapped_model.compute_mtp_single_step = deterministic_mtp

        for i in range(2):
            engine.add_request(
                request_id=i,
                prompt=torch.zeros(4, dtype=torch.int64, device='cuda'),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=test_config.vocab_size - 1,
                ),
            )

        # Run a few steps to get into decode with speculative tokens in flight.
        for _ in range(3):
            if not engine.has_unfinished_requests():
                break
            engine.step_modern()

        # Suspend mid-generation.
        engine.suspend()

        # Re-attach wrappers: resume rebuilds model state, but our closures
        # still hold the right `real_forward`/`real_mtp` references since the
        # model object itself is not recreated.
        unwrapped_model.forward = deterministic_forward
        unwrapped_model.compute_mtp_single_step = deterministic_mtp

        # Resume.
        engine.resume()

        # Run to completion.
        finished_records = []
        step_count = 0
        while engine.has_unfinished_requests():
            res = engine.step_modern()
            finished_records.extend(res["finished_request_records"])
            step_count += 1
            assert step_count < 200, "Engine did not converge after resume"

        # In recompute mode, requests are re-prefilled from prompt + generated_tokens.
        # In persist mode, requests continue from where they left off.
        # Either way, all requests must complete.
        for record in finished_records:
            req = record.merge()
            assert req.status == Status.COMPLETED, f"Request {req.request_id}: status={req.status}"
            assert len(req.generated_tokens) == num_tokens_to_generate, (
                f"Request {req.request_id}: expected {num_tokens_to_generate} "
                f"tokens, got {len(req.generated_tokens)}"
            )
            # All tokens should be 0 (deterministic prediction).
            assert all(t == 0 for t in req.generated_tokens), (
                f"Request {req.request_id}: expected all token 0, " f"got {req.generated_tokens}"
            )

        assert engine.context.active_token_count == 0
        assert engine.context.total_request_count == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "num_tokens_to_generate", [5, 6, 8, 9], ids=["gen5", "gen6", "gen8", "gen9"]
    )
    @torch.inference_mode()
    def test_speculative_decoding_finish_detection_accuracy(self, num_tokens_to_generate):
        """Verify that requests generate exactly num_tokens_to_generate tokens
        with speculative decoding, even when the requested count does not align
        with 1 + (num_speculative_tokens + 1) * N.

        With num_speculative_tokens=2 and all speculative tokens accepted,
        each decode step commits 3 tokens (2 accepted + 1 new base).
        Token counts of the form 1 + 3*N (i.e. 4, 7, 10 ...) align exactly
        with step boundaries.  Counts in between (5, 6, 8, 9 ...) require
        the engine to correctly detect that the request still needs more
        tokens after a full-acceptance decode step.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=num_tokens_to_generate,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            position_embedding_type="none",
        )
        env = self._build_test_env(test_config)
        unwrapped_model = env.engine.controller.inference_wrapped_model.model

        # Deterministic forward: always predict token 0.
        real_forward = unwrapped_model.forward

        def deterministic_forward(*args, **kwargs):
            logits = real_forward(*args, **kwargs)
            logits.zero_()
            logits[..., 0] = 100.0
            return logits

        # Deterministic MTP: also predict token 0 → all speculative tokens accepted.
        real_mtp = unwrapped_model.compute_mtp_single_step

        def deterministic_mtp(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            hidden_states, logits = real_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
            )
            logits.zero_()
            logits[..., 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = deterministic_forward
        unwrapped_model.compute_mtp_single_step = deterministic_mtp

        env.engine._add_request(env.requests[0])

        while env.engine.has_unfinished_requests():
            env.engine.step_modern()

        req = env.requests[0]
        assert req.status == Status.COMPLETED
        assert len(req.generated_tokens) == num_tokens_to_generate, (
            f"Expected {num_tokens_to_generate} tokens, "
            f"got {len(req.generated_tokens)}: {req.generated_tokens}"
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_mixed_prefill_decode_heterogeneous_acceptance(self):
        """Test speculative decoding with a mixed prefill/decode batch where
        decode requests have different acceptance outcomes.

        Adds 3 requests staggered so that when the 3rd request is still in
        prefill, the first 2 are in decode with speculative tokens. The base
        model and MTP heads are set up so that:
          - Request 0 (decode): all speculative tokens accepted (MTP agrees with base)
          - Request 1 (decode): all speculative tokens rejected (MTP predicts wrong tokens)
          - Request 2 (prefill): no speculative tokens (still in prefill)

        This exercises the critical decode/prefill indexing boundary in
        _dynamic_step_sample_logits_and_verify_tokens and heterogeneous
        accepted_token_counts in the same batch.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=6,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            position_embedding_type="none",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        # Wrap the real forward: run the actual model then overwrite logits
        # deterministically. Token 0 always has high logit.
        real_forward = unwrapped_model.forward

        def deterministic_forward(*args, **kwargs):
            logits = real_forward(*args, **kwargs)
            logits.zero_()
            logits[..., 0] = 100.0
            return logits

        # For MTP: predict token 0 for request 0 (accepted) but token 50
        # for request 1 (rejected, since base predicts token 0).
        # During prefill, no MTP runs, so request 2 is unaffected.
        real_mtp = unwrapped_model.compute_mtp_single_step

        def heterogeneous_mtp(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            hidden_states, logits = real_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
            )
            n = logits.size(0)
            logits.zero_()
            if n >= 2:
                logits[0, :, 0] = 100.0  # Request 0: accept (token 0)
                logits[1, :, 50] = 100.0  # Request 1: reject (token 50 != base's token 0)
            else:
                logits[:, :, 0] = 100.0  # Single request: accept
            return hidden_states, logits

        unwrapped_model.forward = deterministic_forward
        unwrapped_model.compute_mtp_single_step = heterogeneous_mtp

        # Add request 0 and 1 first, let them start decoding.
        for i in range(2):
            env.engine.add_request(
                request_id=i,
                prompt=torch.zeros(4, dtype=torch.int64, device='cuda'),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=6, termination_id=test_config.vocab_size - 1
                ),
            )

        # Step once to process prefill for requests 0 and 1.
        env.engine.step_modern()

        # Add request 2 while 0 and 1 are in decode → creates mixed batch.
        env.engine.add_request(
            request_id=2,
            prompt=torch.zeros(4, dtype=torch.int64, device='cuda'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=6, termination_id=test_config.vocab_size - 1
            ),
        )

        # Run to completion.
        finished_records = []
        step_count = 0
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])
            step_count += 1
            assert step_count < 200, "Engine did not converge"

        assert len(finished_records) == 3

        for record in finished_records:
            req = record.merge()
            assert (
                req.status == Status.COMPLETED
            ), f"Request {req.request_id} not completed: {req.status}"
            assert len(req.generated_tokens) == 6, (
                f"Request {req.request_id}: expected 6 tokens, " f"got {len(req.generated_tokens)}"
            )

        # Verify engine state is clean.
        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_logprobs_alignment_under_length_truncation(self):
        """Test that log probs count matches generated_tokens when speculative
        tokens are trimmed by num_tokens_to_generate (not by stop words).

        With num_speculative_tokens=2, each step emits up to 3 tokens.
        num_tokens_to_generate=5 is not divisible by 3, so the final step
        must truncate 1 token. The log probs for that discarded token must
        also be excluded.
        """
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=5,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="gpt",
            position_embedding_type="none",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model
        hidden_size = unwrapped_model.config.hidden_size

        real_forward = unwrapped_model.forward

        def deterministic_forward(*args, **kwargs):
            logits = real_forward(*args, **kwargs)
            logits.zero_()
            logits[..., 0] = 100.0
            return logits

        real_mtp = unwrapped_model.compute_mtp_single_step

        def deterministic_mtp(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            hidden_states, logits = real_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
            )
            logits.zero_()
            logits[..., 0] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = deterministic_forward
        unwrapped_model.compute_mtp_single_step = deterministic_mtp

        env.engine.add_request(
            request_id=0,
            prompt=torch.zeros(4, dtype=torch.int64, device='cuda'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=5,
                termination_id=test_config.vocab_size - 1,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                top_k=1,
            ),
        )

        finished_records = []
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])

        assert len(finished_records) == 1
        req = finished_records[0].merge()

        assert req.status == Status.COMPLETED
        assert len(req.generated_tokens) == 5, f"Expected 5 tokens, got {len(req.generated_tokens)}"

        # This is the critical assertion: log probs must align with tokens
        # even when the final speculative batch was length-truncated.
        assert req.generated_log_probs is not None
        assert len(req.generated_log_probs) == len(req.generated_tokens), (
            f"Log probs count {len(req.generated_log_probs)} != "
            f"token count {len(req.generated_tokens)}. "
            f"Log probs were not trimmed when length truncation discarded "
            f"speculative tokens."
        )

        for j, lp in enumerate(req.generated_log_probs):
            assert isinstance(lp, float)
            assert -0.1 < lp <= 0.0, f"Token {j}: expected log prob near 0.0, got {lp}"

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "rejection_mode",
        ["all_accepted", "all_rejected", "partial"],
        ids=["accept_all", "reject_all", "partial_reject"],
    )
    @torch.inference_mode()
    def test_speculative_decoding_mamba_hybrid(self, rejection_mode):
        """Test speculative decoding with a Mamba hybrid model.

        Exercises the intermediate Mamba state commit/rewind path with
        speculative tokens under three acceptance scenarios:
          - all_accepted: all speculative tokens match the base model, no rewind
          - all_rejected: MTP predicts wrong tokens, full rewind every step
          - partial: first speculative token accepted, second rejected

        The rewind path (text_generation_controller._rewind_kv_cache) indexes
        into mamba_intermediate_{conv,ssm}_states using accepted_token_counts
        to restore the correct Mamba state. This test verifies that state is
        not corrupted across multiple rewind cycles and that the model produces
        the correct number of tokens.

        Two requests run simultaneously to exercise batched rewind indexing
        where mamba_metadata.request_to_mamba_state_idx differs per request.
        """
        skip_if_mamba_sequence_packing_not_available("hybrid")

        num_tokens_to_generate = 8
        test_config = DynamicEngineTestConfig(
            num_requests=0,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=num_tokens_to_generate,
            num_speculative_tokens=2,
            materialize_only_last_token_logits=False,
            model_provider="hybrid",
        )
        env = self._build_test_env(test_config)

        unwrapped_model = env.engine.controller.inference_wrapped_model.model

        # Wrap real forward: run real Mamba layers (conv/SSM state updates)
        # then substitute deterministic logits.
        real_forward = unwrapped_model.forward

        def deterministic_forward(*args, **kwargs):
            logits = real_forward(*args, **kwargs)
            # Base model always predicts token 0.
            logits.zero_()
            logits[..., 0] = 100.0
            return logits

        real_mtp = unwrapped_model.compute_mtp_single_step

        def mtp_with_rejection(
            hidden_states, next_token_ids, position_ids, depth, eager=False, cache_key=None
        ):
            # Run real MTP to exercise Mamba intermediate state saving.
            hidden_states, logits = real_mtp(
                hidden_states, next_token_ids, position_ids, depth, eager=eager, cache_key=cache_key
            )
            logits.zero_()
            if rejection_mode == "all_accepted":
                # Predict token 0 (same as base) → accepted.
                logits[..., 0] = 100.0
            elif rejection_mode == "all_rejected":
                # Predict token 50 (differs from base's token 0) → rejected.
                # Forces full rewind of Mamba intermediate states every step.
                logits[..., 50] = 100.0
            else:
                # partial: depth 0 accepted (token 0), depth 1 rejected (token 50).
                # This exercises the rewind to an intermediate depth, verifying
                # that mamba_intermediate_states[accepted_count] is correct.
                if depth == 0:
                    logits[..., 0] = 100.0
                else:
                    logits[..., 50] = 100.0
            return hidden_states, logits

        unwrapped_model.forward = deterministic_forward
        unwrapped_model.compute_mtp_single_step = mtp_with_rejection

        # Add 2 requests to exercise batched Mamba state indexing.
        for i in range(2):
            env.engine.add_request(
                request_id=i,
                prompt=torch.zeros(4, dtype=torch.int64, device='cuda'),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=test_config.vocab_size - 1,
                ),
            )

        finished_records = []
        step_count = 0
        while env.engine.has_unfinished_requests():
            res = env.engine.step_modern()
            finished_records.extend(res["finished_request_records"])
            step_count += 1
            assert step_count < 200, "Engine did not converge"

        assert len(finished_records) == 2

        for record in finished_records:
            req = record.merge()
            assert req.status == Status.COMPLETED, f"Request {req.request_id}: status={req.status}"
            assert len(req.generated_tokens) == num_tokens_to_generate, (
                f"Request {req.request_id}: expected {num_tokens_to_generate} "
                f"tokens, got {len(req.generated_tokens)}"
            )
            # All tokens should be 0 (deterministic base model prediction).
            assert all(t == 0 for t in req.generated_tokens), (
                f"Request {req.request_id}: expected all token 0, " f"got {req.generated_tokens}"
            )

        # Verify engine state is clean.
        assert env.engine.context.active_token_count == 0
        assert env.engine.context.total_request_count == 0


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestGDNDynamicInferenceEngine(DynamicInferenceEngineTestBase):
    """Exercise GDN through the production scheduler and local CUDA graphs."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    @staticmethod
    def _generated_tokens(env):
        return [list(request.generated_tokens) for request in env.requests]

    def test_cuda_graph_parity(self):
        common = dict(
            model_provider="hybrid",
            ssm_mixer="gdn",
            num_requests=4,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=8,
            num_gap_steps=0,
            top_k=1,
            context_max_requests=32,
            use_cuda_graphs_for_non_decode_steps=False,
        )
        eager = self._run_test(**common, num_cuda_graphs=None)
        graphed = self._run_test(
            **common,
            num_cuda_graphs=3,
            force_build_cuda_graphs=True,
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
        )

        model = graphed.engine.controller.inference_wrapped_model.model
        assert model.cudagraph_manager.cudagraph_runners
        assert self._generated_tokens(graphed) == self._generated_tokens(eager)

    def test_scheduling_invariance(self):
        """Staggered admission must not change per-request greedy output."""
        common = dict(
            model_provider="hybrid",
            ssm_mixer="gdn",
            num_requests=4,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=8,
            top_k=1,
            context_max_requests=32,
            num_cuda_graphs=None,
        )
        dense = self._run_test(**common, num_gap_steps=0)
        staggered = self._run_test(**common, num_gap_steps=3)

        assert self._generated_tokens(staggered) == self._generated_tokens(dense)
