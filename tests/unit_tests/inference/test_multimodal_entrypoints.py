# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mini end-to-end coverage for multimodal inference entrypoints.

These tests keep the HTTP/API plumbing and real LLaVA/Omni prompt expansion,
while replacing media decoding and the vision backbone with deterministic toy
tensors. They intentionally do not require checkpoints or distributed setup.
"""

import asyncio
import base64
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.apis.async_llm import MegatronAsyncLLM
from megatron.core.inference.config import ImageProcessingConfig, VideoProcessingConfig
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    resolve_multimodal_data_for_engine,
    serialize_multimodal_data,
)
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.multimodal.utils import (
    dynamic_media_embedding_counts,
    dynamic_media_replacement_counts,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.coordinator_test_utils import make_coordinator_direct

_MEDIA_TOKEN_ID = 99
_PROMPT_TOKENS = [10, _MEDIA_TOKEN_ID, 20]
_MEDIA_BYTES = b"toy-media"


class _ToyTokenizer:
    unk_token_id = 0
    eod = 0

    def convert_tokens_to_ids(self, token):
        return _MEDIA_TOKEN_ID if token == "<image>" else self.unk_token_id

    def detokenize(self, token_ids):
        return " ".join(str(token_id) for token_id in token_ids)


class _InlineLoopManager:
    async def run_async(self, awaitable):
        return await awaitable


class _ToyLanguageModel:
    def __init__(self):
        self.last_decoder_input = None

    def embedding(self, input_ids, position_ids):
        del position_ids
        batch_size, sequence_length = input_ids.shape
        return torch.zeros(sequence_length, batch_size, 4)

    def __call__(self, *, decoder_input, **_kwargs):
        self.last_decoder_input = decoder_input
        return decoder_input


def _toy_preprocessed_media(modality):
    if modality == "video":
        return {
            "imgs": torch.ones(2, 3, 4, 4),
            "imgs_sizes": torch.tensor([[4, 4], [4, 4]], dtype=torch.int32),
            "num_frames": torch.tensor([2], dtype=torch.int32),
        }
    return {"imgs": torch.ones(1, 3, 4, 4), "imgs_sizes": torch.tensor([[4, 4]], dtype=torch.int32)}


def _build_toy_wrapper(wrapper_cls):
    language_model = _ToyLanguageModel()
    model = SimpleNamespace(
        image_token_index=-200,
        dynamic_resolution=True,
        _dynamic_resolution=True,
        patch_dim=2,
        _pixel_shuffle=False,
        _conv_merging=False,
        _drop_vision_class_token=True,
        _class_token_len=0,
        temporal_patch_dim=1,
        vision_model=SimpleNamespace(temporal_patch_dim=1),
        language_model=language_model,
        sequence_parallel_lm=False,
    )
    model.forward_lm_only = mock.Mock(
        side_effect=lambda *, combined_embeddings, **_kwargs: combined_embeddings
    )
    wrapper = object.__new__(wrapper_cls)
    wrapper.model = SimpleNamespace(module=model) if wrapper_cls is VLMInferenceWrapper else model
    wrapper.inference_context = None
    wrapper.pp_group = None

    def toy_vision_forward(_images, num_image_tiles=None, imgs_sizes=None, num_frames=None):
        assert num_image_tiles is None
        frame_counts = dynamic_media_embedding_counts(
            imgs_sizes,
            patch_dim=model.patch_dim,
            pixel_shuffle=wrapper_cls is NemotronOmniInferenceWrapper,
        )
        replacement_counts = dynamic_media_replacement_counts(
            frame_counts,
            num_frames=num_frames,
            temporal_patch_size=model.vision_model.temporal_patch_dim,
        )
        return torch.ones(sum(replacement_counts), 1, 4)

    wrapper._forward_vision_encoder = mock.Mock(side_effect=toy_vision_forward)
    return wrapper


class _ToyInferenceService:
    """Mimic the client/coordinator boundary and admit into a real VLM request builder."""

    def __init__(self, wrapper_cls, *, deserialize):
        self.deserialize = deserialize
        self.tokenizer = _ToyTokenizer()
        wrapper = _build_toy_wrapper(wrapper_cls)
        self.wrapper = wrapper
        self.model = wrapper.model.module if wrapper_cls is VLMInferenceWrapper else wrapper.model

        self.engine = object.__new__(DynamicInferenceEngine)
        self.engine.controller = SimpleNamespace(
            inference_wrapped_model=wrapper, tokenizer=self.tokenizer, pp_group=None
        )
        self.engine.context = SimpleNamespace(
            block_size_tokens=16, enable_prefix_caching=False, add_vlm_request_data=mock.Mock()
        )
        self.engine.vision_embedding_cache_max_bytes = 1 << 20
        self.engine._vision_embedding_cache = OrderedDict()
        self.engine._vision_embedding_cache_bytes = 0
        self.coordinator = make_coordinator_direct(
            enable_prefix_caching=False, prefix_caching_routing_alpha=1.0
        )
        self.selected_ranks = []
        self._request_id = 0
        self.last_request = None
        self.last_wire_data = None

        image_config = ImageProcessingConfig(patch_dim=2, dynamic_resolution=True)
        self.image_config = image_config
        self.video_config = VideoProcessingConfig(
            image_config=image_config, num_frames=2, temporal_patch_size=1
        )

    def _run_toy_decoder(self, request):
        sequence_length = len(request.prompt_tokens)
        inference_input = {
            "tokens": request.prompt_tokens.unsqueeze(0),
            "position_ids": torch.arange(sequence_length).unsqueeze(0),
            "attention_mask": None,
            "image_token_mask": request.image_token_mask.unsqueeze(0),
            "image_embeddings": request.image_embeddings,
        }
        if isinstance(self.wrapper, VLMInferenceWrapper):
            with mock.patch(
                "megatron.core.inference.model_inference_wrappers.multimodal."
                "vlm_inference_wrapper.is_pipeline_first_stage",
                return_value=True,
            ):
                decoder_output = self.wrapper._forward_dynamic(inference_input)
        else:
            decoder_output = self.wrapper._forward_dynamic(inference_input).transpose(0, 1)

        image_positions = inference_input["image_token_mask"] >= 0
        assert torch.all(decoder_output[image_positions] == 1)
        assert torch.all(decoder_output[~image_positions] == 0)

    def add_request(self, prompt, sampling_params, *, multi_modal_data=None):
        future = asyncio.get_running_loop().create_future()
        try:
            self.last_wire_data = serialize_multimodal_data(multi_modal_data)
            media_cache_key = self.last_wire_data["media_cache_key"]
            request_hashes = self.coordinator.compute_request_hashes(
                prompt, cache_salt=media_cache_key
            )
            selected_rank = self.coordinator.get_best_data_parallel_rank(
                request_hashes, media_cache_key=media_cache_key
            )
            self.coordinator._update_media_affinity(media_cache_key, selected_rank)
            rank_index = self.coordinator.identity_to_rank_index[selected_rank]
            self.coordinator._pending_counts[rank_index] += 1
            self.selected_ranks.append(selected_rank)

            engine_kwargs = resolve_multimodal_data_for_engine(
                self.last_wire_data,
                image_preprocessing_config=self.image_config,
                video_preprocessing_config=self.video_config,
            )
            media_tokens_preexpanded = engine_kwargs.pop("media_tokens_preexpanded", False)
            engine_kwargs.setdefault("num_tiles", None)
            self._request_id += 1
            request = self.engine._build_vlm_request(
                request_id=self._request_id,
                prompt_str=None,
                tokens=torch.tensor(prompt, dtype=torch.int64),
                sampling_params=sampling_params,
                num_img_embeddings_per_tile=0,
                precomputed_block_hashes=None,
                media_tokens_preexpanded=media_tokens_preexpanded,
                **engine_kwargs,
            )
            request.generated_text = "toy output"
            request.generated_tokens = [7]
            self._run_toy_decoder(request)
            self.last_request = request
            if self.deserialize:
                result = request
            else:
                result = {
                    "uid": "toy-request",
                    "status": "COMPLETED",
                    "generated_text": request.generated_text,
                    "generated_tokens": request.generated_tokens,
                    "prompt_tokens": request.prompt_tokens.tolist(),
                    "sampling_params": {"num_tokens_to_generate": 1},
                    "routing_indices": None,
                }
            future.set_result(result)
        except Exception as error:
            future.set_exception(error)
        return future


@pytest.fixture
def toy_media_preprocessing(monkeypatch):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        image_preprocessing,
    )

    monkeypatch.setattr(
        image_preprocessing,
        "preprocess_image_bytes_list",
        lambda media, _config, device=None: (
            _toy_preprocessed_media("image") if media == [_MEDIA_BYTES] else None
        ),
    )
    monkeypatch.setattr(
        image_preprocessing,
        "preprocess_video_bytes_list",
        lambda media, _config, device=None: (
            _toy_preprocessed_media("video") if media == [_MEDIA_BYTES] else None
        ),
    )
    # Keep this toy E2E path on CPU even when the test worker can see a GPU.
    # Otherwise resolve_multimodal_data_for_engine constructs a CUDA device
    # before the mocked preprocessors have a chance to ignore it.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))


def _build_toy_async_llm(service):
    llm = object.__new__(MegatronAsyncLLM)
    llm._is_primary_rank = True
    llm._use_coordinator = True
    llm._loop_manager = _InlineLoopManager()
    llm._coord_runtime = SimpleNamespace(client=service)
    return llm


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_cls", [VLMInferenceWrapper, NemotronOmniInferenceWrapper])
@pytest.mark.parametrize("modality", ["image", "video"])
async def test_generate_multimodal_entrypoint_with_toy_model(
    wrapper_cls, modality, toy_media_preprocessing
):
    service = _ToyInferenceService(wrapper_cls, deserialize=True)
    llm = _build_toy_async_llm(service)

    result = await llm.generate(
        _PROMPT_TOKENS,
        SamplingParams(num_tokens_to_generate=1, termination_id=0),
        multi_modal_data={modality: _MEDIA_BYTES},
    )

    assert result is service.last_request
    assert result.compact_prompt_tokens.tolist() == _PROMPT_TOKENS
    assert result.generated_text == "toy output"
    assert service.last_wire_data[modality] == [_MEDIA_BYTES]
    assert service.wrapper._forward_vision_encoder.call_count == 1
    assert int((result.image_token_mask >= 0).sum()) == result.image_embeddings.shape[0]


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_cls", [VLMInferenceWrapper, NemotronOmniInferenceWrapper])
@pytest.mark.parametrize("modality", ["image", "video"])
async def test_repeated_media_hits_vision_cache_and_coordinator_affinity(
    wrapper_cls, modality, toy_media_preprocessing
):
    service = _ToyInferenceService(wrapper_cls, deserialize=True)
    llm = _build_toy_async_llm(service)
    sampling_params = SamplingParams(num_tokens_to_generate=1, termination_id=0)
    multi_modal_data = {modality: _MEDIA_BYTES}

    first = await llm.generate(_PROMPT_TOKENS, sampling_params, multi_modal_data=multi_modal_data)
    second = await llm.generate(_PROMPT_TOKENS, sampling_params, multi_modal_data=multi_modal_data)

    assert service.selected_ranks[0] == service.selected_ranks[1]
    assert service.wrapper._forward_vision_encoder.call_count == 1
    assert len(service.engine._vision_embedding_cache) == 1
    assert second.image_embeddings is first.image_embeddings


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_cls", [VLMInferenceWrapper, NemotronOmniInferenceWrapper])
@pytest.mark.parametrize("modality", ["image", "video"])
async def test_completions_multimodal_entrypoint_with_toy_model(
    wrapper_cls, modality, toy_media_preprocessing
):
    quart = pytest.importorskip("quart")
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.completions import (
        bp,
    )

    service = _ToyInferenceService(wrapper_cls, deserialize=False)
    app = quart.Quart(__name__)
    app.config.update(client=service, tokenizer=service.tokenizer, verbose=False)
    app.register_blueprint(bp)

    response = await app.test_client().post(
        "/v1/completions",
        json={
            "prompt": _PROMPT_TOKENS,
            "max_tokens": 1,
            "multi_modal_data": {modality: base64.b64encode(_MEDIA_BYTES).decode("ascii")},
        },
    )

    assert response.status_code == 200
    payload = await response.get_json()
    assert payload["choices"][0]["text"] == "toy output"
    assert service.last_request.compact_prompt_tokens.tolist() == _PROMPT_TOKENS
    assert service.last_wire_data[modality] == [_MEDIA_BYTES]
    assert service.wrapper._forward_vision_encoder.call_count == 1
    assert int((service.last_request.image_token_mask >= 0).sum()) == (
        service.last_request.image_embeddings.shape[0]
    )
