# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import base64
import copy
import io
import json
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from megatron.core import parallel_state
from megatron.core.activations import squared_relu
from megatron.core.inference.config import (
    ImageProcessingConfig,
    InferenceConfig,
    MambaInferenceStateConfig,
    MultimodalPromptConfig,
    VideoProcessingConfig,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import (
    PREFIX_EXPANDED_TOKEN_COUNT_FIELD,
    PREFIX_MEDIA_COUNT_FIELD,
    compute_media_cache_key,
    resolve_multimodal_data_for_engine,
    serialize_multimodal_data,
)
from megatron.core.inference.model_inference_wrappers.multimodal.nemotron_omni_inference_wrapper import (
    NemotronOmniInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.chat_completions import (
    _extract_multimodal_from_messages,
    _tokenize_with_media_slots,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_local_spec,
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.inference.engines.ssm_test_helpers import (
    hybrid_mixer_kwargs,
    hybrid_stack_spec_for,
    skip_if_sequence_packing_not_available,
)
from tests.unit_tests.test_utilities import Utils

IMAGE_TOKEN_ID, VOCAB_SIZE = 126, 128
pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.fixture
def model_parallel(request):
    """Initialize one topology: "tp" (default), "dp", "ep", "etp", or "opt".

    Tensor parallelism is capped at two ranks so every test model divides its heads evenly
    under any launch size; the remaining ranks are data-parallel replicas. The other
    topologies are selected with indirect parametrization and need at least two ranks.
    """
    topology = getattr(request, "param", "tp")
    if topology != "tp" and Utils.world_size < 2:
        pytest.skip(f"the {topology} topology needs at least two ranks")
    old_token_rounder = DynamicInferenceContext.TOKEN_ROUNDER
    old_request_rounder = DynamicInferenceContext.REQUEST_ROUNDER
    DynamicInferenceContext.TOKEN_ROUNDER = 1
    DynamicInferenceContext.REQUEST_ROUNDER = 1
    init_kwargs = {}
    if topology == "ep":
        init_kwargs["expert_model_parallel_size"] = 2
    elif topology == "etp":
        init_kwargs["expert_tensor_parallel_size"] = 2
    tp_size = min(Utils.world_size, 2) if topology == "tp" else 1
    Utils.initialize_model_parallel(tp_size, 1, **init_kwargs)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
    yield topology
    InferenceMode.unset_active()
    Utils.destroy_model_parallel()
    DynamicInferenceContext.TOKEN_ROUNDER = old_token_rounder
    DynamicInferenceContext.REQUEST_ROUNDER = old_request_rounder


def _transformer_config(hidden_size, num_attention_heads, **kwargs):
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=max(
            num_attention_heads, parallel_state.get_tensor_model_parallel_world_size()
        ),
        kv_channels=64,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        use_cpu_initialization=True,
        transformer_impl="local",
        **kwargs,
    )


def _language_defaults(**overrides):
    return dict(
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        use_cpu_initialization=True,
        inference_rng_tracker=True,
        inference_sampling_seed=123,
        **overrides,
    )


def _llava_model(**kwargs):
    model = LLaVAModel(
        language_vocab_size=VOCAB_SIZE,
        language_max_sequence_length=64,
        img_h=8,
        img_w=8,
        patch_dim=4,
        image_token_index=-200,
        **kwargs,
    ).cuda()
    return model.bfloat16().eval()


def _component_configs(projection_hidden, projection_ffn):
    vision = _transformer_config(16, 2)
    vision.vision_model_type = "clip"
    projection = _transformer_config(projection_hidden, 1, ffn_hidden_size=projection_ffn)
    return vision, projection, get_gpt_layer_local_submodules()


def _build_model(
    modality,
    vision_model_type=None,
    pixel_shuffle=False,
    temporal_patch_dim=1,
    separate_video_embedder=False,
    **language_config_overrides,
):
    language_config = _transformer_config(
        32, 4, inference_rng_tracker=True, inference_sampling_seed=123, **language_config_overrides
    )
    language_config.language_model_type = "dummy"
    vision_model_type = vision_model_type or ("clip" if modality == "image" else "radio")
    vision_config = _transformer_config(16, 2)
    vision_config.vision_model_type = vision_model_type
    native_merged = vision_model_type in ("pixtral-vit-large", "qwen-vl", "kimi-vit")
    dynamic_resolution = (
        pixel_shuffle
        or modality == "video"
        or vision_model_type
        in ("radio-g", "cradio-g", "pixtral-vit", "pixtral-vit-large", "qwen-vl", "kimi-vit")
    )
    if vision_model_type.startswith("pixtral"):
        vision_config.normalization = "RMSNorm"
        vision_config.gated_linear_unit = True
        vision_config.activation_func = F.silu
        vision_config.add_bias_linear = False
        vision_config.rotary_interleaved = True
    elif vision_model_type in ("qwen-vl", "kimi-vit"):
        vision_config.activation_func = lambda value: F.gelu(value, approximate="tanh")
        vision_config.rotary_interleaved = vision_model_type == "kimi-vit"
        vision_config.num_position_embeddings = 4
        vision_config.pos_embed_height = 2
        vision_config.pos_embed_width = 2
    projection_config = _transformer_config(32, 1, ffn_hidden_size=64)
    layer_submodules = get_gpt_layer_local_submodules()
    projection_spec = copy.deepcopy(get_submodules(layer_submodules.mlp))
    assert isinstance(projection_spec, MLPSubmodules)
    return _llava_model(
        language_transformer_config=language_config,
        language_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=copy.deepcopy(layer_submodules)
        ),
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=(
            get_vit_layer_with_transformer_engine_spec()
            if dynamic_resolution
            else get_vit_layer_with_local_spec()
        ),
        drop_vision_class_token=vision_model_type != "siglip",
        vision_projection_config=projection_config,
        vision_projection_layer_spec=projection_spec,
        dynamic_resolution=dynamic_resolution,
        class_token_len=0 if dynamic_resolution else None,
        conv_merging=native_merged,
        pixel_shuffle=pixel_shuffle,
        temporal_patch_dim=temporal_patch_dim,
        separate_video_embedder=separate_video_embedder,
    )


def _attach_omni_encoder_api(model):
    def encode_images(self, images, imgs_sizes, vision_packed_seq_params, num_frames):
        old_add_decoder = self.add_decoder
        self.add_decoder = False
        images = images.to(dtype=self.vision_model.embedder.weight.dtype)
        try:
            embeddings, _ = self(
                images,
                torch.empty((1, 0), dtype=torch.long, device=images.device),
                position_ids=None,
                attention_mask=None,
                imgs_sizes=imgs_sizes,
                vision_packed_seq_params=vision_packed_seq_params,
                num_frames=num_frames,
                runtime_gather_output=True,
            )
        finally:
            self.add_decoder = old_add_decoder
        return embeddings.squeeze(1)

    model._encode_images = MethodType(encode_images, model)
    return model


def _build_hybrid_model(ssm_mixer, with_mtp=False):
    is_gdn = ssm_mixer == "gdn"
    language_config = TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=16,
        kv_channels=64,
        transformer_impl="transformer_engine",
        is_hybrid_model=True,
        mtp_num_layers=1 if with_mtp else None,
        normalization="RMSNorm" if is_gdn else "LayerNorm",
        layernorm_zero_centered_gamma=is_gdn,
        activation_func=F.silu if is_gdn else F.gelu,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        **hybrid_mixer_kwargs(ssm_mixer),
        **_language_defaults(),
    )
    language_config.language_model_type = "dummy"
    vision_config, projection_config, layer_submodules = _component_configs(256, 512)
    return _llava_model(
        language_transformer_config=language_config,
        language_transformer_layer_spec=hybrid_stack_spec_for(ssm_mixer),
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=copy.deepcopy(layer_submodules)
        ),
        drop_vision_class_token=True,
        vision_projection_config=projection_config,
        vision_projection_layer_spec=copy.deepcopy(get_submodules(layer_submodules.mlp)),
        hybrid_layer_pattern=("G" if is_gdn else "M") + "*-" + ("/M" if with_mtp else ""),
        language_position_embedding_type="none",
    )


def _build_moe_model():
    expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
    expert_tensor_parallel_size = parallel_state.get_expert_tensor_parallel_world_size()
    language_config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        kv_channels=64,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        transformer_impl="transformer_engine",
        **_language_defaults(),
    )
    language_config.language_model_type = "dummy"
    vision_config, projection_config, layer_submodules = _component_configs(32, 64)
    return _llava_model(
        language_transformer_config=language_config,
        language_transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(num_experts=2),
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=copy.deepcopy(layer_submodules)
        ),
        drop_vision_class_token=True,
        vision_projection_config=projection_config,
        vision_projection_layer_spec=copy.deepcopy(get_submodules(layer_submodules.mlp)),
    )


def _build_te_parallel_model(topology):
    optimized = topology == "opt"
    language_config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        kv_channels=64,
        transformer_impl="inference_optimized" if optimized else "transformer_engine",
        normalization="RMSNorm" if optimized else "LayerNorm",
        add_bias_linear=not optimized,
        activation_func=squared_relu if optimized else F.gelu,
        **_language_defaults(),
    )
    language_config.language_model_type = "dummy"
    vision_config, projection_config, layer_submodules = _component_configs(32, 64)
    return _llava_model(
        language_transformer_config=language_config,
        language_transformer_layer_spec=(
            get_gpt_layer_with_inference_spec()
            if optimized
            else get_gpt_layer_with_transformer_engine_spec()
        ),
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=copy.deepcopy(layer_submodules)
        ),
        drop_vision_class_token=True,
        vision_projection_config=projection_config,
        vision_projection_layer_spec=copy.deepcopy(get_submodules(layer_submodules.mlp)),
    )


def _build_attention_variant_model(variant):
    common = _language_defaults(num_layers=1, hidden_size=32, num_attention_heads=4, kv_channels=64)
    language_config = TransformerConfig(
        transformer_impl="local", window_size=(4, 0) if variant == "swa" else None, **common
    )
    layer_submodules = get_gpt_layer_local_submodules()
    language_spec = ModuleSpec(module=TransformerLayer, submodules=copy.deepcopy(layer_submodules))
    return _variant_llava_model(language_config, language_spec, "learned_absolute")


def _variant_llava_model(language_config, language_spec, position_embedding_type):
    language_config.language_model_type = "dummy"
    vision_config, projection_config, projection_submodules = _component_configs(
        language_config.hidden_size, 128
    )
    return _llava_model(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_spec,
        language_position_embedding_type=position_embedding_type,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=copy.deepcopy(projection_submodules)
        ),
        drop_vision_class_token=True,
        vision_projection_config=projection_config,
        vision_projection_layer_spec=copy.deepcopy(get_submodules(projection_submodules.mlp)),
    )


def _inference_config(model, **overrides):
    config_kwargs = dict(
        block_size_tokens=4,
        buffer_size_gb=0.01,
        max_sequence_length=64,
        max_requests=4,
        max_tokens=64,
        vision_embedding_cache_max_bytes=1024 * 1024,
        mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
    )
    config_kwargs.update(overrides)
    return InferenceConfig(**config_kwargs)


def _build_engine(modality, model=None, wrapper_cls=VLMInferenceWrapper, **config_overrides):
    model = model or _build_model(modality)
    context = DynamicInferenceContext(model.config, _inference_config(model, **config_overrides))
    wrapper = wrapper_cls(model, context)
    wrapper.prep_model_for_inference()
    tokenizer = SimpleNamespace(
        eod=VOCAB_SIZE - 1,
        unk_token_id=-1,
        vocab_size=VOCAB_SIZE,
        convert_tokens_to_ids=lambda token: IMAGE_TOKEN_ID if token == "<image>" else -1,
        tokenize=lambda text: (
            [9, IMAGE_TOKEN_ID, 10] * text.count("<image>")
            if "<image>" in text
            else ([9] if text else [])
        ),
        detokenize=lambda tokens: " ".join(map(str, tokens)),
    )
    controller = TextGenerationController(wrapper, tokenizer)
    engine = DynamicInferenceEngine(controller, context)
    return engine, wrapper


def _encoder_patch(wrapper):
    return mock.patch.object(
        wrapper, "_forward_vision_encoder", wraps=wrapper._forward_vision_encoder
    )


def _greedy_token(step):
    return int(step["logits"][0, -1].argmax())


def _drain(engine):
    finished = []
    while engine.has_unfinished_requests():
        finished.extend(engine.step_modern()["finished_requests"])
    return finished


def _same(first, first_step, first_finished, reuse, reuse_step, reuse_finished):
    assert first["mask"].equal(reuse["mask"]) and first["embeddings"].equal(reuse["embeddings"])
    assert torch.equal(first_step["embeddings"], reuse_step["embeddings"])
    assert torch.equal(first_step["logits"], reuse_step["logits"])
    assert first_finished.generated_tokens == reuse_finished.generated_tokens


def _assert_step(admission, step, finished):
    assert step["mask"].ge(0).sum().item() == 4
    assert torch.equal(step["embeddings"], admission["embeddings"])
    assert step["logits"].shape == (1, 1, VOCAB_SIZE) and torch.isfinite(step["logits"]).all()
    assert finished.generated_tokens == [_greedy_token(step)]


def _install_forward_trace(engine):
    trace, original_forward = [], engine.controller._dynamic_step_forward_logits

    def traced_forward(input_ids, position_ids):
        active_slice = slice(
            engine.context.paused_request_count, engine.context.total_request_count
        )
        request_ids = engine.context.request_ids[active_slice].tolist()
        query_lengths = engine.context.request_query_lengths[active_slice].tolist()
        kv_offsets = engine.context.request_kv_length_offsets[active_slice].tolist()
        mask = engine.context.current_image_token_mask()
        embeddings = engine.context.current_image_embeddings()
        original_forward(input_ids, position_ids)
        trace.append(
            {
                "request_ids": request_ids,
                "query_lengths": query_lengths,
                "kv_offsets": kv_offsets,
                "input_ids": input_ids.detach().cpu().clone(),
                "mask": None if mask is None else mask.detach().cpu().clone(),
                "embeddings": (None if embeddings is None else embeddings.detach().cpu().clone()),
                "logits": engine.controller._all_logits_cuda.detach().cpu().clone(),
                "graphed": engine.context.using_cuda_graph_this_step(),
            }
        )

    engine.controller._dynamic_step_forward_logits = traced_forward
    return trace


def _media(modality, offset):
    if modality == "image":
        imgs = torch.linspace(-1, 1, 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
        imgs = imgs + offset
        fields = {
            "imgs": imgs,
            "num_tiles": torch.tensor([1], dtype=torch.int32),
            "num_img_embeddings_per_tile": 4,
        }
    else:
        patch_features = 3 * 4 * 4
        imgs = torch.linspace(-1, 1, 8 * patch_features, dtype=torch.float32).reshape(
            1, 8, patch_features
        )
        imgs = imgs + offset
        fields = {
            "imgs": imgs,
            "imgs_sizes": torch.tensor([[8, 8], [8, 8]], dtype=torch.int32),
            "num_frames": torch.tensor([2], dtype=torch.int32),
            "video_frame_indices": [[0, 3]],
            "video_fps": [30.0],
        }
    fields["media_cache_key"] = compute_media_cache_key(modality, fields)
    return fields


def _run_request(engine, request_id, media, trace, prompt=None, num_tokens=1):
    params = SamplingParams(num_tokens_to_generate=num_tokens, termination_id=-1, top_k=1)
    engine.add_request(
        request_id=request_id,
        prompt=prompt or [5, IMAGE_TOKEN_ID, 6],
        sampling_params=params,
        **media,
    )
    admitted = engine.get_request(request_id)
    admission = {
        "key": admitted.media_cache_key,
        "tokens": admitted.prompt_tokens.detach().cpu().clone(),
        "mask": admitted.image_token_mask.detach().cpu().clone(),
        "embeddings": admitted.image_embeddings.detach().cpu().clone(),
    }
    finished = _drain(engine)
    assert len(finished) == 1 and len(finished[0].generated_tokens) == num_tokens
    matching_steps = [step for step in trace if step["request_ids"] == [request_id]]
    matching_steps = [step for step in matching_steps if step["embeddings"] is not None]
    assert len(matching_steps) == 1
    return admission, matching_steps[0], finished[0]


def _multiple_videos(offset):
    patch_features = 3 * 4 * 4
    imgs = torch.linspace(-1, 1, 14 * patch_features, dtype=torch.float32).reshape(
        1, 14, patch_features
    )
    fields = {
        "imgs": imgs + offset,
        "imgs_sizes": torch.tensor([[8, 8], [8, 8], [8, 12]], dtype=torch.int32),
        "num_frames": torch.tensor([2, 1], dtype=torch.int32),
        "video_frame_indices": [[0, 3], [5]],
        "video_fps": [30.0, 24.0],
    }
    fields["media_cache_key"] = compute_media_cache_key("video", fields)
    return fields


def _two_images():
    first = _media("image", offset=0.0)["imgs"]
    second = _media("image", offset=0.5)["imgs"]
    fields = {
        "imgs": torch.cat((first, second)),
        "num_tiles": torch.tensor([1, 1], dtype=torch.int32),
        "num_img_embeddings_per_tile": 4,
    }
    fields["media_cache_key"] = compute_media_cache_key("image", fields)
    return fields


def _dynamic_image(offset):
    patch_features = 3 * 4 * 4
    fields = {
        "imgs": torch.linspace(-1, 1, 4 * patch_features, dtype=torch.float32).reshape(
            1, 4, patch_features
        )
        + offset,
        "imgs_sizes": torch.tensor([[8, 8]], dtype=torch.int32),
    }
    fields["media_cache_key"] = compute_media_cache_key("image", fields)
    return fields


@pytest.mark.parametrize("modality", ["image", "video"])
def test_real_encoder_cache_reuse_matches_uncached_and_separates_distinct_media(
    model_parallel, modality, tmp_path
):
    engine, wrapper = _build_engine(modality)
    trace = _install_forward_trace(engine)
    first_media, distinct_media = _media(modality, 0.0), _media(modality, 0.5)
    with _encoder_patch(wrapper) as encoder:
        first, first_step, first_finished = _run_request(engine, 10, first_media, trace)
        reuse, reuse_step, reuse_finished = _run_request(
            engine, 11, {"media_cache_key": first_media["media_cache_key"]}, trace
        )
        distinct, distinct_step, distinct_finished = _run_request(engine, 12, distinct_media, trace)
        if modality == "video":
            frame_paths = []
            for index, color in enumerate(((32, 64, 96), (96, 64, 32))):
                path = tmp_path / f"frame-{index}.png"
                Image.new("RGB", (8, 8), color=color).save(path)
                frame_paths.append(str(path))
            manifest = (
                b"frames:"
                + json.dumps(
                    {
                        "frame_paths": frame_paths,
                        "metadata": {"frames_indices": [0, 3], "fps": 30.0},
                    }
                ).encode()
            )
            video_config = VideoProcessingConfig(
                image_config=ImageProcessingConfig(
                    patch_dim=4,
                    dynamic_resolution=True,
                    dynamic_resolution_min_patches=4,
                    dynamic_resolution_max_patches=4,
                    vision_model_type="radio",
                ),
                num_frames=2,
                frame_manifest_magic=b"frames:",
            )
            raw_args = resolve_multimodal_data_for_engine(
                serialize_multimodal_data({"video": [manifest]}),
                video_preprocessing_config=video_config,
            )
            tensor_args = {
                key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
                for key, value in raw_args.items()
                if key != "media_cache_key"
            }
            tensor_args["media_cache_key"] = compute_media_cache_key("video", tensor_args)
            raw, raw_step, raw_finished = _run_request(engine, 15, raw_args, trace)
            tensor, tensor_step, tensor_finished = _run_request(engine, 16, tensor_args, trace)
    assert encoder.call_count == (4 if modality == "video" else 2)
    assert first["key"] == reuse["key"] != distinct["key"]
    _same(first, first_step, first_finished, reuse, reuse_step, reuse_finished)
    assert torch.equal(first_step["input_ids"], reuse_step["input_ids"])
    assert torch.equal(first_step["mask"], reuse_step["mask"])
    assert torch.equal(first["tokens"], distinct["tokens"])
    assert torch.equal(first["mask"], distinct["mask"])
    assert not torch.equal(first["embeddings"], distinct["embeddings"])
    assert not torch.equal(first_step["logits"], distinct_step["logits"])
    if modality == "video":
        assert raw_args["video_frame_indices"] == [[0, 3]] and raw_args["video_fps"] == [30.0]
        assert raw["key"] != tensor["key"]
        assert torch.equal(raw["tokens"], tensor["tokens"])
        _same(raw, raw_step, raw_finished, tensor, tensor_step, tensor_finished)


@pytest.mark.parametrize(
    ("vision_model_type", "dynamic", "expected_embeddings"),
    [
        ("siglip", False, 4),
        ("internvit", False, 4),
        ("internvit300M", False, 4),
        ("radio-g", True, 4),
        ("cradio-g", True, 4),
        ("pixtral-vit", True, 4),
        ("pixtral-vit-large", True, 1),
        ("qwen-vl", True, 1),
        ("kimi-vit", True, 1),
    ],
)
def test_native_vision_family_real_encoder_reuse_and_distinct_media(
    model_parallel, vision_model_type, dynamic, expected_embeddings
):
    model = _build_model("image", vision_model_type=vision_model_type)
    engine, wrapper = _build_engine("image", model=model)
    trace = _install_forward_trace(engine)
    media_factory = _dynamic_image if dynamic else lambda offset: _media("image", offset)
    first_media = media_factory(0.0)
    distinct_media = media_factory(0.5)
    with _encoder_patch(wrapper) as encoder:
        first, first_step, first_finished = _run_request(engine, 170, first_media, trace)
        reuse, reuse_step, reuse_finished = _run_request(
            engine, 171, {"media_cache_key": first_media["media_cache_key"]}, trace
        )
        distinct, distinct_step, distinct_finished = _run_request(
            engine, 172, distinct_media, trace
        )
    assert encoder.call_count == 2
    assert type(model.vision_model).__name__ in {
        "CLIPViTModel",
        "RADIOViTModel",
        "ViTModel",
        "QwenVLViTModel",
        "KimiViTModel",
    }
    assert first["mask"].ge(0).sum().item() == expected_embeddings
    _same(first, first_step, first_finished, reuse, reuse_step, reuse_finished)
    assert not torch.equal(first["embeddings"], distinct["embeddings"])
    assert not torch.equal(first_step["logits"], distinct_step["logits"])


@pytest.mark.parametrize("separate_video_embedder", [False, True])
def test_temporal_tubelet_encoder_reuse_and_exact_media_rows(
    model_parallel, separate_video_embedder
):
    model = _build_model(
        "video", temporal_patch_dim=2, separate_video_embedder=separate_video_embedder
    )
    engine, wrapper = _build_engine("video", model=model)
    trace = _install_forward_trace(engine)
    first_media, distinct_media = _media("video", 0.0), _media("video", 0.5)
    with (
        _encoder_patch(wrapper) as encoder,
        mock.patch.object(
            model.vision_model,
            "_apply_temporal_grouping",
            wraps=model.vision_model._apply_temporal_grouping,
        ) as grouping,
    ):
        first, first_step, first_finished = _run_request(engine, 178, first_media, trace)
        reuse, reuse_step, reuse_finished = _run_request(
            engine, 179, {"media_cache_key": first_media["media_cache_key"]}, trace
        )
        distinct, distinct_step, distinct_finished = _run_request(
            engine, 180, distinct_media, trace
        )
    assert encoder.call_count == grouping.call_count == 2
    assert model.vision_model.temporal_patch_dim == 2
    assert hasattr(model.vision_model, "video_embedder") is separate_video_embedder
    assert first["mask"].ge(0).sum().item() == 4
    assert first["mask"][first["mask"] >= 0].tolist() == list(range(4))
    _same(first, first_step, first_finished, reuse, reuse_step, reuse_finished)
    assert not torch.equal(first["embeddings"], distinct["embeddings"])
    assert not torch.equal(first_step["logits"], distinct_step["logits"])


@pytest.mark.parametrize(("modality", "expected_embeddings"), [("image", 1), ("video", 2)])
def test_nemotron_omni_real_pixel_shuffle_encoder_reuse_and_prompt_ownership(
    model_parallel, modality, expected_embeddings
):
    model = _attach_omni_encoder_api(
        _build_model(modality, vision_model_type="radio", pixel_shuffle=True)
    )
    engine, wrapper = _build_engine(modality, model=model, wrapper_cls=NemotronOmniInferenceWrapper)
    trace = _install_forward_trace(engine)
    media_factory = (
        _dynamic_image if modality == "image" else lambda offset: _media("video", offset)
    )
    first_media = media_factory(0.0)
    distinct_media = media_factory(0.5)
    with _encoder_patch(wrapper) as encoder:
        first, first_step, first_finished = _run_request(engine, 180, first_media, trace)
        reuse, reuse_step, reuse_finished = _run_request(
            engine, 181, {"media_cache_key": first_media["media_cache_key"]}, trace
        )
        distinct, distinct_step, distinct_finished = _run_request(
            engine, 182, distinct_media, trace
        )
    assert encoder.call_count == 2 and model._pixel_shuffle
    assert first["mask"].ge(0).sum().item() == expected_embeddings
    assert first["embeddings"].shape == (expected_embeddings, 1, model.config.hidden_size)
    assert first["mask"][first["mask"] >= 0].tolist() == list(range(expected_embeddings))
    if modality == "video":
        assert first["tokens"].numel() > expected_embeddings + 2
    _same(first, first_step, first_finished, reuse, reuse_step, reuse_finished)
    assert not torch.equal(first["embeddings"], distinct["embeddings"])
    assert not torch.equal(first_step["logits"], distinct_step["logits"])


def test_multiple_videos_preexpanded_equivalence_and_real_cache_eviction(model_parallel):
    engine, wrapper = _build_engine("video")
    trace = _install_forward_trace(engine)
    compact_prompt = [5, IMAGE_TOKEN_ID, 6, IMAGE_TOKEN_ID, 7]
    first_media = _multiple_videos(0.0)
    distinct_media = _multiple_videos(0.5)
    with _encoder_patch(wrapper) as encoder:
        compact, compact_step, compact_finished = _run_request(
            engine, 20, first_media, trace, compact_prompt
        )
        assert compact["mask"].ge(0).sum().item() == 14
        assert compact["mask"][1:9].tolist() == list(range(8))
        assert compact["mask"][10:16].tolist() == list(range(8, 14))
        engine.clear_vision_embedding_cache()
        preexpanded_media = {**first_media, "media_tokens_preexpanded": True}
        preexpanded_prompt = compact["tokens"].clone()
        preexpanded_prompt[compact["mask"] >= 0] = wrapper.get_preexpanded_media_token_id("video")
        expanded, expanded_step, expanded_finished = _run_request(
            engine, 21, preexpanded_media, trace, preexpanded_prompt.tolist()
        )
        entry_size = engine._vision_cache_entry_nbytes(
            engine._vision_embedding_cache[first_media["media_cache_key"]]
        )
        engine.vision_embedding_cache_max_bytes = entry_size
        _run_request(engine, 22, distinct_media, trace, compact_prompt)
        assert list(engine._vision_embedding_cache) == [distinct_media["media_cache_key"]]
        replay, replay_step, replay_finished = _run_request(
            engine, 23, first_media, trace, compact_prompt
        )
    assert encoder.call_count == 4
    assert torch.equal(expanded["tokens"], preexpanded_prompt)
    assert torch.equal(
        compact["tokens"][compact["mask"] < 0], expanded["tokens"][expanded["mask"] < 0]
    )
    _same(compact, compact_step, compact_finished, expanded, expanded_step, expanded_finished)
    _same(compact, compact_step, compact_finished, replay, replay_step, replay_finished)


def test_expanded_media_prefix_stitching_matches_compact_real_model(model_parallel):
    engine, wrapper = _build_engine("image")
    trace = _install_forward_trace(engine)
    media = _two_images()
    compact_prompt = [5, IMAGE_TOKEN_ID, 6, 7, IMAGE_TOKEN_ID, 8]
    stitched_prompt = (
        [5] + [wrapper.get_preexpanded_media_token_id("image")] * 4 + [6, 7, IMAGE_TOKEN_ID, 8]
    )
    stitching = {PREFIX_MEDIA_COUNT_FIELD: 1, PREFIX_EXPANDED_TOKEN_COUNT_FIELD: 6}
    with _encoder_patch(wrapper) as encoder:
        compact, compact_step, compact_finished = _run_request(
            engine, 24, media, trace, compact_prompt
        )
        stitched, stitched_step, stitched_finished = _run_request(
            engine,
            25,
            {"media_cache_key": media["media_cache_key"], "offload_params": stitching},
            trace,
            stitched_prompt,
        )
    assert encoder.call_count == 1
    assert compact["mask"].tolist() == [-1, 0, 1, 2, 3, -1, -1, 4, 5, 6, 7, -1]
    assert torch.equal(compact["mask"], stitched["mask"])
    normalized_stitched_tokens = stitched["tokens"].clone()
    normalized_stitched_tokens[stitched["mask"] >= 0] = IMAGE_TOKEN_ID
    assert torch.equal(compact["tokens"], normalized_stitched_tokens)
    _same(compact, compact_step, compact_finished, stitched, stitched_step, stitched_finished)


@pytest.mark.parametrize(
    ("prompt", "expected_offsets", "expected_query_lengths"),
    [
        ([5, IMAGE_TOKEN_ID, 6], [[0], [4], [8]], [[4], [4], [2]]),
        ([5, 6, 7, 8, IMAGE_TOKEN_ID, 9, 10, 11, 12], [[0], [4], [8], [12]], [[4]] * 4),
    ],
)
def test_chunk_boundaries_inside_media_match_unchunked_real_model(
    model_parallel, prompt, expected_offsets, expected_query_lengths
):
    model = _build_model("video")
    baseline_engine, baseline_wrapper = _build_engine("video", model=model)
    baseline_trace = _install_forward_trace(baseline_engine)
    media = _media("video", offset=0.0)
    with _encoder_patch(baseline_wrapper) as baseline_encoder:
        baseline, baseline_step, baseline_finished = _run_request(
            baseline_engine, 30, media, baseline_trace, prompt
        )
    chunk_engine, chunk_wrapper = _build_engine(
        "video", model=model, enable_chunked_prefill=True, max_tokens=4
    )
    chunk_trace = _install_forward_trace(chunk_engine)
    params = SamplingParams(num_tokens_to_generate=1, termination_id=-1, temperature=1.0, top_k=1)
    with _encoder_patch(chunk_wrapper) as chunk_encoder:
        chunk_engine.add_request(request_id=31, prompt=prompt, sampling_params=params, **media)
        chunk_admitted = chunk_engine.get_request(31)
        chunk_finished = _drain(chunk_engine)
    request_steps = [step for step in chunk_trace if step["request_ids"] == [31]]
    assert baseline_encoder.call_count == chunk_encoder.call_count == 1
    assert [step["kv_offsets"] for step in request_steps] == expected_offsets
    assert [step["query_lengths"] for step in request_steps] == expected_query_lengths
    consumed_media_indices = torch.cat([step["mask"][step["mask"] >= 0] for step in request_steps])
    assert consumed_media_indices.tolist() == list(range(8))
    assert all(torch.equal(step["embeddings"], baseline["embeddings"]) for step in request_steps)
    assert torch.equal(chunk_admitted.prompt_tokens.cpu(), baseline["tokens"])
    assert torch.equal(request_steps[-1]["logits"], baseline_step["logits"])
    assert len(chunk_finished) == 1
    assert chunk_finished[0].generated_tokens == baseline_finished.generated_tokens


def test_async_mixed_text_media_batch_preserves_ownership_through_middle_completion(
    model_parallel, record_property
):
    model = _build_model("image")
    engine, wrapper = _build_engine("image", model=model)
    trace = _install_forward_trace(engine)
    first_media, second_media = _media("image", 0.0), _media("image", 0.5)
    params = {
        request_id: SamplingParams(
            num_tokens_to_generate=count, termination_id=-1, temperature=1.0, top_k=1
        )
        for request_id, count in ((40, 1), (41, 2), (42, 3))
    }
    with _encoder_patch(wrapper) as encoder:
        engine.add_request(40, [5, IMAGE_TOKEN_ID, 6], params[40], **first_media)
        first_embedding = engine.get_request(40).image_embeddings.detach().cpu().clone()
        engine.add_request(41, [8, 9, 10], params[41])
        engine.add_request(42, [5, IMAGE_TOKEN_ID, 6], params[42], **second_media)
        second_embedding = engine.get_request(42).image_embeddings.detach().cpu().clone()
        batched_finished = []
        finish_steps = {}
        step_index = 0
        while engine.has_unfinished_requests():
            for request in engine.step_modern()["finished_requests"]:
                batched_finished.append(request)
                finish_steps[request.request_id] = step_index
            step_index += 1
    first_step = trace[0]
    assert encoder.call_count == 2
    assert first_step["request_ids"] == [40, 41, 42]
    assert first_step["query_lengths"] == [6, 3, 6]
    assert first_step["mask"][0, :6].tolist() == [-1, 0, 1, 2, 3, -1]
    assert first_step["mask"][0, 6:9].tolist() == [-1, -1, -1]
    assert first_step["mask"][0, 9:15].tolist() == [-1, 4, 5, 6, 7, -1]
    expected_embeddings = torch.cat([first_embedding, second_embedding], dim=0)
    assert torch.equal(first_step["embeddings"], expected_embeddings)
    assert engine.context.async_sched_step_count > 0
    assert finish_steps[40] < finish_steps[41] < finish_steps[42]
    baseline_engine, _ = _build_engine("image", model=model)
    _install_forward_trace(baseline_engine)
    baseline_finished = {}
    for request_id, prompt, media in (
        (50, [5, IMAGE_TOKEN_ID, 6], first_media),
        (51, [8, 9, 10], {}),
        (52, [5, IMAGE_TOKEN_ID, 6], second_media),
    ):
        count = request_id - 49
        baseline_engine.add_request(
            request_id,
            prompt,
            SamplingParams(
                num_tokens_to_generate=count, termination_id=-1, temperature=1.0, top_k=1
            ),
            **media,
        )
        while baseline_engine.has_unfinished_requests():
            for request in baseline_engine.step_modern()["finished_requests"]:
                baseline_finished[request_id] = request.generated_tokens
    assert [request.generated_tokens for request in batched_finished] == [
        baseline_finished[50],
        baseline_finished[51],
        baseline_finished[52],
    ]
    record_property("async_request_id_trace", repr([step["request_ids"] for step in trace]))
    assert any(step["request_ids"] == [42] for step in trace)


def test_real_media_prefix_cache_reuses_only_matching_text_and_media(
    model_parallel, record_property
):
    engine, wrapper = _build_engine("image", enable_prefix_caching=True)
    trace = _install_forward_trace(engine)
    first_media, distinct_media = _media("image", 0.0), _media("image", 0.5)
    with _encoder_patch(wrapper) as encoder:
        first, first_step, first_finished = _run_request(engine, 60, first_media, trace)
        skipped_after_first = engine._prefill_tokens_skipped
        different_text, _, _ = _run_request(
            engine,
            61,
            {"media_cache_key": first_media["media_cache_key"]},
            trace,
            [9, IMAGE_TOKEN_ID, 6],
        )
        skipped_after_different_text = engine._prefill_tokens_skipped
        matching, matching_step, matching_finished = _run_request(
            engine, 62, {"media_cache_key": first_media["media_cache_key"]}, trace
        )
        skipped_after_match = engine._prefill_tokens_skipped
        distinct, _, _ = _run_request(engine, 63, distinct_media, trace)
        skipped_after_distinct = engine._prefill_tokens_skipped
    assert encoder.call_count == 2
    assert skipped_after_first == skipped_after_different_text == 0
    assert skipped_after_match == 4
    assert skipped_after_distinct == skipped_after_match
    assert torch.equal(first["embeddings"], different_text["embeddings"])
    assert torch.equal(first["embeddings"], matching["embeddings"])
    assert not torch.equal(first["embeddings"], distinct["embeddings"])
    assert matching_step["kv_offsets"] == [4]
    assert matching_step["query_lengths"] == [2]
    max_abs = (matching_step["logits"] - first_step["logits"]).abs().max().item()
    record_property("bf16_prefix_cache_logit_max_abs", f"{max_abs:.8g}")
    torch.testing.assert_close(matching_step["logits"], first_step["logits"], rtol=0.01, atol=0.01)
    assert matching_finished.generated_tokens == first_finished.generated_tokens


def test_recompute_refit_refreshes_real_media_without_changing_output(model_parallel):
    model = _build_model("image")
    media = _media("image", offset=0.0)
    baseline_engine, _ = _build_engine("image", model=model)
    baseline_engine.add_request(
        70,
        [5, IMAGE_TOKEN_ID, 6],
        SamplingParams(num_tokens_to_generate=3, termination_id=-1, top_k=1),
        **media,
    )
    baseline_finished = _drain(baseline_engine)
    treated_engine, treated_wrapper = _build_engine(
        "image", model=model, kv_cache_management_mode="recompute", static_kv_memory_pointers=False
    )
    treated_trace = _install_forward_trace(treated_engine)
    with _encoder_patch(treated_wrapper) as encoder:
        treated_engine.add_request(
            71,
            [5, IMAGE_TOKEN_ID, 6],
            SamplingParams(num_tokens_to_generate=3, termination_id=-1, top_k=1),
            **media,
        )
        original_embedding = treated_engine.get_request(71).image_embeddings.detach().cpu().clone()
        first_result = treated_engine.step_modern()
        assert not first_result["finished_requests"]
        treated_engine.suspend()
        assert not treated_engine._vision_embedding_cache
        assert treated_engine.get_request(71).image_embeddings is None
        treated_engine.resume()
        refreshed_request = treated_engine.get_request(71)
        refreshed_embedding = refreshed_request.image_embeddings.detach().cpu().clone()
        treated_finished = _drain(treated_engine)
    assert encoder.call_count == 2 and treated_engine._weight_epoch == 1
    assert torch.equal(original_embedding, refreshed_embedding)
    assert len(baseline_finished) == len(treated_finished) == 1
    assert treated_finished[0].generated_tokens == baseline_finished[0].generated_tokens
    assert any(step["kv_offsets"] == [0] for step in treated_trace)


@pytest.mark.parametrize("ssm_mixer", ["mamba", "gdn", "gdp"])
def test_real_media_executes_hybrid_language_mixer_on_prefill(model_parallel, ssm_mixer):
    if ssm_mixer == "gdn" and not HAVE_FLA:
        pytest.skip("GDN requires FLA")
    skip_if_sequence_packing_not_available(ssm_mixer)
    model = _build_hybrid_model(ssm_mixer)
    engine, wrapper = _build_engine("image", model=model)
    trace = _install_forward_trace(engine)
    mixer_class = {"mamba": "MambaMixer", "gdn": "GatedDeltaNet", "gdp": "GatedDeltaProductMixer"}[
        ssm_mixer
    ]
    mixer_events = []
    handles = [
        module.register_forward_pre_hook(
            lambda _module, args: mixer_events.append(tuple(args[0].shape))
        )
        for module in model.language_model.modules()
        if type(module).__name__ == mixer_class
    ]
    try:
        with _encoder_patch(wrapper) as encoder:
            admission, step, finished = _run_request(engine, 80, _media("image", offset=0.0), trace)
    finally:
        for handle in handles:
            handle.remove()
    assert encoder.call_count == 1
    assert mixer_events
    _assert_step(admission, step, finished)


def test_raw_image_matches_preprocessed_dynamic_input_on_real_model(model_parallel):
    image = Image.new("RGB", (8, 8))
    image.putdata([(x * 16, y * 16, (x + y) * 8) for y in range(8) for x in range(8)])
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    image_config = ImageProcessingConfig(
        patch_dim=4,
        dynamic_resolution=True,
        dynamic_resolution_min_patches=4,
        dynamic_resolution_max_patches=4,
        vision_model_type="radio",
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    raw_wire = serialize_multimodal_data({"image": [payload.getvalue()]})
    raw_args = resolve_multimodal_data_for_engine(raw_wire, image_preprocessing_config=image_config)
    preprocessed_args = {
        "imgs": raw_args["imgs"].clone(),
        "imgs_sizes": raw_args["imgs_sizes"].clone(),
    }
    preprocessed_args["media_cache_key"] = compute_media_cache_key("image", preprocessed_args)
    model = _build_model("video")
    engine, wrapper = _build_engine("video", model=model)
    trace = _install_forward_trace(engine)
    with _encoder_patch(wrapper) as encoder:
        raw, raw_step, raw_finished = _run_request(engine, 90, raw_args, trace)
        preprocessed, preprocessed_step, preprocessed_finished = _run_request(
            engine, 91, preprocessed_args, trace
        )
    assert encoder.call_count == 2
    assert raw["key"] != preprocessed["key"]
    assert raw["tokens"].tolist() == [5] + [IMAGE_TOKEN_ID] * 4 + [6]
    assert torch.equal(raw["tokens"], preprocessed["tokens"])
    _same(raw, raw_step, raw_finished, preprocessed, preprocessed_step, preprocessed_finished)


def test_chat_media_slots_feed_real_preprocessor_encoder_and_exact_positions(model_parallel):
    image = Image.new("RGB", (8, 8), color=(32, 96, 160))
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    encoded = base64.b64encode(payload.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    prompt_config = MultimodalPromptConfig()
    rewritten, image_bytes, video_bytes, media_slots = _extract_multimodal_from_messages(
        messages, prompt_config
    )
    sentinel = media_slots[0][0]

    class ChatTokenizer:
        unk_token_id = -1

        def convert_tokens_to_ids(self, token):
            return IMAGE_TOKEN_ID if token == "<image>" else self.unk_token_id

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [5] if "before" in text else ([6] if "after" in text else [])

    prompt = _tokenize_with_media_slots(
        f"before{sentinel}after", media_slots, ChatTokenizer(), prompt_config
    )
    assert rewritten[0]["content"][1]["text"] == sentinel
    assert not video_bytes and prompt == [5, IMAGE_TOKEN_ID, 6]
    image_config = ImageProcessingConfig(
        patch_dim=4,
        dynamic_resolution=True,
        dynamic_resolution_min_patches=4,
        dynamic_resolution_max_patches=4,
        vision_model_type="radio",
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    media = resolve_multimodal_data_for_engine(
        serialize_multimodal_data({"image": image_bytes}), image_preprocessing_config=image_config
    )
    engine, wrapper = _build_engine("video")
    trace = _install_forward_trace(engine)
    with _encoder_patch(wrapper) as encoder:
        admission, step, finished = _run_request(engine, 95, media, trace, prompt)
    assert encoder.call_count == 1
    assert admission["tokens"].tolist() == [5] + [IMAGE_TOKEN_ID] * 4 + [6]
    assert admission["mask"].tolist() == [-1, 0, 1, 2, 3, -1]
    _assert_step(admission, step, finished)


def test_dynamic_image_row_boundaries_share_real_encoder_step(model_parallel):
    model = _build_model("video")
    image_config = ImageProcessingConfig(
        patch_dim=4, dynamic_resolution=True, image_break_token_id=7, image_end_token_id=8
    )
    engine, wrapper = _build_engine("image", model=model, image_preprocessing_config=image_config)
    trace = _install_forward_trace(engine)
    with _encoder_patch(wrapper) as encoder:
        admission, step, finished = _run_request(engine, 96, _dynamic_image(offset=0.0), trace)
    assert encoder.call_count == 1
    assert admission["tokens"].tolist() == [5] + [IMAGE_TOKEN_ID] * 2 + [7] + [
        IMAGE_TOKEN_ID
    ] * 2 + [8, 6]
    assert admission["mask"].tolist() == [-1, 0, 1, -1, 2, 3, -1, -1]
    _assert_step(admission, step, finished)


@pytest.mark.parametrize("model_parallel", ["tp", "ep", "etp"], indirect=True)
def test_real_media_executes_moe_router_with_configured_expert_topology(model_parallel):
    model = _build_moe_model()
    engine, wrapper = _build_engine("image", model=model)
    trace = _install_forward_trace(engine)
    router_events = []
    handles = [
        module.register_forward_pre_hook(
            lambda _module, args: router_events.append(tuple(args[0].shape))
        )
        for module in model.language_model.modules()
        if type(module).__name__ == "TopKRouter"
    ]
    try:
        with _encoder_patch(wrapper) as encoder:
            admission, step, finished = _run_request(
                engine, 100, _media("image", offset=0.0), trace
            )
    finally:
        for handle in handles:
            handle.remove()
    assert encoder.call_count == 1 and router_events
    cfg = model.config
    assert cfg.expert_model_parallel_size == parallel_state.get_expert_model_parallel_world_size()
    assert cfg.expert_tensor_parallel_size == parallel_state.get_expert_tensor_parallel_world_size()
    _assert_step(admission, step, finished)
    local_logits = step["logits"].cuda().flatten()
    rank_logits = local_logits.new_empty(Utils.world_size * VOCAB_SIZE)
    torch.distributed.all_gather_into_tensor(rank_logits, local_logits)
    rank_logits = rank_logits.view(Utils.world_size, -1)
    assert torch.all(torch.isfinite(rank_logits) & (rank_logits == rank_logits[:1]))


@pytest.mark.parametrize("model_parallel", ["tp", "dp", "ep", "etp", "opt"], indirect=True)
def test_real_media_executes_configured_language_implementation_or_parallel_path(
    model_parallel, record_property
):
    topology = model_parallel
    model = _build_te_parallel_model(topology)
    rotary_type = type(getattr(model.language_model, "rotary_pos_emb", None)).__name__
    record_property("position_embedding_type", rotary_type)
    model.language_model.forward = mock.Mock(wraps=model.language_model.forward)
    engine, wrapper = _build_engine("image", model=model)
    trace = _install_forward_trace(engine)
    optimized_events = []
    optimized_handles = [
        module.register_forward_pre_hook(
            lambda _module, args: optimized_events.append(tuple(args[0].shape))
        )
        for module in model.language_model.modules()
        if topology == "opt"
        and type(module).__name__
        in ("InferenceColumnParallelLinear", "InferenceLayerNormColumnParallelLinear")
    ]
    try:
        with _encoder_patch(wrapper) as encoder:
            media_offset = (
                0.5 * parallel_state.get_data_parallel_rank() if topology == "dp" else 0.0
            )
            admission, step, finished = _run_request(
                engine, 110, _media("image", offset=media_offset), trace
            )
    finally:
        for handle in optimized_handles:
            handle.remove()
    assert encoder.call_count == 1
    if topology == "opt":
        assert optimized_handles and optimized_events
    if topology == "dp":
        assert parallel_state.get_data_parallel_world_size() == Utils.world_size
        local = admission["embeddings"].float().sum().cuda().reshape(1)
        gathered = [torch.empty_like(local) for _ in range(Utils.world_size)]
        torch.distributed.all_gather(
            gathered, local, group=parallel_state.get_data_parallel_group()
        )
        assert len({float(value.item()) for value in gathered}) == Utils.world_size
    assert torch.equal(step["mask"][0, : admission["mask"].numel()], admission["mask"])
    assert torch.equal(step["embeddings"], admission["embeddings"])
    assert step["logits"].shape == (1, 1, VOCAB_SIZE)
    assert torch.isfinite(step["logits"]).all()
    assert finished.generated_tokens[0] == _greedy_token(step)


@pytest.mark.parametrize("variant", ["swa"])
def test_real_media_executes_attention_variant_and_reuses_exact_result(model_parallel, variant):
    model = _build_attention_variant_model(variant)
    block_size_tokens = 4
    engine, wrapper = _build_engine("image", model=model, block_size_tokens=block_size_tokens)
    trace = _install_forward_trace(engine)
    media = _media("image", offset=0.0)
    companion_events = []
    patchers = []
    if variant == "swa":
        original_flash = Attention.flash_decode_and_prefill

        def record_window(attention, *args, **kwargs):
            companion_events.append((attention.config.window_size, attention.layer_number))
            return original_flash(attention, *args, **kwargs)

        patchers.append(mock.patch.object(Attention, "flash_decode_and_prefill", new=record_window))
    for patcher in patchers:
        patcher.start()
    try:
        with _encoder_patch(wrapper) as encoder:
            first, first_step, first_finished = _run_request(
                engine, 120, media, trace, [5, IMAGE_TOKEN_ID, 6, 7, 8, 9]
            )
            reuse, reuse_step, reuse_finished = _run_request(
                engine,
                121,
                {"media_cache_key": media["media_cache_key"]},
                trace,
                [5, IMAGE_TOKEN_ID, 6, 7, 8, 9],
            )
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
    assert encoder.call_count == 1 and companion_events
    if variant == "swa":
        assert all(window == (4, 0) for window, _ in companion_events)
    assert first_step["mask"].ge(0).sum().item() == 4
    _same(first, first_step, first_finished, reuse, reuse_step, reuse_finished)
    assert torch.isfinite(first_step["logits"]).all()


def test_real_media_prefill_replays_actual_cuda_graph_with_eager_parity(model_parallel):
    eager_model = _build_model("image")
    graphed_model = _build_model(
        "image", cuda_graph_impl="local", inference_cuda_graph_scope=InferenceCudaGraphScope.block
    )
    graphed_model.load_state_dict(eager_model.state_dict())
    media = _media("image", offset=0.0)
    eager_engine, eager_wrapper = _build_engine("image", model=eager_model)
    eager_trace = _install_forward_trace(eager_engine)
    with _encoder_patch(eager_wrapper) as eager_encoder:
        eager, eager_step, eager_finished = _run_request(eager_engine, 130, media, eager_trace)
    graph_engine, graph_wrapper = _build_engine(
        "image",
        model=graphed_model,
        num_cuda_graphs=-1,
        # CUDA graphs need max_requests to be a multiple of the tensor-parallel size.
        max_requests=parallel_state.get_tensor_model_parallel_world_size(),
        max_tokens=16,
        cuda_graph_all_prefills=True,
        cuda_graph_max_tokens=16,
    )
    graph_trace = _install_forward_trace(graph_engine)
    with _encoder_patch(graph_wrapper) as graph_encoder:
        graphed, graph_step, graph_finished = _run_request(graph_engine, 131, media, graph_trace)
    assert eager_encoder.call_count == graph_encoder.call_count == 1 and not eager_step["graphed"]
    assert graph_step["graphed"]
    # The graph pads its batch to max_requests, so compare only the real request's logits.
    graph_step = dict(graph_step, logits=graph_step["logits"][:, : eager_step["logits"].shape[1]])
    assert torch.equal(eager["tokens"], graphed["tokens"])
    _same(eager, eager_step, eager_finished, graphed, graph_step, graph_finished)


def test_real_media_prefill_and_mtp_draft_share_step_with_greedy_parity(model_parallel):
    skip_if_sequence_packing_not_available("mamba")
    baseline_model = _build_hybrid_model("mamba")
    mtp_model = _build_hybrid_model("mamba", with_mtp=True)
    incompatible = mtp_model.load_state_dict(baseline_model.state_dict(), strict=False)
    assert incompatible.missing_keys and not incompatible.unexpected_keys
    media = _media("image", offset=0.0)
    params = SamplingParams(num_tokens_to_generate=3, termination_id=-1, top_k=1)
    baseline_engine, baseline_wrapper = _build_engine("image", model=baseline_model)
    with _encoder_patch(baseline_wrapper) as baseline_encoder:
        baseline_engine.add_request(140, [5, IMAGE_TOKEN_ID, 6], params, **media)
        baseline_finished = _drain(baseline_engine)
    mtp_engine, mtp_wrapper = _build_engine("image", model=mtp_model, num_speculative_tokens=1)
    mtp_trace = _install_forward_trace(mtp_engine)
    with (
        _encoder_patch(mtp_wrapper) as mtp_encoder,
        mock.patch.object(
            mtp_model.language_model,
            "compute_mtp_single_step",
            wraps=mtp_model.language_model.compute_mtp_single_step,
        ) as mtp_forward,
    ):
        mtp_engine.add_request(141, [5, IMAGE_TOKEN_ID, 6], params, **media)
        mtp_finished = list(mtp_engine.step_modern()["finished_requests"])
        mtp_finished.extend(mtp_engine.step_modern()["finished_requests"])
        calls_after_media_prefill = mtp_forward.call_count
        mtp_finished.extend(_drain(mtp_engine))
    assert baseline_encoder.call_count == mtp_encoder.call_count == 1
    assert calls_after_media_prefill > 0 and mtp_trace[0]["mask"].ge(0).sum().item() == 4
    assert mtp_trace[0]["embeddings"].shape[0] == 4
    assert len(baseline_finished) == len(mtp_finished) == 1
    assert mtp_finished[0].generated_tokens == baseline_finished[0].generated_tokens


def test_real_media_flashinfer_samples_only_from_requested_top_k(model_parallel):
    pytest.importorskip("flashinfer")
    from megatron.core.inference.sampling import flashinfer_sampling

    engine, wrapper = _build_engine("image", sampling_backend="flashinfer")
    trace = _install_forward_trace(engine)
    params = SamplingParams(num_tokens_to_generate=3, termination_id=-1, temperature=0.8, top_k=8)
    support_checks = []
    original_kernel = flashinfer_sampling.flashinfer.sampling.top_k_sampling_from_probs

    def checked_top_k(probs, top_k, *args, **kwargs):
        sampled = original_kernel(probs, top_k, *args, **kwargs)
        for row, token, k in zip(probs, sampled, top_k):
            support = torch.topk(row, int(k.item())).indices
            support_checks.append(bool((support == token).any().item()))
        return sampled

    with (
        _encoder_patch(wrapper) as encoder,
        mock.patch.object(
            flashinfer_sampling.flashinfer.sampling, "top_k_sampling_from_probs", new=checked_top_k
        ),
    ):
        engine.add_request(150, [5, IMAGE_TOKEN_ID, 6], params, **_media("image", offset=0.0))
        finished = _drain(engine)
    assert encoder.call_count == 1 and support_checks and all(support_checks)
    assert trace[0]["mask"].ge(0).sum().item() == 4
    assert len(finished) == 1
    assert len(finished[0].generated_tokens) == 3


@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
def test_real_media_top_p_seed_and_logprobs_match_distribution(model_parallel, logprobs_mode):
    engine, wrapper = _build_engine("image", logprobs_mode=logprobs_mode)
    trace = _install_forward_trace(engine)
    media = _media("image", offset=0.0)
    temperature, top_p = 0.7, 0.6

    def run(request_id, media_args):
        params = SamplingParams(
            num_tokens_to_generate=3,
            termination_id=-1,
            temperature=temperature,
            top_p=top_p,
            seed=211,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            top_n_logprobs=3,
        )
        engine.add_request(request_id, [5, IMAGE_TOKEN_ID, 6], params, **media_args)
        finished = _drain(engine)
        steps = [step for step in trace if step["request_ids"] == [request_id]]
        assert len(finished) == 1 and len(steps) == len(finished[0].generated_tokens) + 1
        return finished[0], steps

    with _encoder_patch(wrapper) as encoder:
        first, first_steps = run(155, media)
        repeated, repeated_steps = run(156, {"media_cache_key": media["media_cache_key"]})
    assert encoder.call_count == 1
    assert first_steps[0]["mask"].ge(0).sum().item() == 4
    assert first.generated_tokens == repeated.generated_tokens
    assert first.generated_log_probs == repeated.generated_log_probs
    assert first.generated_top_n_logprobs == repeated.generated_top_n_logprobs
    assert all(torch.equal(a["logits"], b["logits"]) for a, b in zip(first_steps, repeated_steps))
    for step, token, reported, top_n in zip(
        first_steps,
        first.generated_tokens,
        first.generated_log_probs,
        first.generated_top_n_logprobs,
    ):
        raw_logits = step["logits"][0, 0].float()
        scaled_logits = raw_logits / temperature
        order = torch.argsort(scaled_logits, descending=True)
        cumulative = torch.softmax(scaled_logits[order], dim=0).cumsum(dim=0)
        crossing = torch.nonzero(cumulative > top_p, as_tuple=False)
        support_count = int(crossing[0].item()) + 1 if crossing.numel() else VOCAB_SIZE
        support = order[:support_count]
        assert int(token) in support.tolist()
        processed_logits = torch.full_like(scaled_logits, float("-inf"))
        processed_logits[support] = scaled_logits[support]
        scored_logits = raw_logits if logprobs_mode == "raw_logprobs" else processed_logits
        expected = torch.log_softmax(scored_logits, dim=0)
        assert reported == pytest.approx(float(expected[token]), abs=1e-6)
        threshold = torch.topk(expected, k=3).values[-1]
        assert len(top_n) == 3
        for token_id, value in top_n.items():
            assert expected[int(token_id)] >= threshold
            assert value == pytest.approx(float(expected[int(token_id)]), abs=1e-6)


def test_real_media_eos_and_explicit_stop_end_at_first_greedy_token(model_parallel):
    engine, wrapper = _build_engine("image")
    trace = _install_forward_trace(engine)
    media = _media("image", offset=0.0)

    def run(request_id, params):
        engine.clear_vision_embedding_cache()
        trace_start = len(trace)
        engine.add_request(request_id, [5, IMAGE_TOKEN_ID, 6], params, **media)
        embedding = engine.get_request(request_id).image_embeddings.detach().cpu().clone()
        finished = _drain(engine)
        assert len(finished) == 1
        return finished[0], embedding, trace[trace_start:]

    with _encoder_patch(wrapper) as encoder:
        baseline, baseline_embedding, baseline_steps = run(
            157, SamplingParams(num_tokens_to_generate=1, termination_id=-1, top_k=1)
        )
        first_token = baseline.generated_tokens[0]
        eos, eos_embedding, eos_steps = run(
            158, SamplingParams(num_tokens_to_generate=5, termination_id=first_token, top_k=1)
        )
        explicit, explicit_embedding, explicit_steps = run(
            159,
            SamplingParams(
                num_tokens_to_generate=5, termination_id=-1, stop_token_ids=[first_token], top_k=1
            ),
        )
    assert encoder.call_count == 3
    assert baseline.generated_tokens == eos.generated_tokens == explicit.generated_tokens
    assert baseline.finish_reason == "length"
    assert eos.finish_reason == explicit.finish_reason == "stop"
    assert torch.equal(baseline_embedding, eos_embedding)
    assert torch.equal(baseline_embedding, explicit_embedding)
    assert all(
        steps[0]["mask"].ge(0).sum().item() == 4
        for steps in (baseline_steps, eos_steps, explicit_steps)
    )
    assert torch.equal(baseline_steps[0]["logits"], eos_steps[0]["logits"])
    assert torch.equal(baseline_steps[0]["logits"], explicit_steps[0]["logits"])
