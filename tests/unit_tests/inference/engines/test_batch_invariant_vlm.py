# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real multimodal coverage for dynamic batch-invariant inference."""

from copy import deepcopy
from functools import partial, wraps

import pytest
import torch

from megatron.core.inference.model_inference_wrappers.multimodal import vlm_inference_wrapper
from megatron.core.models.gpt import gpt_layer_specs
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.inference.engines import batch_invariant_test_utils as bi

MEDIA_TOKEN = 126
CACHE_REQUEST = 102


def _config(layers, hidden, heads, backend, fa_version):
    return TransformerConfig(
        **bi.model_defaults(backend, fa_version),
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        ffn_hidden_size=4 * hidden,
        batch_invariant_collective="ordered",
    )


def _build_engine(case, backend, fa_version, *, modality, engines, wire=False):
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(
        321, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
    )
    language = _config(2, 128, 4, backend, fa_version)
    vision = _config(1, 64, 4, backend, fa_version)
    projection = _config(1, 128, 1, backend, fa_version)
    language.language_model_type = "dummy"
    vision.vision_model_type = "radio"
    submodules = gpt_layer_specs.get_gpt_layer_with_transformer_engine_submodules()
    model = LLaVAModel(
        language_transformer_config=language,
        language_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=deepcopy(submodules)
        ),
        language_vocab_size=bi.VOCAB,
        language_max_sequence_length=256,
        vision_transformer_config=vision,
        vision_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=deepcopy(submodules)
        ),
        drop_vision_class_token=False,
        vision_projection_config=projection,
        vision_projection_layer_spec=deepcopy(get_submodules(submodules.mlp)),
        img_h=28,
        img_w=28,
        patch_dim=14,
        dynamic_resolution=True,
        radio_force_eval_mode=True,
    ).cuda()
    model = Float16Module(language, model).eval()
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    options = dict(
        max_sequence_length=256,
        buffer_size_gb=0.125,
        block_size_tokens=64,
        max_requests=128,
        max_tokens=256,
        materialize_only_last_token_logits=False,
        vision_embedding_cache_max_bytes=1 << 20,
    )
    options.update(case.context)
    context = bi.DynamicInferenceContext(language, bi.InferenceConfig(**options))
    tokenizer = bi.DummyTokenizer(bi.VOCAB, bos=1, eod=bi.VOCAB - 1)
    tokenizer.convert_tokens_to_ids = {"<image>": MEDIA_TOKEN}.get
    wrapper = vlm_inference_wrapper.VLMInferenceWrapper(model, context)
    evidence = []
    vision_forward = wrapper._forward_vision_encoder

    @wraps(vision_forward)
    def observe_vision(images, **kwargs):
        output = vision_forward(images, **kwargs)
        evidence.append(
            {
                "kwargs": {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                },
                "projected": output.detach().clone(),
            }
        )
        return output

    wrapper._forward_vision_encoder = observe_vision
    decoder_inputs, current = [], {}
    dynamic_forward, lm_forward = wrapper._forward_dynamic, model.module.forward_lm_only

    def observe_dynamic(inputs):
        current["mask"] = inputs.get("image_token_mask")  # Text-only decode has no media mask.
        return dynamic_forward(inputs)

    def observe_lm(*args, **kwargs):
        active = context.request_ids[: context.total_request_count]
        if context.num_prefill_requests and bi.TARGET in active:
            decoder_inputs.append(
                (kwargs["combined_embeddings"].detach().clone(), current["mask"].clone())
            )
        return lm_forward(*args, **kwargs)

    wrapper._forward_dynamic, model.module.forward_lm_only = observe_dynamic, observe_lm
    engine = bi.DynamicInferenceEngine(bi.TextGenerationController(wrapper, tokenizer), context)
    frames = 2 if modality == "video" else 1
    media = torch.arange(frames * 4 * 3 * 14 * 14, dtype=torch.float32)
    media = ((media % 31) / 16 - 1).to(torch.bfloat16).reshape(1, frames * 4, -1)
    media_kwargs = {
        "imgs": media,
        "imgs_sizes": torch.tensor([[28, 28]] * frames, dtype=torch.int32),
    }
    if modality == "video":
        media_kwargs["num_frames"] = torch.tensor([frames], dtype=torch.int32)
    add_request = engine.add_request

    @wraps(add_request)
    def add_media(request_id, prompt, *args, **kwargs):
        if request_id in (bi.TARGET, CACHE_REQUEST):
            prompt = list(prompt)
            prompt[3] = MEDIA_TOKEN
            kwargs.update(media_kwargs)
        return add_request(request_id, prompt, *args, **kwargs)

    engine.add_request = add_request if wire else add_media
    engine._bi_wire_media = {modality: media_kwargs}
    engine._bi_media_evidence = evidence
    engine._bi_decoder_inputs = decoder_inputs
    engines.append(engine)
    return engine


def _assert_projected(reference, actual):
    assert reference.shape == actual.shape
    assert torch.equal(reference, actual), "projected media embeddings changed"


def _drain(engine):
    for _ in range(32):
        if not engine.has_unfinished_requests():
            return
        engine.step_modern()
    raise AssertionError("cache-hit request did not drain")


@pytest.mark.parametrize("modality", ["image", "video"])
def test_vlm_dynamic_batch_invariance(modality):
    case = bi.Case(f"vlm-{modality}")
    engines = []
    with bi.invariant_runtime(case) as (backend, version), pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            bi, "build_engine", partial(_build_engine, modality=modality, engines=engines)
        )
        reference = bi.run_order(case, backend, version, "solo")
        actual = bi.run_order(case, backend, version, "back")
        bi.assert_same_target(reference, actual)
        expected, observed = (engine._bi_media_evidence for engine in engines)
        assert len(expected) == len(observed) == 1
        _assert_projected(expected[0]["projected"], observed[0]["projected"])
        frames = 2 if modality == "video" else 1
        assert observed[0]["projected"].shape == (12 * frames, 1, 128)
        assert ("num_frames" in observed[0]["kwargs"]) == (modality == "video")
        for engine in engines:
            assert len(engine._bi_decoder_inputs) == 1, "target media never reached decoder"
            decoder_input, mask = engine._bi_decoder_inputs[0]
            _assert_projected(expected[0]["projected"].squeeze(1), decoder_input[mask >= 0])
        assert {s["physical"] for s in reference[1].steps if s["decode"]} == {64}
        assert any(s["physical"] == 128 and s["target_row"] > 0 for s in actual[1].steps)

        engine = engines[1]
        engine.add_request(
            CACHE_REQUEST,
            bi.target_prompt(case.prompt_length),
            bi.SamplingParams(num_tokens_to_generate=1, top_k=1, termination_id=-1),
        )
        target = engine.get_request(CACHE_REQUEST)
        assert len(observed) == 1 and len(engine._vision_embedding_cache) == 1
        engine._invalidate_vision_state()
        assert not engine._vision_embedding_cache
        assert target.image_embeddings is None
        engine._refresh_vlm_request_data(target)
        assert len(observed) == 2
        _assert_projected(expected[0]["projected"], observed[1]["projected"])
        _drain(engine)

        engine.reset()
        print("BI_VLM_WITNESS", modality, backend, version, "physical", [64, 128])
