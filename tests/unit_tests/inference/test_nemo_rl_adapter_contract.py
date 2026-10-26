# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import copy
import importlib
import os
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode, PrefixCachingEvictionPolicy
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.inference.quantization.utils import quantize_params_to_mxfp8
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_inference_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import attention as attention_module
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.inference.engines import test_dynamic_engine as dynamic_engine_tests
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    DynamicInferenceEngineTestBase,
    reset_rounder,
)
from tests.unit_tests.test_utilities import Utils

NEMO_RL_COMMIT = "b26733300f0f7f5421c43c8ba4cd9e37e5d5f65d"
class _CoordinatorTokenizer:
    vocab_size = 100
    eod, bos = 99, None
    def tokenize(self, prompt):
        return prompt if isinstance(prompt, list) else [int(token) for token in prompt.split()]
    def detokenize(self, tokens):
        return " ".join(map(str, tokens))
def _mount_nemo_rl_generation_packages(source):
    for name, relative_path in (
        ("nemo_rl.models.generation", "nemo_rl/models/generation"),
        ("nemo_rl.models.generation.megatron", "nemo_rl/models/generation/megatron"),
        ("nemo_rl.weight_sync", "nemo_rl/weight_sync"),
    ):
        package = ModuleType(name)
        package.__path__ = [str(source / relative_path)]
        package.__package__ = name
        sys.modules[name] = package
def _load_nemo_rl_adapter():
    source_value = os.environ.get("NEMO_RL_SOURCE")
    if source_value is None:
        pytest.skip("NEMO_RL_SOURCE is supplied by the NeMo-RL acceptance worker")
    assert os.environ.get("NEMO_RL_SOURCE_COMMIT") == NEMO_RL_COMMIT
    source = Path(source_value).resolve(strict=True)
    sys.path.insert(0, str(source))
    try:
        data_module = importlib.import_module("nemo_rl.distributed.batched_data_dict")
        _mount_nemo_rl_generation_packages(source)
        worker_module = importlib.import_module(
            "nemo_rl.models.generation.megatron.megatron_worker"
        )
    finally:
        sys.path.remove(str(source))
    assert Path(worker_module.__file__).resolve().is_relative_to(source)
    assert Path(data_module.__file__).resolve().is_relative_to(source)
    adapter_cls = type(
        "_TestMegatronPolicyAdapter",
        (worker_module.MegatronGenerationMixin, worker_module.MegatronGenerationRefitMixin),
        {},
    )
    return data_module.BatchedDataDict, adapter_cls
def _load_nemo_rl_token_capture():
    source_value = os.environ.get("NEMO_RL_SOURCE")
    if source_value is None:
        pytest.skip("NEMO_RL_SOURCE is supplied by the NeMo-RL acceptance worker")
    assert os.environ.get("NEMO_RL_SOURCE_COMMIT") == NEMO_RL_COMMIT
    source = Path(source_value).resolve(strict=True)
    gym_source = Path(
        os.environ.get("NEMO_GYM_SOURCE", source / "3rdparty" / "Gym-workspace" / "Gym")
    ).resolve(strict=True)
    sys.path[:0] = [str(source), str(gym_source)]
    try:
        _mount_nemo_rl_generation_packages(source)
        capture_module = importlib.import_module("nemo_rl.models.generation.megatron.token_capture")
        gym_module = importlib.import_module("nemo_gym.token_id_capture")
        records_module = importlib.import_module("nemo_gym.token_id_capture.staging.records")
    finally:
        sys.path.remove(str(gym_source))
        sys.path.remove(str(source))
    assert Path(capture_module.__file__).resolve().is_relative_to(source)
    assert Path(gym_module.__file__).resolve().is_relative_to(gym_source)
    return capture_module, gym_module, records_module
def _protocol_adapter(mixin_cls, *, stop_token_ids=None, ignore_eos=False):
    adapter = object.__new__(mixin_cls)
    adapter.cfg = {
        "generation": {
            "temperature": 0.7,
            "top_k": None,
            "top_p": None,
            "max_new_tokens": 3,
            "ignore_eos": ignore_eos,
            "stop_token_ids": stop_token_ids,
            "stop_strings": None,
        }
    }
    adapter.tokenizer = SimpleNamespace(pad_token_id=0)
    adapter.megatron_tokenizer = SimpleNamespace(eod=99)
    return adapter
def _reply(tokens, scores, reason, prompt=None):
    return SimpleNamespace(
        prompt_tokens=prompt,
        generated_tokens=tokens,
        generated_log_probs=scores,
        finish_reason=reason,
    )
def _assert_exact_scores(actual, expected):
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
async def _wait_for_epoch(engine, epoch):
    for _ in range(100):
        if engine._generation_epoch == epoch:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"generation epoch did not reach {epoch}")
async def _drive_engine(engine, futures, witness=None):
    step_count = 0
    while not all(future.done() for future in futures):
        with torch.inference_mode():
            step = await engine.async_step()
        if witness is not None and step["cuda_graph_request_count"] is not None:
            request_ids = tuple(map(int, step["active_request_ids"]))
            graph_ids = witness["cuda_graph_request_ids"]
            finished_ids = tuple(item.request_id for item in step["finished_requests"])
            graph_ids.append(request_ids + finished_ids)
        step_count += 1
        assert step_count < 100, "dynamic engine did not drain the adapter requests"
        await asyncio.sleep(0)
    return await asyncio.gather(*futures)
def _build_inference_mxfp8_env(test_case, config, *, quantize=True):
    config_class = dynamic_engine_tests.TransformerConfig
    def build_mxfp8_config(**kwargs):
        kwargs.update(fp8_recipe="mxfp8", fp8_param=True)
        return config_class(**kwargs)
    with (
        mock.patch.object(dynamic_engine_tests, "TransformerConfig", build_mxfp8_config),
        mock.patch.object(
            dynamic_engine_tests,
            "get_gpt_layer_with_transformer_engine_spec",
            lambda num_experts=None: get_gpt_layer_with_inference_spec(num_experts=num_experts),
        ),
    ):
        env = test_case._build_test_env(config)
    model = env.engine.controller.inference_wrapped_model.model
    assert any(type(module).__name__.startswith("Inference") for module in model.modules())
    buffers = quantize_params_to_mxfp8(model.decoder, backend="triton") if quantize else None
    assert not quantize or buffers
    return env
class _InProcessInferenceClient:
    def __init__(self, engine, expected_requests, first_request_id, witness):
        self.engine = engine
        self.expected_requests = expected_requests
        self.next_request_id = first_request_id
        self.witness = witness
        self.pending = []
        self.driver = None
    def add_request_with_id(self, prompt, sampling_params, *, multi_modal_data=None):
        assert multi_modal_data is None
        request_id = self.next_request_id
        self.next_request_id += 1
        self.witness["adapter_submissions"].append(
            {
                "request_id": request_id,
                "prompt": tuple(prompt),
                "temperature": sampling_params.temperature,
                "top_k": sampling_params.top_k,
                "top_p": sampling_params.top_p,
            }
        )
        future = self.engine.add_request(request_id, prompt, sampling_params)
        self.pending.append(future)
        if len(self.pending) == self.expected_requests:
            drive = _drive_engine(self.engine, self.pending, self.witness)
            self.driver = asyncio.create_task(drive)
        return request_id, future
    def abort_request(self, request_id):  # pragma: no cover - failure path assertion
        raise AssertionError(f"unexpected adapter abort for request {request_id}")
class TestNeMoRLAdapterContract(DynamicInferenceEngineTestBase):
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)
    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        reset_rounder()
        Utils.destroy_model_parallel()
    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "feature_config",
        [
            {},
            {
                "context_block_size_tokens": 256,
                "context_max_tokens": 1024,
                "enable_prefix_caching": True,
                "prefix_caching_eviction_policy": PrefixCachingEvictionPolicy.REF_ZERO,
            },
            {"model_provider": "hybrid", "ssm_mixer": "mamba"},
            {"model_provider": "hybrid", "ssm_mixer": "gdn"},
            {"model_provider": "hybrid", "ssm_mixer": "gdp"},
            {"transformer_impl": "transformer_engine"},
            {"transformer_impl": "inference_optimized"},
            {"fp8": True},
            {"fp8": True, "mxfp8": True, "transformer_impl": "inference_optimized"},
            {"window_size": (4, 0), "softmax_type": "off-by-one"},
            {
                "num_cuda_graphs": 2,
                "force_build_cuda_graphs": True,
                "use_cuda_graphs_for_non_decode_steps": False,
            },
            {"num_speculative_tokens": 1, "mtp_use_repeated_layer": True},
            {"adapter_sampling": {"temperature": 0.73, "top_k": 1, "top_p": 0.91}},
            {"logprobs_mode": "processed_logprobs", "materialize_only_last_token_logits": False},
            dict(hidden_size=256, position_embedding_type="rope", use_flashinfer_fused_rope=True),
        ],
        ids=[
            "dense",
            "prefix-ref-zero",
            "mamba-hybrid",
            "gdn-hybrid",
            "gdp-hybrid",
            "transformer-engine",
            "inference-optimized",
            "fp8",
            "mxfp8",
            "sliding-window-sink",
            "cuda-graphs",
            "mtp",
            "non-greedy-parameters",
            "processed-logprobs",
            "flashinfer-fused-rope",
        ],
    )
    async def test_direct_and_real_adapter_greedy_outputs_match(self, feature_config, monkeypatch):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        stresses_chunking = feature_config.get("enable_chunked_prefill", False)
        stresses_prefix = feature_config.get("enable_prefix_caching", False)
        adapter_sampling = feature_config.get("adapter_sampling")
        uses_mla = feature_config.get("mla", False)
        uses_mxfp8 = feature_config.get("mxfp8", False)
        if uses_mla and not attention_module.HAVE_FMLA:
            pytest.skip("dynamic latent-cache MLA requires FlashMLA")
        if stresses_prefix:
            prefix_len, suffix_lens = (256, (258, 260)) if stresses_chunking else (256, (2, 4))
            shared_prefix = [index % 89 + 1 for index in range(prefix_len)]
            prompts = [
                shared_prefix + [(index + shift) % 89 + 1 for index in range(length)]
                for shift, length in zip((0, 41), suffix_lens, strict=True)
            ]
        else:
            prompts = [[3, 5, 7, 9], [2, 4, 6, 8, 10, 12]]
        config = DynamicEngineTestConfig(
            num_requests=0,
            max_prompt_length=max(map(len, prompts)),
            max_sequence_length=max(map(len, prompts)) + 8,
            context_max_requests=8,
            context_max_tokens=feature_config.get("context_max_tokens", 64),
            num_gap_steps=0,
            sampling_backend="flashinfer",
            async_sched_mode=AsyncScheduleMode.ASYNC,
            flash_attention_version=None if uses_mla else 2,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            logprobs_mode=feature_config.get("logprobs_mode", "raw_logprobs"),
            **{
                key: value
                for key, value in feature_config.items()
                if key
                not in ("adapter_sampling", "context_max_tokens", "logprobs_mode", "mla", "mxfp8")
            },
        )
        if uses_mla:
            def build_mla_config(**kwargs):
                kwargs.update(
                    hidden_size=512,
                    num_attention_heads=8,
                    add_bias_linear=False,
                    normalization="RMSNorm",
                    qk_layernorm=True,
                    multi_latent_attention=True,
                    q_lora_rank=128,
                    kv_lora_rank=512,
                    qk_head_dim=128,
                    qk_pos_emb_head_dim=64,
                    v_head_dim=128,
                    rope_type="rope",
                    cache_mla_latents=True,
                )
                return MLATransformerConfig(**kwargs)
            with (
                mock.patch.object(dynamic_engine_tests, "TransformerConfig", build_mla_config),
                mock.patch.object(
                    dynamic_engine_tests,
                    "get_gpt_layer_with_transformer_engine_spec",
                    lambda num_experts=None: get_gpt_layer_with_transformer_engine_spec(
                        num_experts=num_experts, qk_layernorm=True, multi_latent_attention=True
                    ),
                ),
                mock.patch.object(
                    dynamic_engine_tests,
                    "get_gpt_layer_local_spec",
                    lambda num_experts=None: ModuleSpec(
                        module=TransformerLayer,
                        submodules=get_gpt_layer_local_submodules(
                            num_experts=num_experts,
                            qk_layernorm=True,
                            multi_latent_attention=True,
                            normalization="RMSNorm",
                        ),
                    ),
                ),
            ):
                env = self._build_test_env(config)
        elif uses_mxfp8:
            env = _build_inference_mxfp8_env(self, config)
        else:
            env = self._build_test_env(config)
        engine = env.engine
        assert not uses_mla or engine.context.cache_mla_latent
        if uses_mxfp8:
            model_config = engine.controller.inference_wrapped_model.model.config
            assert "mxfp8" in str(model_config.fp8_recipe).lower()
        generation_length = 4
        witness = {
            "adapter_submissions": [],
            "model_request_ids": [],
            "sampler_request_ids": [],
            "chunked_prefill_request_ids": [],
            "cuda_graph_request_ids": [],
            "mtp_request_ids": [],
            "flash_mla_request_ids": [],
            "fused_rope_request_ids": [],
        }
        def active_request_ids():
            return tuple(engine.context.request_ids[: engine.context.total_request_count].tolist())
        model = engine.controller.inference_wrapped_model.model
        mixer_calls = []
        mixer_name = {
            "mamba": "MambaMixer",
            "gdn": "GatedDeltaNet",
            "gdp": "GatedDeltaProductMixer",
        }.get(feature_config.get("ssm_mixer"))
        mixer_hooks = [
            module.register_forward_hook(lambda _module, _inputs, _output: mixer_calls.append(1))
            for module in model.modules()
            if type(module).__name__ == mixer_name
        ]
        original_forward = model.forward
        def traced_forward(*args, **kwargs):
            request_ids = active_request_ids()
            witness["model_request_ids"].append(request_ids)
            result = original_forward(*args, **kwargs)
            ctx = engine.context
            shape = result.shape[1], ctx.padded_active_token_count, ctx.num_last_token_logits
            witness.setdefault("logit_rows", []).append((request_ids, *shape))
            return result
        model.forward = traced_forward
        sampler = engine.controller._sampling
        original_sample_kernel = sampler.sample_kernel
        def traced_sample_kernel(*args, **kwargs):
            witness["sampler_request_ids"].append(active_request_ids())
            return original_sample_kernel(*args, **kwargs)
        sampler.sample_kernel = traced_sample_kernel
        if feature_config.get("use_flashinfer_fused_rope"):
            original_fused_rope = engine.context.apply_fused_qk_rotary_emb
            def traced_fused_rope(*args, **kwargs):
                witness["fused_rope_request_ids"].append(active_request_ids())
                return original_fused_rope(*args, **kwargs)
            engine.context.apply_fused_qk_rotary_emb = traced_fused_rope
        if uses_mla:
            original_flash_mla = attention_module.flash_mla_with_kvcache
            def traced_flash_mla(*args, **kwargs):
                witness["flash_mla_request_ids"].append(active_request_ids())
                return original_flash_mla(*args, **kwargs)
            monkeypatch.setattr(attention_module, "flash_mla_with_kvcache", traced_flash_mla)
        if feature_config.get("num_speculative_tokens", 0):
            original_mtp_step = engine.controller._run_async_sched_sample_mtp
            def traced_mtp_step(*args, **kwargs):
                witness["mtp_request_ids"].append(active_request_ids())
                return original_mtp_step(*args, **kwargs)
            engine.controller._run_async_sched_sample_mtp = traced_mtp_step
        direct_params = [
            SamplingParams(
                temperature=(adapter_sampling or {}).get("temperature", 0),
                top_k=(adapter_sampling or {}).get("top_k", 1),
                top_p=(adapter_sampling or {}).get("top_p", 0.0),
                skip_prompt_log_probs=True,
                return_log_probs=True,
                num_tokens_to_generate=generation_length,
                termination_id=-1,
                detokenize_stop_sequence=True,
            )
            for _ in prompts
        ]
        if stresses_chunking:
            warm_params = copy.deepcopy(direct_params[0])
            warm_params.num_tokens_to_generate = 1
            await _drive_engine(engine, [engine.add_request(90, shared_prefix, warm_params)])
        direct_futures = [
            engine.add_request(request_id, prompt, params)
            for request_id, prompt, params in zip(
                range(100, 100 + len(prompts)), prompts, direct_params, strict=True
            )
        ]
        direct = await _drive_engine(engine, direct_futures)
        if stresses_chunking:
            assert torch.is_inference(engine.context.request_ids)
            engine.reset()
            await _drive_engine(
                engine, [engine.add_request(190, shared_prefix, copy.deepcopy(warm_params))]
            )
        engine._prefill_tokens_skipped = 0
        witness["mtp_request_ids"].clear()
        witness["flash_mla_request_ids"].clear()
        witness["fused_rope_request_ids"].clear()
        original_schedule = engine.schedule_waiting_requests
        def traced_schedule():
            original_schedule()
            if (request_id := engine.context.chunked_prefill_request_id) >= 0:
                witness["chunked_prefill_request_ids"].append(int(request_id))
        engine.schedule_waiting_requests = traced_schedule
        if uses_mla:
            for layer in engine.controller.inference_wrapped_model.model.decoder.layers:
                assert layer.self_attention.kv_layernorm.weight.is_cuda
                assert layer.self_attention.linear_kv_up_proj_linear.weight.is_cuda
        adapter = object.__new__(mixin_cls)
        adapter.cfg = {
            "generation": {
                "temperature": (adapter_sampling or {}).get("temperature", 0.73),
                "top_k": (adapter_sampling or {}).get("top_k", 17),
                "top_p": (adapter_sampling or {}).get("top_p", 0.91),
                "max_new_tokens": generation_length,
                "ignore_eos": True,
                "stop_token_ids": None,
                "stop_strings": None,
            }
        }
        adapter.tokenizer = SimpleNamespace(pad_token_id=0)
        adapter.megatron_tokenizer = SimpleNamespace(eod=99)
        adapter.inference_client = _InProcessInferenceClient(
            engine, expected_requests=len(prompts), first_request_id=200, witness=witness
        )
        max_prompt_length = max(map(len, prompts))
        padded = torch.tensor(
            [prompt + [0] * (max_prompt_length - len(prompt)) for prompt in prompts],
            dtype=torch.long,
            device="cuda",
        )
        data = BatchedDataDict(
            {
                "input_ids": padded,
                "input_lengths": torch.tensor(
                    list(map(len, prompts)), dtype=torch.long, device="cuda"
                ),
            }
        )
        lowered_prompts, media, lowered_params = adapter._prepare_data_for_generation(
            data, greedy=adapter_sampling is None
        )
        assert lowered_prompts == prompts and media == [None, None]
        expected_sampling = (
            (0, 1, 0.0, -1)
            if adapter_sampling is None
            else (
                adapter_sampling["temperature"],
                adapter_sampling["top_k"],
                adapter_sampling["top_p"],
                -1,
            )
        )
        assert all(
            (params.temperature, params.top_k, params.top_p, params.termination_id)
            == expected_sampling
            for params in lowered_params
        )
        adapter_requests = await adapter._generate_with_persistent_engine(
            lowered_prompts, media, lowered_params
        )
        await adapter.inference_client.driver
        output = adapter._parse_result_to_batched_data_dict(data, adapter_requests)
        assert [entry["prompt"] for entry in witness["adapter_submissions"]] == [
            tuple(prompt) for prompt in prompts
        ]
        assert all(
            (entry["temperature"], entry["top_k"], entry["top_p"]) == expected_sampling[:3]
            for entry in witness["adapter_submissions"]
        )
        for index, expected in enumerate(direct):
            prompt_length = len(prompts[index])
            actual_length = int(output["generation_lengths"][index])
            assert actual_length == generation_length
            assert output["output_ids"][index, :prompt_length].tolist() == prompts[index]
            end = prompt_length + actual_length
            actual_tokens = output["output_ids"][index, prompt_length:end]
            assert actual_tokens.tolist() == expected.generated_tokens
            assert output["unpadded_sequence_lengths"][index].item() == end
            assert output["output_ids"][index, end:].eq(0).all()
            assert output["logprobs"][index, end:].eq(0).all()
            actual_scores = output["logprobs"][index, prompt_length:end]
            _assert_exact_scores(actual_scores, expected.generated_log_probs)
            if feature_config.get("logprobs_mode") == "processed_logprobs":
                assert actual_scores.eq(0).all()
        adapter_ids = {200, 201}
        assert adapter_ids.issubset(
            {request_id for step in witness["model_request_ids"] for request_id in step}
        )
        assert adapter_ids.issubset(
            {request_id for step in witness["sampler_request_ids"] for request_id in step}
        )
        if not feature_config or not engine.context.config.materialize_only_last_token_logits:
            rows = [row[1:] for row in witness["logit_rows"] if adapter_ids & set(row[0])]
            expected_index = 2 if engine.context.config.materialize_only_last_token_logits else 1
            assert rows and all(row[0] == row[expected_index] for row in rows)
            assert any(padded > selected for _, padded, selected in rows)
        assert engine.context.async_sched_step_count > 0
        if stresses_chunking:
            assert witness["chunked_prefill_request_ids"]
            assert set(witness["chunked_prefill_request_ids"]) <= adapter_ids
            assert engine._prefill_tokens_skipped > 0
        if stresses_prefix:
            allocator = engine.context.kv_block_allocator
            assert engine._prefill_tokens_skipped > 0
            assert not torch.count_nonzero(allocator.block_ref_counts)
            if (
                feature_config["prefix_caching_eviction_policy"]
                == PrefixCachingEvictionPolicy.REF_ZERO
            ):
                assert not allocator.kv_hash_to_block_id
            else:
                assert allocator.kv_hash_to_block_id
        if feature_config.get("num_cuda_graphs") is not None:
            graph_ids = {
                request_id
                for step_request_ids in witness["cuda_graph_request_ids"]
                for request_id in step_request_ids
            }
            assert graph_ids and graph_ids <= adapter_ids
        if feature_config.get("num_speculative_tokens", 0):
            mtp_ids = {
                request_id
                for step_request_ids in witness["mtp_request_ids"]
                for request_id in step_request_ids
            }
            assert mtp_ids and mtp_ids <= adapter_ids
        if uses_mla:
            mla_ids = {
                request_id for step in witness["flash_mla_request_ids"] for request_id in step
            }
            assert mla_ids and mla_ids <= adapter_ids
        if feature_config.get("use_flashinfer_fused_rope"):
            rope_ids = {
                request_id for step in witness["fused_rope_request_ids"] for request_id in step
            }
            assert rope_ids and rope_ids <= adapter_ids
        if mixer_name is not None:
            assert mixer_hooks and mixer_calls
        for hook in mixer_hooks:
            hook.remove()
        assert not engine.requests and not engine.has_unfinished_requests()
    @pytest.mark.internal
    @pytest.mark.asyncio
    async def test_real_adapter_joint_sampling_matches_analytic_distribution(self):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        sample_count = 128
        prompt = [3, 5, 7, 9]
        engine = self._build_test_env(
            DynamicEngineTestConfig(
                num_requests=0,
                max_prompt_length=len(prompt),
                max_sequence_length=len(prompt) + 1,
                context_max_requests=sample_count,
                context_max_tokens=sample_count * len(prompt),
                sampling_backend="flashinfer",
                async_sched_mode=AsyncScheduleMode.ASYNC,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                logprobs_mode="raw_logprobs",
            )
        ).engine
        observed = []
        sampler = engine.controller._sampling
        original_sample_kernel = sampler.sample_kernel
        def traced_sample_kernel(logits, n, context, **kwargs):
            result = original_sample_kernel(logits, n, context, **kwargs)
            observed.append(
                {
                    "logits": logits[:n].detach().float().cpu(),
                    "temperature": context.gpu_view.temperature[:n].cpu().clone(),
                    "top_k": context.gpu_view.top_k[:n].cpu().clone(),
                    "top_p": context.gpu_view.top_p[:n].cpu().clone(),
                    "tokens": result[:n].cpu().clone(),
                    "no_top_k": kwargs["no_top_k"],
                    "no_top_p": kwargs["no_top_p"],
                }
            )
            return result
        sampler.sample_kernel = traced_sample_kernel
        adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
        adapter.cfg["generation"].update(temperature=0.73, top_k=4, top_p=0.9, max_new_tokens=1)
        adapter.inference_client = _InProcessInferenceClient(
            engine,
            expected_requests=sample_count,
            first_request_id=900,
            witness={"adapter_submissions": []},
        )
        prompts = [prompt] * sample_count
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor(prompts, device="cuda"),
                "input_lengths": torch.full((sample_count,), len(prompt), device="cuda"),
            }
        )
        lowered_prompts, media, params = adapter._prepare_data_for_generation(data, greedy=False)
        replies = await adapter._generate_with_persistent_engine(lowered_prompts, media, params)
        await adapter.inference_client.driver
        output = adapter._parse_result_to_batched_data_dict(data, replies)
        assert len(observed) == 1
        step = observed[0]
        assert not step["no_top_k"] and not step["no_top_p"]
        assert step["temperature"].tolist() == pytest.approx([0.73] * sample_count)
        assert step["top_k"].tolist() == [4] * sample_count
        assert step["top_p"].tolist() == pytest.approx([0.9] * sample_count)
        torch.testing.assert_close(
            step["logits"], step["logits"][0].expand_as(step["logits"]), rtol=0, atol=0
        )
        raw_log_probs = torch.log_softmax(step["logits"][0], dim=-1)
        scaled_probs = torch.softmax(step["logits"][0] / 0.73, dim=-1)
        top_probs, top_tokens = scaled_probs.topk(4)
        sorted_probs, order = top_probs.sort(descending=True)
        sorted_tokens = top_tokens[order]
        keep = sorted_probs.cumsum(0) - sorted_probs < 0.9
        expected_probs = sorted_probs[keep]
        expected_probs /= expected_probs.sum()
        allowed_tokens = sorted_tokens[keep]
        assert len(allowed_tokens) > 1
        sampled = step["tokens"]
        assert torch.isin(sampled, allowed_tokens).all()
        assert output["output_ids"][:, len(prompt)].cpu().equal(sampled)
        torch.testing.assert_close(
            output["logprobs"][:, len(prompt)].cpu(), raw_log_probs[sampled], rtol=2e-4, atol=2e-4
        )
        for token, probability in zip(allowed_tokens, expected_probs, strict=True):
            observed_probability = (sampled == token).float().mean()
            standard_error = torch.sqrt(probability * (1 - probability) / sample_count)
            assert abs(observed_probability - probability) <= 5 * standard_error + (
                1 / sample_count
            )
        assert not engine.requests and not engine.has_unfinished_requests()
    @pytest.mark.internal
    @pytest.mark.asyncio
    async def test_public_async_adapter_preserves_row_attribution(self):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        engine = self._build_test_env(
            DynamicEngineTestConfig(
                num_requests=0,
                max_prompt_length=6,
                max_sequence_length=10,
                context_max_requests=4,
                context_max_tokens=32,
                sampling_backend="flashinfer",
                async_sched_mode=AsyncScheduleMode.ASYNC,
                return_log_probs=True,
                skip_prompt_log_probs=True,
                logprobs_mode="raw_logprobs",
            )
        ).engine
        prompts = [[3, 5, 7, 9], [2, 4, 6, 8, 10, 12]]
        direct = await _drive_engine(
            engine,
            [
                engine.add_request(
                    request_id,
                    prompt,
                    SamplingParams(
                        temperature=0,
                        top_k=1,
                        top_p=0,
                        return_log_probs=True,
                        skip_prompt_log_probs=True,
                        num_tokens_to_generate=3,
                        termination_id=-1,
                    ),
                )
                for request_id, prompt in enumerate(prompts, 1200)
            ],
        )
        adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
        adapter.cfg["generation"]["max_new_tokens"] = 3
        adapter.inference_client = _InProcessInferenceClient(
            engine, expected_requests=2, first_request_id=1210, witness={"adapter_submissions": []}
        )
        adapter._inference_loop = asyncio.get_running_loop()
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([prompts[0] + [0, 0], prompts[1]], device="cuda"),
                "input_lengths": torch.tensor([4, 6], device="cuda"),
            }
        )
        completed = {}
        async for index, row in adapter.generate_async(data, greedy=True):
            completed[index] = row
        assert set(completed) == {0, 1}
        for index, expected in enumerate(direct):
            row = completed[index]
            start = len(prompts[index])
            assert row["output_ids"][0, start : start + 3].tolist() == (expected.generated_tokens)
            _assert_exact_scores(
                row["logprobs"][0, start : start + 3], expected.generated_log_probs
            )
        assert engine.context.async_sched_step_count > 0
        assert not engine.requests and not engine.has_unfinished_requests()
@pytest.fixture
def nemo_parallel(request, monkeypatch):
    topology = request.param
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=topology["tp"],
        pipeline_model_parallel_size=topology["pp"],
        expert_model_parallel_size=topology["ep"],
        expert_tensor_parallel_size=topology.get("etp", 1),
    )
    synchronization_group = torch.distributed.new_group(backend="gloo")
    try:
        yield topology, synchronization_group
    finally:
        torch.distributed.destroy_process_group(synchronization_group)
        delete_cuda_graphs()
        reset_rounder()
        Utils.destroy_model_parallel()
class TestNeMoRLAdapterParallelContract(DynamicInferenceEngineTestBase):
    @pytest.mark.internal
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "nemo_parallel",
        [
            pytest.param({"tp": 2, "pp": 1, "ep": 1}, id="tp2"),
            pytest.param(
                {"tp": 2, "pp": 1, "ep": 1, "sp": True, "transformer_impl": "inference_optimized"},
                id="tp2-sp-inference-optimized",
            ),
            pytest.param({"tp": 1, "pp": 2, "ep": 1}, id="pp2"),
            pytest.param({"tp": 2, "pp": 1, "ep": 1, "etp": 2, "moe": True}, id="etp2-moe-local"),
            pytest.param(
                {
                    "tp": 2,
                    "pp": 1,
                    "ep": 1,
                    "etp": 2,
                    "moe": True,
                    "transformer_impl": "transformer_engine",
                },
                id="etp2-moe-te",
            ),
            pytest.param({"tp": 1, "pp": 1, "ep": 1}, id="dp2"),
        ],
        indirect=True,
    )
    async def test_real_adapter_parallel_topology(self, nemo_parallel):
        BatchedDataDict, mixin_cls = _load_nemo_rl_adapter()
        topology, synchronization_group = nemo_parallel
        rank = torch.distributed.get_rank()
        assert torch.distributed.get_world_size() == 2
        uses_moe = topology["ep"] > 1 or topology.get("moe", False)
        config = DynamicEngineTestConfig(
            num_requests=0,
            max_prompt_length=8,
            max_sequence_length=16,
            context_max_requests=8,
            context_max_tokens=64,
            tensor_model_parallel_size=topology["tp"],
            pipeline_model_parallel_size=topology["pp"],
            expert_model_parallel_size=topology["ep"],
            use_moe_layer_spec=uses_moe,
            sequence_parallel=topology.get("sp", False),
            transformer_impl=topology.get("transformer_impl", "local"),
            sampling_backend="flashinfer",
            async_sched_mode=AsyncScheduleMode.ASYNC,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            logprobs_mode="raw_logprobs",
        )
        transformer_config = dynamic_engine_tests.TransformerConfig
        def build_transformer_config(**kwargs):
            kwargs.update(
                expert_tensor_parallel_size=topology.get("etp", 1),
                num_moe_experts=2 if uses_moe else kwargs["num_moe_experts"],
                add_bias_linear=False if uses_moe else kwargs["add_bias_linear"],
            )
            return transformer_config(**kwargs)
        with mock.patch.object(dynamic_engine_tests, "TransformerConfig", build_transformer_config):
            engine = self._build_test_env(config).engine
        engine.controller.tokenizer = _CoordinatorTokenizer()
        is_dp = topology["tp"] == topology["pp"] == topology["ep"] == 1
        prompts = [[3, 5, 7, 9], [3, 5, 7, 9]] if is_dp else [[3, 5, 7, 9], [2, 4, 6, 8, 10]]
        params = [
            SamplingParams(
                temperature=0.7 if is_dp else 0,
                top_k=0 if is_dp else 1,
                top_p=0,
                skip_prompt_log_probs=True,
                return_log_probs=True,
                num_tokens_to_generate=3,
                termination_id=-1,
            )
            for _ in (prompts[:1] if is_dp else prompts)
        ]
        initial_seed = engine.controller.sampling_rng.initial_seed()
        direct = await _drive_engine(
            engine,
            [
                engine.add_request(request_id, prompt, request_params)
                for request_id, prompt, request_params in zip(
                    range(600, 600 + len(params)), prompts[: len(params)], params, strict=True
                )
            ],
        )
        direct_replies = [(reply.generated_tokens, reply.generated_log_probs) for reply in direct]
        expected_by_dp, seeds_by_dp = [None, None], [None, None]
        if is_dp:
            await asyncio.to_thread(
                torch.distributed.all_gather_object,
                expected_by_dp,
                (direct[0].generated_tokens, direct[0].generated_log_probs),
                group=synchronization_group,
            )
            await asyncio.to_thread(
                torch.distributed.all_gather_object,
                seeds_by_dp,
                initial_seed,
                group=synchronization_group,
            )
            assert seeds_by_dp[1] == seeds_by_dp[0] + 1
        engine.controller.sampling_rng.manual_seed(initial_seed)
        adapter_forward_batches, model = [], engine.controller.inference_wrapped_model.model
        original_forward = model.forward
        def traced_forward(*args, **kwargs):
            adapter_forward_batches.append(
                tuple(engine.context.request_ids[: engine.context.total_request_count].tolist())
            )
            return original_forward(*args, **kwargs)
        model.forward = traced_forward
        moe_forward_batches = []
        moe_hooks = [
            module.register_forward_hook(
                lambda _module, _inputs, _output: moe_forward_batches.append(
                    adapter_forward_batches[-1]
                )
            )
            for module in model.modules()
            if type(module).__name__ == "MoELayer"
        ]
        client, failure = None, None
        try:
            coordinator_address = await engine.start_listening_to_data_parallel_coordinator(
                hostname="127.0.0.1"
            )
            await asyncio.to_thread(torch.distributed.barrier, group=synchronization_group)
            if rank == 0:
                try:
                    client = InferenceClient(coordinator_address, deserialize=True)
                    client.start(loop=asyncio.get_running_loop(), connect_timeout_seconds=10)
                    adapter = _protocol_adapter(mixin_cls, ignore_eos=True)
                    adapter.inference_client = client
                    max_prompt_length = max(map(len, prompts))
                    data = BatchedDataDict(
                        {
                            "input_ids": torch.tensor(
                                [
                                    prompt + [0] * (max_prompt_length - len(prompt))
                                    for prompt in prompts
                                ],
                                device="cuda",
                            ),
                            "input_lengths": torch.tensor(list(map(len, prompts)), device="cuda"),
                        }
                    )
                    lowered_prompts, media, lowered_params = adapter._prepare_data_for_generation(
                        data, greedy=not is_dp
                    )
                    replies = await asyncio.wait_for(
                        adapter._generate_with_persistent_engine(
                            lowered_prompts, media, lowered_params
                        ),
                        timeout=30,
                    )
                    output = adapter._parse_result_to_batched_data_dict(data, replies)
                    expected_replies = expected_by_dp if is_dp else direct_replies
                    actual_replies = [
                        (reply.generated_tokens, reply.generated_log_probs) for reply in replies
                    ]
                    if is_dp:
                        assert sorted(actual_replies) == sorted(expected_replies)
                    else:
                        assert actual_replies == expected_replies
                    assert [reply.request_id for reply in replies] == list(range(len(prompts)))
                    for index, reply in enumerate(replies):
                        prompt_length = len(prompts[index])
                        assert (
                            output["output_ids"][index, prompt_length : prompt_length + 3].tolist()
                            == reply.generated_tokens
                        )
                        _assert_exact_scores(
                            output["logprobs"][index, prompt_length : prompt_length + 3],
                            reply.generated_log_probs,
                        )
                except BaseException as error:
                    failure = error
            await asyncio.to_thread(torch.distributed.barrier, group=synchronization_group)
        finally:
            if rank == 0 and client is not None:
                client.pause_engines()
            await asyncio.wait_for(engine.wait_until(EngineState.PAUSED), timeout=15)
            if rank == 0 and client is not None:
                client.stop_engines()
            await asyncio.wait_for(engine.engine_loop_task, timeout=15)
            if rank == 0 and client is not None:
                client.shutdown_coordinator()
                coordinator_process = engine.inference_coordinator_process
                await asyncio.to_thread(coordinator_process.join, 10)
                assert not coordinator_process.is_alive()
                client.stop()
        if failure is not None:
            raise failure
        assert adapter_forward_batches
        if topology["ep"] > 1:
            adapter_ids = set(range(len(prompts)))
            assert any(adapter_ids.issubset(batch) for batch in adapter_forward_batches)
        if uses_moe:
            assert moe_hooks and moe_forward_batches
            if topology["ep"] > 1:
                assert any(adapter_ids.issubset(batch) for batch in moe_forward_batches)
        for hook in moe_hooks:
            hook.remove()
