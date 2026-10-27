# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise cache-hit parity, including chunked prefill, graph replay and refit."""

import gc
import math
from unittest import mock

import msgpack
import pytest
import torch

from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    MambaInferenceStateConfig,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import dynamic_engine
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.headers import Headers
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines.ssm_test_helpers import (
    skip_if_sequence_packing_not_available,
)
from tests.unit_tests.models.test_gpt_model_batch_invariant import (
    HAVE_FA3,
    HAVE_FA4,
    DummyTokenizer,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
@pytest.mark.skipif(
    not te_supports_batch_invariant_attention() or not (HAVE_FA3 or HAVE_FA4),
    reason="Requires batch-invariant TE and FA3/4",
)
class TestBatchInvariantMambaPrefixCaching:
    def setup_method(self):
        skip_if_sequence_packing_not_available()
        Utils.initialize_model_parallel(1, 1)
        self._previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

    def teardown_method(self):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()
        torch.use_deterministic_algorithms(self._previous_deterministic)

    @staticmethod
    def _set_env(monkeypatch):
        for name in ("MAMBA_DETERMINISTIC", "NVTE_FLASH_ATTN"):
            monkeypatch.setenv(name, "1")
        for name in ("NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
            monkeypatch.setenv(name, "0")

    @staticmethod
    def _run(
        *,
        prefix,
        graphs,
        chunked,
        kv_cache_management_mode=KVCacheManagementMode.PERSIST,
        sampling_backend="torch",
        exercise_real_refit=True,
        target_seed=321,
        source_seed=654,
    ):
        model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
        torch.manual_seed(321)
        speculative = exercise_real_refit and sampling_backend == "torch"
        cfg = TransformerConfig(
            num_layers=3,
            hidden_size=128,
            num_attention_heads=4,
            is_hybrid_model=True,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_num_groups=1,
            mamba_state_dim=16,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            batch_invariant_mode=not speculative,
            batch_invariant_backend="te_native",
            flash_attention_version=4 if HAVE_FA4 else 3,
            normalization="RMSNorm",
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_backend=AttnBackend.flash,
            cuda_graph_impl="local" if graphs else "none",
            inference_rng_tracker=True,
            mtp_num_layers=1 if speculative else None,
        )

        def build_model(seed):
            model_parallel_cuda_manual_seed(seed, inference_rng_tracker=True, force_reset_rng=True)
            torch.manual_seed(seed)
            return (
                HybridModel(
                    config=cfg,
                    hybrid_stack_spec=hybrid_stack_spec,
                    hybrid_layer_pattern="M*-/*" if speculative else "M*-",
                    vocab_size=128,
                    max_sequence_length=1024,
                )
                .cuda()
                .eval()
            )

        target_core = build_model(target_seed)
        model = Float16Module(cfg, target_core).cuda().eval()
        source_core = build_model(source_seed) if exercise_real_refit else None
        mamba_config = MambaInferenceStateConfig.from_model(target_core)
        assert mamba_config.ssm_states_dtype == (torch.bfloat16 if speculative else torch.float32)
        target_parameters = dict(model.named_parameters())
        source_model, source_parameters, target_storage = None, {}, {}
        if exercise_real_refit:
            source_model = Float16Module(cfg, source_core).cuda().eval()
            source_parameters = dict(source_model.named_parameters())
            assert source_parameters.keys() == target_parameters.keys()
            if source_seed != target_seed:
                assert any(
                    not torch.equal(source_parameters[name], target_parameters[name])
                    for name in source_parameters
                )
            target_storage = {
                name: parameter.data_ptr() for name, parameter in target_parameters.items()
            }
            prepare_swap_model_weights([source_model], [model])
        context = DynamicInferenceContext(
            cfg,
            InferenceConfig(
                mamba_inference_state_config=mamba_config,
                prefix_caching_mamba_gb=0.125,
                max_sequence_length=1024,
                buffer_size_gb=0.125,
                block_size_tokens=256,
                max_requests=4,
                max_tokens=512 if chunked else 2048,
                num_cuda_graphs=4 if graphs else None,
                cuda_graph_max_tokens=256,
                materialize_only_last_token_logits=True,
                use_cuda_graphs_for_non_decode_steps=True,
                enable_chunked_prefill=chunked,
                enable_prefix_caching=prefix,
                kv_cache_management_mode=kv_cache_management_mode,
                static_kv_memory_pointers=False,
                use_flashinfer_fused_rope=False,
                unified_memory_level=0,
                sampling_backend=sampling_backend,
                num_speculative_tokens=int(speculative),
            ),
        )
        engine = DynamicInferenceEngine(
            TextGenerationController(
                GPTInferenceWrapper(model, context), DummyTokenizer(vocab_size=128, eod=127)
            ),
            context,
        )
        graph_recaptures = 0
        graph_recapture_reset_context = []
        if graphs and kv_cache_management_mode != KVCacheManagementMode.PERSIST:
            create_cuda_graphs = engine.create_cuda_graphs

            def observe_graph_recapture(*args, **kwargs):
                nonlocal graph_recaptures
                graph_recaptures += 1
                graph_recapture_reset_context.append(kwargs.get("reset_context", True))
                return create_cuda_graphs(*args, **kwargs)

            engine.create_cuda_graphs = observe_graph_recapture
        prompts = [[i % 120 + 1 for i in range(n)] for n in (600, 650, 512, 513, 768, 769)]
        outputs = {}
        graph_steps = post_refit_graph_steps = post_refit_chunked_steps = 0
        post_refit_speculative_start = None
        torch_top_k = int(exercise_real_refit and sampling_backend == "torch")
        for wave, order in enumerate(([0, 1], [1, 0], [2], [3], [4], [5], [0])):
            if wave == 6:
                if exercise_real_refit:
                    engine.add_request(
                        5999,
                        [125] * 577,
                        SamplingParams(
                            seed=None,
                            num_tokens_to_generate=8,
                            temperature=1.0,
                            top_k=torch_top_k,
                            top_p=1.0,
                            return_log_probs=True,
                            skip_prompt_log_probs=True,
                            termination_id=-1,
                        ),
                    )
                    for _ in range(2):
                        pre_refit_result = engine.step_modern()
                        assert not pre_refit_result["finished_requests"]
                    assert engine.controller._async_sched_forward.is_valid
                    resident = engine.get_request(5999)
                    assert resident.finished_chunk_token_count > 0 or resident.generated_tokens
                    resident_salt = resident.block_hash_salt

                    engine.add_request(
                        6000,
                        [126] * 577,
                        SamplingParams(
                            seed=None,
                            num_tokens_to_generate=8,
                            temperature=1.0,
                            top_k=torch_top_k,
                            top_p=1.0,
                            return_log_probs=True,
                            skip_prompt_log_probs=True,
                            termination_id=-1,
                        ),
                    )
                    queued = engine.get_request(6000)
                    assert queued.finished_chunk_token_count == 0 and not queued.generated_tokens
                    old_salt = queued.block_hash_salt
                    post_refit_speculative_start = int(engine._spec_tokens_proposed_per_pos.sum())
                    engine.suspend()
                    pending_survived_suspend = engine.controller._async_sched_forward.is_valid
                    swap_model_weights([source_model], [model], refit_method="nccl")
                    assert all(
                        torch.equal(source_parameters[name], target_parameters[name])
                        for name in source_parameters
                    )
                    assert {
                        name: parameter.data_ptr() for name, parameter in target_parameters.items()
                    } == target_storage
                    assert pending_survived_suspend is (
                        kv_cache_management_mode != KVCacheManagementMode.RECOMPUTE
                    )

                    engine.is_mp_coordinator = True
                    engine.socket_for_receiving_requests = mock.Mock()
                    engine.model_parallel_publisher_socket = mock.Mock()
                    epoch_payload = msgpack.packb([Headers.SET_GENERATION_EPOCH.value, 1])
                    engine.socket_for_receiving_requests.recv_multipart.side_effect = [
                        [epoch_payload],
                        dynamic_engine.zmq.Again,
                    ]
                    assert engine.schedule_requests() == 1
                    engine.resume()
                    assert not engine.controller._async_sched_forward.is_valid
                    assert engine._generation_epoch == 1
                    assert all(
                        engine.get_request(request_id).policy_epoch[-1][1] == 1
                        for request_id in (5999, 6000)
                    )
                    assert queued.block_hash_salt != old_salt
                    resumed_resident_salt = engine.get_request(5999).block_hash_salt
                    if kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE:
                        assert resumed_resident_salt != resident_salt
                        assert resumed_resident_salt == queued.block_hash_salt
                else:
                    engine.suspend()
                    for parameter in model.parameters():
                        parameter.mul_(1.03)
                    engine.resume()
                skipped_before_refit_request = engine._prefill_tokens_skipped
            for i in order:
                engine.add_request(
                    wave * 10 + i + 1,
                    prompts[i],
                    SamplingParams(
                        seed=(
                            100 + i
                            if sampling_backend == "torch" and not exercise_real_refit
                            else None
                        ),
                        num_tokens_to_generate=8,
                        temperature=1.0,
                        top_k=torch_top_k,
                        top_p=1.0,
                        return_log_probs=True,
                        skip_prompt_log_probs=True,
                        termination_id=-1,
                    ),
                )
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                result = engine.step_modern()
                used_graph = int(context.using_cuda_graph_this_step())
                graph_steps += used_graph
                if wave == 6:
                    post_refit_graph_steps += used_graph
                    post_refit_chunked_steps += int(context.chunked_prefill_request_id != -1)
                for request in result["finished_requests"]:
                    outputs[request.request_id] = (
                        list(request.generated_tokens),
                        list(request.generated_log_probs),
                    )
            assert not engine.has_unfinished_requests(), "Cached suffix made no scheduling progress"
            if wave == 6:
                assert engine._prefill_tokens_skipped == skipped_before_refit_request
                if graphs:
                    assert post_refit_graph_steps > 0
                if chunked:
                    assert post_refit_chunked_steps > 0
        if sampling_backend == "torch":
            if not exercise_real_refit:
                for first_id, repeated_id in ((1, 11), (2, 12)):
                    first_tokens, first_log_probs = outputs[first_id]
                    repeated_tokens, repeated_log_probs = outputs[repeated_id]
                    assert first_tokens == repeated_tokens
                    assert len(first_log_probs) == len(repeated_log_probs)
                    for first, repeated in zip(first_log_probs, repeated_log_probs):
                        if math.isnan(first) or math.isnan(repeated):
                            assert math.isnan(first) and math.isnan(repeated)
                        else:
                            assert first == repeated

            before_tokens, before_log_probs = outputs[1]
            after_tokens, after_log_probs = outputs[61]
            assert len(before_log_probs) == len(after_log_probs)
            finite_log_probs_differ = any(
                (math.isnan(before) != math.isnan(after))
                or (not math.isnan(before) and before != after)
                for before, after in zip(before_log_probs, after_log_probs)
            )
            if source_seed != target_seed:
                assert before_tokens != after_tokens or finite_log_probs_differ
        if speculative:
            assert len(outputs[5999][0]) == 8
            assert len(outputs[6000][0]) == 8
            assert engine.num_speculative_tokens == 1
            assert post_refit_speculative_start is not None
            proposed = engine._spec_tokens_proposed_per_pos
            accepted = engine._spec_tokens_accepted_per_pos
            assert int(proposed.sum()) > post_refit_speculative_start
            assert torch.all(accepted <= proposed)
        if prefix:
            assert engine._prefill_tokens_skipped > 0
        if graphs:
            assert graph_steps > 0
            if kv_cache_management_mode != KVCacheManagementMode.PERSIST:
                assert graph_recaptures == 1
                assert graph_recapture_reset_context == [
                    kv_cache_management_mode != KVCacheManagementMode.OFFLOAD
                ]
        del engine, context, model, source_model, target_core, source_core
        clear_all_caches()
        delete_cuda_graphs()
        gc.collect()
        return outputs

    @pytest.mark.parametrize("graphs", [False, True])
    @pytest.mark.parametrize("chunked", [False, True])
    @torch.inference_mode()
    def test_cache_matches_uncached_eager(self, graphs, chunked, monkeypatch):
        self._set_env(monkeypatch)
        with set_batch_invariant_mode(True, backend="te_native"):
            baseline = self._run(
                prefix=False, graphs=False, chunked=False, exercise_real_refit=False
            )
            cached = self._run(
                prefix=True, graphs=graphs, chunked=chunked, exercise_real_refit=False
            )
        assert cached == baseline

    @pytest.mark.parametrize("kv_cache_management_mode", [KVCacheManagementMode.RECOMPUTE])
    @pytest.mark.parametrize("chunked", [False, True])
    @torch.inference_mode()
    def test_refit_across_reallocating_cache_lifecycle(
        self, kv_cache_management_mode, chunked, monkeypatch
    ):
        self._set_env(monkeypatch)
        with set_batch_invariant_mode(False):
            self._run(
                prefix=True,
                graphs=True,
                chunked=chunked,
                kv_cache_management_mode=kv_cache_management_mode,
            )
