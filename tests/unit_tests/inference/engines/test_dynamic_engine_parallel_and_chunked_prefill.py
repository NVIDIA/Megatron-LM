# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import os
import random
import types

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.config import (
    AsyncScheduleMode,
    InferenceConfig,
    MambaInferenceStateConfig,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.gated_delta_net import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.ssm_test_helpers import (
    hybrid_mixer_kwargs,
    hybrid_stack_spec_for,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase,
    set_rounder,
    skip_if_mamba_sequence_packing_not_available,
)
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


class TestDynamicInferenceEngineParallel(DynamicInferenceEngineTestBase):
    """Tests that require non-default parallel configs (tp>1, pp>1, or ep>1).

    Each test initializes its own parallel state and tears it down afterward,
    so these are separated from TestDynamicInferenceEngine to avoid accumulating
    NCCL communicator memory from repeated init/destroy cycles.
    """

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    @classmethod
    @torch.inference_mode()
    def _build_test_env(cls, test_config):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=test_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=test_config.pipeline_model_parallel_size,
            expert_model_parallel_size=test_config.expert_model_parallel_size,
            expert_tensor_parallel_size=1,
        )
        return super()._build_test_env(test_config)

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_gdn_tensor_parallel(self):
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            pytest.skip("GDN TP=2 inference requires at least two GPUs.")
        env = self._run_test(
            model_provider="hybrid",
            ssm_mixer="gdn",
            tensor_model_parallel_size=2,
            num_requests=4,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=4,
            num_gap_steps=0,
            top_k=1,
            context_max_requests=16,
        )
        assert all(request.status == Status.COMPLETED for request in env.requests)

    @pytest.mark.internal
    @pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_gdn_pipeline_parallel(self):
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            pytest.skip("GDN PP=2 inference requires at least two GPUs.")
        env = self._run_test(
            model_provider="hybrid",
            ssm_mixer="gdn",
            pipeline_model_parallel_size=2,
            num_requests=4,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=4,
            num_gap_steps=0,
            top_k=1,
            context_max_requests=16,
        )
        assert all(request.status == Status.COMPLETED for request in env.requests)
        assert all(len(request.generated_tokens) == 4 for request in env.requests)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize(
        "async_sched_mode", [AsyncScheduleMode.LEGACY, AsyncScheduleMode.ASYNC]
    )
    @torch.inference_mode()
    def test_non_greedy_sampling_with_mamba_mtp_ep(self, async_sched_mode):
        """Run cumulative Mamba, MTP, and EP sampling support to completion.

        Args:
            async_sched_mode (AsyncScheduleMode): Scheduling mode under test.
        """
        skip_if_mamba_sequence_packing_not_available("hybrid")
        if int(os.environ.get("WORLD_SIZE", "1")) < 2:
            pytest.skip("Test requires at least 2 GPUs")

        env = self._run_test(
            num_requests=2,
            min_prompt_length=4,
            max_prompt_length=4,
            num_tokens_to_generate=4,
            num_gap_steps=0,
            use_fixed_output_lengths=True,
            model_provider="hybrid",
            expert_model_parallel_size=2,
            num_speculative_tokens=1,
            sampling_backend="torch",
            temperature=0.8,
            top_k=8,
            return_log_probs=True,
            skip_prompt_log_probs=True,
            context_max_requests=8,
            async_sched_mode=async_sched_mode,
        )

        assert all(request.status == Status.COMPLETED for request in env.requests)
        assert all(
            len(request.generated_log_probs) == len(request.generated_tokens)
            for request in env.requests
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("materialize_only_last_token_logits", [False, True])
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    @pytest.mark.parametrize("ep_size", [1, 2])
    @pytest.mark.parametrize("pp_size", [1, 2])
    @pytest.mark.parametrize("tp_size", [1, 2])
    @pytest.mark.parametrize("model_provider", ["gpt", "hybrid"])
    @pytest.mark.parametrize("transformer_impl", ["local", "inference_optimized"])
    @torch.inference_mode()
    def test_parallel_inference(
        self,
        model_provider,
        tp_size,
        pp_size,
        ep_size,
        sequence_parallel,
        materialize_only_last_token_logits,
        transformer_impl,
    ):
        skip_if_mamba_sequence_packing_not_available(model_provider)

        if tp_size == 1 and pp_size == 1 and ep_size == 1:
            pytest.skip(reason="Test requires tp_size > 1 or pp_size > 1 or ep_size > 1")
        elif not torch.distributed.is_initialized():
            pytest.skip("Distributed not initialized")
        world_size = torch.distributed.get_world_size()
        min_world_size = tp_size * pp_size * ep_size
        if world_size < min_world_size:
            pytest.skip(f"Test requires at least {min_world_size} GPUs")
        elif tp_size == 1 and sequence_parallel:
            pytest.skip(reason="Sequence parallelism requires tp_size > 1")
        elif tp_size > 1 and ep_size > 1 and not sequence_parallel:
            pytest.skip(reason="Sequence parallelism must be used with tp_size > 1 and ep_size > 1")
        elif transformer_impl == "inference_optimized":
            if ep_size > 1:
                pytest.skip(
                    reason="MoE models are not supported with the inference optimized transformer."
                )
            if tp_size > 1 and not sequence_parallel:
                pytest.skip(
                    reason=(
                        "The inference optimized transformer requires sequence parallelism "
                        "when tp_size > 1."
                    )
                )

        env = self._run_test(
            model_provider=model_provider,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=sequence_parallel,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            transformer_impl=transformer_impl,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @pytest.mark.parametrize("materialize_only_last_token_logits", [False, True])
    def test_sequence_parallel_fp8_inference(self, materialize_only_last_token_logits: bool):
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        self._run_test(
            min_prompt_length=19,
            max_prompt_length=19,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            materialize_only_last_token_logits=True,
            fp8=True,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_speculative_decoding_pipeline_parallel(self):
        """Test speculative decoding with pipeline parallelism (pp_size=2)."""
        if not torch.distributed.is_initialized():
            pytest.skip("Distributed not initialized")
        world_size = torch.distributed.get_world_size()
        pp_size = 2
        if world_size < pp_size:
            pytest.skip(f"Test requires at least {pp_size} GPUs")

        env = self._run_test(
            model_provider="gpt",
            pipeline_model_parallel_size=pp_size,
            num_speculative_tokens=2,
            num_tokens_to_generate=6,
            materialize_only_last_token_logits=False,
        )

        for request in env.requests:
            assert (
                request.status == Status.COMPLETED
            ), f"Request {request.request_id}: status={request.status}"
            num_expected = request.sampling_params.num_tokens_to_generate
            assert len(request.generated_tokens) <= num_expected


CHUNKED_CG_BLOCK_SIZE = 256
CHUNKED_CG_VOCAB_SIZE = 10000
CHUNKED_CG_MAX_SEQ_LEN = 2048


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestChunkedPrefillCudaGraphs:
    """Verify correctness across chunked prefill and CUDA graph combinations.

    For each model type, runs a baseline config (no chunked prefill, no CUDA graphs)
    and compares output tokens against every other combination.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        set_rounder(64)
        Utils.destroy_model_parallel()

    def _create_model(self, model_provider, num_cuda_graphs, ssm_mixer="mamba"):
        """Create a GPT or hybrid model with optional CUDA graph support.

        `ssm_mixer` selects the hybrid stack's linear-attention mixer ("mamba"
        or "gdp"); it is ignored for GPT.
        """
        cuda_graph_impl = "local" if num_cuda_graphs else "none"

        if model_provider == "gpt":
            config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=4,
                hidden_size=32,
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl=cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=True,
            )
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_local_spec(),
                vocab_size=CHUNKED_CG_VOCAB_SIZE,
                max_sequence_length=CHUNKED_CG_MAX_SEQ_LEN,
                parallel_output=True,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
        elif model_provider == "hybrid":
            config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=3,
                hidden_size=256,
                **hybrid_mixer_kwargs(ssm_mixer),
                num_attention_heads=16,
                use_cpu_initialization=True,
                cuda_graph_impl=cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=True,
                is_hybrid_model=True,
            )
            model = HybridModel(
                config=config,
                hybrid_stack_spec=hybrid_stack_spec_for(ssm_mixer),
                vocab_size=CHUNKED_CG_VOCAB_SIZE,
                max_sequence_length=CHUNKED_CG_MAX_SEQ_LEN,
                parallel_output=True,
                hybrid_layer_pattern="M*-",
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
        else:
            raise ValueError(f"Invalid model_provider {model_provider}")

        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        return model

    def _build_engine(self, model, enable_chunked_prefill, num_cuda_graphs, context_max_tokens):
        """Build an engine with the given chunked prefill / CUDA graph config."""
        set_rounder(4)
        # FP32 recurrent state. Chunked prefill hands a request's recurrence
        # through the state cache at every chunk boundary, while the baseline
        # keeps it in the kernel's FP32 accumulator from start to finish. With a
        # BF16 cache that round trip rounds the boundary value, and the two runs
        # genuinely diverge -- the same reason batch-invariant mode forces FP32
        # (see MambaInferenceStateConfig.from_model). Pinning FP32 here makes the
        # comparison test the state plumbing rather than cache precision.
        mamba_config = MambaInferenceStateConfig.from_model(model, ssm_states_dtype=torch.float32)

        inference_config_kwargs = dict(
            max_sequence_length=CHUNKED_CG_MAX_SEQ_LEN,
            buffer_size_gb=0.5,
            block_size_tokens=CHUNKED_CG_BLOCK_SIZE,
            materialize_only_last_token_logits=False,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
            enable_chunked_prefill=enable_chunked_prefill,
            max_tokens=context_max_tokens,
            max_requests=128,
            sampling_backend='torch',
        )
        if mamba_config is not None:
            inference_config_kwargs.update(mamba_inference_state_config=mamba_config)
        context = DynamicInferenceContext(
            model_config=model.config, inference_config=InferenceConfig(**inference_config_kwargs)
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=CHUNKED_CG_VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        delete_cuda_graphs()
        return DynamicInferenceEngine(controller, context)

    def _run_to_completion(self, engine, prompts, num_tokens_to_generate, conv_snapshots=None):
        """Add all prompts and run to completion, returning {req_id: generated_tokens}.

        `conv_snapshots`, if given, is appended one clone of request 0's conv
        state per step, so a caller can inspect the state at a chosen step
        instead of only the generated tokens.
        """
        for i, prompt in enumerate(prompts):
            request = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate, termination_id=-1, top_k=1
                ),
                block_size_tokens=CHUNKED_CG_BLOCK_SIZE,
            )
            engine._add_request(request)

        finished = {}
        step_count = 0
        # The slot index is read once and reused: by the final step the request
        # has finished and its entry may already be cleared, but the cache row
        # itself is only overwritten when a later request is allocated into it.
        mamba_idx = None
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            step_count += 1
            if conv_snapshots is not None:
                if mamba_idx is None:
                    mamba_idx = engine.context.mamba_metadata.request_to_mamba_state_idx[0].item()
                    assert mamba_idx >= 0, "request 0 has no mamba slot after its first step"
                conv_snapshots.append(engine.context.mamba_conv_states[:, mamba_idx].clone())
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        return finished, step_count

    # Paired rather than crossed: the mixer only means something for a hybrid
    # stack, so a gpt x gdp cell would just re-run the GPT case.
    @pytest.mark.parametrize(
        "model_provider,ssm_mixer",
        [("gpt", None), ("hybrid", "mamba"), ("hybrid", "gdp")],
        ids=["gpt", "mamba", "gdp"],
    )
    @pytest.mark.parametrize("chunked_prefill", [False, True])
    @pytest.mark.parametrize("num_cuda_graphs", [None, 2])
    @torch.inference_mode()
    def test_chunked_prefill_cuda_graphs(
        self, model_provider, ssm_mixer, chunked_prefill, num_cuda_graphs
    ):
        """Verify generated tokens match across chunked prefill and CUDA graph configs."""
        skip_if_mamba_sequence_packing_not_available(model_provider, ssm_mixer or "mamba")

        clear_nvte_env_vars()

        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

        # Create model with CUDA graph support so it can be used for both CG and non-CG engines.
        model = self._create_model(model_provider, num_cuda_graphs=2, ssm_mixer=ssm_mixer)

        # 3 prompts of 512 tokens each, disjoint token ranges (no prefix sharing).
        device = torch.cuda.current_device()
        prompts = [
            torch.arange(i * 600, i * 600 + 512, dtype=torch.int64, device=device) for i in range(3)
        ]
        num_tokens_to_generate = 8

        # Token budget: 768 forces chunking when chunked_prefill=True
        # (only ~1.5 of the 512-token prompts fit per step).
        context_max_tokens = 768 if chunked_prefill else None

        # Baseline: no chunked prefill, no CUDA graphs.
        baseline_engine = self._build_engine(
            model, enable_chunked_prefill=False, num_cuda_graphs=None, context_max_tokens=None
        )
        baseline_outputs, baseline_steps = self._run_to_completion(
            baseline_engine, prompts, num_tokens_to_generate
        )

        # Test config.
        test_engine = self._build_engine(
            model,
            enable_chunked_prefill=chunked_prefill,
            num_cuda_graphs=num_cuda_graphs,
            context_max_tokens=context_max_tokens,
        )
        test_outputs, test_steps = self._run_to_completion(
            test_engine, prompts, num_tokens_to_generate
        )

        # Correctness: generated tokens must match baseline.
        for req_id in range(3):
            assert baseline_outputs[req_id] == test_outputs[req_id], (
                f"req {req_id}: baseline {baseline_outputs[req_id]} != "
                f"test {test_outputs[req_id]} "
                f"(chunked_prefill={chunked_prefill}, num_cuda_graphs={num_cuda_graphs})"
            )

        # When chunked prefill is enabled with a constrained token budget, the engine
        # needs more scheduling steps than the non-chunked baseline.
        if chunked_prefill:
            assert test_steps > baseline_steps, (
                f"chunked prefill should need more steps than baseline "
                f"({test_steps} <= {baseline_steps})"
            )

    # d_conv is 4 for both mixers, so a final chunk of 2 or 3 tokens is shorter
    # than the conv window. Deriving the conv state from that slice alone
    # zero-fills the columns that predate it, and the first decode step then
    # convolves against zeros instead of the previous chunk's tail.
    @pytest.mark.internal
    @pytest.mark.parametrize("ssm_mixer", ["mamba", "gdp"])
    @pytest.mark.parametrize("final_chunk_len", [2, 3])
    @pytest.mark.parametrize("num_cuda_graphs", [None, 2])
    @torch.inference_mode()
    def test_short_final_prefill_chunk_carries_conv_state(
        self, ssm_mixer, final_chunk_len, num_cuda_graphs
    ):
        """A final prefill chunk shorter than d_conv must still match the baseline.

        Regression test for the conv-state carry: without
        `causal_conv1d_varlen_carry_states` the saved state loses its leading
        columns here and the generated tokens diverge from the unchunked run.

        Run under capture as well as eagerly. The kernel reads each slice length
        from `cu_seqlens` on the device, so a single captured graph serves both a
        full-length chunk and the short final one: the per-column choice has to
        come out right on replay, from lengths the capture never saw.
        """
        skip_if_mamba_sequence_packing_not_available("hybrid", ssm_mixer)

        clear_nvte_env_vars()

        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

        # Built with CUDA graph support so the same model serves both the eager
        # baseline and the captured test engine.
        model = self._create_model("hybrid", num_cuda_graphs=2, ssm_mixer=ssm_mixer)

        # Budget the prompt so the last chunk is exactly `final_chunk_len` tokens.
        # Prefix caching is off, so no block-alignment snapping shifts the split,
        # and the engine's flash-attn guard only special-cases a 1-token tail.
        context_max_tokens = 128
        prompt_len = 2 * context_max_tokens + final_chunk_len
        device = torch.cuda.current_device()
        prompts = [torch.arange(prompt_len, dtype=torch.int64, device=device)]
        num_tokens_to_generate = 8

        baseline_snapshots = []
        baseline_engine = self._build_engine(
            model, enable_chunked_prefill=False, num_cuda_graphs=None, context_max_tokens=None
        )
        baseline_outputs, _ = self._run_to_completion(
            baseline_engine, prompts, num_tokens_to_generate, conv_snapshots=baseline_snapshots
        )

        chunked_snapshots = []
        chunked_engine = self._build_engine(
            model,
            enable_chunked_prefill=True,
            num_cuda_graphs=num_cuda_graphs,
            context_max_tokens=context_max_tokens,
        )
        chunked_outputs, _ = self._run_to_completion(
            chunked_engine, prompts, num_tokens_to_generate, conv_snapshots=chunked_snapshots
        )

        assert baseline_outputs[0] == chunked_outputs[0], (
            f"{ssm_mixer}: a {final_chunk_len}-token final prefill chunk diverged from "
            f"the unchunked baseline (num_cuda_graphs={num_cuda_graphs}); baseline "
            f"{baseline_outputs[0]} != chunked {chunked_outputs[0]}"
        )

        # Generated tokens are too blunt on their own: a zero-filled carry
        # perturbs two of the four conv-state columns, and Mamba2 absorbs that
        # without moving a top-k=1 argmax, so assert on the state directly.
        #
        # It has to be the last prefill step. The conv state holds only the last
        # d_conv tokens, so a few decode steps later those columns have been
        # rolled out of it and only their knock-on effect remains.
        baseline_prefill_steps = 1
        chunked_prefill_steps = math.ceil(prompt_len / context_max_tokens)
        for name, snapshots, prefill_steps in (
            ("baseline", baseline_snapshots, baseline_prefill_steps),
            ("chunked", chunked_snapshots, chunked_prefill_steps),
        ):
            # The first generated token comes out of the final prefill step, so
            # the rest cost one decode step each. Checked rather than assumed: a
            # different prompt split would otherwise leave the comparison below
            # on two decode-step snapshots, which match no matter what.
            assert len(snapshots) == prefill_steps + num_tokens_to_generate - 1, (
                f"{name}: expected {prefill_steps} prefill steps and "
                f"{num_tokens_to_generate - 1} decode steps, got {len(snapshots)} steps"
            )

        baseline_conv = baseline_snapshots[baseline_prefill_steps - 1]
        chunked_conv = chunked_snapshots[chunked_prefill_steps - 1]

        # Both runs end prefill having seen the same prompt tokens, so the states
        # should agree. The only SSM layer sits first in the "M*-" pattern, so its
        # conv input is a projection of the embeddings and accumulates no
        # cross-chunk error; the tolerance covers GEMM tiling differences between
        # a long and a short prefill, while a dropped carry shifts a column by O(1).
        torch.testing.assert_close(
            baseline_conv,
            chunked_conv,
            rtol=2e-2,
            atol=2e-2,
            msg=lambda default: (
                f"{ssm_mixer}: conv state after a {final_chunk_len}-token final prefill "
                f"chunk does not match the unchunked baseline "
                f"(num_cuda_graphs={num_cuda_graphs}). The leading columns are the ones "
                f"the carry restores.\n{default}"
            ),
        )
