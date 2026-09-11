# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end CUDA graph test for a Gated Delta Product hybrid model.

The kernel-level tests in `tests/unit_tests/ssm/test_gdp_dynamic_inference.py`
cover the in-tree kernels in isolation. This one drives a real GDP model through
`DynamicInferenceEngine` so the parts those tests cannot reach are exercised:
the GDP chunk descriptors built by `MambaMetadata` each step, their plumbing
through `MambaInferenceStateConfig.gdp_num_householder`, and graph capture of
whole layers rather than a hand-captured mixer call.

The assertion is the one that matters for a serving run: enabling CUDA graphs
must not change a single generated token. Greedy sampling (`top_k=1`) makes
that a well-defined comparison. Both decode-only and mixed prefill+decode steps
are covered, since `use_cuda_graphs_for_non_decode_steps` captures both and
they take different paths through the mixer.

Both engines run with batch-invariant GEMM and attention reductions, since a
graphed step pads the token count and so would otherwise pick different
reduction orders than the eager step running the same requests. That is the
kernel-level half only; GDP has no batch-invariant mixer path, so
`config.batch_invariant_mode` stays off.
"""

import random
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference import batch_dimensions_utils
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_specs import gated_delta_product_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.packed_seq_helpers import check_fla_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import HAVE_FA3, HAVE_FA4, Attention
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.ssm_test_helpers import HAVE_GDP_DEPS
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

VOCAB_SIZE = 4000
MAX_SEQ_LEN = 1024
NUM_TOKENS_TO_GENERATE = 8
NUM_CUDA_GRAPHS = 2

# Batch-invariant attention needs a pinned FlashAttention version; FA2 does not
# expose num_splits, so it cannot be pinned to a fixed split-K schedule.
FLASH_ATTENTION_VERSION = 4 if HAVE_FA4 else (3 if HAVE_FA3 else None)


# Batch rounding for this module's tests, small enough that the tiny test
# batches still hit padded shapes.
ROUNDER = 4


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="GDP requires fla + mamba_ssm + einops")
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
@pytest.mark.skipif(
    FLASH_ATTENTION_VERSION is None, reason="batch-invariant attention needs FA3 or FA4"
)
class TestGDPCudaGraphE2E:
    """A GDP hybrid model must generate identical tokens with and without graphs."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.fixture(autouse=True)
    def rounders(self, monkeypatch):
        """Shrink the batch rounding for the duration of each test.

        monkeypatch rather than plain assignment: `batch_dimensions_utils`'
        TOKEN_ROUNDER is a module-level global in production code, so restoring
        it by hand would both restate a constant its own module asks callers not
        to restate and leak into other test modules if a test raised partway.
        """
        monkeypatch.setattr(DynamicInferenceContext, "ROUNDER", ROUNDER, raising=False)
        monkeypatch.setattr(DynamicInferenceContext, "TOKEN_ROUNDER", ROUNDER)
        monkeypatch.setattr(DynamicInferenceContext, "REQUEST_ROUNDER", ROUNDER)
        # Batch-invariant CUDA-graph buckets align their token counts to the
        # module-level TOKEN_ROUNDER via _batch_invariant_token_align, which the
        # context class attributes above do not reach. Left at its default while
        # the eager path rounds to ROUNDER, a decode-only bucket's token_count
        # rounds up but its decode_req_count does not, producing an inconsistent
        # (token_count != decode_req_count) shape that breaks the decode reshape;
        # it also puts graph steps in a different norm/GEMM M-alignment class
        # than eager. Keep the two rounders in lockstep so decode buckets stay
        # square and both paths match.
        monkeypatch.setattr(batch_dimensions_utils, "TOKEN_ROUNDER", ROUNDER)

    def setup_method(self, method):
        # conftest's set_env fixture pins the NVTE backend selection; clear it so
        # TE picks the backend the attention layers expect.
        clear_nvte_env_vars()
        ok, reason = check_fla_sequence_packing_support()
        if not ok:
            pytest.skip(reason)
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def _create_model(self):
        """A 3-layer GDP / attention / MLP hybrid, built for local CUDA graphs."""
        # Seeded so every rank builds the same weights: the two engines are
        # compared against each other per rank, but identical weights keep a
        # failure reproducible across ranks.
        torch.manual_seed(1234)
        config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=3,
            hidden_size=256,
            num_attention_heads=16,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_num_groups=8,
            mamba_state_dim=64,
            gdp_num_householder=2,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
            is_hybrid_model=True,
            attention_dropout=0.0,
            attention_backend=AttnBackend.flash,
            flash_attention_version=FLASH_ATTENTION_VERSION,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=gated_delta_product_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            parallel_output=True,
            hybrid_layer_pattern="M*-",
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        # Set on the modules, not the config. `_set_attention_backend` gates
        # `batch_invariant_mode` on a TE version that only matters for TE's own
        # attention, which dynamic inference does not use, and the config flag
        # would also trip GatedDeltaProductMixer's assert that GDP has no
        # batch-invariant path. `Attention` reads the flag off the config once in
        # __init__, so flipping it on the modules is what this test needs; GEMM
        # and CUDA-graph bucket alignment come from `set_batch_invariant_mode`.
        for module in model.modules():
            if isinstance(module, Attention):
                module.batch_invariant_mode = True
        return model

    @staticmethod
    def _reset_cuda_graph_state(model):
        """Clear global and per-module CUDA graph state between engines."""
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.custom_cudagraphs_lookup_table.clear()

    def _build_engine(self, model, num_cuda_graphs):
        self._reset_cuda_graph_state(model)
        mamba_config = MambaInferenceStateConfig.from_model(model)
        assert mamba_config is not None, "GDP layers must register as Mamba layers"
        # The descriptors the forked prefill kernels read are sized from this.
        assert mamba_config.gdp_num_householder == model.config.gdp_num_householder

        context = DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=MAX_SEQ_LEN,
                buffer_size_gb=0.5,
                materialize_only_last_token_logits=False,
                mamba_inference_state_config=mamba_config,
                unified_memory_level=0,
                num_cuda_graphs=num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=True,
                sampling_backend='torch',
            ),
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        return DynamicInferenceEngine(controller, context)

    @staticmethod
    def _make_request(req_id, prompt):
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=NUM_TOKENS_TO_GENERATE, termination_id=-1, top_k=1
            ),
        )

    def _create_prompts(self):
        """Uneven lengths, none a multiple of the 64-token GDP chunk size.

        Sequence lengths that are not multiples of 64 are the interesting case:
        that is exactly when the Householder-expanded chunking stops being a
        rescaling of the unexpanded one (ceil(L*M/64) != M*ceil(L/64)), which is
        why the two descriptor sets exist.
        """
        device = torch.cuda.current_device()
        return [
            torch.randint(0, VOCAB_SIZE, (n,), dtype=torch.int64, device=device)
            for n in (70, 129, 33, 200)
        ]

    def _run(self, engine, prompts, stagger):
        """Drive the engine to completion, returning tokens and a per-step log.

        With `stagger`, two requests are held back and added once the first
        batch has reached decode, which forces mixed prefill+decode steps.
        """
        initial = prompts[:2] if stagger else prompts
        for i, prompt in enumerate(initial):
            engine._add_request(self._make_request(i, prompt))

        ctx = engine.context
        finished, step_log = {}, []
        pending = list(enumerate(prompts))[2:] if stagger else []

        while engine.has_unfinished_requests() or pending:
            if pending and step_log and step_log[-1][0] == 0 and step_log[-1][1] > 0:
                for i, prompt in pending:
                    engine._add_request(self._make_request(i, prompt))
                pending = []

            result = engine.step_modern()
            dims = ctx.batch_dimensions
            step_log.append(
                (dims.prefill_req_count, dims.decode_req_count, ctx.using_cuda_graph_this_step())
            )
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        return finished, step_log

    @pytest.mark.parametrize("stagger", [False, True], ids=["uniform", "mixed_steps"])
    @torch.inference_mode()
    def test_cuda_graphs_do_not_change_generated_tokens(self, stagger):
        """Same prompts, same model, graphs on vs off: identical output tokens."""
        # Patches aten::mm/addmm and TE's GEMM and RMSNorm entry points onto
        # fixed-reduction kernels; `batch_invariant_mode` on the config is what
        # pins attention to num_splits=1.
        with set_batch_invariant_mode(True):
            model = self._create_model()
            prompts = self._create_prompts()

            eager_engine = self._build_engine(model, num_cuda_graphs=None)
            eager_outputs, eager_log = self._run(eager_engine, prompts, stagger)

            cg_engine = self._build_engine(model, num_cuda_graphs=NUM_CUDA_GRAPHS)
            cg_outputs, cg_log = self._run(cg_engine, prompts, stagger)

        assert set(eager_outputs) == set(range(len(prompts))), "not every request finished"
        for req_id in sorted(eager_outputs):
            assert eager_outputs[req_id] == cg_outputs[req_id], (
                f"request {req_id}: CUDA graphs changed the output\n"
                f"  eager: {eager_outputs[req_id]}\n"
                f"  graph: {cg_outputs[req_id]}"
            )

        # Guard against the comparison being vacuous: graphs must actually run,
        # and never in the eager engine.
        assert any(cg for _, _, cg in cg_log), f"no CUDA graph step was taken: {cg_log}"
        assert not any(cg for _, _, cg in eager_log), f"eager engine used graphs: {eager_log}"
        assert (
            _CudagraphGlobalRecord.cudagraph_inference_record
        ), "no CUDA graph was captured for any layer"

        if stagger:
            assert any(
                p > 0 and d > 0 and cg for p, d, cg in cg_log
            ), f"no graphed mixed prefill+decode step found: {cg_log}"
        else:
            assert any(
                p == 0 and d > 0 and cg for p, d, cg in cg_log
            ), f"no graphed decode-only step found: {cg_log}"
