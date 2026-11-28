# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shortcut-MoE inference overlap at tensor-parallel size 2.

With overlap, routing, NVLS dispatch, experts and combine run on a side stream while the
paired GDP or attention layer and the shared expert run on the main stream, together with
every TP sequence-parallel collective. The kernels and their inputs are the same as in the
serial schedule, so the generated tokens and their log-probs must match it exactly, in
eager mode and with CUDA graphs, for expert-tensor-parallel and expert-parallel layouts.
"""

import gc
import random
import types

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_specs import (
    wide_residual_gated_delta_product_inference_stack_spec,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.packed_seq_helpers import check_fla_sequence_packing_support
from megatron.core.tensor_parallel import inference_layers
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import HAVE_FA3
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.ssm_test_helpers import HAVE_GDP_DEPS
from tests.unit_tests.inference.engines.test_gdp_cuda_graph_e2e import (
    FLASH_ATTENTION_VERSION,
    MAX_SEQ_LEN,
    NUM_CUDA_GRAPHS,
    VOCAB_SIZE,
)
from tests.unit_tests.inference.engines.test_gdp_cuda_graph_e2e import (
    TestGDPCudaGraphE2E as _GDPGraphTest,
)
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

NUM_TOKENS_TO_GENERATE = 8

# FA4 cannot split KV on SM90, and it picks a split for these batches unless batch-invariant mode
# pins num_splits=1, which overlap rejects. Pin FA3, Hopper's native kernel, there instead.
_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
_FLASH_VERSION = (3 if HAVE_FA3 else None) if _SM90 else FLASH_ATTENTION_VERSION

# (expert TP, EP, TP collectives, layer pattern) at TP 2. "nccl" forces the TP layers onto
# their NCCL fallback, so NCCL TP collectives on the main stream overlap the NVLS dispatch on
# the side stream. "*E*E" pairs each MoE with attention; "ME*E" also pairs one with GDP.
LAYOUTS = [
    pytest.param(2, 1, "nvls", "*E*E", id="tp2_etp2"),
    pytest.param(1, 2, "nvls", "*E*E", id="tp2_ep2"),
    pytest.param(2, 1, "nccl", "*E*E", id="tp2_etp2_nccl_tp"),
    pytest.param(2, 1, "nvls", "ME*E", id="tp2_etp2_gdp"),
]


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
@pytest.mark.skipif(_SM90 and not HAVE_FA3, reason="FA4 cannot split KV on SM90; needs FA3")
class TestShortcutMoEOverlapTP:
    """Overlap must reproduce the serial schedule bit for bit at TP 2."""

    def setup_method(self, method):
        self._model = None
        clear_nvte_env_vars()
        Utils.initialize_distributed()
        if torch.distributed.get_world_size() % 2:
            pytest.skip("needs an even number of ranks")

    def teardown_method(self, method):
        # Free the captured graphs before their communicators are destroyed: they hold NCCL
        # work (e.g. the TP logits gather), and some torch builds then hang in
        # destroy_process_group.
        if self._model is not None:
            _GDPGraphTest._reset_cuda_graph_state(self._model)
            self._model = None
        gc.collect()
        torch.cuda.synchronize()
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def _create_model(self, expert_tp, ep, pattern):
        torch.manual_seed(1234)
        config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=4,
            hidden_size=256,
            num_attention_heads=16,
            ffn_hidden_size=512,
            activation_func=squared_relu,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_num_groups=8,
            mamba_state_dim=64,
            gdp_num_householder=2,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_pre_softmax=True,
            moe_router_dtype="fp32",
            moe_token_dispatcher_type="alltoall",
            moe_shortcut_connection=True,
            moe_shared_expert_intermediate_size=512,
            moe_shortcut_post_norm=True,
            wide_residual=WideResidualConfig(num_streams=3),
            use_cpu_initialization=True,
            transformer_impl="inference_optimized",
            inference_moe_token_dispatcher_type="nvls",
            normalization="RMSNorm",
            cuda_graph_impl="local",
            inference_cuda_graph_scope="block",
            inference_rng_tracker=True,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            expert_tensor_parallel_size=expert_tp,
            expert_model_parallel_size=ep,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=False,
            is_hybrid_model=True,
            attention_dropout=0.0,
            attention_backend=AttnBackend.flash,
            flash_attention_version=_FLASH_VERSION,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=wide_residual_gated_delta_product_inference_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            parallel_output=True,
            hybrid_layer_pattern=pattern,
            pre_process=True,
            post_process=True,
        ).cuda()
        blocks = [layer for layer in model.decoder.layers if isinstance(layer, ShortcutMoEBlock)]
        assert len(blocks) == 2
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        return model

    def _build_engine(self, model, num_cuda_graphs, overlap):
        _GDPGraphTest._reset_cuda_graph_state(model)
        context = DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=MAX_SEQ_LEN,
                buffer_size_gb=0.5,
                materialize_only_last_token_logits=False,
                mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
                unified_memory_level=0,
                num_cuda_graphs=num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=True,
                sampling_backend="torch",
                inference_shortcut_moe_overlap=overlap,
            ),
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = False
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
                num_tokens_to_generate=NUM_TOKENS_TO_GENERATE,
                termination_id=-1,
                top_k=1,
                return_log_probs=True,
            ),
        )

    def _run(self, engine, prompts):
        """Run to completion, adding half of the requests once the first half decodes.

        The late arrivals force mixed prefill+decode steps.
        """
        for i, prompt in enumerate(prompts[:2]):
            engine._add_request(self._make_request(i, prompt))
        pending = list(enumerate(prompts))[2:]
        finished, graphed = {}, []
        while engine.has_unfinished_requests() or pending:
            dims = engine.context.batch_dimensions
            if pending and graphed and dims.decode_req_count > 0:
                for i, prompt in pending:
                    engine._add_request(self._make_request(i, prompt))
                pending = []
            result = engine.step_modern()
            graphed.append(engine.context.using_cuda_graph_this_step())
            for request in result["finished_requests"]:
                finished[request.request_id] = (
                    list(request.generated_tokens),
                    [float(x) for x in request.generated_log_probs],
                )
        return finished, graphed

    @pytest.mark.parametrize("expert_tp,ep,tp_comm,pattern", LAYOUTS)
    @torch.inference_mode()
    def test_overlap_matches_serial(self, monkeypatch, expert_tp, ep, tp_comm, pattern):
        if "M" in pattern:
            if not HAVE_GDP_DEPS:
                pytest.skip("GDP requires fla + mamba_ssm + einops")
            ok, reason = check_fla_sequence_packing_support()
            if not ok:
                pytest.skip(reason)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=expert_tp,
        )
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )
        if tp_comm == "nccl":
            monkeypatch.setattr(inference_layers, "are_tensors_nvls_eligible", lambda *t: False)
        overlap_calls = []
        overlap_forward = ShortcutMoEBlock._forward_inference_overlap

        def counting_overlap_forward(block, *args, **kwargs):
            overlap_calls.append(block.layer_number)
            return overlap_forward(block, *args, **kwargs)

        monkeypatch.setattr(
            ShortcutMoEBlock, "_forward_inference_overlap", counting_overlap_forward
        )

        model = self._model = self._create_model(expert_tp, ep, pattern)
        device = torch.cuda.current_device()
        # Same prompts on every rank (seeded above); none a multiple of the GDP chunk size.
        prompts = [
            torch.randint(0, VOCAB_SIZE, (n,), dtype=torch.int64, device=device)
            for n in (70, 129, 33, 200)
        ]
        for num_cuda_graphs in (None, NUM_CUDA_GRAPHS):
            serial, serial_graphed = self._run(
                self._build_engine(model, num_cuda_graphs, overlap=False), prompts
            )
            assert not overlap_calls
            overlapped, overlap_graphed = self._run(
                self._build_engine(model, num_cuda_graphs, overlap=True), prompts
            )
            assert overlap_calls, "the overlap schedule never ran"
            overlap_calls.clear()

            assert sorted(serial) == list(range(len(prompts)))
            for req_id, (tokens, log_probs) in serial.items():
                assert overlapped[req_id][0] == tokens, f"request {req_id}: tokens differ"
                # Bitwise: both schedules run the same kernels on the same inputs.
                assert overlapped[req_id][1] == log_probs, f"request {req_id}: log-probs differ"
            assert any(overlap_graphed) == (num_cuda_graphs is not None), overlap_graphed
            assert serial_graphed == overlap_graphed
