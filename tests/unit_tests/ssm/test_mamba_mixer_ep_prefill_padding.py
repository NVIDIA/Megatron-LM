# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test MambaMixer._dynamic_inference with expert-parallel batch dimension sync.

When expert parallelism > 1 with strict matching (hybrid models), batch
dimensions are MAX-reduced across EP ranks.  A decode-only rank may get
padded_prefill_count > 0 because another EP rank has real prefill requests.
These tests verify that the Mamba layer produces correct output shapes in
this scenario using the real EP synchronization path.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils


def _get_mamba_mixer(model: MambaModel):
    """Return the first Mamba mixer from a MambaModel."""
    decoder = model.decoder
    for layer_type, layer in zip(decoder.layer_type_list, decoder.layers):
        if layer_type == Symbols.MAMBA:
            return layer.mixer
    raise RuntimeError("No Mamba layer found in model")


@pytest.mark.internal
class TestMambaMixerEPPrefillPadding:
    """Verify _dynamic_inference output shapes when EP strict matching forces
    padded prefill slots onto a decode-only rank."""

    HIDDEN_SIZE = 256
    NUM_ATTN_HEADS = 4
    PROMPT_LEN = 16
    MAX_SEQ_LEN = 512

    @classmethod
    def setup_class(cls):
        available, reason = _check_mamba_sequence_packing_support(for_inference_not_training=True)
        if not available:
            pytest.skip(reason, allow_module_level=True)
        if not is_fa_min_version("2.7.3"):
            pytest.skip("need flash-attn >= 2.7.3 for dynamic batching", allow_module_level=True)
        if Utils.world_size < 2:
            pytest.skip("EP test requires at least 2 GPUs", allow_module_level=True)

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=Utils.world_size,
        )

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _build_model(self):
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.NUM_ATTN_HEADS,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
        )
        model = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=128,
            max_sequence_length=self.MAX_SEQ_LEN,
            hybrid_layer_pattern="M*",
        )
        model.cuda()
        model.eval()
        return model

    def _build_context(self, model, num_cuda_graphs=16):
        mamba_config = MambaInferenceStateConfig.from_model(model)
        return DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=self.MAX_SEQ_LEN,
                buffer_size_gb=1.0,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
                mamba_inference_state_config=mamba_config,
                num_cuda_graphs=num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=True,
            ),
        )

    def _prefill_and_decode_step(self, model, ctx, num_requests):
        """Add requests, run one prefill forward pass, then advance to decode."""
        for i in range(num_requests):
            req = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=torch.arange(self.PROMPT_LEN, dtype=torch.long, device="cuda"),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            ctx.add_request(req)

        # Prefill step
        ctx.initialize_attention_state()
        input_ids, position_ids = ctx.current_input_and_position_ids()
        with torch.inference_mode():
            model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                inference_context=ctx,
                runtime_gather_output=True,
            )

        # Transition all requests from prefill → decode
        active_mask = torch.ones(num_requests, dtype=torch.uint8, device="cuda")
        new_tokens = torch.zeros(num_requests, dtype=torch.long, device="cuda")
        ctx.update_requests(active_mask, new_tokens)

    @pytest.mark.internal
    @torch.inference_mode()
    def test_decode_only_rank_with_ep_prefill_padding(self):
        """Even ranks are decode-only; odd ranks add a fresh prefill request.

        After initialize_attention_state, the EP MAX-reduce (strict mode for
        hybrid models) should cause even ranks to match a mixed CUDA graph
        with padded_prefill_count > 0.  The Mamba mixer must produce output
        with the full padded token count.
        """
        rank = dist.get_rank()
        model = self._build_model()
        ctx = self._build_context(model)

        # Phase 1: All ranks prefill 2 requests and move them to decode.
        self._prefill_and_decode_step(model, ctx, num_requests=2)

        # Phase 2: Odd ranks add a new prefill request; even ranks do not.
        if rank % 2 == 1:
            new_req = DynamicInferenceRequest(
                request_id=100 + rank,
                prompt_tokens=torch.arange(
                    self.PROMPT_LEN * 2, dtype=torch.long, device="cuda"
                ),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
            ctx.add_request(new_req)

        # Phase 3: initialize_attention_state does real EP sync via
        # adjust_batch_dims_for_expert_parallelism inside match_graph_config.
        ctx.initialize_attention_state()

        padded = ctx.padded_batch_dimensions

        # Verify: on even ranks (decode-only), strict EP sync should have
        # given us padded_prefill_count > 0 because odd ranks have prefill.
        if rank % 2 == 0:
            assert ctx.batch_dimensions.prefill_req_count == 0, (
                "Even ranks should have 0 real prefill requests"
            )
            # The padded dims should have prefill slots due to EP sync
            assert padded.prefill_req_count > 0, (
                f"Rank {rank}: expected padded_prefill_count > 0 after EP sync, "
                f"got {padded.prefill_req_count}"
            )
        else:
            assert ctx.batch_dimensions.prefill_req_count > 0

        # Verify consistency: all EP ranks must agree on padded token count
        tc = torch.tensor([padded.token_count], dtype=torch.int32, device="cuda")
        tc_max = tc.clone()
        tc_min = tc.clone()
        ep_group = parallel_state.get_expert_model_parallel_group()
        dist.all_reduce(tc_max, op=dist.ReduceOp.MAX, group=ep_group)
        dist.all_reduce(tc_min, op=dist.ReduceOp.MIN, group=ep_group)
        assert tc_max.item() == tc_min.item(), (
            f"Padded token count mismatch: min={tc_min.item()}, max={tc_max.item()}"
        )

        # Phase 4: Run the Mamba mixer directly with padded hidden states.
        mixer = _get_mamba_mixer(model)
        hidden_states = torch.randn(
            padded.token_count,
            1,
            self.HIDDEN_SIZE,
            device="cuda",
            dtype=model.config.params_dtype,
        )

        out, out_bias = mixer._dynamic_inference(hidden_states, ctx)

        assert out.shape == (padded.token_count, 1, self.HIDDEN_SIZE), (
            f"Rank {rank}: expected output shape ({padded.token_count}, 1, {self.HIDDEN_SIZE}), "
            f"got {tuple(out.shape)}"
        )
