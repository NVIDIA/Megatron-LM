# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Combined 1F1B (EP A2A overlap) takes the loss scale and the last pipeline stage from the
schedule's process groups, not from the global parallel grid."""

import gc
from contextlib import ExitStack, contextmanager
from unittest import mock

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel import schedules
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
    set_streams,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    build_gpt_model,
    build_input_data,
    deterministic_mode,
    forward_step_func,
    get_test_config,
)
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 32
VOCAB_SIZE = 128
NUM_MICROBATCHES = 4


@contextmanager
def _record_global_stage_queries():
    """Record every query of the global grid for the CP size or the last pipeline stage.

    The queries are recorded rather than raised: a stage that raised would stop sending
    activations, and the next stage would wait for them forever.
    """
    queries = []

    def _recorded(name, query):
        def _record(*args, **kwargs):
            queries.append(name)
            return query(*args, **kwargs)

        return _record

    with ExitStack() as stack:
        for name in ('get_context_parallel_world_size', 'is_pipeline_last_stage'):
            query = getattr(schedules.parallel_state, name)
            stack.enter_context(
                mock.patch.object(schedules.parallel_state, name, _recorded(name, query))
            )
        yield queries


def _interleaved_config():
    """get_test_config's MLA + MoE model on PP=2 x VPP=2 with EP=2, one layer per stage."""
    return MLATransformerConfig(
        attention_backend="unfused",
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
        # With two pipeline stages the interleaved schedule needs overlapped P2P communication.
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        deterministic_mode=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_layers=4,
        hidden_size=512,
        add_bias_linear=False,
        num_attention_heads=128,
        ffn_hidden_size=512,
        kv_channels=128,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        multi_latent_attention=True,
        num_moe_experts=8,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
    )


def _train_step(schedule, model_chunks, data, pg_collection, overlap, **schedule_kwargs):
    """Run one forward-backward step; return the reduced losses and the gradients."""
    config = model_chunks[0].config
    config.overlap_moe_expert_parallel_comm = overlap
    for chunk in model_chunks:
        chunk.zero_grad()
    with _record_global_stage_queries() as global_queries:
        losses_reduced = schedule(
            forward_step_func=forward_step_func,
            data_iterator=[iter([data] * NUM_MICROBATCHES) for _ in model_chunks],
            model=model_chunks,
            num_microbatches=NUM_MICROBATCHES,
            seq_length=SEQ_LEN,
            micro_batch_size=1,
            forward_only=False,
            pg_collection=pg_collection,
            **schedule_kwargs,
        )
    torch.cuda.synchronize()
    assert not global_queries, f"read parallel_state.{sorted(set(global_queries))}"
    grads = {
        f"{chunk_id}.{name}": param.grad.clone()
        for chunk_id, chunk in enumerate(model_chunks)
        for name, param in chunk.named_parameters()
        if param.grad is not None
    }
    return [loss['lm loss'] for loss in losses_reduced], grads


def _assert_same_step(reference, overlapped):
    reference_losses, reference_grads = reference
    losses, grads = overlapped
    assert len(losses) == len(reference_losses)
    for loss, reference_loss in zip(losses, reference_losses):
        torch.testing.assert_close(loss, reference_loss)
    assert grads.keys() == reference_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, reference_grads[name], msg=name)


@pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
@pytest.mark.skipif(Utils.world_size % 4 != 0, reason="needs a multiple of 4 ranks")
class TestCombined1F1BProcessGroups:
    """The overlapped step matches the plain step and never queries the global grid."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

    def test_no_pipelining(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        with deterministic_mode():
            config = get_test_config(
                num_layers=2,
                extra_kwargs={
                    "moe_token_dispatcher_type": "alltoall",
                    "overlap_moe_expert_parallel_comm": True,
                },
            )
            model_chunks = [build_gpt_model(config, vocab_size=VOCAB_SIZE)]
            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
            schedule = schedules.forward_backward_no_pipelining
            reference = _train_step(schedule, model_chunks, data, pg_collection, overlap=False)
            overlapped = _train_step(schedule, model_chunks, data, pg_collection, overlap=True)

        _assert_same_step(reference, overlapped)
        assert len(overlapped[0]) == NUM_MICROBATCHES

    def test_interleaved_pipelining(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            expert_model_parallel_size=2,
        )
        set_streams()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pp_group = pg_collection.pp
        with deterministic_mode():
            config = _interleaved_config()
            vp_size = config.virtual_pipeline_model_parallel_size
            # The pipeline sends activations in config.pipeline_dtype, so cast the chunks to
            # bf16: the learned position embedding is created in fp32.
            model_chunks = [
                GPTModel(
                    config=config,
                    transformer_layer_spec=get_gpt_decoder_block_spec(
                        config=config, use_transformer_engine=True, vp_stage=vp_stage
                    ),
                    vocab_size=VOCAB_SIZE,
                    max_sequence_length=300,
                    pre_process=is_vp_first_stage(vp_stage, vp_size)
                    and is_pp_first_stage(pp_group),
                    post_process=is_vp_last_stage(vp_stage, vp_size) and is_pp_last_stage(pp_group),
                    share_embeddings_and_output_weights=False,
                    vp_stage=vp_stage,
                )
                .bfloat16()
                .cuda()
                for vp_stage in range(vp_size)
            ]
            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
            schedule = schedules.forward_backward_pipelining_with_interleaving
            p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
            reference = _train_step(
                schedule,
                model_chunks,
                data,
                pg_collection,
                overlap=False,
                p2p_communicator=p2p_communicator,
            )
            overlapped = _train_step(
                schedule,
                model_chunks,
                data,
                pg_collection,
                overlap=True,
                p2p_communicator=p2p_communicator,
            )

        _assert_same_step(reference, overlapped)
        # Only the last virtual stage of the last pipeline stage computes the loss.
        assert len(overlapped[0]) == (NUM_MICROBATCHES if is_pp_last_stage(pp_group) else 0)
