# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import (
    get_batch_on_this_rank_for_sequence_packing,
    wrap_data_iterator,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RouterDecoder:
    """Exercise HybridModel's layout masks with real GPU/SP groups and MoE routing."""

    def __init__(self, cp_batch, pg_collection, config):
        self.cp_batch = cp_batch
        self.pg_collection = pg_collection
        self.router = TopKRouter(config=config, pg_collection=pg_collection).cuda()
        self.router.set_layer_number(1)
        self.input_tensor = None

    def __call__(self, hidden_states, padding_mask_by_layout, **kwargs):
        for layout, batch in self.cp_batch.batches_by_layout.items():
            cp_mask = batch['padding_mask']
            expected = cp_mask.chunk(self.pg_collection.tp.size(), dim=1)[
                self.pg_collection.tp.rank()
            ]
            mask = padding_mask_by_layout[layout]
            torch.testing.assert_close(mask, expected)
            # Derive the activation shape from the CP batch, not the mask under test.
            tokens = batch['tokens'] if batch['tokens'] is not None else batch['labels']
            local_tokens = tokens.shape[1] // self.pg_collection.tp.size()
            activations = torch.randn(local_tokens, 1, 8, device='cuda', requires_grad=True)
            probs, _ = self.router(
                activations,
                padding_mask=mask.transpose(0, 1).contiguous(),
                packed_seq_params=self.cp_batch.get_packed_seq_params(layout),
            )
            probs.square().sum().backward()
            assert torch.isfinite(activations.grad).all()
            assert torch.isfinite(self.router.weight.grad).all()
            self.router.zero_grad(set_to_none=True)
        return hidden_states if hidden_states is not None else self.input_tensor


@pytest.mark.parametrize('dynamic_cp', [False, True])
@pytest.mark.parametrize('cp_size', [1, 2])
@pytest.mark.parametrize('hybrid', [False, True])
def test_packed_schedule_pipeline_cp_and_hybrid_sp(dynamic_cp, cp_size, hybrid):
    """Run scheduling -> PP field pruning -> TP broadcast -> CP slice -> SP routing."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=cp_size,
        dynamic_context_parallel=dynamic_cp,
        min_dynamic_context_parallel_size=cp_size,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = SimpleNamespace(
            sequence_packing_scheduler='default_dynamic_cp' if dynamic_cp else 'dp_balanced',
            min_dynamic_context_parallel_size=cp_size,
            max_seqlen_per_dp_cp_rank=32,
            pad_packed_seq_alignment=4,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_layout=None,
            mtp_num_layers=None,
            sequence_parallel=True,
            linear_cp_layout='contiguous',
            attention_cp_layout='zigzag',
        )
        iterator = None
        if pg.tp.rank() == 0:
            samples = []
            for _ in range(4):
                tokens = torch.arange(11, device='cuda', dtype=torch.int64)
                samples.append(
                    {
                        'tokens': tokens.clone(),
                        'labels': tokens.clone() + 1,
                        'position_ids': tokens.clone(),
                        'loss_mask': torch.ones(11, device='cuda'),
                        'cu_seqlens': torch.tensor([0, 11], dtype=torch.int32, device='cuda'),
                    }
                )
            iterator = iter(samples)
        iterator, count, _, _ = wrap_data_iterator(iterator, config, 4, pg_collection=pg)
        assert count > 0
        for _ in range(count):
            result = get_batch_on_this_rank_for_sequence_packing(
                iterator,
                dynamic_cp=dynamic_cp,
                pg_collection=pg,
                config=config,
                dynamic_cp_group_func=parallel_state.get_dynamic_data_context_parallel_groups,
                dynamic_tp_cp_group_func=parallel_state.get_dynamic_tensor_data_context_parallel_group,
                return_context_parallel_batch=hybrid,
            )
            if hybrid:
                batches = list(result.batches_by_layout.values())
            else:
                tokens, labels, loss_mask, _, positions, _, mask = result
                batches = [
                    dict(
                        tokens=tokens,
                        labels=labels,
                        loss_mask=loss_mask,
                        position_ids=positions,
                        padding_mask=mask,
                    )
                ]
            for batch in batches:
                assert (batch['tokens'] is not None) == (pg.pp.rank() == 0)
                assert (batch['position_ids'] is not None) == (pg.pp.rank() == 0)
                assert (batch['labels'] is not None) == (pg.pp.rank() == 1)
                assert (batch['loss_mask'] is not None) == (pg.pp.rank() == 1)
                assert batch['padding_mask'].shape[1] <= 32
                assert batch['padding_mask'].shape[1] % 4 == 0
            if not hybrid:
                continue

            router_config = TransformerConfig(
                num_layers=2,
                hidden_size=8,
                num_attention_heads=2,
                add_bias_linear=False,
                num_moe_experts=4,
                moe_router_topk=2,
                moe_aux_loss_coeff=0.01,
                sequence_parallel=True,
                tensor_model_parallel_size=2,
                context_parallel_size=cp_size,
                use_cpu_initialization=True,
            )
            decoder = _RouterDecoder(result, pg, router_config)
            model = SimpleNamespace(
                config=router_config,
                decoder=decoder,
                position_embedding_type='none',
                pre_process=pg.pp.rank() == 0,
                post_process=False,
                share_embeddings_and_output_weights=False,
                mtp_process=False,
                pg_collection=pg,
            )
            boundary = result.get_batch('contiguous')
            decoder.input_tensor = torch.zeros(
                boundary['padding_mask'].shape[1] // pg.tp.size(), 1, 8, device='cuda'
            )
            HybridModel.forward(
                model,
                input_ids=None,
                position_ids=None,
                attention_mask=None,
                decoder_input=decoder.input_tensor if model.pre_process else None,
                padding_mask=boundary['padding_mask'],
                packed_seq_params=result.get_packed_seq_params('contiguous'),
                cp_batch=result,
            )
    finally:
        Utils.destroy_model_parallel()
