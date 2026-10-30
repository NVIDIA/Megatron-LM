# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native DDP/distributed Adam normalization against an independent sample objective."""

import os

import pytest
import torch
import torch.distributed as dist

from examples.mimo.data.source_batch import SourceBatchIterator
from examples.mimo.training.packed_step import PackedMimoStep
from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import wrap_data_iterator
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.mimo.partition.utils import PartitionAdapter, PartitionConfig
from megatron.core.models.mimo.planning import MediaTask, assign_media_to_producers
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _source_batch(pure_text=False):
    samples, media = {}, []
    for sid, physical in enumerate((96, 64, 48, 32, 32, 16, 16, 16, 16)):
        logical = physical - 3
        tokens = torch.full((physical,), sid + 1, dtype=torch.long)
        tokens[logical:] = 0
        image_ids = []
        if not pure_text and sid != 8:
            tokens[1:5] = 511
            image_ids = [sid]
            media.append(dict(image_id=sid, size=(16, 16), length=4))
        loss_mask = torch.zeros(physical)
        loss_mask[:logical] = 1
        loss_mask[1:4] = 0
        positions = torch.zeros(3, physical, dtype=torch.long)
        positions[:, :logical] = (
            torch.arange(logical)[None, :] + 100 * sid + 1000 * torch.arange(3)[:, None]
        )
        samples[sid] = dict(
            tokens=tokens,
            labels=tokens.clone(),
            loss_mask=loss_mask,
            position_ids=positions,
            original_seq_len=logical,
            padded_seq_len=physical,
            media_ids=image_ids,
        )
    return dict(step=7, samples=samples, media=media)


def test_source_resume_reconstructs_the_same_global_batch():
    class Dataset:
        def build_global_batch(self, step, size):
            return tuple(range(step * size, (step + 1) * size)), ()

    source = SourceBatchIterator(Dataset(), 64)
    for _ in range(5):
        next(source)
    resumed = SourceBatchIterator(Dataset(), 64, consumed_samples=320)
    assert next(source) == next(resumed)
    assert next(source) == next(resumed)
    with pytest.raises(ValueError, match='complete source global batch'):
        SourceBatchIterator(Dataset(), 64, consumed_samples=321)


_WIDTH = 512
_LR = 0.125


def _normalization_vision_inputs(media, device):
    if not media:
        return {}
    rows = torch.cat([torch.arange(item.length) + 10 * item.media_id + 1 for item in media])
    return {'rows': rows.to(device=device, dtype=torch.float32)[:, None] / 32}


class _NormalizationModel(torch.nn.Module):
    def __init__(self, config, pg):
        super().__init__()
        self.config = config
        self.encoder_weight = torch.nn.Parameter(torch.full((_WIDTH,), 0.5, device='cuda'))
        self.decoder_weight = torch.nn.Parameter(torch.full((_WIDTH,), 0.25, device='cuda'))
        self.special_token_ids = {'images': 511}
        self.partition = PartitionAdapter(
            PartitionConfig.from_mp_config(
                config, max_seq_len=128, kv_format='thd', cp_group=pg.cp, tp_group=pg.tp
            )
        )

    def encode_modalities(self, inputs):
        return {'images': inputs['rows'] @ self.encoder_weight[None, :]} if inputs else {}

    def forward(self, input_ids, labels, loss_mask, modality_embeddings, **kwargs):
        embeddings = (input_ids.float()[..., None] / 8).expand(-1, -1, _WIDTH).clone()
        if modality_embeddings:
            embeddings[input_ids == 511] = modality_embeddings['images']
        batch = self.partition.partition(
            embeddings=embeddings.transpose(0, 1),
            input_ids=input_ids,
            position_ids=kwargs["position_ids"],
            labels=labels,
            loss_mask=loss_mask,
            mtp_input_mask=None,
            cu_seqlens=kwargs["cu_seqlens"],
            cu_seqlens_padded=kwargs["cu_seqlens_padded"],
            max_seqlen=kwargs["max_seqlen"],
            padding_mask=kwargs["padding_mask"],
            cp_group=kwargs["cp_group"],
            tp_cp_group=kwargs["tp_cp_group"],
        ).get_batch()
        output = (
            (batch["decoder_input"].transpose(0, 1) * self.decoder_weight).square().mean(dim=-1)
        )
        return output, batch["loss_mask"]


def _independent_reference(source):
    """Original samples only; no scheduler, bridge, partition or loss helpers."""
    encoder = torch.full((_WIDTH,), 0.5, dtype=torch.float64, requires_grad=True)
    decoder = torch.full((_WIDTH,), 0.25, dtype=torch.float64, requires_grad=True)
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    for sid, sample in source['samples'].items():
        length = int(sample['original_seq_len'])
        tokens = sample['tokens'][:length]
        values = (tokens.double()[:, None] / 8).expand(-1, _WIDTH).clone()
        if sample['media_ids']:
            assert sample['media_ids'] == [sid]
            image_rows = (torch.arange(4, dtype=torch.float64) + 10 * sid + 1) / 32
            values[tokens == 511] = image_rows[:, None] * encoder
        mask = sample['loss_mask'][:length].double()
        total = total + ((values * decoder).square().mean(dim=-1) * mask).sum()
        count += int(mask.sum())
    loss = total / count
    loss.backward()
    return {
        'loss': loss.item(),
        'tokens': count,
        'initial': {'encoder_weight': encoder.detach(), 'decoder_weight': decoder.detach()},
        'grads': {'encoder_weight': encoder.grad, 'decoder_weight': decoder.grad},
    }


@pytest.mark.parametrize('layout', ('cp1', 'static_cp2', 'dynamic_cp'))
@pytest.mark.parametrize('mask_kind', ('prompt_and_padding', 'sparse'))
def test_packed_step_distributed_adam_normalization(layout, mask_kind):
    if int(os.environ.get('WORLD_SIZE', '1')) != 4:
        pytest.skip('This oracle uses exactly four CUDA ranks')
    dynamic = layout == 'dynamic_cp'
    cp_size = 1 if layout == 'cp1' else 2
    distributed = True
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=1,
        dynamic_context_parallel=dynamic,
        create_gloo_process_groups=False,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=_WIDTH,
            num_attention_heads=1,
            params_dtype=torch.float32,
            use_cpu_initialization=True,
            context_parallel_size=cp_size,
            calculate_per_token_loss=True,
            sequence_packing_scheduler='default_dynamic_cp' if dynamic else 'dp_balanced',
            dynamic_context_parallel=dynamic,
            max_seqlen_per_dp_cp_rank={'cp1': 128, 'static_cp2': 64, 'dynamic_cp': 32}[layout],
            pad_packed_seq_alignment='max',
            hidden_dropout=0,
            attention_dropout=0,
            mtp_num_layers=None,
            moe_aux_loss_coeff=0,
        )
        module = _NormalizationModel(config, pg)
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                use_distributed_optimizer=distributed,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                average_in_collective=False,
            ),
            module,
            pg_collection=pg,
        )
        opt_config = OptimizerConfig(
            optimizer='adam' if distributed else 'sgd',
            lr=_LR,
            weight_decay=0,
            sgd_momentum=0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            clip_grad=0,
            params_dtype=torch.float32,
            use_distributed_optimizer=distributed,
        )
        optimizer = get_megatron_optimizer(
            opt_config, [ddp], config_overrides={}, pg_collection=pg, use_gloo_process_groups=False
        )

        def prepare_source(source):
            tasks, slices = assign_media_to_producers(
                [MediaTask(item["image_id"], item["length"]) for item in source["media"]],
                dist.get_process_group_ranks(pg.dp_cp),
            )
            local = {task.media_id: task for task in tasks[dist.get_rank()]}

            def encoder_inputs(media_ids):
                return _normalization_vision_inputs(
                    [local[media_id] for media_id in media_ids], torch.device("cuda")
                )

            return (source["samples"], slices, encoder_inputs)

        coordinator = PackedMimoStep(module, pg, prepare_source, modality_name="images")
        coordinator.install()
        source = _source_batch()
        if mask_kind == 'sparse':
            for sample in source['samples'].values():
                sample['loss_mask'].zero_()
            source['samples'][0]['loss_mask'][4:8] = 1
        reference = _independent_reference(source)
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        iterator, rounds, _, _ = wrap_data_iterator(iter([source]), config, 1, pg)
        local_tokens = torch.zeros((), device='cuda', dtype=torch.int)
        runtime_sizes = []
        loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
        for _ in range(rounds):
            item = next(iterator)
            runtime_sizes.append(item.cp_group.size())
            # Encoder precision must not inherit decoder-only autocast when its
            # execution moves from source preparation to the forward callback.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output, loss_fn = coordinator.forward(item, ddp)
            raw_loss, tokens, _ = loss_fn(output)
            raw_loss.backward()
            local_tokens += tokens
            loss_sum += raw_loss.detach().double()
        original_local_count = int(local_tokens.item())
        coordinator.finalize([ddp], local_tokens, pg_collection=pg)
        dist.all_reduce(loss_sum, group=pg.dp_cp)
        torch.testing.assert_close(
            loss_sum.cpu() / reference['tokens'],
            torch.tensor(reference['loss'], dtype=torch.float64),
            rtol=2e-6,
            atol=2e-8,
        )
        assert int(local_tokens) == reference['tokens']
        assert [buffer.gradient_scaling_factor for buffer in ddp.buffers] == [1.0]
        assert not coordinator.runtime.active
        inner = optimizer.chained_optimizers[0]
        owned_ranges = {}
        for name, parameter in module.named_parameters():
            if distributed:
                if parameter not in inner.model_param_group_index_map:
                    continue
                shard = inner._get_model_param_range_map(parameter)['param']
                start, end = shard.start, shard.end
            else:
                start, end = 0, parameter.numel()
            owned_ranges[name] = [start, end]
            torch.testing.assert_close(
                parameter.main_grad.view(-1)[start:end].cpu().double(),
                reference['grads'][name][start:end],
                rtol=2e-6,
                atol=0,
            )
        update_successful, _, _ = optimizer.step()
        assert update_successful
        for name, parameter in module.named_parameters():
            gradient = reference['grads'][name]
            if distributed:
                expected_delta = -_LR * gradient / (gradient.abs() + opt_config.adam_eps)
                if name in owned_ranges:
                    start, end = owned_ranges[name]
                    states = inner._get_main_param_and_optimizer_states(parameter)
                    torch.testing.assert_close(
                        states['exp_avg'].cpu().double(),
                        (1 - opt_config.adam_beta1) * gradient[start:end],
                        rtol=2e-6,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        states['exp_avg_sq'].cpu().double(),
                        (1 - opt_config.adam_beta2) * gradient[start:end].square(),
                        rtol=2e-6,
                        atol=0,
                    )
            else:
                expected_delta = -_LR * gradient
            torch.testing.assert_close(
                parameter.detach().cpu().double() - reference['initial'][name],
                expected_delta,
                rtol=2e-6,
                atol=4e-8,
            )
        peers = [None] * pg.dp_cp.size()
        dist.all_gather_object(
            peers, (runtime_sizes, original_local_count, owned_ranges), group=pg.dp_cp
        )
        assert sum(count for _, count, _ in peers) == reference['tokens']
        assert {size for sizes, _, _ in peers for size in sizes} == {
            'cp1': {1},
            'static_cp2': {2},
            'dynamic_cp': {1, 2, 4},
        }[layout]
        if mask_kind == 'sparse':
            assert any(count == 0 for _, count, _ in peers)
        if distributed:
            for name in reference['grads']:
                intervals = sorted(ranges[name] for _, _, ranges in peers if name in ranges)
                assert intervals[0][0] == 0 and intervals[-1][1] == _WIDTH
                assert all(0 <= start < end <= _WIDTH for start, end in intervals)
                assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    finally:
        Utils.destroy_model_parallel()
