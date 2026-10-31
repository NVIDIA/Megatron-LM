# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def loop_config(repeats=2, **kwargs):
    options = dict(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        persist_layer_norm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        decoder_num_repeats=repeats,
        decoder_repeat_start=1,
        decoder_repeat_end=3,
    )
    options.update(kwargs)
    return TransformerConfig(**options)


@pytest.mark.parametrize('repeats', [1, 2, 3])
def test_execution_order(repeats):
    config = loop_config(repeats)
    assert config.get_decoder_layer_execution_order() == (0,) + (1, 2) * repeats + (3,)


@pytest.mark.parametrize('start,end', [(0, 1), (0, 4), (3, 4), (1, 3)])
def test_execution_order_boundary_segments(start, end):
    config = loop_config(3, decoder_repeat_start=start, decoder_repeat_end=end)
    assert config.get_decoder_layer_execution_order() == (
        tuple(range(start)) + tuple(range(start, end)) * 3 + tuple(range(end, 4))
    )


def test_default_execution_order():
    config = TransformerConfig(num_layers=4, hidden_size=64, num_attention_heads=4)
    assert config.get_decoder_layer_execution_order() == (0, 1, 2, 3)


@pytest.mark.parametrize(
    'options',
    [
        dict(decoder_num_repeats=0),
        dict(decoder_num_repeats=-1),
        dict(decoder_num_repeats=True),
        dict(decoder_num_repeats=1.5),
        dict(decoder_repeat_start=-1),
        dict(decoder_repeat_start=3),
        dict(decoder_repeat_end=5),
        dict(decoder_repeat_end=None),
        dict(decoder_repeat_start=None),
        dict(decoder_repeat_start=1.5),
        dict(bf16=False),
        dict(params_dtype=torch.float32),
        dict(tensor_model_parallel_size=2),
        dict(pipeline_model_parallel_size=2),
        dict(context_parallel_size=2),
        dict(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1),
        dict(recompute_granularity='selective'),
        dict(fp8='hybrid'),
        dict(fp8=False),
        dict(fp4=False),
        dict(gtp_weight_remat_size=2),
        dict(dynamic_context_parallel=True),
        dict(fp4='nvfp4'),
        dict(num_moe_experts=2),
        dict(mtp_num_layers=1),
        dict(cpu_offloading=True),
        dict(cuda_graph_impl='local'),
        dict(enable_mhc_connections=True),
        dict(heterogeneous_block_specs=True),
        dict(fine_grained_activation_offloading=True),
        dict(quant_recipe=object()),
        dict(use_kitchen=True),
        dict(use_kitchen_attention=True),
        dict(delay_wgrad_compute=True),
    ],
)
def test_unsupported_config(options):
    with pytest.raises(ValueError, match='decoder'):
        loop_config(**options)


@pytest.mark.parametrize(
    'flag', ['overlap_grad_reduce', 'overlap_param_gather', 'delay_wgrad_compute']
)
def test_unsupported_ddp_overlap(flag):
    with pytest.raises(ValueError, match='decoder'):
        DistributedDataParallel._validate_config(
            loop_config(), DistributedDataParallelConfig(**{flag: True})
        )


@pytest.mark.skipif(Utils.world_size != 1, reason='Numerical reference requires a one-rank GPU run')
class TestLoopedDecoder:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @staticmethod
    def model(config):
        return (
            GPTModel(
                config,
                get_gpt_layer_local_spec(),
                vocab_size=128,
                max_sequence_length=16,
                position_embedding_type='learned_absolute',
            )
            .cuda()
            .bfloat16()
        )

    @pytest.mark.parametrize('repeats', [1, 2, 3])
    def test_output_gradients_update_and_identity(self, repeats):
        model = self.model(loop_config(repeats))
        # Independently construct a flat unrolled model, then tie its leaf
        # Parameters by physical layer. ProcessGroups cannot be deep-copied.
        order = [0] + [1, 2] * repeats + [3]
        reference = self.model(loop_config(1, num_layers=len(order)))
        for name, child in model.named_children():
            if name != 'decoder':
                reference.get_submodule(name).load_state_dict(child.state_dict())
        reference.decoder.final_layernorm.load_state_dict(
            model.decoder.final_layernorm.state_dict()
        )
        for logical, physical in enumerate(order):
            layer = reference.decoder.layers[logical]
            layer.load_state_dict(model.decoder.layers[physical].state_dict())
            layer.layer_number = physical + 1
            first = order.index(physical)
            for name, _ in list(layer.named_parameters()):
                parent, _, leaf = name.rpartition('.')
                param = reference.decoder.layers[first].get_parameter(name)
                layer.get_submodule(parent).register_parameter(leaf, param)
                assert param is not model.decoder.layers[physical].get_parameter(name)
        keys = tuple(model.state_dict())
        identities = tuple(id(p) for p in model.parameters())
        count = sum(p.numel() for p in model.parameters())
        baseline = self.model(loop_config(1))
        assert keys == tuple(baseline.state_dict())
        assert count == sum(p.numel() for p in baseline.parameters())
        del baseline
        actual_order = []
        handles = [
            layer.register_forward_pre_hook(lambda module, args, i=i: actual_order.append(i))
            for i, layer in enumerate(model.decoder.layers)
        ]
        ids = torch.arange(16, device='cuda').view(1, -1)
        positions = torch.arange(16, device='cuda').view(1, -1)
        mask = torch.ones(1, 1, 16, 16, device='cuda', dtype=torch.bool).triu(1)
        actual = model(ids, positions, mask)
        expected = reference(ids, positions, mask)
        assert actual_order == order
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        assert tuple(id(p) for p in model.parameters()) == identities
        # Match each physical parameter to its first reference occurrence.
        for name, param in model.named_parameters():
            ref_name = name
            if name.startswith('decoder.layers.'):
                parts = name.split('.')
                parts[2] = str(order.index(int(parts[2])))
                ref_name = '.'.join(parts)
            torch.testing.assert_close(
                param.grad, reference.get_parameter(ref_name).grad, rtol=0, atol=0
            )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        opt.step()
        ref_opt.step()
        assert len(opt.state) == len(identities)
        assert all(state['step'].item() == 1 for state in opt.state.values())
        for name, param in model.named_parameters():
            ref_name = name
            if name.startswith('decoder.layers.'):
                parts = name.split('.')
                parts[2] = str(order.index(int(parts[2])))
                ref_name = '.'.join(parts)
            torch.testing.assert_close(param, reference.get_parameter(ref_name), rtol=0, atol=0)

    def test_cached_inference_and_extraction_rejected(self):
        from megatron.core.inference.contexts import StaticInferenceContext

        model = self.model(loop_config())
        hidden = torch.randn(16, 1, 64, device='cuda', dtype=torch.bfloat16)
        with pytest.raises(ValueError, match='decoder'):
            model.decoder(hidden, None, inference_context=StaticInferenceContext(1, 16))
        with pytest.raises(ValueError, match='decoder'):
            model.decoder(hidden, None, extract_layer_indices={1})

    @pytest.mark.parametrize('repeats', [1, 2, 3])
    @pytest.mark.parametrize('distributed_optimizer', [False, True])
    def test_megatron_ddp_optimizer_and_checkpoint(
        self, repeats, distributed_optimizer, tmp_path_dist_ckpt
    ):
        from megatron.core import dist_checkpointing
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

        config = loop_config(repeats)
        model = self.model(config)
        reference = self.model(loop_config(repeats))
        reference.load_state_dict(model.state_dict())
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=distributed_optimizer,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        )
        ddp = DistributedDataParallel(config, ddp_config, model)
        opt_config = OptimizerConfig(
            lr=1e-3,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=distributed_optimizer,
            weight_decay=0.0,
            clip_grad=0.0,
        )
        optimizer = get_megatron_optimizer(opt_config, [ddp])
        masters = {
            name: torch.nn.Parameter(p.detach().float().clone())
            for name, p in reference.named_parameters()
        }
        ref_opt = torch.optim.AdamW(masters.values(), lr=1e-3, weight_decay=0.0)
        ids = torch.arange(16, device='cuda').view(1, -1)
        positions = torch.arange(16, device='cuda').view(1, -1)
        mask = torch.ones(1, 1, 16, 16, device='cuda', dtype=torch.bool).triu(1)

        def train_step(wrapped, opt):
            opt.zero_grad()
            wrapped.zero_grad_buffer()
            for offset in (0, 1):
                loss = wrapped(ids + offset, positions, mask).float().square().mean() / 2
                loss.backward()
            wrapped.finish_grad_sync()
            assert opt.step()[0]

        optimizer.zero_grad()
        ddp.zero_grad_buffer()
        expected_grads = {name: torch.zeros_like(p) for name, p in masters.items()}
        for offset in (0, 1):
            reference.zero_grad(set_to_none=True)
            actual = ddp(ids + offset, positions, mask)
            expected = reference(ids + offset, positions, mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (actual.float().square().mean() / 2).backward()
            (expected.float().square().mean() / 2).backward()
            for name, param in reference.named_parameters():
                expected_grads[name].add_(param.grad.float())
        ddp.finish_grad_sync()
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.main_grad, expected_grads[name], rtol=0, atol=0)
            masters[name].grad = expected_grads[name]
        assert optimizer.step()[0]
        ref_opt.step()
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, masters[name].bfloat16(), rtol=0, atol=0)
        # Inspect the physical optimizer tensors, including sharded master weights.
        state_numel = sum(
            value.numel()
            for state in optimizer.optimizer.state.values()
            for key, value in state.items()
            if key in ('exp_avg', 'exp_avg_sq')
        )
        assert state_numel == 2 * sum(p.numel() for p in model.parameters())
        for state in optimizer.optimizer.state.values():
            if 'step' in state:
                assert state['step'].item() == 1

        # Distributed checkpoint serialization includes both model and Adam state.
        metadata = dict(distrib_optim_sharding_type='dp_reshardable', checkpoint_version=3.1)
        model_sd = model.sharded_state_dict()
        baseline = self.model(loop_config(1))
        assert tuple(model_sd) == tuple(baseline.sharded_state_dict())
        del baseline
        checkpoint = {
            'model': model_sd,
            'optimizer': optimizer.sharded_state_dict(model_sd, metadata=metadata),
        }
        path = tmp_path_dist_ckpt / f'loop-{repeats}-{distributed_optimizer}'
        path.mkdir(parents=True, exist_ok=True)
        dist_checkpointing.save(checkpoint, path)
        bootstrap_path = path.parent / f'{path.name}-optimizer-common.pt'
        if distributed_optimizer:
            torch.save(optimizer.state_dict(), bootstrap_path)
        restored = self.model(loop_config(repeats))
        restored_ddp = DistributedDataParallel(config, ddp_config, restored)
        restored_opt = get_megatron_optimizer(opt_config, [restored_ddp])
        # Native distributed Adam needs common metadata before tensor templates.
        # Load it from disk, never from the running optimizer's in-memory tensors.
        if distributed_optimizer:
            restored_opt.load_state_dict(torch.load(bootstrap_path, weights_only=False))
        else:
            restored_ddp.zero_grad_buffer()
            assert restored_opt.step()[0]
        for state in restored_opt.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    if key != 'step':
                        value.fill_(float('nan'))
                    elif not distributed_optimizer:
                        value.zero_()
        for group in restored_opt.optimizer.param_groups:
            for param in group['params']:
                param.data.fill_(float('nan'))
            if 'step' in group and not distributed_optimizer:
                group['step'] = 0
        restore_sd = restored.sharded_state_dict()
        template = {
            'model': restore_sd,
            'optimizer': restored_opt.sharded_state_dict(
                restore_sd, is_loading=True, metadata=metadata
            ),
        }
        loaded = dist_checkpointing.load(template, path)
        restored.load_state_dict(loaded['model'])
        restored_opt.load_state_dict(loaded['optimizer'])
        for group, restored_group in zip(
            optimizer.optimizer.param_groups, restored_opt.optimizer.param_groups
        ):
            for param, restored_param in zip(group['params'], restored_group['params']):
                torch.testing.assert_close(param, restored_param, rtol=0, atol=0)
                for key, value in optimizer.optimizer.state[param].items():
                    restored_value = restored_opt.optimizer.state[restored_param][key]
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, restored_value, rtol=0, atol=0)
                    else:
                        assert value == restored_value
        if distributed_optimizer and isinstance(restored_opt.optimizer, torch.optim.AdamW):
            steps = [state['step'] for state in restored_opt.optimizer.state.values()]
            assert len({step.data_ptr() for step in steps}) == len(steps)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, restored.get_parameter(name), rtol=0, atol=0)
        train_step(ddp, optimizer)
        train_step(restored_ddp, restored_opt)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, restored.get_parameter(name), rtol=0, atol=0)


def test_fsdp_wrappers_rejected():
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
        FullyShardedDataParallelV1,
        FullyShardedDataParallelV2,
    )
    from megatron.core.distributed.torch_fully_sharded_data_parallel import (
        TorchFullyShardedDataParallel,
    )

    for wrapper in (
        FullyShardedDataParallelV1,
        FullyShardedDataParallelV2,
        TorchFullyShardedDataParallel,
    ):
        with pytest.raises(ValueError, match='decoder'):
            wrapper(loop_config(), DistributedDataParallelConfig(), torch.nn.Identity())


def test_hybrid_and_schedule_plan_rejected():
    from types import SimpleNamespace

    from megatron.core.models.common.model_chunk_schedule_plan import (
        TransformerModelChunkSchedulePlan,
    )
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    config = loop_config()
    with pytest.raises(ValueError, match='decoder'):
        HybridModel(config, None, vocab_size=128, max_sequence_length=16)
    with pytest.raises(ValueError, match='decoder'):
        TransformerModelChunkSchedulePlan(SimpleNamespace(config=config), None, None, None)
