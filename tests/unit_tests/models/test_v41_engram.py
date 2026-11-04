# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hash, gate and real two-rank communication checks, runnable without CUDA fixtures."""

import copy
import tempfile
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core import fp8_utils
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import (
    copy_back_gathered_bf16_into_fp8_params,
    copy_tensor_to_quantized_param,
    copy_tensors_to_quantized_params,
    get_fp8_context,
    is_mxfp8tensor,
)
from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.core.models.deepseek_v41.engram import Engram, EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from megatron.core.models.engram import distributed_embedding as embedding_impl
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    get_standard_config_overrides,
)
from megatron.core.optimizer.emerging_optimizers import (
    Sinkhorn,
    _is_muon_excluded,
    _muon_param_overrides_factory,
)
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    tag_params_for_buffer_routing,
)
from megatron.core.optimizer.optimizer import (
    Float16OptimizerWithFloat16Params,
    copy_optimizer_param_metadata,
)
from megatron.core.optimizer.param_layout import BufferKey, FullParamLayout
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _hasher():
    args = SimpleNamespace(
        engram_layer_ids=[0],
        engram_num_embeddings=[0],
        engram_max_ngram_size=4,
        engram_vocab_size=11,
        engram_n_heads=2,
        engram_head_dim=3,
    )
    layout = EngramLayout.from_args(args)
    rows = sum(sum(primes) for primes in layout.primes[0])
    options = EngramConfig([0], [rows], 4, 11, 2, 3, 0, 20)
    return EngramHasher(options, torch.arange(20))


def _config():
    return SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=True,
        init_method=lambda t: torch.nn.init.normal_(t, std=0.1),
        mhc_num_residual_streams=2,
        hidden_size=4,
        layernorm_epsilon=1e-6,
    )


def _cuda_mxfp8_engram(pg_collection, *, fp8_param=True, communication_backend="nccl"):
    """Small real TE module, also used by the kernel replay test."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        mhc_num_residual_streams=2,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        fp8='e4m3',
        fp8_recipe=Fp8Recipe.mxfp8,
        fp8_param=fp8_param,
        deterministic_mode=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    layout = EngramLayout(
        max_ngram_size=2,
        layer_ids=(0,),
        num_embeddings=(198,),
        primes=(((97, 101),),),
        n_heads=2,
        head_dim=32,
    )
    config.engram_config = EngramConfig(
        [0],
        [198],
        2,
        97,
        2,
        32,
        0,
        20,
        communication_backend=communication_backend,
        hybridep_chunk_size=64,
    )
    return Engram(config, layout, 0, pg_collection).cuda()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='TE MXFP8 needs Blackwell')
@pytest.mark.parametrize('width', [32, 256])
@pytest.mark.parametrize('counts', [(0,), (37,), (5, 0, 17)])
def test_mxfp8_fused_lookup_pack(width, counts):
    """Gather bytes/scales into request order, including strided IDs and empty owners."""
    pytest.importorskip('transformer_engine.pytorch')
    pytest.importorskip('triton')
    torch.manual_seed(123)
    quantizer = embedding_impl._mxfp8_quantizer()
    weights = [
        quantizer(torch.randn(128, width, device='cuda', dtype=torch.bfloat16)) for _ in counts
    ]
    requests = torch.randint(0, 128, (sum(counts), 2), device='cuda')
    requests[:5, 1] = 0
    order = torch.randperm(sum(counts), device='cuda') if len(counts) > 1 else None
    fused = embedding_impl._mxfp8_pack_lookup(weights, requests[:, 1], order, counts, width)
    native = embedding_impl._mxfp8_pack_lookup(
        weights, requests[:, 1], order, counts, width, use_fused=False
    )
    assert torch.equal(fused, native)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='TE MXFP8 needs Blackwell')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('needs_wgrad', [False, True])
@pytest.mark.parametrize(
    'tokens,dim,heads,permuted',
    [
        (0, 32, 1, True),
        (1, 32, 1, False),
        (35, 32, 3, True),
        (65, 256, 2, True),
        (129, 96, 2, True),
        (32, 256, 1, False),
    ],
)
def test_mxfp8_fused_activation_matches_te(tokens, dim, heads, permuted, needs_wgrad, dtype):
    """Bit parity with native repack + TE dequantization/columnwise quantization."""
    pytest.importorskip('transformer_engine.pytorch')
    pytest.importorskip('triton')
    torch.manual_seed(123)
    rows = tokens * heads
    values = torch.randn(max(32, (rows + 31) // 32 * 32), dim, device='cuda')
    values *= torch.exp2(
        torch.randint(-10, 10, (values.shape[0], dim // 32), device='cuda')
    ).repeat_interleave(32, dim=1)
    values[:1] = 0
    weight = embedding_impl._mxfp8_quantizer()(values.to(dtype))
    packed = torch.cat(
        (weight._rowwise_data[:rows], weight._rowwise_scale_inv[:rows, : dim // 32]), dim=-1
    )
    if (tokens, dim, heads) == (32, 256, 1):
        # Exercise signed zero, FP8 subnormals/ties, tiny logical values and the
        # upper finite range. Each 32-channel group retains its own scale range.
        raw = torch.arange(rows * dim, device='cuda').reshape(rows, dim)
        packed[:, :dim] = ((raw % 127) | ((raw // 127 % 2) * 128)).to(torch.uint8)
        scales = (
            [0, 1, 3, 20, 80, 123, 127, 245]
            if dtype == torch.bfloat16
            else [0, 1, 64, 103, 110, 123, 127, 134]
        )
        packed[:, dim:] = torch.tensor(scales, device='cuda', dtype=torch.uint8)
    order = torch.randperm(rows, device='cuda') if permuted else None
    fused = embedding_impl._mxfp8_lookup_activation(
        packed, dim, dim * heads, dtype, needs_wgrad, order
    )
    native = embedding_impl._mxfp8_lookup_activation(
        packed, dim, dim * heads, dtype, needs_wgrad, order, use_fused=False
    )
    assert torch.equal(fused._rowwise_data, native._rowwise_data)
    assert torch.equal(fused._rowwise_scale_inv, native._rowwise_scale_inv)
    torch.testing.assert_close(fused.dequantize(), native.dequantize(), rtol=0, atol=0)
    if needs_wgrad:
        assert torch.equal(fused._columnwise_data, native._columnwise_data)
        blocks, width = fused.shape[0] // 32, dim * heads
        assert torch.equal(
            fused._columnwise_scale_inv[:blocks, :width],
            native._columnwise_scale_inv[:blocks, :width],
        )
        assert torch.all(fused._columnwise_scale_inv[blocks:] == 127)
        assert torch.all(fused._columnwise_scale_inv[:, width:] == 127)
    else:
        assert fused._columnwise_data is None and fused._columnwise_scale_inv is None


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 GEMMs need Blackwell')
@pytest.mark.parametrize('ep_size', [1, 2])
@pytest.mark.parametrize("communication_backend", ["nccl", "hybridep"])
def test_mxfp8_engram_training_and_optimizer(ep_size, communication_backend):
    """Real TE storage, EP lookup, projection backward, masters and requantization."""
    if communication_backend == "hybridep" and ep_size > 1:
        pytest.importorskip("deep_ep", exc_type=ImportError)
    pytest.importorskip('transformer_engine.pytorch')
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    from megatron.core.extensions.transformer_engine import te_checkpoint
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    Utils.initialize_distributed()
    if dist.get_world_size() % ep_size:
        pytest.skip('world size must be divisible by EP size')
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    try:
        model_parallel_cuda_manual_seed(123)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model = _cuda_mxfp8_engram(pg, communication_backend=communication_backend)
        table = model.embed.tables[0]
        weight = table.weight
        assert is_mxfp8tensor(weight) and is_mxfp8tensor(model.wkv.weight)
        assert weight._columnwise_data is None
        assert weight._columnwise_scale_inv is None
        assert not weight._with_gemm_swizzled_scales
        initial = weight.get_high_precision_init_val().cuda().float()

        # Compare the actual TE payload to complete-table high-precision embedding lookup.
        reference = torch.zeros(198, 32, device='cuda', dtype=torch.float32)
        reference[table.row_start : table.row_end] = (
            weight.detach().dequantize()[: table.local_num_embeddings].float()
        )
        dist.all_reduce(reference, group=pg.ep)
        reference.requires_grad_()
        tokens = 35 + pg.ep.rank()
        hashes = torch.randint(0, 198, (1, tokens, 1, 2), device='cuda')
        hashes[:, :3] = 0
        activation = model.embed.lookup_quantized(hashes[:, :, 0].unsqueeze(-1), 64)
        assert is_mxfp8tensor(activation) and activation._columnwise_data is not None
        decoded = activation.dequantize()
        expected = torch.nn.functional.embedding(hashes[:, :, 0], reference).reshape(tokens, 64)
        torch.testing.assert_close(decoded[:tokens].float(), expected, rtol=0, atol=0)
        assert torch.count_nonzero(decoded[tokens:]) == 0
        gradient = torch.randn(tokens, 64, device='cuda', dtype=torch.bfloat16)
        (decoded[:tokens] * gradient).sum().backward()
        (expected * gradient.float()).sum().backward()
        dist.all_reduce(reference.grad, group=pg.ep)
        torch.testing.assert_close(
            weight.grad[: table.local_num_embeddings],
            reference.grad[table.row_start : table.row_end].bfloat16(),
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(weight.grad[table.local_num_embeddings :]) == 0
        model.zero_grad(set_to_none=True)

        optimizer = Float16OptimizerWithFloat16Params(
            Sinkhorn([weight], lr=0.01),
            OptimizerConfig(bf16=True, clip_grad=0.0),
            grad_scaler=None,
            init_state_fn=None,
        )
        master = optimizer.fp32_from_float16_groups[0][0]
        assert master.dtype == torch.float32
        torch.testing.assert_close(master, initial, rtol=0, atol=0)
        assert master.row_shard_local_rows == table.local_num_embeddings
        assert weight.get_high_precision_init_val() is None
        hidden = torch.randn(
            tokens, 1, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True
        )
        with get_fp8_context(model.config):
            output = model(hidden, hashes)
        output.float().square().mean().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        for parameter in model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert model.wkv.weight.grad.abs().sum() > 0

        # The hybrid stack restores FP8 context inside the checkpointed region on replay.
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        input_gradient = hidden.grad.clone()
        model.zero_grad(set_to_none=True)
        hidden.grad = None

        def region(hidden_states, hash_ids):
            with get_fp8_context(model.config):
                return model(hidden_states, hash_ids)

        replayed = te_checkpoint(
            region, False, get_cuda_rng_tracker, pg.tp, hidden, hashes, use_reentrant=False
        )
        replayed.float().square().mean().backward()
        torch.testing.assert_close(replayed, output, rtol=0, atol=0)
        torch.testing.assert_close(hidden.grad, input_gradient, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)

        assert not optimizer.prepare_grads()
        assert optimizer.step_with_ready_grads()
        assert not torch.equal(master, initial)
        state = optimizer.optimizer.state[master]
        assert state['momentum_buffer'].dtype == torch.float32
        assert torch.count_nonzero(master[table.local_num_embeddings :]) == 0
        assert weight._columnwise_data is None

        # Reuse the actual DDP copy-back helper: only GEMM weights get both orientations.
        bf16_master = master.detach().bfloat16()
        weight._get_quantizer().set_usage(rowwise=True, columnwise=True)
        copy_back_gathered_bf16_into_fp8_params([weight], [bf16_master])
        expected_weight = embedding_impl._mxfp8_quantizer()(bf16_master)
        assert torch.equal(weight._rowwise_data, expected_weight._rowwise_data)
        torch.testing.assert_close(
            weight.dequantize(), expected_weight.dequantize(), rtol=0, atol=0
        )
        assert weight._columnwise_data is None
        assert not weight._get_quantizer().columnwise_usage
        factory = table.sharded_state_dict()['weight']
        saved_model = factory.build()[0].data.clone()
        saved_master = factory.build_fn('master', master, (0, 0, 0), None)[0].data.clone()
        saved_momentum = factory.build_fn('momentum', state['momentum_buffer'], (0, 0, 0), None)[
            0
        ].data.clone()
        table.load_state_dict({'weight': factory.merge_fn([saved_model])})
        torch.testing.assert_close(factory.merge_fn([saved_master]), master, rtol=0, atol=0)
        torch.testing.assert_close(
            factory.merge_fn([saved_momentum]), state['momentum_buffer'], rtol=0, atol=0
        )
        assert weight._columnwise_data is None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 needs Blackwell')
@pytest.mark.parametrize('cpu_initialization', [False, True])
def test_mxfp8_table_conversion_preserves_metadata(cpu_initialization):
    pytest.importorskip('transformer_engine.pytorch')
    config = _config()
    config.params_dtype = torch.bfloat16
    config.use_cpu_initialization = cpu_initialization
    config.sequence_parallel = True
    model = EPShardedMultiTableEmbedding(config, (97,), 32, config.init_method, fp8_storage=True)
    weight = model.tables[0].weight
    weight.lr_mult = 5.0
    initial = (
        weight.detach().clone() if cpu_initialization else weight.get_high_precision_init_val()
    )
    assert is_mxfp8tensor(weight) != cpu_initialization
    # get_model moves an already-CUDA model, then Float16Module converts its dtype.
    model.cuda()
    model.cuda()
    convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
    old_data = model.tables[0].weight._rowwise_data
    model._apply(lambda tensor: tensor.clone())
    weight = model.tables[0].weight
    assert is_mxfp8tensor(weight)
    assert weight._rowwise_data.data_ptr() != old_data.data_ptr()
    torch.testing.assert_close(weight._rowwise_data, old_data, rtol=0, atol=0)
    assert weight.get_high_precision_init_val.__self__ is weight
    assert weight.clear_high_precision_init_val.__self__ is weight
    torch.testing.assert_close(weight.get_high_precision_init_val(), initial, rtol=0, atol=0)
    assert not weight.allreduce and weight.is_engram_embedding
    assert weight.is_embedding_or_output_parameter and weight.sequence_parallel
    assert weight.lr_mult == 5.0 and not weight.mxfp8_requires_columnwise
    assert weight.row_shard_group is None
    assert weight.row_shard_start == 0 and weight.row_shard_global_rows == 97
    assert weight.row_shard_local_rows == 97 and weight._columnwise_data is None
    tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=True)
    assert weight.use_sinkhorn and _is_muon_excluded(weight)
    buffers = group_params_for_buffers([weight], grad_reduce_in_fp32=True)
    assert list(buffers) == [BufferKey(torch.uint8, torch.float32, True, True)]
    weight.clear_high_precision_init_val()
    assert weight.get_high_precision_init_val() is None


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 needs Blackwell')
@pytest.mark.parametrize('use_layout', [False, True])
@pytest.mark.parametrize('communication_backend', ['nccl', 'symmetric'])
def test_mxfp8_lookup_overlapped_param_gather(use_layout, communication_backend):
    """EP2 x replica-DP2 lookup must publish updated masters before reading FP8 bytes."""
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    if communication_backend == 'symmetric':
        pytest.importorskip('nccl.core.device.cute', exc_type=ImportError)
    Utils.initialize_distributed()
    if dist.get_world_size() < 4 or dist.get_world_size() % 2:
        pytest.skip('requires EP2 and at least two expert-DP replicas')
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    runtime = None
    try:
        torch.manual_seed(123)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=False,
            fp8='e4m3',
            fp8_recipe=Fp8Recipe.mxfp8,
            fp8_param=True,
        )
        # These row counts produce different physical shard sizes on the two EP ranks.
        model = EPShardedMultiTableEmbedding(
            config,
            (129, 65),
            32,
            config.init_method,
            pg.ep,
            pg.tp,
            pg.expt_dp,
            fp8_storage=True,
            communication_backend=communication_backend,
        ).cuda()
        convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
        weights = [table.weight for table in model.tables]
        for weight in weights:
            assert weight.row_shard_group is pg.ep and not weight.allreduce
            weight.lr_mult = 5.0
        if communication_backend == 'symmetric':
            # Keep real TE tensors/quantization, but force unaligned allocator
            # suballocations for both bytes and scales before optimizer/DDP setup.
            for weight in weights:
                for name in ('_rowwise_data', '_rowwise_scale_inv'):
                    original = getattr(weight, name)
                    parent = torch.empty(original.numel() + 4096, dtype=torch.uint8, device='cuda')
                    start = (512 - parent.data_ptr()) % 4096
                    replacement = torch.from_dlpack(parent[start : start + original.numel()])
                    replacement = replacement.view_as(original)
                    replacement.copy_(original)
                    setattr(weight, name, replacement)
                    assert replacement.untyped_storage().data_ptr() % 4096 == 512
            runtime = embedding_impl.EngramRuntime(8, defer_grad_transfer=True)
            runtime.attach(model)
        tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=True)
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            bf16=True,
            lr=0.02,
            clip_grad=0.0,
            muon_use_sinkhorn=True,
            use_distributed_optimizer=True,
            use_layer_wise_param_layout=use_layout,
        )
        overrides = get_standard_config_overrides(optimizer_config)
        overrides.update(_muon_param_overrides_factory())
        groups = _get_param_groups([model], optimizer_config, overrides)
        assert all(group['optimizer'] == 'sinkhorn' for group in groups if group['params'])
        assert all(group['lr_mult'] == 5.0 for group in groups if group['params'])
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                use_distributed_optimizer=True,
                use_layer_wise_param_layout=use_layout,
                fp8_param_gather=True,
                reuse_grad_buf_for_mxfp8_param_ag=True,
            ),
            model,
            pg_collection=pg,
        )
        assert not ddp.buffers and ddp.expert_parallel_buffers
        optimizer = LayerWiseDistributedOptimizer(
            [Sinkhorn([{'params': weights, 'is_expert_parallel': True}], lr=0.02)],
            optimizer_config,
            pg_collection=pg,
            model_chunks=[ddp],
        )
        child = optimizer.chained_optimizers[0]
        masters = {
            id(weight): master
            for model_group, master_group in zip(
                child.float16_groups, child.fp32_from_float16_groups
            )
            for weight, master in zip(model_group, master_group)
        }
        rows = torch.arange(129, device='cuda')
        hashes = torch.stack((rows, rows % 65), dim=-1)
        previous = None
        for _ in range(3):
            ddp.zero_grad_buffer()
            optimizer.zero_grad()
            if runtime is not None:
                runtime.reset()
            optimizer.prepare_model_params_for_param_sync()
            references = []
            for index, table in enumerate(model.tables):
                expected_shard = torch.zeros(
                    table.weight.shape, device='cuda', dtype=torch.bfloat16
                )
                if id(table.weight) in masters:
                    expected_shard.copy_(masters[id(table.weight)].detach())
                dist.broadcast(
                    expected_shard, dist.get_global_rank(pg.expt_dp, index), group=pg.expt_dp
                )
                expected_shard = embedding_impl._mxfp8_quantizer()(expected_shard).dequantize()
                reference = torch.zeros(
                    table.global_num_embeddings, 32, device='cuda', dtype=torch.bfloat16
                )
                reference[table.row_start : table.row_end] = expected_shard[
                    : table.local_num_embeddings
                ]
                dist.all_reduce(reference, group=pg.ep)
                references.append(reference[hashes[:, index]])
            # Engram's FP8 path never calls table.forward(), where DDP installs its hooks.
            activation = (
                runtime.consume(runtime.prefetch(model, hashes), 64)
                if runtime is not None
                else model.lookup_quantized(hashes, 64)
            )
            actual = activation.dequantize()[:129]
            expected = torch.cat(references, dim=-1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if previous is not None:
                assert not torch.equal(actual, previous)
            previous = actual.detach().clone()
            actual.float().square().sum().backward()
            ddp.finish_grad_sync()
            assert not optimizer.prepare_grads()
            assert optimizer.step_with_ready_grads()
            assert all(weight._columnwise_data is None for weight in weights)
    finally:
        if runtime is not None:
            runtime.close()
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 needs Blackwell')
@pytest.mark.parametrize('rows', [128, 160, 416])
@pytest.mark.parametrize('width', [32, 256])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_mxfp8_chunked_weight_copy(monkeypatch, rows, width, dtype):
    """Chunk boundaries and padded scale views preserve TE bytes without reallocating."""
    monkeypatch.setattr(fp8_utils, '_MXFP8_QUANTIZE_CHUNK_ROWS', 128)
    quantizer = embedding_impl._mxfp8_quantizer()
    # Exercise a strided master and a last chunk with only 32 logical rows.
    source = torch.randn(rows, width * 2, dtype=dtype, device='cuda')[:, ::2]
    source[::7] = 0
    expected = quantizer(source.contiguous())
    weight = torch.nn.Parameter(
        quantizer.make_empty(source.shape, dtype=torch.bfloat16, device='cuda'), requires_grad=False
    )
    data_ptr = weight._rowwise_data.data_ptr()
    scale_ptr = weight._rowwise_scale_inv.data_ptr()
    for copy_weight in (
        lambda: copy_tensor_to_quantized_param(weight, source),
        lambda: copy_tensors_to_quantized_params([weight], [source.reshape(-1)]),
    ):
        weight._rowwise_data.zero_()
        weight._rowwise_scale_inv.zero_()
        copy_weight()
        torch.testing.assert_close(weight._rowwise_data, expected._rowwise_data, rtol=0, atol=0)
        torch.testing.assert_close(
            weight._rowwise_scale_inv[:rows, : width // 32],
            expected._rowwise_scale_inv[:rows, : width // 32],
            rtol=0,
            atol=0,
        )
        assert weight._rowwise_data.data_ptr() == data_ptr
        assert weight._rowwise_scale_inv.data_ptr() == scale_ptr
        assert weight._columnwise_data is None


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 needs Blackwell')
def test_mxfp8_engram_large_table_lifecycle():
    """Regress the real EP64 shard that exceeded TE's 65535-block grid.y limit."""
    config = _config()
    config.params_dtype = torch.bfloat16
    config.use_cpu_initialization = False
    config.init_method = lambda tensor: torch.nn.init.constant_(tensor, 0.5)
    logical_rows = 6000097
    model = EPShardedMultiTableEmbedding(
        config, (logical_rows,), 256, config.init_method, fp8_storage=True
    )
    table = model.tables[0]
    weight = table.weight
    assert weight.shape == (6000128, 256)
    data_ptr = weight._rowwise_data.data_ptr()
    scale_ptr = weight._rowwise_scale_inv.data_ptr()

    def check_weight(value):
        with torch.no_grad():
            decoded = weight.dequantize()
            assert (decoded[:logical_rows] == value).all()
            assert (decoded[logical_rows:] == 0).all()
        assert weight._rowwise_data.data_ptr() == data_ptr
        assert weight._rowwise_scale_inv.data_ptr() == scale_ptr
        assert weight._columnwise_data is None

    check_weight(0.5)
    optimizer = Float16OptimizerWithFloat16Params(
        torch.optim.SGD([weight], lr=0.01),
        OptimizerConfig(bf16=True, clip_grad=0.0),
        grad_scaler=None,
        init_state_fn=None,
    )
    master = optimizer.fp32_from_float16_groups[0][0]
    assert master.dtype == torch.float32 and weight.get_high_precision_init_val() is None
    with torch.no_grad():
        master[:logical_rows].fill_(0.75)
        optimizer._copy_main_params_to_model_params()
    check_weight(0.75)

    with torch.no_grad():
        master[:logical_rows].fill_(0.875)
        copy_back_gathered_bf16_into_fp8_params([weight], [master.bfloat16()])
    check_weight(0.875)

    # Save logical rows and load a CPU checkpoint through the parent Module.
    factory = table.sharded_state_dict()['weight']
    saved = factory.build()[0].data.cpu()
    state = {'tables.0.weight': factory.merge_fn([saved])}
    source = state['tables.0.weight']
    weight._rowwise_data.zero_()
    model.load_state_dict(state)
    check_weight(0.875)
    assert model.tables[0].weight is weight and state['tables.0.weight'] is source
    assert weight.row_shard_local_rows == logical_rows


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 needs Blackwell')
def test_mxfp8_chunked_checkpoint_validation(monkeypatch):
    """The custom weight copy retains ordinary state-dict key/shape validation."""
    monkeypatch.setattr(fp8_utils, '_MXFP8_QUANTIZE_CHUNK_ROWS', 128)
    config = _config()
    config.params_dtype = torch.bfloat16
    config.use_cpu_initialization = False
    model = EPShardedMultiTableEmbedding(config, (161,), 32, config.init_method, fp8_storage=True)
    weight = model.tables[0].weight
    with pytest.raises(RuntimeError, match='size mismatch'):
        model.load_state_dict({'tables.0.weight': torch.zeros(32, 32)})
    with pytest.raises(RuntimeError, match='Missing key'):
        model.load_state_dict({})
    with pytest.raises(RuntimeError, match='Unexpected key'):
        model.load_state_dict({'tables.0.weight': weight, 'extra': torch.zeros(1)})
    incompatible = model.load_state_dict({}, strict=False)
    assert incompatible.missing_keys == ['tables.0.weight']
    source = torch.randn(weight.shape, dtype=torch.bfloat16)
    model.load_state_dict({'tables.0.weight': source}, assign=True)
    expected = embedding_impl._mxfp8_quantizer()(source.cuda())
    assert model.tables[0].weight is weight
    torch.testing.assert_close(weight._rowwise_data, expected._rowwise_data, rtol=0, atol=0)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 GEMMs need Blackwell')
def test_mxfp8_projection_with_bf16_table():
    pytest.importorskip('transformer_engine.pytorch')
    Utils.initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(123)
        model = _cuda_mxfp8_engram(ProcessGroupCollection.use_mpu_process_groups(), fp8_param=False)
        assert not is_mxfp8tensor(model.embed.tables[0].weight)
        assert not is_mxfp8tensor(model.wkv.weight)
        hidden = torch.randn(64, 1, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        hashes = torch.randint(0, 198, (1, 64, 1, 2), device='cuda')
        with get_fp8_context(model.config):
            model(hidden, hashes).float().square().sum().backward()
        assert model.wkv.fp8
        assert model.embed.tables[0].weight.grad.abs().sum() > 0
    finally:
        Utils.destroy_model_parallel()


def test_hash_resets_and_memory_gradients():
    hasher = _hasher()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    sequence = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    hashes = hasher(ids, sequence_ids=sequence)
    torch.testing.assert_close(hashes[:, 4:], hasher(ids[:, 4:]))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 3] = False
    torch.testing.assert_close(hasher(ids, mask)[:, 4:], hasher(ids[:, 4:]))
    model = Engram(_config(), hasher.layout, 0, SimpleNamespace(ep=None, tp=None, expt_dp=None))
    overrides = get_standard_config_overrides(OptimizerConfig())
    for name in ("q_weight", "k_weight"):
        parameter = getattr(model, name)
        assert _is_muon_excluded(parameter)
        assert any(
            key.matches(parameter, name) and override.get("wd_mult") == 0.0
            for key, override in overrides.items()
        )
    hidden = torch.randn(8, 1, 8, requires_grad=True)
    out = model(hidden, hashes, mask)
    torch.testing.assert_close(out[3], hidden[3])
    out.square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('lr_multiplier', [1.0, 5.0])
@pytest.mark.parametrize('norm_decay', [False, True])
def test_engram_optimizer_groups_and_schedule(monkeypatch, lr_multiplier, norm_decay):
    """Only lookup tables get the Engram LR; fused norm vectors follow the norm policy."""
    monkeypatch.setattr(dist, 'get_world_size', lambda **kwargs: 1)
    monkeypatch.setattr(
        dist, 'all_gather_object', lambda out, value, **kwargs: out.__setitem__(0, value)
    )
    config = _config()
    config.engram_config = SimpleNamespace(embedding_lr_multiplier=lr_multiplier)
    model = torch.nn.ModuleDict(
        {
            'engram': Engram(
                config, _hasher().layout, 0, SimpleNamespace(ep=None, tp=None, expt_dp=None)
            ),
            'embedding': torch.nn.Embedding(20, 4),
        }
    )
    model['embedding'].weight.is_embedding_or_output_parameter = True
    tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=True)
    optimizer_config = OptimizerConfig(
        optimizer='muon', lr=0.01, min_lr=0.001, apply_wd_to_norm=norm_decay
    )
    overrides = get_standard_config_overrides(optimizer_config)
    overrides.update(_muon_param_overrides_factory())
    groups = _get_param_groups([model], optimizer_config, overrides)
    scheduler = OptimizerParamScheduler(
        SimpleNamespace(param_groups=groups),
        init_lr=0.002,
        max_lr=0.01,
        min_lr=0.001,
        lr_warmup_steps=10,
        lr_decay_steps=100,
        lr_decay_style='cosine',
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=100,
        wd_incr_style='constant',
    )
    by_param = {id(p): group for group in groups for p in group['params']}
    table = model['engram'].embed.tables[0].weight
    for step in (0, 5, 10, 50, 100):
        scheduler.step(step - scheduler.num_steps)
        base_lr = by_param[id(model['embedding'].weight)]['lr']
        assert by_param[id(table)]['lr'] == pytest.approx(base_lr * lr_multiplier)
        assert by_param[id(table)]['optimizer'] == 'sinkhorn'
        assert by_param[id(table)]['weight_decay'] == 0
        for param in (model['engram'].q_weight, model['engram'].k_weight):
            assert by_param[id(param)]['optimizer'] == 'adam'
            assert by_param[id(param)]['lr'] == pytest.approx(base_lr)
            assert by_param[id(param)]['weight_decay'] == pytest.approx(0.1 if norm_decay else 0)
        assert by_param[id(model['engram'].wkv.weight)]['lr'] == pytest.approx(base_lr)
        assert by_param[id(model['engram'].wkv.weight)]['weight_decay'] == pytest.approx(0.1)


def _sinkhorn_distributed_worker(rank, path, use_layout, padded):
    """Compare EP2 x replica-DP2 updates/resume with complete-matrix EO updates."""
    dist.init_process_group(
        'gloo', init_method='file://' + path, rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    ep_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
    replica_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
    ep = ep_groups[rank // 2]
    replica = replica_groups[rank % 2]
    try:
        torch.manual_seed(109)
        model = EPShardedMultiTableEmbedding(
            _config(), (17, 19), 5, _config().init_method, ep, expt_dp_group=replica
        )
        if padded:
            for table in model.tables:
                table.weight.data = torch.nn.functional.pad(table.weight.data, (0, 0, 0, 7))
        # Different local sizes deliberately perturb ordinary bin packing across EP peers.
        model.extra = torch.nn.Parameter(torch.zeros(8 if rank % 2 == 0 else 11, 5))
        model.extra.allreduce = False
        tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=True)
        full_weights = [torch.nn.Parameter(torch.randn(rows, 5)) for rows in (17, 19)]
        references = [Sinkhorn([param], lr=0.02) for param in full_weights]
        for table, full in zip(model.tables, full_weights):
            table.weight.data[: table.local_num_embeddings].copy_(
                full[table.row_start : table.row_end]
            )
        optimizer = Sinkhorn(
            [{'params': [t.weight for t in model.tables], 'is_expert_parallel': True}], lr=0.02
        )
        extra_optimizer = torch.optim.SGD(
            [{'params': [model.extra], 'is_expert_parallel': True}], lr=0.02
        )
        owner = object.__new__(LayerWiseDistributedOptimizer)
        owner.dp_cp, owner.expt_dp = dist.group.WORLD, replica
        layouts = None
        if use_layout:
            layout = owner._compute_per_buffer_param_layout(
                list(model.parameters()),
                50,
                2,
                SimpleNamespace(pad_buckets_for_high_nccl_busbw=False),
            )
            layouts = [
                FullParamLayout({BufferKey(torch.float32, torch.float32, True, True): layout})
            ]
        owner.shard_params([optimizer, extra_optimizer], layouts)
        local_params = optimizer.param_groups[0]['params']
        assert len(local_params) == 1
        assert local_params[0] is model.tables[replica.rank()].weight
        restored = None
        generator = torch.Generator().manual_seed(311)
        for iteration in range(5):
            gradients = [torch.randn(p.shape, generator=generator) for p in full_weights]
            for table, grad, reference in zip(model.tables, gradients, references):
                grad[: table.global_num_embeddings // 2] *= torch.tensor([100, 1, 0.1, 0.01, 2])
                grad[0].zero_()
                grad[1].mul_(1e-7)
                if iteration == 1:
                    # A missing local gradient must still update old momentum globally.
                    grad[(table.global_num_embeddings + 1) // 2 :].zero_()
                absent = iteration == 3 or (iteration == 1 and ep.rank() == 1)
                local_grad = torch.zeros_like(table.weight)
                local_grad[: table.local_num_embeddings].copy_(
                    grad[table.row_start : table.row_end]
                )
                table.weight.grad = None if absent else local_grad
                reference.param_groups[0]['params'][0].grad = None if iteration == 3 else grad
                reference.step()
            optimizer.step()
            if restored is not None:
                for actual, resumed in zip(local_params, restored.param_groups[0]['params']):
                    resumed.grad = None if actual.grad is None else actual.grad.clone()
                restored.step()
                torch.testing.assert_close(
                    restored.param_groups[0]['params'][0], local_params[0], rtol=0, atol=0
                )
            for index, (table, full) in enumerate(zip(model.tables, full_weights)):
                # Mirror LayerWise's replica parameter synchronization after each update.
                dist.broadcast(
                    table.weight.data, dist.get_global_rank(replica, index), group=replica
                )
                torch.testing.assert_close(
                    table.weight[: table.local_num_embeddings],
                    full[table.row_start : table.row_end],
                    rtol=3e-6,
                    atol=3e-7,
                )
                assert torch.count_nonzero(table.weight[table.local_num_embeddings :]) == 0
            if iteration == 2:
                resumed_param = torch.nn.Parameter(local_params[0].detach().clone())
                copy_optimizer_param_metadata(resumed_param, local_params[0])
                assert resumed_param.row_shard_group is ep
                restored = Sinkhorn([{'params': [resumed_param], 'is_expert_parallel': True}])
                restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))

        # A valid logical matrix may include an empty row shard.
        empty = torch.nn.Parameter(torch.ones(4 if ep.rank() == 0 else 0, 4))
        empty.row_shard_group = ep
        empty.row_shard_global_rows = 4
        empty.row_shard_start = 0 if ep.rank() == 0 else 4
        empty_optimizer = Sinkhorn([empty])
        empty.grad = torch.ones_like(empty) if ep.rank() == 0 else None
        empty_optimizer.step()
        torch.testing.assert_close(empty, torch.full_like(empty, 1 - 0.02 * 0.18))
        if ep.rank() == 1:
            assert empty.grad is None
        empty.row_shard_global_rows = 4 + ep.rank()
        with pytest.raises(ValueError, match='logical matrix metadata'):
            Sinkhorn([empty])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('use_layout', [False, True])
@pytest.mark.parametrize('padded', [False, True])
def test_sinkhorn_ep_and_replica_owners_match_complete_update(use_layout, padded):
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _sinkhorn_distributed_worker,
            args=(directory + '/rendezvous', use_layout, padded),
            nprocs=4,
            join=True,
        )


def test_fp8_table_checkpoint_excludes_physical_padding():
    """Model and optimizer factory transformations preserve the unpadded global table."""
    config = _config()
    config.params_dtype = torch.bfloat16
    model = EPShardedMultiTableEmbedding(config, (97,), 32, config.init_method, fp8_storage=True)
    table = model.tables[0]
    assert table.weight.shape == (128, 32)
    assert table.local_num_embeddings == 97
    assert model.local_parameter_count == model.global_parameter_count == 97 * 32
    assert not table.weight.mxfp8_requires_columnwise
    assert torch.count_nonzero(table.weight[97:]) == 0
    factory = table.sharded_state_dict(sharded_offsets=((0, 1, 2),))['weight']
    shard = factory.build()[0]
    assert shard.local_shape == (97, 32)
    assert shard.global_shape == (2, 97, 32)
    assert shard.global_offset == (1, 0, 0)
    torch.testing.assert_close(factory.merge_fn([shard.data]), table.weight)
    # The optimizer replaces the factory data/key; no closure may read model values instead.
    momentum = torch.randn(128, 32)
    saved = factory.build_fn('optimizer.momentum', momentum, (0, 0, 0), None)[0]
    assert saved.key == 'optimizer.momentum'
    restored = factory.merge_fn([saved.data])
    torch.testing.assert_close(restored[:97], momentum[:97])
    assert torch.count_nonzero(restored[97:]) == 0
    with pytest.raises(ValueError, match='whole row-shard'):
        factory.build_fn('weight', momentum.flatten()[:32], (0, 0, 0), slice(0, 32))


def test_sinkhorn_padding_does_not_enter_statistics():
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    torch.manual_seed(71)
    logical = torch.nn.Parameter(torch.randn(97, 32))
    padded = torch.nn.Parameter(torch.nn.functional.pad(logical.detach(), (0, 0, 0, 31)))
    padded.row_shard_local_rows = 97
    reference, optimizer = Sinkhorn([logical], lr=0.01), Sinkhorn([padded], lr=0.01)
    for _ in range(3):
        logical.grad = torch.randn_like(logical)
        # Poison padding so a wrong row mean, column norm or update cannot pass unnoticed.
        padded.grad = torch.full_like(padded, 1e5)
        padded.grad[:97] = logical.grad
        reference.step()
        optimizer.step()
        torch.testing.assert_close(padded[:97], logical, rtol=3e-6, atol=3e-7)
        assert torch.count_nonzero(padded[97:]) == 0


def _fp8_wire_worker(rank, path, num_tables, ep_size):
    """Real byte AllToAll/autograd, with only TE's GEMM-input wrapper replaced on CPU."""
    dist.init_process_group(
        'gloo', init_method='file://' + path, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        config = _config()
        model = EPShardedMultiTableEmbedding(
            config,
            (97, 101)[:num_tables],
            64,
            config.init_method,
            ep_group=dist.group.WORLD if ep_size > 1 else None,
        )
        generator = torch.Generator().manual_seed(918)
        reference_weights = []
        for table in model.tables:
            data = torch.randn(table.global_num_embeddings, 64, generator=generator).to(
                torch.float8_e4m3fn
            )
            scales = torch.randint(
                123, 132, (table.global_num_embeddings, 2), generator=generator, dtype=torch.uint8
            )
            values = data.float().view(-1, 2, 32) * torch.exp2(scales.float() - 127).unsqueeze(-1)
            reference_weights.append(values.flatten(-2).requires_grad_())
            table.weight._rowwise_data = data.view(torch.uint8)[table.row_start : table.row_end]
            table.weight._rowwise_scale_inv = scales[table.row_start : table.row_end]

        def cpu_activation(packed, embedding_dim, output_width, dtype, needs_wgrad, order=None):
            assert packed.dtype == torch.uint8
            assert packed.shape[1] == embedding_dim + embedding_dim // 32
            if order is not None:
                ordered = torch.empty_like(packed)
                ordered.index_copy_(0, order, packed)
                packed = ordered
            values = packed[:, :embedding_dim].contiguous().view(torch.float8_e4m3fn).float()
            scales = torch.exp2(packed[:, embedding_dim:].float() - 127)
            values = (values.view(-1, embedding_dim // 32, 32) * scales.unsqueeze(-1)).flatten(-2)
            tokens = values.numel() // output_width
            values = values.view(tokens, output_width)
            return torch.nn.functional.pad(
                values, (0, 0, 0, max(32, (tokens + 31) // 32 * 32) - tokens)
            )

        with (
            patch.object(embedding_impl, 'is_mxfp8tensor', lambda p: hasattr(p, '_rowwise_data')),
            patch.object(embedding_impl, '_mxfp8_lookup_activation', cpu_activation),
        ):
            for empty_requester in (False, True):
                tokens = 0 if empty_requester and rank == 0 else 3 + rank
                ids = torch.randint(0, 97, (tokens, 3, num_tables), generator=generator)
                if tokens:
                    ids[0] = 0  # duplicate rows in multiple heads
                output = model.lookup_quantized(ids, 3 * num_tables * 64)
                expected = torch.stack(
                    [
                        torch.nn.functional.embedding(ids[..., index], weight)
                        for index, weight in enumerate(reference_weights)
                    ],
                    dim=-2,
                ).reshape(tokens, 3 * num_tables * 64)
                torch.testing.assert_close(output[:tokens], expected, rtol=0, atol=0)
                assert torch.count_nonzero(output[tokens:]) == 0
                gradient = torch.randn(expected.shape, generator=generator)
                (expected * gradient).sum().backward()
                (output[:tokens] * gradient).sum().backward()
                for table, reference in zip(model.tables, reference_weights):
                    if ep_size > 1:
                        dist.all_reduce(reference.grad)
                    torch.testing.assert_close(
                        table.weight.grad,
                        reference.grad[table.row_start : table.row_end],
                        rtol=1e-6,
                        atol=1e-6,
                    )
                    table.weight.grad = None
                    reference.grad = None
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('num_tables', [1, 2])
@pytest.mark.parametrize('ep_size', [1, 2])
def test_two_rank_fp8_payload_and_embedding_gradients(num_tables, ep_size):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _fp8_wire_worker,
            args=(directory + '/rendezvous', num_tables, ep_size),
            nprocs=2,
            join=True,
        )


def _distributed_worker(rank, path):
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    group = dist.group.WORLD
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    model = EPShardedMultiTableEmbedding(_config(), (11, 13), 3, _config().init_method, group)
    assert torch.equal(before, torch.random.get_rng_state())
    full = [torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) for n in (11, 13)]
    with torch.no_grad():
        for table, ref in zip(model.tables, full):
            table.weight.copy_(ref[table.row_start : table.row_end])
    ids = torch.tensor([[0, 12], [10, 0], [0, 12]])[: rank + 2]
    out = model(ids)
    expected = torch.stack([ref[ids[:, i]] for i, ref in enumerate(full)], 1)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    for i, table in enumerate(model.tables):
        counts = torch.bincount(ids[:, i], minlength=full[i].shape[0]).float()
        dist.all_reduce(counts)
        torch.testing.assert_close(
            table.weight.grad, counts[table.row_start : table.row_end, None].expand_as(table.weight)
        )
        shard = table.sharded_state_dict()['weight']
        assert shard.global_shape == tuple(full[i].shape)
    hasher = _hasher()
    # Check both normal halo exchange and tiny shards spanning multiple neighbors.
    for local_length in (4, 1):
        tokens = torch.arange(1, 2 * local_length + 1).unsqueeze(0)
        sequences = torch.arange(2 * local_length).unsqueeze(0) // 3
        local = slice(rank * local_length, (rank + 1) * local_length)
        actual = hasher(tokens[:, local], sequence_ids=sequences[:, local], cp_group=group)
        expected = hasher(tokens, sequence_ids=sequences)[:, local]
        torch.testing.assert_close(actual, expected)
    dist.destroy_process_group()


def test_two_rank_lookup_gradients_and_cp_hash_history():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_worker, args=(directory + '/rendezvous',), nprocs=2, join=True)


@pytest.mark.parametrize("size", [4, 64])
@pytest.mark.parametrize(
    "pattern", ["uniform", "uneven", "hotspot", "empty_requester", "empty", "tiny"]
)
def test_hybridep_engram_interleaved_plan(size, pattern):
    """Exact per-round capacities agree with independently enumerated routed rows."""
    counts = (torch.arange(size * size, device="cpu").reshape(size, size) * 37 + 17) % 133
    if pattern == "uniform":
        counts.fill_(128)
    elif pattern == "hotspot":
        counts.zero_()
        counts[:, 0] = 131 + torch.arange(size, device="cpu")
    elif pattern == "empty_requester":
        counts[0].zero_()
    elif pattern in ("empty", "tiny"):
        counts.zero_()
        if pattern == "tiny":
            counts[0, -1] = 1
    transport = embedding_impl._EngramAllToAll(None, "hybridep", 64)
    for matrix in (counts, counts.T):
        destinations = [
            [peer for peer, count in enumerate(row) for _ in range(count)]
            for row in matrix.tolist()
        ]
        max_rows = max(map(len, destinations))
        rounds = (max_rows + 63) // 64
        expected = [[0] * size for _ in range(rounds)]
        for rows in destinations:
            for round_id in range(rounds):
                targets = rows[round_id::rounds]
                assert len(targets) <= 64
                for peer in targets:
                    expected[round_id][peer] += 1
        for rank in range(size):
            with patch.object(transport, "rank", rank):
                plan = transport._hybridep_plan(matrix)
            assert plan.max_rows == max_rows
            assert plan.recv_rows == tuple(row[rank] for row in expected)
            assert sum(plan.recv_rows) == int(matrix[:, rank].sum())
            if pattern == "uniform":
                # Every rank receives work in every round, without a first-round hotspot.
                assert set(plan.recv_rows) == {64}


@pytest.fixture
def hybridep_group():
    """Exercise the installed HybridEP kernels on a real multi-rank CUDA group."""
    if not torch.cuda.is_available():
        pytest.skip("HybridEP needs CUDA")
    deep_ep = pytest.importorskip("deep_ep", exc_type=ImportError)
    if not hasattr(deep_ep, "HybridEPBuffer"):
        pytest.skip("DeepEP build has no HybridEPBuffer")
    Utils.initialize_distributed()
    if dist.get_world_size() < 2:
        pytest.skip("HybridEP transport test needs at least two ranks")
    yield dist.group.WORLD
    torch.cuda.synchronize()
    dist.barrier()


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("chunk_size", [64, 4096])
@pytest.mark.parametrize(
    "dtype,width",
    [(torch.int64, 2), (torch.uint8, 264), (torch.bfloat16, 256), (torch.float32, 256)],
)
def test_hybridep_engram_bytes_counts_and_empty_ranks(hybridep_group, dtype, width, chunk_size):
    """Byte-exact transport including BF16 NaN payloads, large IDs and all-to-one load."""
    group = hybridep_group
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    reference = embedding_impl._EngramAllToAll(group)
    transport = embedding_impl._EngramAllToAll(
        group, "hybridep", chunk_size, width * dtype.itemsize
    )
    for pattern in ("uneven", "hotspot", "empty_requester", "empty"):
        splits = [chunk_size + 1 + (rank + peer) % 3 for peer in range(size)]
        if pattern == "hotspot":
            splits = [2 * chunk_size + 1 + rank] + [0] * (size - 1)
        if pattern == "empty" or (pattern == "empty_requester" and rank == 0):
            splits = [0] * size
        # Two tables per peer ensure the count transpose preserves both axes.
        counts = torch.tensor([[n // 2, n - n // 2] for n in splits], device="cuda").flatten()
        sent, recv_splits, table_counts, _ = reference.exchange_counts(counts)
        assert sent == splits
        elements = torch.arange(sum(splits) * width, device="cuda").reshape(sum(splits), width)
        if dtype == torch.bfloat16:
            # Covers every 16-bit pattern, including NaNs/subnormals/signed zero.
            payload = (elements + rank).to(torch.int16).view(torch.bfloat16)
        elif dtype == torch.int64:
            payload = elements + (1 << 50) + rank * (1 << 40)
        else:
            payload = elements.to(dtype)
        expected = reference.exchange(payload, recv_splits, splits)
        with patch.object(dist, "all_to_all_single", side_effect=AssertionError("NCCL A2A used")):
            actual_sent, actual_recv, actual_tables, plans = transport.exchange_counts(counts)
            assert (actual_sent, actual_recv, actual_tables) == (splits, recv_splits, table_counts)
            actual = transport.exchange(payload, recv_splits, splits, plans[0])
            torch.testing.assert_close(
                actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )
            # Replaying a planned exchange must not synchronize max_rows again.
            with (
                patch.object(dist, "all_reduce", side_effect=AssertionError("max_rows all-reduce")),
                patch.object(
                    transport.buffer,
                    "dispatch_with_permute",
                    wraps=transport.buffer.dispatch_with_permute,
                ) as dispatch,
            ):
                replay = transport.exchange(payload, recv_splits, splits, plans[0])
            torch.testing.assert_close(
                replay.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )
            assert len(dispatch.call_args_list) == len(plans[0].recv_rows)
            for call, expected_rows in zip(dispatch.call_args_list, plans[0].recv_rows):
                assert call.kwargs["non_blocking"] is True
                assert call.kwargs["num_permuted_tokens"] == expected_rows
            restored = transport.exchange(actual, splits, recv_splits, plans[1])
            torch.testing.assert_close(
                restored.view(torch.uint8), payload.view(torch.uint8), rtol=0, atol=0
            )


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("fp8_storage", [False, True])
def test_hybridep_engram_lookup_and_gradients(hybridep_group, fp8_storage):
    """Two live lookups per module retain independent plans and match NCCL exactly."""
    if fp8_storage and not _IS_BLACKWELL:
        pytest.skip("MXFP8 needs Blackwell")
    group = hybridep_group
    rank = dist.get_rank(group)
    config = _config()
    config.params_dtype = torch.bfloat16
    config.use_cpu_initialization = False
    models = [
        EPShardedMultiTableEmbedding(
            config,
            (97, 101),
            256,
            config.init_method,
            group,
            fp8_storage=fp8_storage,
            communication_backend=backend,
            hybridep_chunk_size=64,
        )
        for backend in ("nccl", "hybridep", "hybridep")
    ]
    # The same initialized weights must survive two simultaneously live transports.
    for model in models[1:]:
        model.load_state_dict(models[0].state_dict())
    for pattern in ("uneven", "hotspot", "empty_requester", "empty"):
        tokens = (
            0 if pattern == "empty" or (pattern == "empty_requester" and rank == 0) else 37 + rank
        )
        ids = torch.randint(0, 97, (tokens, 3, 2), device="cuda")
        ids[:3] = 0
        if pattern == "hotspot":
            ids.zero_()
        gradient = torch.randn(tokens, 3 * 2 * 256, device="cuda", dtype=torch.bfloat16)
        outputs = []
        for model in models:
            model.zero_grad(set_to_none=True)
            context = patch.object(
                dist, "all_to_all_single", side_effect=AssertionError("NCCL A2A used")
            )
            # The reference must still exercise real NCCL A2A.
            if model is models[0]:
                context = nullcontext()
            with context:
                results, losses = [], []
                for lookup_ids, lookup_grad in ((ids, gradient), (ids.flip(0)[::2], gradient[::2])):
                    count = lookup_ids.shape[0]
                    if fp8_storage:
                        output = model.lookup_quantized(lookup_ids, 3 * 2 * 256)
                        results.extend(
                            (output._rowwise_data.clone(), output._rowwise_scale_inv.clone())
                        )
                        decoded = output.dequantize()[:count]
                    else:
                        decoded = model(lookup_ids).reshape(count, 3 * 2 * 256)
                        results.append(decoded.detach())
                    losses.append((decoded * lookup_grad).sum())
                # Backward for the first lookup runs after the second lookup has
                # installed different counts, including a different round count.
                for loss in losses:
                    loss.backward()
                outputs.append(results)
        for result, model in zip(outputs[1:], models[1:]):
            for actual, expected in zip(result, outputs[0]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(model.parameters(), models[0].parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)
        assert models[1].communication.buffer is not models[2].communication.buffer


def test_engram_hybridep_single_rank_needs_no_extension():
    """Selecting HybridEP at EP1 stays a local lookup, including on CPU."""
    config = _config()
    model = EPShardedMultiTableEmbedding(
        config, (97,), 3, config.init_method, communication_backend="hybridep"
    )
    ids = torch.tensor([[0], [96], [0]])
    output = model(ids)
    torch.testing.assert_close(output, model.tables[0].weight[ids])
    output.sum().backward()
    torch.testing.assert_close(model.tables[0].weight.grad[0], torch.full((3,), 2.0))
    assert model.communication.buffer is None


@pytest.mark.parametrize('defer', [False, True])
@pytest.mark.parametrize('tables', [(13,), (13, 7)])
def test_prefetch_duplicate_rows_and_deferred_gradients(defer, tables):
    """Reordered microbatch backwards match dense lookup, without dense per-lookup wgrad."""
    torch.manual_seed(91)
    model = EPShardedMultiTableEmbedding(_config(), tables, 4, _config().init_method)
    reference = [table.weight.detach().clone().requires_grad_() for table in model.tables]
    runtime = embedding_impl.EngramRuntime(1, defer_grad_transfer=defer)
    runtime.attach(model)
    ids = torch.tensor([[0, 1, 1, 0, 2, 1], [3, 3, 0, 1, 3, 3]])[..., None].expand(
        -1, -1, len(tables)
    )
    tickets = [runtime.prefetch(model, ids), runtime.prefetch(model, ids.flip(1))]
    outputs = [runtime.consume(ticket) for ticket in tickets]
    for i in (1, 0):
        addresses = ids if i == 0 else ids.flip(1)
        expected = torch.stack(
            [weight[addresses[..., j]] for j, weight in enumerate(reference)], -2
        )
        torch.testing.assert_close(outputs[i], expected, rtol=0, atol=0)
        grad = torch.randn_like(expected)
        outputs[i].backward(grad)
        expected.backward(grad)
    if defer:
        assert all(table.weight.grad is None for table in model.tables)
        runtime.finish_deferred_gradients()
    for table, ref in zip(model.tables, reference):
        torch.testing.assert_close(table.weight.grad, ref.grad, rtol=1e-6, atol=2e-6)
    assert runtime.used == 0
    with pytest.raises(RuntimeError, match='already been consumed'):
        runtime.consume(tickets[0])


def test_parameter_domains_have_stable_optimizer_groups(monkeypatch):
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.process_groups_config import get_parameter_parallel_domain

    monkeypatch.setattr(dist, 'get_world_size', lambda **kwargs: 1)
    monkeypatch.setattr(
        dist, 'all_gather_object', lambda out, value, **kwargs: out.__setitem__(0, value)
    )
    ordered = []
    for names in (['dense', 'engram'], ['engram', 'dense']):
        model = torch.nn.Module()
        for name in names:
            param = torch.nn.Parameter(torch.ones(2, 2))
            if name == 'engram':
                param.parallel_domain = name
                param.allreduce = False
            model.register_parameter(name, param)
        ordered.append(_get_param_groups([model], OptimizerConfig(lr=0.1), {}))
    assert [get_parameter_parallel_domain(group) for group in ordered[0]] == ['dense', 'engram']
    assert [get_parameter_parallel_domain(group) for group in ordered[1]] == ['dense', 'engram']
    saved = [
        {**group, '_tag': get_parameter_parallel_domain(group)} for group in reversed(ordered[0])
    ]
    restored = MegatronOptimizer._filter_and_reorder_param_groups(ordered[1], saved)
    assert [group['_tag'] for group in restored] == ['dense', 'engram']


def test_prefetch_eviction_and_main_grad_accumulation():
    model = EPShardedMultiTableEmbedding(_config(), (19,), 4, _config().init_method)
    runtime = embedding_impl.EngramRuntime(1, defer_grad_transfer=True)
    runtime.attach(model)
    param = model.tables[0].weight
    param.main_grad = torch.zeros_like(param)
    ids = torch.tensor([[1], [1], [3]])
    first = runtime.prefetch(model, ids)
    runtime.budget = first.capacity
    second = runtime.prefetch(model, ids + 1)
    assert first.payload is not None and second.payload is None
    runtime.consume(first).sum().backward()
    assert second.payload is None
    runtime.consume(second).sum().backward()
    runtime.finish_deferred_gradients()
    expected = torch.zeros_like(param)
    expected[1:5] = torch.tensor([2.0, 2.0, 1.0, 1.0])[:, None]
    torch.testing.assert_close(param.main_grad, expected)
    assert param.grad is None and runtime.used == 0
    assert runtime.peak <= runtime.budget
    stale = runtime.prefetch(model, ids)
    runtime.reset()
    with pytest.raises(RuntimeError, match='earlier training step'):
        runtime.consume(stale)


@pytest.mark.parametrize(
    'field,value',
    [
        ('parallel_size', 0),
        ('parallel_size', True),
        ('prefetch', 1),
        ('defer_grad_transfer', 'true'),
        ('buffer_budget_mb', 0),
    ],
)
def test_engram_runtime_config_rejects_invalid_values(field, value):
    from dataclasses import asdict

    options = asdict(EngramConfig([0], [11], 2, 11, 1, 4, 0, 20))
    options[field] = value
    with pytest.raises(ValueError):
        EngramConfig(**options)


def _independent_engram_worker(rank, path, use_layout):
    from megatron.core.process_groups_config import create_row_parallel_groups
    from megatron.core.utils import check_param_hashes_across_dp_replicas
    from megatron.training.utils import common_utils

    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=4)
    try:
        world = dist.group.WORLD
        shard, replicas = create_row_parallel_groups(
            world, 2, world_group=world, name='ENGRAM_TEST'
        )
        for member in range(4):
            group = dist.new_group([member], backend='gloo')
            if rank == member:
                local = group
        assert dist.get_process_group_ranks(shard) == [2 * (rank // 2), 2 * (rank // 2) + 1]
        assert dist.get_process_group_ranks(replicas) == [rank % 2, rank % 2 + 2]
        groups = ProcessGroupCollection(
            dp_cp=world,
            expt_dp=world,
            tp=local,
            expt_tp=local,
            mp=local,
            tp_ep_pp=local,
            gtp_remat=local,
            expt_gtp_remat=local,
            engram=shard,
            engram_dp=replicas,
            parameter_replica_groups={'engram': replicas},
        )
        model = EPShardedMultiTableEmbedding(
            _config(), (17,), 4, _config().init_method, ep_group=shard, expt_dp_group=replicas
        )
        weight = model.tables[0].weight
        weight.parallel_domain = 'engram'
        weight.is_managed_by_layer_wise_optimizer = True
        weight.use_sinkhorn = True
        weight.optimizer_owner_index = 0
        full = torch.arange(68, dtype=torch.float32).view(17, 4).div(100).requires_grad_()
        start, end = model.tables[0].row_start, model.tables[0].row_end
        weight.data.copy_(full[start:end])
        weight.main_grad = torch.zeros_like(weight)
        runtime = embedding_impl.EngramRuntime(1, defer_grad_transfer=True)
        runtime.attach(model)

        # Empty requester, duplicate rows, and uneven ownership in the same exchange.
        def addresses(peer, microbatch):
            return torch.tensor(
                [] if peer == 0 else [peer, peer, 16, (peer + microbatch) % 17], dtype=torch.long
            ).view(-1, 1)

        tickets = [runtime.prefetch(model, addresses(rank, mb)) for mb in range(2)]
        for mb in (1, 0):
            result = runtime.consume(tickets[mb])
            torch.testing.assert_close(result, full[addresses(rank, mb)], rtol=0, atol=0)
            result.sum().backward()
        runtime.finish_deferred_gradients()
        dist.all_reduce(weight.main_grad, group=replicas)
        for peer in range(4):
            for mb in range(2):
                full[addresses(peer, mb)].sum().backward()
        torch.testing.assert_close(weight.main_grad, full.grad[start:end], rtol=0, atol=0)
        config = DistributedDataParallelConfig(
            use_distributed_optimizer=use_layout, use_layer_wise_param_layout=use_layout
        )
        layout = LayerWiseDistributedOptimizer.compute_full_param_layout(
            [weight],
            None,
            4,
            config,
            expert_data_parallel_world_size=4,
            parallel_group_sizes={'engram': 2},
        )
        key, buffer = next(iter(layout.layouts.items()))
        assert key.domain == 'engram' and not key.is_expert_parallel
        if use_layout:
            assert buffer.num_optimizer_shards == 2
        optimizer = object.__new__(LayerWiseDistributedOptimizer)
        optimizer.pg_collection, optimizer.dp_cp, optimizer.expt_dp = groups, world, world
        inner = torch.optim.SGD(
            [{'params': [weight], 'parallel_domain': 'engram', 'is_expert_parallel': False}], lr=0.1
        )
        optimizer.shard_params([inner], [layout] if use_layout else None)
        assert bool(inner.param_groups[0]['params']) == (replicas.rank() == 0)
        weight.grad = weight.main_grad
        inner.step()
        DistributedDataParallel.broadcast_params(
            SimpleNamespace(module=model, parameter_replica_groups={'engram': replicas})
        )
        torch.testing.assert_close(weight, (full - 0.1 * full.grad)[start:end])
        assert check_param_hashes_across_dp_replicas([model], True, pg_collection=groups)

        # Norm logging counts each row shard once, also when only one replica
        # owns its FP32 master. Use CPU math in place of the CUDA norm kernel.
        def norm_squared(tensors):
            return sum((tensor.float().square().sum() for tensor in tensors), torch.zeros(()))

        for master in (False, True):
            if master:
                weight.main_param_sharded = True
                weight.main_param = weight.detach().clone() if replicas.rank() == 0 else None
            with (
                patch.object(common_utils, 'get_args', return_value=SimpleNamespace(bf16=master)),
                patch.object(common_utils, '_compute_norm_2', side_effect=norm_squared),
            ):
                norm = common_utils.calc_params_l2_norm(model, pg_collection=groups)
            assert norm == pytest.approx((full - 0.1 * full.grad).norm().item(), rel=1e-6)
        with torch.no_grad():
            weight.add_(replicas.rank())
        assert not check_param_hashes_across_dp_replicas([model], True, pg_collection=groups)
        checkpoint = model.sharded_state_dict()
        saved = checkpoint['tables.0.weight']
        assert saved.global_offset[0] == start
        assert saved.replica_id[-1] == replicas.rank()
        # Every rank must reject invalid geometry before starting any new group.
        with pytest.raises(ValueError, match='positive divisor'):
            create_row_parallel_groups(world, 3, world_group=world, name='INVALID')
        with pytest.raises(ValueError, match='agree on all ranks'):
            create_row_parallel_groups(
                world, 1 if rank == 0 else 2, world_group=world, name='MISMATCH'
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('use_layout', [False, True])
def test_independent_engram_groups_lookup_gradients_and_optimizer_owners(use_layout):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_independent_engram_worker, args=(directory + '/rendezvous', use_layout), nprocs=4)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not _IS_BLACKWELL, reason='MXFP8 needs Blackwell')
@pytest.mark.parametrize('communication_backend', ['nccl', 'hybridep', 'symmetric'])
def test_prefetched_mxfp8_deferred_gradient_ddp(communication_backend):
    """Real FP8 storage, independent shards and DDP's external gradient readiness."""
    from megatron.core.process_groups_config import create_row_parallel_groups

    Utils.initialize_distributed()
    if dist.get_world_size() != 4:
        pytest.skip('Requires four GPUs: row-shards 2, replicas 2, MoE EP4')
    if communication_backend == 'hybridep':
        pytest.importorskip('deep_ep')
    if communication_backend == 'symmetric':
        pytest.importorskip('nccl.core.device.cute', exc_type=ImportError)
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    try:
        model_parallel_cuda_manual_seed(123)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        pg.engram, pg.engram_dp = create_row_parallel_groups(
            pg.dp_cp, 2, world_group=dist.group.WORLD, name='ENGRAM_RUNTIME_TEST'
        )
        pg.parameter_replica_groups = {'engram': pg.engram_dp}
        model = _cuda_mxfp8_engram(pg, communication_backend=communication_backend)
        embedding, weight = model.embed, model.embed.tables[0].weight
        runtime = embedding_impl.EngramRuntime(8, defer_grad_transfer=True)
        runtime.attach(embedding)
        ddp = DistributedDataParallel(
            model.config,
            DistributedDataParallelConfig(overlap_grad_reduce=True, grad_reduce_in_fp32=True),
            embedding,
            pg_collection=pg,
        )
        assert ddp.buffers == [] and ddp.expert_parallel_buffers == []
        ddp.zero_grad_buffer()
        ids = torch.tensor([[[0], [0], [197], [dist.get_rank()]]], device='cuda')
        ticket = runtime.prefetch(embedding, ids)
        consumer = torch.cuda.Stream()
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            activation = runtime.consume(ticket, 64)
            activation.dequantize()[:2].sum().backward()
        assert weight.grad is None and torch.count_nonzero(weight.main_grad) == 0
        ddp.start_grad_sync()
        for bucket in ddp.bucket_groups_by_domain['engram']:
            assert bucket.grad_reduce_handle is None
        ddp.finish_grad_sync()
        full_grad = torch.zeros(198, 32, device='cuda')
        full_grad[0] += 2
        full_grad[197] += 1
        full_grad[:4] += 0.25
        table = embedding.tables[0]
        torch.testing.assert_close(
            weight.main_grad[: table.local_num_embeddings],
            full_grad[table.row_start : table.row_end],
            rtol=0,
            atol=0,
        )
        assert torch.count_nonzero(weight.main_grad[table.local_num_embeddings :]) == 0
        assert runtime.used == runtime.resident
        runtime.close()
    finally:
        Utils.destroy_model_parallel()


def test_deferred_producer_finishes_before_ddp_reduction():
    from unittest.mock import Mock

    events = []
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    producer = Mock()
    producer.finish_deferred_gradients.side_effect = lambda: events.append('produce')
    bucket = Mock()
    bucket.register_grad_ready.side_effect = lambda *args: events.append('ready')
    bucket.finish_grad_sync.side_effect = lambda **kwargs: events.append('reduce')
    ddp = object.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(ddp)
    ddp.ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True)
    ddp.force_all_reduce = False
    ddp.deferred_grad_producers = (producer,)
    ddp.deferred_grad_params = [parameter]
    ddp.deferred_bucket_groups = {bucket}
    ddp._deferred_grads_finished = False
    ddp.param_to_bucket_group = {parameter: bucket}
    ddp.bucket_groups_by_domain = {'memory': [bucket]}
    ddp.start_grad_sync()
    bucket.start_grad_sync.assert_not_called()
    ddp.finish_grad_sync()
    assert events == ['produce', 'ready', 'reduce']
    ddp.finish_deferred_gradients()
    producer.finish_deferred_gradients.assert_called_once()


class _ReferenceDeviceTransport:
    """CPU reference for the runtime's local-request/ID-plus-gradient contract."""

    def __init__(self, group):
        self.group = group

    def exchange(self, source, recv, send, plan, workspace_bytes):
        output = source.new_empty((sum(recv), *source.shape[1:]))
        dist.all_to_all_single(output, source.contiguous(), recv, send, group=self.group)
        return output

    def lookup(self, key, data, scales, requests, send, recv, plans, workspace_bytes):
        assert not scales
        received = self.exchange(requests, recv, send, plans[0], workspace_bytes)
        result = data[0].new_empty((sum(recv), data[0].shape[1]))
        for table, weight in enumerate(data):
            mask = received[:, 0] == table
            result[mask] = weight[received[mask, 1]]
        return self.exchange(result, send, recv, plans[1], workspace_bytes).view(torch.uint8)

    def close(self):
        pass


def _direct_request_runtime_worker(rank, path, defer):
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=4)
    try:
        with patch.object(
            embedding_impl,
            'require',
            return_value=SimpleNamespace(EngramDeviceTransport=_ReferenceDeviceTransport),
        ):
            model = EPShardedMultiTableEmbedding(
                _config(),
                (7, 2),
                4,
                _config().init_method,
                ep_group=dist.group.WORLD,
                communication_backend='symmetric',
            )
        reference = [torch.arange(size * 4).reshape(size, 4).float() / 13 for size in (7, 2)]
        for table, full in zip(model.tables, reference):
            with torch.no_grad():
                table.weight.copy_(full[table.row_start : table.row_end])
        runtime = embedding_impl.EngramRuntime(1, defer_grad_transfer=defer)
        runtime.attach(model)
        tickets, expected, gradients = [], [], []
        full_grad = [torch.zeros_like(full) for full in reference]
        for step in range(3):
            # Rank 0 asks for no rows; the smaller table has two empty owners.
            rows = torch.arange(0 if rank == 0 else 11 + rank)
            ids = torch.stack(((rows + step) % 7, rows % 2), dim=-1)
            ticket = runtime.prefetch(model, ids)
            assert ticket.requests.shape[0] == sum(ticket.send_splits)
            tickets.append(ticket)
            expected.append(torch.stack([full[ids[:, i]] for i, full in enumerate(reference)], 1))
            grad = torch.arange(ids.numel() * 4).reshape(*ids.shape, 4).float() / 31
            gradients.append(grad)
            for i in range(2):
                full_grad[i].index_add_(0, ids[:, i], grad[:, i])
        for step in (1, 0, 2):
            output = runtime.consume(tickets[step])
            torch.testing.assert_close(output, expected[step], rtol=0, atol=0)
            output.backward(gradients[step])
        runtime.finish_deferred_gradients()
        for table, full in zip(model.tables, full_grad):
            dist.all_reduce(full)
            torch.testing.assert_close(table.weight.grad, full[table.row_start : table.row_end])
        runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('defer', [False, True])
def test_direct_request_runtime_gradient_packets(defer):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_direct_request_runtime_worker, args=(directory + '/rendezvous', defer), nprocs=4)
