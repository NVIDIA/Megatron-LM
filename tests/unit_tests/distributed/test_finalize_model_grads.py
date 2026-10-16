# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import importlib
import inspect
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _get_distributed_optimizer_shard_range,
    _select_embedding_grad_shard_range,
    _update_router_expert_bias,
    _update_router_qb_beta,
    finalize_model_grads,
    reset_model_temporary_tensors,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RouterExpertBiasModel(torch.nn.Module):
    def __init__(self, config, local_tokens_per_expert):
        super().__init__()
        self.config = config
        self.ddp_config = DistributedDataParallelConfig()
        self.router = torch.nn.Module()
        self.router.register_buffer("local_tokens_per_expert", local_tokens_per_expert)
        self.router.register_buffer("expert_bias", torch.zeros_like(local_tokens_per_expert))
        self.finish_grad_sync_calls = 0

    def finish_grad_sync(self, force_all_reduce=False):
        del force_all_reduce
        self.finish_grad_sync_calls += 1


class _HashRouterWithoutExpertBias(torch.nn.Module):
    """Match hash-router layers, which intentionally do not own expert-bias state."""

    def __init__(self):
        super().__init__()
        self.expert_bias = None
        self.local_tokens_per_expert = None


def test_hash_router_without_expert_bias_is_ignored():
    router = _HashRouterWithoutExpertBias()
    config = SimpleNamespace(
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="none",
        moe_router_bias_update_rate=0.25,
    )

    reset_model_temporary_tensors(config, [router])
    _update_router_expert_bias([router], config)

    assert router.expert_bias is None
    assert router.local_tokens_per_expert is None


def _router_expert_bias_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_router_bias_update_rate=0.25,
        moe_router_load_balancing_type="none",
    )


_NO_TP_DP_CP = object()


def _router_bias_pg_collection(tp_dp_cp=_NO_TP_DP_CP):
    kwargs = {
        'tp': dist.group.WORLD,
        'pp': dist.group.WORLD,
        'embd': None,
        'pos_embd': None,
        'dp_cp': dist.group.WORLD,
    }
    if tp_dp_cp is not _NO_TP_DP_CP:
        kwargs['tp_dp_cp'] = tp_dp_cp
    return ProcessGroupCollection(**kwargs)


class TestFinalizeModelGradsMoEExpertBias:
    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()
        Utils.initialize_distributed()
        parallel_state.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_model_grads_updates_router_expert_bias_with_custom_group(self):
        assert not parallel_state.model_parallel_is_initialized()

        config = _router_expert_bias_config()
        device = torch.device("cuda", torch.cuda.current_device())
        local_tokens = torch.tensor(
            [0.0, 2.0] if dist.get_rank() == 0 else [0.0, 0.0], device=device
        )
        model = _RouterExpertBiasModel(config, local_tokens)

        finalize_model_grads(
            [model], pg_collection=_router_bias_pg_collection(tp_dp_cp=dist.group.WORLD)
        )

        expected_bias = torch.tensor([0.25, -0.25], device=device)
        torch.testing.assert_close(model.router.expert_bias, expected_bias)
        torch.testing.assert_close(
            model.router.local_tokens_per_expert, torch.zeros_like(local_tokens)
        )
        assert model.finish_grad_sync_calls == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_model_grads_requires_custom_group_before_grad_sync(self):
        assert not parallel_state.model_parallel_is_initialized()
        config = _router_expert_bias_config()
        device = torch.device("cuda", torch.cuda.current_device())
        pg_collections = [
            _router_bias_pg_collection(),
            _router_bias_pg_collection(tp_dp_cp=dist.group.WORLD),
        ]
        pg_collections[1].tp_dp_cp = None

        for pg_collection in pg_collections:
            model = _RouterExpertBiasModel(config, torch.tensor([1.0, 0.0], device=device))
            with pytest.raises(AssertionError, match="tp_dp_cp"):
                finalize_model_grads([model], pg_collection=pg_collection)
            assert model.finish_grad_sync_calls == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("force_all_reduce", [False, True])
    def test_finalize_model_grads_forwards_force_all_reduce_to_embedding_grads(
        self, force_all_reduce
    ):
        config = _router_expert_bias_config()
        device = torch.device("cuda", torch.cuda.current_device())
        model = _RouterExpertBiasModel(config, torch.zeros(2, device=device))
        finalize_module = importlib.import_module("megatron.core.distributed.finalize_model_grads")

        with mock.patch.object(
            finalize_module, "_allreduce_word_embedding_grads"
        ) as allreduce_word_embedding_grads:
            finalize_model_grads(
                [model],
                pg_collection=_router_bias_pg_collection(tp_dp_cp=dist.group.WORLD),
                force_all_reduce=force_all_reduce,
            )

        allreduce_word_embedding_grads.assert_called_once()
        assert allreduce_word_embedding_grads.call_args.kwargs["force_all_reduce"] is (
            force_all_reduce
        )


class TestAllReduceLNGrads:

    def init_model(self, share_embeddings_and_output_weights: bool = False):
        qk_layernorm = True
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            qk_layernorm=qk_layernorm,
            pipeline_dtype=torch.float32,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                qk_layernorm=qk_layernorm
            ),
            vocab_size=100,
            max_sequence_length=4,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("freeze_model,tp_size", [(True, 2), (False, 2)])
    def test_allreduce_layernorm_grads(self, freeze_model, tp_size):
        self.tp_size = tp_size
        self.pp_size = 1
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model()
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)

        _allreduce_non_tensor_model_parallel_grads(
            [self.model], self.transformer_config, parallel_state.get_tensor_model_parallel_group()
        )

    @pytest.mark.parametrize(
        ("freeze_model", "pp_size", "share_embeddings"),
        [(True, 2, True), (False, 2, True), (True, 2, False), (False, 2, False)],
    )
    def test_allreduce_word_embedding_grads(self, freeze_model, pp_size, share_embeddings):
        self.tp_size = 1
        self.pp_size = pp_size
        Utils.initialize_model_parallel(pipeline_model_parallel_size=self.pp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model(share_embeddings)
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group()

        _allreduce_word_embedding_grads([self.model], self.transformer_config, embd_group, pp_group)


def _distributed_optimizer_param_range(
    param_start, param_end, bucket_numel, shard_rank, num_shards, bucket_offset=1000
):
    """The slice of a param that DistributedOptimizer reads for one shard of its bucket."""
    param = object()
    shard_numel = bucket_numel // num_shards
    shard_world_start = bucket_offset + shard_rank * shard_numel
    param_map = DistributedOptimizer._build_model_gbuf_param_range_map(
        {param: (bucket_offset + param_start, bucket_offset + param_end, 0)},
        Range(shard_world_start, shard_world_start + shard_numel),
        bucket_offset,
    )
    if param not in param_map:
        return (0, 0)
    param_range = param_map[param]["param"]
    return (param_range.start, param_range.end)


# (param_start, param_end, bucket_numel, num_shards)
_SHARD_LAYOUTS = [
    (0, 16, 16, 4),  # param fills the bucket exactly
    (0, 222, 256, 2),  # padded bucket tail
    (0, 222, 256, 4),
    (5, 19, 24, 3),  # param at an offset, spanning three shards
    (9, 14, 32, 4),  # param inside a single shard
    (0, 100, 512, 4),  # trailing shards hold only padding
    (64, 286, 512, 8),  # param after another param, padded tail
    (0, 37, 37, 1),  # a single shard owns everything
]


class TestDistributedOptimizerShardRange:

    @pytest.mark.parametrize(
        ("param_start", "param_end", "bucket_numel", "num_shards"), _SHARD_LAYOUTS
    )
    def test_matches_distributed_optimizer_param_ranges(
        self, param_start, param_end, bucket_numel, num_shards
    ):
        ranges = [
            _get_distributed_optimizer_shard_range(
                param_start, param_end, bucket_numel, shard_rank, num_shards
            )
            for shard_rank in range(num_shards)
        ]
        for shard_rank, shard_range in enumerate(ranges):
            assert shard_range == _distributed_optimizer_param_range(
                param_start, param_end, bucket_numel, shard_rank, num_shards
            )

        # The non-empty shard ranges tile the whole param exactly once, in shard order.
        owned = [shard_range for shard_range in ranges if shard_range[1] > shard_range[0]]
        assert owned[0][0] == 0
        assert owned[-1][1] == param_end - param_start
        for previous, current in zip(owned, owned[1:]):
            assert previous[1] == current[0]

    def test_explicit_ranges(self):
        def ranges(param_start, param_end, bucket_numel, num_shards):
            return [
                _get_distributed_optimizer_shard_range(
                    param_start, param_end, bucket_numel, shard_rank, num_shards
                )
                for shard_rank in range(num_shards)
            ]

        assert ranges(0, 222, 256, 2) == [(0, 128), (128, 222)]
        assert ranges(5, 19, 24, 3) == [(0, 3), (3, 11), (11, 14)]
        assert ranges(9, 14, 32, 4) == [(0, 0), (0, 5), (0, 0), (0, 0)]
        assert ranges(0, 100, 512, 4) == [(0, 100), (0, 0), (0, 0), (0, 0)]

    @pytest.mark.parametrize(
        ("param_start", "param_end", "bucket_numel", "shard_rank", "num_shards"),
        [
            (0, 10, 10, 0, 3),  # bucket not divisible into equal shards
            (0, 16, 16, 4, 4),  # shard rank out of range
            (0, 16, 16, 0, 0),  # no shards
            (8, 24, 16, 0, 2),  # param ends past the bucket
            (9, 8, 16, 0, 2),  # empty param range with start > end
        ],
    )
    def test_rejects_invalid_layouts(
        self, param_start, param_end, bucket_numel, shard_rank, num_shards
    ):
        with pytest.raises(ValueError):
            _get_distributed_optimizer_shard_range(
                param_start, param_end, bucket_numel, shard_rank, num_shards
            )


def _sharded_layout(bucket_numel, param_start, param_end, dist_opt_rank, dist_opt_size):
    """A layout row of a reduce-scattered distributed-optimizer bucket."""
    reduced = _get_distributed_optimizer_shard_range(
        param_start, param_end, bucket_numel, dist_opt_rank, dist_opt_size
    )
    return (1, bucket_numel, param_start, param_end, dist_opt_rank, dist_opt_size, *reduced)


def _whole_grad_layout(numel):
    """A layout row of a gradient that is not reduce-scattered (fully reduced)."""
    return (0, 0, 0, 0, 0, 0, 0, numel)


class TestSelectEmbeddingGradShardRange:
    """The embedding group's decision from the all-gathered layouts (numel 222)."""

    @pytest.mark.parametrize(
        "layouts",
        [
            # DP=1: both stages hold the whole gradient reduced, whatever their buckets look like.
            [_sharded_layout(256, 0, 222, 0, 1), _sharded_layout(384, 64, 286, 0, 1)],
            # DP=1 with the partner's embedding in an all-reduced (e.g. LayerWise) bucket.
            [_sharded_layout(256, 0, 222, 0, 1), _whole_grad_layout(222)],
            # No stage reduce-scatters the embedding.
            [_whole_grad_layout(222), _whole_grad_layout(222)],
            # The whole param lies in this DP rank's shard on both stages.
            [_sharded_layout(512, 0, 222, 0, 2), _sharded_layout(768, 64, 286, 0, 2)],
        ],
    )
    def test_whole_gradient_reduced_everywhere_keeps_full_allreduce(self, layouts):
        assert _select_embedding_grad_shard_range(layouts, 222, [0, 4]) is None

    def test_identical_partial_layouts_select_the_shard(self):
        layouts = [_sharded_layout(256, 0, 222, 1, 2)] * 2
        assert _select_embedding_grad_shard_range(layouts, 222, [1, 5]) == (128, 222)

    def test_same_reduced_range_with_different_buckets_selects_the_shard(self):
        # The shard boundary falls at the same param offset on both stages.
        layouts = [_sharded_layout(16, 4, 12, 1, 2), _sharded_layout(24, 8, 16, 1, 2)]
        assert layouts[0][-2:] == layouts[1][-2:] == (4, 8)
        assert _select_embedding_grad_shard_range(layouts, 8, [1, 5]) == (4, 8)

    def test_no_reduced_elements_selects_an_empty_slice(self):
        # Neither stage's optimizer reads any of the gradient on this DP rank.
        layouts = [_sharded_layout(256, 0, 100, 1, 2), _sharded_layout(512, 0, 100, 1, 2)]
        assert _select_embedding_grad_shard_range(layouts, 100, [1, 5]) == (0, 0)

    @pytest.mark.parametrize(
        "partner",
        [
            _sharded_layout(320, 64, 286, 1, 2),  # same numel, shards cut it elsewhere
            _sharded_layout(256, 0, 222, 0, 2),  # mispaired DP ranks
            _whole_grad_layout(222),  # partner all-reduced, this rank only owns a shard
        ],
    )
    def test_different_reduced_ranges_raise(self, partner):
        layouts = [_sharded_layout(256, 0, 222, 1, 2), partner]
        with pytest.raises(RuntimeError, match="different distributed-optimizer layouts"):
            _select_embedding_grad_shard_range(layouts, 222, [1, 5])

    def test_invalid_layout_raises_on_every_rank(self):
        invalid = (-1, 10, 0, 10, 0, 3, 0, 0)
        layouts = [invalid, invalid]
        with pytest.raises(RuntimeError, match="rank 5: eligible=-1"):
            _select_embedding_grad_shard_range(layouts, 10, [1, 5])
        with pytest.raises(RuntimeError, match="bucket layout is invalid: bad shards"):
            _select_embedding_grad_shard_range(layouts, 10, [1, 5], local_error="bad shards")


class _SharedEmbeddingStage(torch.nn.Module):
    """One pipeline stage holding a copy of a tied word embedding plus a few dense layers."""

    def __init__(
        self,
        pre_process: bool,
        mark_shared_embedding: bool = True,
        extra_param: bool = False,
        vocab_size: int = 61,
        hidden_size: int = 6,
    ):
        super().__init__()
        self.pre_process = pre_process
        self.share_embeddings_and_output_weights = True
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(2)]
        )
        self.word_embeddings = torch.nn.Parameter(torch.zeros(vocab_size, hidden_size))
        if mark_shared_embedding:
            self.word_embeddings.shared_embedding = True
        if extra_param:
            # Registered last, so DDP's reverse-order layout places it before the embedding.
            self.extra = torch.nn.Parameter(torch.zeros(7))

    def shared_embedding_or_output_weight(self):
        return self.word_embeddings


def _local_grad(param, pp_rank, dp_rank):
    """A distinct, exactly representable local gradient per (param, pp rank, dp rank)."""
    base = torch.arange(param.numel(), device=param.device) % 13 + 1
    return (base.float() * (1 + 10 * pp_rank + dp_rank)).view(param.shape)


def _fill_local_grads(model):
    model.zero_grad_buffer()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
    for param in model.module.parameters():
        if param.requires_grad:
            param.main_grad.copy_(_local_grad(param, pp_rank, dp_rank))


def _expected_embedding_grad(weight):
    """Data-parallel mean of the local grads, summed over the two pipeline stages."""
    dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    total = sum(
        _local_grad(weight, pp_rank, dp_rank).double()
        for pp_rank in range(pp_size)
        for dp_rank in range(dp_size)
    )
    return (total / dp_size).float()


def _owned_param_range(model, weight):
    """The slice of ``weight`` that this rank's DistributedOptimizer reads after the grad sync."""
    for buffer in model.buffers:
        if weight in buffer.param_to_bucket:
            bucket_id = buffer.param_to_bucket[weight].bucket_id
            param_map = DistributedOptimizer._build_model_gbuf_range(buffer, bucket_id)["param_map"]
            if weight not in param_map:
                return (0, 0)
            param_range = param_map[weight]["param"]
            return (param_range.start, param_range.end)
    raise AssertionError("weight is not in any grad buffer")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSharedEmbeddingGradShardAllReduce:
    """PP=2 x DP: all-reduce only the distributed-optimizer shard of the tied embedding grad."""

    def setup_method(self, method):
        Utils.destroy_model_parallel()
        if Utils.world_size < 4 or Utils.world_size % 2 != 0:
            pytest.skip("Needs an even number of ranks >= 4 for PP=2 x DP>=2.")
        self._init_parallel()
        self.config = TransformerConfig(num_layers=1, hidden_size=6, num_attention_heads=1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _init_parallel(self, tensor_model_parallel_size=1):
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        self.pp_group = parallel_state.get_pipeline_model_parallel_group()
        self.embd_group = parallel_state.get_embedding_group()
        self.is_last_stage = parallel_state.get_pipeline_model_parallel_rank() == 1

    def _build(
        self,
        ddp_config,
        use_layer_wise=False,
        tag_embedding=False,
        freeze_embedding=False,
        **stage_kwargs,
    ):
        from megatron.training.training import wrap_model_chunks_with_ddp

        module = _SharedEmbeddingStage(pre_process=not self.is_last_stage, **stage_kwargs).cuda()
        if tag_embedding:
            module.word_embeddings.is_embedding_or_output_parameter = True
        if freeze_embedding:
            module.word_embeddings.requires_grad_(False)
        model = wrap_model_chunks_with_ddp(
            [module],
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_config,
            use_layer_wise_distributed_optimizer=use_layer_wise,
        )[0]
        return model, module.word_embeddings

    def _allreduce(self, model, force_all_reduce=False):
        _allreduce_word_embedding_grads(
            [model], self.config, self.embd_group, self.pp_group, force_all_reduce=force_all_reduce
        )

    def _old_full_allreduce(self, model, weight, force_all_reduce=False):
        """The previous behaviour: the same grad sync followed by a full all-reduce."""
        _fill_local_grads(model)
        model.finish_grad_sync(force_all_reduce=force_all_reduce)
        dist.all_reduce(weight.main_grad, group=self.embd_group)
        return weight.main_grad.detach().clone()

    @pytest.mark.parametrize("use_layer_wise", [False, True])
    def test_shard_allreduce_matches_full_allreduce_on_owned_shard(self, use_layer_wise):
        # With LayerWise (Muon) the DDP wrapper forces the chunk-level DistOpt flag on, and the
        # tagged embedding lands in the reduce-scattered DistOpt (Adam) buffer.
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=not use_layer_wise
        )
        model, weight = self._build(
            ddp_config, use_layer_wise=use_layer_wise, tag_embedding=use_layer_wise
        )
        assert model.ddp_config.use_distributed_optimizer
        assert model.param_to_bucket_group[weight].ddp_config.use_distributed_optimizer

        numel = weight.numel()
        start, end = _owned_param_range(model, weight)
        assert 0 <= start < end <= numel and end - start < numel

        _fill_local_grads(model)
        model.finish_grad_sync()
        after_dp_sync = weight.main_grad.detach().clone().view(-1)
        self._allreduce(model)
        sharded = weight.main_grad.detach().clone().view(-1)
        assert model._embedding_grad_shard_ranges[weight] == (start, end)

        full = self._old_full_allreduce(model, weight).view(-1)
        expected = _expected_embedding_grad(weight).view(-1)

        # The shard the optimizer reads is bitwise identical to the old full all-reduce.
        assert torch.equal(sharded[start:end], full[start:end])
        torch.testing.assert_close(sharded[start:end], expected[start:end])
        # The rest keeps what the reduce-scatter left there, which the optimizer never reads;
        # it differs from the full all-reduce, so the test does see the shard-only path.
        not_owned = torch.ones(numel, dtype=torch.bool, device=sharded.device)
        not_owned[start:end] = False
        assert torch.equal(sharded[not_owned], after_dp_sync[not_owned])
        assert not torch.equal(sharded[not_owned], full[not_owned])

    def test_cached_layout_is_reused_inside_cuda_graph_capture(self):
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        )
        model, weight = self._build(ddp_config)
        _fill_local_grads(model)
        model.finish_grad_sync()
        # Eager step: runs the one-time layout check and caches the shard range.
        self._allreduce(model)
        start, end = model._embedding_grad_shard_ranges[weight]

        _fill_local_grads(model)
        model.finish_grad_sync()
        after_dp_sync = weight.main_grad.detach().clone().view(-1)
        with (
            mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            mock.patch.object(dist, "all_gather", side_effect=AssertionError("layout rechecked")),
        ):
            self._allreduce(model)
        torch.cuda.synchronize()
        sharded = weight.main_grad.detach().clone().view(-1)

        expected = _expected_embedding_grad(weight).view(-1)
        torch.testing.assert_close(sharded[start:end], expected[start:end])
        not_owned = torch.ones(weight.numel(), dtype=torch.bool, device=sharded.device)
        not_owned[start:end] = False
        assert torch.equal(sharded[not_owned], after_dp_sync[not_owned])

    def test_sliced_allreduce_replays_in_cuda_graph(self):
        if torch.cuda.nccl.version() < (2, 9, 6):
            pytest.skip("Capturing NCCL collectives in a CUDA graph needs NCCL >= 2.9.6.")
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        )
        model, weight = self._build(ddp_config)
        _fill_local_grads(model)
        model.finish_grad_sync()
        # Eager step: runs the one-time layout check (a host sync) and caches the shard range.
        self._allreduce(model)
        start, end = model._embedding_grad_shard_ranges[weight]
        assert 0 <= start < end and end - start < weight.numel()

        # Capture the cross-stage all-reduce the way FullCudaGraphWrapper captures a whole
        # iteration: synchronized, behind barriers, on a side stream.
        _fill_local_grads(model)
        model.finish_grad_sync()
        after_dp_sync = weight.main_grad.detach().clone().view(-1)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        dist.barrier()
        with mock.patch.object(dist, "all_gather", side_effect=AssertionError("layout rechecked")):
            with torch.cuda.graph(
                graph, stream=torch.cuda.Stream(), capture_error_mode="thread_local"
            ):
                self._allreduce(model)
        torch.cuda.synchronize()
        dist.barrier()
        # Capturing only records the all-reduce.
        assert torch.equal(weight.main_grad.view(-1), after_dp_sync)

        # Replay on freshly synced local grads in the same grad buffer.
        _fill_local_grads(model)
        model.finish_grad_sync()
        graph.replay()
        torch.cuda.synchronize()
        replayed = weight.main_grad.detach().clone().view(-1)

        full = self._old_full_allreduce(model, weight).view(-1)
        assert not torch.equal(after_dp_sync[start:end], full[start:end])
        assert torch.equal(replayed[start:end], full[start:end])
        not_owned = torch.ones(weight.numel(), dtype=torch.bool, device=replayed.device)
        not_owned[start:end] = False
        assert torch.equal(replayed[not_owned], after_dp_sync[not_owned])

    def test_capture_without_checked_layout_falls_back_to_full_allreduce(self):
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        )
        model, weight = self._build(ddp_config)
        _fill_local_grads(model)
        model.finish_grad_sync()
        with (
            mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            mock.patch.object(dist, "all_gather", side_effect=AssertionError("checked in capture")),
        ):
            self._allreduce(model)
        torch.cuda.synchronize()
        result = weight.main_grad.detach().clone()
        assert weight not in model._embedding_grad_shard_ranges

        assert torch.equal(result, self._old_full_allreduce(model, weight))

    @pytest.mark.parametrize("case", ["force_all_reduce", "no_distributed_optimizer", "layer_wise"])
    def test_full_allreduce_fallbacks(self, case):
        force_all_reduce = case == "force_all_reduce"
        use_layer_wise = case == "layer_wise"
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=force_all_reduce
        )
        # Untagged, the 2D embedding is a LayerWise (Muon) param in an all-reduced buffer,
        # although the wrapper forces the chunk-level DistOpt flag on.
        model, weight = self._build(ddp_config, use_layer_wise=use_layer_wise)
        assert model.ddp_config.use_distributed_optimizer == (force_all_reduce or use_layer_wise)
        if use_layer_wise:
            assert not model.param_to_bucket_group[weight].ddp_config.use_distributed_optimizer

        _fill_local_grads(model)
        model.finish_grad_sync(force_all_reduce=force_all_reduce)
        self._allreduce(model, force_all_reduce=force_all_reduce)
        result = weight.main_grad.detach().clone()
        assert getattr(model, "_embedding_grad_shard_ranges", {}).get(weight) is None

        # Every element is reduced, exactly as before.
        assert torch.equal(result, self._old_full_allreduce(model, weight, force_all_reduce))
        torch.testing.assert_close(result, _expected_embedding_grad(weight))

    def test_frozen_embedding_is_skipped(self):
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        )
        model, weight = self._build(ddp_config, freeze_embedding=True)
        assert weight.grad is None and not hasattr(weight, "main_grad")
        _fill_local_grads(model)
        model.finish_grad_sync()
        with (
            mock.patch.object(dist, "all_reduce", side_effect=AssertionError("all-reduced")),
            mock.patch.object(dist, "all_gather", side_effect=AssertionError("checked")),
        ):
            self._allreduce(model)

    def test_mismatched_layouts_raise_on_every_stage(self):
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        )
        # The last stage neither isolates the embedding in its own bucket nor starts it at
        # offset 0, so its shards cut the embedding differently from the first stage.
        model, weight = self._build(
            ddp_config, mark_shared_embedding=not self.is_last_stage, extra_param=self.is_last_stage
        )
        _fill_local_grads(model)
        model.finish_grad_sync()
        with pytest.raises(RuntimeError, match="different distributed-optimizer layouts"):
            self._allreduce(model)

    def test_tiny_embedding_is_whole_or_empty_in_each_shard(self):
        if parallel_state.get_data_parallel_world_size(with_context_parallel=True) > 4:
            pytest.skip("The tiny embedding must fit in one shard of its 128-element bucket.")
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        )
        # 5 x 6 = 30 elements in a bucket padded to 128: DP rank 0's shard holds all of the
        # embedding and every other DP rank's shard holds none of it, on both stages.
        model, weight = self._build(ddp_config, vocab_size=5)
        numel = weight.numel()
        holds_all = parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
        assert _owned_param_range(model, weight) == ((0, numel) if holds_all else (0, 0))

        _fill_local_grads(model)
        model.finish_grad_sync()
        after_dp_sync = weight.main_grad.detach().clone()
        if holds_all:
            self._allreduce(model)
        else:
            # Neither stage's optimizer reads the gradient here, so nothing is all-reduced.
            with mock.patch.object(dist, "all_reduce", side_effect=AssertionError("all-reduced")):
                self._allreduce(model)
        result = weight.main_grad.detach().clone()
        full = self._old_full_allreduce(model, weight)

        if holds_all:
            assert model._embedding_grad_shard_ranges[weight] is None
            assert torch.equal(result, full)
            torch.testing.assert_close(result, _expected_embedding_grad(weight))
        else:
            assert model._embedding_grad_shard_ranges[weight] == (0, 0)
            assert torch.equal(result, after_dp_sync)

    def test_force_all_reduce_attribute_keeps_full_allreduce(self):
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        )
        model, weight = self._build(ddp_config)
        # Training sets the DDP attribute along with the argument; with overlap_grad_reduce the
        # backward hooks pick the data-parallel collective from the attribute alone.
        model.force_all_reduce = True
        _fill_local_grads(model)
        model.finish_grad_sync(force_all_reduce=True)
        self._allreduce(model, force_all_reduce=False)
        result = weight.main_grad.detach().clone()
        assert weight not in getattr(model, "_embedding_grad_shard_ranges", {})
        model.force_all_reduce = False

        assert torch.equal(result, self._old_full_allreduce(model, weight, force_all_reduce=True))
        torch.testing.assert_close(result, _expected_embedding_grad(weight))

    @pytest.mark.parametrize("case", ["different_buckets", "mixed_buckets"])
    def test_no_data_parallelism_keeps_full_allreduce(self, case):
        # TP takes the ranks that PP leaves, so DP=1 and every rank holds the whole gradient
        # reduced, however each stage lays out its buckets.
        self._init_parallel(tensor_model_parallel_size=Utils.world_size // 2)
        assert parallel_state.get_data_parallel_world_size(with_context_parallel=True) == 1
        if case == "mixed_buckets":
            # Only the first stage tags its embedding, so it is in a reduce-scattered DistOpt
            # bucket there and in an all-reduced LayerWise bucket on the last stage.
            ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True)
            model, weight = self._build(
                ddp_config, use_layer_wise=True, tag_embedding=not self.is_last_stage
            )
            assert model.param_to_bucket_group[weight].ddp_config.use_distributed_optimizer == (
                not self.is_last_stage
            )
        else:
            # The stages lay out the embedding differently, as in the mismatch test above.
            ddp_config = DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, use_distributed_optimizer=True
            )
            model, weight = self._build(
                ddp_config,
                mark_shared_embedding=not self.is_last_stage,
                extra_param=self.is_last_stage,
            )

        _fill_local_grads(model)
        model.finish_grad_sync()
        self._allreduce(model)
        result = weight.main_grad.detach().clone()
        assert model._embedding_grad_shard_ranges[weight] is None

        assert torch.equal(result, self._old_full_allreduce(model, weight))
        torch.testing.assert_close(result, _expected_embedding_grad(weight))
