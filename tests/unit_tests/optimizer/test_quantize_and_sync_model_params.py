# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Plumbing of ``quantize_and_sync_model_params_from_main_params``.

Chained optimizers (dense + expert, or a LayerWise/DistOpt pair) commonly own the *same*
DDP model chunk, so the per-chunk work has to happen exactly once and in a fixed order:
every optimizer stages its own shards, then the chunk is gathered once.

The refresh must not zero the DDP buffers. Staging writes each rank's own shard and the
forced gather rewrites every byte that is later read back as a param, so a zero buys
nothing -- and under ``--reuse-grad-buf-for-mxfp8-param-ag`` the param buffer aliases the
grad buffer, where clearing a bucket that mixes quantized and non-quantized params would
wipe the non-quantized weights that alias it. ``_post_param_sync`` skips exactly those
buckets for that reason. None of this needs a GPU to pin.
"""

from types import SimpleNamespace

import pytest

from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer


class _FakeModelChunk:
    """Records the per-chunk calls the refresh is supposed to make."""

    def __init__(self, log, name):
        self._log = log
        self._name = name

    def zero_grad_buffer(self):
        self._log.append(('zero', self._name))

    def start_param_sync(self, *unused, force_sync=False, force_dispatch=False):
        self._log.append(('sync', self._name, force_sync))


class _FakeOptimizer:
    """Minimal stand-in for a chained member optimizer."""

    def __init__(self, log, name, config, model_chunks, is_stub_optimizer=False):
        self._log = log
        self._name = name
        self.config = config
        self.model_chunks = model_chunks
        self.is_stub_optimizer = is_stub_optimizer

    def _stage_model_params_from_main_params(self):
        if self.is_stub_optimizer:
            return
        self._log.append(('stage', self._name))


def _build(reuse_grad_buf, share_chunk=True, with_stub=False):
    log = []
    config = SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=reuse_grad_buf)
    chunk_a = _FakeModelChunk(log, 'chunk_a')
    chunk_b = chunk_a if share_chunk else _FakeModelChunk(log, 'chunk_b')
    members = [
        _FakeOptimizer(log, 'dense', config, [chunk_a]),
        _FakeOptimizer(log, 'expert', config, [chunk_b]),
    ]
    if with_stub:
        stub_chunk = _FakeModelChunk(log, 'stub_chunk')
        members.append(_FakeOptimizer(log, 'stub', config, [stub_chunk], is_stub_optimizer=True))
    return ChainedOptimizer(members), log


def test_shared_model_chunk_is_gathered_once():
    """Two optimizers, one shared chunk: one gather, not one per optimizer."""
    optimizer, log = _build(reuse_grad_buf=True)

    optimizer.quantize_and_sync_model_params_from_main_params()

    assert log.count(('sync', 'chunk_a', True)) == 1, log
    assert [entry for entry in log if entry[0] == 'stage'] == [
        ('stage', 'dense'),
        ('stage', 'expert'),
    ], log


def test_every_optimizer_stages_before_the_gather():
    """Gathering early would broadcast shards a sibling optimizer has not written yet."""
    optimizer, log = _build(reuse_grad_buf=True)

    optimizer.quantize_and_sync_model_params_from_main_params()

    order = [entry[0] for entry in log]
    assert max(i for i, k in enumerate(order) if k == 'stage') < order.index('sync'), log


def test_distinct_model_chunks_are_each_gathered():
    optimizer, log = _build(reuse_grad_buf=True, share_chunk=False)

    optimizer.quantize_and_sync_model_params_from_main_params()

    for name in ('chunk_a', 'chunk_b'):
        assert log.count(('sync', name, True)) == 1, log


@pytest.mark.parametrize("reuse_grad_buf", [True, False])
def test_the_refresh_never_zeroes_the_grad_buffer(reuse_grad_buf):
    """Staging plus the forced gather rewrites every param byte, so zeroing buys nothing.

    It is not merely redundant under --reuse-grad-buf-for-mxfp8-param-ag: zeroing a bucket
    that mixes quantized and non-quantized params would clear the non-quantized weights,
    which alias that buffer and are not restaged from the masters.
    """
    optimizer, log = _build(reuse_grad_buf=reuse_grad_buf)

    optimizer.quantize_and_sync_model_params_from_main_params()

    assert not [entry for entry in log if entry[0] == 'zero'], log
    assert log.count(('sync', 'chunk_a', True)) == 1, log


def test_stub_optimizers_stage_nothing():
    """A stub member has no main params to re-derive from, so it must not stage."""
    optimizer, log = _build(reuse_grad_buf=True, with_stub=True)

    optimizer.quantize_and_sync_model_params_from_main_params()

    assert ('stage', 'stub') not in log, log
    assert [entry[1] for entry in log if entry[0] == 'stage'] == ['dense', 'expert'], log


def test_gather_uses_the_chain_s_own_model_chunks():
    """The gather must read self.model_chunks, not re-walk the members.

    ChainedOptimizer.__init__ collects chunks only from members exposing a model_chunks
    attribute. LayerWiseDistributedOptimizer's members are Float16OptimizerWithFloat16Params,
    which do not have one, so __init__ leaves the list empty and LayerWise reassigns
    self.model_chunks afterwards. Recomputing from the members here would discard that
    and silently gather nothing on the whole Muon path.
    """
    log = []
    config = SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=True)

    class _MemberWithoutChunks:
        """Stands in for Float16OptimizerWithFloat16Params: no model_chunks attribute."""

        def __init__(self, name):
            self._name = name
            self.config = config
            self.is_stub_optimizer = False

        def _stage_model_params_from_main_params(self):
            log.append(('stage', self._name))

    chained = ChainedOptimizer([_MemberWithoutChunks('a'), _MemberWithoutChunks('b')])
    assert chained.model_chunks == [], "members expose no chunks, so __init__ finds none"

    # What LayerWiseDistributedOptimizer.__init__ does after super().__init__().
    chunk = _FakeModelChunk(log, 'layerwise_chunk')
    chained.model_chunks = [chunk]

    chained.quantize_and_sync_model_params_from_main_params()

    assert [e[1] for e in log if e[0] == 'stage'] == ['a', 'b'], log
    assert log.count(('sync', 'layerwise_chunk', True)) == 1, log


def test_a_nested_chained_optimizer_stages_its_own_members():
    """LayerWiseDistributedOptimizer is a ChainedOptimizer nested inside the outer one.

    Without ChainedOptimizer._stage_model_params_from_main_params the nested chain inherits
    the base no-op, and its params -- the quantized weights under Muon -- are never
    re-derived. That is silent: no error, just weights that did not get refreshed.
    """
    log = []
    config = SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=True)
    chunk = _FakeModelChunk(log, 'chunk')
    inner = ChainedOptimizer(
        [
            _FakeOptimizer(log, 'inner_a', config, [chunk]),
            _FakeOptimizer(log, 'inner_b', config, [chunk]),
        ]
    )
    outer = ChainedOptimizer([inner, _FakeOptimizer(log, 'outer', config, [chunk])])

    outer.quantize_and_sync_model_params_from_main_params()

    staged = [entry[1] for entry in log if entry[0] == 'stage']
    assert staged == ['inner_a', 'inner_b', 'outer'], log
    assert log.count(('sync', 'chunk', True)) == 1, log


def test_an_empty_chain_refreshes_without_touching_config():
    """A rank with no trainable parameters gets ChainedOptimizer([]).

    __init__ takes the else branch there and never assigns self.config, and it does not
    call super().__init__(), so the attribute does not exist rather than holding None.
    Any config read added to the refresh would raise AttributeError on exactly those
    ranks, which is what the is_stub_optimizer guard and this test exist to prevent.
    """
    chained = ChainedOptimizer([])
    assert chained.is_stub_optimizer
    assert not hasattr(chained, 'config')

    chained.quantize_and_sync_model_params_from_main_params()
    chained._stage_model_params_from_main_params()


def test_a_chain_of_only_stubs_stages_nothing():
    """Every member being a stub makes the chain itself a stub."""
    log = []
    config = SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=True)
    stub = _FakeOptimizer(log, 'stub', config, [], is_stub_optimizer=True)
    chained = ChainedOptimizer([stub])
    assert chained.is_stub_optimizer

    chained.quantize_and_sync_model_params_from_main_params()

    assert log == []


def test_base_optimizer_refresh_is_a_no_op():
    """Optimizers with no main params inherit a no-op rather than failing."""

    class _Bare(MegatronOptimizer):
        def __init__(self):  # deliberately skips MegatronOptimizer.__init__
            pass

        def prepare_grads(self) -> bool: ...
        def step_with_ready_grads(self) -> bool: ...
        def zero_grad(self, set_to_none=True): ...
        def get_loss_scale(self): ...
        def reload_model_params(self, state_dict=None): ...
        def state_dict(self): ...
        def load_state_dict(self, state_dict): ...
        def step(self): ...
        def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None): ...

    _Bare().quantize_and_sync_model_params_from_main_params()
