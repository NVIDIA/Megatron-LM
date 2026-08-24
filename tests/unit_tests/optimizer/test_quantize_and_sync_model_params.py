# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Plumbing of ``quantize_and_sync_model_params_from_main_params``.

Chained optimizers (dense + expert, or a LayerWise/DistOpt pair) commonly own the *same*
DDP model chunk, and under ``--reuse-grad-buf-for-mxfp8-param-ag`` the param buffer aliases
the grad buffer. So the per-chunk work has to happen exactly once and in a fixed order:
zero every shared buffer, then let each optimizer stage its own shards into it, then gather
once. Getting that wrong corrupts weights silently -- a second zero wipes the shards a
sibling optimizer already staged -- which is why it is worth pinning without a GPU.
"""

from types import SimpleNamespace

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


def test_shared_model_chunk_is_zeroed_and_gathered_once():
    """Two optimizers, one shared chunk: one zero and one gather, not two of each."""
    optimizer, log = _build(reuse_grad_buf=True)

    optimizer.quantize_and_sync_model_params_from_main_params()

    assert log.count(('zero', 'chunk_a')) == 1, log
    assert log.count(('sync', 'chunk_a', True)) == 1, log
    assert [entry for entry in log if entry[0] == 'stage'] == [
        ('stage', 'dense'),
        ('stage', 'expert'),
    ], log


def test_buffer_is_zeroed_before_any_optimizer_stages_into_it():
    """A zero after a stage would wipe shards the sibling optimizer already wrote."""
    optimizer, log = _build(reuse_grad_buf=True)

    optimizer.quantize_and_sync_model_params_from_main_params()

    order = [entry[0] for entry in log]
    assert order.index('zero') < order.index('stage'), log
    assert max(i for i, k in enumerate(order) if k == 'stage') < order.index('sync'), log


def test_distinct_model_chunks_are_each_zeroed_and_gathered():
    optimizer, log = _build(reuse_grad_buf=True, share_chunk=False)

    optimizer.quantize_and_sync_model_params_from_main_params()

    for name in ('chunk_a', 'chunk_b'):
        assert log.count(('zero', name)) == 1, log
        assert log.count(('sync', name, True)) == 1, log


def test_grad_buffer_is_not_zeroed_when_the_buffer_is_not_reused():
    """Without the shared param/grad buffer there is nothing to zero, only stage and gather."""
    optimizer, log = _build(reuse_grad_buf=False)

    optimizer.quantize_and_sync_model_params_from_main_params()

    assert not [entry for entry in log if entry[0] == 'zero'], log
    assert log.count(('sync', 'chunk_a', True)) == 1, log


def test_stub_optimizers_contribute_no_chunks_and_stage_nothing():
    optimizer, log = _build(reuse_grad_buf=True, with_stub=True)

    optimizer.quantize_and_sync_model_params_from_main_params()

    assert not [entry for entry in log if entry[1] == 'stub_chunk'], log
    assert ('stage', 'stub') not in log, log


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
    assert log.count(('zero', 'chunk')) == 1, log
    assert log.count(('sync', 'chunk', True)) == 1, log


def test_an_empty_chain_refreshes_without_touching_config():
    """A rank with no trainable parameters gets ChainedOptimizer([]).

    __init__ takes the else branch there and never assigns self.config, so reading
    self.config.reuse_grad_buf_for_mxfp8_param_ag raises AttributeError rather than
    returning a default. Loading a quantized checkpoint on such a rank must not crash.
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
