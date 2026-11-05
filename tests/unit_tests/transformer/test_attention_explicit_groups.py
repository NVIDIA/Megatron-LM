# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""SelfAttention takes its tensor- and data-parallel groups from its process-group collection.

The distributed tests build the collection from a TP=2 grid whose data-parallel axis varies
fastest, while the global grid has TP=1: every rank is alone in its global tensor-parallel group
and shares one global data-parallel group with all other ranks. Neither global group matches the
module's groups, so reading the global grid compares or slices across the wrong ranks.

The remaining tests build the module on CPU with stand-in groups and a recording
``torch.distributed.all_gather``, so they need neither GPUs nor an initialized process group.
"""

import contextlib
from unittest import mock

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core import process_groups_config
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import attention as attention_module
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

TP_SIZE = 2

# The global-grid accessors that SelfAttention's consistency check and output gate can read.
_GLOBAL_ACCESSORS = (
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_tensor_model_parallel_group",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
)


@contextlib.contextmanager
def _forbid_global_groups():
    """Make the accessors raise, on parallel_state and wherever attention.py bound them by name."""

    def forbidden(name):
        def read_global_grid(*args, **kwargs):
            raise AssertionError(f"parallel_state.{name}() reads the global parallel grid")

        return read_global_grid

    with contextlib.ExitStack() as stack:
        for name in _GLOBAL_ACCESSORS:
            for module in (parallel_state, attention_module):
                if name in vars(module):
                    stack.enter_context(mock.patch.object(module, name, forbidden(name)))
        yield


class _StandInGroup:
    """A process group as SelfAttention sees it: a rank and a size, no communicator."""

    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


@pytest.fixture
def fresh_fallback_warnings(monkeypatch):
    """Each test observes the first global-group fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@pytest.fixture
def global_grid():
    """A TP=1 global grid: one global data-parallel group over all ranks."""
    if Utils.world_size < 2 * TP_SIZE or Utils.world_size % TP_SIZE:
        pytest.skip(f"needs a multiple of {TP_SIZE} ranks, at least {2 * TP_SIZE}")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


@pytest.fixture
def module_groups(global_grid):
    """The module's collection: TP=2, with the data-parallel axis varying fastest."""
    grid = HyperCommGrid([Utils.world_size // TP_SIZE, TP_SIZE, 1], ["dp", "tp", "cp"])
    groups = {"tp": grid.create_pg("tp"), "cp": grid.create_pg("cp"), "dp": grid.create_pg("dp")}
    yield ProcessGroupCollection(
        tp=groups["tp"], cp=groups["cp"], dp_gtp_remat=groups["dp"], gtp_remat=None
    )
    # Every rank finishes its collectives before any communicator goes away.
    torch.cuda.synchronize()
    torch.distributed.barrier()
    for group in groups.values():
        torch.distributed.destroy_process_group(group)


def _build_attention(
    pg_collection, *, tensor_model_parallel_size=TP_SIZE, on_gpu=True, **config_kwargs
):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=tensor_model_parallel_size,
        use_cpu_initialization=not on_gpu,
        **config_kwargs,
    )
    submodules = get_gpt_layer_local_submodules(
        qk_layernorm=config.qk_layernorm
    ).self_attention.submodules
    attention = SelfAttention(
        config,
        submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )
    return attention.cuda() if on_gpu else attention


def test_realtime_tests_use_module_groups(module_groups):
    attention = _build_attention(module_groups, qk_layernorm=True, test_mode=True)

    with _forbid_global_groups():
        attention.run_realtime_tests()


def test_realtime_tests_compare_within_module_groups(module_groups):
    """A mismatch across the module's TP group is reported as a TP discrepancy on every rank.

    Ranks at the module's TP rank 1 get different Q layernorm weights. Each module DP group holds
    ranks of a single TP rank, so its comparison passes, and every module TP group pairs both
    weights. The global DP group spans all ranks, so a check over it fails as a DP discrepancy.
    """
    attention = _build_attention(module_groups, qk_layernorm=True, test_mode=True)
    if module_groups.tp.rank() == 1:
        attention.q_layernorm.weight.data.add_(1.0)

    with pytest.raises(AssertionError, match="q_w in TP ranks"):
        attention.run_realtime_tests()


def test_realtime_tests_gather_over_module_groups():
    """The check gathers over the collection's dp_gtp_remat group, then over its TP group.

    The collection also sets dp, the replicate group without GTP-remat peers. The check must not
    use it: the layernorm parameters are replicated across those peers too, and dp_gtp_remat, like
    get_data_parallel_group(), includes them.
    """
    tp_group = _StandInGroup(rank=1, size=TP_SIZE)
    dp_group = _StandInGroup(rank=0, size=3)
    pg_collection = ProcessGroupCollection(
        tp=tp_group, cp=_StandInGroup(), dp=_StandInGroup(), dp_gtp_remat=dp_group, gtp_remat=None
    )
    attention = _build_attention(pg_collection, on_gpu=False, qk_layernorm=True, test_mode=True)
    gathers = []

    def all_gather(tensor_list, tensor, group=None, async_op=False):
        gathers.append((group, len(tensor_list)))
        for slot in tensor_list:
            slot.copy_(tensor)

    with _forbid_global_groups(), mock.patch.object(torch.distributed, "all_gather", all_gather):
        attention.run_realtime_tests()

    assert len(gathers) == 2
    assert gathers[0][0] is dp_group and gathers[0][1] == dp_group.size()
    assert gathers[1][0] is tp_group and gathers[1][1] == tp_group.size()


@pytest.mark.parametrize("dp_gtp_remat", ["unset", None])
def test_realtime_tests_require_data_parallel_group(dp_gtp_remat):
    """Without dp_gtp_remat the check raises before any collective could run over WORLD.

    The collection sets dp, which does not stand in for dp_gtp_remat.
    """
    groups = {
        "tp": _StandInGroup(),
        "cp": _StandInGroup(),
        "dp": _StandInGroup(),
        "gtp_remat": None,
    }
    if dp_gtp_remat != "unset":
        groups["dp_gtp_remat"] = dp_gtp_remat
    attention = _build_attention(
        ProcessGroupCollection(**groups),
        tensor_model_parallel_size=1,
        on_gpu=False,
        qk_layernorm=True,
        test_mode=True,
    )

    with mock.patch.object(torch.distributed, "all_gather") as all_gather:
        with pytest.raises(ValueError, match="dp_gtp_remat"):
            attention.run_realtime_tests()
    all_gather.assert_not_called()


def test_realtime_tests_require_tensor_parallel_group():
    """Without a TP group the check raises before any collective could run over WORLD."""
    pg_collection = ProcessGroupCollection(
        tp=_StandInGroup(), cp=_StandInGroup(), dp_gtp_remat=_StandInGroup(), gtp_remat=None
    )
    attention = _build_attention(
        pg_collection, tensor_model_parallel_size=1, on_gpu=False, qk_layernorm=True, test_mode=True
    )
    attention.tp_group = None

    with mock.patch.object(torch.distributed, "all_gather") as all_gather:
        with pytest.raises(ValueError, match="must set tp and dp_gtp_remat"):
            attention.run_realtime_tests()
    all_gather.assert_not_called()


@pytest.mark.parametrize("test_mode", [False, True])
@pytest.mark.parametrize("qk_layernorm", [False, True])
def test_default_collection_resolves_dp_gtp_remat_for_realtime_tests(
    fresh_fallback_warnings, test_mode, qk_layernorm
):
    """Only a model that runs the Q/K layernorm check resolves the global dp_gtp_remat group."""
    pg_collection = ProcessGroupCollection(tp=_StandInGroup(), cp=_StandInGroup(), gtp_remat=None)
    with mock.patch.object(
        ProcessGroupCollection, "use_mpu_process_groups", return_value=pg_collection
    ) as use_mpu_process_groups:
        with pytest.warns(FutureWarning, match="SelfAttention was called without `pg_collection`"):
            _build_attention(
                None,
                tensor_model_parallel_size=1,
                on_gpu=False,
                qk_layernorm=qk_layernorm,
                test_mode=test_mode,
            )

    use_mpu_process_groups.assert_called_once()
    required_pgs = use_mpu_process_groups.call_args.kwargs["required_pgs"]
    assert ("dp_gtp_remat" in required_pgs) == (test_mode and qk_layernorm)


def test_realtime_tests_default_collection_has_data_parallel_group(
    global_grid, fresh_fallback_warnings
):
    with pytest.warns(FutureWarning, match="SelfAttention was called without `pg_collection`"):
        attention = _build_attention(
            None, tensor_model_parallel_size=1, qk_layernorm=True, test_mode=True
        )

    assert attention.pg_collection.dp_gtp_remat is parallel_state.get_data_parallel_group()
    attention.run_realtime_tests()


def test_output_gate_uses_module_tensor_parallel_rank(module_groups):
    """The gate heads a rank keeps are the heads of its query, at its module TP rank.

    With one query group, each TP rank gathers the projection of the whole group and keeps the
    query and gate heads of its TP rank. The gate rows of the projection weight repeat the query
    rows, so the gate equals the query only when both are sliced at the same rank.
    """
    attention = _build_attention(
        module_groups, num_query_groups=1, attention_output_gate=True, add_bias_linear=False
    )
    config = attention.config
    query_rows = config.num_attention_heads * config.kv_channels
    generator = torch.Generator().manual_seed(1234)
    # The projection of one query group is laid out as [query, gate, key, value].
    weight = torch.randn(attention.linear_qkv_out_dim, config.hidden_size, generator=generator)
    weight = weight / config.hidden_size**0.5
    weight[query_rows : 2 * query_rows] = weight[:query_rows]
    local_rows = attention.linear_qkv.weight.shape[0]
    tp_rank = module_groups.tp.rank()
    attention.linear_qkv.weight.data.copy_(
        weight[tp_rank * local_rows : (tp_rank + 1) * local_rows]
    )
    hidden_states = torch.randn(8, 2, config.hidden_size, generator=generator).cuda()

    with torch.no_grad(), _forbid_global_groups():
        query, _, _, gate = attention.get_query_key_value_tensors(hidden_states, output_gate=True)

    assert query.shape[2] == config.num_attention_heads // TP_SIZE
    torch.testing.assert_close(gate, query, rtol=1e-2, atol=1e-2)
