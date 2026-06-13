# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for GTP + distributed checkpointing.

Verifies that ``make_sharded_tensors_for_checkpoint_with_gtp`` emits
ShardedTensor offsets that correctly encode TP × GTP sharding, and that
the helper is a no-op (delegates to vanilla) when no ``GTPShardedParam``
is present in the input state_dict.

"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.experimental.gtp import (
    GTP_CONFIG,
    GTPShardedParam,
    HAVE_GTP,
    make_sharded_tensors_for_checkpoint_with_gtp,
    reset_gtp_quantize_cache,
    update_gtp_config,
    wrap_module_params_gtp,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def _no_pad_alignment():
    """Disable GTP padding for the duration of each test so local shard sizes
    are exactly ``per_tp_out / gtp_size`` and the test math stays simple.
    DCP semantics with padding are exercised by the integration tests.
    """
    orig = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=0)
    yield
    update_gtp_config(pad_for_alignment=orig)


pytestmark = pytest.mark.skipif(not HAVE_GTP, reason="GTP requires TE with hook registry")


@pytest.fixture(scope="module", autouse=True)
def _torchrun_dist_init():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _require_world_size(n):
    if dist.get_world_size() != n:
        pytest.skip(
            f"Requires world_size={n}, got {dist.get_world_size()} "
            f"(launch with torchrun --nproc-per-node={n})"
        )


def _make_gtp_shard(out_features, in_features, gtp_group, dtype=torch.bfloat16):
    """Build a small GTPShardedParam by wrapping a one-param dummy module."""

    class _Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(out_features * in_features, dtype=dtype, device="cuda").reshape(
                    out_features, in_features
                )
            )

    mod = _Dummy()
    wrap_module_params_gtp(mod, ["weight"], gtp_group)
    return mod.weight  # now a GTPShardedParam


def _worker_helper_offsets_tp_eq_gtp_axis(rank, world_size, port):
    """TP=2, GTP=2 (4 ranks total). Weight is GTPShardedParam.

    Production flow: Mcore TE constructs the Linear with already-TP-sliced
    out_features (i.e. full / tp_size). GTP then slices that further by
    gtp_size. We mimic that by starting with a per-TP-rank tensor of size
    ``full // tp_size`` and letting wrap_module_params_gtp slice it.
    """
    gtp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])
    tp_group = dist.new_group([0, 2]) if rank in (0, 2) else dist.new_group([1, 3])

    full_out_features = 8
    tp_size, gtp_size = 2, 2
    per_tp_out = full_out_features // tp_size  # 4
    per_shard_out = per_tp_out // gtp_size  # 2
    in_features = 4

    weight = _make_gtp_shard(per_tp_out, in_features, gtp_group)
    assert weight.shape == (per_shard_out, in_features), (
        f"rank={rank} local shard shape {tuple(weight.shape)} != "
        f"({per_shard_out}, {in_features})"
    )

    sharded = make_sharded_tensors_for_checkpoint_with_gtp(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"

    # Composite offset: (axis=0, tp_rank*gtp_size+gtp_rank, tp_size*gtp_size)
    # rank → (tp_rank, gtp_rank): 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1)
    tp_rank = rank // 2
    gtp_rank = rank % 2
    expected_offset = (tp_rank * gtp_size + gtp_rank) * per_shard_out
    assert st.global_offset[0] == expected_offset, (
        f"rank={rank} expected axis-0 offset {expected_offset}, got {st.global_offset[0]}"
    )
    assert st.global_shape[0] == full_out_features, (
        f"rank={rank} expected global axis-0 size {full_out_features}, got {st.global_shape[0]}"
    )


def _worker_helper_offsets_tp_neq_gtp_axis(rank, world_size, port):
    """Row-parallel: TP=2 shards axis 1, GTP=2 shards axis 0.

    Per-TP-rank tensor: (full_out, full_in/tp_size). GTP further shards
    axis 0 to (full_out/gtp_size, full_in/tp_size).
    """
    gtp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])
    tp_group = dist.new_group([0, 2]) if rank in (0, 2) else dist.new_group([1, 3])

    full_out, full_in = 8, 4
    tp_size, gtp_size = 2, 2
    per_tp_in = full_in // tp_size  # 2
    per_shard_out = full_out // gtp_size  # 4

    weight = _make_gtp_shard(full_out, per_tp_in, gtp_group)
    assert weight.shape == (per_shard_out, per_tp_in)

    sharded = make_sharded_tensors_for_checkpoint_with_gtp(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 1},  # row-parallel
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    tp_rank = rank // 2
    gtp_rank = rank % 2
    assert st.global_offset[0] == gtp_rank * per_shard_out, (
        f"rank={rank} axis-0 offset wrong: {st.global_offset[0]}"
    )
    assert st.global_offset[1] == tp_rank * per_tp_in, (
        f"rank={rank} axis-1 offset wrong: {st.global_offset[1]}"
    )
    assert st.global_shape == (full_out, full_in), (
        f"rank={rank} global shape {st.global_shape} != ({full_out}, {full_in})"
    )


def _worker_helper_no_op_no_gtp(rank, world_size, port):
    """Helper must delegate to vanilla when state_dict has no GTPShardedParam.

    Per-TP-rank shape under column-parallel TP=2: (full_out//tp_size, in).
    """
    tp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])

    full_out, in_features, tp_size = 8, 4, 2
    per_tp_out = full_out // tp_size

    plain = torch.nn.Parameter(
        torch.zeros(per_tp_out, in_features, dtype=torch.bfloat16, device="cuda")
    )
    bias = torch.nn.Parameter(torch.zeros(per_tp_out, dtype=torch.bfloat16, device="cuda"))

    sharded = make_sharded_tensors_for_checkpoint_with_gtp(
        {"weight": plain, "bias": bias},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0, "bias": 0},
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    # tp_group is [0,1] for ranks 0,1 and [2,3] for ranks 2,3 here — local tp_rank = rank % 2
    tp_rank = rank % 2
    assert sharded["weight"].global_offset[0] == tp_rank * per_tp_out, (
        f"rank={rank} fallback path produced wrong offset for weight: "
        f"{sharded['weight'].global_offset[0]}"
    )
    assert sharded["weight"].global_shape == (full_out, in_features)


def _worker_helper_padded_inproj_no_pad_case(rank, world_size, port):
    """``in_proj.weight`` shape modeled after the production case (z|x|B|C|dt
    concat along dim 0). With GTP=4 and these dim-0 sizes the alignment
    constraint ``dim0 % (gtp_size * pad_for_alignment) == 0`` is satisfied —
    *no* padding fires. Verify the helper emits the expected offsets.
    """
    update_gtp_config(pad_for_alignment=16)
    # dim0 = 512+512+64+64+8 = 1160 → 1160 % (4*16=64) = 8 ⇒ NOT aligned.
    # Pick sizes that ARE aligned to 64 to exercise the no-pad path:
    dim0 = 1152  # = 18 * 64; alignment-clean for gtp_size=4, pad=16
    in_features = 4

    # All 4 ranks form a single GTP group.
    gtp_group = dist.new_group(list(range(world_size)))
    weight = _make_gtp_shard(dim0, in_features, gtp_group)

    # No padding ⇒ local shape is exactly dim0 / 4 = 288
    expected_local = dim0 // 4
    assert weight.shape == (expected_local, in_features), (
        f"rank={rank}: padding should NOT have fired (dim0 aligned); "
        f"got local shape {tuple(weight.shape)}, expected ({expected_local}, {in_features})"
    )
    assert getattr(weight, "pad_length", 0) == 0

    sharded = make_sharded_tensors_for_checkpoint_with_gtp(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=dist.new_group([rank]),  # trivial 1-rank TP group
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    assert st.global_shape[0] == dim0, (
        f"rank={rank} no-pad case: global_shape[0] {st.global_shape[0]} != {dim0}"
    )
    assert st.global_offset[0] == rank * expected_local


def _worker_helper_padded_inproj_pad_case(rank, world_size, port):
    """Same in_proj layout but with a dim-0 size that requires GTP padding.

    z=512, x=512, B=64, C=64, dt=8 → dim0=1160. With gtp_size=4 and
    pad_for_alignment=16, alignment block = 64; 1160 % 64 = 8 so 56 pad
    rows are appended. Padded dim0 = 1216, per-rank shard = 304 (uniform
    across all 4 ranks; the pad rows live at the tail of rank-3's slice).

    The helper today saves the *padded* global shape (1216) — round-trip is
    correct under save_gtp_size == load_gtp_size. This test pins that
    behaviour and serves as a regression for the future "unpadded global"
    fix.
    """
    update_gtp_config(pad_for_alignment=16)
    dim0_unpadded = 1160  # z(512) + x(512) + B(64) + C(64) + dt(8)
    in_features = 4
    gtp_size = world_size
    alignment_block = 16 * gtp_size  # = 64
    pad = (alignment_block - dim0_unpadded % alignment_block) % alignment_block
    dim0_padded = dim0_unpadded + pad
    per_shard = dim0_padded // gtp_size

    gtp_group = dist.new_group(list(range(world_size)))
    weight = _make_gtp_shard(dim0_unpadded, in_features, gtp_group)

    assert weight.shape == (per_shard, in_features), (
        f"rank={rank}: post-pad shard shape {tuple(weight.shape)} != ({per_shard}, {in_features})"
    )
    # Only rank-3 (the last GTP rank) carries the trailing pad rows; all ranks
    # report the same pad_length (an invariant set by _gtp_slice_one_param).
    assert getattr(weight, "pad_length", 0) == pad, (
        f"rank={rank}: pad_length {getattr(weight, 'pad_length', 0)} != {pad}"
    )

    sharded = make_sharded_tensors_for_checkpoint_with_gtp(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=dist.new_group([rank]),
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    # Helper saves the padded global. ``allow_shape_mismatch=True`` is what
    # makes the saved tensor portable to a different load-time GTP topology
    # (different alignment choice yields a different padded size).
    assert st.global_shape[0] == dim0_padded, (
        f"rank={rank} pad case: global_shape[0] {st.global_shape[0]} != {dim0_padded}"
    )
    assert st.global_offset[0] == rank * per_shard
    assert st.allow_shape_mismatch is True, (
        f"rank={rank} pad case: allow_shape_mismatch must be True when GTP padding fires; "
        f"otherwise the ckpt cannot be loaded at a different GTP topology."
    )


def _worker_helper_cross_topology_reshard_metadata(rank, world_size, port):
    """Pin the cross-topology reshard contract via ShardedTensor metadata.

    We can't run a real DCP save/load against itself within a single torchrun
    (need separate worlds), but we can verify the saved ShardedTensor carries
    everything DCP needs to do the reshard: ``allow_shape_mismatch=True`` and
    a global_shape large enough to cover any compatible load-side topology
    (≥ unpadded original).
    """
    update_gtp_config(pad_for_alignment=16)
    dim0_unpadded = 1160
    in_features = 4
    gtp_size = world_size
    alignment_block = 16 * gtp_size  # 64
    dim0_padded = (
        dim0_unpadded + (alignment_block - dim0_unpadded % alignment_block) % alignment_block
    )
    per_shard = dim0_padded // gtp_size

    gtp_group = dist.new_group(list(range(world_size)))
    weight = _make_gtp_shard(dim0_unpadded, in_features, gtp_group)

    sharded = make_sharded_tensors_for_checkpoint_with_gtp(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=dist.new_group([rank]),
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    # 1. The saved global covers >= unpadded original size.
    assert st.global_shape[0] >= dim0_unpadded, (
        f"rank={rank} saved global_shape ({st.global_shape[0]}) < unpadded ({dim0_unpadded}); "
        f"would lose valid data on cross-topology reshard."
    )
    # 2. ``allow_shape_mismatch=True`` lets DCP tolerate that the load-side
    #    padded size may differ.
    assert st.allow_shape_mismatch is True
    # 3. Each rank's offset+local_shape covers a contiguous slice of the
    #    padded global; together the ranks cover [0, padded_global).
    assert st.global_offset[0] + st.local_shape[0] <= st.global_shape[0]
    assert st.global_offset[0] + st.local_shape[0] == (rank + 1) * per_shard


def _worker_save_then_load_offsets_symmetric(rank, world_size, port):
    """Save-side and load-side ShardedTensors must produce identical offsets
    and global_shape so DCP can correctly resharded between them.

    We don't run the real DCP save (avoids filesystem / async-writer issues
    in CI); we just verify the symmetry property the load path relies on.
    """
    update_gtp_config(pad_for_alignment=0)
    dim0 = 16
    in_features = 4
    gtp_group = dist.new_group(list(range(world_size)))

    def _build(prefix):
        weight = _make_gtp_shard(dim0, in_features, gtp_group)
        return make_sharded_tensors_for_checkpoint_with_gtp(
            {"weight": weight},
            prefix=prefix,
            tensor_parallel_layers_axis_map={"weight": 0},
            sharded_offsets=(),
            tp_group=dist.new_group([rank]),
            dp_cp_group=dist.new_group(list(range(world_size))),
        )["layer.weight"]

    save_st = _build("layer.")
    load_st = _build("layer.")
    assert save_st.global_shape == load_st.global_shape
    assert save_st.global_offset == load_st.global_offset
    assert save_st.local_shape == load_st.local_shape
    assert save_st.replica_id == load_st.replica_id


def _worker_reset_quantize_cache(rank, world_size, port):
    """`reset_gtp_quantize_cache` must flip did_cast_to_low_precision back to False."""
    gtp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])

    class _Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(4, 4, dtype=torch.bfloat16, device="cuda")
            )

    mod = _Dummy()
    wrap_module_params_gtp(mod, ["weight"], gtp_group)
    p = mod.weight
    p.did_cast_to_low_precision = True

    reset_gtp_quantize_cache(mod)
    assert p.did_cast_to_low_precision is False


def _worker_helper_offsets_ep_egtp(rank, world_size, port):
    """EP=2, EGTP=2 (4 ranks): routed-expert weight.

    Mirrors ``TEGroupedLinear.sharded_state_dict``: expert parallelism prepends a
    global-expert axis through ``sharded_offsets``, and EGTP shards each expert's
    ``out_features`` (axis 0). The GTP-aware checkpoint helper layers the EGTP
    axis-0 split on top of the prepended expert offset.

    rank → (ep_rank, egtp_rank): 0→(0,0) 1→(0,1) 2→(1,0) 3→(1,1).
    """
    egtp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])

    ep_size, egtp_size, num_gemms = 2, 2, 1
    ep_rank = rank // 2
    egtp_rank = rank % 2
    per_expert_out = 4
    per_shard_out = per_expert_out // egtp_size  # 2
    in_features = 4
    num_global_experts = ep_size * num_gemms  # 2
    global_expert_idx = ep_rank * num_gemms  # + gemm_idx (0)

    weight = _make_gtp_shard(per_expert_out, in_features, egtp_group)
    assert weight.shape == (per_shard_out, in_features), (
        f"rank={rank} EGTP shard shape {tuple(weight.shape)} != ({per_shard_out}, {in_features})"
    )

    sharded = make_sharded_tensors_for_checkpoint_with_gtp(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        # EP prepends the global-expert axis; EGTP shards out_features below it.
        sharded_offsets=((0, global_expert_idx, num_global_experts),),
        tp_group=dist.new_group([rank]),  # no TP in this case
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"
    # global shape = (num_global_experts, full_out_features, in_features)
    assert st.global_shape == (num_global_experts, per_expert_out, in_features), (
        f"rank={rank} global_shape {st.global_shape} != "
        f"({num_global_experts}, {per_expert_out}, {in_features})"
    )
    # Prepended expert axis (axis 0): offset == this rank's global expert index.
    assert st.global_offset[0] == global_expert_idx, (
        f"rank={rank} expert-axis offset {st.global_offset[0]} != {global_expert_idx}"
    )
    # EGTP axis (weight axis 0, shifted to global axis 1): offset == egtp_rank · per_shard.
    assert st.global_offset[1] == egtp_rank * per_shard_out, (
        f"rank={rank} EGTP axis-1 offset {st.global_offset[1]} != {egtp_rank * per_shard_out}"
    )


def _worker_helper_embedding_offsets(rank, world_size, port):
    """Embedding / output_layer path: ``VocabParallelEmbedding.sharded_state_dict`` calls
    ``make_tp_sharded_tensor_for_checkpoint`` DIRECTLY (it needs allow_shape_mismatch for
    vocab padding), bypassing the GTP-aware wrapper. So that helper itself must layer the
    GTP axis-0 split. TP=2, GTP=2, tp_axis=0 (vocab) → composite axis-0 offset, same as the
    column-parallel case.
    """
    from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

    gtp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])
    tp_group = dist.new_group([0, 2]) if rank in (0, 2) else dist.new_group([1, 3])

    full_vocab, hidden = 8, 4
    tp_size, gtp_size = 2, 2
    per_tp = full_vocab // tp_size  # 4
    per_shard = per_tp // gtp_size  # 2

    weight = _make_gtp_shard(per_tp, hidden, gtp_group)
    assert weight.shape == (per_shard, hidden)

    st = make_tp_sharded_tensor_for_checkpoint(
        tensor=weight,
        key="embedding.word_embeddings.weight",
        tp_axis=0,
        allow_shape_mismatch=True,  # how VocabParallelEmbedding calls it
        prepend_offsets=(),
        tp_group=tp_group,
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"
    tp_rank = rank // 2
    gtp_rank = rank % 2
    expected_offset = (tp_rank * gtp_size + gtp_rank) * per_shard
    assert st.global_offset[0] == expected_offset, (
        f"rank={rank} embedding axis-0 offset {st.global_offset[0]} != {expected_offset}"
    )
    assert st.global_shape[0] == full_vocab, (
        f"rank={rank} embedding global axis-0 {st.global_shape[0]} != {full_vocab}"
    )


def _worker_helper_public_wrapper_delegates(rank, world_size, port):
    """The public ``make_sharded_tensors_for_checkpoint`` (the entry point most layers call,
    e.g. ColumnParallelLinear / output_layer) must detect a GTPShardedParam and produce the
    GTP-composite offset — i.e. it delegates to the GTP-aware path rather than the vanilla
    TP-only one. TP=2, GTP=2, column-parallel (tp_axis=0).
    """
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    gtp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])
    tp_group = dist.new_group([0, 2]) if rank in (0, 2) else dist.new_group([1, 3])

    full_out, in_features = 8, 4
    tp_size, gtp_size = 2, 2
    per_tp_out = full_out // tp_size  # 4
    per_shard_out = per_tp_out // gtp_size  # 2

    weight = _make_gtp_shard(per_tp_out, in_features, gtp_group)

    sharded = make_sharded_tensors_for_checkpoint(
        {"weight": weight},
        prefix="layer.",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=dist.new_group(list(range(world_size))),
    )
    st = sharded["layer.weight"]
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"
    tp_rank = rank // 2
    gtp_rank = rank % 2
    expected_offset = (tp_rank * gtp_size + gtp_rank) * per_shard_out
    assert st.global_offset[0] == expected_offset, (
        f"rank={rank} public wrapper did not produce the GTP-composite offset: "
        f"{st.global_offset[0]} != {expected_offset} (delegation to the GTP path failed?)"
    )
    assert st.global_shape[0] == full_out, (
        f"rank={rank} global axis-0 {st.global_shape[0]} != {full_out}"
    )


def _worker_helper_replicated_sink_rejects_gtp(rank, world_size, port):
    """Sanity guard: a GTPShardedParam must NEVER be saved via the replicated
    make_sharded_tensor_for_checkpoint (it would record a shard-sized global shape).
    The helper asserts; this pins that behaviour.
    """
    from megatron.core.utils import make_sharded_tensor_for_checkpoint

    gtp_group = dist.new_group([0, 1]) if rank in (0, 1) else dist.new_group([2, 3])
    weight = _make_gtp_shard(4, 4, gtp_group)
    with pytest.raises(AssertionError):
        make_sharded_tensor_for_checkpoint(
            weight,
            "weight",
            tp_group=dist.new_group([rank]),
            dp_cp_group=dist.new_group(list(range(world_size))),
        )


# ---------------------------------------------------------------------------
# Test class wrappers (4-GPU)
# ---------------------------------------------------------------------------


@pytest.mark.run_only_on_devices_with_compute_capability(compute_capability=(10, 0))
class TestGtpDcpHelper:
    def test_composite_offset_same_axis(self):
        _require_world_size(4)
        _worker_helper_offsets_tp_eq_gtp_axis(dist.get_rank(), 4, None)

    def test_dual_offsets_cross_axis(self):
        _require_world_size(4)
        _worker_helper_offsets_tp_neq_gtp_axis(dist.get_rank(), 4, None)

    def test_ep_egtp_offsets(self):
        _require_world_size(4)
        _worker_helper_offsets_ep_egtp(dist.get_rank(), 4, None)

    def test_embedding_offsets(self):
        _require_world_size(4)
        _worker_helper_embedding_offsets(dist.get_rank(), 4, None)

    def test_public_wrapper_delegates(self):
        _require_world_size(4)
        _worker_helper_public_wrapper_delegates(dist.get_rank(), 4, None)

    def test_replicated_sink_rejects_gtp(self):
        _require_world_size(4)
        _worker_helper_replicated_sink_rejects_gtp(dist.get_rank(), 4, None)

    def test_no_op_no_gtp(self):
        _require_world_size(4)
        _worker_helper_no_op_no_gtp(dist.get_rank(), 4, None)

    def test_reset_quantize_cache(self):
        _require_world_size(4)
        _worker_reset_quantize_cache(dist.get_rank(), 4, None)

    def test_inproj_no_pad(self):
        _require_world_size(4)
        _worker_helper_padded_inproj_no_pad_case(dist.get_rank(), 4, None)

    def test_inproj_with_pad(self):
        _require_world_size(4)
        _worker_helper_padded_inproj_pad_case(dist.get_rank(), 4, None)

    def test_cross_topology_reshard_metadata(self):
        _require_world_size(4)
        _worker_helper_cross_topology_reshard_metadata(dist.get_rank(), 4, None)

    def test_save_then_load_offsets_symmetric(self):
        _require_world_size(4)
        _worker_save_then_load_offsets_symmetric(dist.get_rank(), 4, None)
