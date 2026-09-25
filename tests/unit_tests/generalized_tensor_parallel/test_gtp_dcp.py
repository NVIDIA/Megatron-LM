# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for GTP_remat + distributed checkpointing.

Verifies that ``make_sharded_tensors_for_checkpoint_with_gtp_remat`` emits
ShardedTensor offsets that correctly encode TP × GTP_remat sharding, and that
the helper is a no-op (delegates to vanilla) when no ``GTPShardedParam``
is present in the input state_dict.

"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TE with hook registry", allow_module_level=True)

import transformer_engine.pytorch as te  # noqa: E402
from transformer_engine.common.recipe import MXFP8BlockScaling  # noqa: E402
from transformer_engine.pytorch import fp8_autocast, fp8_model_init  # noqa: E402

from megatron.core.dist_checkpointing.mapping import (  # noqa: E402
    ShardedObject,
    ShardedTensorFactory,
    is_main_replica,
)
from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fp8_utils import is_float8tensor  # noqa: E402
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add  # noqa: E402
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer  # noqa: E402
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.ssm.gated_delta_net import HAVE_FLA as HAVE_GDN_FLA  # noqa: E402
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules  # noqa: E402
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules  # noqa: E402
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules  # noqa: E402
from megatron.core.tensor_parallel.generalized_tensor_parallelism import (  # noqa: E402
    GTP_CONFIG,
    GTPShardedParam,
    make_sharded_tensors_for_checkpoint_with_gtp_remat,
    reset_gtp_state,
    update_gtp_config,
    wrap_module_params_gtp,
)
from megatron.core.tensor_parallel.gtp_api import (  # noqa: E402
    attach_gtp_to_presharded_module,
    dequantize_gtp_native_fp8,
    gtp_native_fp8_load_context,
    gtp_remat_shard_dim0,
    is_gtp_param,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.spec_utils import ModuleSpec  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint  # noqa: E402
from megatron.core.utils import (  # noqa: E402
    get_pg_rank,
    get_pg_size,
    make_tp_sharded_tensor_for_checkpoint,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: E402,F401
    _requires_mxfp8,
    _torchrun_dist_init,
)


@pytest.fixture(autouse=True)
def _no_pad_alignment():
    """Disable GTP_remat padding for the duration of each test so local shard sizes
    are exactly ``per_tp_out / gtp_remat_size`` and the test math stays simple.
    DCP semantics with padding are exercised by the integration tests.
    """
    orig = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=0)
    yield
    update_gtp_config(pad_for_alignment=orig)


def _require_world_size(n):
    if dist.get_world_size() != n:
        pytest.skip(
            f"Requires world_size={n}, got {dist.get_world_size()} "
            f"(launch with torchrun --nproc-per-node={n})"
        )


# Many workers need the same TP/GTP subgroups. Memoize by rank-set so the process holds a
# handful of communicators instead of re-creating (and leaking) one per worker.
_GROUP_CACHE = {}


def _cached_new_group(ranks):
    """Memoized ``dist.new_group`` keyed by rank-set (see note above)."""
    key = tuple(ranks)
    if key not in _GROUP_CACHE:
        _GROUP_CACHE[key] = dist.new_group(list(ranks))
    return _GROUP_CACHE[key]


@pytest.fixture(scope="module", autouse=True)
def _precreate_subgroups(_torchrun_dist_init):
    """Pre-create the shared TP/GTP subgroups once, on all ranks, in a fixed order.

    ``dist.new_group`` is a world-collective (all ranks must call it in the same order); the
    per-member ``new_group([0,1]) if rank in (0,1) else ...`` idiom collides disjoint groups on
    the call-order tag and hangs NCCL. Pre-creating makes every later ``_cached_new_group`` a hit.
    """
    if dist.is_initialized() and dist.get_world_size() == 4:
        for ranks in ([0], [1], [2], [3], [0, 1], [2, 3], [0, 2], [1, 3], [0, 1, 2, 3]):
            _cached_new_group(ranks)
    yield


def _make_gtp_shard(
    out_features, in_features, gtp_remat_group, dtype=torch.bfloat16, replica_group=None
):
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
    wrap_module_params_gtp(mod, ["weight"], gtp_remat_group, replica_group=replica_group)
    return mod.weight  # now a GTPShardedParam


def _make_native_fp8_gtp_shard(per_tp_out, in_f, gtp_remat_group, recipe):
    """Build a native-FP8 GTP weight the production way (extensions/transformer_engine.py):
    pass the pre-sharded out_features into a stock ``fp8_model_init`` ``te.Linear`` so TE inits a
    native MXFP8 shard, attach the GTP surface post-init, then run one FP8 forward to populate the
    rowwise/columnwise FP8 data. Returns the reclassed ``GTP_<Fp8Tensor>`` weight."""
    shard_out, pad = gtp_remat_shard_dim0(per_tp_out, gtp_remat_group)
    with fp8_model_init(enabled=True, recipe=recipe):
        lin = te.Linear(in_f, shard_out, bias=False, params_dtype=torch.bfloat16, device="cuda")
    lin.gtp_remat_size = gtp_remat_group.size()
    attach_gtp_to_presharded_module(lin, gtp_remat_group, pad)
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        _ = lin(torch.randn(32, in_f, dtype=torch.bfloat16, device="cuda"))
    return lin.weight


def _worker_native_fp8_dcp_save(rank, world_size, port):
    """Native-FP8 GTP weight: DCP save must emit a dequantized BF16 ShardedTensor with the full
    (TP x GTP_remat) global shape and correct composite axis-0 offset -- not raw FP8 bytes under
    a fake BF16 dtype (a55b save-crash guard: recognition gates / TE tex.dequantize miss the
    native-FP8 GTP_<Fp8Tensor> subclass; dequantize_gtp_native_fp8 restores the base class).
    """
    _requires_mxfp8()

    # TP=2, GTP_remat=2 (4 ranks). MXFP8 needs dims % 32, so use fp8-valid sizes.
    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    tp_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])
    full_out, in_f = 128, 128
    tp_size, gtp_remat_size = 2, 2
    per_tp_out = full_out // tp_size  # 64
    per_shard_out = per_tp_out // gtp_remat_size  # 32 (== MXFP8 block size)

    recipe = MXFP8BlockScaling()
    w = _make_native_fp8_gtp_shard(per_tp_out, in_f, gtp_remat_group, recipe)
    # The live weight is a native FP8 GTP param (a QuantizedTensor subclass), sharded.
    assert is_float8tensor(w), "weight should be a native FP8 tensor"
    assert getattr(w, "is_gtp_weight_remat", False), "GTP surface missing"
    assert type(w).__name__.startswith("GTP_"), type(w).__name__
    assert tuple(w.shape) == (per_shard_out, in_f)

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": w},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    assert isinstance(st, ShardedTensor), type(st)

    # Saved data must be dequantized BF16 — not raw FP8 bytes under a fake dtype.
    assert st.data.dtype == torch.bfloat16, f"expected bf16 saved data, got {st.data.dtype}"
    assert not is_float8tensor(st.data), "checkpoint data must be dequantized, not FP8"
    assert tuple(st.data.shape) == (per_shard_out, in_f)

    # Full (TP x GTP) global shape + composite axis-0 offset (not sharded-as-full).
    assert st.global_shape[0] == full_out, (st.global_shape, full_out)
    tp_rank, gtp_rank = rank // 2, rank % 2
    assert st.global_offset[0] == (tp_rank * gtp_remat_size + gtp_rank) * per_shard_out, (
        rank,
        st.global_offset,
    )

    # The live param must be untouched by the save (class restored, still native FP8).
    assert is_float8tensor(w) and type(w).__name__.startswith(
        "GTP_"
    ), "dequantize must not mutate the live param's class"


def _worker_native_fp8_dcp_load_copy(rank, world_size, port):
    """A DCP load copies a BF16 payload into a live native-FP8 GTP weight. That copy must work
    through ``gtp_native_fp8_load_context``, keep the GTP subclass, and leave the weight holding
    the loaded values.

    The context exists because TE's ``IsMXFP8Tensor`` compared the *exact* class and so rejected
    GTP's dynamic ``GTP_<Fp8Tensor>`` subclass. TE #3393 made that check subclass-tolerant, so on
    newer TE a bare copy succeeds as well — hence step (1) probes rather than requires a raise.
    """
    _requires_mxfp8()

    def is_gtp_fp8(param):
        return is_float8tensor(param) and type(param).__name__.startswith("GTP_")

    # Single-rank concern (the __class__ swap during copy_), so use the default WORLD group as the
    # gtp_remat_group rather than dist.new_group subgroups — the latter's secondary NCCL socket
    # bootstrap is flaky on some multi-node allocations and would mask the fp8 copy behavior.
    gtp_remat_group = dist.group.WORLD
    per_tp_out, in_f = 128, 128  # MXFP8 needs dims % 32; shard = 128/world(4) = 32
    recipe = MXFP8BlockScaling()

    shard_out, pad = gtp_remat_shard_dim0(per_tp_out, gtp_remat_group)
    with fp8_model_init(enabled=True, recipe=recipe):
        lin = te.Linear(in_f, shard_out, bias=False, params_dtype=torch.bfloat16, device="cuda")
    lin.gtp_remat_size = gtp_remat_group.size()
    attach_gtp_to_presharded_module(lin, gtp_remat_group, pad)
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        _ = lin(torch.randn(32, in_f, dtype=torch.bfloat16, device="cuda"))

    assert is_gtp_fp8(lin.weight), f"setup gave {type(lin.weight).__name__}, not a GTP FP8 weight"

    # The dequantized BF16 payload a DCP load would hand back for this shard.
    target_bf16 = torch.randn(shard_out, in_f, dtype=torch.bfloat16, device="cuda")

    # (1) Bare copy_, the way production's _load_from_state_dict does it. Raises on pre-#3393 TE
    # and succeeds after, so accept either — but copy a DISTINCT value, or a copy that lands would
    # pre-seed the target and make (2) pass trivially.
    probe_bf16 = torch.randn(shard_out, in_f, dtype=torch.bfloat16, device="cuda")
    try:
        with torch.no_grad():
            lin.weight.copy_(probe_bf16)
    except Exception as e:  # noqa: BLE001
        assert "MXFP8" in str(e) or "IsMXFP8Tensor" in str(e), str(e)
    assert is_gtp_fp8(lin.weight), "bare copy_ must leave the GTP subclass intact"

    # (2) Under the context the copy always lands, on every TE version.
    with torch.no_grad(), gtp_native_fp8_load_context(lin):
        lin.weight.copy_(target_bf16)
    assert is_gtp_fp8(lin.weight), "load context must reclass back to the GTP subclass"

    # MXFP8 round-trip is lossy; check the weight tracks the target, not the pre-copy values.
    loaded = dequantize_gtp_native_fp8(lin.weight)
    rel = (loaded - target_bf16).abs().max() / target_bf16.abs().max().clamp_min(1e-6)
    assert rel < 0.2, f"loaded weight does not match checkpoint values (max rel {rel:.3f})"


def _worker_helper_offsets_tp_eq_gtp_axis(rank, world_size, port):
    """TP=2, GTP_remat=2 (4 ranks total). Weight is GTPShardedParam.

    Production flow: Mcore TE constructs the Linear with already-TP-sliced
    out_features (i.e. full / tp_size). GTP_remat then slices that further by
    gtp_remat_size. We mimic that by starting with a per-TP-rank tensor of size
    ``full // tp_size`` and letting wrap_module_params_gtp slice it.
    """
    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    tp_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])

    full_out_features = 8
    tp_size, gtp_remat_size = 2, 2
    per_tp_out = full_out_features // tp_size  # 4
    per_shard_out = per_tp_out // gtp_remat_size  # 2
    in_features = 4

    weight = _make_gtp_shard(per_tp_out, in_features, gtp_remat_group)
    assert weight.shape == (per_shard_out, in_features), (
        f"rank={rank} local shard shape {tuple(weight.shape)} != "
        f"({per_shard_out}, {in_features})"
    )

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"

    # Composite offset: (axis=0, tp_rank*gtp_remat_size+gtp_rank, tp_size*gtp_remat_size)
    # rank → (tp_rank, gtp_rank): 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1)
    tp_rank = rank // 2
    gtp_rank = rank % 2
    expected_offset = (tp_rank * gtp_remat_size + gtp_rank) * per_shard_out
    assert (
        st.global_offset[0] == expected_offset
    ), f"rank={rank} expected axis-0 offset {expected_offset}, got {st.global_offset[0]}"
    assert (
        st.global_shape[0] == full_out_features
    ), f"rank={rank} expected global axis-0 size {full_out_features}, got {st.global_shape[0]}"


def _worker_helper_offsets_tp_neq_gtp_axis(rank, world_size, port):
    """Row-parallel: TP=2 shards axis 1, GTP_remat=2 shards axis 0.

    Per-TP-rank tensor: (full_out, full_in/tp_size). GTP_remat further shards
    axis 0 to (full_out/gtp_remat_size, full_in/tp_size).
    """
    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    tp_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])

    full_out, full_in = 8, 4
    tp_size, gtp_remat_size = 2, 2
    per_tp_in = full_in // tp_size  # 2
    per_shard_out = full_out // gtp_remat_size  # 4

    weight = _make_gtp_shard(full_out, per_tp_in, gtp_remat_group)
    assert weight.shape == (per_shard_out, per_tp_in)

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 1},  # row-parallel
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    tp_rank = rank // 2
    gtp_rank = rank % 2
    assert (
        st.global_offset[0] == gtp_rank * per_shard_out
    ), f"rank={rank} axis-0 offset wrong: {st.global_offset[0]}"
    assert (
        st.global_offset[1] == tp_rank * per_tp_in
    ), f"rank={rank} axis-1 offset wrong: {st.global_offset[1]}"
    assert st.global_shape == (
        full_out,
        full_in,
    ), f"rank={rank} global shape {st.global_shape} != ({full_out}, {full_in})"


def _worker_helper_no_op_no_gtp_remat(rank, world_size, port):
    """Helper must delegate to vanilla when state_dict has no GTPShardedParam.

    Per-TP-rank shape under column-parallel TP=2: (full_out//tp_size, in).
    """
    tp_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])

    full_out, in_features, tp_size = 8, 4, 2
    per_tp_out = full_out // tp_size

    plain = torch.nn.Parameter(
        torch.zeros(per_tp_out, in_features, dtype=torch.bfloat16, device="cuda")
    )
    bias = torch.nn.Parameter(torch.zeros(per_tp_out, dtype=torch.bfloat16, device="cuda"))

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": plain, "bias": bias},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0, "bias": 0},
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=_cached_new_group(list(range(world_size))),
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
    concat along dim 0). With GTP_remat=4 and these dim-0 sizes the alignment
    constraint ``dim0 % (gtp_remat_size * pad_for_alignment) == 0`` is satisfied —
    *no* padding fires. Verify the helper emits the expected offsets.
    """
    update_gtp_config(pad_for_alignment=16)
    # dim0 = 512+512+64+64+8 = 1160 → 1160 % (4*16=64) = 8 ⇒ NOT aligned.
    # Pick sizes that ARE aligned to 64 to exercise the no-pad path:
    dim0 = 1152  # = 18 * 64; alignment-clean for gtp_remat_size=4, pad=16
    in_features = 4

    # All 4 ranks form a single GTP_remat group.
    gtp_remat_group = _cached_new_group(list(range(world_size)))
    weight = _make_gtp_shard(dim0, in_features, gtp_remat_group)

    # No padding ⇒ local shape is exactly dim0 / 4 = 288
    expected_local = dim0 // 4
    assert weight.shape == (expected_local, in_features), (
        f"rank={rank}: padding should NOT have fired (dim0 aligned); "
        f"got local shape {tuple(weight.shape)}, expected ({expected_local}, {in_features})"
    )
    assert getattr(weight, "pad_length", 0) == 0

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=_cached_new_group([rank]),  # trivial 1-rank TP group
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    assert (
        st.global_shape[0] == dim0
    ), f"rank={rank} no-pad case: global_shape[0] {st.global_shape[0]} != {dim0}"
    assert st.global_offset[0] == rank * expected_local


def _worker_helper_padded_inproj_pad_case(rank, world_size, port):
    """in_proj with a dim-0 size needing GTP_remat padding (dim0=1160, gtp_remat_size=4,
    pad_for_alignment=16 -> 56 pad rows -> padded 1216, per-rank shard 304). Pins that the
    padded global shape round-trips when save_gtp_remat_size == load_gtp_remat_size.
    """
    update_gtp_config(pad_for_alignment=16)
    dim0_unpadded = 1160  # z(512) + x(512) + B(64) + C(64) + dt(8)
    in_features = 4
    gtp_remat_size = world_size
    alignment_block = 16 * gtp_remat_size  # = 64
    pad = (alignment_block - dim0_unpadded % alignment_block) % alignment_block
    dim0_padded = dim0_unpadded + pad
    per_shard = dim0_padded // gtp_remat_size

    gtp_remat_group = _cached_new_group(list(range(world_size)))
    weight = _make_gtp_shard(dim0_unpadded, in_features, gtp_remat_group)

    assert weight.shape == (
        per_shard,
        in_features,
    ), f"rank={rank}: post-pad shard shape {tuple(weight.shape)} != ({per_shard}, {in_features})"
    # Only rank-3 (the last GTP_remat rank) carries the trailing pad rows; all ranks
    # report the same pad_length (an invariant set by _gtp_slice_one_param).
    assert (
        getattr(weight, "pad_length", 0) == pad
    ), f"rank={rank}: pad_length {getattr(weight, 'pad_length', 0)} != {pad}"

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=_cached_new_group([rank]),
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    # Alignment padding is excluded from the logical checkpoint layout.
    start = rank * per_shard
    expected_rows = min(per_shard, max(0, dim0_unpadded - start))
    assert (
        st.global_shape[0] == dim0_unpadded
    ), f"rank={rank} pad case: global_shape[0] {st.global_shape[0]} != {dim0_unpadded}"
    assert st.global_offset[0] == start
    assert st.local_shape[0] == expected_rows, (
        f"rank={rank} pad case: local rows {st.local_shape[0]} != {expected_rows} "
        "(alignment pad rows must not reach the checkpoint)"
    )
    assert (
        not st.allow_shape_mismatch
    ), f"rank={rank} pad case: the logical layout is exact, so the shape check must stay armed"


def _worker_helper_cross_topology_reshard_metadata(rank, world_size, port):
    """Pin the cross-topology reshard contract via ShardedTensor metadata.

    We can't run a real DCP save/load against itself within a single torchrun
    (need separate worlds), but we can verify the saved ShardedTensor carries
    what DCP needs for the reshard: the LOGICAL global shape, which is the same
    at every save-time alignment/GTP degree, so no waiver is required.
    """
    update_gtp_config(pad_for_alignment=16)
    dim0_unpadded = 1160
    in_features = 4
    gtp_remat_size = world_size
    alignment_block = 16 * gtp_remat_size  # 64
    dim0_padded = (
        dim0_unpadded + (alignment_block - dim0_unpadded % alignment_block) % alignment_block
    )
    per_shard = dim0_padded // gtp_remat_size

    gtp_remat_group = _cached_new_group(list(range(world_size)))
    weight = _make_gtp_shard(dim0_unpadded, in_features, gtp_remat_group)

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=_cached_new_group([rank]),
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    # Logical coordinates are independent of save-time alignment.
    start = rank * per_shard
    expected_rows = min(per_shard, dim0_unpadded - start)
    assert st.global_shape[0] == dim0_unpadded, (
        f"rank={rank} saved global_shape ({st.global_shape[0]}) != logical ({dim0_unpadded}); "
        "alignment padding leaked into the checkpoint layout."
    )
    assert (
        not st.allow_shape_mismatch
    ), "the logical layout is exact, so the shape check must stay armed"
    assert st.global_offset[0] == start, f"rank={rank} offset {st.global_offset[0]} != {start}"
    assert (
        st.local_shape[0] == expected_rows
    ), f"rank={rank} local rows {st.local_shape[0]} != {expected_rows} (pad rows not dropped)"
    assert st.global_offset[0] + st.local_shape[0] <= st.global_shape[0]


def _worker_save_then_load_offsets_symmetric(rank, world_size, port):
    """Save-side and load-side ShardedTensors must produce identical offsets
    and global_shape so DCP can correctly resharded between them.

    We don't run the real DCP save (avoids filesystem / async-writer issues
    in CI); we just verify the symmetry property the load path relies on.
    """
    update_gtp_config(pad_for_alignment=0)
    dim0 = 16
    in_features = 4
    gtp_remat_group = _cached_new_group(list(range(world_size)))

    def _build(prefix):
        weight = _make_gtp_shard(dim0, in_features, gtp_remat_group)
        return make_sharded_tensors_for_checkpoint_with_gtp_remat(
            {"weight": weight},
            prefix=prefix,
            tensor_parallel_layers_axis_map={"weight": 0},
            sharded_offsets=(),
            tp_group=_cached_new_group([rank]),
            dp_cp_group=_cached_new_group(list(range(world_size))),
        )["layer.weight"]

    save_st = _build("layer.")
    load_st = _build("layer.")
    assert save_st.global_shape == load_st.global_shape
    assert save_st.global_offset == load_st.global_offset
    assert save_st.local_shape == load_st.local_shape
    assert save_st.replica_id == load_st.replica_id


def _worker_helper_offsets_ep_egtp(rank, world_size, port):
    """EP=2, EGTP_remat=2 (4 ranks): routed-expert weight.

    Mirrors ``TEGroupedLinear.sharded_state_dict``: expert parallelism prepends a
    global-expert axis through ``sharded_offsets``, and EGTP_remat shards each expert's
    ``out_features`` (axis 0). The GTP_remat-aware checkpoint helper layers the EGTP_remat
    axis-0 split on top of the prepended expert offset.

    rank → (ep_rank, egtp_rank): 0→(0,0) 1→(0,1) 2→(1,0) 3→(1,1).
    """
    egtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])

    ep_size, egtp_remat_size, num_gemms = 2, 2, 1
    ep_rank = rank // 2
    egtp_rank = rank % 2
    per_expert_out = 4
    per_shard_out = per_expert_out // egtp_remat_size  # 2
    in_features = 4
    num_global_experts = ep_size * num_gemms  # 2
    global_expert_idx = ep_rank * num_gemms  # + gemm_idx (0)

    # A routed-expert weight replicates over EXPERT DP, not dense dp_cp: with EP=2 the expert
    # replicas of rank 0's shard are its expt_dp peers ([0,2] here), so the writer election must
    # run over that group. Non-grouped expert linears stamp it from pg_collection.expt_dp.
    expt_dp_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])
    weight = _make_gtp_shard(
        per_expert_out, in_features, egtp_remat_group, replica_group=expt_dp_group
    )
    weight.allreduce = False  # routed-expert tag, as set by _set_expert_parameter_attributes
    assert weight.shape == (
        per_shard_out,
        in_features,
    ), f"rank={rank} EGTP_remat shape {tuple(weight.shape)} != ({per_shard_out}, {in_features})"

    sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {"weight": weight},
        prefix="",
        tensor_parallel_layers_axis_map={"weight": 0},
        # EP prepends the global-expert axis; EGTP_remat shards out_features below it.
        sharded_offsets=((0, global_expert_idx, num_global_experts),),
        tp_group=_cached_new_group([rank]),  # no TP in this case
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["weight"]
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"
    # global shape = (num_global_experts, full_out_features, in_features)
    assert st.global_shape == (num_global_experts, per_expert_out, in_features), (
        f"rank={rank} global_shape {st.global_shape} != "
        f"({num_global_experts}, {per_expert_out}, {in_features})"
    )
    # Prepended expert axis (axis 0): offset == this rank's global expert index.
    assert (
        st.global_offset[0] == global_expert_idx
    ), f"rank={rank} expert-axis offset {st.global_offset[0]} != {global_expert_idx}"
    # EGTP_remat axis (weight axis 0, shifted to global axis 1): offset == egtp_rank · per_shard.
    assert (
        st.global_offset[1] == egtp_rank * per_shard_out
    ), f"rank={rank} EGTP_remat axis-1 offset {st.global_offset[1]} != {egtp_rank * per_shard_out}"
    # Writer elected over EXPERT DP: rank 0 of [0,2] / [1,3], i.e. ranks 0 and 1 win. Electing
    # over dense dp_cp instead would misalign the election with the real replica sets.
    assert st.replica_id[2] == get_pg_rank(expt_dp_group), (
        f"rank={rank} expert replica coord {st.replica_id[2]} != "
        f"{get_pg_rank(expt_dp_group)} (elected over the dense group instead of expt_dp?)"
    )


def _worker_helper_embedding_offsets(rank, world_size, port):
    """Embedding / output_layer path: ``VocabParallelEmbedding.sharded_state_dict`` calls
    ``make_tp_sharded_tensor_for_checkpoint`` DIRECTLY (it needs allow_shape_mismatch for
    vocab padding), bypassing the GTP_remat-aware wrapper. So that helper itself must layer the
    GTP_remat axis-0 split. TP=2, GTP_remat=2, tp_axis=0 → composite axis-0 offset, same as the
    column-parallel case.
    """
    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    tp_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])

    full_vocab, hidden = 8, 4
    tp_size, gtp_remat_size = 2, 2
    per_tp = full_vocab // tp_size  # 4
    per_shard = per_tp // gtp_remat_size  # 2

    weight = _make_gtp_shard(per_tp, hidden, gtp_remat_group)
    assert weight.shape == (per_shard, hidden)

    st = make_tp_sharded_tensor_for_checkpoint(
        tensor=weight,
        key="embedding.word_embeddings.weight",
        tp_axis=0,
        allow_shape_mismatch=True,  # how VocabParallelEmbedding calls it
        prepend_offsets=(),
        tp_group=tp_group,
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"
    tp_rank = rank // 2
    gtp_rank = rank % 2
    expected_offset = (tp_rank * gtp_remat_size + gtp_rank) * per_shard
    assert (
        st.global_offset[0] == expected_offset
    ), f"rank={rank} embedding axis-0 offset {st.global_offset[0]} != {expected_offset}"
    assert (
        st.global_shape[0] == full_vocab
    ), f"rank={rank} embedding global axis-0 {st.global_shape[0]} != {full_vocab}"


def _worker_helper_public_wrapper_delegates(rank, world_size, port):
    """The public ``make_sharded_tensors_for_checkpoint`` (the entry point most layers call,
    e.g. ColumnParallelLinear / output_layer) must detect a GTPShardedParam and produce the
    GTP_remat-composite offset — i.e. it delegates to the GTP_remat-aware path not the vanilla
    TP-only one. TP=2, GTP_remat=2, column-parallel (tp_axis=0).
    """
    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    tp_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])

    full_out, in_features = 8, 4
    tp_size, gtp_remat_size = 2, 2
    per_tp_out = full_out // tp_size  # 4
    per_shard_out = per_tp_out // gtp_remat_size  # 2

    weight = _make_gtp_shard(per_tp_out, in_features, gtp_remat_group)

    sharded = make_sharded_tensors_for_checkpoint(
        {"weight": weight},
        prefix="layer.",
        tensor_parallel_layers_axis_map={"weight": 0},
        sharded_offsets=(),
        tp_group=tp_group,
        dp_cp_group=_cached_new_group(list(range(world_size))),
    )
    st = sharded["layer.weight"]
    assert isinstance(st, ShardedTensor), f"Expected ShardedTensor, got {type(st)}"
    tp_rank = rank // 2
    gtp_rank = rank % 2
    expected_offset = (tp_rank * gtp_remat_size + gtp_rank) * per_shard_out
    assert st.global_offset[0] == expected_offset, (
        f"rank={rank} public wrapper did not produce the GTP_remat-composite offset: "
        f"{st.global_offset[0]} != {expected_offset} (delegation to the GTP_remat path failed?)"
    )
    assert (
        st.global_shape[0] == full_out
    ), f"rank={rank} global axis-0 {st.global_shape[0]} != {full_out}"


def _worker_save_without_mpu_uses_stamped_replica_group(rank, world_size, port):
    """GTP save must run off the CALLER's groups, with ``parallel_state`` torn down.

    A model built on an explicit process-group grid (MiMo-style / pg_collection-only embedders)
    never initializes the MPU globals, but the GTP checkpoint path used to elect its shard writer
    via ``parallel_state.get_data_parallel_rank(..., with_gtp_remat=False)`` -- an unconditional
    read that asserts "data parallel group with CP is not initialized". Both GTP save helpers now
    take the gtp_remat-EXCLUDED DP x CP group stamped on the param at wrap time
    (``pg_collection.dp_cp``), and only fall back to the MPU globals when it is absent.

    world=4 -> tp1 * gtp_remat2 * dp2: gtp peers [0,1] / [2,3] hold DIFFERENT shards, replicas
    [0,2] / [1,3] hold the SAME shard, so a writer election over the wrong group is visible.
    """
    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    replica_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])
    world_group = _cached_new_group(list(range(world_size)))
    gtp_remat_size, per_tp_out, in_features = 2, 8, 4
    per_shard_out = per_tp_out // gtp_remat_size  # 4
    gtp_rank = rank % 2

    # Build the params while the MPU is still up (construction is not what regressed), then tear
    # it down so every group read below must come from the caller-supplied / stamped groups.
    stamped = _make_gtp_shard(per_tp_out, in_features, gtp_remat_group, replica_group=replica_group)
    unstamped = _make_gtp_shard(per_tp_out, in_features, gtp_remat_group)
    assert stamped.gtp_replica_group is replica_group, "wrap did not stamp the replica group"
    assert not hasattr(unstamped, "gtp_replica_group")

    ps.destroy_model_parallel()
    try:
        assert not ps.is_initialized(), "MPU must be down for this regression to bite"

        def _check(st, what):
            assert isinstance(st, ShardedTensor), f"{what}: got {type(st)}"
            assert st.global_shape[0] == per_tp_out, f"{what}: global {st.global_shape}"
            assert (
                st.global_offset[0] == gtp_rank * per_shard_out
            ), f"{what}: rank={rank} offset {st.global_offset} (gtp_rank={gtp_rank})"
            # Writer election over the replica group: rank 0 of [0,2] / [1,3] -> ranks 0 and 1.
            expected_replica = 0 if rank in (0, 1) else 1
            assert st.replica_id == (0, 0, expected_replica), f"{what}: {st.replica_id}"

        # Single-tensor helper (VocabParallelEmbedding / direct callers).
        _check(
            make_tp_sharded_tensor_for_checkpoint(
                tensor=stamped,
                key="single.weight",
                tp_axis=0,
                tp_group=None,  # TP=1
                dp_cp_group=world_group,
            ),
            "make_tp_sharded_tensor_for_checkpoint",
        )
        # Multi-tensor helper (the path every module's sharded_state_dict takes).
        _check(
            make_sharded_tensors_for_checkpoint_with_gtp_remat(
                {"weight": stamped},
                prefix="multi.",
                tensor_parallel_layers_axis_map={"weight": 0},
                tp_group=None,
                dp_cp_group=world_group,
            )["multi.weight"],
            "make_sharded_tensors_for_checkpoint_with_gtp_remat",
        )

        # Exactly one writer per GTP shard -- the property a wrong (gtp-inclusive) group breaks
        # by leaving one of the two shards with no main replica at all.
        st = make_tp_sharded_tensor_for_checkpoint(
            tensor=stamped, key="elect.weight", tp_axis=0, tp_group=None, dp_cp_group=world_group
        )
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (gtp_rank, is_main_replica(st.replica_id)))
        for shard in range(gtp_remat_size):
            writers = [g for g, main in gathered if g == shard and main]
            assert len(writers) == 1, f"shard {shard} has {len(writers)} writers: {gathered}"

        # The GTP-REPLICATED entries alongside a GTP weight (bias here) take their replica
        # coordinate from the same stamped group -- or from an explicit override when given.
        bias = torch.zeros(per_shard_out, dtype=torch.bfloat16, device="cuda")
        expected_replica = 0 if rank in (0, 1) else 1
        sd = make_sharded_tensors_for_checkpoint_with_gtp_remat(
            {"weight": stamped, "bias": bias},
            prefix="",
            tensor_parallel_layers_axis_map={"weight": 0},
            tp_group=None,
            dp_cp_group=world_group,
        )
        assert sd["bias"].replica_id == (0, gtp_rank, expected_replica), sd["bias"].replica_id

        override = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
        sd = make_sharded_tensors_for_checkpoint_with_gtp_remat(
            {"weight": stamped, "bias": bias},
            prefix="",
            tensor_parallel_layers_axis_map={"weight": 0},
            tp_group=None,
            dp_cp_group=world_group,
            intra_dp_cp_group=override,
        )
        assert sd["bias"].replica_id == (0, gtp_rank, gtp_rank), sd["bias"].replica_id

        # No stamped group and no MPU: refuse with an actionable error instead of guessing a
        # writer (which would silently drop or duplicate shards).
        with pytest.raises(RuntimeError, match="gtp_replica_group"):
            make_tp_sharded_tensor_for_checkpoint(
                tensor=unstamped,
                key="unstamped.weight",
                tp_axis=0,
                tp_group=None,
                dp_cp_group=world_group,
            )
    finally:
        ps.initialize_model_parallel()


def _worker_gtp_sharded_tp_replicated_roundtrip(rank, world_size, ckpt_base):
    """A GTP-sharded duplicated weight uses TP only as a checkpoint replica coordinate.

    world=4 -> TP2 x GTP2: GTP peers hold different axis-0 shards, while TP peers hold
    identical copies of each shard. The checkpoint must therefore describe GTP2, not TP2 x GTP2,
    and elect exactly one TP replica to write each GTP shard without reading MPU globals.
    """
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    tp_replica_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])
    dp_replica_group = _cached_new_group([rank])
    world_group = _cached_new_group(list(range(world_size)))
    logical_out, in_features = 8, 4
    gtp_size = 2
    local_out = logical_out // gtp_size
    tp_rank = rank // gtp_size
    gtp_rank = rank % gtp_size

    ps.destroy_model_parallel()
    try:
        weight = _make_gtp_shard(
            logical_out, in_features, gtp_remat_group, replica_group=dp_replica_group
        )
        assert not ps.is_initialized(), "MPU must be down so all groups come from the caller"

        sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
            {"weight": weight},
            prefix="duplicated.",
            tensor_parallel_layers_axis_map={},
            tp_group=tp_replica_group,
            dp_cp_group=world_group,
        )
        entry = sharded["duplicated.weight"]
        assert isinstance(entry, ShardedTensor), type(entry)
        assert entry.global_shape == (logical_out, in_features), entry.global_shape
        assert entry.global_offset == (gtp_rank * local_out, 0), entry.global_offset
        assert entry.axis_fragmentations == (gtp_size, 1), entry.axis_fragmentations
        assert entry.replica_id == (0, tp_rank, 0), entry.replica_id

        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            (gtp_rank, entry.global_offset, entry.replica_id, is_main_replica(entry.replica_id)),
        )
        for shard_rank in range(gtp_size):
            shard_records = [record for record in gathered if record[0] == shard_rank]
            assert len(shard_records) == 2, shard_records
            assert len({record[1] for record in shard_records}) == 1, shard_records
            assert sum(record[3] for record in shard_records) == 1, shard_records

        with TempNamedDir(ckpt_base / 'gtp_tp_replicated_roundtrip', sync=True) as ckpt_dir:
            save(sharded, ckpt_dir)
            loaded = load(sharded, ckpt_dir)
        torch.testing.assert_close(
            loaded["duplicated.weight"].cpu(), weight.detach().cpu(), rtol=0, atol=0
        )
    finally:
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


def _worker_helper_replicated_sink_rejects_gtp(rank, world_size, port):
    """Sanity guard: a GTPShardedParam must NEVER be saved via the replicated
    make_sharded_tensor_for_checkpoint (it would record a shard-sized global shape).
    The helper asserts; this pins that behaviour.
    """
    from megatron.core.utils import make_sharded_tensor_for_checkpoint

    gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    weight = _make_gtp_shard(4, 4, gtp_remat_group)
    with pytest.raises(AssertionError):
        make_sharded_tensor_for_checkpoint(
            weight,
            "weight",
            tp_group=_cached_new_group([rank]),
            dp_cp_group=_cached_new_group(list(range(world_size))),
        )


def _worker_mamba_replicated_param_replica_ids(rank, world_size, port):
    """MambaMixer.sharded_state_dict under GTP_remat: replicated directly-owned params
    (A_log / dt_bias / D / conv1d.*) must get conflict-free replica_ids -- unique across the
    peers holding each chunk, exactly one writer -- so DCP elects a single writer per chunk.
    """
    GTP_remat = 2  # world=4 -> tp1 * gtp2 * dp2 (exercises both gtp_remat peers and replicate DP)
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=GTP_remat
    )
    model_parallel_cuda_manual_seed(42)
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])

    config = TransformerConfig(
        num_attention_heads=32,
        num_layers=1,
        hidden_size=4096,
        mamba_num_heads=128,
        mamba_head_dim=64,
        mamba_state_dim=128,
        mamba_num_groups=8,
        use_mamba_mem_eff_path=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    submodules = MambaLayerSubmodules(
        mixer=ModuleSpec(
            module=MambaMixer,
            submodules=MambaMixerSubmodules(
                in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
            ),
        ),
        mamba_bda=get_bias_dropout_add,
    )
    layer = MambaLayer(config, submodules, layer_number=1, pg_collection=pg).cuda()
    assert any(
        isinstance(p, GTPShardedParam) for p in layer.parameters()
    ), "GTP_remat not active: no GTPShardedParam in the GTP_remat=2 Mamba layer"

    # Checkpoint replica election for gtp_remat-REPLICATED params needs the gtp_remat-INCLUSIVE
    # group so gtp_remat peers get distinct replica_ids (matches production's get_default
    # metadata, which uses the gtp_remat-inclusive default). The replicate group would collide.
    metadata = {'dp_cp_group': ps.get_data_parallel_group(with_context_parallel=True)}
    sd = layer.mixer.sharded_state_dict(prefix='mixer.', metadata=metadata)

    target_bases = {'A_log', 'dt_bias', 'D', 'conv1d.weight', 'conv1d.bias'}
    local = {}
    for key, val in sd.items():
        base = key.split('mixer.', 1)[-1]
        if base in target_bases and isinstance(
            val, (ShardedTensor, ShardedTensorFactory, ShardedObject)
        ):
            rid = val.replica_id
            if isinstance(rid, tuple):
                local[base] = tuple(rid)

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}

    if rank == 0:
        bases = set(gathered[0])
        assert bases, "no GTP_remat-replicated tiny params found in MambaMixer sharded_state_dict"
        for base in sorted(bases):
            rids = [g[base] for g in gathered]
            assert (
                len(set(rids)) == world_size
            ), f"{base}: replica_id collision across ranks -> DCP write conflict: {rids}"
            n_writers = sum(is_main_replica(r) for r in rids)
            assert n_writers == 1, f"{base}: expected exactly 1 writer, got {n_writers}: {rids}"


def _worker_replicated_param_needs_gtp_inclusive_dp_cp(rank, world_size, port):
    """Regression for the checkpoint-save duplicate-writer bug in save_checkpoint_and_time.

    A REPLICATED param's replica_id must use the gtp_remat-INCLUSIVE group (``pg.dp_cp_gtp_remat``);
    the gtp-excluded ``pg.dp_cp`` collapses gtp_remat peers to one replica_id -> multiple writers
    -> save validation failure. world=4 -> tp1*gtp2*dp2 (replicate=2 ranks, inclusive=4).
    """
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    pg = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=['tp', 'dp_cp', 'dp_cp_gtp_remat']
    )

    # The two attributes save_checkpoint_and_time may read must differ under GTP_remat, else the
    # group choice would be moot: full = replicate x gtp_remat(2).
    assert (
        get_pg_size(pg.dp_cp_gtp_remat) == get_pg_size(pg.dp_cp) * 2
    ), f"full={get_pg_size(pg.dp_cp_gtp_remat)} replicate={get_pg_size(pg.dp_cp)}"

    replicated = torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16, device="cuda"))

    def _gather_replica_ids(dp_cp_group):
        sd = make_sharded_tensors_for_checkpoint(
            {"w": replicated},
            prefix="",
            tensor_parallel_layers_axis_map={},
            tp_group=pg.tp,
            dp_cp_group=dp_cp_group,
        )
        out = [None] * world_size
        dist.all_gather_object(out, tuple(sd["w"].replica_id))
        return out

    rids_replicate = _gather_replica_ids(pg.dp_cp)  # the bug
    rids_full = _gather_replica_ids(pg.dp_cp_gtp_remat)  # the fix

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()

    if rank == 0:
        # Replicate (gtp-excluded) group: gtp_remat peers collapse -> >1 writer (reproduces bug).
        assert (
            sum(is_main_replica(r) for r in rids_replicate) > 1
        ), f"replicate dp_cp should collide across gtp_remat peers, got {rids_replicate}"
        # gtp_remat-inclusive group: every holder distinct, exactly one writer.
        assert len(set(rids_full)) == world_size, f"full-group replica_id collision: {rids_full}"
        assert (
            sum(is_main_replica(r) for r in rids_full) == 1
        ), f"full group must elect exactly one writer, got {rids_full}"


def _worker_embedding_writer_election_gtp_inclusive_default(rank, world_size, port):
    """VocabParallelEmbedding calls make_tp_sharded_tensor_for_checkpoint directly (needs
    allow_shape_mismatch), so its GTP writer election must use the gtp_remat-EXCLUDED DP group.
    Assert every axis-0 offset has exactly one main-replica writer and the offsets tile the vocab.

    world=4 -> tp1 * gtp2 * dp2: vocab split in 2 (gtp), each half replicated on 2 dp ranks.
    """
    from collections import defaultdict

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    gtp_group = ps.get_gtp_weight_remat_group()
    assert gtp_group.size() == 2, f"expected gtp_remat_size=2, got {gtp_group.size()}"

    full_vocab, hidden = 8, 4
    per_shard = full_vocab // gtp_group.size()  # 4 (tp=1 -> per_tp == full_vocab)
    weight = _make_gtp_shard(full_vocab, hidden, gtp_group)
    assert weight.shape == (per_shard, hidden)

    st = make_tp_sharded_tensor_for_checkpoint(
        tensor=weight,
        key="embedding.word_embeddings.weight",
        tp_axis=0,
        allow_shape_mismatch=True,  # how VocabParallelEmbedding calls it
        prepend_offsets=(),
        tp_group=ps.get_tensor_model_parallel_group(),
        dp_cp_group=ps.get_data_parallel_group(with_context_parallel=True),
    )
    mine = (int(st.global_offset[0]), tuple(st.replica_id))
    gathered = [None] * world_size
    dist.all_gather_object(gathered, mine)

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}

    if rank == 0:
        by_offset = defaultdict(list)
        for off, rid in gathered:
            by_offset[off].append(rid)
        assert set(by_offset) == {0, per_shard}, f"vocab offsets must tile: {sorted(by_offset)}"
        for off, rids in by_offset.items():
            n_writers = sum(is_main_replica(r) for r in rids)
            assert n_writers == 1, (
                f"vocab offset {off}: expected exactly 1 checkpoint writer, got {n_writers} "
                f"(replica_ids {rids}); a gtp_remat-inclusive default DP group leaves a shard "
                f"with no main-replica writer -> 'Invalid access pattern' at save"
            )


def _worker_mamba_inproj_optim_param_map(rank, world_size, port):
    """GTP_remat+Muon ckpt fix: in_proj's gathered+split model entry does NOT id-match the
    per-shard optimizer param, so get_param_id_to_sharded_param_map misses it (the KeyError seen in
    Float16OptimizerWithFloat16Params.sharded_state_dict). Verify the per-shard fallback used by the
    fix restores a ShardedTensor with local_shape == the optimizer param shape, which
    make_sharded_optimizer_tensor then accepts.
    """
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
    )
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        tag_gtp_params_with_names,
    )

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    model_parallel_cuda_manual_seed(42)
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp', 'gtp_remat'])
    config = TransformerConfig(
        num_attention_heads=32,
        num_layers=1,
        hidden_size=4096,
        mamba_num_heads=128,
        mamba_head_dim=64,
        mamba_state_dim=128,
        mamba_num_groups=8,
        use_mamba_mem_eff_path=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        bias_dropout_fusion=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    submodules = MambaLayerSubmodules(
        mixer=ModuleSpec(
            module=MambaMixer,
            submodules=MambaMixerSubmodules(
                in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
            ),
        ),
        mamba_bda=get_bias_dropout_add,
    )
    layer = MambaLayer(config, submodules, layer_number=1, pg_collection=pg).cuda()
    tag_gtp_params_with_names(layer)  # set _debug_name (mirrors production setup)

    in_proj_w = layer.mixer.in_proj.weight
    assert isinstance(in_proj_w, GTPShardedParam), "in_proj.weight should be GTP_remat-sharded"

    metadata = {'dp_cp_group': ps.get_data_parallel_group(with_context_parallel=True)}
    model_sd = layer.mixer.sharded_state_dict(prefix='mixer.', metadata=metadata)

    # Reproduce the gap: in_proj's per-shard optim param has no id-match in the model dict.
    id_map = get_param_id_to_sharded_param_map(model_sd, [in_proj_w])
    assert 0 not in id_map, "expected in_proj to be MISSING from id map (the KeyError gap)"

    # The fix's per-shard fallback restores a matching entry.
    key = in_proj_w._debug_name or '_gtp_optim_param_0'
    entry = make_sharded_tensors_for_checkpoint_with_gtp_remat(
        {key: in_proj_w},
        prefix='',
        tensor_parallel_layers_axis_map={key: 0},
        tp_group=ps.get_tensor_model_parallel_group(),
        dp_cp_group=ps.get_data_parallel_group(with_context_parallel=True),
    )[key]
    assert tuple(entry.local_shape) == tuple(in_proj_w.shape), (
        f"per-shard entry local_shape {tuple(entry.local_shape)} != param shape "
        f"{tuple(in_proj_w.shape)}"
    )
    # make_sharded_optimizer_tensor must accept it for a same-shape optimizer state tensor.
    opt_state = torch.zeros_like(in_proj_w)
    osh = make_sharded_optimizer_tensor(entry, opt_state, prefix='optimizer.state.exp_avg')
    assert osh is not None

    ps.destroy_model_parallel()
    ps.initialize_model_parallel()
    GTPShardedParam._chain_state = {}


# ---------------------------------------------------------------------------
# Gated-delta-product (GDP) in_proj: gather+split under GTP_remat
#
# GDP's ``in_proj.weight`` is GTP-sliced along axis 0 and zero-padded to an alignment multiple,
# while the checkpoint splits it into householder-major chunks whose boundaries do NOT line up with
# the GTP slice boundaries. ``GatedDeltaProductMixer.sharded_state_dict`` therefore all-gathers the
# shards back to the TP-local width and strips the pad before splitting (§3.3), and wraps the
# factory's merge_fn to re-pad + re-slice on load. The three workers below cover that contract.
# ---------------------------------------------------------------------------

# in_proj width = d_inner(256)*4 + 4*ngroups(2)*d_state(128) + nheads(4)*4 = 2064. With
# pad_for_alignment=32 (what setup_gtp_remat_from_recipe picks for MXFP8) and gtp_remat_size=2 the
# alignment block is 64, so 48 pad rows fire -- the padded-shard case the split path must handle.
_GDP_HIDDEN_SIZE = 256


def _build_gdp_mixer(required_pgs):
    """Build a 1-layer GatedDeltaProductMixer. Returns ``(mixer, pg, in_proj_dim)``.

    Callers must have set ``update_gtp_config(pad_for_alignment=32)`` and initialized model
    parallel with ``gtp_remat_size=2`` first.
    """
    from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec
    from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer

    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=required_pgs)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_GDP_HIDDEN_SIZE,
        num_attention_heads=4,
        mamba_num_heads=4,
        mamba_head_dim=64,
        mamba_num_groups=2,
        mamba_state_dim=128,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    mixer = GatedDeltaProductMixer(
        config,
        gdp_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        config.hidden_size,
        layer_number=1,
        pg_collection=pg,
    ).cuda()
    in_proj_dim = (
        mixer.d_inner_local_tp * (1 + mixer.num_householder)
        + (1 + mixer.num_householder) * mixer.ngroups_local_tp * mixer.d_state
        + mixer.nheads_local_tp * (1 + mixer.num_householder)
    )
    in_proj_w = mixer.in_proj.weight
    assert isinstance(in_proj_w, GTPShardedParam), "in_proj.weight should be GTP_remat-sharded"
    assert in_proj_w.data.size(0) * 2 > in_proj_dim, (
        f"expected GTP alignment padding to fire (got {in_proj_w.data.size(0)} * 2 == "
        f"{in_proj_dim}); these tests must cover the strip-pad / re-pad path"
    )
    return mixer, pg, in_proj_dim


def _gdp_valid_rows(in_proj_w, in_proj_dim):
    """Rows of this rank's GTP shard that hold real weights (the rest are alignment pad)."""
    local_rows = in_proj_w.data.size(0)
    gtp_remat_rank = torch.distributed.get_rank(in_proj_w.group)
    return max(0, min(local_rows, in_proj_dim - gtp_remat_rank * local_rows))


def _worker_gdp_inproj_gather_split(rank, world_size, port):
    """GatedDeltaProductMixer.sharded_state_dict under GTP_remat.

    Regression for the GDP save crash: the raw GTP shard neither matches ``in_proj_dim`` nor lines
    up with the in_proj split boundaries -- the pre-fix code asserted here. Verify the mixer
    gathers back to TP-local size, splits into the 6 chunks a non-GTP_remat run would write, and
    that the load-side merge_fn re-pads + re-slices back to the live GTP shard.
    """
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    try:
        model_parallel_cuda_manual_seed(42)
        update_gtp_config(pad_for_alignment=32)  # MXFP8 alignment
        mixer, _, in_proj_dim = _build_gdp_mixer(['tp', 'cp', 'gtp_remat'])
        in_proj_w = mixer.in_proj.weight

        metadata = {'dp_cp_group': ps.get_data_parallel_group(with_context_parallel=True)}
        # Pre-fix this raised AssertionError((in_proj_dim, ShardedTensor(...))).
        sd = mixer.sharded_state_dict(prefix='mixer.', metadata=metadata)

        factory = sd['mixer.in_proj.weight']
        assert isinstance(factory, ShardedTensorFactory), type(factory)
        # Save side: the gathered tensor is the full TP-local width, pad stripped.
        assert factory.data.size(0) == in_proj_dim, (factory.data.size(0), in_proj_dim)

        from megatron.core.ssm.gated_delta_product import _get_in_proj_checkpoint_split_layout

        # The chunk names/sizes come from _get_in_proj_checkpoint_split_layout (householder-major:
        # z, V0..V(M-1), K0..K(M-1), Q, b0..b(M-1), a). Derive the expectation from that helper so
        # this pins "GTP splits exactly like a non-GTP_remat run" rather than a frozen key list.
        _, expected_names = _get_in_proj_checkpoint_split_layout(
            mixer.d_inner_local_tp,
            mixer.ngroups_local_tp * mixer.d_state,
            mixer.nheads_local_tp,
            mixer.num_householder,
        )
        chunks = factory.build_fn(factory.key, factory.data, factory.replica_id, None)
        assert [t.key.rsplit('.', 1)[-1] for t in chunks] == expected_names
        assert sum(t.data.size(0) for t in chunks) == in_proj_dim

        # Load side: cat the chunks, re-pad, re-slice -> exactly this rank's live GTP shard.
        merged = factory.merge_fn([t.data for t in chunks])
        assert tuple(merged.shape) == tuple(in_proj_w.data.shape), (
            tuple(merged.shape),
            tuple(in_proj_w.data.shape),
        )
        # The pad rows the last GTP rank carries are never written to the ckpt -> back as zeros.
        n_valid = _gdp_valid_rows(in_proj_w, in_proj_dim)
        torch.testing.assert_close(merged[:n_valid], in_proj_w.data[:n_valid], rtol=0, atol=0)
        assert torch.equal(merged[n_valid:], torch.zeros_like(merged[n_valid:]))
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


def _worker_gdp_save_load_roundtrip(rank, world_size, ckpt_base):
    """End-to-end DCP save->load of a GatedDeltaProductMixer under GTP_remat.

    Companion to ``_worker_gdp_inproj_gather_split``, which only checks the factory build/merge
    functions in isolation. This drives the real ``save``/``load`` so the load-side merge_fn
    (re-pad + re-slice back to the live GTP shard) is exercised through DCP, and so a
    duplicate-writer replica_id would surface as an 'Invalid access pattern'.
    """
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    try:
        model_parallel_cuda_manual_seed(42)
        update_gtp_config(pad_for_alignment=32)  # MXFP8 alignment
        mixer, pg, in_proj_dim = _build_gdp_mixer(
            ['tp', 'cp', 'gtp_remat', 'dp_cp', 'dp_cp_gtp_remat']
        )
        in_proj_w = mixer.in_proj.weight

        # ``save_checkpoint_and_time`` threads the gtp_remat-INCLUSIVE group; using the
        # gtp_remat-excluding pg.dp_cp here collides replica_ids across gtp_remat peers.
        metadata = {'dp_cp_group': pg.dp_cp_gtp_remat}
        golden = {k: v.detach().clone() for k, v in mixer.state_dict().items()}

        with TempNamedDir(ckpt_base / 'gdp_gtp_dcp_roundtrip', sync=True) as ckpt_dir:
            save(mixer.sharded_state_dict(prefix='mixer.', metadata=metadata), ckpt_dir)

            # Scribble over every param so a no-op load cannot pass.
            with torch.no_grad():
                for p in mixer.parameters():
                    p.data.fill_(float(rank + 1))
            loaded = load(mixer.sharded_state_dict(prefix='mixer.', metadata=metadata), ckpt_dir)

        # in_proj comes back through the 6-way split + the GTP re-pad/re-slice merge_fn.
        merged = loaded['mixer.in_proj.weight']
        assert tuple(merged.shape) == tuple(in_proj_w.data.shape), (
            tuple(merged.shape),
            tuple(in_proj_w.data.shape),
        )
        n_valid = _gdp_valid_rows(in_proj_w, in_proj_dim)
        torch.testing.assert_close(
            merged[:n_valid].cpu(), golden['in_proj.weight'][:n_valid].cpu(), rtol=0, atol=0
        )

        # The rest of the mixer must round-trip too -- a colliding replica_id across gtp_remat
        # peers would either fail the load or return another rank's data.
        for name in (
            'A_log',
            'dt_bias',
            'conv1d.weight',
            'norm.weight',
            'out_proj.weight',
            'in_proj.layer_norm_weight',
        ):
            key = f'mixer.{name}'
            assert key in loaded, f"{key} missing from the loaded state dict: {sorted(loaded)}"
            torch.testing.assert_close(
                loaded[key].cpu(), golden[name].cpu(), rtol=0, atol=0, msg=f"{name} drifted"
            )
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


def _worker_gdp_inproj_optim_param_map(rank, world_size, port):
    """GDP ``in_proj`` must survive the optimizer id->ShardedTensor match (Muon path, §1.6).

    Same gap as the Mamba case (``_worker_mamba_inproj_optim_param_map``): the model entry for a
    gathered+split ``in_proj`` exposes the *gathered* tensor, so it never id-matches the per-shard
    GTP optimizer param and ``get_param_id_to_sharded_param_map`` drops it -> KeyError in
    ``Float16OptimizerWithFloat16Params.sharded_state_dict``. Unlike that test, this one drives the
    real production backfill (``_backfill_gtp_sharded_param_map``) rather than reproducing its
    rebuild, so it also pins that GDP takes the per-shard rebuild branch, not the EP refusal.
    """
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
    )
    from megatron.core.optimizer.optimizer import _backfill_gtp_sharded_param_map
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        tag_gtp_params_with_names,
    )

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    try:
        model_parallel_cuda_manual_seed(42)
        update_gtp_config(pad_for_alignment=32)  # MXFP8 alignment
        mixer, pg, in_proj_dim = _build_gdp_mixer(
            ['tp', 'cp', 'gtp_remat', 'dp_cp', 'dp_cp_gtp_remat']
        )
        tag_gtp_params_with_names(mixer)  # sets _debug_name, mirrors production setup
        in_proj_w = mixer.in_proj.weight

        metadata = {'dp_cp_group': pg.dp_cp_gtp_remat}
        model_sd = mixer.sharded_state_dict(prefix='mixer.', metadata=metadata)

        # The gap: the gathered+split factory does not id-match the per-shard optimizer param.
        id_map = get_param_id_to_sharded_param_map(model_sd, [in_proj_w])
        assert 0 not in id_map, "expected in_proj to be MISSING from the id map (the KeyError gap)"

        # The production backfill must fill it via the per-shard rebuild. An expert-parallel param
        # would raise instead; in_proj is dense, so it must rebuild cleanly.
        _backfill_gtp_sharded_param_map(id_map, [[in_proj_w]], model_sd)
        assert 0 in id_map, "backfill did not restore in_proj"
        entry = id_map[0]
        # A plain per-shard ShardedTensor keyed by the tagged name -- NOT the model's gathered+split
        # factory (reusing that would hand the optimizer the wrong shape).
        assert isinstance(entry, ShardedTensor), type(entry)
        assert entry is not model_sd['mixer.in_proj.weight']
        assert entry.key == in_proj_w._debug_name, (entry.key, in_proj_w._debug_name)
        # The trailing rank's logical checkpoint shard excludes its padding.
        gtp_remat_rank = torch.distributed.get_rank(in_proj_w.group)
        shard_rows = in_proj_w.shape[0]
        start = min(gtp_remat_rank * shard_rows, in_proj_dim)
        expected_rows = min(shard_rows, max(0, in_proj_dim - start))
        assert tuple(entry.local_shape) == (expected_rows, in_proj_w.shape[1]), (
            f"rebuilt local_shape {tuple(entry.local_shape)} != logical slice "
            f"{(expected_rows, in_proj_w.shape[1])} (param shape {tuple(in_proj_w.shape)})"
        )
        assert entry.global_offset[0] == start, (entry.global_offset, gtp_remat_rank)
        assert entry.global_shape[0] == in_proj_dim, (entry.global_shape, in_proj_dim)

        # The optimizer state spans the full padded shard; make_sharded_optimizer_tensor must
        # accept it and trim it to the same logical rows the model entry kept.
        opt_state = torch.zeros_like(in_proj_w)
        osh = make_sharded_optimizer_tensor(entry, opt_state, prefix='optimizer.state.exp_avg')
        assert osh is not None
        assert tuple(osh.local_shape) == (expected_rows, in_proj_w.shape[1]), (
            f"optimizer state {tuple(osh.local_shape)} not trimmed to "
            f"{(expected_rows, in_proj_w.shape[1])}"
        )
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


def _worker_fused_projection_checkpoint(world_size, *, kind, native_fp8=False):
    """Real fused projections preserve TP sections and padded GTP shards, including native FP8."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    model_parallel_cuda_manual_seed(42)
    original_pad = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=32 if native_fp8 else 16)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp', 'gtp_remat', 'dp_cp']
        )
        assert pg.tp.size() == pg.gtp_remat.size() == 2
        assert pg.dp_cp.size() == world_size // 4
        config_args = dict(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            normalization='RMSNorm',
            layernorm_zero_centered_gamma=False,
            activation_func=F.silu,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            transformer_impl='transformer_engine',
            fp8='e4m3' if native_fp8 else None,
            fp8_recipe='mxfp8',
            fp8_param=native_fp8,
        )
        tp_rank, gtp_rank = dist.get_rank(pg.tp), dist.get_rank(pg.gtp_remat)
        if kind == 'gdn':
            config = TransformerConfig(
                **config_args,
                linear_conv_kernel_dim=4,
                linear_key_head_dim=64,
                linear_value_head_dim=64,
                linear_num_key_heads=4,
                linear_num_value_heads=16 if native_fp8 else 8,
                experimental_attention_variant='gated_delta_net',
                linear_attention_freq=[1],
            )
            model = GatedDeltaNet(
                config,
                GatedDeltaNetSubmodules(
                    in_proj=TELayerNormColumnParallelLinear,
                    out_norm=TENorm,
                    out_proj=TERowParallelLinear,
                ),
                layer_number=1,
                bias=False,
                conv_bias=False,
                pg_collection=pg,
            ).cuda()
            weight, key = model.in_proj.weight, 'in_proj.weight'
            sizes = [128, 128, 512, 512, 8, 8] if native_fp8 else [128, 128, 256, 256, 4, 4]
            names = ['query', 'key', 'value', 'z', 'beta', 'alpha']
            expected_parts = [
                (f'{kind}.{key}.{name}', size, 2 * size, tp_rank * size)
                for name, size in zip(names, sizes)
            ]
        else:
            from megatron.core.transformer.mlp import MLP, MLPSubmodules

            config = TransformerConfig(
                **config_args, ffn_hidden_size=776, gated_linear_unit=True, add_bias_linear=False
            )
            model = MLP(
                config,
                MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
                tp_group=pg.tp,
                pg_collection=pg,
            ).cuda()
            weight, key = model.linear_fc1.weight, 'linear_fc1.weight'
            expected_parts = [
                (f'{kind}.{key}', 388, 1552, rank * 388) for rank in (tp_rank, tp_rank + 2)
            ]
            # Seed TP-local logical rows independently of the checkpoint factories.
            logical_weight = (
                torch.randn(
                    776,
                    config.hidden_size,
                    device=weight.device,
                    dtype=weight.dtype,
                    generator=torch.Generator(device=weight.device).manual_seed(1234 + tp_rank),
                )
                * 0.02
            )
            with torch.no_grad():
                weight.copy_(F.pad(logical_weight, (0, 0, 0, 24)).chunk(2)[gtp_rank])
        assert is_gtp_param(weight)
        assert is_float8tensor(weight) == native_fp8
        expected_weight = dequantize_gtp_native_fp8(weight) if native_fp8 else weight.detach()
        sharded_sd = model.sharded_state_dict(prefix=f'{kind}.', metadata={'dp_cp_group': pg.dp_cp})
        factory = sharded_sd[f'{kind}.{key}']
        assert isinstance(factory, ShardedTensorFactory)

        # Source-bound companions let model-space optimizer formats pass the early guard.
        optimizer = SimpleNamespace(
            ddp_config=SimpleNamespace(use_megatron_fsdp=False),
            buffers=[SimpleNamespace(param_index_map={weight: None})],
            state_dict=mock.Mock(side_effect=RuntimeError("optimizer state reached")),
        )
        assert factory.for_optimizer().data is weight
        for sharding_type in ('fully_reshardable', 'fully_sharded_model_space', 'dp_reshardable'):
            with pytest.raises(RuntimeError, match="optimizer state reached"):
                DistributedOptimizer.sharded_state_dict(
                    optimizer, sharded_sd, metadata={'distrib_optim_sharding_type': sharding_type}
                )
        optimizer.state_dict.reset_mock()
        # An unbound gathered tensor must still fail before optimizer state or collectives.
        with mock.patch.object(factory, 'optimizer_factory', None):
            for sharding_type in ('fully_reshardable', 'fully_sharded_model_space'):
                with pytest.raises(NotImplementedError, match="requires source-bound"):
                    DistributedOptimizer.sharded_state_dict(
                        optimizer,
                        sharded_sd,
                        metadata={'distrib_optim_sharding_type': sharding_type},
                    )
        optimizer.state_dict.assert_not_called()

        logical_rows, pad_rows = (1296, 48) if native_fp8 else (776, 24)
        assert tuple(factory.data.shape) == (logical_rows, config.hidden_size)
        assert factory.data.dtype == torch.bfloat16 and not is_float8tensor(factory.data)
        parts = factory.build()
        assert len(parts) == len(expected_parts)
        for part, (part_key, local_rows, global_rows, offset) in zip(parts, expected_parts):
            assert part.key == part_key
            assert tuple(part.local_shape) == (local_rows, config.hidden_size)
            assert tuple(part.global_shape) == (global_rows, config.hidden_size)
            assert tuple(part.global_offset) == (offset, 0)
            assert part.replica_id == (0, gtp_rank, dist.get_rank(pg.dp_cp))

        # Merge reconstructs this rank's contiguous physical rows and zeroes alignment padding.
        merged = factory.merge_fn([part.data for part in parts])
        assert tuple(merged.shape) == tuple(weight.shape)
        assert weight.pad_length == pad_rows
        valid_rows = merged.shape[0] - (pad_rows if gtp_rank == pg.gtp_remat.size() - 1 else 0)
        torch.testing.assert_close(
            merged[:valid_rows], expected_weight[:valid_rows], rtol=0, atol=0
        )
        assert torch.count_nonzero(merged[valid_rows:]) == 0

        if native_fp8:
            # Clear live storage so a no-op load cannot pass the roundtrip check.
            assert torch.count_nonzero(merged) > 0
            with torch.no_grad(), gtp_native_fp8_load_context(model):
                weight.copy_(torch.zeros_like(merged))
            assert torch.count_nonzero(dequantize_gtp_native_fp8(weight)) == 0
            with gtp_native_fp8_load_context(model):
                model.load_state_dict({key: merged}, strict=False)
            assert is_float8tensor(weight) and is_gtp_param(weight)
            restored = dequantize_gtp_native_fp8(weight)
            error = (restored - merged).abs().max() / merged.abs().max().clamp_min(1e-6)
            assert error < 0.2  # MXFP8 requantization is lossy.
        elif kind == 'swiglu':
            # One real forward checks that restored gate/up rows need no runtime permutation.
            with torch.no_grad():
                weight.zero_()
                model.load_state_dict({key: merged}, strict=False)
                inputs = torch.randn(
                    8, 2, config.hidden_size, device=weight.device, dtype=weight.dtype
                )
                actual, _ = model.linear_fc1(inputs)
                normalized = F.rms_norm(
                    inputs.float(),
                    (config.hidden_size,),
                    model.linear_fc1.layer_norm_weight.float(),
                    config.layernorm_epsilon,
                ).to(inputs.dtype)
                expected = F.linear(normalized, logical_weight)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    finally:
        update_gtp_config(pad_for_alignment=original_pad)
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()
        reset_gtp_state()


def _worker_save_load_roundtrip_needs_gtp_inclusive_group(rank, world_size, ckpt_base):
    """Save->load roundtrip: save and load must use the gtp_remat-INCLUSIVE replica group.

    Loading with the gtp_remat-EXCLUDING ``pg.dp_cp`` collides replica_ids across gtp_remat
    peers -> DCP 'Invalid access pattern' (the a55b load failure). world=4 -> tp1*gtp2*dp2.
    """
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'dp_cp', 'dp_cp_gtp_remat']
        )
        # The two group choices must differ under GTP_remat, else the test is moot.
        assert (
            get_pg_size(pg.dp_cp_gtp_remat) == get_pg_size(pg.dp_cp) * 2
        ), f"full={get_pg_size(pg.dp_cp_gtp_remat)} replicate={get_pg_size(pg.dp_cp)}"

        # A GTP-replicated param: byte-identical on every rank (like decoder.final_norm.weight).
        replicated = torch.nn.Parameter(
            torch.arange(32, dtype=torch.bfloat16, device="cuda").reshape(8, 4)
        )

        def _sd(dp_cp_group):
            return make_sharded_tensors_for_checkpoint(
                {"w": replicated},
                prefix="",
                tensor_parallel_layers_axis_map={},
                tp_group=pg.tp,
                dp_cp_group=dp_cp_group,
            )

        # Negative invariant (uniform, no failed collective): under the gtp_remat-EXCLUDING
        # replicate group the gtp_remat peers collapse to the same replica_id -> >1 main writer
        # for the identical element. That is exactly what makes DCP load-validation raise the
        # a55b 'Invalid access pattern'. (validate_sharding_integrity raises on rank 0 only, so a
        # real failed load() can't be asserted cleanly across ranks; we assert the root condition.)
        rids_excl = [None] * world_size
        dist.all_gather_object(rids_excl, tuple(_sd(pg.dp_cp)["w"].replica_id))
        rids_incl = [None] * world_size
        dist.all_gather_object(rids_incl, tuple(_sd(pg.dp_cp_gtp_remat)["w"].replica_id))
        if rank == 0:
            assert (
                sum(is_main_replica(r) for r in rids_excl) > 1
            ), f"gtp-excluding dp_cp must collide across gtp_remat peers (the bug): {rids_excl}"
            assert (
                sum(is_main_replica(r) for r in rids_incl) == 1
            ), f"gtp-inclusive group must elect exactly one writer: {rids_incl}"

        # Positive end-to-end roundtrip through the real DCP save/load with the gtp_remat-inclusive
        # group (what save_checkpoint_and_time and the fixed load_checkpoint both thread): save and
        # load must agree on this group, and the replicated data must round-trip intact.
        with TempNamedDir(ckpt_base / 'gtp_dcp_roundtrip', sync=True) as ckpt_dir:
            save(_sd(pg.dp_cp_gtp_remat), ckpt_dir)
            loaded = load(_sd(pg.dp_cp_gtp_remat), ckpt_dir)
            assert torch.equal(loaded["w"].cpu(), replicated.detach().cpu()), loaded["w"]
    finally:
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_cross_gtp_degree_save_load_roundtrip(rank, world_size, ckpt_base):
    """GTP4 saves a padded allocation that GTP2 reads in strict logical coordinates."""
    from megatron.core.dist_checkpointing import load, save
    from megatron.core.utils import grant_shape_mismatch_for_gtp_padding
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    update_gtp_config(pad_for_alignment=32)  # mxfp8-like alignment
    dim0, in_features = 192, 4  # 192 % 128 (GTP4 align) != 0 -> pads; 192 % 64 (GTP2 align) == 0

    world_group = _cached_new_group(list(range(world_size)))
    save_gtp_group = world_group  # gtp_remat_size=4 -> pads (like the reported GTP64 save)
    load_gtp_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
    # tp_group also doubles as the (trivial) GTP replica_group: save_gtp_group is the whole
    # world, so no two ranks share a shard -- no real replication.
    trivial_replica_group = _cached_new_group([rank])
    # ranks 0&2 and 1&3 land on the same GTP2 shard (two independent GTP2 groups over 4 ranks),
    # so they're REPLICAS and must share a replica_group -- else DCP sees two ranks claim the
    # same (offset, replica_id) and raises "Invalid access pattern".
    load_replica_group = _cached_new_group([0, 2]) if rank % 2 == 0 else _cached_new_group([1, 3])

    golden = torch.arange(dim0 * in_features, dtype=torch.bfloat16, device="cuda").reshape(
        dim0, in_features
    )

    def _wrap(tensor, key, **extra):
        return {
            key: make_tp_sharded_tensor_for_checkpoint(
                tensor=tensor,
                key=key,
                tp_axis=0,
                prepend_offsets=(),
                tp_group=trivial_replica_group,
                dp_cp_group=world_group,
                **extra,
            )
        }

    ps.destroy_model_parallel()
    try:
        weight_save = _make_gtp_shard(
            dim0, in_features, save_gtp_group, replica_group=trivial_replica_group
        )  # save_gtp_group covers the whole world -- no replication, trivial replica group
        assert getattr(weight_save, "pad_length", 0) == 64, weight_save.pad_length
        sharded_save = _wrap(weight_save, "weight")
        assert sharded_save["weight"].global_shape[0] == dim0, sharded_save["weight"].global_shape

        weight_load = _make_gtp_shard(
            dim0, in_features, load_gtp_group, replica_group=load_replica_group
        )
        assert getattr(weight_load, "pad_length", 0) == 0, weight_load.pad_length
        with torch.no_grad():
            weight_load.fill_(-999.0)  # scribble so a no-op load cannot pass
        sharded_load = _wrap(weight_load, "weight")
        assert sharded_load["weight"].global_shape[0] == 192, sharded_load["weight"].global_shape

        with TempNamedDir(ckpt_base / 'gtp_cross_degree_roundtrip', sync=True) as ckpt_dir:
            save(sharded_save, ckpt_dir)
            grant_shape_mismatch_for_gtp_padding(sharded_load, ckpt_dir, pad_for_alignment=32)
            assert not sharded_load["weight"].allow_shape_mismatch
            loaded = load(sharded_load, ckpt_dir)

        load_gtp_rank = torch.distributed.get_rank(load_gtp_group)
        shard_dim0_load = dim0 // load_gtp_group.size()  # 96
        expected = golden[load_gtp_rank * shard_dim0_load : (load_gtp_rank + 1) * shard_dim0_load]
        torch.testing.assert_close(loaded["weight"].cpu(), expected.cpu(), rtol=0, atol=0)
    finally:
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


def _worker_restrict_shape_mismatch_to_explainable_padding(rank, world_size, ckpt_base):
    """grant_shape_mismatch_for_gtp_padding sets allow_shape_mismatch from whether the
    checkpoint-vs-expected difference is caused by GTP padding: OFF for a mismatch that isn't
    (catches a wrong checkpoint DCP would otherwise silently accept), ON for one that is --
    including for a plain non-GTP tensor, closing the GTP-saved -> non-GTP-loaded gap where that
    consumer previously had no signal at all.
    """
    from megatron.core.dist_checkpointing import save
    from megatron.core.utils import grant_shape_mismatch_for_gtp_padding
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    pad_for_alignment = 32  # mxfp8-like alignment
    update_gtp_config(pad_for_alignment=pad_for_alignment)
    dim0, in_features = 192, 4  # same weight as the cross-degree roundtrip test above

    world_group = _cached_new_group(list(range(world_size)))
    trivial_replica_group = _cached_new_group([rank])

    def _wrap(tensor, key, prepend_offsets=(), **extra):
        return {
            key: make_tp_sharded_tensor_for_checkpoint(
                tensor=tensor,
                key=key,
                tp_axis=0,
                prepend_offsets=prepend_offsets,
                tp_group=trivial_replica_group,
                dp_cp_group=world_group,
                **extra,
            )
        }

    ps.destroy_model_parallel()
    try:
        # Construct the legacy padded checkpoint layout explicitly; new saves omit padding.
        weight_save = _make_gtp_shard(
            dim0, in_features, world_group, replica_group=trivial_replica_group
        )
        sharded_save = {
            "weight": ShardedTensor.from_rank_offsets("weight", weight_save, (0, rank, world_size))
        }
        assert sharded_save["weight"].global_shape[0] == 256, sharded_save["weight"].global_shape

        with TempNamedDir(ckpt_base / 'gtp_restrict_bound', sync=True) as ckpt_dir:
            save(sharded_save, ckpt_dir)

            # Legit: SAME logical weight (dim0=192), read at GTP2 -- expected global_shape=192,
            # dim0_unpadded=192. declared=256 IS padding of 192 (256>=192, 256%32==0).
            legit_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
            weight_legit = _make_gtp_shard(
                dim0, in_features, legit_group, replica_group=trivial_replica_group
            )
            sharded_legit = _wrap(weight_legit, "weight")
            grant_shape_mismatch_for_gtp_padding(sharded_legit, ckpt_dir, pad_for_alignment)
            assert (
                sharded_legit["weight"].allow_shape_mismatch is True
            ), "padding-caused shape difference (256 vs dim0_unpadded=192) must NOT be restricted"

            # Bogus: a DIFFERENT weight (dim0=300) reusing the same key -- declared=256 <
            # dim0_unpadded=300, not padding, flag must flip to False. Needs legit_group (size 2):
            # a size-1 group makes wrap_module_params_gtp a no-op, so is_gtp_param never fires.
            weight_bogus = _make_gtp_shard(
                300, in_features, legit_group, replica_group=trivial_replica_group
            )
            sharded_bogus = _wrap(weight_bogus, "weight")
            grant_shape_mismatch_for_gtp_padding(sharded_bogus, ckpt_dir, pad_for_alignment)
            assert sharded_bogus["weight"].allow_shape_mismatch is False, (
                "unexplainable shape discrepancy (declared=256 < dim0_unpadded=300) must stay "
                "strictly validated, never granted a mismatch-tolerant bypass"
            )

            # Non-GTP: a PLAIN tensor (never wrapped by wrap_module_params_gtp) loading the same
            # GTP4-saved key -- the GTP-saved -> non-GTP-loaded gap. allow_shape_mismatch starts
            # False and must flip ON: declared=256 is still explainable padding of dim0_unpadded=192.
            plain_weight = torch.nn.Parameter(
                torch.zeros(dim0, in_features, dtype=torch.bfloat16, device="cuda")
            )
            assert not hasattr(plain_weight, "pad_length"), "must be a genuinely plain tensor"
            sharded_plain = _wrap(plain_weight, "weight")
            assert sharded_plain["weight"].global_shape[0] == 192, sharded_plain[
                "weight"
            ].global_shape
            assert (
                sharded_plain["weight"].allow_shape_mismatch is False
            ), "a plain non-GTP tensor must never get allow_shape_mismatch from the GTP branch"
            grant_shape_mismatch_for_gtp_padding(sharded_plain, ckpt_dir, pad_for_alignment)
            assert sharded_plain["weight"].allow_shape_mismatch is True, (
                "GTP-saved (256) -> non-GTP-loaded (192) is explainable padding and must be "
                "reconciled to allow_shape_mismatch=True, closing the non-GTP load gap"
            )

        # A new padded destination has the same logical shape as an unpadded save.
        plain_unaligned = torch.nn.Parameter(
            torch.zeros(100, in_features, dtype=torch.bfloat16, device="cuda")
        )
        sharded_plain_save = _wrap(plain_unaligned, "weight_unaligned")
        assert sharded_plain_save["weight_unaligned"].global_shape[0] == 100

        with TempNamedDir(ckpt_base / 'gtp_restrict_bound_reverse', sync=True) as ckpt_dir2:
            save(sharded_plain_save, ckpt_dir2)

            weight_gtp_load = _make_gtp_shard(
                100, in_features, legit_group, replica_group=trivial_replica_group
            )
            assert getattr(weight_gtp_load, "pad_length", 0) == 28, weight_gtp_load.pad_length
            sharded_gtp_load = _wrap(weight_gtp_load, "weight_unaligned")
            assert sharded_gtp_load["weight_unaligned"].global_shape[0] == 100
            grant_shape_mismatch_for_gtp_padding(sharded_gtp_load, ckpt_dir2, pad_for_alignment)
            assert not sharded_gtp_load["weight_unaligned"].allow_shape_mismatch

        # Pre-set: a tensor whose allow_shape_mismatch=True was already granted for a reason
        # unrelated to GTP (e.g. VocabParallelEmbedding's own vocab-padding logic) -- declared=50
        # vs expected=80 is NOT explainable as GTP padding (50 < 80, and 50 % 32 != 0), so without
        # the "already True -> skip" guard this function would incorrectly clobber it to False.
        weight_vocab_like = torch.nn.Parameter(
            torch.zeros(50, in_features, dtype=torch.bfloat16, device="cuda")
        )
        sharded_vocab_save = _wrap(weight_vocab_like, "weight_vocab_like")
        with TempNamedDir(ckpt_base / 'gtp_restrict_bound_preset', sync=True) as ckpt_dir3:
            save(sharded_vocab_save, ckpt_dir3)

            weight_vocab_like_load = torch.nn.Parameter(
                torch.zeros(80, in_features, dtype=torch.bfloat16, device="cuda")
            )
            sharded_vocab_load = _wrap(
                weight_vocab_like_load,
                "weight_vocab_like",
                allow_shape_mismatch=True,  # e.g. what VocabParallelEmbedding sets
            )
            grant_shape_mismatch_for_gtp_padding(sharded_vocab_load, ckpt_dir3, pad_for_alignment)
            assert sharded_vocab_load["weight_vocab_like"].allow_shape_mismatch is True, (
                "a pre-set allow_shape_mismatch=True (non-GTP reason) must never be clobbered, "
                "even when the shape difference (50 vs 80) doesn't look like GTP padding"
            )

        # Bf16 bound: pad_for_alignment=1 (bf16 -- no tile-size requirement) must not make the
        # alignment check vacuous. declared0 % 1 == 0 for every integer, so without gating the
        # alignment disjunct on pad_for_alignment > 1, any declared0 >= expected0 would be wrongly
        # accepted as "padding" even for two genuinely incompatible plain tensors.
        weight_incompatible = torch.nn.Parameter(
            torch.zeros(200, in_features, dtype=torch.bfloat16, device="cuda")
        )
        sharded_incompatible_save = _wrap(weight_incompatible, "weight_incompatible")
        with TempNamedDir(ckpt_base / 'gtp_restrict_bound_bf16', sync=True) as ckpt_dir4:
            save(sharded_incompatible_save, ckpt_dir4)

            weight_expected = torch.nn.Parameter(
                torch.zeros(150, in_features, dtype=torch.bfloat16, device="cuda")
            )
            sharded_incompatible_load = _wrap(weight_expected, "weight_incompatible")
            grant_shape_mismatch_for_gtp_padding(sharded_incompatible_load, ckpt_dir4, 1)
            assert sharded_incompatible_load["weight_incompatible"].allow_shape_mismatch is False, (
                "declared0=200 vs expected0=150 at pad_for_alignment=1 is a genuine shape "
                "mismatch, not GTP padding -- must stay strictly validated"
            )

        # Prepended axis: a PP-layer axis ahead of dim0 (both sides' global_shape[0] is
        # num_layers=3, always equal) -- would wrongly early-exit as "no difference" if the
        # padded axis were read from index 0 instead of prepend_axis_num. Mocks the metadata
        # read (no real save/load) -- only the axis-indexing math is under test here.
        layer_axis = ((0, 1, 3),)  # (axis, layer_idx=1, num_layers=3)
        weight_layer_load = _make_gtp_shard(
            dim0, in_features, legit_group, replica_group=trivial_replica_group
        )
        sharded_layer_load = _wrap(weight_layer_load, "layer.weight", prepend_offsets=layer_axis)
        assert sharded_layer_load["layer.weight"].global_shape == (3, 192, in_features)
        fake_metadata = {"layer.weight": SimpleNamespace(global_shape=(3, 256, in_features))}
        with mock.patch(
            "megatron.core.dist_checkpointing.serialization.load_tensors_metadata",
            return_value=fake_metadata,
        ):
            grant_shape_mismatch_for_gtp_padding(sharded_layer_load, "unused", pad_for_alignment)
        assert sharded_layer_load["layer.weight"].allow_shape_mismatch is True, (
            "prepended-axis padding (declared dim0=256 vs expected dim0=192) must be "
            "recognized via prepend_axis_num, not silently no-op'd by comparing num_layers"
        )
    finally:
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


# ---------------------------------------------------------------------------
# Test class wrappers (4-GPU)
# ---------------------------------------------------------------------------


_PREFIX = "layer."  # arbitrary checkpoint key prefix; these tests do not depend on the name


def _sharded_weight(weight, rank, world_group, *, gtp, expect_global_shape):
    """Build ShardedTensors for one weight, with or without the GTP-aware builder.

    Both builders describe the unpadded global shape, independent of GTP alignment.
    """
    build = (
        make_sharded_tensors_for_checkpoint_with_gtp_remat
        if gtp
        else make_sharded_tensors_for_checkpoint
    )
    sharded = build(
        {"weight": weight},
        prefix=_PREFIX,
        tensor_parallel_layers_axis_map={},
        tp_group=_cached_new_group([rank]),
        dp_cp_group=world_group,
    )
    entry = sharded[_PREFIX + "weight"]
    assert entry.global_shape == expect_global_shape, entry.global_shape
    return sharded


def _worker_save_without_gtp_load_with_gtp_remat(rank, world_size, ckpt_base):
    """A checkpoint saved with GTP_remat disabled must load cleanly into a
    gtp_remat_size=4 run (e.g. a weight saved before GTP was turned on for that
    module). The saved ShardedTensor is the full,
    unsharded weight; on load each GTP rank must slice its own row-chunk out
    of it with no shape-mismatch error.
    """
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    ps.destroy_model_parallel()
    world_group = _cached_new_group(list(range(world_size)))
    out_features, in_features = 16, 8
    gtp_size = world_size
    local_out = out_features // gtp_size

    try:
        # ---- "save" side: GTP disabled, single full (unsharded) tensor ----
        full_weight = torch.arange(
            out_features * in_features, dtype=torch.bfloat16, device="cuda"
        ).reshape(out_features, in_features)

        no_gtp_sharded = _sharded_weight(
            full_weight,
            rank,
            world_group,
            gtp=False,
            expect_global_shape=(out_features, in_features),
        )

        with TempNamedDir(ckpt_base / 'no_gtp_save', sync=True) as ckpt_dir:
            save(no_gtp_sharded, ckpt_dir)

            # ---- "load" side: gtp_remat_size=4, weight is now GTP-sharded ----
            gtp_weight = _make_gtp_shard(
                out_features, in_features, world_group, replica_group=_cached_new_group([rank])
            )
            assert tuple(gtp_weight.shape) == (local_out, in_features), gtp_weight.shape

            # Same global_shape as the non-GTP save recorded -> no shape mismatch on load.
            gtp_sharded = _sharded_weight(
                gtp_weight,
                rank,
                world_group,
                gtp=True,
                expect_global_shape=(out_features, in_features),
            )
            loaded = load(gtp_sharded, ckpt_dir)

        torch.testing.assert_close(
            loaded[_PREFIX + "weight"].cpu(), gtp_weight.detach().cpu(), rtol=0, atol=0
        )
    finally:
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


def _worker_save_with_gtp_load_without_gtp_aligned(rank, world_size, ckpt_base):
    """Reverse direction: save WITH gtp_remat_size=4, then load into a
    run with GTP fully disabled. When out_features needs no alignment padding
    (pad_length == 0), the saved (unpadded) global_shape equals the non-GTP loader's expected
    global_shape, so the load succeeds like any ordinary resharding.
    """
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    ps.destroy_model_parallel()
    world_group = _cached_new_group(list(range(world_size)))
    out_features, in_features = 16, 8  # divisible by gtp_remat_size=4 -> no padding needed

    try:
        full_weight = torch.arange(
            out_features * in_features, dtype=torch.bfloat16, device="cuda"
        ).reshape(out_features, in_features)

        # ---- "save" side: gtp_remat_size=4 ----
        gtp_weight = _make_gtp_shard(
            out_features, in_features, world_group, replica_group=_cached_new_group([rank])
        )
        assert gtp_weight.pad_length == 0, gtp_weight.pad_length

        gtp_sharded = _sharded_weight(
            gtp_weight, rank, world_group, gtp=True, expect_global_shape=(out_features, in_features)
        )

        with TempNamedDir(ckpt_base / 'gtp_save', sync=True) as ckpt_dir:
            save(gtp_sharded, ckpt_dir)

            # ---- "load" side: GTP disabled, plain full tensor ----
            plain_weight = torch.zeros(
                out_features, in_features, dtype=torch.bfloat16, device="cuda"
            )
            no_gtp_sharded = _sharded_weight(
                plain_weight,
                rank,
                world_group,
                gtp=False,
                expect_global_shape=(out_features, in_features),
            )
            loaded = load(no_gtp_sharded, ckpt_dir)

        torch.testing.assert_close(
            loaded[_PREFIX + "weight"].cpu(), full_weight.cpu(), rtol=0, atol=0
        )
    finally:
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


def _worker_save_with_gtp_padded_load_without_gtp(rank, world_size, ckpt_base):
    """A padded GTP save loads into an ordinary model without a shape-mismatch waiver."""
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    ps.destroy_model_parallel()
    world_group = _cached_new_group(list(range(world_size)))
    out_features, in_features = 20, 8  # Needs padding to a multiple of alignment 2 x GTP4.
    orig_pad = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=2)  # alignment = pad_for_alignment * gtp_remat_size = 8

    try:
        gtp_weight = _make_gtp_shard(
            out_features, in_features, world_group, replica_group=_cached_new_group([rank])
        )
        assert gtp_weight.pad_length == 4, gtp_weight.pad_length  # (8 - 20%8) % 8

        # The checkpoint describes the unpadded (20, 8) tensor.
        gtp_sharded = _sharded_weight(
            gtp_weight, rank, world_group, gtp=True, expect_global_shape=(out_features, in_features)
        )

        with TempNamedDir(ckpt_base / 'gtp_padded_save', sync=True) as ckpt_dir:
            save(gtp_sharded, ckpt_dir)

            # ---- "load" side: GTP disabled, expects the TRUE (unpadded) shape ----
            plain_weight = torch.zeros(
                out_features, in_features, dtype=torch.bfloat16, device="cuda"
            )
            no_gtp_sharded = _sharded_weight(
                plain_weight,
                rank,
                world_group,
                gtp=False,
                expect_global_shape=(out_features, in_features),
            )

            assert not no_gtp_sharded[_PREFIX + "weight"].allow_shape_mismatch
            loaded = load(no_gtp_sharded, ckpt_dir)
            expected = torch.arange(out_features * in_features, dtype=torch.bfloat16).reshape(
                out_features, in_features
            )
            torch.testing.assert_close(loaded[_PREFIX + "weight"].cpu(), expected, rtol=0, atol=0)
    finally:
        update_gtp_config(pad_for_alignment=orig_pad)
        ps.initialize_model_parallel()
        GTPShardedParam._chain_state = {}


class TestGtpFc1SwigluDcp:
    def test_fc1_swiglu_checkpoint(self):
        world_size = dist.get_world_size()
        if world_size not in (4, 8):
            pytest.skip(f"Requires world_size=4 or 8, got {world_size}")
        _worker_fused_projection_checkpoint(world_size, kind='swiglu')


@pytest.mark.skipif(not HAVE_GDN_FLA, reason="FLA is not installed.")
class TestGtpGdnDcp:
    @pytest.mark.parametrize('native_fp8', [False, True], ids=['bf16', 'native_fp8'])
    def test_gdn_inproj_checkpoint(self, native_fp8):
        world_size = dist.get_world_size()
        if world_size not in (4, 8):
            pytest.skip(f"Requires world_size=4 or 8, got {world_size}")
        if native_fp8:
            _requires_mxfp8()
        with fp8_model_init(enabled=native_fp8, recipe=MXFP8BlockScaling()):
            _worker_fused_projection_checkpoint(world_size, kind='gdn', native_fp8=native_fp8)


@pytest.mark.run_only_on_devices_with_compute_capability(compute_capability=(10, 0))
class TestGtpDcpHelper:
    def test_mamba_replicated_param_replica_ids(self):
        _require_world_size(4)
        _worker_mamba_replicated_param_replica_ids(dist.get_rank(), 4, None)

    def test_mamba_inproj_optim_param_map(self):
        _require_world_size(4)
        _worker_mamba_inproj_optim_param_map(dist.get_rank(), 4, None)

    def test_gdp_inproj_gather_split(self):
        _require_world_size(4)
        _worker_gdp_inproj_gather_split(dist.get_rank(), 4, None)

    def test_gdp_save_load_roundtrip(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_gdp_save_load_roundtrip(dist.get_rank(), 4, tmp_path_dist_ckpt)

    def test_gdp_inproj_optim_param_map(self):
        _require_world_size(4)
        _worker_gdp_inproj_optim_param_map(dist.get_rank(), 4, None)

    def test_replicated_param_needs_gtp_inclusive_dp_cp(self):
        _require_world_size(4)
        _worker_replicated_param_needs_gtp_inclusive_dp_cp(dist.get_rank(), 4, None)

    def test_save_load_roundtrip_needs_gtp_inclusive_group(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_save_load_roundtrip_needs_gtp_inclusive_group(
            dist.get_rank(), 4, tmp_path_dist_ckpt
        )

    def test_composite_offset_same_axis(self):
        _require_world_size(4)
        _worker_helper_offsets_tp_eq_gtp_axis(dist.get_rank(), 4, None)

    def test_native_fp8_dcp_save(self):
        _require_world_size(4)
        _worker_native_fp8_dcp_save(dist.get_rank(), 4, None)

    def test_native_fp8_dcp_load_copy(self):
        _require_world_size(4)
        _worker_native_fp8_dcp_load_copy(dist.get_rank(), 4, None)

    def test_dual_offsets_cross_axis(self):
        _require_world_size(4)
        _worker_helper_offsets_tp_neq_gtp_axis(dist.get_rank(), 4, None)

    def test_ep_egtp_offsets(self):
        _require_world_size(4)
        _worker_helper_offsets_ep_egtp(dist.get_rank(), 4, None)

    def test_embedding_offsets(self):
        _require_world_size(4)
        _worker_helper_embedding_offsets(dist.get_rank(), 4, None)

    def test_embedding_writer_election(self):
        _require_world_size(4)
        _worker_embedding_writer_election_gtp_inclusive_default(dist.get_rank(), 4, None)

    def test_public_wrapper_delegates(self):
        _require_world_size(4)
        _worker_helper_public_wrapper_delegates(dist.get_rank(), 4, None)

    def test_save_without_mpu_uses_stamped_replica_group(self):
        _require_world_size(4)
        _worker_save_without_mpu_uses_stamped_replica_group(dist.get_rank(), 4, None)

    def test_gtp_sharded_tp_replicated_roundtrip(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_gtp_sharded_tp_replicated_roundtrip(dist.get_rank(), 4, tmp_path_dist_ckpt)

    def test_replicated_sink_rejects_gtp(self):
        _require_world_size(4)
        _worker_helper_replicated_sink_rejects_gtp(dist.get_rank(), 4, None)

    def test_no_op_no_gtp_remat(self):
        _require_world_size(4)
        _worker_helper_no_op_no_gtp_remat(dist.get_rank(), 4, None)

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

    def test_cross_gtp_degree_save_load_roundtrip(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_cross_gtp_degree_save_load_roundtrip(dist.get_rank(), 4, tmp_path_dist_ckpt)

    def test_restrict_shape_mismatch_to_explainable_padding(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_restrict_shape_mismatch_to_explainable_padding(
            dist.get_rank(), 4, tmp_path_dist_ckpt
        )

    def test_save_without_gtp_load_with_gtp_remat(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_save_without_gtp_load_with_gtp_remat(dist.get_rank(), 4, tmp_path_dist_ckpt)

    def test_save_with_gtp_load_without_gtp_aligned(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_save_with_gtp_load_without_gtp_aligned(dist.get_rank(), 4, tmp_path_dist_ckpt)

    def test_save_with_gtp_padded_load_without_gtp(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_save_with_gtp_padded_load_without_gtp(dist.get_rank(), 4, tmp_path_dist_ckpt)


def _worker_padded_shard_logical_offsets(rank, world_size, port):
    """Alignment padding must not shift TP offsets or enter the global shape.

    TP2/GTP2, 12 rows per TP rank and alignment 4 give four logical spans:
    8+4 | 8+4. Check exact coverage and a trailing shard containing only padding.
    """
    orig = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=4)
    try:
        gtp_remat_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
        tp_group = _cached_new_group([0, 2]) if rank in (0, 2) else _cached_new_group([1, 3])

        hidden = 4
        tp_size, gtp_remat_size = 2, 2
        per_tp, pad_length = 12, 4
        full_vocab = per_tp * tp_size  # 24 logical rows
        shard_rows = (per_tp + pad_length) // gtp_remat_size  # 8

        weight = _make_gtp_shard(per_tp, hidden, gtp_remat_group)
        assert weight.shape == (shard_rows, hidden), weight.shape
        assert weight.pad_length == pad_length, weight.pad_length

        st = make_tp_sharded_tensor_for_checkpoint(
            tensor=weight,
            key="embedding.word_embeddings.weight",
            tp_axis=0,
            prepend_offsets=(),
            tp_group=tp_group,
            dp_cp_group=_cached_new_group(list(range(world_size))),
        )

        tp_rank, gtp_rank = rank // 2, rank % 2
        start = gtp_rank * shard_rows
        expected_rows = min(shard_rows, per_tp - start)
        expected_offset = tp_rank * per_tp + start

        assert st.global_shape[0] == full_vocab, (
            f"rank={rank} global rows {st.global_shape[0]} != logical {full_vocab} "
            "(padding leaked into the global shape)"
        )
        assert st.global_offset[0] == expected_offset, (
            f"rank={rank} axis-0 offset {st.global_offset[0]} != {expected_offset} "
            f"(shifted by {st.global_offset[0] - expected_offset})"
        )
        assert (
            st.local_shape[0] == expected_rows
        ), f"rank={rank} local rows {st.local_shape[0]} != {expected_rows} (pad rows not dropped)"
        assert (
            st.global_offset[0] + st.local_shape[0] <= st.global_shape[0]
        ), f"rank={rank} shard runs past the end of the global tensor"

        # The shards must tile the logical tensor exactly -- no gap, no overlap.
        span = torch.tensor([st.global_offset[0], st.local_shape[0]], device="cuda")
        spans = [torch.empty_like(span) for _ in range(world_size)]
        dist.all_gather(spans, span)
        covered = torch.zeros(full_vocab, dtype=torch.int32)
        for off, rows in (t.cpu().tolist() for t in spans):
            covered[off : off + rows] += 1
        assert torch.all(covered == 1), f"shards do not tile the logical tensor: {covered.tolist()}"

        # A trailing shard may be entirely padding (real case: vision linear_proj, dim0 1152 with
        # alignment 128*4 -> 4 shards of 384, the last covering [1152, 1536)). That is a legal
        # configuration and must produce an empty slice, not an error. dim0 2 over GTP=2 with
        # alignment 3*2=6 reproduces it: pad 4, padded 6, shards of 3, and the second shard
        # starts at row 3 -- past the 2 logical rows.
        update_gtp_config(pad_for_alignment=3)
        tiny = _make_gtp_shard(2, hidden, gtp_remat_group)
        assert tiny.pad_length == 4, tiny.pad_length
        st_tiny = make_tp_sharded_tensor_for_checkpoint(
            tensor=tiny,
            key="all_pad.weight",
            tp_axis=0,
            prepend_offsets=(),
            tp_group=tp_group,
            dp_cp_group=_cached_new_group(list(range(world_size))),
        )
        assert st_tiny.global_shape[0] == 2 * tp_size, st_tiny.global_shape
        expected_tiny = 2 if gtp_rank == 0 else 0
        assert st_tiny.local_shape[0] == expected_tiny, (
            f"rank={rank} all-padding shard should keep {expected_tiny} rows, "
            f"got {st_tiny.local_shape[0]}"
        )
        assert st_tiny.global_offset[0] + st_tiny.local_shape[0] <= st_tiny.global_shape[0]
    finally:
        update_gtp_config(pad_for_alignment=orig)


def _worker_padded_save_load_roundtrip(rank, world_size, ckpt_base, *, native_fp8=False):
    """DCP restores logical rows into a full padded buffer for BF16 and native MXFP8.

    TP2/GTP2 uses 10 logical rows with 2 pad rows for BF16, or 144 with 48
    pad rows for MXFP8. FP8 optimizer identity and BF16 load storage must stay separate.
    Cross-topology coordinates are checked by the non-GTP -> padded-GTP test.
    """
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    if native_fp8:
        _requires_mxfp8()
    per_tp, hidden, alignment = (144, 128, 32) if native_fp8 else (10, 4, 3)
    original_pad = GTP_CONFIG.pad_for_alignment
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    update_gtp_config(pad_for_alignment=alignment)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'gtp_remat', 'dp_cp_gtp_remat']
        )
        assert pg.gtp_remat.size() == 2
        tp_rank, gtp_rank = pg.tp.rank(), pg.gtp_remat.rank()

        def _weight():
            if native_fp8:
                return _make_native_fp8_gtp_shard(per_tp, hidden, pg.gtp_remat, MXFP8BlockScaling())
            return _make_gtp_shard(per_tp, hidden, pg.gtp_remat)

        def _sd(weight):
            return {
                'w': make_tp_sharded_tensor_for_checkpoint(
                    weight, 'w', tp_axis=0, tp_group=pg.tp, dp_cp_group=pg.dp_cp_gtp_remat
                )
            }

        weight = _weight()
        if not native_fp8:
            with torch.no_grad():
                weight.add_(tp_rank * 1000.0)
        assert is_float8tensor(weight) == native_fp8
        assert weight.shape == ((96 if native_fp8 else 6), hidden)
        assert weight.pad_length == (48 if native_fp8 else 2)
        keep = min(weight.shape[0], per_tp - gtp_rank * weight.shape[0])
        saved = _sd(weight)
        entry = saved['w']
        assert entry.data.dtype == torch.bfloat16 and not is_float8tensor(entry.data)
        assert entry.local_shape == (keep, hidden)
        assert entry.global_shape == (per_tp * 2, hidden)
        if keep < weight.shape[0]:
            assert entry.gtp_pad_src is weight
            if native_fp8:
                # DCP writes the dequantized buffer; the live parameter only supplies identity.
                buffer = entry.gtp_pad_buffer
                assert buffer is not weight and not is_float8tensor(buffer)
                assert buffer.shape == weight.shape
                assert entry.data.data_ptr() == buffer.data_ptr()
        elif native_fp8:
            assert entry.data._gtp_dequant_src is weight
        else:
            assert entry.data is weight

        values = dequantize_gtp_native_fp8(weight) if native_fp8 else weight.detach()
        expected = values[:keep].clone()
        with TempNamedDir(ckpt_base / 'gtp_padded_roundtrip', sync=True) as ckpt_dir:
            save(saved, ckpt_dir)
            target = _sd(_weight())
            # Zero the destination rows so a no-op load cannot pass, including the FP8 copy.
            with torch.no_grad():
                target['w'].data.zero_()
            loaded = load(target, ckpt_dir)['w']
        assert loaded.shape == weight.shape and not is_float8tensor(loaded)
        torch.testing.assert_close(loaded[:keep], expected, rtol=0, atol=0)
    finally:
        update_gtp_config(pad_for_alignment=original_pad)
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_padded_optimizer_state_mapping(rank, world_size, port):
    """The optimizer must still find a padded GTP shard, and its state must trim with it.

    Trimming the alignment-pad tail makes ShardedTensor.data a different object from the
    optimizer's param on the shard that actually loses rows. The optimizer matches the two by
    object identity (get_param_id_to_sharded_param_map) and, in the Muon backfill, by shape.
    Without the gtp_pad_src backlink the param stops matching, which does NOT raise -- the
    state is silently left out of the checkpoint -- and where it does raise, it raises in
    make_sharded_optimizer_tensor's local_shape assert. Pin both halves here.

    TP=1 x GTP=4 (world 4), dim0 10, alignment 3*4=12 -> pad 2, padded 12, shards of 3.
    gtp ranks 0-2 keep 3 rows each, gtp rank 3 keeps 1.
    """
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
    )

    orig = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=3)
    try:
        gtp_group = _cached_new_group(list(range(world_size)))
        dim0, hidden = 10, 4
        weight = _make_gtp_shard(dim0, hidden, gtp_group)
        shard_rows = weight.shape[0]
        assert (shard_rows, weight.pad_length) == (3, 2), (shard_rows, weight.pad_length)

        sharded = {
            "w": make_tp_sharded_tensor_for_checkpoint(
                tensor=weight,
                key="w",
                tp_axis=0,
                prepend_offsets=(),
                tp_group=_cached_new_group([rank]),
                dp_cp_group=gtp_group,
            )
        }
        keep = min(shard_rows, max(0, dim0 - rank * shard_rows))
        assert sharded["w"].local_shape[0] == keep, (sharded["w"].local_shape, keep)

        # 1. The optimizer must resolve this param even on the trimmed shard.
        id_map = get_param_id_to_sharded_param_map(sharded, [weight])
        assert id_map, (
            f"rank={rank} (keep={keep}/{shard_rows}) param did not resolve; its optimizer "
            "state would be silently dropped from the checkpoint"
        )
        assert id_map[0] is sharded["w"], id_map

        # 2. Optimizer state spans the full padded shard and must trim to the logical rows.
        exp_avg = torch.arange(shard_rows * hidden, dtype=torch.float32, device="cuda").reshape(
            shard_rows, hidden
        )
        opt_st = make_sharded_optimizer_tensor(
            sharded["w"], exp_avg, prefix="optimizer.state.exp_avg"
        )
        assert tuple(opt_st.data.shape) == (
            keep,
            hidden,
        ), f"rank={rank} optimizer state {tuple(opt_st.data.shape)} != trimmed {(keep, hidden)}"
        assert torch.equal(opt_st.data, exp_avg[:keep]), "trimmed the wrong rows"
        assert opt_st.global_offset == sharded["w"].global_offset, (
            opt_st.global_offset,
            sharded["w"].global_offset,
        )
    finally:
        update_gtp_config(pad_for_alignment=orig)


def _worker_padded_fused_projection_gather(rank, world_size, port):
    """Gather full physical shards before trimming the logical projection.

    Ten rows over GTP4 with alignment 3 produce four shards of three rows.
    The trailing entry has one logical row, but the collective still needs all three.
    """
    from megatron.core.tensor_parallel.gtp_utils import _gtp_gather_rows_for_save

    orig = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=3)
    try:
        gtp_group = _cached_new_group(list(range(world_size)))
        dim0, hidden = 10, 4
        weight = _make_gtp_shard(dim0, hidden, gtp_group)
        assert (weight.shape[0], weight.pad_length) == (3, 2), (weight.shape, weight.pad_length)

        tp_group = _cached_new_group([rank])
        dp_cp_group = _cached_new_group(list(range(world_size)))
        sh_ten = make_tp_sharded_tensor_for_checkpoint(
            tensor=weight,
            key="in_proj.weight",
            tp_axis=0,
            prepend_offsets=(),
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )
        gathered = _gtp_gather_rows_for_save(
            sh_ten, "in_proj.weight", weight, dim0, tp_group, dp_cp_group, ()
        )
        assert tuple(gathered.data.shape) == (
            dim0,
            hidden,
        ), f"rank={rank} gathered {tuple(gathered.data.shape)} != {(dim0, hidden)}"
        # _make_gtp_shard fills the logical tensor with arange, so order is checkable.
        expected = torch.arange(
            dim0 * hidden, dtype=gathered.data.dtype, device=gathered.data.device
        ).reshape(dim0, hidden)
        assert torch.equal(gathered.data, expected), (
            f"rank={rank} gathered rows out of order or wrong\n"
            f"  got  {gathered.data.flatten()[:8].tolist()}\n"
            f"  want {expected.flatten()[:8].tolist()}"
        )
    finally:
        update_gtp_config(pad_for_alignment=orig)


def _worker_non_gtp_checkpoint_into_padded_gtp(rank, world_size, ckpt_base):
    """Load plain TP checkpoints into padded GTP shards using independent layouts.

    TP2/GTP2 with 10 rows per TP rank yields physical shards of 6 and logical
    spans of 6+4. Distinct row values expose offsets computed from padded TP widths.
    """
    from megatron.core.dist_checkpointing import load, save
    from tests.unit_tests.dist_checkpointing import TempNamedDir

    orig = GTP_CONFIG.pad_for_alignment
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'dp_cp', 'dp_cp_gtp_remat']
        )
        gtp_group = ps.get_gtp_weight_remat_group()
        per_tp, hidden = 10, 4
        tp_rank = ps.get_tensor_model_parallel_rank()
        gtp_rank = ps.get_gtp_weight_remat_rank()

        # Row r of the global tensor carries the value r, so any shift is visible.
        full = torch.arange(per_tp * 2, dtype=torch.bfloat16, device="cuda").reshape(-1, 1)
        full = full.expand(per_tp * 2, hidden).contiguous()
        my_rows = full[tp_rank * per_tp : (tp_rank + 1) * per_tp].clone()

        with TempNamedDir(ckpt_base / 'non_gtp_to_gtp', sync=True) as ckpt_dir:
            # --- save WITHOUT GTP: plain TP-sharded, logical global (20, 4) ---
            update_gtp_config(pad_for_alignment=0)
            plain = {
                "w": make_tp_sharded_tensor_for_checkpoint(
                    tensor=torch.nn.Parameter(my_rows.clone()),
                    key="w",
                    tp_axis=0,
                    prepend_offsets=(),
                    tp_group=pg.tp,
                    # GTP-inclusive replicate group: the two GTP peers hold identical rows,
                    # so with the GTP-excluding group they collide on one replica_id and the
                    # save deadlocks (see test_save_load_roundtrip_needs_gtp_inclusive_group).
                    dp_cp_group=pg.dp_cp_gtp_remat,
                )
            }
            assert plain["w"].global_shape[0] == per_tp * 2, plain["w"].global_shape
            save(plain, ckpt_dir)

            # --- load INTO a padded GTP model ---
            update_gtp_config(pad_for_alignment=3)
            weight = _make_gtp_shard(per_tp, hidden, gtp_group)
            assert (weight.shape[0], weight.pad_length) == (6, 2), (weight.shape, weight.pad_length)
            with torch.no_grad():
                weight.zero_()
            target = {
                "w": make_tp_sharded_tensor_for_checkpoint(
                    tensor=weight,
                    key="w",
                    tp_axis=0,
                    prepend_offsets=(),
                    tp_group=pg.tp,
                    dp_cp_group=pg.dp_cp_gtp_remat,
                )
            }
            loaded = load(target, ckpt_dir)

        got = loaded["w"]
        got = got.data if hasattr(got, "data") else got
        start = gtp_rank * 6
        keep = min(6, max(0, per_tp - start))
        want = my_rows[start : start + keep]
        assert torch.equal(got[:keep].cpu(), want.cpu()), (
            f"rank={rank} tp={tp_rank} gtp={gtp_rank}: non-GTP checkpoint loaded into the padded "
            f"GTP model gave the wrong rows -- the layout is shifted.\n"
            f"  want rows {want[:, 0].tolist()}\n"
            f"  got  rows {got[:keep, 0].cpu().tolist()}"
        )
    finally:
        update_gtp_config(pad_for_alignment=orig)
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


def _worker_padded_ep_egtp_offsets(rank, world_size, port):
    """EP2/EGTP2 keeps expert coordinates while excluding alignment padding.

    Ten rows per expert with alignment 3 give logical spans of 6+4;
    gather their metadata to verify each expert's rows have exactly one writer.
    """
    orig = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=3)
    try:
        egtp_group = _cached_new_group([0, 1]) if rank in (0, 1) else _cached_new_group([2, 3])
        ep_size, egtp_size, num_gemms = 2, 2, 1
        ep_rank, egtp_rank = rank // 2, rank % 2
        per_expert_out, in_features = 10, 4
        num_global_experts = ep_size * num_gemms
        global_expert_idx = ep_rank * num_gemms

        weight = _make_gtp_shard(per_expert_out, in_features, egtp_group)
        assert (weight.shape[0], weight.pad_length) == (6, 2), (weight.shape, weight.pad_length)

        sharded = make_sharded_tensors_for_checkpoint_with_gtp_remat(
            {"weight": weight},
            prefix="",
            tensor_parallel_layers_axis_map={"weight": 0},
            sharded_offsets=((0, global_expert_idx, num_global_experts),),
            tp_group=_cached_new_group([rank]),
            dp_cp_group=_cached_new_group(list(range(world_size))),
        )
        st = sharded["weight"]

        start = egtp_rank * 6
        keep = min(6, max(0, per_expert_out - start))
        assert st.global_shape[1] == per_expert_out, (
            f"rank={rank} expert axis-0 global {st.global_shape[1]} != logical {per_expert_out} "
            "-- alignment padding leaked into the checkpoint"
        )
        assert st.global_offset[1] == start, (st.global_offset, start)
        assert (
            st.local_shape[0] == keep
        ), f"rank={rank} local rows {st.local_shape[0]} != {keep} (pad rows not dropped)"
        assert st.global_offset[0] == global_expert_idx, st.global_offset
        assert not st.allow_shape_mismatch, "the logical layout is exact; keep the check armed"

        # The two EGTP shards of one expert must tile that expert's rows exactly.
        span = torch.tensor([global_expert_idx, start, keep], device="cuda")
        spans = [torch.empty_like(span) for _ in range(world_size)]
        dist.all_gather(spans, span)
        covered = torch.zeros(num_global_experts, per_expert_out, dtype=torch.int32)
        for e, off, rows in (t.cpu().tolist() for t in spans):
            covered[e, off : off + rows] += 1
        assert torch.all(covered == 1), f"expert rows not tiled exactly:\n{covered.tolist()}"
    finally:
        update_gtp_config(pad_for_alignment=orig)


class TestGTPPaddingCheckpoint:
    def test_padded_ep_egtp_offsets(self):
        _require_world_size(4)
        _worker_padded_ep_egtp_offsets(dist.get_rank(), 4, None)

    def test_padded_fused_projection_gather(self):
        _require_world_size(4)
        _worker_padded_fused_projection_gather(dist.get_rank(), 4, None)

    def test_padded_optimizer_state_mapping(self):
        _require_world_size(4)
        _worker_padded_optimizer_state_mapping(dist.get_rank(), 4, None)

    def test_non_gtp_checkpoint_into_padded_gtp(self, tmp_path_dist_ckpt):
        _require_world_size(4)
        _worker_non_gtp_checkpoint_into_padded_gtp(dist.get_rank(), 4, tmp_path_dist_ckpt)

    @pytest.mark.parametrize('native_fp8', [False, True], ids=['bf16', 'native_fp8'])
    def test_padded_save_load_roundtrip(self, tmp_path_dist_ckpt, native_fp8):
        _require_world_size(4)
        _worker_padded_save_load_roundtrip(
            dist.get_rank(), 4, tmp_path_dist_ckpt, native_fp8=native_fp8
        )

    def test_padded_shard_logical_offsets(self):
        _require_world_size(4)
        _worker_padded_shard_logical_offsets(dist.get_rank(), 4, None)


def _worker_egtp_grouped_fc1_swiglu_checkpoint(rank, world_size, port):
    """Real grouped experts preserve logical sections, writer ownership and optimizer sources.

    EP2 x EGTP2 uses 776 logical rows per expert and 24 padding rows; the GTP
    boundary crosses the gate/up split, so a per-shard split cannot pass.
    """
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_gtp_remat_size=2,
    )
    model_parallel_cuda_manual_seed(42)
    original_pad = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=16)
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp', 'ep', 'expt_tp', 'expt_dp', 'expt_gtp_remat', 'dp_cp']
        )
        num_local_experts = 1
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            ffn_hidden_size=776,
            moe_ffn_hidden_size=388,
            num_moe_experts=2,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            normalization='RMSNorm',
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            transformer_impl='transformer_engine',
            moe_grouped_gemm=True,
        )
        mlp = TEGroupedMLP(
            num_local_experts,
            config,
            MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
            ),
            pg_collection=pg,
        ).cuda()

        from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
            tag_gtp_params_with_names,
        )

        tag_gtp_params_with_names(mlp)
        weight = getattr(mlp.linear_fc1, "weight0")
        assert is_gtp_param(weight), "expert fc1 weight0 should be EGTP-sharded"
        egtp_size = dist.get_world_size(pg.expt_gtp_remat)
        egtp_rank = dist.get_rank(pg.expt_gtp_remat)
        assert weight.pad_length == 24, f"expected 24 pad rows, got {weight.pad_length}"

        metadata = {'dp_cp_group': pg.dp_cp}
        sharded_sd = mlp.sharded_state_dict(prefix='mlp.', metadata=metadata)
        key = next(k for k in sharded_sd if k.endswith('linear_fc1.weight0'))
        factory = sharded_sd[key]
        assert isinstance(factory, ShardedTensorFactory), type(factory)

        # The gathered tensor is the ETP-local fused [gate | up] weight with the pad stripped.
        assert tuple(factory.data.shape) == (776, config.hidden_size), factory.data.shape

        # Adam uses a source-bound physical companion; Muon keeps unsplit source metadata.
        from megatron.core.optimizer.distrib_optimizer import _get_param_id_to_sharded_metadata
        from megatron.core.optimizer.optimizer import _backfill_gtp_sharded_param_map

        companion = _get_param_id_to_sharded_metadata(sharded_sd)[id(weight)]
        assert companion.data is weight
        assert companion.key == factory.key
        muon_map = {}
        _backfill_gtp_sharded_param_map(muon_map, [[weight]], sharded_sd)
        assert isinstance(muon_map[0], ShardedTensor)
        assert muon_map[0].key == factory.key
        assert muon_map[0].global_shape == (2, 776, config.hidden_size)
        assert muon_map[0].global_offset == (pg.ep.rank(), egtp_rank * 400, 0)

        # Save side: the same two entries a non-EGTP run writes -- gate and up halves, keyed
        # WITHOUT a per-expert suffix (TEGroupedLinear carries the expert index as an offset).
        parts = factory.build()
        assert len(parts) == 2, len(parts)
        half = 776 // 2
        for part in parts:
            assert part.key.endswith('linear_fc1.weight'), part.key
            assert tuple(part.local_shape) == (half, config.hidden_size), part.local_shape

        # Load side: merge cats gate+up, re-pads, and slices back this rank's CONTIGUOUS rows,
        # element-wise equal to the live shard. Pad rows are re-zeroed, not round-tripped.
        merged = factory.merge_fn([part.data for part in parts])
        assert tuple(merged.shape) == tuple(weight.shape), (merged.shape, weight.shape)
        valid = merged.shape[0] - (weight.pad_length if egtp_rank == egtp_size - 1 else 0)
        torch.testing.assert_close(merged[:valid], weight[:valid])
        assert torch.count_nonzero(merged[valid:]) == 0

        # Writer election with a REAL EP group: each logical chunk is described by both EGTP
        # peers, and exactly one of them is the main replica.
        local_meta = [(p.key, tuple(p.global_offset), tuple(p.replica_id)) for p in parts]
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_meta)
        if rank == 0:
            by_chunk = {}
            for rank_meta in gathered:
                for k, off, rep in rank_meta:
                    by_chunk.setdefault((k, off), []).append(rep)
            for (k, off), reps in by_chunk.items():
                assert len(reps) == egtp_size, (k, off, reps)
                assert len(set(reps)) == egtp_size, f"EGTP peers share a replica_id: {reps}"
                mains = sum(is_main_replica(r) for r in reps)
                assert mains == 1, f"chunk {(k, off)} has {mains} writers: {reps}"
    finally:
        update_gtp_config(pad_for_alignment=original_pad)
        ps.destroy_model_parallel()
        ps.initialize_model_parallel()


class TestEGTPGroupedCheckpoint:
    def test_egtp_grouped_fc1_swiglu_checkpoint(self):
        _require_world_size(4)
        _worker_egtp_grouped_fc1_swiglu_checkpoint(dist.get_rank(), 4, None)
