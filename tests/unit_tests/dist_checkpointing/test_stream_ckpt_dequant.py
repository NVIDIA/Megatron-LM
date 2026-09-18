# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the streaming dequantize load path (`stream_ckpt_dequant`).

With `stream_ckpt_dequant`, `TorchDistLoadShardedStrategy` keeps the quantized destinations
(TE `QuantizedTensor`) in the state dict and quantizes each of them in place, one destination
at a time, instead of dequantizing the whole state dict to high precision before the load.

The invariant under test: the streaming path must leave bit-identical quantized storage to the
upfront-dequantize path (mirroring `load_state_dict`'s `param.copy_(loaded)`), for every recipe,
for aligned and resharding loads, with and without the fully parallel load wrapper, while
returning the quantized destinations themselves from `load` (no high-precision copies).
"""

import functools
import traceback

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from megatron.core.dist_checkpointing.utils import is_delayed_scaling_fp8tensor
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
        Float8Quantizer,
    )

    HAVE_TE = is_te_min_version("2.2.0")
except ImportError:
    HAVE_TE = False

try:
    from megatron.training.utils import get_device_arch_version

    _DEVICE_ARCH = get_device_arch_version()
except Exception:
    _DEVICE_ARCH = 0

# Delayed scaling is rejected by the streaming load (its scale is only known once
# `load_state_dict` restores the fp8 metadata); these are the supported recipes.
RECIPES = ["current", "mxfp8", "blockwise", "nvfp4"]
GEOMETRIES = [(0, 0), (0, 1), (1, 0)]
SENTINEL = 99.0
# Storage attributes holding element codes. Scale arrays are compared through the dequantized
# values only: TE pads block scale arrays, and the padding is not reproducible.
_CODE_ATTRS = ("_data", "_rowwise_data", "_columnwise_data")


def _make_quantizer(recipe):
    if recipe == "delayed":
        return Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device="cuda"),
            amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
    if recipe == "current":
        return Float8CurrentScalingQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, device="cuda")
    if recipe == "mxfp8":
        if _DEVICE_ARCH < 10:
            pytest.skip("MXFP8 requires Blackwell+")
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

        return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
    if recipe == "blockwise":
        if _DEVICE_ARCH < 9:
            pytest.skip("Blockwise FP8 requires Hopper+")
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

        return Float8BlockQuantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    if recipe == "nvfp4":
        if _DEVICE_ARCH < 10:
            pytest.skip("NVFP4 requires Blackwell+")
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

        return NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1)
    raise ValueError(recipe)


def _quantize(recipe, tensor):
    try:
        return _make_quantizer(recipe)(tensor.contiguous())
    except RuntimeError as e:
        pytest.skip(f"{recipe} quantization not supported on this device: {e}")


def _random_data(shape, seed):
    """Random BF16 data whose two halves differ 32x in dynamic range, so that a scale derived
    from the wrong region (per-tensor or per-block) cannot go unnoticed."""
    torch.manual_seed(seed)
    data = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    data[: shape[0] // 2] *= 1.0 / 32
    return data


def _storage_state(tensor):
    """Element codes and dequantized values of a quantized tensor (or the plain tensor)."""
    if not is_float8tensor(tensor):
        return {"dense": tensor.detach().clone()}
    state = {
        attr: getattr(tensor, attr).detach().clone()
        for attr in _CODE_ATTRS
        if torch.is_tensor(getattr(tensor, attr, None))
    }
    assert state, f"no quantized storage found on {type(tensor).__name__}"
    state["dequantized"] = tensor.dequantize().clone()
    return state


def _assert_same_storage(a, b, context):
    # Print before asserting: a failure on one rank only leaves the other rank waiting in the
    # next collective, and pytest's report is never reached.
    problems = []
    if a.keys() != b.keys():
        problems.append(f"storage attributes differ: {sorted(a)} vs {sorted(b)}")
    else:
        for attr in a:
            if a[attr].shape != b[attr].shape or not torch.equal(a[attr], b[attr]):
                num = int((a[attr] != b[attr]).sum()) if a[attr].shape == b[attr].shape else -1
                problems.append(f"{attr} differs in {num}/{a[attr].numel()} elements")
    if problems:
        print(f"[rank {Utils.rank}] {context}: " + "; ".join(problems), flush=True)
    assert not problems, f"{context}: {problems}"


def _storage_ptr(tensor):
    for attr in _CODE_ATTRS:
        data = getattr(tensor, attr, None)
        if torch.is_tensor(data):
            return data.data_ptr()
    return tensor.data_ptr()


def _print_rank_failures(test_fn):
    """Prints a failing rank's traceback right away: a failure on one rank only leaves the other
    rank waiting in the next collective, and pytest's report is never reached."""

    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        try:
            return test_fn(*args, **kwargs)
        except Exception:
            print(
                f"[rank {Utils.rank}] {test_fn.__name__} failed:\n{traceback.format_exc()}",
                flush=True,
            )
            raise

    return wrapper


@pytest.mark.skipif(not HAVE_TE, reason="TransformerEngine >= 2.2 not available")
class TestStreamCkptDequant:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _load_both_ways(self, ckpt_dir, recipe, make_sharded_tensor, truth, load_strategy_fn):
        """Loads the checkpoint with streaming off and on into fresh sentinel-filled quantized
        destinations, mirrors `load_state_dict`'s `dst.copy_(loaded)` and returns per-path
        (storage state, loaded tensor, destination)."""
        results = {}
        for stream in (False, True):
            dst = _quantize(recipe, torch.full_like(truth, SENTINEL))
            sharded_state_dict = {'w': make_sharded_tensor(dst)}
            try:
                loaded = load(sharded_state_dict, ckpt_dir, load_strategy_fn(stream))['w']
            except Exception as e:
                # See `_assert_same_storage`: make rank-local failures visible in the log.
                print(
                    f"[rank {Utils.rank}] {recipe} stream={stream}: load raised {e!r}", flush=True
                )
                raise
            with torch.no_grad():
                dst.copy_(loaded)
            results[stream] = (_storage_state(dst), loaded, dst)
        return results

    def _check_results(self, results, truth, context):
        legacy_state, legacy_loaded, _ = results[False]
        stream_state, stream_loaded, stream_dst = results[True]
        _assert_same_storage(legacy_state, stream_state, context)
        # Both paths quantize the same values, so equality alone could also mean "both did
        # nothing": check the load actually replaced the sentinel.
        err = (stream_state["dequantized"].float() - truth.float()).abs().max().item()
        assert err < 0.25 * SENTINEL, f"{context}: loaded values do not match the checkpoint"
        # The upfront path hands back the high-precision buffer DCP loaded into; the streaming
        # path hands back (a view of) the destination itself.
        assert not is_float8tensor(legacy_loaded)
        assert is_float8tensor(stream_loaded), context
        assert _storage_ptr(stream_loaded) == _storage_ptr(stream_dst), context

    def test_default_is_off(self):
        assert TorchDistLoadShardedStrategy().stream_ckpt_dequant is False
        assert MCoreLoadPlanner().stream_ckpt_dequant is False
        wrapped = FullyParallelLoadStrategyWrapper(
            TorchDistLoadShardedStrategy(stream_ckpt_dequant=True)
        )
        assert wrapped.stream_ckpt_dequant is True

    @pytest.mark.parametrize('recipe', RECIPES + ["delayed"])
    @_print_rank_failures
    def test_delayed_scaling_detection(self, recipe):
        tensor = _quantize(recipe, torch.ones(128, 128, dtype=torch.bfloat16, device='cuda'))
        assert is_delayed_scaling_fp8tensor(tensor) == (recipe == "delayed")

    @_print_rank_failures
    def test_delayed_scaling_rejected(self, tmp_path_dist_ckpt):
        """Delayed scaling is not supported: its scale is only restored by `load_state_dict`."""
        data = _random_data((128, 128), seed=11)
        with TempNamedDir(tmp_path_dist_ckpt / 'stream_dequant_delayed', sync=True) as ckpt_dir:
            save(
                {'w': ShardedTensor.from_rank_offsets('w', data, replica_id=Utils.rank)},
                ckpt_dir,
                TorchDistSaveShardedStrategy('torch_dist', 1),
            )
            dst = _quantize("delayed", torch.full_like(data, SENTINEL))
            with pytest.raises(CheckpointingException, match="delayed-scaling"):
                load(
                    {'w': ShardedTensor.from_rank_offsets('w', dst, replica_id=Utils.rank)},
                    ckpt_dir,
                    TorchDistLoadShardedStrategy(stream_ckpt_dequant=True),
                )

    @pytest.mark.parametrize('recipe', RECIPES)
    @pytest.mark.parametrize('save_axis,load_axis', GEOMETRIES)
    @_print_rank_failures
    def test_matches_upfront_dequantize(self, tmp_path_dist_ckpt, recipe, save_axis, load_axis):
        """Aligned (same save/load axis) and resharding (different axes) loads.

        Resharding is the load-bearing case: each destination is then fed by several partial
        read items, which is where a per-read-item quantization corrupts data-derived scales
        and where narrowing a TE quantized tensor silently drops the loaded bytes.
        """
        world = Utils.world_size
        if save_axis != load_axis and world < 2:
            pytest.skip("Resharding needs at least 2 ranks")
        # 128 is a multiple of every block size in play (blockwise: 128, MXFP8: 32, NVFP4: 16)
        # and stays so after the per-rank split.
        n = 128 * world
        full = _random_data((n, n), seed=1234)
        chunk = n // world

        def rank_slice(axis):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(Utils.rank * chunk, (Utils.rank + 1) * chunk)
            return full[tuple(sl)].contiguous()

        truth = rank_slice(load_axis)
        with TempNamedDir(
            tmp_path_dist_ckpt / f'stream_dequant_{recipe}_{save_axis}{load_axis}', sync=True
        ) as ckpt_dir:
            save(
                {
                    'w': ShardedTensor.from_rank_offsets(
                        'w', rank_slice(save_axis), (save_axis, Utils.rank, world)
                    )
                },
                ckpt_dir,
                TorchDistSaveShardedStrategy('torch_dist', 1),
            )
            results = self._load_both_ways(
                ckpt_dir,
                recipe,
                lambda dst: ShardedTensor.from_rank_offsets(
                    'w', dst, (load_axis, Utils.rank, world)
                ),
                truth,
                lambda stream: TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream),
            )
        self._check_results(results, truth, f"{recipe} save{save_axis}->load{load_axis}")

    @pytest.mark.parametrize('recipe', RECIPES)
    @_print_rank_failures
    def test_prepend_axis(self, tmp_path_dist_ckpt, recipe):
        """Destinations with prepended singleton axes (e.g. per-expert weights) go through
        `view` on both the way in and the way out, never through a dequantizing `select`."""
        world = Utils.world_size
        full = _random_data((world, 128, 256), seed=42)
        truth = full[Utils.rank].contiguous()
        with TempNamedDir(tmp_path_dist_ckpt / f'stream_dequant_prepend_{recipe}', sync=True) as (
            ckpt_dir
        ):
            save(
                {
                    'w': ShardedTensor.from_rank_offsets(
                        'w', truth, (0, Utils.rank, world), prepend_axis_num=1
                    )
                },
                ckpt_dir,
                TorchDistSaveShardedStrategy('torch_dist', 1),
            )
            results = self._load_both_ways(
                ckpt_dir,
                recipe,
                lambda dst: ShardedTensor.from_rank_offsets(
                    'w', dst, (0, Utils.rank, world), prepend_axis_num=1
                ),
                truth,
                lambda stream: TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream),
            )
        assert results[True][1].shape == truth.shape
        self._check_results(results, truth, f"{recipe} prepend_axis_num=1")

    @pytest.mark.parametrize('recipe', RECIPES)
    @_print_rank_failures
    def test_fully_parallel_load(self, tmp_path_dist_ckpt, recipe):
        """With the fully parallel load, replicated tensors are loaded by one rank and broadcast:
        the loading rank's raw quantized storage (codes and scales) is replicated straight into
        the receivers' destinations."""
        world = Utils.world_size
        if world < 2:
            pytest.skip("Fully parallel load needs at least 2 ranks")
        full = _random_data((256, 256), seed=7)
        truth = full.contiguous()
        with TempNamedDir(tmp_path_dist_ckpt / f'stream_dequant_fpsl_{recipe}', sync=True) as (
            ckpt_dir
        ):
            save(
                {'w': ShardedTensor.from_rank_offsets('w', truth, replica_id=Utils.rank)},
                ckpt_dir,
                TorchDistSaveShardedStrategy('torch_dist', 1),
            )
            results = self._load_both_ways(
                ckpt_dir,
                recipe,
                lambda dst: ShardedTensor.from_rank_offsets('w', dst, replica_id=Utils.rank),
                truth,
                lambda stream: FullyParallelLoadStrategyWrapper(
                    TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream), None, False
                ),
            )
        self._check_results(results, truth, f"{recipe} fully parallel broadcast")

    @pytest.mark.parametrize('exchange_algo', ['gather_rounds', 'gather_object'])
    @_print_rank_failures
    def test_fully_parallel_other_exchange_algorithms_rejected(
        self, tmp_path_dist_ckpt, exchange_algo
    ):
        """Only the `broadcast` exchange replicates quantized storage; the others are refused."""
        world = Utils.world_size
        if world < 2:
            pytest.skip("Fully parallel load needs at least 2 ranks")
        data = _random_data((128, 128), seed=3)
        with TempNamedDir(
            tmp_path_dist_ckpt / f'stream_dequant_fpsl_{exchange_algo}', sync=True
        ) as ckpt_dir:
            save(
                {'w': ShardedTensor.from_rank_offsets('w', data, replica_id=Utils.rank)},
                ckpt_dir,
                TorchDistSaveShardedStrategy('torch_dist', 1),
            )
            dst = _quantize("current", torch.full_like(data, SENTINEL))
            with pytest.raises(CheckpointingException, match="broadcast"):
                load(
                    {'w': ShardedTensor.from_rank_offsets('w', dst, replica_id=Utils.rank)},
                    ckpt_dir,
                    FullyParallelLoadStrategyWrapper(
                        TorchDistLoadShardedStrategy(stream_ckpt_dequant=True),
                        None,
                        False,
                        exchange_algo,
                    ),
                )

    @pytest.mark.parametrize('recipe', RECIPES)
    @pytest.mark.parametrize('half_rows', [128, 64])
    @_print_rank_failures
    def test_swiglu_factory(self, tmp_path_dist_ckpt, recipe, half_rows):
        """SwiGLU `linear_fc1` is checkpointed through a ShardedTensorFactory that splits the
        parameter in two views. Views that alias the quantized storage (MXFP8 with 128-row
        aligned halves) are loaded in place and the merge returns the parameter itself; views
        sharing one per-tensor scale (Float8), dequantized by TE (NVFP4, blockwise) or with padded
        scale copies (MXFP8, 64-row halves) take the upfront path. Both must match it bit for bit.
        """
        from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

        world = Utils.world_size
        rows = 2 * half_rows
        full = _random_data((rows * world, 128), seed=9)
        truth = full[Utils.rank * rows : (Utils.rank + 1) * rows].contiguous()

        def make_factory(data):
            return apply_swiglu_sharded_factory(
                ShardedTensor.from_rank_offsets('w', data, (0, Utils.rank, world)), ()
            )

        with TempNamedDir(
            tmp_path_dist_ckpt / f'stream_dequant_swiglu_{recipe}_{half_rows}', sync=True
        ) as ckpt_dir:
            save(
                {'w': make_factory(truth)}, ckpt_dir, TorchDistSaveShardedStrategy('torch_dist', 1)
            )
            results = self._load_both_ways(
                ckpt_dir,
                recipe,
                make_factory,
                truth,
                lambda stream: TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream),
            )
        legacy_state, legacy_loaded, _ = results[False]
        stream_state, stream_loaded, stream_dst = results[True]
        context = f"{recipe} swiglu factory half_rows={half_rows}"
        _assert_same_storage(legacy_state, stream_state, context)
        err = (stream_state["dequantized"].float() - truth.float()).abs().max().item()
        assert err < 0.25 * SENTINEL, f"{context}: loaded values do not match the checkpoint"
        assert not is_float8tensor(legacy_loaded)
        if recipe == "mxfp8" and half_rows % 128 == 0:
            assert is_float8tensor(stream_loaded), context
            assert _storage_ptr(stream_loaded) == _storage_ptr(stream_dst), context
        else:
            assert not is_float8tensor(stream_loaded), context

    @_print_rank_failures
    def test_planner_state_cleanup(self, tmp_path_dist_ckpt):
        """No scratch buffer may survive the load: holding them is what the streaming path
        exists to avoid."""
        captured = []
        original_init = MCoreLoadPlanner.__init__

        def capturing_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured.append(self)

        data = _random_data((128, 128), seed=3)
        with TempNamedDir(tmp_path_dist_ckpt / 'stream_dequant_cleanup', sync=True) as ckpt_dir:
            save(
                {
                    f'w{i}': ShardedTensor.from_rank_offsets(f'w{i}', data, replica_id=Utils.rank)
                    for i in range(4)
                },
                ckpt_dir,
                TorchDistSaveShardedStrategy('torch_dist', 1),
            )
            MCoreLoadPlanner.__init__ = capturing_init  # type: ignore[assignment]
            try:
                load(
                    {
                        f'w{i}': ShardedTensor.from_rank_offsets(
                            f'w{i}',
                            _quantize("current", torch.full_like(data, SENTINEL)),
                            replica_id=Utils.rank,
                        )
                        for i in range(4)
                    },
                    ckpt_dir,
                    TorchDistLoadShardedStrategy(stream_ckpt_dequant=True),
                )
            finally:
                MCoreLoadPlanner.__init__ = original_init  # type: ignore[assignment]
        assert len(captured) == 1
        assert captured[0]._stream_dequant_buffers == {}

    @pytest.mark.parametrize('recipe', ["current", "mxfp8"])
    @_print_rank_failures
    def test_peak_memory(self, tmp_path_dist_ckpt, recipe):
        """The upfront path holds a high-precision copy of every quantized tensor at once; the
        streaming path holds at most one scratch per in-flight destination."""
        num_tensors, shape = 8, (2048, 2048)
        tensor_bytes = shape[0] * shape[1] * 2
        data = _random_data(shape, seed=5)
        with TempNamedDir(tmp_path_dist_ckpt / f'stream_dequant_mem_{recipe}', sync=True) as (
            ckpt_dir
        ):
            save(
                {
                    f'w{i}': ShardedTensor.from_rank_offsets(f'w{i}', data, replica_id=Utils.rank)
                    for i in range(num_tensors)
                },
                ckpt_dir,
                TorchDistSaveShardedStrategy('torch_dist', 1),
            )
            peaks = {}
            for stream in (False, True):
                # Keep the destinations referenced, as model params are: the upfront path's
                # high-precision copies come on top of them, not instead of them.
                dsts = [
                    _quantize(recipe, torch.full_like(data, SENTINEL)) for _ in range(num_tensors)
                ]
                sharded_state_dict = {
                    f'w{i}': ShardedTensor.from_rank_offsets(
                        f'w{i}', dsts[i], replica_id=Utils.rank
                    )
                    for i in range(num_tensors)
                }
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                load(
                    sharded_state_dict,
                    ckpt_dir,
                    TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream),
                )
                torch.cuda.synchronize()
                peaks[stream] = torch.cuda.max_memory_allocated() - baseline
                del sharded_state_dict, dsts
        assert peaks[False] >= num_tensors * tensor_bytes, peaks
        assert peaks[True] <= 3 * tensor_bytes, peaks
