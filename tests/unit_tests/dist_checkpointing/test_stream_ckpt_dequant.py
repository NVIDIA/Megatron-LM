# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the streaming per-tensor dequantize path used when loading
distributed checkpoints with quantized (FP8 / MXFP8 / blockwise / NVFP4)
model parameters.

The feature under test is ``stream_ckpt_dequant`` on ``MCoreLoadPlanner`` and
``TorchDistLoadShardedStrategy``. When on, the LoadPlanner dequantizes each
quantized destination one at a time inside ``resolve_tensor``/``commit_tensor``
instead of up-front in ``force_all_tensors_to_non_fp8``. Tests cover:

- Loaded-content equivalence vs. the legacy upfront path (FP8).
- Delayed-scaling ``amax_history`` is not polluted across a streaming load.
- Peak GPU memory during load is strictly lower with streaming on.
- ``_unwrap_pyt_sharded_tensor`` uses view-based axis stripping (no
  dequantize fallback) — exercised implicitly by the MXFP8 save/load test.
- MXFP8 save/load round-trip.
- No-op fall-through for plain (non-quantized) tensors.
"""

import gc
from contextlib import contextmanager

import pytest
import torch

try:
    from transformer_engine.pytorch.float8_tensor import Float8Tensor
    from transformer_engine.pytorch.tensor import QuantizedTensor

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
    Float8Tensor = None  # type: ignore
    QuantizedTensor = None  # type: ignore

try:
    import transformer_engine.pytorch.tensor.mxfp8_tensor  # noqa: F401

    HAVE_MXFP8 = True
except ImportError:
    HAVE_MXFP8 = False

try:
    import transformer_engine.pytorch.tensor.nvfp4_tensor  # noqa: F401

    HAVE_NVFP4 = True
except ImportError:
    HAVE_NVFP4 = False

try:
    from megatron.training.utils import get_device_arch_version

    _DEVICE_ARCH = get_device_arch_version()
except Exception:
    _DEVICE_ARCH = 0

# NVFP4 requires Blackwell (arch 10+).
HAVE_NVFP4_HW = HAVE_NVFP4 and _DEVICE_ARCH >= 10

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _to_float8(tensor: torch.Tensor):
    """Convert a BF16 tensor to delayed-scaling Float8Tensor (TE 2.x API)."""
    try:
        return Float8Tensor.to_float8(tensor)
    except Exception:
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

        quantizer = Float8Quantizer(
            scale=torch.full([1], 1.0, dtype=torch.float32, device="cuda"),
            amax=torch.empty([1], dtype=torch.float32, device="cuda"),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        return quantizer(tensor.cuda())


def _to_mxfp8(tensor: torch.Tensor):
    """Convert a BF16 tensor to MXFP8Tensor."""
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
    return quantizer(tensor.cuda().contiguous())


def _to_nvfp4(tensor: torch.Tensor):
    """Convert a BF16 tensor to NVFP4Tensor (Blackwell+ only)."""
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
        with_random_sign_mask=False,
    )
    return quantizer(tensor.cuda().contiguous())


@contextmanager
def _measure_peak_cuda(device: int = 0):
    """Measure peak CUDA allocation inside the block as a delta above a clean baseline.

    Flushes Python GC, synchronizes, empties the CUDA cache, and records the
    current ``memory_allocated`` as the baseline before resetting peak stats.
    This keeps the measurement stable across back-to-back calls and strips out
    allocator fragmentation carried over from earlier work (e.g. the save
    phase). Populates ``baseline``, ``peak``, and ``delta`` on exit.
    """
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = {"baseline": baseline}
    try:
        yield out
    finally:
        torch.cuda.synchronize(device)
        out["peak"] = torch.cuda.max_memory_allocated(device)
        out["delta"] = out["peak"] - baseline


@pytest.mark.skipif(not HAVE_TE, reason="TransformerEngine not available")
class TestStreamCkptDequant:
    """Unit tests for streaming per-tensor dequantize during ckpt load."""

    # ---------------------------------------------------------------
    # Baseline: FP8 (delayed scaling) save/load equivalence + amax safety
    # ---------------------------------------------------------------

    @pytest.mark.parametrize('stream_ckpt_dequant', [False, True])
    def test_fp8_save_load_content_equivalence(self, tmp_path_dist_ckpt, stream_ckpt_dequant):
        """Loaded FP8 contents must match regardless of which dequantize path is used."""
        Utils.initialize_model_parallel(1, 1)

        fill_val = 0.5

        def get_fp8_tensor(val):
            return _to_float8(torch.full((8,), val, dtype=torch.bfloat16, device='cuda'))

        def get_state_dict(val):
            return {
                'w': ShardedTensor.from_rank_offsets(
                    'w', get_fp8_tensor(val), replica_id=Utils.rank
                )
            }

        with TempNamedDir(tmp_path_dist_ckpt / f'fp8_eq_{stream_ckpt_dequant}') as ckpt_dir:
            save(get_state_dict(fill_val), ckpt_dir, TorchDistSaveShardedStrategy())

            # Fresh state dict with a different fill — the load must overwrite it.
            sd_to_load = get_state_dict(99.0)
            strategy = TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream_ckpt_dequant)
            loaded = load(sd_to_load, ckpt_dir, strategy)
            # Dequantize the loaded tensor (may be Float8 or BF16 depending on path)
            loaded_w = loaded['w']
            if isinstance(loaded_w, QuantizedTensor):
                loaded_w = loaded_w.dequantize()
            # fill_val (0.5) is exactly representable in FP8 E4M3 and the per-tensor
            # scale is a power of 2, so the round-trip is numerically lossless modulo
            # bf16 rounding. Tight tolerance catches real regressions.
            torch.testing.assert_close(
                loaded_w,
                torch.full((8,), fill_val, dtype=torch.bfloat16, device='cuda'),
                rtol=1e-3,
                atol=1e-3,
            )

        Utils.destroy_model_parallel()

    def test_fp8_amax_history_not_polluted(self, tmp_path_dist_ckpt):
        """Delayed-scaling amax must be snapshotted & restored across a streaming load."""
        Utils.initialize_model_parallel(1, 1)

        def get_fp8_tensor(val):
            return _to_float8(torch.full((8,), val, dtype=torch.bfloat16, device='cuda'))

        sd_to_save = {
            'w': ShardedTensor.from_rank_offsets('w', get_fp8_tensor(0.25), replica_id=Utils.rank)
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'fp8_amax') as ckpt_dir:
            save(sd_to_save, ckpt_dir, TorchDistSaveShardedStrategy())

            # Rebuild destination with a known (distinct) amax value we can check
            # survives the streaming load.
            dst = ShardedTensor.from_rank_offsets('w', get_fp8_tensor(99.0), replica_id=Utils.rank)
            q = getattr(dst.data, "_quantizer", None)
            if q is None or not isinstance(getattr(q, "amax", None), torch.Tensor):
                pytest.skip("This TE build's Float8Tensor has no quantizer.amax scalar")
            sentinel = 42.0
            q.amax.fill_(sentinel)
            pre_load_amax = q.amax.detach().clone()

            strategy = TorchDistLoadShardedStrategy(stream_ckpt_dequant=True)
            loaded = load({'w': dst}, ckpt_dir, strategy)

            # The loaded tensor is the same QuantizedTensor object; its quantizer.amax
            # must be exactly what we put in before the load.
            loaded_q = getattr(loaded['w'], "_quantizer", None)
            assert loaded_q is not None
            assert torch.equal(loaded_q.amax, pre_load_amax), (
                f"amax was not restored after streaming load; "
                f"before={pre_load_amax.item()} after={loaded_q.amax.item()}"
            )

        Utils.destroy_model_parallel()

    # ---------------------------------------------------------------
    # Memory: streaming must allocate a smaller peak than the bulk path.
    # ---------------------------------------------------------------

    def test_fp8_streaming_reduces_peak_memory(self, tmp_path_dist_ckpt):
        """Peak CUDA allocation during load must be strictly lower with streaming on.

        We stress-test with many moderately-large FP8 tensors so the N-upfront-BF16
        allocation dominates the plateau.
        """
        Utils.initialize_model_parallel(1, 1)

        # 32 tensors * 4 MiB FP8 per rank = 128 MiB of FP8, 256 MiB of upfront BF16
        # under the legacy path; streaming should only keep ~8 MiB scratch live at once.
        num_tensors = 32
        numel = 4 * 1024 * 1024  # 4 Mi elements per rank -> 4 MiB FP8, 8 MiB BF16

        def get_fp8_tensor():
            return _to_float8(torch.zeros(numel, dtype=torch.bfloat16, device='cuda'))

        # Shard each tensor along axis 0 across ranks with replica_id=0 so every rank
        # is the main replica of its own slice and actually materializes the load
        # work (dequant allocations, streaming scratches). Using replica_id=Utils.rank
        # would make only rank 0 the main replica and leave the other ranks with
        # near-zero memory deltas that fail the comparison.
        def build_state_dict():
            return {
                f'w{i}': ShardedTensor.from_rank_offsets(
                    f'w{i}',
                    get_fp8_tensor(),
                    (0, Utils.rank, Utils.world_size),
                    replica_id=0,
                )
                for i in range(num_tensors)
            }

        with TempNamedDir(tmp_path_dist_ckpt / 'fp8_mem') as ckpt_dir:
            save(build_state_dict(), ckpt_dir, TorchDistSaveShardedStrategy())

            # Legacy path.
            dst_legacy = build_state_dict()
            with _measure_peak_cuda() as legacy:
                load(dst_legacy, ckpt_dir, TorchDistLoadShardedStrategy(stream_ckpt_dequant=False))
            # Drop legacy state so it doesn't contribute to the streaming baseline.
            del dst_legacy
            gc.collect()
            torch.cuda.empty_cache()

            # Streaming path.
            dst_stream = build_state_dict()
            with _measure_peak_cuda() as streaming:
                load(dst_stream, ckpt_dir, TorchDistLoadShardedStrategy(stream_ckpt_dequant=True))

            legacy_delta = legacy["delta"]
            streaming_delta = streaming["delta"]
            # Streaming's scratch allocations are bounded per-tensor; legacy holds all
            # BF16 scratches upfront. Compare the load-induced delta above each path's
            # clean baseline. Two assertions make this robust:
            #   1. Streaming uses at most half the load-time memory of legacy.
            #   2. Absolute saving is at least `num_tensors // 4` tensors' worth of BF16.
            # The factor 1/2 accounts for the fact that legacy also frees the original
            # FP8 bytes as it replaces `.data` with BF16, so the net growth is smaller
            # than `N * bf16_bytes`.
            bf16_bytes_per_tensor = numel * 2
            min_absolute_saving = (num_tensors // 4) * bf16_bytes_per_tensor
            saving = legacy_delta - streaming_delta
            assert streaming_delta * 2 <= legacy_delta, (
                f"Streaming did not halve the load-time memory. "
                f"legacy_delta={legacy_delta/1e6:.1f} MB, "
                f"streaming_delta={streaming_delta/1e6:.1f} MB"
            )
            assert saving >= min_absolute_saving, (
                f"Streaming saving below the meaningful floor. "
                f"legacy_delta={legacy_delta/1e6:.1f} MB, "
                f"streaming_delta={streaming_delta/1e6:.1f} MB, "
                f"saving={saving/1e6:.1f} MB, "
                f"min_absolute_saving={min_absolute_saving/1e6:.1f} MB"
            )

        Utils.destroy_model_parallel()

    # ---------------------------------------------------------------
    # MXFP8: exercises both the streaming dequant AND the view-based
    # _unwrap_pyt_sharded_tensor fix (without it, ten[0] on MXFP8 OOMs).
    # ---------------------------------------------------------------

    @pytest.mark.skipif(not HAVE_MXFP8, reason="MXFP8Tensor not available in this TE build")
    @pytest.mark.parametrize('stream_ckpt_dequant', [False, True])
    def test_mxfp8_save_load_content_equivalence(self, tmp_path_dist_ckpt, stream_ckpt_dequant):
        Utils.initialize_model_parallel(1, 1)

        # MXFP8 requires 2D with last-dim aligned to block size (32).
        fill_val = 0.25

        def get_mxfp8_tensor(val):
            return _to_mxfp8(torch.full((64, 128), val, dtype=torch.bfloat16, device='cuda'))

        def get_state_dict(val):
            return {
                'w': ShardedTensor.from_rank_offsets(
                    'w', get_mxfp8_tensor(val), replica_id=Utils.rank
                )
            }

        with TempNamedDir(tmp_path_dist_ckpt / f'mxfp8_eq_{stream_ckpt_dequant}') as ckpt_dir:
            save(get_state_dict(fill_val), ckpt_dir, TorchDistSaveShardedStrategy())

            sd_to_load = get_state_dict(99.0)
            strategy = TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream_ckpt_dequant)
            loaded = load(sd_to_load, ckpt_dir, strategy)
            loaded_w = loaded['w']
            if isinstance(loaded_w, QuantizedTensor):
                loaded_w = loaded_w.dequantize()
            # fill_val (0.25) is exactly representable in FP8 E4M3, and MXFP8 stores
            # block scales in E8M0 (power-of-2), so the per-block scale is exact and
            # every element encodes to the same FP8 code. Round-trip is near-lossless.
            torch.testing.assert_close(
                loaded_w,
                torch.full((64, 128), fill_val, dtype=torch.bfloat16, device='cuda'),
                rtol=1e-3,
                atol=1e-3,
            )

        Utils.destroy_model_parallel()

    # ---------------------------------------------------------------
    # NVFP4: round-trip under both paths. Same invariants as MXFP8 but
    # with NVFP4Tensor — validates that `is_float8tensor` (which binds to
    # QuantizedTensor under TE 2.x) correctly covers the FP4 path, that
    # NVFP4Tensor.view works inside _unwrap_pyt_sharded_tensor, and that
    # BF16->NVFP4 copy through QuantizedTensor.__torch_dispatch__ -> quantize_
    # produces correct values. Requires Blackwell+ for the FP4 kernels.
    # ---------------------------------------------------------------

    @pytest.mark.skipif(
        not HAVE_NVFP4_HW,
        reason="NVFP4 requires TransformerEngine NVFP4Tensor and Blackwell+ (arch 10+)",
    )
    @pytest.mark.parametrize('stream_ckpt_dequant', [False, True])
    def test_nvfp4_save_load_content_equivalence(self, tmp_path_dist_ckpt, stream_ckpt_dequant):
        Utils.initialize_model_parallel(1, 1)

        # NVFP4BlockScaling uses 16-element blocks along the last dim; use a
        # shape that's a multiple of both common block sizes.
        fill_val = 0.25

        def get_nvfp4_tensor(val):
            return _to_nvfp4(torch.full((64, 128), val, dtype=torch.bfloat16, device='cuda'))

        def get_state_dict(val):
            return {
                'w': ShardedTensor.from_rank_offsets(
                    'w', get_nvfp4_tensor(val), replica_id=Utils.rank
                )
            }

        with TempNamedDir(tmp_path_dist_ckpt / f'nvfp4_eq_{stream_ckpt_dequant}') as ckpt_dir:
            save(get_state_dict(fill_val), ckpt_dir, TorchDistSaveShardedStrategy())

            sd_to_load = get_state_dict(99.0)
            strategy = TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream_ckpt_dequant)
            loaded = load(sd_to_load, ckpt_dir, strategy)
            loaded_w = loaded['w']
            if isinstance(loaded_w, QuantizedTensor):
                loaded_w = loaded_w.dequantize()
            # For a constant block every FP4 code is identical and the dominant error
            # source is the per-block scale being stored in FP8 E4M3 (unlike MXFP8's
            # power-of-2 E8M0). That rounding is bounded below ~1% relative; 1e-2 is
            # tight enough to catch real bugs and loose enough to absorb E4M3 scale
            # rounding + bf16 output rounding.
            torch.testing.assert_close(
                loaded_w,
                torch.full((64, 128), fill_val, dtype=torch.bfloat16, device='cuda'),
                rtol=1e-2,
                atol=1e-2,
            )

        Utils.destroy_model_parallel()

    # ---------------------------------------------------------------
    # Corner cases
    # ---------------------------------------------------------------

    @pytest.mark.parametrize('stream_ckpt_dequant', [False, True])
    def test_plain_tensor_untouched_by_streaming_path(
        self, tmp_path_dist_ckpt, stream_ckpt_dequant
    ):
        """Non-quantized tensors in the state dict must round-trip losslessly under either path."""
        Utils.initialize_model_parallel(1, 1)

        src = torch.arange(64, dtype=torch.bfloat16, device='cuda')
        sd_to_save = {'w': ShardedTensor.from_rank_offsets('w', src.clone(), replica_id=Utils.rank)}

        with TempNamedDir(tmp_path_dist_ckpt / f'plain_{stream_ckpt_dequant}') as ckpt_dir:
            save(sd_to_save, ckpt_dir, TorchDistSaveShardedStrategy())

            dst = {
                'w': ShardedTensor.from_rank_offsets(
                    'w', torch.zeros_like(src), replica_id=Utils.rank
                )
            }
            strategy = TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream_ckpt_dequant)
            loaded = load(dst, ckpt_dir, strategy)
            # Plain BF16 must round-trip exactly.
            torch.testing.assert_close(loaded['w'], src)

        Utils.destroy_model_parallel()

    def test_default_is_off(self):
        """The default for stream_ckpt_dequant must be False (old behaviour preserved)."""
        strat = TorchDistLoadShardedStrategy()
        assert (
            strat.stream_ckpt_dequant is False
        ), "Default must be False so users opt-in explicitly."
        planner = MCoreLoadPlanner()
        assert planner.stream_ckpt_dequant is False

    def test_planner_state_cleanup_after_load(self, tmp_path_dist_ckpt):
        """``_intermediate_read_items`` must be empty after a streaming load completes.

        Lingering entries would indicate a scratch tensor we forgot to drop, defeating
        the memory win.
        """
        Utils.initialize_model_parallel(1, 1)

        def get_fp8_tensor(val):
            return _to_float8(torch.full((32,), val, dtype=torch.bfloat16, device='cuda'))

        sd_to_save = {
            f'w{i}': ShardedTensor.from_rank_offsets(
                f'w{i}', get_fp8_tensor(0.125), replica_id=Utils.rank
            )
            for i in range(4)
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'planner_cleanup') as ckpt_dir:
            save(sd_to_save, ckpt_dir, TorchDistSaveShardedStrategy())

            # Instrument: intercept MCoreLoadPlanner to capture the live instance.
            captured: list[MCoreLoadPlanner] = []
            original_init = MCoreLoadPlanner.__init__

            def capturing_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                captured.append(self)

            MCoreLoadPlanner.__init__ = capturing_init  # type: ignore[assignment]
            try:
                dst = {
                    f'w{i}': ShardedTensor.from_rank_offsets(
                        f'w{i}', get_fp8_tensor(99.0), replica_id=Utils.rank
                    )
                    for i in range(4)
                }
                load(dst, ckpt_dir, TorchDistLoadShardedStrategy(stream_ckpt_dequant=True))
            finally:
                MCoreLoadPlanner.__init__ = original_init  # type: ignore[assignment]

            assert len(captured) == 1
            assert captured[0]._intermediate_read_items == {}, (
                f"Planner left intermediate state after load: "
                f"{list(captured[0]._intermediate_read_items.keys())}"
            )

        Utils.destroy_model_parallel()

    def test_streaming_flag_forwards_through_fpsl_wrapper(self):
        """FullyParallelLoadStrategyWrapper must surface the base strategy's flag."""
        from megatron.core.dist_checkpointing.strategies.fully_parallel import (
            FullyParallelLoadStrategyWrapper,
        )

        base_off = TorchDistLoadShardedStrategy(stream_ckpt_dequant=False)
        base_on = TorchDistLoadShardedStrategy(stream_ckpt_dequant=True)
        # parallelization_group left default -> GroupMember.WORLD; that's fine since
        # we're only reading the forwarded property, not calling load().
        wrapped_off = FullyParallelLoadStrategyWrapper(base_off)
        wrapped_on = FullyParallelLoadStrategyWrapper(base_on)
        assert wrapped_off.stream_ckpt_dequant is False
        assert wrapped_on.stream_ckpt_dequant is True
