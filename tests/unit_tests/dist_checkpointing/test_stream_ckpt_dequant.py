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

from contextlib import contextmanager
from typing import Optional

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
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

    HAVE_MXFP8 = True
except ImportError:
    HAVE_MXFP8 = False
    MXFP8Tensor = None  # type: ignore

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


@contextmanager
def _measure_peak_cuda(device: int = 0):
    """Reset peak, yield a dict, populate ``peak`` on exit."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = {}
    try:
        yield out
    finally:
        torch.cuda.synchronize(device)
        out["peak"] = torch.cuda.max_memory_allocated(device)


@pytest.mark.skipif(not HAVE_TE, reason="TransformerEngine not available")
class TestStreamCkptDequant:
    """Unit tests for streaming per-tensor dequantize during ckpt load."""

    # ---------------------------------------------------------------
    # Baseline: FP8 (delayed scaling) save/load equivalence + amax safety
    # ---------------------------------------------------------------

    @pytest.mark.parametrize('stream_ckpt_dequant', [False, True])
    def test_fp8_save_load_content_equivalence(
        self, tmp_path_dist_ckpt, stream_ckpt_dequant
    ):
        """Loaded FP8 contents must match regardless of which dequantize path is used."""
        Utils.initialize_model_parallel(1, 1)

        fill_val = 0.5

        def get_fp8_tensor(val):
            return _to_float8(torch.full((8,), val, dtype=torch.bfloat16, device='cuda'))

        def get_state_dict(val):
            return {
                'w': ShardedTensor.from_rank_offsets(
                    'w', get_fp8_tensor(val), replica_id=Utils.rank
                ),
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
            torch.testing.assert_close(
                loaded_w,
                torch.full((8,), fill_val, dtype=torch.bfloat16, device='cuda'),
                rtol=0.1, atol=0.1,  # FP8 quantization tolerance
            )

        Utils.destroy_model_parallel()

    def test_fp8_amax_history_not_polluted(self, tmp_path_dist_ckpt):
        """Delayed-scaling amax must be snapshotted & restored across a streaming load."""
        Utils.initialize_model_parallel(1, 1)

        def get_fp8_tensor(val):
            return _to_float8(torch.full((8,), val, dtype=torch.bfloat16, device='cuda'))

        sd_to_save = {
            'w': ShardedTensor.from_rank_offsets(
                'w', get_fp8_tensor(0.25), replica_id=Utils.rank
            ),
        }

        with TempNamedDir(tmp_path_dist_ckpt / 'fp8_amax') as ckpt_dir:
            save(sd_to_save, ckpt_dir, TorchDistSaveShardedStrategy())

            # Rebuild destination with a known (distinct) amax value we can check
            # survives the streaming load.
            dst = ShardedTensor.from_rank_offsets(
                'w', get_fp8_tensor(99.0), replica_id=Utils.rank
            )
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

        # 32 tensors * 4 MiB FP8 each = 128 MiB of FP8, 256 MiB of upfront BF16
        # under the legacy path; streaming should only keep ~8 MiB scratch live at once.
        num_tensors = 32
        numel = 4 * 1024 * 1024  # 4 Mi elements -> 4 MiB FP8, 8 MiB BF16

        def get_fp8_tensor():
            return _to_float8(torch.zeros(numel, dtype=torch.bfloat16, device='cuda'))

        def build_state_dict():
            return {
                f'w{i}': ShardedTensor.from_rank_offsets(
                    f'w{i}', get_fp8_tensor(), replica_id=Utils.rank
                )
                for i in range(num_tensors)
            }

        with TempNamedDir(tmp_path_dist_ckpt / 'fp8_mem') as ckpt_dir:
            save(build_state_dict(), ckpt_dir, TorchDistSaveShardedStrategy())

            # Legacy path
            dst_legacy = build_state_dict()
            with _measure_peak_cuda() as legacy:
                load(dst_legacy, ckpt_dir, TorchDistLoadShardedStrategy(stream_ckpt_dequant=False))
            # Free legacy state before measuring streaming
            del dst_legacy
            torch.cuda.empty_cache()

            # Streaming path
            dst_stream = build_state_dict()
            with _measure_peak_cuda() as streaming:
                load(dst_stream, ckpt_dir, TorchDistLoadShardedStrategy(stream_ckpt_dequant=True))

            legacy_peak = legacy["peak"]
            streaming_peak = streaming["peak"]
            # Streaming must save at least half of the bulk BF16 allocation.
            expected_min_saving = (num_tensors - 1) * numel * 2 // 2  # 2 bytes/elem BF16
            saving = legacy_peak - streaming_peak
            assert saving >= expected_min_saving, (
                f"Streaming did not reduce peak as expected. "
                f"legacy_peak={legacy_peak/1e6:.1f} MB, "
                f"streaming_peak={streaming_peak/1e6:.1f} MB, "
                f"saving={saving/1e6:.1f} MB, "
                f"expected_min_saving={expected_min_saving/1e6:.1f} MB"
            )

        Utils.destroy_model_parallel()

    # ---------------------------------------------------------------
    # MXFP8: exercises both the streaming dequant AND the view-based
    # _unwrap_pyt_sharded_tensor fix (without it, ten[0] on MXFP8 OOMs).
    # ---------------------------------------------------------------

    @pytest.mark.skipif(not HAVE_MXFP8, reason="MXFP8Tensor not available in this TE build")
    @pytest.mark.parametrize('stream_ckpt_dequant', [False, True])
    def test_mxfp8_save_load_content_equivalence(
        self, tmp_path_dist_ckpt, stream_ckpt_dequant
    ):
        Utils.initialize_model_parallel(1, 1)

        # MXFP8 requires 2D with last-dim aligned to block size (32).
        fill_val = 0.25

        def get_mxfp8_tensor(val):
            return _to_mxfp8(torch.full((64, 128), val, dtype=torch.bfloat16, device='cuda'))

        def get_state_dict(val):
            return {
                'w': ShardedTensor.from_rank_offsets(
                    'w', get_mxfp8_tensor(val), replica_id=Utils.rank
                ),
            }

        with TempNamedDir(tmp_path_dist_ckpt / f'mxfp8_eq_{stream_ckpt_dequant}') as ckpt_dir:
            save(get_state_dict(fill_val), ckpt_dir, TorchDistSaveShardedStrategy())

            sd_to_load = get_state_dict(99.0)
            strategy = TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream_ckpt_dequant)
            loaded = load(sd_to_load, ckpt_dir, strategy)
            loaded_w = loaded['w']
            if isinstance(loaded_w, QuantizedTensor):
                loaded_w = loaded_w.dequantize()
            torch.testing.assert_close(
                loaded_w,
                torch.full((64, 128), fill_val, dtype=torch.bfloat16, device='cuda'),
                rtol=0.1, atol=0.1,
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
        sd_to_save = {
            'w': ShardedTensor.from_rank_offsets('w', src.clone(), replica_id=Utils.rank),
        }

        with TempNamedDir(tmp_path_dist_ckpt / f'plain_{stream_ckpt_dequant}') as ckpt_dir:
            save(sd_to_save, ckpt_dir, TorchDistSaveShardedStrategy())

            dst = {
                'w': ShardedTensor.from_rank_offsets(
                    'w', torch.zeros_like(src), replica_id=Utils.rank
                ),
            }
            strategy = TorchDistLoadShardedStrategy(stream_ckpt_dequant=stream_ckpt_dequant)
            loaded = load(dst, ckpt_dir, strategy)
            # Plain BF16 must round-trip exactly.
            torch.testing.assert_close(loaded['w'], src)

        Utils.destroy_model_parallel()

    def test_default_is_off(self):
        """The default for stream_ckpt_dequant must be False (old behaviour preserved)."""
        strat = TorchDistLoadShardedStrategy()
        assert strat.stream_ckpt_dequant is False, (
            "Default must be False so users opt-in explicitly."
        )
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
