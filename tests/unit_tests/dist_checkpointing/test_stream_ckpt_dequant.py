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
- ``_unwrap_pyt_sharded_tensor`` uses view-based axis stripping (no
  dequantize fallback) — exercised implicitly by the MXFP8 save/load test.
- MXFP8 save/load round-trip.
- NVFP4 save/load round-trip (Blackwell+ only, skipped otherwise).
- No-op fall-through for plain (non-quantized) tensors.
"""

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

# MXFP8 and NVFP4 require Blackwell (arch 10+).
HAVE_MXFP8_HW = HAVE_MXFP8 and _DEVICE_ARCH >= 10
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


@pytest.mark.skipif(not HAVE_TE, reason="TransformerEngine not available")
class TestStreamCkptDequant:
    """Unit tests for streaming per-tensor dequantize during ckpt load."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

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

    # ---------------------------------------------------------------
    # MXFP8: exercises both the streaming dequant AND the view-based
    # _unwrap_pyt_sharded_tensor fix (without it, ten[0] on MXFP8 OOMs).
    # ---------------------------------------------------------------

    @pytest.mark.skipif(
        not HAVE_MXFP8_HW,
        reason="MXFP8 requires TransformerEngine MXFP8Tensor and Blackwell+ (arch 10+)",
    )
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

    def test_default_is_on(self):
        """The default for stream_ckpt_dequant must be True (streaming path is now default)."""
        strat = TorchDistLoadShardedStrategy()
        assert (
            strat.stream_ckpt_dequant is True
        ), "Default must be True; users opt out via --no-stream-ckpt-dequant."
        planner = MCoreLoadPlanner()
        assert planner.stream_ckpt_dequant is True

    def test_planner_state_cleanup_after_load(self, tmp_path_dist_ckpt):
        """Per-target streaming state must be empty after a streaming load completes.

        Lingering entries in ``_intermediate_read_items`` (per-read-item) or
        ``_stream_targets`` (per-target BF16 buffer) would indicate a scratch
        tensor we forgot to drop, defeating the memory win.
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
            assert captured[0]._stream_targets == {}, (
                f"Planner left per-target streaming buffers after load: "
                f"{list(captured[0]._stream_targets.keys())}"
            )

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
