# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager, nullcontext
from functools import partial
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from megatron.core import fp8_utils, parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    HAVE_MXFP8_TENSOR = True
except ImportError:
    HAVE_MXFP8_TENSOR = False

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support

    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )

    fp8_available, reason_for_no_fp8 = check_fp8_support()
except ImportError:
    fp8_available, reason_for_no_fp8 = False, "FP8 requires Transformer Engine"

# MXFP8 needs Blackwell or newer.
mxfp8_available = HAVE_MXFP8_TENSOR and get_device_arch_version() >= 10
reason_for_no_mxfp8 = "MXFP8 requires Transformer Engine and device arch >= 10"


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    ("is_init", "config_values", "te_helper"),
    [
        (
            False,
            {"fp8": "hybrid", "fp4": None, "fp8_param": False, "fp4_param": False},
            "fp8_autocast",
        ),
        (True, {"fp8": None, "fp4": None, "fp8_param": True, "fp4_param": False}, "fp8_model_init"),
    ],
)
def test_get_fp8_disabled_context_uses_disabled_te_context(is_init, config_values, te_helper):
    config = Mock(**config_values)
    disabled_context = Mock()

    with patch.object(
        fp8_utils.transformer_engine.pytorch, te_helper, return_value=disabled_context
    ) as te_context:
        result = fp8_utils.get_fp8_disabled_context(config, is_init=is_init)

    assert result is disabled_context
    te_context.assert_called_once_with(enabled=False)


class MockTELinear(nn.Module):
    """Mock TE Linear module for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t()


class TestFP8Padding:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        # Clear the wrapped modules set before each test
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def test_prepare_model_for_fp8_inference_basic(self):
        """Test prepare_model_for_fp8_inference wraps TE modules."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.te_layer = MockTELinear(128, 128)
                self.regular_layer = nn.Linear(128, 128)

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding'),
            patch.object(fp8_utils, 'Fp8Unpadding'),
            patch.object(fp8_utils, 'TE_LINEAR_TYPES', (MockTELinear,)),
        ):

            model = SimpleModel()
            original_te_forward = model.te_layer.forward
            original_regular_forward = model.regular_layer.forward

            # Prepare model
            prepared_model = fp8_utils.prepare_model_for_fp8_inference(model)

            # Check same model returned
            assert prepared_model is model

            # Check TE layer was wrapped
            assert model.te_layer.forward != original_te_forward
            assert model.te_layer in fp8_utils._fp8_inference_wrapped_modules

            # Check regular layer was not wrapped
            assert model.regular_layer.forward == original_regular_forward

    def test_padding_mechanism_works(self):
        """Test that the padding mechanism actually pads and unpads correctly."""

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding') as mock_pad_class,
            patch.object(fp8_utils, 'Fp8Unpadding') as mock_unpad_class,
        ):

            # Setup padding mock to pad from 6 to 16
            mock_pad_instance = Mock()
            mock_pad_instance.return_value = (torch.zeros(16, 8192), [16])
            mock_pad_class.return_value = mock_pad_instance

            # Setup unpadding mock to unpad from 16 to 6
            mock_unpad_instance = Mock()
            mock_unpad_instance.return_value = torch.zeros(6, 8192)
            mock_unpad_class.return_value = mock_unpad_instance

            # Create module and get access to padded_forward directly
            module = MockTELinear(4096, 4096)
            module.cuda()

            # Store original forward to track what it receives
            original_forward_input = None

            def track_forward(x):
                nonlocal original_forward_input
                original_forward_input = x
                return torch.randn(x.shape[0], x.shape[1], 4096).cuda()

            module.forward = track_forward

            # Manually create the wrapped forward function
            fp8_utils._wrap_te_linear_for_padding(module)
            padded_forward = module.forward

            # Mock FP8GlobalStateManager.is_fp8_enabled to return True
            with patch(
                'transformer_engine.pytorch.fp8.FP8GlobalStateManager.is_fp8_enabled',
                return_value=True,
            ):
                # Create input: (seq_len=6, batch=2, hidden=4096)
                input_tensor = torch.randn(6, 2, 4096).cuda()

                # Call padded_forward directly
                output = padded_forward(input_tensor)

            # Verify padding was called with correct reshaped input
            mock_pad_instance.assert_called_once()
            call_args = mock_pad_instance.call_args[0]
            assert call_args[0].shape == (6, 8192)  # Reshaped to 2D
            assert call_args[1] == [6]  # Split info

            # Verify the original forward received padded input with correct shape
            assert original_forward_input.shape == (16, 2, 4096)  # Padded to 16

            # Verify unpadding was called
            mock_unpad_instance.assert_called_once()
            unpad_args = mock_unpad_instance.call_args[0]
            assert unpad_args[0].shape == (16, 8192)  # Padded 2D tensor
            assert unpad_args[1] == [6]  # Original split

            # Verify output has original shape
            assert output.shape == (6, 2, 4096)  # Back to original seq_len


class _FakeSequenceParallelTELinear(nn.Module):
    """One rank of a TE linear with sequence parallelism and an identity GEMM. The other ranks
    contribute zeros to the reduce-scatter and copies of this rank's chunk to the all-gather."""

    def __init__(self, parallel_mode, tp_size, rank):
        super().__init__()
        self.parallel_mode, self.tp_size, self.rank = parallel_mode, tp_size, rank
        self.sequence_parallel = True

    def forward(self, x):
        if self.sequence_parallel and self.parallel_mode == "row":
            return x.chunk(self.tp_size)[self.rank], None
        if self.sequence_parallel and self.parallel_mode == "column":
            return torch.cat([x] * self.tp_size), None
        return x, None


@contextmanager
def _fake_fp8_padding(align):
    """Pure-torch Fp8Padding / Fp8Unpadding with a fixed alignment, inside an FP8 context."""

    def padded(splits):
        return [-(-m // align) * align for m in splits]

    class Padding:
        def __init__(self, num_gemms):
            self.num_gemms = num_gemms

        def __call__(self, inp, splits):
            assert len(splits) == self.num_gemms
            pads = [(0, 0, 0, p - m) for m, p in zip(splits, padded(splits))]
            chunks = [nn.functional.pad(c, pad) for c, pad in zip(inp.split(splits), pads)]
            return torch.cat(chunks), padded(splits)

    class Unpadding(Padding):
        def __call__(self, inp, splits):
            assert len(splits) == self.num_gemms
            return torch.cat([c[:m] for c, m in zip(inp.split(padded(splits)), splits)])

    with (
        patch.object(fp8_utils, "Fp8Padding", Padding),
        patch.object(fp8_utils, "Fp8Unpadding", Unpadding),
        patch.object(fp8_utils.FP8GlobalStateManager, "is_fp8_enabled", return_value=True),
    ):
        yield


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize("parallel_mode", ["row", "column", None])
def test_fp8_padding_pads_each_sequence_parallel_chunk(parallel_mode):
    """With sequence parallelism, each rank's chunk of the sequence is padded separately, as TE's
    parallel_mode splits it, so the last rank gets exactly its rows back."""
    tp_size, chunk_len, align = 4, 3, 4
    module = _FakeSequenceParallelTELinear(parallel_mode, tp_size, rank=tp_size - 1)
    seq_len = chunk_len if parallel_mode == "column" else chunk_len * tp_size
    x = torch.randn(seq_len, 2, 8)
    if parallel_mode == "row":
        expected = x[-chunk_len:]
    elif parallel_mode == "column":
        expected = torch.cat([x] * tp_size)
    else:
        expected = x

    with _fake_fp8_padding(align):
        fp8_utils._wrap_te_linear_for_padding(module)
        output, _ = module(x)

    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine is not installed")
def test_fp8_padding_rejects_sequence_not_divisible_by_tp():
    module = _FakeSequenceParallelTELinear("row", tp_size=4, rank=0)
    with _fake_fp8_padding(align=4), pytest.raises(AssertionError, match="not divisible"):
        fp8_utils._wrap_te_linear_for_padding(module)
        module(torch.randn(10, 2, 8))


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFP8PaddingSequenceParallel:
    """The padding wrapper on tensor-parallel TE linears with sequence parallelism.

    The module's own all-gather (column-parallel) or reduce-scatter (row-parallel) must stay the
    only communication, so the wrapped FP8 output equals the unwrapped one.
    """

    TP = 2
    BATCH = 2
    PARALLEL_MODES = ["row", "column", "layernorm_column"]
    # With the tensorwise recipe's 16-row alignment, the wrapper pads 40 tokens (20 per rank) and
    # leaves 32 tokens (16 per rank) as they are.
    SEQ_LENS = [pytest.param(32, id="aligned"), pytest.param(40, id="padded")]
    RECIPE_ALIGNMENT = 16
    # Padding alignments: the recipe's, and the 32 rows of MXFP8 (which itself needs Blackwell).
    ALIGNMENTS = [pytest.param(None, id="recipe_alignment"), pytest.param(32, id="mxfp8_alignment")]

    def setup_method(self, method):
        if Utils.world_size % self.TP != 0:
            pytest.skip(f"requires a world size divisible by TP={self.TP}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.TP)
        model_parallel_cuda_manual_seed(123)
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def _config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            tensor_model_parallel_size=self.TP,
            sequence_parallel=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            add_bias_linear=False,
            fp8="e4m3",
            fp8_recipe="tensorwise",
        )

    def _setup(self, config, parallel_mode, seq_len):
        """Build the TE linear from a full weight and return it with this rank's input and the
        expected output, computed in fp64 from the full operands."""
        tp_group = parallel_state.get_tensor_model_parallel_group()
        rank = tp_group.rank()
        in_features, out_features = (256, 128) if parallel_mode == "row" else (128, 256)
        generator = torch.Generator().manual_seed(seq_len)
        x = torch.randn(seq_len, self.BATCH, in_features, generator=generator).bfloat16()
        weight = (torch.randn(out_features, in_features, generator=generator) * 0.05).bfloat16()
        gemm_input = x.double()
        if parallel_mode == "layernorm_column":
            # TE initializes the norm's weight to one and its bias to zero.
            gemm_input = torch.nn.functional.layer_norm(
                gemm_input, (in_features,), eps=config.layernorm_epsilon
            )
        expected = torch.einsum("sbk,nk->sbn", gemm_input, weight.double())

        kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_group=tp_group,
        )
        if parallel_mode == "row":
            module = TERowParallelLinear(
                in_features, out_features, input_is_parallel=True, **kwargs
            )
            weight_shard = weight.chunk(self.TP, dim=1)[rank]
            x_local = x.chunk(self.TP, dim=2)[rank]
            expected = expected.chunk(self.TP, dim=0)[rank]
        else:
            if parallel_mode == "layernorm_column":
                linear_class = TELayerNormColumnParallelLinear
            else:
                linear_class = TEColumnParallelLinear
            module = linear_class(in_features, out_features, gather_output=False, **kwargs)
            weight_shard = weight.chunk(self.TP, dim=0)[rank]
            x_local = x.chunk(self.TP, dim=0)[rank]
            expected = expected.chunk(self.TP, dim=2)[rank]
        assert module.sequence_parallel
        with torch.no_grad():
            module.weight.copy_(weight_shard)
        return module.eval(), x_local.contiguous().cuda(), expected.cuda()

    @staticmethod
    def _forward(module, x, fp8_config=None):
        context = nullcontext() if fp8_config is None else fp8_utils.get_fp8_context(fp8_config)
        with torch.no_grad(), context:
            output, _ = module(x)
        return output

    @staticmethod
    def _rel_err(actual, expected):
        """Relative Frobenius error over the whole tensor-parallel group."""
        sums = torch.stack(
            [((actual.double() - expected.double()) ** 2).sum(), (expected.double() ** 2).sum()]
        )
        torch.distributed.all_reduce(sums, group=parallel_state.get_tensor_model_parallel_group())
        return (sums[0] / sums[1]).sqrt().item()

    @pytest.mark.parametrize("parallel_mode", PARALLEL_MODES)
    def test_wrapper_is_noop_outside_fp8(self, parallel_mode):
        module, x, expected = self._setup(self._config(), parallel_mode, seq_len=40)
        unwrapped = self._forward(module, x)
        fp8_utils.prepare_model_for_fp8_inference(module)
        wrapped = self._forward(module, x)

        torch.testing.assert_close(wrapped, unwrapped)
        torch.testing.assert_close(wrapped.double(), expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("parallel_mode", PARALLEL_MODES)
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    @pytest.mark.parametrize("align_size", ALIGNMENTS)
    def test_wrapped_output_matches_unwrapped(self, parallel_mode, seq_len, align_size):
        config = self._config()
        module, x, expected = self._setup(config, parallel_mode, seq_len)
        unwrapped = self._forward(module, x, config)

        inner_seq_lens = []
        inner_forward = module.forward

        def recording_forward(input_tensor, *args, **kwargs):
            inner_seq_lens.append(input_tensor.shape[0])
            return inner_forward(input_tensor, *args, **kwargs)

        module.forward = recording_forward
        with (
            patch.object(
                fp8_utils, "Fp8Padding", partial(fp8_utils.Fp8Padding, align_size=align_size)
            ),
            patch.object(
                fp8_utils, "Fp8Unpadding", partial(fp8_utils.Fp8Unpadding, align_size=align_size)
            ),
        ):
            fp8_utils.prepare_model_for_fp8_inference(module)
        wrapped = self._forward(module, x, config)

        if (seq_len // self.TP) % (align_size or self.RECIPE_ALIGNMENT) != 0:
            assert inner_seq_lens[0] > x.shape[0], "the wrapper did not pad the FP8 input"
        assert module.sequence_parallel
        assert wrapped.shape == unwrapped.shape == expected.shape
        # E4M3 quantization of both operands keeps the result within a few percent.
        assert self._rel_err(unwrapped, expected) < 0.15
        # Zero padding rows leave the FP8 scales unchanged, so only GEMM rounding differs. A second
        # reduction across the TP ranks would scale the output by TP (relative error TP - 1).
        assert self._rel_err(wrapped, unwrapped) < 0.05


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
class TestCopyTensorsToQuantizedParams:
    """Cover the batched MXFP8 param copy-back used by _post_param_sync.

    ``copy_tensors_to_quantized_params`` bypasses ``copy_`` and calls the destination quantizer
    directly, so the contract to protect is that it still writes exactly what the per-param
    ``copy_tensor_to_quantized_param`` would have written.
    """

    SHAPES = [(1024, 512), (2048, 256), (512, 1024)]

    def _make_param(self, shape):
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        tensor = quantizer.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        return torch.nn.Parameter(tensor, requires_grad=False)

    def _raw_buffers(self, param):
        """The four buffers MXFP8 storage is made of, i.e. everything a cast writes."""
        data = param.data
        return (
            data._rowwise_data,
            data._rowwise_scale_inv,
            data._columnwise_data,
            data._columnwise_scale_inv,
        )

    def test_matches_per_param_copy(self):
        """Batched copy-back is bitwise identical to copying one param at a time."""
        torch.manual_seed(0)
        reference_params = [self._make_param(shape) for shape in self.SHAPES]
        batched_params = [self._make_param(shape) for shape in self.SHAPES]
        # Sources are flat slices, matching how _post_param_sync views the param buffer.
        srcs = [
            torch.randn(shape, dtype=torch.bfloat16, device="cuda").view(-1)
            for shape in self.SHAPES
        ]

        for param, src in zip(reference_params, srcs):
            fp8_utils.copy_tensor_to_quantized_param(param, src)
        fp8_utils.copy_tensors_to_quantized_params(batched_params, srcs)
        torch.cuda.synchronize()

        for reference, batched in zip(reference_params, batched_params):
            for expected, actual in zip(self._raw_buffers(reference), self._raw_buffers(batched)):
                assert torch.equal(expected, actual)

    def test_falls_back_without_quantizer(self):
        """A destination with no quantizer of its own still gets written."""
        param = self._make_param(self.SHAPES[0])
        param.data._quantizer = None
        src = torch.randn(self.SHAPES[0], dtype=torch.bfloat16, device="cuda").view(-1)

        fp8_utils.copy_tensors_to_quantized_params([param], [src])
        torch.cuda.synchronize()

        # A quantized copy of a non-zero source cannot be all zeros.
        assert param.data._rowwise_data.any()

    def test_empty_input(self):
        """No params is a no-op rather than an error."""
        fp8_utils.copy_tensors_to_quantized_params([], [])
