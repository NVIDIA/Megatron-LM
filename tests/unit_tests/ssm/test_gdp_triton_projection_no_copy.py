# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDP projection equivalence when the Triton TE GEMM uses weight strides."""

import random

import pytest
import torch

from megatron.core.inference import batch_dimensions_utils
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines import test_gdp_cuda_graph_e2e as gdp_e2e
from tests.unit_tests.inference.engines.ssm_test_helpers import HAVE_GDP_DEPS
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor):
    """Compare raw bits, including the intentionally NaN-filled padding row."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="GDP requires fla + mamba_ssm + einops")
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
@pytest.mark.skipif(
    gdp_e2e.FLASH_ATTENTION_VERSION is None, reason="batch-invariant attention needs FA3 or FA4"
)
class TestGDPTritonProjectionNoCopy:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.fixture(autouse=True)
    def rounders(self, monkeypatch):
        monkeypatch.setattr(DynamicInferenceContext, "ROUNDER", gdp_e2e.ROUNDER, raising=False)
        monkeypatch.setattr(DynamicInferenceContext, "TOKEN_ROUNDER", gdp_e2e.ROUNDER)
        monkeypatch.setattr(DynamicInferenceContext, "REQUEST_ROUNDER", gdp_e2e.ROUNDER)
        monkeypatch.setattr(batch_dimensions_utils, "TOKEN_ROUNDER", gdp_e2e.ROUNDER)

    def setup_method(self, method):
        clear_nvte_env_vars()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    @staticmethod
    def _case():
        return gdp_e2e.TestGDPCudaGraphE2E()

    @torch.inference_mode()
    def test_mixer_output_and_cache_match_copying_gemm(self, monkeypatch):
        """A strided weight view gives the old copying path's exact bits."""
        case = self._case()
        with set_batch_invariant_mode(True, backend="triton"):
            model = case._create_model()
            mixer = next(m for m in model.modules() if isinstance(m, GatedDeltaProductMixer))
            weights = (mixer.in_proj.weight, mixer.out_proj.weight)
            storage_pointers = tuple(weight.untyped_storage().data_ptr() for weight in weights)
            engine = case._build_engine(model, num_cuda_graphs=None)
            prompts = case._create_prompts()
            for i, prompt in enumerate(prompts):
                engine._add_request(case._make_request(i, prompt))
            for _ in range(8):
                engine.step_modern()
                dims = engine.context.batch_dimensions
                if dims.prefill_req_count == 0 and dims.decode_req_count == len(prompts):
                    break
            else:
                pytest.fail("engine did not reach a full-batch decode step")

            context = engine.context
            row = mixer.layer_number - mixer.pp_layer_offset
            conv, ssm = context.mamba_states_cache(row)
            initial_conv, initial_ssm = conv.clone(), ssm.clone()
            indices = context.mamba_metadata.batch_indices_decode
            slots = conv.shape[0]
            torch.manual_seed(9204)
            indices[: len(prompts)] = torch.randperm(slots, device="cuda", dtype=torch.int32)[
                : len(prompts)
            ]
            indices[len(prompts) - 1] = -1
            indices[len(prompts) :].fill_(-1)
            hidden = torch.randn(
                context.padded_batch_dimensions.token_count,
                1,
                model.config.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            hidden[len(prompts) - 1] = float("nan")

            original_mm = bik.mm_batch_invariant

            def copying_mm(a, b):
                return original_mm(a, b.contiguous())

            # The old TE shim copied its transposed weight immediately before
            # mm_batch_invariant. Force exactly that copy for the reference.
            monkeypatch.setattr(bik, "mm_batch_invariant", copying_mm)
            expected, expected_bias = mixer.ssm_dynamic_inference(hidden, context)
            expected = expected.clone()
            expected_conv, expected_ssm = conv.clone(), ssm.clone()
            monkeypatch.setattr(bik, "mm_batch_invariant", original_mm)

            conv.copy_(initial_conv)
            ssm.copy_(initial_ssm)
            actual, actual_bias = mixer.ssm_dynamic_inference(hidden, context)
            assert torch.isfinite(actual[: len(prompts) - 1]).all()
            _assert_bitwise_equal(actual, expected)
            if expected_bias is None:
                assert actual_bias is None
            else:
                _assert_bitwise_equal(actual_bias, expected_bias)
            _assert_bitwise_equal(conv, expected_conv)
            _assert_bitwise_equal(ssm, expected_ssm)
            assert (
                tuple(weight.untyped_storage().data_ptr() for weight in weights) == storage_pointers
            )
            assert all(weight.stride(1) == 1 for weight in weights)

    @torch.inference_mode()
    def test_fresh_model_graph_first_matches_eager_tokens(self):
        """Capture must be correct before a model has ever run an eager engine."""
        case = self._case()
        with set_batch_invariant_mode(True, backend="triton"):
            graph_model = case._create_model()
            prompts = case._create_prompts()
            graph_engine = case._build_engine(graph_model, num_cuda_graphs=gdp_e2e.NUM_CUDA_GRAPHS)
            graph_tokens, graph_log = case._run(graph_engine, prompts, stagger=False)

            eager_model = case._create_model()
            eager_engine = case._build_engine(eager_model, num_cuda_graphs=None)
            eager_tokens, eager_log = case._run(eager_engine, prompts, stagger=False)

        assert graph_tokens == eager_tokens
        assert set(graph_tokens) == set(range(len(prompts)))
        assert any(graph for _, _, graph in graph_log)
        assert not any(graph for _, _, graph in eager_log)
