# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.vision_cuda_graphs import (
    HAVE_TE_GRAPHS,
    VisionTECudaGraphHelper,
    _vision_layer_is_graphable,
    _wrap_graph_for_vision,
    get_vision_cuda_graph_seq_length,
)
from megatron.core.utils import is_te_min_version
from megatron.core import parallel_state
from tests.unit_tests.test_utilities import Utils

TE_MIN_VERSION = "2.13.0"
_te_version_ok = HAVE_TE and is_te_min_version(TE_MIN_VERSION)
if not _te_version_ok and __name__ != "__main__":
    pytest.skip(
        f"Vision CUDA graph tests require TransformerEngine >= {TE_MIN_VERSION}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Tests for _vision_layer_is_graphable
# ---------------------------------------------------------------------------
class TestVisionLayerIsGraphable:
    def test_non_transformer_layer_returns_false(self):
        config = SimpleNamespace(cuda_graph_impl="transformer_engine")
        layer = torch.nn.Linear(4, 4)
        assert _vision_layer_is_graphable(layer, config) is False

    def test_wrong_cuda_graph_impl_returns_false(self):
        from megatron.core.transformer.transformer_layer import TransformerLayer

        config = SimpleNamespace(cuda_graph_impl="local")
        layer = MagicMock(spec=TransformerLayer)
        # isinstance check with MagicMock(spec=...) should pass
        assert _vision_layer_is_graphable(layer, config) is False

    def test_correct_config_with_transformer_layer(self):
        """Real TransformerLayer + cuda_graph_impl='transformer_engine' -> True."""
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
        )
        from megatron.core.transformer.transformer_block import TransformerBlock

        block = TransformerBlock(config, get_vit_layer_with_transformer_engine_spec())
        layer = block.layers[0]
        assert _vision_layer_is_graphable(layer, config) is True

        Utils.destroy_model_parallel()


# ---------------------------------------------------------------------------
# Tests for _wrap_graph_for_vision
# ---------------------------------------------------------------------------
class TestWrapGraphForVision:
    def test_filters_none_from_tuple(self):
        def fake_graph(*args, **kwargs):
            return (torch.tensor(1.0), None)

        wrapped = _wrap_graph_for_vision(fake_graph)
        result = wrapped()
        assert result == (torch.tensor(1.0),)

    def test_returns_non_tuple_unchanged(self):
        t = torch.tensor(42.0)

        def fake_graph(*args, **kwargs):
            return t

        wrapped = _wrap_graph_for_vision(fake_graph)
        result = wrapped()
        assert result is t

    def test_preserves_all_non_none(self):
        a, b = torch.tensor(1.0), torch.tensor(2.0)

        def fake_graph(*args, **kwargs):
            return (a, b)

        wrapped = _wrap_graph_for_vision(fake_graph)
        result = wrapped()
        assert result == (a, b)

    def test_all_none_returns_original(self):
        def fake_graph(*args, **kwargs):
            return (None, None)

        wrapped = _wrap_graph_for_vision(fake_graph)
        result = wrapped()
        # filtered is empty -> returns original tuple
        assert result == (None, None)

    def test_preserves_te_attributes(self):
        def fake_graph(*args, **kwargs):
            return (torch.tensor(1.0),)

        fake_graph.backward_dw = "bwd_dw_fn"
        fake_graph.reset = "reset_fn"

        wrapped = _wrap_graph_for_vision(fake_graph)
        assert wrapped.backward_dw == "bwd_dw_fn"
        assert wrapped.reset == "reset_fn"

    def test_missing_te_attributes_not_set(self):
        def fake_graph(*args, **kwargs):
            return (torch.tensor(1.0),)

        wrapped = _wrap_graph_for_vision(fake_graph)
        assert not hasattr(wrapped, 'backward_dw')
        assert not hasattr(wrapped, 'reset')


# ---------------------------------------------------------------------------
# Tests for get_vision_cuda_graph_seq_length
# ---------------------------------------------------------------------------
class TestGetVisionCudaGraphSeqLength:
    def test_explicit_max_seq_length(self):
        config = SimpleNamespace(max_vision_cuda_graph_seq_length=2048)
        assert get_vision_cuda_graph_seq_length(config) == 2048

    def test_explicit_max_seq_length_zero_falls_through(self):
        """max_vision_cuda_graph_seq_length=0 is falsy, should fall through."""
        config = SimpleNamespace(max_vision_cuda_graph_seq_length=0)
        assert get_vision_cuda_graph_seq_length(config, default_seq_length=999) == 999

    def test_num_position_embeddings_only(self):
        config = SimpleNamespace(num_position_embeddings=1024)
        assert get_vision_cuda_graph_seq_length(config) == 1024

    def test_num_position_embeddings_with_spatial_merge(self):
        config = SimpleNamespace(num_position_embeddings=1024, spatial_merge_size=2)
        # merge_factor = 2**2 = 4, seq = 1024 // 4 = 256
        assert get_vision_cuda_graph_seq_length(config) == 256

    def test_spatial_merge_size_3(self):
        config = SimpleNamespace(num_position_embeddings=900, spatial_merge_size=3)
        # merge_factor = 9, seq = 900 // 9 = 100
        assert get_vision_cuda_graph_seq_length(config) == 100

    def test_default_seq_length(self):
        config = SimpleNamespace()
        assert get_vision_cuda_graph_seq_length(config) == 4096

    def test_custom_default(self):
        config = SimpleNamespace()
        assert get_vision_cuda_graph_seq_length(config, default_seq_length=512) == 512

    def test_explicit_overrides_position_embeddings(self):
        config = SimpleNamespace(
            max_vision_cuda_graph_seq_length=8192, num_position_embeddings=1024
        )
        assert get_vision_cuda_graph_seq_length(config) == 8192


# ---------------------------------------------------------------------------
# Integration test for VisionTECudaGraphHelper with LLaVA model
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
class TestVisionTECudaGraphHelper:
    """Test VisionTECudaGraphHelper initialization, sample args, and graph lifecycle."""

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        )
        model_parallel_cuda_manual_seed(123)

        from megatron.core.models.multimodal.llava_model import LLaVAModel

        self.language_hidden_size = 64
        self.vision_hidden_size = 16
        self.vision_num_layers = 2

        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        self.vision_config = TransformerConfig(
            num_layers=self.vision_num_layers,
            hidden_size=self.vision_hidden_size,
            num_attention_heads=2,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            bf16=True,
            pipeline_dtype=torch.bfloat16,
        )

        vision_projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
        )

        language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(language_layer_spec.submodules.mlp.submodules)

        self.vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
        )
        self.llava_model.bfloat16()

        self.vision_seq_length = 576
        self.micro_batch_size = 2

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        gc.collect()

    def _make_helper(self, num_microbatches=1):
        return VisionTECudaGraphHelper(
            model=[self.llava_model],
            vision_config=self.vision_config,
            vision_seq_length=self.vision_seq_length,
            micro_batch_size=self.micro_batch_size,
            num_microbatches=num_microbatches,
        )

    # -- Initialization tests --

    def test_init_finds_vision_layers(self):
        helper = self._make_helper()
        assert helper.vision_model is not None, "Should find vision_model"
        assert helper.num_layers == self.vision_num_layers
        assert len(helper.callables) == self.vision_num_layers
        assert helper.graphs_created() is False

    def test_init_no_vision_model_warns(self):
        """When model has no vision_model attr, helper should degrade gracefully."""
        dummy_model = torch.nn.Linear(4, 4)
        helper = VisionTECudaGraphHelper(
            model=[dummy_model],
            vision_config=self.vision_config,
            vision_seq_length=self.vision_seq_length,
            micro_batch_size=self.micro_batch_size,
        )
        assert helper.vision_model is None
        assert len(helper.callables) == 0
        assert helper.graphs_created() is False

    # -- _get_sample_args tests --

    def test_get_sample_args_shapes(self):
        helper = self._make_helper(num_microbatches=1)
        sample_args, sample_kwargs_list = helper._get_sample_args()

        expected_count = self.vision_num_layers * 1  # layers * microbatches
        assert len(sample_args) == expected_count
        assert len(sample_kwargs_list) == expected_count

        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs_list)):
            assert isinstance(args_item, tuple), f"sample_args[{i}] should be tuple"
            assert len(args_item) == 1, f"sample_args[{i}] should have one element (hidden_states)"
            hs = args_item[0]
            assert hs.shape == (self.vision_seq_length, 1, self.vision_hidden_size), (
                f"Expected ({self.vision_seq_length}, 1, {self.vision_hidden_size}), "
                f"got {hs.shape}"
            )
            assert hs.dtype == torch.bfloat16
            assert hs.device.type == 'cuda'
            assert hs.requires_grad is True

    def test_get_sample_args_multi_microbatch(self):
        helper = self._make_helper(num_microbatches=3)
        sample_args, sample_kwargs_list = helper._get_sample_args()

        expected_count = self.vision_num_layers * 3
        assert len(sample_args) == expected_count
        assert len(sample_kwargs_list) == expected_count

    def test_get_sample_args_empty_when_no_callables(self):
        dummy_model = torch.nn.Linear(4, 4)
        helper = VisionTECudaGraphHelper(
            model=[dummy_model],
            vision_config=self.vision_config,
            vision_seq_length=self.vision_seq_length,
            micro_batch_size=self.micro_batch_size,
        )
        sample_args, sample_kwargs_list = helper._get_sample_args()
        assert sample_args == []
        assert sample_kwargs_list == {}

    # -- create_cudagraphs / delete_cuda_graphs lifecycle --

    @pytest.mark.skipif(
        not (HAVE_TE_GRAPHS and is_te_min_version("2.7.0")),
        reason="TE CUDA graph capture requires TransformerEngine >= 2.7.0",
    )
    def test_create_and_delete_cudagraphs(self):
        """Full lifecycle: create graphs, verify state, delete, verify cleanup."""
        self.llava_model.cuda()
        helper = self._make_helper(num_microbatches=1)

        assert not helper.graphs_created()

        helper.create_cudagraphs()
        assert helper.graphs_created()

        # Each vision layer should have cuda_graphs attached
        for layer in helper.callables:
            assert hasattr(layer, 'cuda_graphs'), "Layer should have cuda_graphs after capture"
            assert len(layer.cuda_graphs) == 1  # 1 microbatch

        # cudagraph_manager should have been removed during capture
        for layer in helper.callables:
            assert not hasattr(layer, 'cudagraph_manager'), (
                "cudagraph_manager should be removed before TE capture"
            )

        helper.delete_cuda_graphs()
        assert not helper.graphs_created()

        # cuda_graphs attribute should be cleaned up
        for layer in helper.callables:
            assert not hasattr(layer, 'cuda_graphs'), (
                "cuda_graphs should be removed after delete"
            )

    @pytest.mark.skipif(
        not (HAVE_TE_GRAPHS and is_te_min_version("2.7.0")),
        reason="TE CUDA graph capture requires TransformerEngine >= 2.7.0",
    )
    def test_create_cudagraphs_multi_microbatch(self):
        """Verify that graphs are created per-microbatch per-layer."""
        self.llava_model.cuda()
        num_mb = 2
        helper = self._make_helper(num_microbatches=num_mb)

        helper.create_cudagraphs()
        assert helper.graphs_created()

        for layer in helper.callables:
            assert hasattr(layer, 'cuda_graphs')
            # PP=1 collapses to 1 microbatch internally
            assert len(layer.cuda_graphs) == helper.num_microbatches

        helper.delete_cuda_graphs()

    def test_create_cudagraphs_no_callables_is_noop(self):
        """create_cudagraphs on empty helper should not crash."""
        dummy_model = torch.nn.Linear(4, 4)
        helper = VisionTECudaGraphHelper(
            model=[dummy_model],
            vision_config=self.vision_config,
            vision_seq_length=self.vision_seq_length,
            micro_batch_size=self.micro_batch_size,
        )
        helper.create_cudagraphs()
        assert not helper.graphs_created()

    def test_delete_cudagraphs_before_create_is_noop(self):
        """delete_cuda_graphs before creation should not crash."""
        helper = self._make_helper()
        helper.delete_cuda_graphs()
        assert not helper.graphs_created()

    # -- cuda_graph_set_manual_hooks --

    def test_set_manual_hooks_before_creation_is_noop(self):
        helper = self._make_helper()
        helper.cuda_graph_set_manual_hooks(make_forward_pre_hook_fn=lambda: None)
        # Should not raise


# ---------------------------------------------------------------------------
# Integration test with PP=2: vision encoder on first pipeline stage only
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
class TestVisionTECudaGraphHelperPP2:
    """Test VisionTECudaGraphHelper with PP=2.

    With pipeline_model_parallel_size=2 the LLaVA model is split so that the
    vision encoder lives exclusively on the first pipeline stage:
      - pp_rank 0: add_encoder=True,  pre_process=True,  post_process=False
      - pp_rank 1: add_encoder=False, pre_process=False, post_process=True

    This test verifies that:
      1. On stage 0 the helper finds and captures vision layers.
      2. On stage 1 the helper gracefully finds no vision layers.
      3. With PP>1, num_microbatches is NOT collapsed to 1.
    """

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=None,
        )
        model_parallel_cuda_manual_seed(123)

        from megatron.core.models.multimodal.llava_model import LLaVAModel

        self.language_hidden_size = 64
        self.vision_hidden_size = 16
        self.vision_num_layers = 2
        self.language_num_layers = 4

        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        is_first_stage = pp_rank == 0
        is_last_stage = pp_rank == (parallel_state.get_pipeline_model_parallel_world_size() - 1)

        language_config = TransformerConfig(
            num_layers=self.language_num_layers,
            hidden_size=self.language_hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_model_parallel_size=2,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
        )

        self.vision_config = TransformerConfig(
            num_layers=self.vision_num_layers,
            hidden_size=self.vision_hidden_size,
            num_attention_heads=2,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            bf16=True,
            pipeline_dtype=torch.bfloat16,
        )

        vision_projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
        )

        language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(language_layer_spec.submodules.mlp.submodules)

        self.vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        self.is_first_stage = is_first_stage
        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=self.vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
            pre_process=is_first_stage,
            post_process=is_last_stage,
            add_encoder=is_first_stage,
            add_decoder=True,
        )
        self.llava_model.bfloat16()

        self.vision_seq_length = 576
        self.micro_batch_size = 2

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        gc.collect()

    def _make_helper(self, num_microbatches=4):
        return VisionTECudaGraphHelper(
            model=[self.llava_model],
            vision_config=self.vision_config,
            vision_seq_length=self.vision_seq_length,
            micro_batch_size=self.micro_batch_size,
            num_microbatches=num_microbatches,
        )

    def test_pp2_first_stage_finds_vision_layers(self):
        """Stage 0 should discover all vision encoder layers."""
        if not self.is_first_stage:
            pytest.skip("This assertion is only for pp_rank 0")

        helper = self._make_helper(num_microbatches=4)
        assert helper.vision_model is not None
        assert helper.num_layers == self.vision_num_layers
        assert len(helper.callables) == self.vision_num_layers

    def test_pp2_last_stage_has_no_vision_layers(self):
        """Stage 1 should find no vision model (encoder lives on stage 0)."""
        if self.is_first_stage:
            pytest.skip("This assertion is only for pp_rank 1")

        helper = self._make_helper(num_microbatches=4)
        assert helper.vision_model is None
        assert len(helper.callables) == 0
        assert not helper.graphs_created()

    def test_pp2_num_microbatches_preserved(self):
        """With PP>1, num_microbatches should NOT be collapsed to 1."""
        if not self.is_first_stage:
            pytest.skip("Vision layers only on pp_rank 0")

        num_mb = 8
        helper = self._make_helper(num_microbatches=num_mb)
        # _get_sample_args generates layers * microbatches entries
        sample_args, sample_kwargs_list = helper._get_sample_args()
        expected_count = self.vision_num_layers * num_mb
        assert len(sample_args) == expected_count, (
            f"With PP>1, expected {expected_count} sample_args "
            f"(layers={self.vision_num_layers} * mb={num_mb}), got {len(sample_args)}"
        )

    @pytest.mark.skipif(
        not (HAVE_TE_GRAPHS and is_te_min_version("2.7.0")),
        reason="TE CUDA graph capture requires TransformerEngine >= 2.7.0",
    )
    def test_pp2_create_cudagraphs_first_stage(self):
        """On stage 0, CUDA graphs should be captured with the full pipeline order."""
        if not self.is_first_stage:
            pytest.skip("Vision layers only on pp_rank 0")

        self.llava_model.cuda()
        num_mb = 4
        helper = self._make_helper(num_microbatches=num_mb)

        assert not helper.graphs_created()

        helper.create_cudagraphs()
        assert helper.graphs_created()

        # num_microbatches should be preserved (PP>1 does not collapse)
        assert helper.num_microbatches == num_mb

        # Each layer should have one graph per microbatch
        for layer in helper.callables:
            assert hasattr(layer, 'cuda_graphs')
            assert len(layer.cuda_graphs) == num_mb, (
                f"Expected {num_mb} graphs per layer, got {len(layer.cuda_graphs)}"
            )

        # Cleanup
        helper.delete_cuda_graphs()
        assert not helper.graphs_created()
        for layer in helper.callables:
            assert not hasattr(layer, 'cuda_graphs')

    @pytest.mark.skipif(
        not (HAVE_TE_GRAPHS and is_te_min_version("2.7.0")),
        reason="TE CUDA graph capture requires TransformerEngine >= 2.7.0",
    )
    def test_pp2_create_cudagraphs_last_stage_noop(self):
        """On stage 1 (no vision model), create_cudagraphs should be a no-op."""
        if self.is_first_stage:
            pytest.skip("This assertion is only for pp_rank 1")

        helper = self._make_helper(num_microbatches=4)
        helper.create_cudagraphs()
        assert not helper.graphs_created()


if __name__ == "__main__":
    if not _te_version_ok:
        print(f"SKIPPED: Vision CUDA graph tests require TransformerEngine >= {TE_MIN_VERSION}")
        exit(0)

    from _pytest.outcomes import Skipped

    def run_test(test_obj, test_fn_name):
        """Run a test method, treating pytest.skip() as a non-error."""
        test_obj.setup_method(method=None)
        try:
            getattr(test_obj, test_fn_name)()
        except Skipped as e:
            print(f"  SKIPPED {test_fn_name}: {e}")
        finally:
            test_obj.teardown_method(method=None)

    # Quick smoke tests for pure functions
    t = TestWrapGraphForVision()
    t.test_filters_none_from_tuple()
    t.test_returns_non_tuple_unchanged()
    t.test_preserves_all_non_none()
    t.test_all_none_returns_original()
    t.test_preserves_te_attributes()
    t.test_missing_te_attributes_not_set()
    print("_wrap_graph_for_vision tests passed.")

    t2 = TestGetVisionCudaGraphSeqLength()
    t2.test_explicit_max_seq_length()
    t2.test_explicit_max_seq_length_zero_falls_through()
    t2.test_num_position_embeddings_only()
    t2.test_num_position_embeddings_with_spatial_merge()
    t2.test_spatial_merge_size_3()
    t2.test_default_seq_length()
    t2.test_custom_default()
    t2.test_explicit_overrides_position_embeddings()
    print("get_vision_cuda_graph_seq_length tests passed.")

    # Integration tests (require GPU + distributed init)
    t3 = TestVisionTECudaGraphHelper()
    run_test(t3, "test_init_finds_vision_layers")
    run_test(t3, "test_get_sample_args_shapes")
    run_test(t3, "test_create_and_delete_cudagraphs")
    print("TestVisionTECudaGraphHelper tests passed.")

    # PP=2 integration tests (require 2+ GPUs)
    if Utils.world_size >= 2:
        t4 = TestVisionTECudaGraphHelperPP2()
        run_test(t4, "test_pp2_first_stage_finds_vision_layers")
        run_test(t4, "test_pp2_last_stage_has_no_vision_layers")
        run_test(t4, "test_pp2_num_microbatches_preserved")
        run_test(t4, "test_pp2_create_cudagraphs_first_stage")
        run_test(t4, "test_pp2_create_cudagraphs_last_stage_noop")
        print("TestVisionTECudaGraphHelperPP2 tests passed.")
    else:
        print("SKIPPED TestVisionTECudaGraphHelperPP2 (requires 2+ GPUs)")

    print("All vision CUDA graph tests passed.")
