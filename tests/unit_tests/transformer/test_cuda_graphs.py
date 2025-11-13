# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import random
import sys
import time
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_block import MambaStack
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version, is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils


class TestParallelTransformerBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)

        # initialize transformer model
        num_layers = 8
        hidden_size = 64
        self.transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_layer_with_transformer_engine_spec()
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_transformer_block = self.parallel_transformer_block
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_transformer_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_transformer_block.layers[0], "cudagraph_manager")
            assert (
                len(parallel_transformer_block.layers[0].cudagraph_manager.cudagraph_runners) == 1
            )
            del (
                parallel_transformer_block.layers[_]
                .cudagraph_manager.cudagraph_runners[0]
                .fwd_graph
            )


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
@pytest.mark.parametrize(
    "total_num_layers, pp, vpp, account_for_embedding_in_pipeline_split, account_for_loss_in_pipeline_split, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage, pp_layout, first_layer_numbers_golden, last_layer_numbers_golden",
    [
        (4, 1, None, False, False, None, None, None, [1], [4]),
        (8, 2, None, False, False, None, None, None, [1, 5], [4, 8]),
        (8, 2, 2, False, False, None, None, None, [1, 3, 5, 7], [2, 4, 6, 8]),
        (14, 4, None, True, True, None, None, None, [1, 4, 8, 12], [3, 7, 11, 14]),
        (
            14,
            4,
            2,
            True,
            True,
            None,
            None,
            None,
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
        (12, 4, None, False, False, 2, 2, None, [1, 3, 7, 11], [2, 6, 10, 12]),
        (
            12,
            4,
            2,
            False,
            False,
            2,
            2,
            None,
            [1, 2, 4, 6, 7, 8, 10, 12],
            [1, 3, 5, 6, 7, 9, 11, 12],
        ),
        (
            14,
            4,
            2,
            False,
            False,
            None,
            None,
            [
                ["embedding", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "loss"],
            ],
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
    ],
)
def test_cuda_graph_determine_first_last_layer_logic(
    total_num_layers,
    pp,
    vpp,
    account_for_embedding_in_pipeline_split,
    account_for_loss_in_pipeline_split,
    num_layers_in_first_pipeline_stage,
    num_layers_in_last_pipeline_stage,
    pp_layout,
    first_layer_numbers_golden,
    last_layer_numbers_golden,
):
    # Initialize RNG tracker
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    # Initialize parallel state
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp, virtual_pipeline_model_parallel_size=vpp
    )

    # initialize model
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    hidden_size = 128
    transformer_config = TransformerConfig(
        num_layers=total_num_layers,
        hidden_size=hidden_size,
        num_attention_heads=1,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_size=pp,
        deallocate_pipeline_outputs=True,
        cuda_graph_impl="local",
        use_te_rng_tracker=True,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        pipeline_model_parallel_layout=pp_layout,
    )
    model = []
    for i in range(vpp or 1):
        this_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=1024,
            position_embedding_type="rope",
            vp_stage=i,
        ).cuda()
        model.append(this_model)

    # create runner by running a fake forward pass
    sequence_length, micro_batch_size = 32, 1
    hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size)).cuda()
    attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
    for m in model:
        _ = m(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=hidden_states,
        )

    # Check if cuda graph is correctly setting is first/last layer
    for m in model:
        for l in m.decoder.layers:
            assert hasattr(l, "cudagraph_manager")
            assert (
                len(l.cudagraph_manager.cudagraph_runners) == 1
            ), "Cuda graph runner should be created"
            runner = l.cudagraph_manager.cudagraph_runners[0]
            assert runner.is_first_layer is not None and runner.is_last_layer is not None
            assert runner.is_first_layer == (l.layer_number in first_layer_numbers_golden)
            assert runner.is_last_layer == (l.layer_number in last_layer_numbers_golden)

            del l.cudagraph_manager.cudagraph_runners[0].fwd_graph

    # Destroy all captured graphs deterministically
    for m in model:
        for l in m.decoder.layers:
            for runner in getattr(l.cudagraph_manager, "cudagraph_runners", []):
                # Safely delete both graphs if present
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph

    # Ensure all pending work is complete and graph destruction runs now
    torch.cuda.synchronize()

    # Teardown
    Utils.destroy_model_parallel()
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    CudaGraphManager.global_mempool = None
    CudaGraphManager.fwd_mempools = None
    CudaGraphManager.bwd_mempools = None


class TestLLaVACudaGraph:
    """Test CUDA graphs with LLaVA model focusing on is_last_layer logic for encoder/decoder transitions."""

    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        )
        model_parallel_cuda_manual_seed(123)

        from copy import deepcopy

        from megatron.core.models.multimodal.llava_model import LLaVAModel
        from megatron.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_transformer_engine_spec,
        )

        # Create language transformer config with CUDA graphs enabled
        self.language_hidden_size = 64
        self.language_num_attention_heads = 4
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs
        )

        # Create vision transformer config
        vision_config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs for vision model too
        )

        # Create vision projection config
        vision_projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )

        # Get layer specs
        language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(language_layer_spec.submodules.mlp.submodules)

        # Set vision model type
        vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        # Create LLaVA model with both encoder and decoder
        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
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

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_llava_cudagraph_is_last_layer_logic(self):
        """Test that is_last_layer logic correctly resets prev_bwd_hidden_state_inputgrad for LLaVA models."""

        # Move model to CUDA
        self.llava_model.cuda()

        set_current_microbatch(self.llava_model.vision_model, 1)
        set_current_microbatch(self.llava_model.language_model, 1)

        # Create test inputs
        batch_size = 2
        seq_length = 1024
        num_images = 1

        images = torch.ones((num_images, 3, 336, 336), dtype=torch.float32).cuda()

        # Create text input with image tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        # Insert image token (using default image token index)
        input_ids[0, 5] = self.llava_model.image_token_index

        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = None

        # Create labels and loss mask for training
        labels = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        loss_mask = torch.ones((batch_size, seq_length), dtype=torch.float32).cuda()

        # Create num_image_tiles
        num_image_tiles = torch.ones(num_images, dtype=torch.int).cuda()

        # First forward pass - this should record the CUDA graphs
        output1, loss_mask1 = self.llava_model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )

        # Verify that CUDA graph managers were created
        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Vision model layers should have CUDA graph managers"

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Language model layers should have CUDA graph managers"

                    # Verify that CUDA graphs were created successfully
                    for runner in layer.cudagraph_manager.cudagraph_runners:
                        assert hasattr(runner, 'fwd_graph')
                        assert hasattr(runner, 'bwd_graph')

        # Perform backward pass to trigger backward graph recording
        if isinstance(output1, tuple):
            loss = output1[0].sum()
        else:
            loss = output1.sum()
        loss.backward()

        # Import the CUDA graph creation function
        from megatron.core.transformer.cuda_graphs import create_cudagraphs

        # Create the CUDA graphs - this is where the is_last_layer logic is tested
        create_cudagraphs()

        # Verify that CUDA graphs were created successfully
        assert _CudagraphGlobalRecord.cudagraph_created, "CUDA graphs should be created"

        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph


class TestParallelMambaBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

        # Ensure that this test is capturing to a fresh memory pool.
        CudaGraphManager.global_mempool = None

        def get_pg_collection():
            return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

        def get_mamba_block(hybrid_override_pattern):
            transformer_config = TransformerConfig(
                hidden_size=256,  # The Mamba layer places several constraints on this
                # Need to specify num_attention_heads and num_layers or TransformerConfig
                # will generate errors.
                num_layers=len(hybrid_override_pattern),
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl="local",
            )
            modules = mamba_stack_spec.submodules
            return MambaStack(
                transformer_config,
                modules,
                hybrid_override_pattern=hybrid_override_pattern,
                pg_collection=get_pg_collection(),
            )

        self.mamba_block = get_mamba_block(hybrid_override_pattern="M-M*-")
        self.transformer_config = self.mamba_block.config

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_mamba_block = self.mamba_block
        parallel_mamba_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_mamba_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_mamba_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_mamba_block.layers[0], "cudagraph_manager")
            assert len(parallel_mamba_block.layers[0].cudagraph_manager.cudagraph_runners) == 1

            del parallel_mamba_block.layers[_].cudagraph_manager.cudagraph_runners[0].fwd_graph


class TestCaptureFreezeGC:

    def capture_cuda_graphs(self, cuda_graph_capture_freeze_gc: bool) -> None:
        """Capture multiple cuda graphs by initializing the `DynamicInferenceEngine`.

        The `DynamicInferenceEngine` is used here because it is currently (as of
        August 2025) one of the heaviest users of multiple cuda graphs, and so
        its setup tests a realistic use-case of multi-batch size cuda graphs.

        Args:
            cuda_graph_capture_freeze_gc (bool): Flag that determines whether to
                freeze garbage collection.
        """

        # Set freeze-gc environment variable.
        os.environ["CUDA_GRAPH_CAPTURE_FREEZE_GC"] = str(int(cuda_graph_capture_freeze_gc))

        # Configuration.
        random_seed = 123
        vocab_size = 100
        num_tokens_to_prompt = 128
        num_tokens_to_generate = 32
        max_sequence_length = num_tokens_to_prompt + num_tokens_to_generate
        num_cuda_graphs = 4

        # Rounder values.
        rounder = 4
        DynamicInferenceContext.ROUNDER = rounder  # For backwards compatibility
        DynamicInferenceContext.TOKEN_ROUNDER = rounder
        DynamicInferenceContext.REQUEST_ROUNDER = rounder

        # Random state.
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        model_parallel_cuda_manual_seed(
            seed=random_seed,
            inference_rng_tracker=True,
            use_cudagraphable_rng=False,
            force_reset_rng=True,
        )

        # Transformer config.
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=4,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,  # needed?
        )

        # Sampling params.
        sampling_params = SamplingParams(num_tokens_to_generate=num_tokens_to_generate)

        # GPT model.
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            parallel_output=True,
        ).cuda()

        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)

        model.eval()

        # Inference config.
        inference_config = InferenceWrapperConfig(
            hidden_size=transformer_config.hidden_size,
            inference_batch_times_seqlen_threshold=400,
            fp32_residual_connection=False,
            params_dtype=transformer_config.params_dtype,
            padded_vocab_size=vocab_size,
        )

        # Inference context.
        context = DynamicInferenceContext(
            params_dtype=transformer_config.params_dtype,
            num_layers=transformer_config.num_layers,
            kv_channels=transformer_config.kv_channels,
            num_attention_heads=transformer_config.num_query_groups,
            max_sequence_length=max_sequence_length,
            num_cuda_graphs=num_cuda_graphs,
            buffer_size_gb=20,
            buffer_guaranteed_fraction=0.05,
            block_size_tokens=256,
            buffer_overflow_factor=1.1,
            max_requests_override=512,
            max_tokens_override=8196,
            tensor_model_parallel_size=transformer_config.tensor_model_parallel_size,
        )

        # Inference model wrapper.
        inference_wrapped_model = GPTInferenceWrapper(model, inference_config, context)

        # Note: the following is taken from AbstractModelInferenceWrapper.prep_model_for_inference().
        inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )

        # Text generation controller.
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=types.SimpleNamespace(vocab_size=vocab_size),
        )

        # Inference engine.
        engine = DynamicInferenceEngine(
            text_generation_controller,
            context,
            termination_id=vocab_size - 1,
            random_seed=random_seed,
        )

        return engine.capture_stats

    @pytest.mark.flaky_in_dev  # Issue #2855
    @pytest.mark.experimental
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_capture_freeze_gc(self):
        """Test cuda graph capture while freezing the GC."""

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        # Run tests with GC freeze off/on.
        result_map = {}
        for freeze_gc in (False, True):

            # Reset global cuda graph state.
            _CudagraphGlobalRecord.cudagraph_created = False
            _CudagraphGlobalRecord.cudagraph_record = []
            CudaGraphManager.global_mempool = None

            # Capture multiple cuda graphs by initializing DynamicInferenceEngine.
            mem_stats_start = torch.cuda.memory_stats()
            time_start = time.time()
            internal_stats = self.capture_cuda_graphs(freeze_gc)
            time_end = time.time()
            mem_stats_end = torch.cuda.memory_stats()

            # Track local (external) stats, in addition to internal stats.
            external_stats = {
                "time": time_end - time_start,
                "allocated_bytes": (
                    mem_stats_end["allocated_bytes.all.current"]
                    - mem_stats_start["allocated_bytes.all.current"]
                ),
                "reserved_bytes": (
                    mem_stats_end["reserved_bytes.all.current"]
                    - mem_stats_start["reserved_bytes.all.current"]
                ),
            }

            # Record results.
            result_map[freeze_gc] = {"internal": internal_stats, "external": external_stats}

        # Extract results.
        freeze_off_results = result_map[False]
        freeze_on_results = result_map[True]
        print(
            "test capture | freeze off: internal %.3f, external %.3f."
            % (freeze_off_results["internal"]["time"], freeze_off_results["external"]["time"])
        )
        print(
            "test capture | freeze on:  internal %.3f, external %.3f."
            % (freeze_on_results["internal"]["time"], freeze_on_results["external"]["time"])
        )

        # Validate time and memory usage.
        assert freeze_on_results["internal"]["time"] < 0.3 * freeze_off_results["internal"]["time"]
        assert freeze_on_results["external"]["time"] < 0.3 * freeze_off_results["external"]["time"]
        assert (
            freeze_on_results["internal"]["allocated_bytes"]
            <= freeze_off_results["internal"]["allocated_bytes"]
        )
        assert (
            freeze_on_results["external"]["allocated_bytes"]
            <= freeze_off_results["external"]["allocated_bytes"]
        )
        assert (
            freeze_on_results["internal"]["reserved_bytes"]
            <= freeze_off_results["internal"]["reserved_bytes"]
        )
        assert (
            freeze_on_results["external"]["reserved_bytes"]
            <= freeze_off_results["external"]["reserved_bytes"]
        )


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


class TestPartialCudaGraph:
    """Test that CUDA graph outputs match non-CUDA graph outputs for various scopes."""

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        # Store original environment variable values
        self.original_env = {
            'CUDA_DEVICE_MAX_CONNECTIONS': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'),
            'NVTE_ALLOW_NONDETERMINISTIC_ALGO': os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO'),
        }
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'

    def teardown_method(self, method):
        # Restore original environment variable values
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(123)
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn()
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=True
            )
        else:
            mtp_block_spec = None
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            mtp_block_spec=mtp_block_spec,
        )

    def create_test_args(
        self, cuda_graph_impl, cuda_graph_scope, cuda_graph_warmup_steps, ep_size, **kwargs
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_cuda_graphs.py']
        args = parse_args()
        args.num_layers = 4
        args.mtp_num_layers = 1
        args.vocab_size = 1024
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.global_batch_size = self.micro_batch_size * 8
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = 2
        args.sequence_parallel = True
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.expert_model_parallel_size = ep_size
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = True
        args.position_embedding_type = "rope"
        args.rotary_percent = 1.0
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0

        # MoE settings
        args.num_experts = 4
        args.expert_model_parallel_size = ep_size
        args.moe_shared_expert_intermediate_size = 1024
        args.moe_layer_freq = "[0,0,1,1]"
        args.moe_permute_fusion = True
        args.moe_router_fusion = True
        args.moe_router_topk = 2

        # CUDA graph settings
        args.cuda_graph_impl = cuda_graph_impl
        args.cuda_graph_scope = cuda_graph_scope
        args.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        args.use_te_rng_tracker = cuda_graph_impl != "none"

        for key, value in kwargs.items():
            assert hasattr(args, key)
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_test_helper(
        self, ep_size, cuda_graph_impl, cuda_graph_scope, cuda_graph_warmup_steps, **kwargs
    ):
        """Test fp8_param with gpt_model."""
        args = self.create_test_args(
            cuda_graph_impl, cuda_graph_scope, cuda_graph_warmup_steps, ep_size, **kwargs
        )

        set_args(args)
        torch.manual_seed(123)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, expert_model_parallel_size=ep_size
        )

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size
        )

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        loss_list = []

        cuda_graph_helper = None
        if cuda_graph_impl == "transformer_engine":
            from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

            cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        for i in range(100):
            gpt_model[0].zero_grad_buffer()
            optimizer.zero_grad()

            # Capture CUDA graphs after warmup if helper is provided
            if cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                cuda_graph_helper.create_cudagraphs()

            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length

            # Verify gradients
            loss = output.mean()
            loss.backward()

            for param in gpt_model[0].parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

        return torch.tensor(loss_list)

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.14.0")),
        reason="Partial CUDA graph support requires TransformerEngine version >= 1.14.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    @pytest.mark.parametrize("moe_dropless_dispatcher", [False, True])
    @pytest.mark.parametrize("moe_dispatcher_type", ["alltoall", "deepep", "hybridep"])
    def test_moe_partial_cudagraph(self, ep_size, moe_dropless_dispatcher, moe_dispatcher_type):
        extra_kwargs = {}
        if moe_dispatcher_type == "deepep":
            if not is_deep_ep_available():
                pytest.skip("Deep EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"
        elif moe_dispatcher_type == "hybridep":
            if not is_hybrid_ep_available():
                pytest.skip("Hybrid EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "hybridep"
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type
        if not moe_dropless_dispatcher:
            if moe_dispatcher_type == "deepep":
                pytest.skip("Deep EP doesn't support drop&pad MoE")
            extra_kwargs["moe_expert_capacity_factor"] = 1.0
            extra_kwargs["moe_pad_expert_input_to_capacity"] = True

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_scope in [
            None,
            ["attn"],
            ["moe"],
            ["mlp", "moe_router"],
            ["attn", "mlp", "moe_router", "moe_preprocess"],
        ]:
            if moe_dropless_dispatcher and (cuda_graph_scope is None or "moe" in cuda_graph_scope):
                # Dropless MoE doesn't work with "moe" scope cudagraph. Skip.
                continue
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_scope,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref)


if __name__ == "__main__":

    test = TestParallelTransformerBlockCudagraphs()
    test.setup_method(method=None)
    test.test_gpu_cudagraph()
    test.teardown_method(method=None)

    llava_test = TestLLaVACudaGraph()
    llava_test.setup_method(method=None)
    llava_test.test_llava_cudagraph_is_last_layer_logic()
    llava_test.teardown_method(method=None)

    test = TestCaptureFreezeGC()
    test.test_capture_freeze_gc()

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_moe_partial_cudagraph(4, True, "alltoall")
    test.teardown_method(method=None)
