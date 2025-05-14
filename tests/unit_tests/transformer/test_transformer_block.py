# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.device_utils import get_current_device, get_current_device_type
import copy
from contextlib import nullcontext

import pytest
import torch
import pytest 
from packaging import version

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core import parallel_state
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed

try:
    import transformer_engine  # pylint: disable=unused-import
    HAVE_TE =True
except ImportError:
    HAVE_TE = False

class TestParallelTransformerBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_device_manual_seed(123)
        self.transformer_config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True)   
        layer_spec = get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        self.parallel_transformer_block = TransformerBlock(self.transformer_config,
                                                           layer_spec)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_transformer_block = self.parallel_transformer_block
        assert isinstance(parallel_transformer_block, TransformerBlock)
        num_weights = sum([p.numel() for p in parallel_transformer_block.parameters()])
        assert num_weights == 100096
        assert parallel_transformer_block.num_layers_per_pipeline_rank == 2
        assert len(parallel_transformer_block.layers) == 2
        layer_0: TransformerLayer = parallel_transformer_block._get_layer(0)
        assert layer_0.layer_number == 1
        layer_1: TransformerLayer = parallel_transformer_block._get_layer(1)
        assert layer_1.layer_number == 2

    def test_gpu_forward(self):
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config

        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.to(device=get_current_device())

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.to(device=get_current_device())

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).to(device=get_current_device())

        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_gpu_forward_full_checkpoint(self):
        self._run_full_checkpoint_test(fp8=None)

    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required for fp8")
    def test_gpu_forward_full_checkpoint_fp8(self):
        self._run_full_checkpoint_test(fp8="e4m3")

    def test_gpu_forward_selective_checkpoint(self):
        self._run_selective_checkpoint_test(fp8=None)

    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required for fp8")
    def test_gpu_forward_selective_checkpoint_fp8(self):
        self._run_selective_checkpoint_test(fp8="e4m3")

    def _run_full_checkpoint_test(self, fp8):
        transformer_config = self.transformer_config
        config = transformer_config
        config.recompute_granularity = 'full'
        config.recompute_method = 'block'
        config.fp8 = fp8
        config.recompute_num_layers = config.num_layers
        layer_spec = get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        full_transformer_block = TransformerBlock(
            config, layer_spec
        )
        assert full_transformer_block.config.recompute_granularity == 'full'
        assert full_transformer_block.config.recompute_method == 'block'
        assert full_transformer_block.config.fp8 == fp8

        sequence_length = 32
        micro_batch_size = 2
        full_transformer_block.to(device=get_current_device())

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.to(device=get_current_device())

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).to(device=get_current_device())

        hidden_states = full_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def _run_selective_checkpoint_test(self, fp8):
        transformer_config = self.transformer_config
        config = transformer_config
        config.recompute_granularity = 'selective'
        config.fp8 = fp8
        layer_spec = get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        selective_transformer_block = TransformerBlock(
            config, layer_spec
        )
        assert selective_transformer_block.config.recompute_granularity == 'selective'
        assert "core_attn" in selective_transformer_block.config.recompute_modules
        assert selective_transformer_block.checkpoint_core_attention
        assert selective_transformer_block.config.fp8 == fp8

        sequence_length = 32
        micro_batch_size = 2
        selective_transformer_block.to(device=get_current_device())

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.to(device=get_current_device())

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).to(device=get_current_device())

        hidden_states = selective_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size


class TestPipelineParallelTransformerBlock:
    @pytest.mark.skip("Test is broken")
    @pytest.mark.parametrize(
        "num_layers, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, "
        "account_for_embedding_in_pipeline_split, account_for_loss_in_pipeline_split, "
        "first_pipeline_num_layers, last_pipeline_num_layers, should_assert_error",
        [
            # Last pipeline stage has specified layers
            (60, 5, None, False, False, None, 4, False),
            # Uneven PP 6*[8]+[6]+[6]=60
            (60, 8, None, False, False, 6, 6, False),
            # Even PP
            (64, 4, None, False, False, None, None, False),
            # Even VPP
            (64, 4, 8, False, False, None, None, False),
            # First pipeline stage has specified layers
            # Should distribute remaining layers evenly among other stages
            (60, 6, None, False, False, 5, None, False),
            # Uneven distribution leading to assertion error
            (101, 8, None, False, False, 13, 13, True),
            # Include embedding in pipeline split without virtual PP
            (63, 4, None, True, False, None, None, False),
            # Include loss in pipeline split without virtual PP
            (63, 4, None, False, True, None, None, False),
            # Include embedding and loss in pipeline split without virtual PP
            (62, 4, None, True, True, None, None, False),
            # Include embedding and loss with virtual PP
            (62, 4, 2, True, True, None, None, False),
            # num_layers not divisible by pipeline size without embedding/loss
            (65, 4, None, False, False, None, None, True),
            # num_layers not divisible by pipeline size with embedding/loss
            (65, 4, None, True, True, None, None, True),
            # Uneven distribution with specified first pipeline layers causing error
            (61, 4, None, False, False, 12, None, True),
            # Too few layers for the number of pipeline stages
            (2, 4, None, False, False, None, None, True),
            # Uneven PP with embedding included (should assert per code)
            (60, 6, None, True, False, 5, 5, True),
            # Virtual PP where num_layers not divisible by total virtual stages
            (50, 2, 7, False, False, None, None, True),
            # Edge case where num_layers per virtual rank is zero
            (4, 4, 4, False, False, None, None, True),
        ],
    )
    def test_layer_builder(
        self,
        num_layers,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
        account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split,
        first_pipeline_num_layers,
        last_pipeline_num_layers,
        should_assert_error,
    ):
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        )
        context = (
            pytest.raises((AssertionError, ValueError)) if should_assert_error else nullcontext()
        )
        with context:
            transformer_config = TransformerConfig(
                num_layers=num_layers,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
                account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
                account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
                num_layers_in_first_pipeline_stage=first_pipeline_num_layers,
                num_layers_in_last_pipeline_stage=last_pipeline_num_layers,
                pipeline_dtype=torch.bfloat16,
                hidden_size=128,
                num_attention_heads=16,
            )
            total_build_layers = 0
            for i in range(pipeline_model_parallel_size):
                parallel_state.set_pipeline_model_parallel_rank(i)
                if virtual_pipeline_model_parallel_size is not None:
                    for j in range(virtual_pipeline_model_parallel_size):
                        parallel_state.set_virtual_pipeline_model_parallel_rank(j)
                        num_layers_to_build = get_num_layers_to_build(transformer_config)
                        total_build_layers += num_layers_to_build
                else:
                    num_layers_to_build = get_num_layers_to_build(transformer_config)
                    total_build_layers += num_layers_to_build
        if not should_assert_error:
            assert (
                total_build_layers == num_layers
            ), f"total build layers {total_build_layers} should be equal to num_layers {num_layers}"
        parallel_state.set_pipeline_model_parallel_world_size(None)
        parallel_state.set_virtual_pipeline_model_parallel_world_size(None)


class TestProcessGroupTransformerBlock:
    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "tp_size,cp_size,dp_size,use_custom_pg",
        [(2, 2, 2, True), (2, 4, 1, True), (2, 2, 2, False), (2, 4, 1, False)],
    )
    def test_pg_input_args(self, tp_size, cp_size, dp_size, use_custom_pg):
        """
        Test TransformerBlock with custom process groups.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_device_manual_seed(123)
        if use_custom_pg:
            # Create custom process groups
            device_mesh = torch.distributed.init_device_mesh(
                get_current_device_type(), (dp_size, tp_size, cp_size), mesh_dim_names=("dp", "tp", "cp")
            )
            # Get process groups from device mesh
            tp_group = device_mesh.get_group(mesh_dim="tp")
            cp_group = device_mesh.get_group(mesh_dim="cp")
            # Create ModelCommProcessGroups with custom process groups
            model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group)
        else:
            # Rely on TransformerBlock to create default process groups
            model_comm_pgs = None

        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        self.transformer_block = TransformerBlock(
            self.transformer_config,
            transformer_layer_spec,
            model_comm_pgs=model_comm_pgs,
        )
        self.transformer_block.to(get_current_device())

        sequence_length = 128
        micro_batch_size = 1

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.transformer_block.config.hidden_size),
            device=get_current_device(),
        )

        hidden_states = self.transformer_block(hidden_states=hidden_states, attention_mask=None)

        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == self.transformer_block.config.hidden_size


class TestMixedProcessGroups:
    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize("tp_size,cp_size", [(2, 4)])
    def test_mixed_pg_transformer_block(self, tp_size, cp_size, monkeypatch):
        """
        Test TransformerBlock with custom process groups.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_device_manual_seed(123)

        # Create a new build_layers method that uses interleaved attention
        def _build_layers_with_interleaved_attention(self):
            def build_layer(layer_spec, layer_number):
                fp8_init_context = get_fp8_context(self.config, layer_number - 1, is_init=True)
                if layer_number % 4 == 0:
                    config = self.local_attn_config
                    model_comm_pgs = self.local_pgs
                else:
                    config = self.config
                    model_comm_pgs = self.model_comm_pgs
                with fp8_init_context:
                    module = build_module(
                        layer_spec,
                        config=config,
                        layer_number=layer_number,
                        model_comm_pgs=model_comm_pgs,
                    )
                return module

            # Modify TransformerConfig and ModelCommProcessGroups for local attention
            self.local_attn_config = copy.deepcopy(self.config)
            self.local_pgs = ModelCommProcessGroups.use_mpu_process_groups()
            self.local_attn_config.context_parallel_size = 1
            self.local_pgs.cp = torch.distributed.new_group(ranks=[torch.distributed.get_rank()])

            # offset is implicit in TransformerLayer
            self.layers = torch.nn.ModuleList(
                [
                    build_layer(layer_spec, i + 1)
                    for i, layer_spec in enumerate(self.submodules.layer_specs)
                ]
            )

            # Copied from TransformerBlock.build_layers
            if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
                self.final_layernorm = build_module(
                    self.submodules.layer_norm,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.final_layernorm = None  # Either this or nn.Identity

        # Replace the default build_layers method
        monkeypatch.setattr(
            TransformerBlock, "_build_layers", _build_layers_with_interleaved_attention
        )

        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            context_parallel_size=cp_size if HAVE_TE else 1,
            bf16=True,
        )
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        self.transformer_block = TransformerBlock(
            self.transformer_config, transformer_layer_spec
        )
        self.transformer_block.to(get_current_device()).bfloat16()

        sequence_length = 128
        micro_batch_size = 1

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.transformer_block.config.hidden_size),
            dtype=torch.bfloat16,
            device=get_current_device(),
        )

        hidden_states = self.transformer_block(hidden_states=hidden_states, attention_mask=None)

        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == self.transformer_block.config.hidden_size
