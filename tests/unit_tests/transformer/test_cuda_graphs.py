# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

import megatron.core.transformer.cuda_graphs as cuda_graphs_module
import megatron.core.transformer.moe.paged_stash as paged_stash_module
import megatron.core.transformer.transformer_config as transformer_config_module
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graph_config import validate_moe_cuda_graph_support
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    TECudaGraphHelper,
    _CudagraphGlobalRecord,
    _layer_is_graphable,
)
from megatron.core.transformer.enums import CudaGraphModule, CudaGraphScope, InferenceCudaGraphScope
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import is_fa_min_version, is_te_min_version
from megatron.training import arguments as training_arguments
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

fp8_available, _ = check_fp8_support()


def _base_cuda_graph_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kwargs)


def _te_whole_moe_paged_stash_config(**overrides) -> TransformerConfig:
    kwargs = {
        "cuda_graph_impl": "transformer_engine",
        "cuda_graph_modules": [CudaGraphModule.moe],
        "num_moe_experts": 4,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "hybridep",
        "moe_expert_rank_capacity_factor": 1.2,
        "moe_paged_stash": True,
        "use_transformer_engine_op_fuser": True,
    }
    kwargs.update(overrides)
    return _base_cuda_graph_config(**kwargs)


def _validated_cuda_graph_cli_args(monkeypatch, cli_args=None, **overrides):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    warning_messages = []
    print_messages = []

    monkeypatch.setattr(
        training_arguments, "warn_rank_0", lambda msg, *args, **kwargs: warning_messages.append(msg)
    )
    monkeypatch.setattr(
        training_arguments, "print_rank_0", lambda msg, *args, **kwargs: print_messages.append(msg)
    )
    monkeypatch.setattr(sys, "argv", ["test_cuda_graphs.py", *(cli_args or [])])

    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1

    for key, value in overrides.items():
        setattr(args, key, value)

    args = validate_args(args)
    return args, warning_messages, print_messages


class TestCudaGraphConfigAndArguments:
    def test_local_impl_defaults_to_layer_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='local')
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_local_impl_allows_expert_activation_offload_scope(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
            fine_grained_activation_offloading=True,
            offload_modules=['expert_fc1', 'moe_act'],
            num_moe_experts=4,
        )

        assert cfg.cuda_graph_impl == 'local'
        assert CudaGraphModule.attn in cfg.cuda_graph_modules
        assert CudaGraphModule.moe_router in cfg.cuda_graph_modules
        assert CudaGraphModule.moe_preprocess in cfg.cuda_graph_modules

    def test_local_impl_rejects_unsupported_activation_offload_scope(self):
        with pytest.raises(
            AssertionError,
            match=(
                "fine-grained activation offloading with cuda_graph_impl='local'.*"
                "Unsupported offload_modules: \\['qkv_linear'\\]"
            ),
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[CudaGraphModule.attn],
                fine_grained_activation_offloading=True,
                offload_modules=['qkv_linear'],
            )

    def test_local_impl_rejects_full_layer_graph_with_activation_offload(self):
        with pytest.raises(
            AssertionError, match="not supported with whole-layer CUDA graph capture"
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[],
                fine_grained_activation_offloading=True,
                offload_modules=['expert_fc1'],
            )

    def test_local_impl_rejects_moe_router_graph_with_mlp_norm_offload(self):
        with pytest.raises(
            AssertionError,
            match=(
                "fine-grained activation offloading with cuda_graph_impl='local'.*"
                "Unsupported offload_modules: \\['mlp_norm'\\]"
            ),
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[CudaGraphModule.moe_router],
                fine_grained_activation_offloading=True,
                offload_modules=['mlp_norm'],
                num_moe_experts=4,
            )

    def test_local_explicit_moe_graph_rejects_dropless_moe(self):
        with pytest.raises(
            AssertionError, match="moe cuda graph is only supported with drop-padding MoE"
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules=[CudaGraphModule.moe], num_moe_experts=4
            )

    def test_local_inference_full_layer_graph_allows_dropless_moe(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            cuda_graph_modules=[],
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
            num_moe_experts=4,
        )

        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_graph_allows_sync_free_hybridep_paged_stash(
        self, monkeypatch, cuda_graph_modules
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        cfg = _te_whole_moe_paged_stash_config(
            cuda_graph_modules=cuda_graph_modules, cuda_graph_warmup_steps=2
        )
        validate_moe_cuda_graph_support(cfg)

        assert cfg.cuda_graph_modules == cuda_graph_modules

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_paged_stash_rejects_dynamic_microbatches(
        self, monkeypatch, cuda_graph_modules
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        with pytest.raises(AssertionError, match="require a fixed runtime microbatch schedule"):
            _te_whole_moe_paged_stash_config(
                cuda_graph_modules=cuda_graph_modules, cuda_graph_dynamic_microbatches=True
            )

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    @pytest.mark.parametrize("warmup_steps", [0, 1])
    def test_te_whole_moe_paged_stash_requires_two_warmup_steps(
        self, monkeypatch, cuda_graph_modules, warmup_steps
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        with pytest.raises(AssertionError, match="require at least 2 cuda_graph_warmup_steps"):
            _te_whole_moe_paged_stash_config(
                cuda_graph_modules=cuda_graph_modules, cuda_graph_warmup_steps=warmup_steps
            )

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_paged_stash_requires_minimum_te_version(
        self, monkeypatch, cuda_graph_modules
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: False)

        with pytest.raises(ValueError, match=r"Transformer Engine >= 2\.19\.0"):
            _te_whole_moe_paged_stash_config(
                cuda_graph_modules=cuda_graph_modules, cuda_graph_warmup_steps=2
            )

    def test_te_moe_router_paged_stash_still_allows_dynamic_microbatches(self):
        cfg = _te_whole_moe_paged_stash_config(
            cuda_graph_modules=[CudaGraphModule.moe_router], cuda_graph_dynamic_microbatches=True
        )

        assert cfg.cuda_graph_dynamic_microbatches

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_graph_rejects_sync_free_hybridep_without_paged_stash(
        self, cuda_graph_modules
    ):
        with pytest.raises(
            AssertionError, match="sync-free HybridEP with rank capacity and paged stash"
        ):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=cuda_graph_modules,
                num_moe_experts=4,
                moe_token_dispatcher_type="flex",
                moe_flex_dispatcher_backend="hybridep",
                moe_expert_rank_capacity_factor=1.2,
                use_transformer_engine_op_fuser=True,
            )
            validate_moe_cuda_graph_support(cfg)

    def test_full_iteration_impl_requires_empty_scope(self):
        with pytest.raises(
            AssertionError,
            match='cuda_graph_modules must be empty when cuda_graph_impl="full_iteration"',
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='full_iteration', cuda_graph_modules=[CudaGraphModule.attn]
            )

    def test_full_iteration_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration'
            )
        assert cfg.cuda_graph_impl == 'full_iteration'
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration_inference'
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_noops_without_local_impl(self):
        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(cuda_graph_modules='full_iteration_inference')
        assert cfg.cuda_graph_impl == 'none'
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_full_iteration_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration' cannot be combined with "
            "inference_cuda_graph_scope='block'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration',
                inference_cuda_graph_scope='block',
            )

    def test_deprecated_full_iteration_inference_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration_inference',
                inference_cuda_graph_scope='layer',
            )

    def test_enable_cuda_graph_flag_migrates_to_local_impl(self, monkeypatch):
        args, _, print_messages = _validated_cuda_graph_cli_args(
            monkeypatch, ['--enable-cuda-graph']
        )
        assert args.cuda_graph_impl == 'local'
        assert any("--enable-cuda-graph is deprecated" in msg for msg in print_messages)

    def test_full_iteration_inference_scope_cli_migrates_to_block_scope(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch,
            ['--cuda-graph-impl', 'local', '--cuda-graph-modules', 'full_iteration_inference'],
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert args.cuda_graph_modules == []
        assert any(
            "--cuda-graph-modules 'full_iteration_inference' is deprecated" in msg
            for msg in warning_messages
        )

    def test_full_iteration_inference_scope_cli_noops_without_local_impl(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-scope', 'full_iteration_inference']
        )
        assert args.cuda_graph_impl == 'none'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert args.cuda_graph_modules == []
        assert any("has no effect when --cuda-graph-impl=none" in msg for msg in warning_messages)

    def test_full_iteration_inference_scope_cli_rejects_conflicting_new_scope(self, monkeypatch):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _validated_cuda_graph_cli_args(
                monkeypatch,
                [
                    '--cuda-graph-impl',
                    'local',
                    '--cuda-graph-modules',
                    'full_iteration_inference',
                    '--inference-cuda-graph-scope',
                    'layer',
                ],
            )

    def test_new_scope_cli_accepts_block(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'block']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block

    def test_new_scope_cli_accepts_layer(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'layer']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_removed_module_scoped_scope_name_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                'test_cuda_graphs.py',
                '--cuda-graph-impl',
                'local',
                '--inference-cuda-graph-scope',
                'module_scoped',
            ],
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_removed_old_inference_bool_flag_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys, "argv", ['test_cuda_graphs.py', '--inference-use-full-iteration-cuda-graph']
        )
        with pytest.raises(SystemExit):
            parse_args()

    # --- Backward compat: cuda_graph_scope → cuda_graph_modules rename ---

    def test_deprecated_cuda_graph_scope_kwarg_migrates_to_modules(self):
        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=['attn'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_new_cuda_graph_modules_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_modules=['attn', 'mlp'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn, CudaGraphModule.mlp]
        assert cfg.cuda_graph_scope is None

    def test_new_full_iteration_impl_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='full_iteration', cuda_graph_modules=[])
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_cli_migrates_to_modules(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--cuda-graph-scope', 'attn']
        )
        assert args.cuda_graph_modules == [CudaGraphModule.attn]
        assert any('--cuda-graph-scope is deprecated' in msg for msg in warning_messages)

    def test_cuda_graph_scope_is_standalone_class_for_pickle_compat(self):
        from megatron.core.transformer.enums import CudaGraphScope

        # CudaGraphScope is preserved as a standalone class (not an alias) so that
        # pre-refactor checkpoints can be deserialized without value-collision errors.
        assert CudaGraphScope is not CudaGraphModule
        assert CudaGraphScope.attn.value == 2  # original ordinals preserved
        assert CudaGraphScope.mamba.value == 7

    def test_cuda_graph_scope_and_inference_scope_in_safe_globals(self):
        from megatron.core.safe_globals import SAFE_GLOBALS
        from megatron.core.transformer.enums import CudaGraphScope

        assert CudaGraphScope in SAFE_GLOBALS
        assert InferenceCudaGraphScope in SAFE_GLOBALS

    def test_deprecated_cuda_graph_scope_enum_instance_migrates_to_modules(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.attn])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_enum_migrates_to_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.full_iteration])
        assert cfg.cuda_graph_impl == "full_iteration"
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_enum_migrates_to_scope(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl="local", cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_noops_without_local_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(
                cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.cuda_graph_impl == "none"
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None


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

    @pytest.mark.flaky_in_dev  # Issue #5474
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
        language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))
        assert isinstance(vision_projection_spec, MLPSubmodules)

        # Set vision model type
        vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        # Create LLaVA model with both encoder and decoder
        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
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
        # Cudagraph backward capture assumes the model has DDP so create main_grads for params
        for param in self.llava_model.parameters():
            param.main_grad = torch.zeros_like(param)

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


class TestParallelHybridBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

        # Ensure that this test is capturing to a fresh memory pool.
        CudaGraphManager.global_mempool = None

        def get_pg_collection():
            return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

        def get_mamba_block(hybrid_layer_pattern):
            layer_type_list = validate_segment_layers(hybrid_layer_pattern)
            transformer_config = TransformerConfig(
                hidden_size=256,  # The Mamba layer places several constraints on this
                # Need to specify num_attention_heads and num_layers or TransformerConfig
                # will generate errors.
                num_layers=len(layer_type_list),
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl="local",
            )
            modules = hybrid_stack_spec.submodules
            return HybridStack(
                transformer_config,
                modules,
                layer_type_list=layer_type_list,
                pp_layer_offset=0,
                pg_collection=get_pg_collection(),
            )

        self.mamba_block = get_mamba_block(hybrid_layer_pattern="M-M*-")
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

    def test_mhc_hybrid_layers_are_te_cudagraph_capturable(self):
        """Regression: a mHC-enabled HybridStack must expose graph-capturable layers.

        When ``enable_hyper_connections=True``, ``HybridStack`` wraps every layer in
        ``HyperConnectionHybridLayer``. That wrapper must subclass
        ``GraphableMegatronModule`` and be recognized by ``_layer_is_graphable`` so TE
        cuda-graph discovery finds the wrapped layers. Before the fix the wrapper
        subclassed plain ``MegatronModule``, so discovery rejected every layer (0
        graphable) and CUDA graph capture was silently skipped for the whole hybrid
        model -- making the mHC hybrid run fully eager (several times slower than the
        graphed GPT mHC path). This test fails on the pre-fix code via both assertions.
        """
        # The wrapper must be graph-capturable by construction.
        assert issubclass(HyperConnectionHybridLayer, GraphableMegatronModule)

        layer_type_list = validate_segment_layers("M-M*-")  # mamba / mlp / attention mix
        config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            enable_hyper_connections=True,
            num_residual_streams=4,
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.mamba, CudaGraphModule.mlp],
        )
        block = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "pp", "cp"]
            ),
        )

        # Every layer is wrapped, and the wrappers are discoverable as graphable.
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in block.layers)
        graphable = [layer for layer in block.layers if _layer_is_graphable(layer, config)]
        assert len(graphable) > 0, (
            "mHC HybridStack produced 0 graphable layers -- TE cuda-graph capture would "
            "be silently skipped for the entire model (the pre-fix bug)."
        )


class TestHybridTECudaGraphDiscovery:
    @staticmethod
    def _bare_hybrid_wrapper(*, offload_in_graph=None, delay_offload=False):
        wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(wrapper)
        # Intentionally minimal: individual CPU mocks provide only the state they exercise.
        wrapper.config = SimpleNamespace(
            cuda_graph_modules=[CudaGraphModule.attn],
            delay_offload_until_cuda_graph=delay_offload,
            fine_grained_activation_offloading=True,
        )
        object.__setattr__(wrapper, '_offload_module_in_cuda_graph_cached', None)
        if offload_in_graph is not None:
            object.__setattr__(
                wrapper, '_compute_offload_module_in_cuda_graph', lambda: offload_in_graph
            )
        return wrapper

    @staticmethod
    def _bare_transformer_inner(*, has_attention, offload_core_attn, is_moe=False):
        from megatron.core.transformer.identity_op import IdentityOp

        inner = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(inner)
        inner.self_attention = torch.nn.Linear(2, 2) if has_attention else IdentityOp()
        inner.cross_attention = IdentityOp()
        inner.mlp = IdentityOp()
        inner.offload_attn_norm = False
        inner.offload_qkv_linear = False
        inner.offload_core_attn = offload_core_attn
        inner.offload_attn_proj = False
        inner.offload_mlp_norm = False
        inner.is_moe_layer = is_moe
        inner.offload_module_in_cuda_graph = offload_core_attn
        return inner

    @staticmethod
    def _recording_offload_interface(calls):
        class RecordingOffloadInterface:
            @staticmethod
            def backward_record(hidden_states):
                calls.append('backward_record')
                return hidden_states + 1

            @staticmethod
            def forward_record():
                calls.append('forward_record')

            @staticmethod
            def enter_replay():
                calls.append('enter_replay')

            @staticmethod
            def flush_delayed_groups():
                calls.append('flush_delayed_groups')

            @staticmethod
            def exit_replay():
                calls.append('exit_replay')

        return RecordingOffloadInterface

    @pytest.mark.parametrize(
        ('has_attention', 'position_embedding_type', 'multi_latent_attention', 'expects_rotary'),
        [
            (True, 'rope', False, True),
            (False, 'rope', False, False),
            (True, 'learned_absolute', False, False),
            (True, 'rope', True, False),
        ],
    )
    def test_hybrid_wrapper_sample_inputs_include_rotary_embeddings(
        self,
        monkeypatch,
        has_attention,
        position_embedding_type,
        multi_latent_attention,
        expects_rotary,
    ):
        """The TE sample signature must match the wrapped attention replay signature."""
        from megatron.core.transformer.identity_op import IdentityOp

        inner = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(inner)
        inner.self_attention = torch.nn.Linear(2, 2) if has_attention else IdentityOp()
        inner.cross_attention = IdentityOp()

        wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(wrapper)
        wrapper.inner_layer = inner
        object.__setattr__(
            wrapper,
            'get_layer_static_inputs',
            lambda _seq_length, _micro_batch_size: {
                'hidden_states': torch.ones(8, 1, 8, requires_grad=True)
            },
        )

        rotary_pos_emb = torch.ones(8, 1, 1, 2)

        class RotaryEmbedding:
            @staticmethod
            def get_rotary_seq_len(*_args):
                return 8

            @staticmethod
            def __call__(_seq_length):
                return rotary_pos_emb

        chunk = SimpleNamespace(
            decoder=SimpleNamespace(layers=[wrapper]),
            position_embedding_type=position_embedding_type,
            rotary_pos_emb=RotaryEmbedding(),
        )
        helper = object.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            multi_latent_attention=multi_latent_attention,
            cuda_graph_modules=[CudaGraphModule.attn],
        )
        helper.seq_length = 8
        helper.micro_batch_size = 1
        helper.num_model_chunks = 1
        helper.num_microbatches = 1
        helper.flattened_callables = [wrapper]
        helper.num_layers_per_chunk = [1]
        helper.callables_per_chunk = [[wrapper]]
        helper.chunks_with_decoder = [chunk]
        helper._needs_full_local_padding_mask = lambda *_args: False
        helper._uses_mhc_direct_write_arena = lambda: False
        monkeypatch.setattr(cuda_graphs_module, 'is_te_min_version', lambda _version: True)

        _sample_args, sample_kwargs = helper._get_sample_arguments([1, -1])

        if expects_rotary:
            assert sample_kwargs[0]['rotary_pos_emb'] is rotary_pos_emb
        else:
            assert 'rotary_pos_emb' not in sample_kwargs[0]

    def test_hybrid_mtp_layers_are_flattened_and_adjacent_layers_are_grouped(self, monkeypatch):
        from megatron.core.transformer import cuda_graphs

        class FakeGraphLayer(torch.nn.Module):
            def __init__(self, group_with_next=False, graphable=True):
                super().__init__()
                self.group_with_next = group_with_next
                self.graphable = graphable
                self.group_tail = None

            def _can_group_te_cuda_graph_with(self, next_layer):
                return self.group_with_next and next_layer.graphable

            def _set_te_cuda_graph_group_tail(self, next_layer):
                self.group_tail = next_layer

        head = FakeGraphLayer(group_with_next=True)
        tail = FakeGraphLayer()
        eager = FakeGraphLayer(graphable=False)
        mtp_stack = HybridStack.__new__(HybridStack)
        torch.nn.Module.__init__(mtp_stack)
        mtp_stack.layers = torch.nn.ModuleList([head, tail, eager])

        monkeypatch.setattr(
            cuda_graphs, '_layer_is_graphable', lambda layer, config: layer.graphable
        )
        callables = cuda_graphs._get_mtp_te_callables(mtp_stack, object())

        assert callables == [head]
        assert head.group_tail is tail

        gpt_mtp_layer = FakeGraphLayer()
        assert cuda_graphs._get_mtp_te_callables(gpt_mtp_layer, object()) == [gpt_mtp_layer]

        class Holder:
            pass

        mtp_layer = Holder()
        mtp_layer.mtp_model_layer = mtp_stack
        chunk = Holder()
        chunk.mtp = Holder()
        chunk.mtp.layers = [mtp_layer]
        assert cuda_graphs._is_mtp_te_callable(head, chunk)
        assert cuda_graphs._is_mtp_te_callable(tail, chunk)
        assert not cuda_graphs._is_mtp_te_callable(eager, Holder())

    def test_capture_group_tail_does_not_change_module_registration(self):
        from megatron.core.transformer.identity_op import IdentityOp

        class Config:
            recompute_granularity = None
            recompute_modules = []
            fp8 = False
            first_last_layers_bf16 = False

        inner = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(inner)
        inner.self_attention = torch.nn.Linear(2, 2)
        inner.cross_attention = IdentityOp()
        inner.mlp = IdentityOp()

        head = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(head)
        head.config = Config()
        head.inner_layer = inner

        tail = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(tail)
        tail.capture_weight = torch.nn.Parameter(torch.ones(1))
        object.__setattr__(tail, '_inner_is_partial_moe_capture', lambda: True)
        object.__setattr__(tail, '_get_submodules_under_cudagraphs', lambda: [tail])

        state_dict_keys = tuple(head.state_dict())
        head._set_te_cuda_graph_group_tail(tail)

        assert head._get_te_cuda_graph_group_tail() is tail
        assert '_te_cuda_graph_group_tail' not in head._modules
        assert tuple(head.state_dict()) == state_dict_keys
        assert any(param is tail.capture_weight for param in head.parameters())

        head.cuda_graphs = [object()]
        head.train()
        assert head._get_active_te_cuda_graph_group_tail() is tail
        head.eval()
        assert head._get_active_te_cuda_graph_group_tail() is None

    def test_capture_group_does_not_cross_first_last_bf16_boundary(self):
        from megatron.core.transformer.identity_op import IdentityOp

        class Config:
            recompute_granularity = None
            recompute_modules = []
            fp8 = True
            first_last_layers_bf16 = True
            num_layers_at_start_in_bf16 = 1
            num_layers_at_end_in_bf16 = 1
            num_layers = 4

        inner = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(inner)
        inner.self_attention = torch.nn.Linear(2, 2)
        inner.cross_attention = IdentityOp()
        inner.mlp = IdentityOp()
        inner.is_mtp_layer = False

        head = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(head)
        head.config = Config()
        head.inner_layer = inner

        tail = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(tail)
        object.__setattr__(tail, '_inner_is_partial_moe_capture', lambda: True)

        head.layer_number, tail.layer_number = 1, 2
        assert not head._can_group_te_cuda_graph_with(tail)

        head.layer_number, tail.layer_number = 2, 3
        assert head._can_group_te_cuda_graph_with(tail)

        head.layer_number, tail.layer_number = 3, 4
        assert not head._can_group_te_cuda_graph_with(tail)

        inner.is_mtp_layer = True
        head.layer_number, tail.layer_number = 1, 2
        assert head._can_group_te_cuda_graph_with(tail)

    @pytest.mark.parametrize('offload_in_graph', [False, True])
    def test_hybrid_offload_graph_replay_args_include_te_stream_and_event(
        self, monkeypatch, offload_in_graph
    ):
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface,
        )

        wrapper = self._bare_hybrid_wrapper(offload_in_graph=offload_in_graph)
        graph_stream = object()
        graph_event = object()
        monkeypatch.setattr(
            FineGrainedActivationOffloadingInterface, 'cuda_graph_stream', lambda: graph_stream
        )
        monkeypatch.setattr(
            FineGrainedActivationOffloadingInterface, 'cuda_graph_event', lambda: graph_event
        )

        cudagraph_args, cudagraph_kwargs = wrapper._get_te_cuda_graph_replay_args(
            torch.ones(2, 1, 4)
        )

        assert len(cudagraph_args) == 1
        if offload_in_graph:
            assert cudagraph_kwargs['cuda_graph_stream'] is graph_stream
            assert cudagraph_kwargs['cuda_graph_event'] is graph_event
        else:
            assert 'cuda_graph_stream' not in cudagraph_kwargs
            assert 'cuda_graph_event' not in cudagraph_kwargs

    def test_hybrid_capture_records_offload_boundary_exactly_once(self):
        calls = []
        wrapper = self._bare_hybrid_wrapper(offload_in_graph=True)
        object.__setattr__(wrapper, 'off_interface', self._recording_offload_interface(calls))
        object.__setattr__(wrapper, '_get_te_cuda_graph_group_tail', lambda: None)
        object.__setattr__(wrapper, '_inner_is_partial_moe_capture', lambda: False)

        def forward(hidden_states, **_kwargs):
            calls.append('body')
            assert torch.equal(hidden_states, torch.full_like(hidden_states, 2))
            return hidden_states * 2, None

        object.__setattr__(wrapper, 'forward', forward)
        output = wrapper._te_cuda_graph_capture(torch.ones(2, 1, 4))

        assert calls == ['backward_record', 'body', 'forward_record']
        assert torch.equal(output[0], torch.full((2, 1, 4), 4.0))

    def test_hybrid_capture_impl_rejects_raw_packed_sequence_kwargs(self):
        wrapper = self._bare_hybrid_wrapper(offload_in_graph=False)

        with pytest.raises(AssertionError):
            wrapper._te_cuda_graph_capture_impl(
                torch.ones(2, 1, 4), cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32)
            )

    def test_transformer_capture_impl_rejects_raw_packed_sequence_kwargs(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(layer)

        with pytest.raises(AssertionError):
            layer._te_cuda_graph_capture_impl(
                torch.ones(2, 1, 4), cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32)
            )

    def test_grouped_hybrid_capture_has_one_outer_offload_boundary(self):
        calls = []
        head = self._bare_hybrid_wrapper(offload_in_graph=True)
        tail = self._bare_hybrid_wrapper(offload_in_graph=True)
        off_interface = self._recording_offload_interface(calls)
        object.__setattr__(head, 'off_interface', off_interface)
        object.__setattr__(tail, 'off_interface', off_interface)
        object.__setattr__(head, '_get_te_cuda_graph_group_tail', lambda: tail)

        def head_forward(hidden_states, **_kwargs):
            calls.append('attention_body')
            return hidden_states + 1, None

        def tail_capture_impl(hidden_states, **_kwargs):
            calls.append('moe_prefix_body')
            return (hidden_states + 1,)

        object.__setattr__(head, 'forward', head_forward)
        object.__setattr__(tail, '_te_cuda_graph_capture_impl', tail_capture_impl)

        output = head._te_cuda_graph_capture(torch.ones(2, 1, 4))

        assert calls == ['backward_record', 'attention_body', 'moe_prefix_body', 'forward_record']
        assert torch.equal(output[0], torch.full((2, 1, 4), 4.0))

    def test_moe_only_tail_does_not_claim_attention_offload(self):
        # Reproduce the stale inner-layer flag: cuda_graph_modules contains ``attn``
        # globally even though this split Hybrid layer has no attention body.
        inner = self._bare_transformer_inner(
            has_attention=False, offload_core_attn=True, is_moe=True
        )

        tail = self._bare_hybrid_wrapper()
        tail.inner_layer = inner

        assert not tail._compute_offload_module_in_cuda_graph()
        assert not tail.offload_module_in_cuda_graph

    def test_hybrid_empty_cuda_graph_scope_does_not_claim_inner_offload(self):
        calls = []
        wrapper = self._bare_hybrid_wrapper()
        wrapper.config.cuda_graph_modules = []
        wrapper.inner_layer = self._bare_transformer_inner(
            has_attention=True, offload_core_attn=True
        )
        object.__setattr__(wrapper, 'off_interface', self._recording_offload_interface(calls))
        object.__setattr__(wrapper, '_get_te_cuda_graph_group_tail', lambda: None)
        object.__setattr__(wrapper, '_inner_is_partial_moe_capture', lambda: False)
        object.__setattr__(
            wrapper,
            'forward',
            lambda hidden_states, **_kwargs: (calls.append('body') or hidden_states, None),
        )

        assert not wrapper.offload_module_in_cuda_graph
        wrapper._te_cuda_graph_capture(torch.ones(2, 1, 4))
        assert calls == ['body']

    def test_hybrid_explicit_attn_scope_claims_inner_offload(self):
        attention_inner = self._bare_transformer_inner(has_attention=True, offload_core_attn=True)

        attention_wrapper = self._bare_hybrid_wrapper()
        attention_wrapper.inner_layer = attention_inner
        assert attention_wrapper.offload_module_in_cuda_graph

    def test_hybrid_offload_property_caches_and_grouping_invalidates(self):
        calls = []
        head = self._bare_hybrid_wrapper()
        tail = self._bare_hybrid_wrapper()
        object.__setattr__(
            head,
            '_compute_inner_offload_module_in_cuda_graph',
            lambda: calls.append('head') or False,
        )
        object.__setattr__(
            tail,
            '_compute_inner_offload_module_in_cuda_graph',
            lambda: calls.append('tail') or True,
        )
        object.__setattr__(head, '_can_group_te_cuda_graph_with', lambda _tail: True)

        assert not head.offload_module_in_cuda_graph
        assert not head.offload_module_in_cuda_graph
        assert calls == ['head']

        head._set_te_cuda_graph_group_tail(tail)
        assert head._offload_module_in_cuda_graph_cached is None
        assert head.offload_module_in_cuda_graph
        assert head.offload_module_in_cuda_graph
        assert calls == ['head', 'head', 'tail']
        assert head._offload_module_in_cuda_graph_cached is True

    @pytest.mark.parametrize(
        ('delay_offload', 'tail_raises'), [(False, False), (True, False), (True, True)]
    )
    @pytest.mark.parametrize('grouped', [False, True])
    def test_hybrid_partial_replay_preserves_delayed_offload_lifecycle(
        self, monkeypatch, delay_offload, tail_raises, grouped
    ):
        calls = []
        head = self._bare_hybrid_wrapper(delay_offload=delay_offload)

        class Tail:
            @staticmethod
            def _resume_partial_moe_cuda_graph(_outputs):
                calls.append('eager_tail')
                if tail_raises:
                    raise RuntimeError('eager tail failed')
                return torch.full((2, 1, 4), 3.0), None

        object.__setattr__(head, 'off_interface', self._recording_offload_interface(calls))
        object.__setattr__(
            head, '_get_te_cuda_graph_group_tail', lambda: Tail() if grouped else None
        )
        object.__setattr__(head, '_inner_is_partial_moe_capture', lambda: not grouped)
        object.__setattr__(
            head, '_resume_partial_moe_cuda_graph', Tail._resume_partial_moe_cuda_graph
        )

        def graph_replay(_self, *_args, **_kwargs):
            calls.append('graph_replay')
            return (torch.ones(2, 1, 4),)

        monkeypatch.setattr(GraphableMegatronModule, '_te_cuda_graph_replay', graph_replay)

        if tail_raises:
            with pytest.raises(RuntimeError, match='eager tail failed'):
                head._te_cuda_graph_replay(torch.ones(2, 1, 4))
        else:
            output = head._te_cuda_graph_replay(torch.ones(2, 1, 4))
            assert torch.equal(output[0], torch.full((2, 1, 4), 3.0))

        expected_calls = ['graph_replay', 'eager_tail']
        if delay_offload:
            expected_calls = [
                'enter_replay',
                'graph_replay',
                'flush_delayed_groups',
                'eager_tail',
                'exit_replay',
            ]
        assert calls == expected_calls


# Global storage for comparing unique buffer counts across different num_microbatches,
# keyed by (pp_size, vpp_size)
_unique_buffer_counts = {}


class TestTECudaGraphHelper:
    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        # Note: _unique_buffer_counts is intentionally NOT cleared here so we can
        # compare values across parametrized test runs

    @pytest.mark.parametrize(
        ("local_layout", "cuda_graph_modules", "expected"),
        [
            ("dense", [CudaGraphModule.attn, CudaGraphModule.moe], False),
            ("dense", [], False),
            ("direct-moe", [CudaGraphModule.attn, CudaGraphModule.moe], True),
            ("nested-moe", [], True),
            ("direct-moe", [CudaGraphModule.moe_router], False),
        ],
    )
    def test_paged_stash_te_capture_context_requires_rank_local_whole_moe(
        self, monkeypatch, local_layout, cuda_graph_modules, expected
    ):
        layer = torch.nn.Module()
        layer.is_moe_layer = local_layout == "direct-moe"
        if local_layout == "nested-moe":
            inner_layer = torch.nn.Module()
            inner_layer.is_moe_layer = True
            layer.inner_layer = inner_layer

        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)

        helper = object.__new__(TECudaGraphHelper)
        helper.config = _te_whole_moe_paged_stash_config(cuda_graph_modules=cuda_graph_modules)
        helper.flattened_callables = [layer]
        helper.callables_per_chunk = []
        helper.num_microbatches = 1
        helper._start_capturing = lambda: 0.0
        helper._finish_capturing = lambda _start_time: None
        helper._get_cuda_graph_input_data = lambda: ([()], {'_order': [1, -1]})
        helper._validate_mhc_static_hidden_inputs = lambda _sample_args: None
        helper._uses_mhc_direct_write_arena = lambda: False

        capture_enabled = []

        def record_capture_context(enabled, order=None, config=None):
            capture_enabled.append(enabled)
            return nullcontext()

        monkeypatch.setattr(
            paged_stash_module, "paged_stash_te_graph_capture", record_capture_context
        )
        monkeypatch.setattr(
            cuda_graphs_module, "make_graphed_callables", lambda *args, **kwargs: ()
        )

        helper.create_cudagraphs()

        assert capture_enabled == [expected]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mhc_static_input_aliasing_requires_disjoint_liveness_windows(self):
        config = _base_cuda_graph_config(
            enable_hyper_connections=True,
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            mhc_recompute_layer_num=2,
            overlap_moe_expert_parallel_comm=True,
            expert_model_parallel_size=2,
            num_moe_experts=4,
            moe_token_dispatcher_type="alltoall",
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn],
            # The aliasing check only runs for the direct-write arena, which is
            # opt-in: without the switch this shape captures the whole attention
            # range and has no arena slot to alias.
            mhc_recompute_attn_cuda_graph_split=True,
            bf16=True,
        )
        helper = object.__new__(TECudaGraphHelper)
        helper.config = config

        shared = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        unique = torch.randn_like(shared)
        # Samples 0 and 2 alias one static buffer (as MCore consumed-sample
        # reuse and TE _reuse_graph_input_output_buffers legally do); sample 1
        # owns its own bytes.
        sample_args = [(shared,), (unique,), (shared.detach().requires_grad_(True),)]

        # Disjoint windows: sample 0 fully retires before sample 2's forward.
        helper._mhc_sample_order_intervals = {0: [0, 3], 1: [1, 4], 2: [5, 6]}
        helper._validate_mhc_static_hidden_inputs(sample_args)

        # Overlapping windows on a shared address must fail at capture time:
        # sample 2's forward starts before sample 0's backward retired, which
        # is exactly the aliasing that corrupts recompute direct-write replay.
        helper._mhc_sample_order_intervals = {0: [0, 5], 1: [1, 4], 2: [3, 6]}
        with pytest.raises(RuntimeError, match="windows overlap"):
            helper._validate_mhc_static_hidden_inputs(sample_args)

        # An entry whose backward never retires in the order is live forever,
        # so any aliasing against it fails.
        helper._mhc_sample_order_intervals = {0: [0, None], 1: [1, 4], 2: [3, 6]}
        with pytest.raises(RuntimeError, match="windows overlap"):
            helper._validate_mhc_static_hidden_inputs(sample_args)

        # A sample with no recorded window at all is rejected outright.
        helper._mhc_sample_order_intervals = {1: [1, 4]}
        with pytest.raises(RuntimeError, match="no recorded"):
            helper._validate_mhc_static_hidden_inputs(sample_args)

    @pytest.mark.parametrize("num_microbatches", [16, 64, 256])
    @pytest.mark.parametrize("pp_size", [1, 2, 4])
    @pytest.mark.parametrize("vpp_size", [None, 2])
    def test_get_cuda_graph_input_data(self, num_microbatches, pp_size, vpp_size):
        """Test _get_cuda_graph_input_data function in TECudaGraphHelper."""

        if vpp_size and pp_size == 1:
            pytest.skip("vpp_size must be None when pp_size is 1")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
        )

        # Set up test configuration
        seq_length = 128
        micro_batch_size = 2
        num_layers = 8
        vocab_size = 1024
        hidden_size = 64
        num_attention_heads = 4

        # Initialize num_microbatches calculator
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=micro_batch_size * num_microbatches,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        # Create transformer config directly
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            use_te_rng_tracker=True,
            bf16=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
            pipeline_dtype=torch.bfloat16,
            context_parallel_size=1,
        )

        # Create model
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        model = []
        for i in range(vpp_size or 1):
            this_model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=vocab_size,
                max_sequence_length=seq_length,
                parallel_output=True,
                position_embedding_type="rope",
                vp_stage=i if vpp_size else None,
            ).cuda()
            model.append(this_model)

        # Initialize TECudaGraphHelper
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=[],
        )

        # Call _get_cuda_graph_input_data (which internally calls _get_sample_arguments)
        sample_args, make_graphed_callables_kwargs = cuda_graph_helper._get_cuda_graph_input_data()

        # Extract sample_kwargs from the kwargs dict
        # For TE >= 1.10.0, sample_kwargs should always be present
        assert (
            'sample_kwargs' in make_graphed_callables_kwargs
        ), "sample_kwargs should be present in make_graphed_callables_kwargs for TE >= 1.10.0"
        sample_kwargs = make_graphed_callables_kwargs['sample_kwargs']

        # Basic checks
        num_graphable_layers = len(cuda_graph_helper.flattened_callables)
        if pp_size > 1:
            expected_length = num_graphable_layers * num_microbatches
        else:
            expected_length = num_graphable_layers
        assert len(sample_args) == expected_length, (
            f"sample_args length mismatch: expected {expected_length}, " f"got {len(sample_args)}"
        )
        assert len(sample_kwargs) == expected_length, (
            f"sample_kwargs length mismatch: expected {expected_length}, "
            f"got {len(sample_kwargs)}"
        )

        # Check that all elements are not None
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            assert args_item is not None, f"sample_args[{i}] is None"
            assert kwargs_item is not None, f"sample_kwargs[{i}] is None"
            assert isinstance(args_item, tuple), f"sample_args[{i}] should be a tuple"
            assert isinstance(kwargs_item, dict), f"sample_kwargs[{i}] should be a dict"
            assert len(args_item) > 0, f"sample_args[{i}] should not be empty"
            # Check that hidden_states is present
            assert "hidden_states" in kwargs_item or (
                len(args_item) > 0 and torch.is_tensor(args_item[0])
            ), f"sample_args[{i}] or sample_kwargs[{i}] should contain hidden_states"

        # Check tensor properties
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get hidden_states from args or kwargs
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                hidden_states = args_item[0]
            elif "hidden_states" in kwargs_item:
                hidden_states = kwargs_item["hidden_states"]
            else:
                continue

            assert torch.is_tensor(hidden_states), f"hidden_states at index {i} should be a tensor"
            # Check shape matches expected (accounting for TP/CP)
            expected_seq_len = seq_length // transformer_config.context_parallel_size
            if transformer_config.sequence_parallel:
                expected_seq_len = expected_seq_len // transformer_config.tensor_model_parallel_size
            assert hidden_states.shape[0] == expected_seq_len, (
                f"hidden_states seq_len mismatch at index {i}: "
                f"expected {expected_seq_len}, got {hidden_states.shape[0]}"
            )
            assert hidden_states.shape[1] == micro_batch_size, (
                f"hidden_states batch_size mismatch at index {i}: "
                f"expected {micro_batch_size}, got {hidden_states.shape[1]}"
            )
            assert hidden_states.shape[2] == transformer_config.hidden_size, (
                f"hidden_states hidden_size mismatch at index {i}: "
                f"expected {transformer_config.hidden_size}, got {hidden_states.shape[2]}"
            )

        # Memory optimization check: verify that buffers with same signature are reused
        # Create a mapping of sample_keys to indices
        sample_keys_to_indices = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Create sample_keys similar to the function
            args_keys = tuple((t.shape, t.dtype, t.layout) for t in args_item if torch.is_tensor(t))
            kwargs_keys = tuple(
                (k, v.shape, v.dtype, v.layout)
                for k, v in sorted(kwargs_item.items())
                if torch.is_tensor(v)
            )
            sample_keys = args_keys + kwargs_keys

            if sample_keys not in sample_keys_to_indices:
                sample_keys_to_indices[sample_keys] = []
            sample_keys_to_indices[sample_keys].append(idx)

        # Check that buffers with same signature share references (memory optimization)
        # The optimization reuses buffers when:
        # 1. They have the same signature (shape, dtype, layout)
        # 2. The backward pass of the original buffer has completed
        # 3. A new forward pass with matching signature needs a buffer
        # Count how many times each tensor is reused
        unique_tensors = set()
        tensor_reuse_count = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get the first tensor from args (hidden_states)
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                tensor_ptr = args_item[0].data_ptr()
                unique_tensors.add(tensor_ptr)
                tensor_reuse_count[tensor_ptr] = tensor_reuse_count.get(tensor_ptr, 0) + 1

        # With memory optimization, we should see some buffers reused
        # (i.e., some tensors should appear multiple times)
        max_reuse = max(tensor_reuse_count.values()) if tensor_reuse_count else 0
        total_entries = len(sample_args)
        unique_buffer_count = len(unique_tensors)

        # Verify that memory optimization is working:
        # - The number of unique buffers should be <= total entries
        # - With the 1F1B schedule and multiple microbatches, we should see some buffer reuse
        # - The number of unique buffers should be bounded as num_microbatches grows.
        assert unique_buffer_count <= total_entries, (
            f"Memory optimization check: unique_buffer_count ({unique_buffer_count}) "
            f"should be <= total_entries ({total_entries})"
        )
        global _unique_buffer_counts
        # Use (pp_size, vpp_size) as key to track unique buffer counts per configuration
        config_key = (pp_size, vpp_size)
        if config_key not in _unique_buffer_counts:
            _unique_buffer_counts[config_key] = unique_buffer_count
        else:
            assert unique_buffer_count == _unique_buffer_counts[config_key], (
                f"Unique buffer count mismatch: expected {_unique_buffer_counts[config_key]}, "
                f"got {unique_buffer_count}"
            )

        # Verify that buffers with the same signature can potentially be reused
        # (the actual reuse depends on the schedule, but the mechanism should work)
        if expected_length > 1:
            # Check that we have multiple entries with the same signature
            has_duplicate_signatures = any(
                len(indices) > 1 for indices in sample_keys_to_indices.values()
            )
            assert has_duplicate_signatures, (
                "Memory optimization: expected duplicate signatures for buffer reuse, "
                "but all signatures are unique"
            )

            # We tested with a large number of microbatches, so we should see some buffer reuse.
            if pp_size > 1:
                assert max_reuse > 1, "Expected some buffer reuse"

        # Verify that make_graphed_callables_kwargs contains expected keys
        assert (
            '_order' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain '_order'"
        assert (
            'num_warmup_iters' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'num_warmup_iters'"
        assert (
            'allow_unused_input' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'allow_unused_input'"

        # Verify the order in kwargs matches expectations
        order = make_graphed_callables_kwargs['_order']
        num_model_chunks = cuda_graph_helper.num_model_chunks
        forward_count = sum(1 for chunk_id in order if chunk_id > 0)
        if pp_size > 1:
            # Verify that all forward passes in order have corresponding entries in sample_args
            assert forward_count == num_microbatches * num_model_chunks, (
                f"Forward count mismatch: expected {num_microbatches * num_model_chunks}, "
                f"got {forward_count}"
            )
            expected_order_length = num_microbatches * num_model_chunks * 2
        else:
            assert num_model_chunks == 1, "Expected only one model chunk for pp_size == 1"
            assert forward_count == 1, "Expected only one forward pass for pp_size == 1"
            expected_order_length = 2
        assert (
            len(order) == expected_order_length
        ), f"Order length mismatch: expected {expected_order_length}, got {len(order)}"


class TestRequiredNumMicrobatchSlots:
    """Pure-Python tests for ``_get_required_num_microbatch_slots_from_order``.

    The method derives the smallest cuda-graph slot count that guarantees no
    in-flight microbatch's static buffer is reused before its backward
    completes. ``order`` is a 1F1B / interleaved-1F1B schedule transcript
    where ``+chunk_id`` denotes a forward and ``-chunk_id`` a backward.
    Non-integer entries (e.g. ``0.5`` for wgrad sub-steps) are skipped.
    """

    @staticmethod
    def _slots(order, num_chunks):
        return TECudaGraphHelper._get_required_num_microbatch_slots_from_order(order, num_chunks)

    def test_single_chunk_single_microbatch(self):
        # F0 then B0: one slot is enough.
        assert self._slots([1, -1], 1) == 1

    def test_single_chunk_pp_pipeline_4_microbatches_pp2(self):
        # PP=2 1F1B with 4 microbatches: warmup F-F, then F-B-F-B-..., then cooldown B-B.
        # Max in-flight = 2.
        order = [1, 1, -1, 1, -1, 1, -1, -1]
        assert self._slots(order, 1) == 2

    def test_two_chunks_independent(self):
        # Two model chunks (VPP=2), each running a tiny PP=2-style 1F1B in turn.
        # Per chunk max in-flight = 2 -> 2 slots.
        order = [1, 1, -1, -1, 2, 2, -2, -2]
        assert self._slots(order, 2) == 2

    def test_two_chunks_interleaved(self):
        # Worst case: forwards stack up across chunks before any backward.
        # F0 F0 F1 F1 B1 B1 B0 B0 -> per-chunk max in-flight = 2.
        order = [1, 1, 2, 2, -2, -2, -1, -1]
        assert self._slots(order, 2) == 2

    def test_skips_non_integer_entries(self):
        # Float c_ids (e.g. 0.5 for wgrad sub-steps) must be ignored.
        order = [1, 0.5, -0.5, -1]
        assert self._slots(order, 1) == 1

    def test_minimum_slot_is_one(self):
        # Empty / no-op order still returns at least 1 (we always need a slot).
        assert self._slots([], 1) == 1

    def test_unbalanced_order_asserts(self):
        # Forward without matching backward -> outstanding != 0 at end -> assert.
        with pytest.raises(AssertionError):
            self._slots([1], 1)

    def test_negative_outstanding_asserts(self):
        # Backward before any forward for a chunk -> outstanding goes negative.
        with pytest.raises(AssertionError):
            self._slots([-1], 1)


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def is_nccl_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP

    return HAVE_TE_EP


class TestPartialCudaGraph:
    """Test that CUDA graph outputs match non-CUDA graph outputs for various scopes."""

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.tp_size = 2
        self.cp_size = 2
        self.cuda_graph_helper = None
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
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_decoder_block_spec,
        **config_kwargs,
    ):
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            config,
            use_transformer_engine=True,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
        )
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
        self, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_cuda_graphs.py']
        args = parse_args()
        args.num_layers = 4
        args.mtp_num_layers = 1
        args.vocab_size = 1024
        args.hidden_size = 512
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.global_batch_size = self.micro_batch_size * 8 // self.tp_size // self.cp_size
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = self.tp_size
        args.sequence_parallel = True if self.tp_size > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = self.cp_size
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
        args.expert_tensor_parallel_size = 1 if ep_size > 1 else self.tp_size
        args.moe_shared_expert_intermediate_size = 1024
        args.moe_layer_freq = [0, 0, 1, 1]
        args.moe_permute_fusion = True
        args.moe_router_fusion = True
        args.moe_router_topk = 2
        args.moe_router_dtype = "fp32"

        # CUDA graph settings
        args.cuda_graph_impl = cuda_graph_impl
        args.cuda_graph_modules = cuda_graph_modules
        args.cuda_graph_warmup_steps = cuda_graph_warmup_steps

        # fp8 settings
        if fp8_available:
            args.fp8 = "e4m3"
            args.fp8_recipe = "tensorwise"
            args.first_last_layers_bf16 = True
            args.num_layers_at_start_in_bf16 = 1
            args.num_layers_at_end_in_bf16 = 1

        for key, value in kwargs.items():
            assert hasattr(args, key) or hasattr(TransformerConfig, key), f"Unknown argument: {key}"
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size, cp_size):
        data = list(range(seq_length // cp_size))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length // cp_size, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length // cp_size).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_test_helper(
        self, ep_size, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, **kwargs
    ):
        """Test fp8_param with gpt_model."""
        args = self.create_test_args(
            cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
        )

        set_args(args)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size, self.cp_size
        )

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        if cuda_graph_impl == "transformer_engine":
            self.cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        loss_list = []

        for i in range(100):
            gpt_model[0].zero_grad_buffer()
            optimizer.zero_grad()

            # Capture CUDA graphs after warmup if helper is provided
            if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                self.cuda_graph_helper.create_cudagraphs()

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length // self.cp_size

            # Verify gradients
            loss = output.mean()
            loss.backward()

            for param in gpt_model[0].parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            # Keep the layer handles for post-run assertions: the helper is
            # nulled below, but the layer objects (and attributes the replay
            # tail set on them) outlive graph teardown.
            self.last_flattened_callables = self.cuda_graph_helper.flattened_callables
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        return torch.tensor(loss_list)

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    @pytest.mark.parametrize("moe_dropless_dispatcher", [False, True])
    @pytest.mark.parametrize("moe_dispatcher_type", ["alltoall", "deepep", "hybridep", "ncclep"])
    def test_moe_partial_cudagraph(self, ep_size, moe_dropless_dispatcher, moe_dispatcher_type):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1 if ep_size > 1 else self.tp_size,
            expert_model_parallel_size=ep_size,
        )

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
        elif moe_dispatcher_type == "ncclep":
            if not is_nccl_ep_available():
                pytest.skip("NCCL EP is not available")
            if ep_size < 2:
                pytest.skip("NCCL EP requires expert_model_parallel_size >= 2 (ep_bootstrap)")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "ncclep"
            # ncclep sizes a per-rank recv buffer from this and overflow hard-traps; size generously.
            extra_kwargs["moe_expert_rank_capacity_factor"] = 8.0
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type
        if not moe_dropless_dispatcher:
            if moe_dispatcher_type in ("deepep", "ncclep"):
                pytest.skip(f"{moe_dispatcher_type} doesn't support drop&pad MoE")
            extra_kwargs["moe_expert_capacity_factor"] = 1.0
            extra_kwargs["moe_pad_expert_input_to_capacity"] = True

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_modules in [
            None,
            [CudaGraphModule.attn],
            [CudaGraphModule.moe],
            [CudaGraphModule.mlp, CudaGraphModule.moe_router],
            [
                CudaGraphModule.attn,
                CudaGraphModule.mlp,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        ]:
            if (moe_dropless_dispatcher or moe_dispatcher_type in ("hybridep", "ncclep")) and (
                cuda_graph_modules is None or CudaGraphModule.moe in cuda_graph_modules
            ):
                # Dropless MoE or a dynamic-shape flex backend (Hybrid EP / NCCL EP) can't be
                # captured at the "moe" scope (the dispatch does a device-to-host sync). Skip;
                # the surrounding compute submodules are still graphed.
                continue
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref)

        if moe_dispatcher_type == "hybridep":
            reset_hybrid_ep_buffer()
        if moe_dispatcher_type == "ncclep":
            from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize

            nccl_ep_finalize()
        Utils.destroy_model_parallel()

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    def test_mhc_moe_partial_cudagraph(self, ep_size):
        """Test that mHC (Hyper Connection) layers produce identical loss curves
        with and without TE partial CUDA graph capture.

        This validates the fix where HyperConnectionTransformerLayer overrides
        _te_cuda_graph_replay_impl (not _te_cuda_graph_replay) so that the parent's
        delay_offload_until_cuda_graph lifecycle and overlap_moe_expert_parallel_comm
        handling are preserved.
        """
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1 if ep_size > 1 else self.tp_size,
            expert_model_parallel_size=ep_size,
        )

        extra_kwargs = {
            "enable_hyper_connections": True,
            "num_residual_streams": 4,
            "mtp_num_layers": None,  # mHC is incompatible with MTP
        }

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_modules in [
            [CudaGraphModule.attn],
            [CudaGraphModule.mlp, CudaGraphModule.moe_router],
            [
                CudaGraphModule.attn,
                CudaGraphModule.mlp,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        ]:
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref), (
                f"mHC loss mismatch with cuda_graph_modules={cuda_graph_modules}, ep_size={ep_size}. "
                f"Max diff: {torch.max(torch.abs(loss_list - loss_list_ref))}"
            )

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    def test_mhc_recompute_whole_attention_cudagraph(self):
        """mHC selective recompute under whole-attention capture matches eager.

        With mhc_recompute_attn_cuda_graph_split off (the default), an attn-scope
        graph captures the whole attention range and the replay's non-split tail
        runs the MLP-side mHC group eagerly: mlp_hyper_connection registers its
        checkpoints against the manager __call__ stashed on the layer, the block
        discards at group end, and the unified hook replays them in backward. A
        graphed run must therefore reproduce the eager loss curve bit for bit --
        this is the executing coverage for that tail, with a live manager rather
        than a mocked boundary.
        """
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=self.tp_size,
            expert_model_parallel_size=1,
        )

        extra_kwargs = {
            "enable_hyper_connections": True,
            "num_residual_streams": 4,
            "mtp_num_layers": None,  # mHC is incompatible with MTP
            "recompute_granularity": "selective",
            "recompute_modules": ["mhc"],
            "mhc_recompute_layer_num": 2,
        }

        loss_list_ref = self._run_test_helper(1, "none", None, 0, **extra_kwargs)
        loss_list = self._run_test_helper(
            1, "transformer_engine", [CudaGraphModule.attn], 3, **extra_kwargs
        )
        assert torch.equal(loss_list, loss_list_ref), (
            "mHC recompute under whole-attention capture diverged from eager. "
            f"Max diff: {torch.max(torch.abs(loss_list - loss_list_ref))}"
        )
        # Loss parity alone is blind to an inert manager (an empty recompute
        # group discards and replays nothing, bit-identically), so pin the
        # layer-side threading directly: the graphed replay tail must have
        # created the pre-MLP checkpoint against a live manager.
        layers = self.last_flattened_callables
        assert any(
            getattr(layer, "pre_mlp_norm_checkpoint", None) is not None for layer in layers
        ), (
            "no layer created pre_mlp_norm_checkpoint during graphed replay: the "
            "whole-attention tail is not threading the recompute manager"
        )

        Utils.destroy_model_parallel()


class _SimpleModule(MegatronModule):
    """Minimal MegatronModule for testing CudaGraphManager with function_name."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def my_op(self, x):
        return self.linear(x)


class _SimpleNonModule:
    """non-nn.Module base_module for testing the function_name= form of `CudaGraphManager`."""

    def __init__(self, config):
        self.weight = torch.randn(config.hidden_size, config.hidden_size, device="cuda")

    def my_op(self, x):
        return x @ self.weight


def _make_simple_module(config):
    return _SimpleModule(config).cuda().eval()


def _make_simple_non_module(config):
    return _SimpleNonModule(config)


class TestInlineCaptureManager:
    """Tests for CudaGraphManager with inline_capture, function_name, eager, and cache_key."""

    def _make_config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            inference_rng_tracker=True,
        )

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "make_module",
        [
            pytest.param(_make_simple_module, id="nn_module"),
            pytest.param(_make_simple_non_module, id="plain_class"),
        ],
    )
    @torch.inference_mode()
    def test_inline_capture_matches_eager(self, make_module):
        """Inline-captured graph output must match eager execution."""
        config = self._make_config()
        module = make_module(config)

        # Get eager reference before wrapping
        x = torch.randn(4, config.hidden_size, device="cuda")
        eager_out = module.my_op(x).clone()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        # First call captures, second replays
        graph_out_1 = module.my_op(x)
        graph_out_2 = module.my_op(x)
        assert torch.equal(eager_out, graph_out_1)
        assert torch.equal(eager_out, graph_out_2)
        assert len(mgr.cudagraph_runners) == 1
        assert mgr.cudagraph_runners[0].fwd_graph_recorded

    @torch.inference_mode()
    def test_eager_bypass(self):
        """eager=True must bypass graph capture entirely."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        _ = module.my_op(x, eager=True)
        _ = module.my_op(x, eager=True)
        assert len(mgr.cudagraph_runners) == 0, "eager=True should not create runners"

    @torch.inference_mode()
    def test_cache_key_routing(self):
        """Different cache_keys must create separate runners."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="key_a")
        module.my_op(x, cache_key="key_b")

        assert len(mgr.cudagraph_runners) == 2
        assert mgr.custom_cudagraphs_lookup_table["key_a"] is not None
        assert mgr.custom_cudagraphs_lookup_table["key_b"] is not None
        assert (
            mgr.custom_cudagraphs_lookup_table["key_a"]
            is not mgr.custom_cudagraphs_lookup_table["key_b"]
        )

        # Same key reuses the runner
        module.my_op(x, cache_key="key_a")
        assert len(mgr.cudagraph_runners) == 2

    @torch.inference_mode()
    def test_num_warmup_steps_override(self):
        """num_warmup_steps on the manager must override the config value on runners."""
        config = self._make_config()
        config.cuda_graph_warmup_steps = 3
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="test")

        runner = mgr.cudagraph_runners[0]
        assert (
            runner.num_warmup_steps == 0
        ), f"Expected 0 warmup steps (manager override), got {runner.num_warmup_steps}"


class TestSkipFp8WeightUpdateTensor:
    """Regression test for the TE 2.15 ``set_skip_fp8_weight_update_tensor`` removal."""

    @staticmethod
    def _read_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        getter = getattr(FP8GlobalStateManager, "get_skip_fp8_weight_update_tensor", None)
        if getter is not None:
            return getter()
        return FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor

    @staticmethod
    def _reset_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        if "skip_fp8_weight_update_tensor" in vars(FP8GlobalStateManager):
            FP8GlobalStateManager.skip_fp8_weight_update_tensor = None
        qstate = getattr(FP8GlobalStateManager, "quantization_state", None)
        if qstate is not None and hasattr(qstate, "skip_fp8_weight_update_tensor"):
            qstate.skip_fp8_weight_update_tensor = None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sets_value_in_place(self):
        """Helper writes the right value and reuses the same storage across calls."""
        from megatron.core.transformer.cuda_graphs import _set_skip_fp8_weight_update_tensor

        self._reset_skip_tensor()
        try:
            _set_skip_fp8_weight_update_tensor(True)
            t = self._read_skip_tensor()
            assert t.shape == (1,) and t.dtype == torch.float32 and t.is_cuda
            assert t.item() == 1.0

            # data_ptr must stay stable so captured cudagraphs read the same address.
            ptr = t.data_ptr()
            _set_skip_fp8_weight_update_tensor(False)
            assert self._read_skip_tensor().data_ptr() == ptr
            assert self._read_skip_tensor().item() == 0.0
        finally:
            self._reset_skip_tensor()


if __name__ == "__main__":

    test = TestParallelTransformerBlockCudagraphs()
    test.setup_method(method=None)
    test.test_gpu_cudagraph()
    test.teardown_method(method=None)

    llava_test = TestLLaVACudaGraph()
    llava_test.setup_method(method=None)
    llava_test.test_llava_cudagraph_is_last_layer_logic()
    llava_test.teardown_method(method=None)

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_moe_partial_cudagraph(4, True, "alltoall")
    test.teardown_method(method=None)

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_mhc_moe_partial_cudagraph(4)
    test.teardown_method(method=None)
