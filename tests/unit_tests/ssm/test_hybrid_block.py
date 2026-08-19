# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
    create_cudagraphs,
)
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestHybridBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

    def get_hybrid_block(self, layer_pattern, **config_kwargs):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_kwargs,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_dsa_hybrid_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            add_bias_linear=False,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_mla_hybrid_block(self, layer_pattern):
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        """Test GPU forward pass."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_hybrid_block(layer_pattern)
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == block.config.hidden_size
        assert output.dtype == torch.float32

    def _run_forward(self, block, sequence_length=32, micro_batch_size=2):
        block.cuda()
        block.train()
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, block.config.hidden_size)
        ).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()
        return block(hidden_states, attention_mask=attention_mask)

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "recompute_kwargs",
        [
            dict(recompute_granularity="full", recompute_method="block", recompute_num_layers=2),
            dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2),
            dict(recompute_granularity="selective", recompute_modules=["core_attn", "mlp"]),
        ],
        ids=["full_block", "full_uniform", "selective"],
    )
    @pytest.mark.parametrize(
        "layer_pattern",
        [
            Symbols.MAMBA * 5,
            Symbols.ATTENTION * 5,
            Symbols.MLP * 5,
            Symbols.ATTENTION + Symbols.MLP + Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP,
            Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP,
        ],
    )
    def test_recompute(self, recompute_kwargs: dict, layer_pattern: str):
        seed = 123
        sequence_length, micro_batch_size = 32, 2

        # When 'mlp' is in recompute_modules, the wrapped MLP's `(out, bias_param)`
        # output triggers a reentrant-backward deadlock in CheckpointFunction.
        # All three in-tree MoE recipes that use `recompute_modules=[..., 'mlp']`
        # set `--disable-bias-linear: true`, so we match that usage pattern here.
        arch_kwargs = {}
        if recompute_kwargs.get(
            "recompute_granularity"
        ) == "selective" and "mlp" in recompute_kwargs.get("recompute_modules", []):
            arch_kwargs["add_bias_linear"] = False

        def build_inputs():
            torch.manual_seed(seed)
            hs = torch.randn(
                (sequence_length, micro_batch_size, 256), device="cuda", requires_grad=True
            )
            am = torch.ones(
                (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool, device="cuda"
            )
            return hs, am

        hs, am = build_inputs()

        def run(block, hs, am):
            out = block(hs, attention_mask=am)
            out.float().sum().backward()
            grads = {
                n: p.grad.detach().float().cpu()
                for n, p in block.named_parameters()
                if p.grad is not None
            }
            return out.detach().float().cpu(), grads

        # --- Baseline (no recompute) ---
        model_parallel_cuda_manual_seed(seed)
        torch.manual_seed(seed)
        base = self.get_hybrid_block(layer_pattern, **arch_kwargs).cuda()
        base.train()
        base_logits, base_grads = run(base, hs, am)
        del base
        torch.cuda.empty_cache()

        # --- Recompute ---
        model_parallel_cuda_manual_seed(seed)
        torch.manual_seed(seed)
        rec = self.get_hybrid_block(layer_pattern, **arch_kwargs, **recompute_kwargs).cuda()
        rec.train()
        rec_logits, rec_grads = run(rec, hs, am)

        # --- Numerical equivalence ---
        assert torch.equal(rec_logits, base_logits), f"Logits should be bitwise matched"
        assert set(rec_grads.keys()) == set(base_grads.keys())
        for name in base_grads:
            gb, gr = base_grads[name], rec_grads[name]
            assert torch.equal(gr, gb), f"Grad should be bitwise matched for {name}"

    def test_layer_types(self):
        """
        Make sure that the layer types specified with layer_pattern
        were honored.
        """
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_hybrid_block(layer_pattern)
        layers = block.layers
        # Note that this matches the order specified by layer_pattern above
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], TransformerLayer)
        assert isinstance(layers[2].mlp, MLP)

    def test_shortcut_pair_is_one_registered_block(self):
        block = self.get_hybrid_block(
            Symbols.MAMBA + Symbols.MOE,
            num_moe_experts=1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=True,
            moe_shared_expert_intermediate_size=256,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

        assert len(block.layers) == 1
        shortcut = block.layers[0]
        assert isinstance(shortcut, ShortcutMoEBlock)
        assert isinstance(shortcut.compute_layer, MambaLayer)
        assert isinstance(shortcut.moe_layer, TransformerLayer)
        assert shortcut.shortcut_pre_mlp_layernorm is not shortcut.moe_layer.pre_mlp_layernorm
        assert isinstance(shortcut.shortcut_post_norm, torch.nn.RMSNorm)
        assert block.num_layers_per_pipeline_rank == 2

        state_keys = set(block.state_dict())
        assert any(key.startswith("layers.0.compute_layer.") for key in state_keys)
        assert any(key.startswith("layers.0.moe_layer.") for key in state_keys)
        assert any(key.startswith("layers.0.shortcut_pre_mlp_layernorm.") for key in state_keys)
        assert "layers.0.shortcut_post_norm.weight" in state_keys

    def test_shortcut_pair_eager_forward_backward(self):
        block = self.get_hybrid_block(
            Symbols.MAMBA + Symbols.MOE,
            num_moe_experts=1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=True,
            moe_shared_expert_intermediate_size=256,
            add_bias_linear=False,
        ).cuda()
        block.train()

        hidden_states = torch.randn(
            16, 2, block.config.hidden_size, device=torch.cuda.current_device(), requires_grad=True
        )
        output = block(hidden_states, attention_mask=None)
        output.float().square().mean().backward()

        assert output.shape == hidden_states.shape
        shortcut = block.layers[0]
        logical_norms = (
            shortcut.shortcut_pre_mlp_layernorm,
            shortcut.moe_layer.pre_mlp_layernorm,
            shortcut.shortcut_post_norm,
        )
        assert len({id(norm.weight) for norm in logical_norms}) == len(logical_norms)
        for norm in logical_norms:
            assert norm.weight.grad is not None
            assert torch.isfinite(norm.weight.grad).all()
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    @pytest.mark.parametrize("tp_size", [1, 2], ids=["tp1", "tp2-sp"])
    def test_shortcut_pair_cuda_graph_replay_matches_eager(self, tp_size):
        if tp_size == 2:
            if torch.distributed.get_world_size() < 4:
                pytest.skip("TP=2/EP=2 shortcut graph parity requires four ranks")
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=2,
            )
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        model_parallel_cuda_manual_seed(123)
        common_config = dict(
            num_moe_experts=4 if tp_size == 2 else 1,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=2 if tp_size == 2 else 1,
            sequence_parallel=tp_size > 1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=True,
            moe_shared_expert_intermediate_size=256,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        eager = self.get_hybrid_block(Symbols.MAMBA + Symbols.MOE, **common_config).cuda()
        graphed = self.get_hybrid_block(
            Symbols.MAMBA + Symbols.MOE,
            cuda_graph_impl="local",
            cuda_graph_warmup_steps=1,
            cuda_graph_modules=["shortcut_block"],
            **common_config,
        ).cuda()
        graphed.load_state_dict(eager.state_dict())
        eager.train()
        graphed.train()
        for parameter in graphed.parameters():
            parameter.main_grad = torch.zeros_like(parameter)

        def logical_norms(block):
            shortcut = block.layers[0]
            return (
                shortcut.shortcut_pre_mlp_layernorm,
                shortcut.moe_layer.pre_mlp_layernorm,
                shortcut.shortcut_post_norm,
            )

        def clear_gradients(block):
            block.zero_grad(set_to_none=True)
            for parameter in block.parameters():
                if hasattr(parameter, "main_grad"):
                    parameter.main_grad.zero_()

        def forward_backward(block, hidden_states, output_gradient):
            output = block(hidden_states, attention_mask=None)
            output_snapshot = output.detach().clone()
            dispatcher = block.layers[0].moe_layer.mlp.token_dispatcher
            expected_tokens = hidden_states.shape[0] * hidden_states.shape[1]
            assert dispatcher.local_map.sum().item() == expected_tokens
            assert dispatcher.local_probs.numel() == expected_tokens
            output.backward(output_gradient)
            gradients = []
            for norm in logical_norms(block):
                gradient = norm.weight.grad
                if gradient is None and hasattr(norm.weight, "main_grad"):
                    gradient = norm.weight.main_grad
                gradients.append(gradient.clone())
            return output_snapshot, gradients

        try:
            capture_input = torch.randn(
                16, 2, graphed.config.hidden_size, device="cuda", requires_grad=True
            )
            capture_gradient = torch.randn_like(capture_input)
            forward_backward(graphed, capture_input, capture_gradient)
            create_cudagraphs()

            for replay_index in range(2):
                torch.manual_seed(1000 + replay_index)
                replay_input = torch.randn(
                    16, 2, eager.config.hidden_size, device="cuda", requires_grad=True
                )
                output_gradient = torch.randn_like(replay_input)

                clear_gradients(eager)
                eager_output, eager_gradients = forward_backward(
                    eager, replay_input.detach().clone().requires_grad_(True), output_gradient
                )
                clear_gradients(graphed)
                graph_output, graph_gradients = forward_backward(
                    graphed, replay_input.detach().clone().requires_grad_(True), output_gradient
                )

                torch.testing.assert_close(graph_output, eager_output, rtol=1e-5, atol=1e-6)
                for graph_gradient, eager_gradient in zip(graph_gradients, eager_gradients):
                    torch.testing.assert_close(graph_gradient, eager_gradient, rtol=1e-5, atol=1e-6)
        finally:
            shortcut = graphed.layers[0]
            for manager in (
                shortcut._graph_state.route_manager,
                shortcut._graph_state.output_manager,
            ):
                for runner in manager.cudagraph_runners:
                    if hasattr(runner, "fwd_graph"):
                        del runner.fwd_graph
                    if hasattr(runner, "bwd_graph"):
                        del runner.bwd_graph
            torch.cuda.synchronize()
            _CudagraphGlobalRecord.cudagraph_created = False
            _CudagraphGlobalRecord.cudagraph_record = []
            _CudagraphGlobalRecord.cudagraph_inference_record = []
            CudaGraphManager.global_mempool = None

    def test_invalid_layer_types_cause_failure(self):
        invalid_symbol = 'X'
        assert invalid_symbol not in Symbols.VALID_LAYERS  # sanity check.
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + invalid_symbol
        # validate_segment_layers() in hybrid_layer_allocation.py throws a ValueError.
        with pytest.raises(ValueError):
            block = self.get_hybrid_block(layer_pattern)

    def test_gdn_layer_types(self):
        """
        Make sure that G creates a TransformerLayer wrapping GatedDeltaNet,
        while * creates a TransformerLayer wrapping SelfAttention.
        """
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        block = self.get_hybrid_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], TransformerLayer)
        assert isinstance(layers[0].self_attention, GatedDeltaNet)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, SelfAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_gdn_gpu_forward(self):
        """Test GPU forward pass with GDN, attention, and Mamba layers."""
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
        )
        modules = hybrid_stack_spec.submodules
        block = HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )
        block.cuda()
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, block.config.hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        attention_mask = attention_mask.cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == block.config.hidden_size
        assert output.dtype == torch.float32

    def test_dsa_layer_types(self):
        """D symbol creates a TransformerLayer with absorbed MLA and DSA core attention."""
        layer_pattern = Symbols.MAMBA + Symbols.DS_ATTENTION + Symbols.MAMBA
        block = self.get_dsa_hybrid_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, AbsorbedMLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, DSAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_dsa_layer_types(self):
        """* and D in the same block fail."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.DS_ATTENTION + Symbols.MAMBA
        with pytest.raises(ValueError):
            block = self.get_dsa_hybrid_block(layer_pattern)

    def test_mla_layer_types(self):
        """+ symbol creates a TransformerLayer with MLASelfAttention but
        standard (non-DSA) core attention."""
        layer_pattern = Symbols.MAMBA + Symbols.MLA + Symbols.MAMBA
        block = self.get_mla_hybrid_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, MLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, TEDotProductAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_mla_layer_types(self):
        """* and + in the same block fail (same reason as * and D)."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLA + Symbols.MAMBA
        with pytest.raises(ValueError):
            block = self.get_mla_hybrid_block(layer_pattern)
