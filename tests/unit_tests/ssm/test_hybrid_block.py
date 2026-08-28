# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_stack_spec,
    hybrid_inference_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.models.hybrid.shortcut_block import ShortcutExecutionMode, ShortcutMoEBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA as HAVE_GDN
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.gated_delta_product import HAVE_FLA as HAVE_GDP
from megatron.core.ssm.gated_delta_product import HAVE_MAMBA_SSM as HAVE_GDP_MAMBA
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
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

    def get_hybrid_block(self, layer_pattern, *, stack_spec=hybrid_stack_spec, **config_kwargs):
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
        modules = stack_spec.submodules
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

    @pytest.mark.parametrize(
        ("compute_symbol", "stack_spec", "compute_config"),
        [
            pytest.param(Symbols.MAMBA, hybrid_stack_spec, {}, id="mamba"),
            pytest.param(
                Symbols.GDN,
                hybrid_stack_spec,
                {
                    "bf16": True,
                    "params_dtype": torch.bfloat16,
                    "activation_func": torch.nn.functional.silu,
                },
                marks=pytest.mark.skipif(not HAVE_GDN, reason="FLA is not installed"),
                id="gdn",
            ),
            pytest.param(Symbols.ATTENTION, hybrid_stack_spec, {}, id="attention"),
            pytest.param(
                Symbols.MAMBA,
                gated_delta_product_stack_spec,
                {
                    "bf16": True,
                    "params_dtype": torch.bfloat16,
                    "mamba_num_heads": 4,
                    "mamba_head_dim": 64,
                    "mamba_num_groups": 4,
                    "mamba_state_dim": 16,
                },
                marks=pytest.mark.skipif(
                    not (HAVE_GDP and HAVE_GDP_MAMBA), reason="GDP dependencies are not installed"
                ),
                id="gdp",
            ),
        ],
    )
    @pytest.mark.parametrize("parallel", [False, True], ids=["serial", "overlap"])
    def test_shortcut_pair_eager_forward_backward(
        self, monkeypatch, compute_symbol, stack_spec, compute_config, parallel
    ):
        block = self.get_hybrid_block(
            compute_symbol + Symbols.MOE,
            stack_spec=stack_spec,
            num_moe_experts=1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=parallel,
            moe_shared_expert_intermediate_size=256,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            **compute_config,
        )

        assert len(block.layers) == 1
        assert block.num_layers_per_pipeline_rank == 2
        shortcut = block.layers[0]
        assert isinstance(shortcut, ShortcutMoEBlock)
        assert shortcut.execution_mode == ShortcutExecutionMode.resolve(overlap_a2a=parallel)
        assert shortcut.compute_layer.supports_split_output_projection()
        assert isinstance(shortcut.moe_layer, TransformerLayer)
        state_keys = set(block.state_dict())
        assert any(key.startswith("layers.0.compute_layer.") for key in state_keys)
        assert any(key.startswith("layers.0.moe_layer.") for key in state_keys)
        assert any(key.startswith("layers.0.shortcut_pre_mlp_layernorm.") for key in state_keys)
        assert "layers.0.shortcut_post_norm.weight" in state_keys

        block = block.cuda()
        block.train()

        hidden_states = torch.randn(
            16, 2, block.config.hidden_size, device=torch.cuda.current_device(), requires_grad=True
        )
        attention_mask = None
        if compute_symbol == Symbols.ATTENTION:
            attention_mask = torch.triu(
                torch.ones(1, 1, 16, 16, dtype=torch.bool, device=hidden_states.device), diagonal=1
            )
            compute_layer = shortcut.compute_layer
            assert compute_layer.supports_split_output_projection()

            def fail_if_mlp_runs(*args, **kwargs):
                pytest.fail("attention shortcut output projection must not execute an MLP")

            monkeypatch.setattr(compute_layer, "_forward_mlp", fail_if_mlp_runs)

        output = block(hidden_states, attention_mask=attention_mask)
        output.float().square().mean().backward()

        assert output.shape == hidden_states.shape
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

    def test_gdn_inference_spec(self):
        """The inference stack must materialize GDN rather than its IdentityOp default."""
        gdn_spec = hybrid_inference_stack_spec.submodules.gdn_layer
        assert gdn_spec.module is TransformerLayer
        assert gdn_spec.submodules.self_attention.module is GatedDeltaNet

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
