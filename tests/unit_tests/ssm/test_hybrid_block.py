# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
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

    def get_dsa_mamba_block(self, layer_pattern):
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

    def test_invalid_layer_types_cause_failure(self):
        invalid_symbol = '+'
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
        """D symbol creates a TransformerLayer with MLASelfAttention."""
        layer_pattern = Symbols.MAMBA + Symbols.DS_ATTENTION + Symbols.MAMBA
        block = self.get_dsa_mamba_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, MLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, DSAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_dsa_layer_types(self):
        """* and D in the same block fail."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.DS_ATTENTION + Symbols.MAMBA
        with pytest.raises(ValueError):
            block = self.get_dsa_mamba_block(layer_pattern)
