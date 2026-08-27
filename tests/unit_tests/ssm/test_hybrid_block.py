# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from copy import deepcopy

import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import (
    HybridModel,
    _get_hash_moe_layer_threshold,
    _validate_hash_moe_pipeline_placement,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA_KDA, GatedDeltaNet, KimiDeltaAttention
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


@pytest.mark.parametrize("n_hash_layers", [-3, -1, 0])
def test_non_positive_hash_moe_count_has_disabled_threshold(n_hash_layers):
    """Non-positive hash-MoE counts normalize to the disabled threshold."""
    assert _get_hash_moe_layer_threshold(Symbols.MOE, n_hash_layers) == 0


@pytest.mark.internal
class TestHybridBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=[
                'tp',
                'pp',
                'embd',
                'cp',
                'dp_cp',
                'ep',
                'expt_tp',
                'expt_dp',
                'tp_ep',
                'tp_cp',
                'tp_dp_cp',
            ]
        )

    @staticmethod
    def _non_fused_norm_submodules():
        """Un-fuse the TE layernorm+linear pairs so the explicit norm modules exist.

        The default dense hybrid spec folds each norm into the following TE linear,
        leaving IdentityOp placeholders that cannot exercise the norm checkpoints.
        """
        submodules = deepcopy(hybrid_stack_spec.submodules)
        attention_submodules = submodules.attention_layer.submodules
        attention_submodules.input_layernorm = TENorm
        attention_submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
        mlp_submodules = submodules.mlp_layer.submodules
        mlp_submodules.pre_mlp_layernorm = TENorm
        mlp_submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear
        return submodules

    def get_mamba_block(self, layer_pattern, enable_hyper_connections=False):
        layer_type_list = validate_segment_layers(layer_pattern)
        mhc_kwargs = (
            {"enable_hyper_connections": True, "hidden_dropout": 0.0, "mhc_sinkhorn_iterations": 5}
            if enable_hyper_connections
            else {}
        )
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            **mhc_kwargs,
        )
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

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

    def get_dsa_mamba_block(self, layer_pattern, enable_hyper_connections=False):
        layer_type_list = validate_segment_layers(layer_pattern)
        mhc_kwargs = (
            {"enable_hyper_connections": True, "hidden_dropout": 0.0, "mhc_sinkhorn_iterations": 5}
            if enable_hyper_connections
            else {}
        )
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
            **mhc_kwargs,
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
            hidden_size=256,
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
        return HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
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

    @pytest.mark.timeout(60)
    def test_hash_moe_hyper_connection_full_recompute(self):
        """Full recompute preserves input IDs through the hybrid mHC wrapper."""
        block = self.get_hybrid_block(
            Symbols.MOE,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            num_moe_experts=4,
            moe_ffn_hidden_size=64,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_router_dtype="fp32",
            moe_n_hash_layers=1,
            actual_vocab_size=128,
            add_bias_linear=False,
        ).cuda()
        block.train()

        sequence_length, micro_batch_size = 8, 2
        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            block.config.hidden_size,
            device="cuda",
            requires_grad=True,
        )
        input_ids = torch.randint(
            0, block.config.actual_vocab_size, (micro_batch_size, sequence_length), device="cuda"
        )

        assert isinstance(block.layers[0], HyperConnectionHybridLayer)
        assert block.layers[0].inner_layer.mlp.router.is_hash_layer

        output = block(hidden_states, attention_mask=None, input_ids=input_ids)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

        output.float().sum().backward()
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    def test_hash_moe_counts_only_moe_layers(self):
        """Hash routing derives a global layer threshold from the MoE positions."""
        layer_pattern = (Symbols.MLP + Symbols.MOE) * 4
        mtp_pattern = Symbols.MOE
        config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_pattern),
            mtp_num_layers=1,
            num_attention_heads=4,
            use_cpu_initialization=True,
            num_moe_experts=4,
            moe_ffn_hidden_size=64,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_router_dtype="fp32",
            moe_n_hash_layers=3,
            actual_vocab_size=128,
            add_bias_linear=False,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern=f"{layer_pattern}/{mtp_pattern}",
            pg_collection=self.get_pg_collection(),
        )
        block = model.decoder

        moe_layers = [
            layer
            for layer_type, layer in zip(block.layer_type_list, block.layers)
            if layer_type == Symbols.MOE
        ]
        routers = [layer.mlp.router for layer in moe_layers]

        assert [layer.layer_number for layer in moe_layers] == [2, 4, 6, 8]
        assert [router.is_hash_layer for router in routers] == [True, True, True, False]
        assert [router.hash_moe_layer_threshold for router in routers] == [6, 6, 6, 6]
        mtp_router = model.mtp.layers[0].mtp_model_layer.layers[0].mlp.router
        assert mtp_router.hash_moe_layer_threshold == 6
        assert not mtp_router.is_hash_layer
        assert model.config.moe_n_hash_layers == 3

    def test_hash_moe_pipeline_placement_validation(self):
        """A stage without the embedding cannot own a hash-routed MoE layer."""
        layer_pattern = Symbols.MAMBA + Symbols.MOE
        config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_pattern),
            num_attention_heads=4,
            use_cpu_initialization=True,
            is_hybrid_model=True,
            num_moe_experts=4,
            moe_ffn_hidden_size=64,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_router_dtype="fp32",
            moe_n_hash_layers=1,
            actual_vocab_size=128,
            add_bias_linear=False,
        )

        with pytest.raises(ValueError, match="same pipeline/virtual-pipeline stage"):
            HybridModel(
                config=config,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=128,
                max_sequence_length=8,
                hybrid_layer_pattern=layer_pattern,
                pre_process=False,
                pg_collection=self.get_pg_collection(),
            )

    def test_hash_moe_pipeline_placement_allows_non_hash_stage(self):
        """A later stage from a pipe-free split is valid when its MoE is not hash-routed."""
        _validate_hash_moe_pipeline_placement(
            [Symbols.MAMBA, Symbols.MOE],
            layer_offset=2,
            hash_moe_layer_threshold=2,
            pre_process=False,
        )

    def test_hybrid_mtp_rejects_expert_parallel_overlap_before_build(self, monkeypatch):
        """Reject overlap before constructing any HybridModel submodule."""
        config = TransformerConfig(
            hidden_size=256, num_layers=1, num_attention_heads=4, use_cpu_initialization=True
        )
        # Mutate after generic config validation to exercise the pattern-specific guard.
        config.overlap_moe_expert_parallel_comm = True

        def fail_build(*args, **kwargs):
            pytest.fail("HybridModel submodule construction must not begin")

        monkeypatch.setattr("megatron.core.models.hybrid.hybrid_model.build_module", fail_build)

        with pytest.raises(ValueError, match="Hybrid MTP does not support"):
            HybridModel(
                config=config,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=128,
                max_sequence_length=8,
                hybrid_layer_pattern=f"{Symbols.MAMBA}/{Symbols.MAMBA}",
                pg_collection=self.get_pg_collection(),
            )

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

    def test_hyper_connection_layer_wrappers(self):
        """mHC wraps each hybrid layer while preserving the layer type underneath."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern, enable_hyper_connections=True)
        layers = block.layers
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in layers)
        assert isinstance(layers[0].inner_layer, MambaLayer)
        assert isinstance(layers[1].inner_layer, TransformerLayer)
        assert isinstance(layers[1].inner_layer.self_attention, SelfAttention)
        assert isinstance(layers[2].inner_layer, TransformerLayer)
        assert isinstance(layers[2].inner_layer.mlp, MLP)

    def test_hyper_connection_recompute_plan_for_hybrid_layers(self):
        """HybridStack creates per-layer mHC recompute managers when requested."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
            recompute_granularity="selective",
            recompute_modules=["core_attn", "mhc"],
        )
        block = HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

        managers, block_ends = block._build_mhc_recompute_layer_plan(use_mhc_recompute=True)
        assert len(managers) == len(block.layers)
        assert all(manager is not None for manager in managers)
        assert block_ends[-1] is True

    @pytest.mark.timeout(60)
    def test_hyper_connection_mhc_recompute_bitwise(self):
        """mHC selective recompute is bitwise identical to the eager path."""
        seed = 123
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        layer_type_list = validate_segment_layers(layer_pattern)
        arch_kwargs = dict(
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
            add_bias_linear=False,
        )

        def build_block(**recompute_kwargs):
            model_parallel_cuda_manual_seed(seed)
            torch.manual_seed(seed)
            config = TransformerConfig(
                hidden_size=256,
                num_layers=len(layer_type_list),
                num_attention_heads=4,
                use_cpu_initialization=True,
                **arch_kwargs,
                **recompute_kwargs,
            )
            return HybridStack(
                config,
                self._non_fused_norm_submodules(),
                layer_type_list=layer_type_list,
                pp_layer_offset=0,
                pg_collection=self.get_pg_collection(),
            ).cuda()

        torch.manual_seed(seed)
        hidden_states = torch.randn(32, 2, 256, device="cuda")
        attention_mask = torch.ones((2, 1, 32, 32), dtype=bool, device="cuda")

        def run(block, inputs):
            block.train()
            output = block(inputs, attention_mask=attention_mask)
            output.float().sum().backward()
            grads = {
                name: param.grad.detach().float().cpu()
                for name, param in block.named_parameters()
                if param.grad is not None
            }
            return output.detach().float().cpu(), grads

        baseline = build_block()
        baseline_output, baseline_grads = run(
            baseline, hidden_states.detach().clone().requires_grad_()
        )
        del baseline
        torch.cuda.empty_cache()

        recomputed = build_block(recompute_granularity="selective", recompute_modules=["mhc"])
        attention_layer = recomputed.layers[1].inner_layer
        mlp_layer = recomputed.layers[2].inner_layer
        assert attention_layer.mhc_checkpoint_input_layernorm
        assert mlp_layer.mhc_checkpoint_pre_mlp_layernorm

        recomputed_output, recomputed_grads = run(
            recomputed, hidden_states.detach().clone().requires_grad_()
        )

        assert torch.equal(recomputed_output, baseline_output)
        assert set(recomputed_grads) == set(baseline_grads)
        for name, baseline_grad in baseline_grads.items():
            assert torch.equal(recomputed_grads[name], baseline_grad), name

        for checkpoint in (
            attention_layer.input_layernorm_checkpoint,
            mlp_layer.pre_mlp_norm_checkpoint,
        ):
            assert checkpoint in checkpoint.ckpt_manager.checkpoints
            assert checkpoint.ctx is None
            assert checkpoint.outputs is None

    @pytest.mark.timeout(60)
    def test_hyper_connection_mlp_fast_path_discards_layernorm_checkpoint(self):
        """The hybrid mHC MLP fast path releases selective layernorm activations."""
        layer_type_list = validate_segment_layers(Symbols.MLP)
        config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
            recompute_granularity="selective",
            recompute_modules=["layernorm"],
        )
        block = HybridStack(
            config,
            self._non_fused_norm_submodules(),
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        ).cuda()
        block.train()

        hidden_states = torch.randn(
            8, 2, block.config.hidden_size, device="cuda", requires_grad=True
        )
        output = block(hidden_states, attention_mask=None)

        layer = block.layers[0]
        assert isinstance(layer, HyperConnectionHybridLayer)
        inner_layer = layer.inner_layer
        assert inner_layer.recompute_pre_mlp_layernorm
        checkpoint = inner_layer.pre_mlp_norm_checkpoint
        assert checkpoint.ckpt_manager is None
        assert checkpoint.outputs[0].untyped_storage().nbytes() == 0

        output.sum().backward()
        assert checkpoint.ctx is None
        assert checkpoint.outputs is None

    def test_hyper_connection_gpu_forward(self):
        """mHC-enabled HybridStack expands internally and contracts back at the output."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP
        block = self.get_mamba_block(layer_pattern, enable_hyper_connections=True)
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

    def test_hyper_connection_gdn_gpu_forward(self):
        """mHC runs through GDN, attention, and Mamba hybrid layers."""
        layer_pattern = Symbols.GDN + Symbols.ATTENTION + Symbols.MAMBA
        layer_type_list = validate_segment_layers(layer_pattern)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
        )
        block = HybridStack(
            transformer_config,
            hybrid_stack_spec.submodules,
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
        ).cuda()
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape == (sequence_length, micro_batch_size, block.config.hidden_size)

    def test_hyper_connection_dsa_layer_wrappers(self):
        """mHC wraps DeepSeek-style DSA and MLP split layers."""
        layer_pattern = Symbols.MAMBA + Symbols.DS_ATTENTION + Symbols.MLP
        block = self.get_dsa_mamba_block(layer_pattern, enable_hyper_connections=True)
        layers = block.layers
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in layers)
        assert isinstance(layers[0].inner_layer, MambaLayer)
        assert isinstance(layers[1].inner_layer, TransformerLayer)
        assert isinstance(layers[1].inner_layer.self_attention, AbsorbedMLASelfAttention)
        assert isinstance(layers[1].inner_layer.self_attention.core_attention, DSAttention)
        assert isinstance(layers[2].inner_layer, TransformerLayer)
        assert isinstance(layers[2].inner_layer.mlp, MLP)

    def test_hyper_connection_pipeline_boundary_shapes(self):
        """HybridStack keeps n-stream tensors between PP stages and contracts at the end."""
        layer_type_list = validate_segment_layers(Symbols.MAMBA)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=5,
        )
        modules = hybrid_stack_spec.submodules
        first_stage = HybridStack(
            transformer_config,
            modules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            post_process=False,
            pg_collection=self.get_pg_collection(),
        ).cuda()
        last_stage = HybridStack(
            transformer_config,
            modules,
            pre_process=False,
            layer_type_list=layer_type_list,
            pp_layer_offset=1,
            post_process=True,
            pg_collection=self.get_pg_collection(),
        ).cuda()

        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, transformer_config.hidden_size), device='cuda'
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool, device='cuda'
        )

        pp_hidden = first_stage(hidden_states, attention_mask=attention_mask)
        assert pp_hidden.shape == (
            sequence_length,
            micro_batch_size,
            transformer_config.hidden_size * transformer_config.num_residual_streams,
        )

        last_stage.set_input_tensor(pp_hidden.detach())
        output = last_stage(hidden_states, attention_mask=attention_mask)
        assert output.shape == (sequence_length, micro_batch_size, transformer_config.hidden_size)

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

    @pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA with KDA support is not installed.")
    def test_kda_layer_type(self):
        """K builds a TransformerLayer wrapping KimiDeltaAttention."""
        block = self.get_hybrid_block(
            Symbols.KDA,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            activation_func=torch.nn.functional.silu,
            add_bias_linear=False,
        )
        assert isinstance(block.layers[0], TransformerLayer)
        assert isinstance(block.layers[0].self_attention, KimiDeltaAttention)

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
        block = self.get_dsa_mamba_block(layer_pattern)
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
            block = self.get_dsa_mamba_block(layer_pattern)

    def test_mla_layer_types(self):
        """+ builds standard MLA rather than DSA."""
        layer_pattern = Symbols.MAMBA + Symbols.MLA + Symbols.MAMBA
        block = self.get_mla_hybrid_block(layer_pattern)
        layers = block.layers
        assert isinstance(layers[0], MambaLayer)
        assert isinstance(layers[1], TransformerLayer)
        assert isinstance(layers[1].self_attention, MLASelfAttention)
        assert isinstance(layers[1].self_attention.core_attention, TEDotProductAttention)
        assert isinstance(layers[2], MambaLayer)

    def test_mixed_attention_and_mla_layer_types(self):
        """* and + in the same block fail."""
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLA + Symbols.MAMBA
        with pytest.raises(ValueError):
            self.get_mla_hybrid_block(layer_pattern)


_HAVE_MAMBA_SSM = __import__("importlib").util.find_spec("mamba_ssm") is not None
requires_mamba_ssm = pytest.mark.skipif(
    not _HAVE_MAMBA_SSM, reason="mamba_ssm not installed in this environment"
)


@pytest.mark.internal
class TestAttnResHybridBlock:
    """Attention residuals in hybrid stacks (AttnResHybridLayer + HybridStack).

    The wrapper is entry-type agnostic, so the always-on tests use pure
    transformer-entry patterns (runnable in containers without mamba_ssm);
    Mamba-entry variants are guarded by ``requires_mamba_ssm`` and run in the
    CI mamba bucket.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

    def _make_config(self, num_layers, block_layers=1, enable=True, eps=1e-6):
        attn_res_kwargs = (
            {"enable_attention_residuals": True, "attn_res_block_layers": block_layers}
            if enable
            else {}
        )
        return TransformerConfig(
            hidden_size=256,
            num_layers=num_layers,
            num_attention_heads=4,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layernorm_epsilon=eps,
            **attn_res_kwargs,
        )

    def _make_stack(self, config, layer_type_list, pp_layer_offset=0, pre=True, post=True):
        return HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=pp_layer_offset,
            pre_process=pre,
            post_process=post,
            pg_collection=self.get_pg_collection(),
        )

    def test_layer_wrappers_and_boundaries(self):
        from megatron.core.models.hybrid.hybrid_block import AttnResHybridLayer

        layer_type_list = validate_segment_layers(
            Symbols.ATTENTION + Symbols.MLP + Symbols.ATTENTION + Symbols.MLP
        )
        config = self._make_config(len(layer_type_list), block_layers=2)
        block = self._make_stack(config, layer_type_list)
        assert all(isinstance(layer, AttnResHybridLayer) for layer in block.layers)
        # k=2 entries per block: boundaries at entries 1 and 3.
        assert [layer.attn_res_is_block_start for layer in block.layers] == [
            True,
            False,
            True,
            False,
        ]
        assert [layer.attn_res_num_sources for layer in block.layers] == [1, 1, 2, 2]
        assert isinstance(block.final_attn_res, torch.nn.Module)

    def test_gpu_forward(self):
        layer_type_list = validate_segment_layers(Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP)
        config = self._make_config(len(layer_type_list), block_layers=1)
        block = self._make_stack(config, layer_type_list).cuda()
        sequence_length, micro_batch_size = 32, 2
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, config.hidden_size), device='cuda'
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool, device='cuda'
        )
        output = block(hidden_states, attention_mask=attention_mask)
        assert output.shape == (sequence_length, micro_batch_size, config.hidden_size)

    @pytest.mark.parametrize(
        "layer_pattern",
        [
            Symbols.ATTENTION + Symbols.MLP + Symbols.ATTENTION + Symbols.MLP,
            pytest.param(
                Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + Symbols.ATTENTION,
                marks=requires_mamba_ssm,
            ),
        ],
    )
    def test_init_equivalence_forward(self, layer_pattern):
        """Zero-init AttnRes hybrid stack matches the baseline at the first forward.

        Zero pseudo-queries make every aggregation the exact mean of the depth
        sources, which partition the baseline residual sum; all consumers are
        (scale-invariant up to eps) norms, and the wrapper's delta
        reconstruction is exact — so with a tiny eps the outputs must agree.
        """
        layer_type_list = validate_segment_layers(layer_pattern)

        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        baseline = self._make_stack(
            self._make_config(len(layer_type_list), enable=False, eps=1e-12), layer_type_list
        ).cuda()

        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        attnres = self._make_stack(
            self._make_config(len(layer_type_list), block_layers=1, eps=1e-12), layer_type_list
        ).cuda()

        # The wrapper nests inner-layer keys under `inner_layer.`; remap the
        # baseline state dict accordingly and copy the shared weights.
        remapped = {}
        for key, value in baseline.state_dict().items():
            if key.startswith("layers."):
                prefix, rest = key.split(".", 2)[0:2], key.split(".", 2)[2]
                remapped[f"{prefix[0]}.{prefix[1]}.inner_layer.{rest}"] = value
            else:
                remapped[key] = value
        missing, unexpected = attnres.load_state_dict(remapped, strict=False)
        assert not unexpected, unexpected
        assert missing and all("attn_res" in key for key in missing), missing

        sequence_length, micro_batch_size = 32, 2
        torch.manual_seed(7)
        hidden_states = torch.randn((sequence_length, micro_batch_size, 256), device='cuda')
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool, device='cuda'
        )

        with torch.no_grad():
            out_base = baseline(hidden_states.clone(), attention_mask=attention_mask)
            out_attn = attnres(hidden_states.clone(), attention_mask=attention_mask)

        torch.testing.assert_close(out_attn, out_base, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("block_layers", [1, 2])
    def test_pipeline_boundary_payload_shapes(self, block_layers):
        """Depth sources + partial cross PP boundaries as one seq-dim-concat payload."""
        from megatron.core.transformer.attention_residual import attn_res_num_payload_slices

        stage0_types = validate_segment_layers(Symbols.ATTENTION + Symbols.MLP)
        stage1_types = validate_segment_layers(Symbols.MLP + Symbols.ATTENTION)
        config = self._make_config(4, block_layers=block_layers)

        first_stage = self._make_stack(
            config, stage0_types, pp_layer_offset=0, pre=True, post=False
        ).cuda()
        last_stage = self._make_stack(
            config, stage1_types, pp_layer_offset=2, pre=False, post=True
        ).cuda()

        sequence_length, micro_batch_size = 32, 2
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, config.hidden_size), device='cuda'
        )
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool, device='cuda'
        )

        payload = first_stage(hidden_states, attention_mask=attention_mask)
        num_slices = attn_res_num_payload_slices(2, block_layers)
        assert payload.shape == (num_slices * sequence_length, micro_batch_size, config.hidden_size)

        last_stage.set_input_tensor(payload.detach())
        output = last_stage(hidden_states, attention_mask=attention_mask)
        assert output.shape == (sequence_length, micro_batch_size, config.hidden_size)
