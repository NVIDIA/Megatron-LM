# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.models.hybrid.hybrid_block as hybrid_block_module
import megatron.core.transformer.utils as transformer_utils
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.models.hybrid.shortcut_block import ShortcutExecutionMode, ShortcutMoEBlock
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _make_pg_collection():
    return SimpleNamespace(pp=None, tp=None, cp=SimpleNamespace(size=lambda: 1), tp_cp=None)


@pytest.mark.parametrize(
    ("layer_pattern", "expected_spec_names"),
    [
        (
            Symbols.MAMBA + Symbols.GDN + Symbols.ATTENTION + Symbols.MLP + Symbols.MOE,
            ["mamba_layer", "gdn_layer", "attention_layer", "mlp_layer", "moe_layer"],
        ),
        (Symbols.DS_ATTENTION + Symbols.MLA, ["dsa_layer", "mla_layer"]),
    ],
)
def test_all_layer_configs_route_to_matching_specs(monkeypatch, layer_pattern, expected_spec_names):
    """Each config marker selects its matching layer spec and config instance."""

    class BuiltLayer(torch.nn.Module):

        def __init__(self, config, layer_number):
            super().__init__()
            self.config = config
            self.layer_number = layer_number

    build_calls = []

    def fake_build_module(module_spec, **kwargs):
        build_calls.append((module_spec, kwargs))
        return BuiltLayer(kwargs["config"], kwargs["layer_number"])

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)

    config = MLATransformerConfig(
        num_layers=len(layer_pattern), hidden_size=64, num_attention_heads=4
    )
    layer_config_list = validate_segment_layers(layer_pattern, config)
    submodules = hybrid_stack_spec.submodules
    expected_specs = [getattr(submodules, spec_name) for spec_name in expected_spec_names]

    block = HybridStack(
        config=config,
        submodules=submodules,
        layer_config_list=layer_config_list,
        pre_process=False,
        pp_layer_offset=5,
        post_layer_norm=False,
        post_process=False,
        pg_collection=_make_pg_collection(),
        name="decoder",
    )

    assert "layer_type_list" not in block.__dict__
    assert block.layer_type_list == list(layer_pattern)
    assert [module_spec for module_spec, _ in build_calls] == expected_specs
    assert all(
        kwargs["config"] is layer_config
        for (_, kwargs), layer_config in zip(build_calls, layer_config_list)
    )
    expected_layer_numbers = list(range(6, 6 + len(layer_pattern)))
    assert [kwargs["layer_number"] for _, kwargs in build_calls] == expected_layer_numbers
    assert [layer.layer_number for layer in block.layers] == expected_layer_numbers


def test_cp_layouts_are_selected_by_layer_config_type(monkeypatch):
    """Each layer config type selects the corresponding context-parallel layout."""

    class BuiltLayer(torch.nn.Module):

        def __init__(self, config, layer_number):
            super().__init__()
            self.config = config
            self.layer_number = layer_number

    layout_manager_kwargs = {}

    class CapturingLayoutManager:

        def __init__(self, **kwargs):
            layout_manager_kwargs.update(kwargs)

    monkeypatch.setattr(hybrid_block_module, "ContextParallelLayoutManager", CapturingLayoutManager)
    monkeypatch.setattr(
        hybrid_block_module,
        "build_module",
        lambda module_spec, **kwargs: BuiltLayer(kwargs["config"], kwargs["layer_number"]),
    )

    config = MLATransformerConfig(
        num_layers=7,
        hidden_size=64,
        num_attention_heads=4,
        linear_cp_layout="contiguous",
        attention_cp_layout="zigzag",
    )
    layer_config_list = validate_segment_layers("MG*-E", config) + validate_segment_layers(
        "D+", config
    )

    HybridStack(
        config=config,
        submodules=hybrid_stack_spec.submodules,
        layer_config_list=layer_config_list,
        pre_process=False,
        post_layer_norm=False,
        post_process=False,
        pg_collection=SimpleNamespace(
            pp=None, tp=None, cp=SimpleNamespace(size=lambda: 2), tp_cp=None
        ),
    )

    assert layout_manager_kwargs["layer_layouts"] == (
        "contiguous",
        "contiguous",
        "zigzag",
        "contiguous",
        "contiguous",
        "zigzag",
        "zigzag",
    )
    assert layout_manager_kwargs["boundary_layout"] == "contiguous"


def test_hybrid_stack_rejects_layer_config_subclasses(monkeypatch):
    """Layer config subclasses must be registered as distinct layer types."""

    class CustomMambaLayerConfig(MambaLayerConfig):
        pass

    class BuiltLayer(torch.nn.Module):

        def __init__(self, config, layer_number):
            super().__init__()
            self.config = config
            self.layer_number = layer_number

    build_calls = []

    def fake_build_module(module_spec, **kwargs):
        build_calls.append(module_spec)
        return BuiltLayer(kwargs["config"], kwargs["layer_number"])

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)

    root_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
    layer_config = CustomMambaLayerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
    with pytest.raises(
        ValueError, match="Unexpected hybrid layer config type: CustomMambaLayerConfig"
    ):
        HybridStack(
            config=root_config,
            submodules=hybrid_stack_spec.submodules,
            layer_config_list=[layer_config],
            pre_process=False,
            post_layer_norm=False,
            post_process=False,
            pg_collection=_make_pg_collection(),
        )

    assert build_calls == []


def test_layer_type_list_rejects_unsupported_tp_overlap():
    """The positional layer-type API rejects unsupported TP overlap."""
    config = MLATransformerConfig(
        num_layers=3, hidden_size=64, num_attention_heads=4, tp_comm_overlap=True
    )
    with pytest.raises(
        ValueError, match="TP communication overlap is not supported with hybrid MLA layers"
    ):
        HybridStack(
            config,
            hybrid_stack_spec.submodules,
            False,
            [Symbols.MAMBA, Symbols.MLA, Symbols.MLP],
            post_layer_norm=False,
            post_process=False,
            pg_collection=_make_pg_collection(),
        )

    assert config.tp_comm_overlap is True


def test_layer_config_list_rejects_unsupported_tp_overlap():
    """Explicit per-layer configs are validated using their own overlap setting."""
    root_config = MLATransformerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4, tp_comm_overlap=False
    )
    layer_config = MLALayerConfig(
        num_layers=1, hidden_size=64, num_attention_heads=4, tp_comm_overlap=True
    )

    with pytest.raises(
        ValueError, match="TP communication overlap is not supported with hybrid MLA layers"
    ):
        HybridStack(
            config=root_config,
            submodules=hybrid_stack_spec.submodules,
            layer_config_list=[layer_config],
            pre_process=False,
            post_layer_norm=False,
            post_process=False,
            pg_collection=_make_pg_collection(),
        )

    assert root_config.tp_comm_overlap is False
    assert layer_config.tp_comm_overlap is True


def test_layer_type_list_configs_follow_root_sequence_parallel_mutations(monkeypatch):
    """Legacy layer symbols still create configs tracked by sequence-parallel utilities."""

    class BuiltLayer(torch.nn.Module):

        def __init__(self, config, layer_number):
            super().__init__()
            self.config = config
            self.layer_number = layer_number

    submodules = hybrid_stack_spec.submodules

    def fake_build_module(module_spec, **kwargs):
        return BuiltLayer(kwargs["config"], kwargs["layer_number"])

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)

    config = MLATransformerConfig(num_layers=3, hidden_size=64, num_attention_heads=4)
    with pytest.warns(
        DeprecationWarning,
        match=r"DEPRECATED\(layer_type_list\): please use `layer_config_list` instead",
    ):
        block = HybridStack(
            config,
            submodules,
            False,
            [Symbols.MAMBA, Symbols.MLA, Symbols.MLP],
            post_layer_norm=False,
            post_process=False,
            pg_collection=_make_pg_collection(),
        )
    layer_config_list = block.layer_config_list

    assert "layer_type_list" not in block.__dict__
    assert block.layer_type_list == [Symbols.MAMBA, Symbols.MLA, Symbols.MLP]
    assert type(layer_config_list) is list
    assert [type(layer_config) for layer_config in layer_config_list] == [
        MambaLayerConfig,
        MLALayerConfig,
        MLPLayerConfig,
    ]
    assert len({id(layer_config) for layer_config in layer_config_list}) == len(layer_config_list)
    assert all(layer_config is not config for layer_config in layer_config_list)
    assert all(
        layer.config is layer_config for layer, layer_config in zip(block.layers, layer_config_list)
    )

    block.position_embedding_type = "rope"
    config.sequence_parallel = True
    for layer_config in layer_config_list:
        layer_config.sequence_parallel = True

    monkeypatch.setattr(transformer_utils, "_sequence_parallel_attr_cache", None)
    transformer_utils.set_model_to_sequence_parallel(block, set_to=False)

    assert config.sequence_parallel is False
    assert all(layer_config.sequence_parallel is False for layer_config in layer_config_list)


def test_explicit_layer_config_mutations_are_isolated(monkeypatch):
    """Mutating one explicitly supplied layer config does not affect the others."""

    class BuiltLayer(torch.nn.Module):

        def __init__(self, config, layer_number):
            super().__init__()
            self.config = config
            self.layer_number = layer_number

    submodules = hybrid_stack_spec.submodules

    def fake_build_module(module_spec, **kwargs):
        if module_spec is submodules.mla_layer:
            kwargs["config"].add_bias_linear = False
        return BuiltLayer(kwargs["config"], kwargs["layer_number"])

    monkeypatch.setattr(hybrid_block_module, "build_module", fake_build_module)

    root_config = MLATransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
    layer_configs = validate_segment_layers(Symbols.MLA + Symbols.MLP, root_config)
    HybridStack(
        config=root_config,
        submodules=submodules,
        layer_config_list=layer_configs,
        pre_process=False,
        post_layer_norm=False,
        post_process=False,
        pg_collection=_make_pg_collection(),
    )

    assert type(layer_configs) is list
    assert root_config.add_bias_linear is True
    assert [layer_config.add_bias_linear for layer_config in layer_configs] == [False, True]


@pytest.mark.parametrize(
    ("provide_layer_type_list", "provide_layer_config_list"),
    [(False, False), (True, True)],
    ids=["neither", "both"],
)
def test_hybrid_stack_requires_exactly_one_layer_list(
    provide_layer_type_list, provide_layer_config_list
):
    """HybridStack requires exactly one legacy symbol list or per-layer config list."""
    config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
    layer_type_list = [Symbols.MAMBA] if provide_layer_type_list else None
    layer_config_list = (
        validate_segment_layers(Symbols.MAMBA, config) if provide_layer_config_list else None
    )

    with pytest.raises(
        ValueError, match="Exactly one of layer_type_list or layer_config_list must be provided"
    ):
        HybridStack(
            config=config,
            submodules=hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            layer_config_list=layer_config_list,
            pre_process=False,
            post_layer_norm=False,
            post_process=False,
            pg_collection=_make_pg_collection(),
        )


def test_hybrid_stack_rejects_multi_character_layer_type():
    """The legacy list treats each entry as one layer symbol."""
    config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)

    with pytest.raises(ValueError, match="Each entry in layer_type_list must be a single"):
        HybridStack(
            config=config,
            submodules=hybrid_stack_spec.submodules,
            layer_type_list=[Symbols.MAMBA + Symbols.ATTENTION],
            pre_process=False,
            post_layer_norm=False,
            post_process=False,
            pg_collection=_make_pg_collection(),
        )


def test_mamba_state_shapes_are_selected_by_layer_config_type():
    """Mamba state shape lookup does not depend on layer symbols or module methods alone."""

    attention_config = object.__new__(AttentionLayerConfig)
    mamba_config = object.__new__(MambaLayerConfig)
    attention_shapes = ((1,), (2,))
    mamba_shapes = ((3,), (4,))
    block = SimpleNamespace(
        layer_config_list=[attention_config, mamba_config],
        layers=[
            SimpleNamespace(mamba_state_shapes_per_request=lambda: attention_shapes),
            SimpleNamespace(mamba_state_shapes_per_request=lambda: mamba_shapes),
        ],
    )

    assert HybridStack.mamba_state_shapes_per_request(block) == mamba_shapes

    block.layer_config_list = [attention_config]
    block.layers = block.layers[:1]
    assert HybridStack.mamba_state_shapes_per_request(block) is None


def test_hybrid_stack_rejects_same_named_config_type():
    root_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
    same_named_config_class = type("MambaLayerConfig", (TransformerConfig,), {})
    layer_config = same_named_config_class(num_layers=1, hidden_size=64, num_attention_heads=4)

    with pytest.raises(ValueError, match="Unexpected hybrid layer config type: MambaLayerConfig"):
        HybridStack(
            config=root_config,
            submodules=hybrid_stack_spec.submodules,
            layer_config_list=[layer_config],
            pre_process=False,
            post_layer_norm=False,
            post_process=False,
            pg_collection=_make_pg_collection(),
        )


@pytest.mark.internal
class TestHybridBlock:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def get_pg_collection(self):
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

    def get_hybrid_block(self, layer_pattern, **config_kwargs):
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_pattern),
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_kwargs,
        )
        layer_config_list = validate_segment_layers(layer_pattern, transformer_config)
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_config_list=layer_config_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_dsa_hybrid_block(self, layer_pattern):
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_pattern),
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
        layer_config_list = validate_segment_layers(layer_pattern, transformer_config)
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_config_list=layer_config_list,
            pp_layer_offset=0,
            pg_collection=self.get_pg_collection(),
        )

    def get_mla_hybrid_block(self, layer_pattern):
        transformer_config = MLATransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=len(layer_pattern),
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
        layer_config_list = validate_segment_layers(layer_pattern, transformer_config)
        modules = hybrid_stack_spec.submodules
        return HybridStack(
            transformer_config,
            modules,
            layer_config_list=layer_config_list,
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
        assert len({id(config) for config in block.layer_config_list}) == len(layer_pattern)
        assert all(
            layer.config is layer_config
            for layer, layer_config in zip(block.layers, block.layer_config_list)
        )

    def test_shortcut_pair_is_one_registered_block(self):
        block = self.get_hybrid_block(
            Symbols.MAMBA + Symbols.MOE,
            num_moe_experts=1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=False,
            moe_shared_expert_intermediate_size=256,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

        assert len(block.layers) == 1
        shortcut = block.layers[0]
        assert isinstance(shortcut, ShortcutMoEBlock)
        assert shortcut.execution_mode == ShortcutExecutionMode.EAGER_SERIAL
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

    @pytest.mark.parametrize("parallel", [False, True], ids=["serial", "overlap"])
    def test_shortcut_pair_eager_forward_backward(self, parallel):
        block = self.get_hybrid_block(
            Symbols.MAMBA + Symbols.MOE,
            num_moe_experts=1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="allgather",
            moe_shortcut_connection=True,
            moe_shortcut_parallel=parallel,
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

    def test_invalid_layer_types_cause_failure(self):
        invalid_pattern_char = 'X'
        assert not layer_utils.is_valid_symbol(invalid_pattern_char)  # sanity check.
        layer_pattern = Symbols.MAMBA + Symbols.ATTENTION + Symbols.MLP + invalid_pattern_char
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
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_pattern),
            num_attention_heads=4,
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
        )
        layer_config_list = validate_segment_layers(layer_pattern, transformer_config)
        modules = hybrid_stack_spec.submodules
        block = HybridStack(
            transformer_config,
            modules,
            layer_config_list=layer_config_list,
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
