# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import os
from datetime import timedelta
from itertools import accumulate

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts import BaseInferenceContext, StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.hybrid.hybrid_block import (
    HyperConnectionHybridLayer,
    HybridStack,
    HybridStackSubmodules,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import divide, is_fa_min_version, is_torch_min_version
from tests.unit_tests.test_utilities import Utils


class _DummyHybridLayer(MegatronModule):
    """Minimal same-shape layer used to test HybridModel/mHC plumbing."""

    def __init__(self, config: TransformerConfig, layer_number: int, **_kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.seen_hidden_shapes = []

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=None,
        **_kwargs,
    ):
        self.seen_hidden_shapes.append(tuple(hidden_states.shape))
        return hidden_states + 0.125 * self.proj(hidden_states)


def _get_dummy_hybrid_stack_spec() -> ModuleSpec:
    """Build a HybridStack spec whose layer symbols all resolve to dummy layers."""
    dummy_layer_spec = ModuleSpec(module=_DummyHybridLayer)
    return ModuleSpec(
        module=HybridStack,
        params={"post_layer_norm": False},
        submodules=HybridStackSubmodules(
            mamba_layer=dummy_layer_spec,
            gdn_layer=dummy_layer_spec,
            attention_layer=dummy_layer_spec,
            dsa_layer=dummy_layer_spec,
            mlp_layer=dummy_layer_spec,
            moe_layer=dummy_layer_spec,
        ),
    )


class TestHybridModel:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        model_config = TransformerConfig(
            num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
            hidden_size=256,  # The Mamba layer places several constraints on this
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        self.model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",  # 1 Mamba, 1 attention, 1 MLP
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, HybridModel)

        assert self.model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1774872

    def test_constructor_with_hyper_connections(self):
        model_config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in model.decoder.layers)
        num_weights = sum([p.numel() for p in model.parameters()])
        assert num_weights > sum([p.numel() for p in self.model.parameters()])

    def test_forward_with_hyper_connections(self):
        model_config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )
        model.cuda()

        sequence_length = model.max_sequence_length
        micro_batch_size = 2
        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == model.vocab_size

    def test_dummy_hybrid_model_with_hyper_connections_forward_backward(self):
        model_config = TransformerConfig(
            num_layers=3,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            hidden_dropout=0.0,
            mhc_sinkhorn_iterations=3,
        )
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=_get_dummy_hybrid_stack_spec(),
            vocab_size=64,
            max_sequence_length=8,
            hybrid_layer_pattern="M*-",
            parallel_output=False,
        )

        assert all(
            isinstance(layer, HyperConnectionHybridLayer) for layer in model.decoder.layers
        )
        assert all(
            isinstance(layer.inner_layer, _DummyHybridLayer) for layer in model.decoder.layers
        )

        model.cuda()
        sequence_length = model.max_sequence_length
        micro_batch_size = 2
        data = torch.arange(sequence_length, dtype=torch.int64, device='cuda')
        input_ids = data.repeat((micro_batch_size, 1))
        position_ids = data.repeat((micro_batch_size, 1))

        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
        )

        assert logits.shape == (micro_batch_size, sequence_length, model.vocab_size)
        assert torch.isfinite(logits).all()

        logits.float().mean().backward()

        for layer in model.decoder.layers:
            assert layer.inner_layer.seen_hidden_shapes == [
                (sequence_length, micro_batch_size, model_config.hidden_size)
            ]
            assert layer.inner_layer.proj.weight.grad is not None
            assert layer.hyper_connection.mapping_proj.weight.grad is not None
            assert torch.isfinite(layer.inner_layer.proj.weight.grad).all()
            assert torch.isfinite(layer.hyper_connection.mapping_proj.weight.grad).all()

    def test_set_input_tensor(self):
        config: TransformerConfig = self.model.config
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape[0] == sequence_length
        assert self.model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.model.decoder.input_tensor.shape[2] == config.hidden_size

    def test_forward(self):
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        self.model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.model.vocab_size

    def test_forward_packed_sequence(self):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        model_config = TransformerConfig(
            num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
            hidden_size=256,  # The Mamba layer places several constraints on this
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,  # Needed for backend=flash
            params_dtype=torch.bfloat16,  # Needed for backend=flash
            attention_backend=AttnBackend.flash,  # Needed for packed sequence
        )
        vocab_size = 100
        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=12,
            hybrid_layer_pattern="M*-",  # 1 Mamba, 1 attention, 1 MLP
        )

        sequence_length = model.max_sequence_length
        micro_batch_size = 1  # must be 1 for packed sequence

        model.cuda()

        data = [i % vocab_size for i in range(sequence_length)]
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        lengths = [4, 3, 5]
        assert sum(lengths) == sequence_length
        positions = [i for n in lengths for i in range(n)]
        position_ids = (
            torch.tensor(positions, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        )
        attention_mask = None

        cumsum = [0] + list(accumulate(lengths))
        cu_seqlens = torch.tensor(cumsum, dtype=torch.int32).cuda()
        max_seqlen = max(lengths)

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_tokens=sequence_length,
        )

        logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == model.vocab_size

    def test_inference(self):
        micro_batch_size = 2
        inference_context: BaseInferenceContext = StaticInferenceContext(
            max_batch_size=micro_batch_size, max_sequence_length=self.model.max_sequence_length
        )
        prompt_length = self.model.max_sequence_length - 1

        self.model.cuda()

        # load-context/first-output-token, step/generate
        for offset in (0, prompt_length):
            if offset == 0:
                sequence_length = prompt_length
            else:
                sequence_length = 1
            inference_context.sequence_len_offset = offset

            data = list(range(sequence_length))
            input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
            position_ids = (
                torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
            )
            attention_mask = torch.ones(
                (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
            ).cuda()

            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inference_context=inference_context,
            )

            assert logits.shape[0] == micro_batch_size
            assert logits.shape[1] == sequence_length
            assert logits.shape[2] == self.model.vocab_size

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))

    def test_layer_numbers(self):
        """
        The layer numbers should start at one (for the embedding # layer) and go up
        incrementally from there. This is required for PEFT to work.
        """
        model = self.model
        for expected, layer in enumerate(model.decoder.layers, start=1):
            assert expected == layer.layer_number, "layer numbers are incorrect"

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"),
        reason="torch.distributed.init_device_mesh requires torch >= 2.4.0",
    )
    @pytest.mark.parametrize("tp_size,cp_size,pp_size", [(2, 1, 4), (1, 1, 8), (8, 1, 1)])
    def test_with_custom_process_groups(self, tmp_path, tp_size, cp_size, pp_size):
        """Test HybridModel with custom process groups."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=pp_size,
        )

        # Create device mesh for custom process groups
        assert torch.distributed.get_world_size() == 8, "Test requires 8 GPUs"

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimensions tp, cp, pp (reversed from device mesh order)
        grid = HyperCommGrid([tp_size, cp_size, pp_size], ["tp", "cp", "pp"])

        pp_group = grid.create_pg("pp")
        cp_group = grid.create_pg("cp")
        tp_group = grid.create_pg("tp")
        embd_group_ranks = parallel_state.default_embedding_ranks(
            torch.distributed.get_process_group_ranks(pp_group)
        )
        embd_group = torch.distributed.new_group(
            ranks=embd_group_ranks, timeout=timedelta(minutes=30)
        )

        # Create model with custom process groups
        from megatron.core.process_groups_config import ProcessGroupCollection

        pg_collection = ProcessGroupCollection(
            tp=tp_group, cp=cp_group, pp=pp_group, embd=embd_group
        )

        # Build pattern with '|' pipeline stage separators: 3 layers per PP stage
        hybrid_layer_pattern = "|".join(["M*-"] * pp_size)

        # Configure model with appropriate sizes for parallelism
        model_config = TransformerConfig(
            num_layers=3 * pp_size,  # Scale layers with PP size
            hidden_size=256 * tp_size,
            num_attention_heads=4 * tp_size,  # Scale heads with TP size
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=pp_size,
            pipeline_dtype=torch.bfloat16,
        )

        model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=4,
            hybrid_layer_pattern=hybrid_layer_pattern,
            pg_collection=pg_collection,
        )

        # Basic forward test
        micro_batch_size = 2
        sequence_length = model.max_sequence_length

        model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == divide(model.vocab_size, tp_size)


class TestHybridQKLayernorm:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_model(self, **config_overrides):
        config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_overrides,
        )
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

    def _get_attention_layer(self, model):
        """Return the SelfAttention submodule from the attention layer."""
        for layer in model.decoder.layers:
            if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'q_layernorm'):
                return layer.self_attention
        return None

    def test_no_qk_norm_by_default(self):
        """Without qk_layernorm, attention has no q/k layernorm."""
        model = self._build_model()
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert attn.q_layernorm is None
        assert attn.k_layernorm is None

    def test_qk_layernorm_from_config(self):
        """config.qk_layernorm=True creates q/k layernorm even with static spec."""
        model = self._build_model(qk_layernorm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        # TENorm is a factory (__new__ returns a TE LayerNorm/RMSNorm), so we
        # verify the norm was created rather than checking for a specific type.
        assert attn.q_layernorm is not None
        assert attn.k_layernorm is not None

    def test_qk_l2_norm_from_config(self):
        """config.qk_l2_norm=True creates L2Norm q/k layernorm."""
        from megatron.core.transformer.torch_norm import L2Norm

        model = self._build_model(qk_l2_norm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert isinstance(attn.q_layernorm, L2Norm)
        assert isinstance(attn.k_layernorm, L2Norm)

    def test_spec_provided_norm_not_overwritten(self):
        """When the spec already provides q/k layernorm, config doesn't override it."""
        import copy

        from megatron.core.extensions.transformer_engine import (
            TEDotProductAttention,
            TELayerNormColumnParallelLinear,
            TERowParallelLinear,
        )
        from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.identity_op import IdentityOp
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.transformer.transformer_layer import (
            TransformerLayer,
            TransformerLayerSubmodules,
        )

        # Build a spec that explicitly sets q/k layernorm to IdentityOp
        spec = copy.deepcopy(hybrid_stack_spec)
        spec.submodules.attention_layer.submodules.self_attention.submodules.q_layernorm = (
            IdentityOp
        )
        spec.submodules.attention_layer.submodules.self_attention.submodules.k_layernorm = (
            IdentityOp
        )

        config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_layernorm=True,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert isinstance(attn.q_layernorm, IdentityOp)
        assert isinstance(attn.k_layernorm, IdentityOp)

    def test_forward_with_qk_layernorm(self):
        """HybridModel forward pass works with qk_layernorm enabled."""
        model = self._build_model(qk_layernorm=True)
        model.cuda()

        sequence_length = 4
        micro_batch_size = 2
        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == 100


class TestHybridWithDynamicInference:
    """Tests HybridModel with dynamic inference."""

    @torch.inference_mode()
    def setup_method(self, method):
        fp8_available, reason_for_no_fp8 = check_fp8_support()
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        model_config = TransformerConfig(
            num_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid",
            fp8_recipe="tensorwise",
        )

        self.model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=DynamicInferenceContext.TOKEN_ROUNDER,
            hybrid_layer_pattern="M*",  # 1 Mamba, 1 attention
        )
        self.model = Float16Module(self.model.config, self.model)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    @torch.inference_mode()
    def test_dynamic_inference_padding_with_fp8(self):
        """
        Tests that logits for padded tokens are zeroed out for fp8 inference.
        """
        self.model.cuda()
        self.model.eval()
        config = self.model.config

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(self.model.module)

        inference_context = DynamicInferenceContext(
            model_config=self.model.config,
            inference_config=InferenceConfig(
                max_sequence_length=self.model.module.max_sequence_length,
                buffer_size_gb=1.0,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
                mamba_inference_state_config=mamba_inference_state_config,
            ),
        )

        # Add a request with 10 tokens. Since 10 is not a multiple of 64 (TOKEN_ROUNDER),
        # this will create padding up to the padded length of 64.
        active_token_count = 10
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=torch.arange(0, active_token_count, dtype=torch.long, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=1),
        )
        inference_context.add_request(request)

        # Prepares the context, including calculating the padded token count.
        inference_context.initialize_attention_state()

        assert inference_context.active_token_count == active_token_count
        assert inference_context.padded_active_token_count == DynamicInferenceContext.TOKEN_ROUNDER

        # Prepare inputs for the forward pass.
        padded_token_count = inference_context.padded_active_token_count
        input_ids, position_ids = inference_context.current_input_and_position_ids()

        # Run the forward pass with inference parameters.
        logits = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            inference_context=inference_context,
            runtime_gather_output=True,
        )

        # Verify the output shape.
        assert logits.shape[0] == 1
        assert logits.shape[1] == padded_token_count
        assert logits.shape[2] == self.model.module.vocab_size

        # Extract the logits corresponding to the padding tokens (from index 10 to 63).
        padding_start_idx = inference_context.active_token_count
        padding_end_idx = inference_context.padded_active_token_count
        padding_logits = logits[0, padding_start_idx:padding_end_idx, :]

        # Assert that all padding logits are zero.
        assert torch.all(padding_logits == 0.0), "Logits for padding tokens are not all zero."


def _make_yarn_config(**kwargs):
    """Build a TransformerConfig with yarn positional embedding attributes."""
    cfg = TransformerConfig(
        num_layers=3,  # 1 Mamba layer, 1 attention layer, 1 MLP layer
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **kwargs,
    )
    # Yarn-specific attributes are set dynamically on the config (not TransformerConfig fields).
    cfg.yarn_rotary_scaling_factor = 2.0
    cfg.yarn_original_max_position_embeddings = 4
    cfg.yarn_beta_fast = 32.0
    cfg.yarn_beta_slow = 1.0
    cfg.yarn_mscale = 1.0
    cfg.yarn_mscale_all_dim = 0.0
    cfg.yarn_correction_range_round_to_int = True
    return cfg


class TestHybridModelWithYarn:
    """Tests for HybridModel with YaRN positional embeddings."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        model_config = _make_yarn_config()
        self.model = HybridModel(
            config=model_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",  # 1 Mamba, 1 attention, 1 MLP
            position_embedding_type='yarn',
            rotary_base=10000,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, HybridModel)
        assert self.model.max_sequence_length == 4
        assert self.model.position_embedding_type == 'yarn'
        # YaRN creates a YarnRotaryEmbedding rather than a plain RotaryEmbedding.
        assert isinstance(self.model.rotary_pos_emb, YarnRotaryEmbedding)

    def test_forward(self):
        sequence_length = self.model.max_sequence_length
        micro_batch_size = 2

        self.model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.model.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.model.vocab_size

    def test_inference(self):
        micro_batch_size = 2
        inference_context: BaseInferenceContext = StaticInferenceContext(
            max_batch_size=micro_batch_size, max_sequence_length=self.model.max_sequence_length
        )
        prompt_length = self.model.max_sequence_length - 1

        self.model.cuda()

        # load-context/first-output-token, step/generate
        for offset in (0, prompt_length):
            sequence_length = prompt_length if offset == 0 else 1
            inference_context.sequence_len_offset = offset

            data = list(range(sequence_length))
            input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
            position_ids = (
                torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
            )
            attention_mask = torch.ones(
                (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
            ).cuda()

            logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inference_context=inference_context,
            )

            assert logits.shape[0] == micro_batch_size
            assert logits.shape[1] == sequence_length
            assert logits.shape[2] == self.model.vocab_size
