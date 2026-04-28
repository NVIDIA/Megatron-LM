# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import os
from datetime import timedelta
from itertools import accumulate
from unittest.mock import patch

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
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import divide, is_fa_min_version, is_torch_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    _HAVE_HADAMARD = True
except ImportError:
    _HAVE_HADAMARD = False
    _hadamard_transform = None


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Identity-with-scale stand-in for `fast_hadamard_transform.hadamard_transform`.

    Mirrors the helper in `tests/unit_tests/transformer/experimental_attention_variant/
    test_attention_variant_dsa.py` so that DSA forward tests run in containers that
    don't ship the upstream library.
    """
    return x * scale


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

    # Subclasses override these to retarget the same tests at MLA's
    # `mla_layer.kv_layernorm` or DSA's `dsa_layer.kv_layernorm`. The base class
    # exercises the SelfAttention path with `attention_layer.k_layernorm`.
    _attention_layer_attr = 'attention_layer'
    _k_norm_attr = 'k_layernorm'

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config = TransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_overrides,
        )
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

    def _get_attention_layer(self, model):
        """Return the self-attention submodule that owns a `q_layernorm`."""
        for layer in model.decoder.layers:
            if hasattr(layer, 'self_attention') and hasattr(layer.self_attention, 'q_layernorm'):
                return layer.self_attention
        return None

    def _get_k_norm(self, attn):
        return getattr(attn, self._k_norm_attr)

    def test_trivial_qk_norm_by_default(self):
        """Without qk_layernorm, attention has trivial q/k layernorm."""
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model()
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert attn.q_layernorm is None or isinstance(attn.q_layernorm, IdentityOp)
        k_norm = self._get_k_norm(attn)
        assert k_norm is None or isinstance(k_norm, IdentityOp)

    def test_qk_layernorm_from_config(self):
        """config.qk_layernorm=True creates q/k layernorm even with static spec."""
        model = self._build_model(qk_layernorm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        # TENorm is a factory (__new__ returns a TE LayerNorm/RMSNorm), so we
        # verify the norm was created rather than checking for a specific type.
        assert attn.q_layernorm is not None
        assert self._get_k_norm(attn) is not None

    def test_qk_l2_norm_from_config(self):
        """config.qk_l2_norm=True creates L2Norm q/k layernorm."""
        from megatron.core.transformer.torch_norm import L2Norm

        model = self._build_model(qk_l2_norm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert isinstance(attn.q_layernorm, L2Norm)
        assert isinstance(self._get_k_norm(attn), L2Norm)

    def test_spec_provided_norm_not_overwritten(self):
        """When the spec already provides q/k layernorm, config doesn't override it."""
        import copy

        from megatron.core.transformer.identity_op import IdentityOp

        # Build a spec that explicitly sets q/k layernorm to IdentityOp on the
        # attention layer that this subclass exercises.
        spec = copy.deepcopy(hybrid_stack_spec)
        attn_submodules = getattr(
            spec.submodules, self._attention_layer_attr
        ).submodules.self_attention.submodules
        attn_submodules.q_layernorm = IdentityOp
        setattr(attn_submodules, self._k_norm_attr, IdentityOp)

        model = self._build_model(spec=spec, qk_layernorm=True)
        attn = self._get_attention_layer(model)
        assert attn is not None
        assert isinstance(attn.q_layernorm, IdentityOp)
        assert isinstance(self._get_k_norm(attn), IdentityOp)

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


class TestHybridMLAQKLayernorm(TestHybridQKLayernorm):
    """Tests QK norm configuration of HybridModel with MLA."""

    _attention_layer_attr = 'mla_layer'
    _k_norm_attr = 'kv_layernorm'

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config = MLATransformerConfig(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            **config_overrides,
        )
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M+-",
        )

    def test_qk_l2_norm_from_config(self):
        with pytest.raises(ValueError, match="qk_l2_norm is not supported"):
            super().test_qk_l2_norm_from_config()


class TestHybridDSAQKLayernorm(TestHybridQKLayernorm):
    """Tests QK norm configuration of HybridModel with DSA."""

    _attention_layer_attr = 'dsa_layer'
    _k_norm_attr = 'kv_layernorm'

    @pytest.fixture(autouse=True)
    def _patch_hadamard_if_needed(self):
        if not _HAVE_HADAMARD:
            with patch(
                'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
                _mock_hadamard_transform,
            ):
                yield
        else:
            yield

    def test_spec_provided_norm_not_overwritten(self):
        # DSA cannot fuse the QK norm into the up-projection, so a trivial
        # `IdentityOp` spec is auto-promoted to `TENorm` when `qk_layernorm=True`.
        # Finer-grained spec-respect behavior is covered by TestDSAQKNormResolution.
        pytest.skip("DSA auto-promotes IdentityOp to TENorm; covered by TestDSAQKNormResolution.")

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config_kwargs = dict(
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            # MLASelfAttention forwards `x` and `qr` to the core attention only when
            # `experimental_attention_variant == "dsa"`; without this the DSA core
            # attention's forward fails on missing positional arguments.
            experimental_attention_variant="dsa",
            # DSA-specific settings; defaults are None and DSAIndexer requires them.
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            # The indexer-loss path runs in training mode and multiplies by this coefficient;
            # leaving it at the default `None` raises `TypeError: ... 'Tensor' and 'NoneType'`.
            dsa_indexer_loss_coeff=1.0,
            # DSA's `rotate_activation` (Hadamard rotation) only supports bf16 input.
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        config_kwargs.update(config_overrides)
        config = MLATransformerConfig(**config_kwargs)
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="MD-",
        )

    def test_qk_l2_norm_from_config(self):
        with pytest.raises(ValueError, match="qk_l2_norm is not supported"):
            super().test_qk_l2_norm_from_config()


class _MLAQKNormTestBase:
    """Common machinery for MLA/DSA QK-norm spec tests.

    Subclasses override `experimental_attention_variant` and
    `hybrid_layer_pattern` to target the MLA vs. DSA code path.
    """

    experimental_attention_variant = None
    hybrid_layer_pattern = "M+-"
    mla_layer_attr = "mla_layer"

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_spec(self, **submodule_overrides):
        """Return a copy of `hybrid_stack_spec` with MLA/DSA submodule overrides."""
        import copy

        spec = copy.deepcopy(hybrid_stack_spec)
        mla_submodules = getattr(
            spec.submodules, self.mla_layer_attr
        ).submodules.self_attention.submodules
        for key, value in submodule_overrides.items():
            setattr(mla_submodules, key, value)
        return spec

    def _build_model(self, spec=None, **config_overrides):
        if spec is None:
            spec = hybrid_stack_spec
        config_kwargs = dict(
            num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )
        if self.experimental_attention_variant is not None:
            config_kwargs["experimental_attention_variant"] = self.experimental_attention_variant
            if self.experimental_attention_variant == "dsa":
                # DSAIndexer requires these; their config defaults are None.
                config_kwargs.setdefault("dsa_indexer_n_heads", 8)
                config_kwargs.setdefault("dsa_indexer_head_dim", 64)
                config_kwargs.setdefault("dsa_indexer_topk", 32)

        config_kwargs.update(config_overrides)
        config = MLATransformerConfig(**config_kwargs)
        return HybridModel(
            config=config,
            hybrid_stack_spec=spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern=self.hybrid_layer_pattern,
        )

    def _get_mla_attention(self, model):
        """Return the MLA self-attention submodule, or None."""
        from megatron.core.transformer.multi_latent_attention import MLASelfAttention

        for layer in model.decoder.layers:
            if hasattr(layer, 'self_attention') and isinstance(
                layer.self_attention, MLASelfAttention
            ):
                return layer.self_attention
        return None


class TestMLAQKNormSpecValidation(_MLAQKNormTestBase):
    """Tests `_validate_qk_norm_spec` in `MLASelfAttention`.

    These errors guard against silently ignoring a configured norm or
    double-applying one through a fused norm+linear.
    """

    experimental_attention_variant = None
    hybrid_layer_pattern = "M+-"
    mla_layer_attr = "mla_layer"

    def test_q_norm_without_q_lora_rank_raises(self):
        """When `q_lora_rank is None`, a non-trivial `q_layernorm` would
        never be reached and must error out.
        """
        from megatron.core.extensions.transformer_engine import TENorm

        spec = self._make_spec(q_layernorm=TENorm)
        with pytest.raises(RuntimeError, match=r"q_lora_rank is None"):
            self._build_model(spec=spec, q_lora_rank=None)

    def test_q_norm_without_q_lora_rank_hint_for_non_fused_linear(self):
        """Error message hints at fused linear when `linear_q_proj` is non-fused."""
        from megatron.core.extensions.transformer_engine import TENorm

        spec = self._make_spec(q_layernorm=TENorm)
        with pytest.raises(RuntimeError, match=r"fused norm\+linear for"):
            self._build_model(spec=spec, q_lora_rank=None)

    def test_fused_linear_q_up_with_q_norm_raises(self):
        """Non-trivial `q_layernorm` combined with a fused `linear_q_up_proj`
        would apply the norm twice.
        """
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(q_layernorm=TENorm, linear_q_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(RuntimeError, match=r"fused norm\+linear"):
            self._build_model(spec=spec)

    def test_fused_linear_kv_up_with_kv_norm_raises(self):
        """Non-trivial `kv_layernorm` combined with a fused `linear_kv_up_proj`
        would apply the norm twice.
        """
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(
            kv_layernorm=TENorm, linear_kv_up_proj=TELayerNormColumnParallelLinear
        )
        with pytest.raises(RuntimeError, match=r"fused norm\+linear"):
            self._build_model(spec=spec)


class TestMLAQKNormResolution(_MLAQKNormTestBase):
    """Tests `_resolve_mla_qk_norm_config` branches.

    Covers fusion auto-selection, spec overrides, and the "disabled"-path
    guards that reject fused/explicit norms when `qk_layernorm` is off.
    """

    experimental_attention_variant = None
    hybrid_layer_pattern = "M+-"
    mla_layer_attr = "mla_layer"

    def test_qk_layernorm_fuses_kv_up_by_default(self):
        """With default (trivial) `kv_layernorm`, enabling `qk_layernorm`
        auto-selects the fused `TELayerNormColumnParallelLinear` for KV up.
        """
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model(qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_kv_up_proj, TELayerNormColumnParallelLinear)
        assert isinstance(attn.kv_layernorm, IdentityOp)

    def test_spec_q_norm_disables_q_up_fusion(self):
        """A non-trivial `q_layernorm` from the spec must force a non-fused
        `linear_q_up_proj` so the norm isn't applied on top of a fused one.
        """
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(q_layernorm=TENorm)
        model = self._build_model(spec=spec, qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_q_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_q_up_proj, TELayerNormColumnParallelLinear)
        # The spec's norm is actually used; it's not reset to IdentityOp.
        assert attn.q_layernorm is not None
        from megatron.core.transformer.identity_op import IdentityOp

        assert not isinstance(attn.q_layernorm, IdentityOp)

    def test_spec_kv_norm_disables_kv_up_fusion(self):
        """Mirror of `test_spec_q_norm_disables_q_up_fusion` for KV."""
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
            TENorm,
        )

        spec = self._make_spec(kv_layernorm=TENorm)
        model = self._build_model(spec=spec, qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_kv_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_kv_up_proj, TELayerNormColumnParallelLinear)
        from megatron.core.transformer.identity_op import IdentityOp

        assert not isinstance(attn.kv_layernorm, IdentityOp)

    def test_disabled_qk_layernorm_rejects_fused_linear_q_up(self):
        """When `qk_layernorm` is off, spec must not force fused linear_q_up_proj."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_q_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"supposed to be disabled"):
            self._build_model(spec=spec)

    def test_disabled_qk_layernorm_rejects_fused_linear_kv_up(self):
        """When `qk_layernorm` is off, spec must not force fused linear_kv_up_proj."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_kv_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"supposed to be disabled"):
            self._build_model(spec=spec)

    def test_disabled_qk_layernorm_rejects_spec_kv_norm(self):
        """When `qk_layernorm` is off, spec must not carry an explicit kv_layernorm."""
        from megatron.core.extensions.transformer_engine import TENorm

        spec = self._make_spec(kv_layernorm=TENorm)
        with pytest.raises(ValueError, match=r"supposed to be disabled"):
            self._build_model(spec=spec)


class TestDSAQKNormResolution(_MLAQKNormTestBase):
    """Tests `_resolve_dsa_qk_norm_config`.

    DSA requires non-fused Q/KV up projections and explicit norms;
    the fused optimization valid for MLA must be rejected here.
    """

    experimental_attention_variant = "dsa"
    hybrid_layer_pattern = "MD-"
    mla_layer_attr = "dsa_layer"

    def test_qk_layernorm_uses_unfused_linear_and_te_norm(self):
        """With default spec, DSA + `qk_layernorm=True` uses non-fused
        `TEColumnParallelLinear` and `TENorm` for Q/KV.
        """
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TELayerNormColumnParallelLinear,
        )
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model(qk_layernorm=True)
        attn = self._get_mla_attention(model)
        assert attn is not None
        assert isinstance(attn.linear_q_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_q_up_proj, TELayerNormColumnParallelLinear)
        assert isinstance(attn.linear_kv_up_proj, TEColumnParallelLinear)
        assert not isinstance(attn.linear_kv_up_proj, TELayerNormColumnParallelLinear)
        assert not isinstance(attn.q_layernorm, IdentityOp)
        assert not isinstance(attn.kv_layernorm, IdentityOp)

    def test_qk_layernorm_rejects_fused_linear_q_up(self):
        """DSA does not support the fused norm+linear optimization."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_q_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(
            RuntimeError, match=r"fused norm\+linear, but this is not supported for DSA"
        ):
            self._build_model(spec=spec, qk_layernorm=True)

    def test_qk_layernorm_without_q_lora_rejects_fused_linear_q(self):
        """DSA does not support fused `linear_q_proj` when `q_lora_rank=None`."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_q_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(
            RuntimeError, match=r"fused norm\+linear, but this is not supported for DSA"
        ):
            self._build_model(spec=spec, qk_layernorm=True, q_lora_rank=None)

    def test_disabled_qk_layernorm_rejects_fused_linear_kv_up(self):
        """When `qk_layernorm` is off, spec must not force fused linear_kv_up_proj."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear

        spec = self._make_spec(linear_kv_up_proj=TELayerNormColumnParallelLinear)
        with pytest.raises(ValueError, match=r"supposed to be disabled"):
            self._build_model(spec=spec)


class TestMLADownProjFusion:
    """Tests `HybridStack._maybe_fuse_mla_down_proj`.

    The method rewrites the MLA `ModuleSpec` in place on a deep-copied
    `HybridStackSubmodules` when `config.mla_down_proj_fusion=True`, swapping
    the self-attention module to `FusedMLASelfAttention` and collapsing the
    separate q/kv down projections into a single fused `linear_qkv_down_proj`
    that also absorbs the input layernorm.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _fresh_submodules(self):
        """Return a deep copy of `hybrid_stack_spec.submodules` so tests don't
        share state through `hybrid_stack_spec`.
        """
        import copy

        return copy.deepcopy(hybrid_stack_spec.submodules)

    def _call_fuse(self, submodules, *, mla_down_proj_fusion):
        """Invoke `_maybe_fuse_mla_down_proj` as an unbound method with a
        minimal stub for `self`. The method only reads `self.config`, so we
        can avoid constructing a full `HybridStack`.
        """
        import types

        from megatron.core.models.hybrid.hybrid_block import HybridStack

        stub = types.SimpleNamespace(
            config=types.SimpleNamespace(mla_down_proj_fusion=mla_down_proj_fusion)
        )
        return HybridStack._maybe_fuse_mla_down_proj(stub, submodules)

    def _build_model(self, pattern="M+-", **config_overrides):
        config_kwargs = dict(
            num_layers=3, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )
        config_kwargs.update(config_overrides)
        config = MLATransformerConfig(**config_kwargs)
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern=pattern,
        )

    def _get_layer_with_mla(self, model):
        """Return the layer whose self-attention is an `MLASelfAttention`
        (which includes its `FusedMLASelfAttention` subclass).
        """
        from megatron.core.transformer.multi_latent_attention import MLASelfAttention

        for layer in model.decoder.layers:
            if hasattr(layer, 'self_attention') and isinstance(
                layer.self_attention, MLASelfAttention
            ):
                return layer
        return None

    def test_disabled_returns_spec_unchanged(self):
        """Flag off: method returns the same object, no copying or rewriting."""
        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=False)
        assert result is submodules

    def test_missing_attr_treated_as_disabled(self):
        """When the config lacks the attribute, `getattr(..., False)` disables fusion."""
        import types

        from megatron.core.models.hybrid.hybrid_block import HybridStack

        submodules = self._fresh_submodules()
        stub = types.SimpleNamespace(config=types.SimpleNamespace())
        result = HybridStack._maybe_fuse_mla_down_proj(stub, submodules)
        assert result is submodules

    def test_enabled_rewrites_mla_spec(self):
        """Flag on: MLA spec is swapped to the fused module and fused linear."""
        from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
        from megatron.core.transformer.identity_op import IdentityOp
        from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention

        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        mla_spec = result.mla_layer
        assert mla_spec.submodules.input_layernorm is IdentityOp
        assert mla_spec.submodules.self_attention.module is FusedMLASelfAttention

        attn_submodules = mla_spec.submodules.self_attention.submodules
        assert attn_submodules.linear_qkv_down_proj is TELayerNormColumnParallelLinear
        assert attn_submodules.linear_q_down_proj is None
        assert attn_submodules.linear_kv_down_proj is None

    def test_enabled_sets_sharded_state_dict_keys_map(self):
        """The keys map is written on the MLA layer submodules for checkpoint
        compatibility with pre-fusion checkpoints.
        """
        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        keys_map = result.mla_layer.submodules.sharded_state_dict_keys_map
        assert keys_map == {
            "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
        }

    def test_enabled_deep_copies_input_submodules(self):
        """The caller's submodules object must not be mutated – the method
        deep-copies before rewriting, so callers can safely reuse their spec.
        """
        from megatron.core.transformer.multi_latent_attention import (
            FusedMLASelfAttention,
            MLASelfAttention,
        )

        submodules = self._fresh_submodules()
        original_mla_module = submodules.mla_layer.submodules.self_attention.module
        original_q_down_proj = (
            submodules.mla_layer.submodules.self_attention.submodules.linear_q_down_proj
        )
        assert original_mla_module is MLASelfAttention  # sanity check of baseline

        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        # Original is unchanged.
        assert submodules.mla_layer.submodules.self_attention.module is original_mla_module
        assert (
            submodules.mla_layer.submodules.self_attention.submodules.linear_q_down_proj
            is original_q_down_proj
        )
        # And result is a different object than the input.
        assert result is not submodules
        assert result.mla_layer is not submodules.mla_layer
        # Plus the fused module only shows up on the returned copy.
        assert result.mla_layer.submodules.self_attention.module is FusedMLASelfAttention

    def test_enabled_leaves_dsa_layer_alone(self):
        """DSA layer spec shares the MLA self-attention class, but fusion
        should only rewrite `mla_layer` — not `dsa_layer`.
        """
        from megatron.core.transformer.multi_latent_attention import (
            FusedMLASelfAttention,
            MLASelfAttention,
        )

        submodules = self._fresh_submodules()
        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        assert result.dsa_layer.submodules.self_attention.module is MLASelfAttention
        assert result.dsa_layer.submodules.self_attention.module is not FusedMLASelfAttention
        # DSA's down projections must remain non-`None` (they're still used
        # via the unfused path).
        assert result.dsa_layer.submodules.self_attention.submodules.linear_q_down_proj is not None
        assert result.dsa_layer.submodules.self_attention.submodules.linear_kv_down_proj is not None

    def test_enabled_leaves_non_mla_layers_alone(self):
        """Unrelated layer specs (mamba, attention, mlp) must survive unchanged."""
        submodules = self._fresh_submodules()
        original_mamba = submodules.mamba_layer
        original_attention = submodules.attention_layer
        original_mlp = submodules.mlp_layer

        result = self._call_fuse(submodules, mla_down_proj_fusion=True)

        # Equality via deep-copy means the returned specs compare as equal to
        # the originals (dataclass equality) even though they are fresh
        # objects.
        assert result.mamba_layer == original_mamba
        assert result.attention_layer == original_attention
        assert result.mlp_layer == original_mlp

    def test_model_uses_fused_mla_when_enabled(self):
        """Integration: a full HybridModel built with the flag uses
        `FusedMLASelfAttention`.
        """
        from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention

        model = self._build_model(mla_down_proj_fusion=True)
        layer = self._get_layer_with_mla(model)
        assert layer is not None
        assert isinstance(layer.self_attention, FusedMLASelfAttention)
        # And the fused down projection is present on the attention module.
        assert hasattr(layer.self_attention, "linear_qkv_down_proj")

    def test_model_uses_unfused_mla_when_disabled(self):
        """Integration: with the flag off, MLA layers use the standard
        `MLASelfAttention` (never the fused subclass).
        """
        from megatron.core.transformer.multi_latent_attention import (
            FusedMLASelfAttention,
            MLASelfAttention,
        )

        model = self._build_model(mla_down_proj_fusion=False)
        layer = self._get_layer_with_mla(model)
        assert layer is not None
        assert isinstance(layer.self_attention, MLASelfAttention)
        assert not isinstance(layer.self_attention, FusedMLASelfAttention)

    def test_enabled_replaces_input_layernorm_with_identity(self):
        """Integration: because the fused down-proj absorbs the input
        layernorm, the transformer layer's own `input_layernorm` must be
        `IdentityOp`.
        """
        from megatron.core.transformer.identity_op import IdentityOp

        model = self._build_model(mla_down_proj_fusion=True)
        layer = self._get_layer_with_mla(model)
        assert layer is not None
        assert isinstance(layer.input_layernorm, IdentityOp)

    def test_forward_with_fused_mla(self):
        """Integration: forward pass works with `mla_down_proj_fusion=True`."""
        model = self._build_model(mla_down_proj_fusion=True)
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
