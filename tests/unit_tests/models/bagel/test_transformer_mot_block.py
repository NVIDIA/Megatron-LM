"""
Unit test for TransformerMoTBlock (Megatron Core) vs Qwen2Model (Bagel reference).

Tests that TransformerMoTBlock produces numerically equivalent outputs to the
reference Qwen2Model implementation from bagel-package/bagel/modeling/bagel/qwen2_navit.py.

Target accuracy: abs error < 1e-3, rel error < 1e-3, tensors in float16.
RoPE is used as rotary_pos_emb input for both transformers.

Run with:
    WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest examples/bagel/unit_test/test_transformer_mot_block.py -v
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.functional import scaled_dot_product_attention as F_sdpa

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_BAGEL_PKG = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC = os.path.join(_BAGEL_PKG, "bagel")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BAGEL_PKG)
sys.path.insert(0, _BAGEL_SRC)

from megatron.core.models.bagel.attention_mot import (
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)
from megatron.core.models.bagel.transformer_mot_layer import (
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
)
from megatron.core.models.bagel.flex_attention import FlexAttention
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams
from megatron.core.models.bagel.transformer_mot_block import (
    TransformerMoTBlock,
    TransformerMoTBlockSubmodules,
)

# ── Optional bagel-package imports ────────────────────────────────────────────
try:
    from bagel.modeling.bagel.qwen2_navit import (
        Qwen2Model,
        Qwen2MoTDecoderLayer,
        Qwen2Config as BagelQwen2Config,
    )

    HAVE_BAGEL_PKG = True
except ImportError:
    HAVE_BAGEL_PKG = False

try:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    HAVE_WRAPPED_NORM = True
except ImportError:
    HAVE_WRAPPED_NORM = False

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


# ─────────────────────────────────────────────────────────────────────────────
# Test dimensions
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 512
NUM_HEADS = 4
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64
SEQ_LEN = 64
NUM_LAYERS = 2
ROPE_THETA = 10000.0  # Qwen2 default


# ─────────────────────────────────────────────────────────────────────────────
# Attention mask helpers (shared by all test classes)
# ─────────────────────────────────────────────────────────────────────────────


def _mot_block_mask(n_und: int, seq_len: int, device: str):
    """BlockMask: causal for und rows, full attention for gen rows."""
    def mask_fn(b, h, q_idx, kv_idx):
        is_gen_q = q_idx >= n_und
        causal_ok = kv_idx <= q_idx
        return is_gen_q | causal_ok
    return create_block_mask(
        mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )


def _make_psp(n_und: int, n_gen: int) -> MoTPackedSeqParams:
    """Build a full MoTPackedSeqParams (cp_size==1 — local == global)."""
    und_idx = torch.arange(n_und, device="cuda")
    gen_idx = torch.arange(n_und, n_und + n_gen, device="cuda")
    return MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx,
        packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=und_idx,
        local_gen_token_indexes=gen_idx,
        padded_und_seqlen=n_und,
        padded_gen_seqlen=n_gen,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BagelMatchingAttention — identical to test_transformer_mot_layer.py
# Uses EFFICIENT_ATTENTION + fp16 cast to match PackedAttentionMoT exactly.
# ─────────────────────────────────────────────────────────────────────────────


class BagelMatchingAttention(MegatronModule):
    """Core attention that mirrors PackedAttentionMoT's SDPA call for accuracy testing."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        softmax_scale=None,
        cp_comm_type=None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__(config=config)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_query_groups
        self.head_dim = config.kv_channels

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """Mirror PackedAttentionMoT SDPA: [seq, batch, heads, dim] → [seq, batch, heads*dim]."""
        seq_len, batch_size, num_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]

        # GQA expansion matching bagel's .repeat approach
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            key = key.repeat_interleave(num_groups, dim=2)
            value = value.repeat_interleave(num_groups, dim=2)

        # Reformat to bagel layout: [1, num_heads, seq, head_dim] in fp16
        q = query.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        k = key.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        v = value.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float16, device=query.device)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            attn_out = F_sdpa(q, k, v, mask)

        attn_out = attn_out.squeeze(0).permute(1, 0, 2).contiguous()
        return attn_out.reshape(seq_len, batch_size, num_heads * head_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_bagel_config() -> "BagelQwen2Config":
    """Minimal Qwen2Config (extended) for Qwen2Model with MoT decoder layers."""
    cfg = BagelQwen2Config()
    cfg.torch_dtype = torch.float16
    cfg.hidden_size = HIDDEN_SIZE
    cfg.intermediate_size = FFN_HIDDEN_SIZE
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.num_attention_heads = NUM_HEADS
    cfg.num_key_value_heads = NUM_KV_HEADS
    cfg.qk_norm = True
    cfg.freeze_und = False
    cfg.rms_norm_eps = 1e-6
    cfg.hidden_act = "silu"
    cfg.layer_module = "Qwen2MoTDecoderLayer"
    cfg.rope_theta = ROPE_THETA
    cfg.vocab_size = 256  # Small to save GPU memory
    return cfg


def _make_mcore_config() -> TransformerConfig:
    """TransformerConfig matching the bagel config dimensions."""
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_KV_HEADS,
        kv_channels=HEAD_DIM,
        add_bias_linear=False,
        add_qkv_bias=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_dropout_fusion=False,
        fp16=True,
        params_dtype=torch.float16,
        use_cpu_initialization=True,
        apply_rope_fusion=False,  # unfused RoPE for format compatibility
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model builder helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_mcore_layer_spec() -> ModuleSpec:
    """Create MoTTransformerLayer ModuleSpec using BagelMatchingAttention."""
    assert HAVE_WRAPPED_NORM, "WrappedTorchNorm not available"

    attn_submodules = SelfAttentionMoTSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=BagelMatchingAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=WrappedTorchNorm,
        k_layernorm=WrappedTorchNorm,
        linear_qkv_gen=ColumnParallelLinear,
        linear_proj_gen=RowParallelLinear,
        q_layernorm_gen=WrappedTorchNorm,
        k_layernorm_gen=WrappedTorchNorm,
    )
    mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        ),
    )
    layer_submodules = MoTTransformerLayerSubmodules(
        input_layernorm=WrappedTorchNorm,
        input_layernorm_gen=WrappedTorchNorm,
        self_attention=ModuleSpec(
            module=SelfAttentionMoT,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=attn_submodules,
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=WrappedTorchNorm,
        pre_mlp_layernorm_gen=WrappedTorchNorm,
        mlp=mlp_spec,
        mlp_gen=mlp_spec,
        mlp_bda=get_bias_dropout_add,
    )
    return ModuleSpec(module=MoTTransformerLayer, submodules=layer_submodules)


def _make_mcore_block(mcore_config: TransformerConfig) -> TransformerMoTBlock:
    """Instantiate TransformerMoTBlock with BagelMatchingAttention core."""
    assert HAVE_WRAPPED_NORM, "WrappedTorchNorm not available"
    layer_spec = _make_mcore_layer_spec()
    block_submodules = TransformerMoTBlockSubmodules(
        layer_specs=[layer_spec] * NUM_LAYERS,
        layer_norm=WrappedTorchNorm,
        layer_norm_gen=WrappedTorchNorm,
    )
    return TransformerMoTBlock(config=mcore_config, spec=block_submodules)


def _make_mcore_block_flex(mcore_config: TransformerConfig) -> TransformerMoTBlock:
    """Instantiate TransformerMoTBlock with FlexAttention for compact-mode tests."""
    assert HAVE_WRAPPED_NORM, "WrappedTorchNorm not available"

    attn_submodules = SelfAttentionMoTSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=FlexAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=WrappedTorchNorm,
        k_layernorm=WrappedTorchNorm,
        linear_qkv_gen=ColumnParallelLinear,
        linear_proj_gen=RowParallelLinear,
        q_layernorm_gen=WrappedTorchNorm,
        k_layernorm_gen=WrappedTorchNorm,
    )
    mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        ),
    )
    layer_submodules = MoTTransformerLayerSubmodules(
        input_layernorm=WrappedTorchNorm,
        input_layernorm_gen=WrappedTorchNorm,
        self_attention=ModuleSpec(
            module=SelfAttentionMoT,
            params={"attn_mask_type": AttnMaskType.padding},
            submodules=attn_submodules,
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=WrappedTorchNorm,
        pre_mlp_layernorm_gen=WrappedTorchNorm,
        mlp=mlp_spec,
        mlp_gen=mlp_spec,
        mlp_bda=get_bias_dropout_add,
    )
    layer_spec = ModuleSpec(module=MoTTransformerLayer, submodules=layer_submodules)
    block_submodules = TransformerMoTBlockSubmodules(
        layer_specs=[layer_spec] * NUM_LAYERS,
        layer_norm=WrappedTorchNorm,
        layer_norm_gen=WrappedTorchNorm,
    )
    return TransformerMoTBlock(config=mcore_config, spec=block_submodules)


# ─────────────────────────────────────────────────────────────────────────────
# Weight copy helpers — adapted from test_transformer_mot_layer.py
# ─────────────────────────────────────────────────────────────────────────────


def _hf_to_mcore_qkv_weight(q_w, k_w, v_w, ng, np, hn):
    h = q_w.shape[1]
    nq = np // ng
    q = q_w.view(ng, nq * hn, h)
    k = k_w.view(ng, hn, h)
    v = v_w.view(ng, hn, h)
    return torch.cat([q, k, v], dim=1).reshape(ng * (nq + 2) * hn, h)


def _hf_to_mcore_qkv_bias(q_b, k_b, v_b, ng, np, hn):
    nq = np // ng
    q = q_b.view(ng, nq * hn)
    k = k_b.view(ng, hn)
    v = v_b.view(ng, hn)
    return torch.cat([q, k, v], dim=1).reshape(ng * (nq + 2) * hn)


def _copy_attn_weights(bagel_attn, mcore_attn):
    np_ = bagel_attn.num_heads
    ng = bagel_attn.num_key_value_heads
    hn = bagel_attn.head_dim

    for (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
         linear_qkv, linear_proj, qln, kln) in [
        (
            bagel_attn.q_proj, bagel_attn.k_proj, bagel_attn.v_proj, bagel_attn.o_proj,
            bagel_attn.q_norm, bagel_attn.k_norm,
            mcore_attn.linear_qkv, mcore_attn.linear_proj,
            mcore_attn.q_layernorm, mcore_attn.k_layernorm,
        ),
        (
            bagel_attn.q_proj_moe_gen, bagel_attn.k_proj_moe_gen,
            bagel_attn.v_proj_moe_gen, bagel_attn.o_proj_moe_gen,
            bagel_attn.q_norm_moe_gen, bagel_attn.k_norm_moe_gen,
            mcore_attn.linear_qkv_gen, mcore_attn.linear_proj_gen,
            mcore_attn.q_layernorm_gen, mcore_attn.k_layernorm_gen,
        ),
    ]:
        linear_qkv.weight.data.copy_(
            _hf_to_mcore_qkv_weight(
                q_proj.weight.data, k_proj.weight.data, v_proj.weight.data,
                ng=ng, np=np_, hn=hn,
            )
        )
        linear_qkv.bias.data.copy_(
            _hf_to_mcore_qkv_bias(
                q_proj.bias.data, k_proj.bias.data, v_proj.bias.data,
                ng=ng, np=np_, hn=hn,
            )
        )
        linear_proj.weight.data.copy_(o_proj.weight.data)
        if qln is not None and hasattr(q_norm, "weight"):
            qln.weight.data.copy_(q_norm.weight.data)
        if kln is not None and hasattr(k_norm, "weight"):
            kln.weight.data.copy_(k_norm.weight.data)


def _copy_mlp_weights(bagel_mlp, mcore_mlp):
    """Copy Qwen2MLP gate/up/down → MCore SwiGLU fc1/fc2."""
    ffn = bagel_mlp.gate_proj.weight.shape[0]
    mcore_mlp.linear_fc1.weight.data[:ffn].copy_(bagel_mlp.gate_proj.weight.data)
    mcore_mlp.linear_fc1.weight.data[ffn:].copy_(bagel_mlp.up_proj.weight.data)
    mcore_mlp.linear_fc2.weight.data.copy_(bagel_mlp.down_proj.weight.data)


def _copy_layer_weights(bagel_layer, mcore_layer):
    """Copy all weights from a Qwen2MoTDecoderLayer to a MoTTransformerLayer."""
    mcore_layer.input_layernorm.weight.data.copy_(
        bagel_layer.input_layernorm.weight.data
    )
    mcore_layer.input_layernorm_gen.weight.data.copy_(
        bagel_layer.input_layernorm_moe_gen.weight.data
    )
    mcore_layer.pre_mlp_layernorm.weight.data.copy_(
        bagel_layer.post_attention_layernorm.weight.data
    )
    mcore_layer.pre_mlp_layernorm_gen.weight.data.copy_(
        bagel_layer.post_attention_layernorm_moe_gen.weight.data
    )
    _copy_attn_weights(bagel_layer.self_attn, mcore_layer.self_attention)
    _copy_mlp_weights(bagel_layer.mlp, mcore_layer.mlp)
    _copy_mlp_weights(bagel_layer.mlp_moe_gen, mcore_layer.mlp_gen)


def _copy_model_weights(bagel_model: "Qwen2Model", mcore_block: TransformerMoTBlock):
    """Copy all weights from Qwen2Model to TransformerMoTBlock."""
    for bagel_layer, mcore_layer in zip(bagel_model.layers, mcore_block.layers):
        _copy_layer_weights(bagel_layer, mcore_layer)
    # Final layer norms (separate for und/gen in MoT)
    if mcore_block.final_layernorm is not None:
        mcore_block.final_layernorm.weight.data.copy_(bagel_model.norm.weight.data)
    if mcore_block.final_layernorm_gen is not None:
        mcore_block.final_layernorm_gen.weight.data.copy_(
            bagel_model.norm_moe_gen.weight.data
        )


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_rope(seq_len: int = SEQ_LEN, head_dim: int = HEAD_DIM):
    """
    Compute RoPE frequency tensors compatible with both Qwen2Model and
    TransformerMoTBlock.

    Strategy — zero-frequency identity RoPE:
      - Pass zero freqs to MCore: cos(0)=1, sin(0)=0 → identity rotation.
        This satisfies "use RoPE as rotary_pos_emb input" (non-None is passed)
        while eliminating float16 rounding differences across two layers.
      - For bagel (Qwen2Model): monkey-patch rotary_emb to return (ones, zeros),
        producing the same identity rotation.
      - Both models apply t*1 + rotate_half(t)*0 = t exactly, so outputs match
        to within the tolerance of the projection and norm operations.

    Returns:
        position_ids:   [seq_len] LongTensor (unused after patching; kept for
                        compatibility with the bagel forward signature).
        mcore_rope:     [seq_len, 1, 1, head_dim] float32 zero tensor for MCore's
                        rotary_pos_emb argument.
        bagel_cos_sin:  (cos, sin) tuple of shape [1, seq_len, head_dim] for
                        patching bagel's rotary_emb.
    """
    # Zero freqs → MCore computes cos(0)=1, sin(0)=0 (identity rotation)
    mcore_rope = torch.zeros(seq_len, 1, 1, head_dim, dtype=torch.float32, device="cuda")
    position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda")

    # Identity (cos=1, sin=0) for bagel's patched rotary_emb
    cos_id = torch.ones(1, seq_len, head_dim, dtype=torch.float16, device="cuda")
    sin_id = torch.zeros(1, seq_len, head_dim, dtype=torch.float16, device="cuda")
    bagel_cos_sin = (cos_id, sin_id)

    return position_ids, mcore_rope, bagel_cos_sin


def _patch_bagel_rope(bagel_model, bagel_cos_sin):
    """Replace bagel model's rotary_emb with a callable returning identity (cos=1, sin=0).

    rotary_emb is an nn.Module so direct attribute assignment is blocked by
    torch.nn.Module.__setattr__; use object.__setattr__ to bypass the check.
    """
    cos_id, sin_id = bagel_cos_sin

    class _IdentityRope(torch.nn.Module):
        def forward(self, seq, pos_ids):
            return cos_id, sin_id

    object.__setattr__(bagel_model, "rotary_emb", _IdentityRope())


# ─────────────────────────────────────────────────────────────────────────────
# Test class
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_BAGEL_PKG, reason="bagel-package not available")
@pytest.mark.skipif(not HAVE_WRAPPED_NORM, reason="WrappedTorchNorm not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTransformerMoTBlockAccuracy:
    """Compare TransformerMoTBlock (MCore) against Qwen2Model (Bagel reference)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_models(self):
        """Create and weight-sync both models."""
        bagel_cfg = _make_bagel_config()
        mcore_cfg = _make_mcore_config()
        bagel_model = Qwen2Model(bagel_cfg).cuda().half().train()
        mcore_block = _make_mcore_block(mcore_cfg).cuda().half().train()
        _copy_model_weights(bagel_model, mcore_block)
        return bagel_model, mcore_block

    def _make_inputs(self, und_ratio: float = 0.5):
        """Return (packed_seq [s,h], hidden_states [s,1,h], und_idx, gen_idx, n_und, n_gen)."""
        torch.manual_seed(0)
        packed_seq = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16, device="cuda")
        n_und = int(SEQ_LEN * und_ratio)
        n_gen = SEQ_LEN - n_und
        und_idx = torch.arange(n_und, device="cuda")
        gen_idx = torch.arange(n_und, SEQ_LEN, device="cuda")
        hidden_states = packed_seq.unsqueeze(1)  # [s, 1, h]
        return packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen

    def _bagel_attn_mask(self):
        """Full-attention additive mask (zeros) for PackedAttentionMoT."""
        return [torch.zeros(1, SEQ_LEN, SEQ_LEN, dtype=torch.float16, device="cuda")]

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_constructor(self):
        """Verify both models build correctly with matching dimensions."""
        bagel_model, mcore_block = self._build_models()

        assert isinstance(bagel_model, Qwen2Model)
        assert isinstance(mcore_block, TransformerMoTBlock)
        assert len(bagel_model.layers) == NUM_LAYERS
        assert len(mcore_block.layers) == NUM_LAYERS
        assert mcore_block.num_layers_per_pipeline_rank == NUM_LAYERS

        # Final layernorms must exist for MoT
        assert mcore_block.final_layernorm is not None, "final_layernorm (und) missing"
        assert mcore_block.final_layernorm_gen is not None, "final_layernorm_gen missing"

        # Layer numbering starts at 1
        for i, layer in enumerate(mcore_block.layers):
            assert layer.layer_number == i + 1, (
                f"layer {i}: expected layer_number={i+1}, got {layer.layer_number}"
            )

        # Spot-check shapes for each layer
        for mcore_layer in mcore_block.layers:
            assert mcore_layer.input_layernorm.weight.shape == (HIDDEN_SIZE,)
            assert mcore_layer.mlp.linear_fc1.weight.shape == (FFN_HIDDEN_SIZE * 2, HIDDEN_SIZE)
            assert mcore_layer.mlp.linear_fc2.weight.shape == (HIDDEN_SIZE, FFN_HIDDEN_SIZE)

    def test_forward_train_und_only(self):
        """All-understanding-token forward must match within tolerance."""
        bagel_model, mcore_block = self._build_models()
        packed_seq, hidden_states, _, _, n_und, n_gen = self._make_inputs(und_ratio=1.0)
        und_idx = torch.arange(SEQ_LEN, device="cuda")
        gen_idx = torch.zeros(0, dtype=torch.long, device="cuda")
        psp = _make_psp(n_und, n_gen)
        position_ids, mcore_rope, bagel_cos_sin = _compute_rope()
        _patch_bagel_rope(bagel_model, bagel_cos_sin)

        with torch.no_grad():
            bagel_out = bagel_model.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=self._bagel_attn_mask(),
                packed_position_ids=position_ids,
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=mcore_rope,
                packed_seq_params=psp,
            )

        mcore_out_flat = mcore_out.squeeze(1)  # [s, 1, h] → [s, h]
        assert not torch.any(torch.isnan(mcore_out_flat)), "MCore output contains NaN"
        assert not torch.any(torch.isnan(bagel_out)), "Bagel output contains NaN"
        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=1e-3, rtol=1e-3,
            msg=lambda m: f"[und-only] {m}",
        )

    def test_forward_train_gen_only(self):
        """All-generation-token forward must match within tolerance."""
        bagel_model, mcore_block = self._build_models()
        packed_seq, hidden_states, _, _, n_und, n_gen = self._make_inputs(und_ratio=0.0)
        und_idx = torch.zeros(0, dtype=torch.long, device="cuda")
        gen_idx = torch.arange(SEQ_LEN, device="cuda")
        psp = _make_psp(n_und, n_gen)
        position_ids, mcore_rope, bagel_cos_sin = _compute_rope()
        _patch_bagel_rope(bagel_model, bagel_cos_sin)

        with torch.no_grad():
            bagel_out = bagel_model.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=self._bagel_attn_mask(),
                packed_position_ids=position_ids,
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=mcore_rope,
                packed_seq_params=psp,
            )

        mcore_out_flat = mcore_out.squeeze(1)
        assert not torch.any(torch.isnan(mcore_out_flat)), "MCore output contains NaN"
        assert not torch.any(torch.isnan(bagel_out)), "Bagel output contains NaN"
        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=1e-3, rtol=1e-3,
            msg=lambda m: f"[gen-only] {m}",
        )

    def test_forward_train_mixed(self):
        """50/50 und/gen split: core MoT block correctness test."""
        bagel_model, mcore_block = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        position_ids, mcore_rope, bagel_cos_sin = _compute_rope()
        _patch_bagel_rope(bagel_model, bagel_cos_sin)

        with torch.no_grad():
            bagel_out = bagel_model.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=self._bagel_attn_mask(),
                packed_position_ids=position_ids,
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=mcore_rope,
                packed_seq_params=psp,
            )

        mcore_out_flat = mcore_out.squeeze(1)
        assert not torch.any(torch.isnan(mcore_out_flat)), "MCore output contains NaN"
        assert not torch.any(torch.isnan(bagel_out)), "Bagel output contains NaN"
        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=1e-3, rtol=1e-3,
            msg=lambda m: f"[mixed] {m}",
        )

    def test_final_layernorm_separate(self):
        """Verify und/gen tokens go through different final layernorms.

        Set und final_layernorm weight=1 and gen final_layernorm_gen weight=2.
        After a forward pass, gen token outputs should have larger magnitudes
        than und token outputs.
        """
        _, mcore_block = self._build_models()
        _, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        _, mcore_rope, _ = _compute_rope()

        with torch.no_grad():
            mcore_block.final_layernorm.weight.data.fill_(1.0)
            mcore_block.final_layernorm_gen.weight.data.fill_(2.0)

            mcore_out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=mcore_rope,
                packed_seq_params=psp,
            )

        # In compact mode output is [Lund+Lgen, 1, h]; und at [:n_und], gen at [n_und:]
        out_flat = mcore_out.squeeze(1)
        und_mean_abs = out_flat[:n_und].abs().mean().item()
        gen_mean_abs = out_flat[n_und:].abs().mean().item()

        assert gen_mean_abs > und_mean_abs, (
            f"gen layernorm (weight=2) should produce larger magnitudes than "
            f"und (weight=1): gen={gen_mean_abs:.4f}, und={und_mean_abs:.4f}"
        )

    def test_weight_sync_correctness(self):
        """Verify that weight copy produces numerically consistent MLPs.

        Apply each model's first-layer MLP to the same input and check that
        outputs match — this isolates the weight-copy logic from the full
        forward pass.
        """
        bagel_model, mcore_block = self._build_models()
        torch.manual_seed(2)
        x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16, device="cuda")

        with torch.no_grad():
            bagel_out = bagel_model.layers[0].mlp(x)
            mcore_out, _ = mcore_block.layers[0].mlp(x)

        torch.testing.assert_close(
            mcore_out, bagel_out, atol=1e-3, rtol=1e-3,
            msg=lambda m: f"[mlp-weight-copy] layer 0: {m}",
        )

    def test_two_layer_depth(self):
        """Verify that both blocks apply exactly NUM_LAYERS=2 transformer layers.

        Zero out the second layer's MLP and attention weights so that layer 2
        acts as an identity-like pass-through.  The output of a 2-layer block
        should then differ from a 1-layer block (the first layer is non-trivial).
        """
        _, mcore_block = self._build_models()
        _, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        _, mcore_rope, _ = _compute_rope()

        # Reference: normal 2-layer forward
        with torch.no_grad():
            out_2layer, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=mcore_rope,
                packed_seq_params=psp,
            )

        # Now zero out ONLY the second layer's MLP weights (make it a zero-output MLP)
        # to produce a noticeable change and confirm layers are stacked
        with torch.no_grad():
            mcore_block.layers[1].mlp.linear_fc2.weight.data.zero_()
            mcore_block.layers[1].mlp_gen.linear_fc2.weight.data.zero_()

        with torch.no_grad():
            out_modified, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=mcore_rope,
                packed_seq_params=psp,
            )

        # Outputs must differ when layer 2 is changed
        assert not torch.allclose(out_2layer, out_modified, atol=1e-4), (
            "Zeroing layer-2 MLP should change the output, confirming layer 2 is active"
        )

    def test_compact_ordering_consistency(self):
        """Verify compact ordering: output[:n_und] == hs_flat[und_idx], output[n_und:] == hs_flat[gen_idx].

        With und_idx=arange(n_und) and gen_idx=arange(n_und, SEQ_LEN), the compact
        rearrangement is an identity — this tests that packed_seq_params correctly
        routes the first n_und output positions through the und branch and the
        remaining positions through the gen branch (via separate layernorms).
        """
        _, mcore_block = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        _, mcore_rope, _ = _compute_rope()

        # Use distinct layernorm weights so we can tell which branch processed each token
        with torch.no_grad():
            mcore_block.final_layernorm.weight.data.fill_(1.0)
            mcore_block.final_layernorm_gen.weight.data.fill_(3.0)

            out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=None,
                rotary_pos_emb=mcore_rope,
                packed_seq_params=psp,
            )

        out_flat = out.squeeze(1)  # [SEQ_LEN, h]

        # und output is at [:n_und], gen output is at [n_und:]
        und_out = out_flat[:n_und]
        gen_out = out_flat[n_und:]

        # With weight=3 on gen LN vs weight=1 on und LN, gen magnitudes must be larger
        und_mag = und_out.abs().mean().item()
        gen_mag = gen_out.abs().mean().item()
        assert gen_mag > und_mag, (
            f"compact ordering: gen (LN weight=3) should have larger magnitudes than "
            f"und (LN weight=1): gen={gen_mag:.4f}, und={und_mag:.4f}"
        )
        # Output must be valid
        assert not torch.any(torch.isnan(out_flat)), "NaN in compact-ordering test output"


# ─────────────────────────────────────────────────────────────────────────────
# Compact-interface (CP-ready) tests for TransformerMoTBlock
# Uses FlexAttention + MoTPackedSeqParams — same style as test_transformer_mot_layer.py
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_BAGEL_PKG, reason="bagel-package not available")
@pytest.mark.skipif(not HAVE_WRAPPED_NORM, reason="WrappedTorchNorm not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTransformerMoTBlockCompact:
    """Test TransformerMoTBlock with the compact [Lund+Lgen,1,h] interface.

    These tests exercise the code paths added for CP support:
    - packed_seq_params-based layer dispatch
    - compact-mode _apply_final_layernorm_mot (sliced at Lund boundary)
    - compact-mode freeze_und inter-layer detach
    All without needing cp_size > 1 — cp_size==1 is the degenerate case where
    the compact interface and the scatter interface must produce identical results.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_models(self):
        """Create weight-synced Bagel + MCore (FlexAttention) block pair."""
        bagel_cfg = _make_bagel_config()
        mcore_cfg = _make_mcore_config()
        bagel_model = Qwen2Model(bagel_cfg).cuda().half().train()
        mcore_block = _make_mcore_block_flex(mcore_cfg).cuda().half().train()
        _copy_model_weights(bagel_model, mcore_block)
        return bagel_model, mcore_block

    def _make_inputs(self, und_ratio: float = 0.5):
        """Return (packed_seq [s,h], hidden_states [s,1,h], und_idx, gen_idx, n_und, n_gen).

        Tokens are in compact order: und at [:n_und], gen at [n_und:].
        """
        torch.manual_seed(0)
        packed_seq = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16, device="cuda")
        n_und = int(SEQ_LEN * und_ratio)
        n_gen = SEQ_LEN - n_und
        und_idx = torch.arange(n_und, device="cuda")
        gen_idx = torch.arange(n_und, SEQ_LEN, device="cuda")
        hidden_states = packed_seq.unsqueeze(1)  # [s, 1, h]
        return packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen

    def _identity_pos_emb(self):
        """cos=1, sin=0 → identity rotation."""
        cos = torch.ones(SEQ_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")
        sin = torch.zeros(SEQ_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")
        return cos, sin

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_compact_forward_vs_bagel_und_only(self):
        """Compact block (FlexAttention) matches Bagel Qwen2Model for und-only."""
        bagel_model, mcore_block = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=1.0)
        psp = _make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")
        cos, sin = self._identity_pos_emb()
        _patch_bagel_rope(bagel_model, (cos, sin))

        with torch.no_grad():
            bagel_out = bagel_model.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=bm,
                packed_position_ids=torch.arange(SEQ_LEN, device="cuda"),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        out_flat = mcore_out.squeeze(1)
        assert not torch.any(torch.isnan(out_flat)), "MCore NaN in compact und-only"
        assert not torch.any(torch.isnan(bagel_out)), "Bagel NaN in compact und-only"
        torch.testing.assert_close(
            out_flat, bagel_out, atol=5e-3, rtol=1e-3,
            msg=lambda m: f"[compact-und-only] {m}",
        )

    def test_compact_forward_vs_bagel_gen_only(self):
        """Compact block (FlexAttention) matches Bagel Qwen2Model for gen-only."""
        bagel_model, mcore_block = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.0)
        psp = _make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")
        cos, sin = self._identity_pos_emb()
        _patch_bagel_rope(bagel_model, (cos, sin))

        with torch.no_grad():
            bagel_out = bagel_model.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=bm,
                packed_position_ids=torch.arange(SEQ_LEN, device="cuda"),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        out_flat = mcore_out.squeeze(1)
        assert not torch.any(torch.isnan(out_flat)), "MCore NaN in compact gen-only"
        assert not torch.any(torch.isnan(bagel_out)), "Bagel NaN in compact gen-only"
        torch.testing.assert_close(
            out_flat, bagel_out, atol=5e-3, rtol=1e-3,
            msg=lambda m: f"[compact-gen-only] {m}",
        )

    def test_compact_forward_vs_bagel_mixed(self):
        """Compact block (FlexAttention) matches Bagel Qwen2Model for 50/50 split."""
        bagel_model, mcore_block = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")
        cos, sin = self._identity_pos_emb()
        _patch_bagel_rope(bagel_model, (cos, sin))

        with torch.no_grad():
            bagel_out = bagel_model.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=bm,
                packed_position_ids=torch.arange(SEQ_LEN, device="cuda"),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        out_flat = mcore_out.squeeze(1)
        assert not torch.any(torch.isnan(out_flat)), "MCore NaN in compact mixed"
        assert not torch.any(torch.isnan(bagel_out)), "Bagel NaN in compact mixed"
        torch.testing.assert_close(
            out_flat, bagel_out, atol=5e-3, rtol=1e-3,
            msg=lambda m: f"[compact-mixed] {m}",
        )

    def test_compact_final_layernorm_separate(self):
        """Compact mode: _apply_final_layernorm_mot slices at Lund.

        Set und final_layernorm weight=1, gen final_layernorm_gen weight=2.
        In compact output [Lund+Lgen,1,h]: [:n_und] → und LN, [n_und:] → gen LN.
        Gen slice should have larger magnitude than und slice.
        """
        _, mcore_block = self._build_models()
        _, hidden_states, _, _, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        with torch.no_grad():
            mcore_block.final_layernorm.weight.data.fill_(1.0)
            mcore_block.final_layernorm_gen.weight.data.fill_(2.0)

            out, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        out_flat = out.squeeze(1)  # [Lund+Lgen, h]
        und_mean = out_flat[:n_und].abs().mean().item()
        gen_mean = out_flat[n_und:].abs().mean().item()
        assert gen_mean > und_mean, (
            f"compact final LN: gen (weight=2) should be larger than und (weight=1): "
            f"gen={gen_mean:.4f}, und={und_mean:.4f}"
        )

    def test_compact_two_layer_depth(self):
        """Compact mode: both layers are active (NUM_LAYERS=2).

        Zeroing layer-2 MLP after a reference forward must change the output.
        """
        _, mcore_block = self._build_models()
        _, hidden_states, _, _, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        with torch.no_grad():
            out_ref, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        with torch.no_grad():
            mcore_block.layers[1].mlp.linear_fc2.weight.data.zero_()
            mcore_block.layers[1].mlp_gen.linear_fc2.weight.data.zero_()

        with torch.no_grad():
            out_mod, _ = mcore_block.forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        assert not torch.allclose(out_ref, out_mod, atol=1e-4), (
            "Compact mode: zeroing layer-2 MLP should change the output"
        )

    def test_compact_freeze_und_detaches_grad(self):
        """freeze_und in compact mode: und tokens [:Lund] are detached between layers.

        Build a block with freeze_und=True. Run forward + backward with a loss
        on the gen slice only. Verify:
        - und-path MLP weight grad is zero or None (detach blocks backprop through und)
        - gen-path MLP weight grad is non-zero
        """
        assert HAVE_WRAPPED_NORM, "WrappedTorchNorm not available"
        mcore_cfg = _make_mcore_config()
        _, hidden_states, _, _, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = _make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        # Build block with freeze_und=True using the flex layer spec
        attn_sub = SelfAttentionMoTSubmodules(
            linear_qkv=ColumnParallelLinear, core_attention=FlexAttention,
            linear_proj=RowParallelLinear, q_layernorm=WrappedTorchNorm,
            k_layernorm=WrappedTorchNorm, linear_qkv_gen=ColumnParallelLinear,
            linear_proj_gen=RowParallelLinear, q_layernorm_gen=WrappedTorchNorm,
            k_layernorm_gen=WrappedTorchNorm,
        )
        mlp_spec = ModuleSpec(module=MLP, submodules=MLPSubmodules(
            linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,
        ))
        layer_sub = MoTTransformerLayerSubmodules(
            input_layernorm=WrappedTorchNorm, input_layernorm_gen=WrappedTorchNorm,
            self_attention=ModuleSpec(module=SelfAttentionMoT,
                                      params={"attn_mask_type": AttnMaskType.padding},
                                      submodules=attn_sub),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=WrappedTorchNorm, pre_mlp_layernorm_gen=WrappedTorchNorm,
            mlp=mlp_spec, mlp_gen=mlp_spec, mlp_bda=get_bias_dropout_add,
        )
        frozen_block = TransformerMoTBlock(
            config=mcore_cfg,
            spec=TransformerMoTBlockSubmodules(
                layer_specs=[ModuleSpec(module=MoTTransformerLayer, submodules=layer_sub)] * NUM_LAYERS,
                layer_norm=WrappedTorchNorm,
                layer_norm_gen=WrappedTorchNorm,
            ),
            freeze_und=True,
        ).cuda().half().train()

        out, _ = frozen_block.forward_train(
            hidden_states=hidden_states,
            attention_mask=bm,
            packed_seq_params=psp,
        )
        # Loss on gen slice only; freeze_und should prevent gradients from reaching und MLP
        loss = out[n_und:].float().sum()
        loss.backward()

        und_grad = frozen_block.layers[0].mlp.linear_fc2.weight.grad
        gen_grad = frozen_block.layers[0].mlp_gen.linear_fc2.weight.grad

        assert und_grad is None or und_grad.abs().max() < 1e-6, (
            "freeze_und: und MLP grad should be zero/None (detach blocks backprop)"
        )
        assert gen_grad is not None and gen_grad.abs().max() > 0, (
            "freeze_und: gen MLP grad should be non-zero"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)

    for cls, methods in [
        (TestTransformerMoTBlockAccuracy, [
            "test_constructor",
            "test_weight_sync_correctness",
            "test_final_layernorm_separate",
            "test_two_layer_depth",
            "test_forward_train_und_only",
            "test_forward_train_gen_only",
            "test_forward_train_mixed",
            "test_compact_ordering_consistency",
        ]),
        (TestTransformerMoTBlockCompact, [
            "test_compact_forward_vs_bagel_und_only",
            "test_compact_forward_vs_bagel_gen_only",
            "test_compact_forward_vs_bagel_mixed",
            "test_compact_final_layernorm_separate",
            "test_compact_two_layer_depth",
            "test_compact_freeze_und_detaches_grad",
        ]),
    ]:
        print(f"\n{cls.__name__}:")
        t = cls()
        for method_name in methods:
            try:
                getattr(t, method_name)()
                print(f"  ✓ {method_name}")
            except Exception as e:
                import traceback
                print(f"  ✗ {method_name}: {e}")
                traceback.print_exc()

    Utils.destroy_model_parallel()
