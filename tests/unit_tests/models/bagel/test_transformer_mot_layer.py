"""
Unit test for MoTTransformerLayer (Megatron Core) vs Qwen2MoTDecoderLayer (Bagel reference).

Tests that MoTTransformerLayer produces numerically equivalent outputs to
Qwen2MoTDecoderLayer from bagel-package/bagel/modeling/bagel/qwen2_navit.py.

Target accuracy: abs error < 1e-3, rel error < 1e-3, tensors in float16.

Run with:
    WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest examples/bagel/unit_test/test_transformer_mot_layer.py -v
"""

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — mirrors test_attention_mot.py
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_BAGEL_PKG = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC = os.path.join(_BAGEL_PKG, "bagel")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BAGEL_PKG)
sys.path.insert(0, _BAGEL_SRC)



import megatron.core.parallel_state as mpu                              # noqa: E402

from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams  # noqa: E402
from megatron.core.models.bagel.flex_attention import FlexAttention               # noqa: E402
from megatron.core.models.bagel.attention_mot import (                            # noqa: E402
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)
from megatron.core.models.bagel.transformer_mot_layer import (                    # noqa: E402
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
)

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# ── Optional bagel-package imports ────────────────────────────────────────────
try:
    from bagel.modeling.bagel.qwen2_navit import PackedAttentionMoT, Qwen2MoTDecoderLayer
    from bagel.modeling.qwen2.configuration_qwen2 import Qwen2Config as BagelQwen2Config

    HAVE_BAGEL_PKG = True
except ImportError:
    HAVE_BAGEL_PKG = False

try:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    HAVE_WRAPPED_NORM = True
except ImportError:
    HAVE_WRAPPED_NORM = False


# ─────────────────────────────────────────────────────────────────────────────
# Minimal ProcessGroupCollection-compatible container
# ─────────────────────────────────────────────────────────────────────────────


class _PGC:
    def __init__(self, tp, cp=None):
        self.tp = tp
        if cp is not None:
            self.cp = cp


# ─────────────────────────────────────────────────────────────────────────────
# Attention mask helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mot_block_mask(n_und: int, seq_len: int, device: str):
    """BlockMask: causal mask for und-token rows, full attention for gen-token rows."""
    def mask_fn(b, h, q_idx, kv_idx):
        is_gen_q = q_idx >= n_und
        causal_ok = kv_idx <= q_idx
        return is_gen_q | causal_ok
    return create_block_mask(
        mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test dimensions
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 512
NUM_HEADS = 4
NUM_KV_HEADS = 4
SEQ_LEN = 64


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_bagel_config() -> "BagelQwen2Config":
    cfg = BagelQwen2Config()
    cfg.torch_dtype = torch.float16
    cfg.hidden_size = HIDDEN_SIZE
    cfg.intermediate_size = FFN_HIDDEN_SIZE
    cfg.num_attention_heads = NUM_HEADS
    cfg.num_key_value_heads = NUM_KV_HEADS
    cfg.qk_norm = True
    cfg.freeze_und = False
    cfg.rms_norm_eps = 1e-6
    cfg.hidden_act = "silu"
    return cfg


def _make_mcore_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_KV_HEADS,
        kv_channels=HIDDEN_SIZE // NUM_HEADS,
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
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────


def _make_mcore_layer(mcore_config: TransformerConfig) -> MoTTransformerLayer:
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
    mlp_submodules = MLPSubmodules(
        linear_fc1=ColumnParallelLinear,
        linear_fc2=RowParallelLinear,
    )
    mlp_spec = ModuleSpec(module=MLP, submodules=mlp_submodules)

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
    return MoTTransformerLayer(
        config=mcore_config,
        submodules=layer_submodules,
        layer_number=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Weight-copy helpers
# ─────────────────────────────────────────────────────────────────────────────


def _hf_to_mcore_qkv_weight(q_w, k_w, v_w, ng, np, hn):
    """Convert separate HF q/k/v weights to Megatron interleaved-by-group layout."""
    h = q_w.shape[1]
    nq = np // ng
    q = q_w.view(ng, nq * hn, h)
    k = k_w.view(ng, hn, h)
    v = v_w.view(ng, hn, h)
    return torch.cat([q, k, v], dim=1).reshape(ng * (nq + 2) * hn, h)


def _hf_to_mcore_qkv_bias(q_b, k_b, v_b, ng, np, hn):
    """Convert separate HF q/k/v biases to Megatron interleaved layout."""
    nq = np // ng
    q = q_b.view(ng, nq * hn)
    k = k_b.view(ng, hn)
    v = v_b.view(ng, hn)
    return torch.cat([q, k, v], dim=1).reshape(ng * (nq + 2) * hn)


def _copy_attn_weights(bagel_attn: "PackedAttentionMoT", mcore_attn: SelfAttentionMoT):
    """Copy PackedAttentionMoT weights into SelfAttentionMoT (und and gen paths)."""
    np = bagel_attn.num_heads
    ng = bagel_attn.num_key_value_heads
    hn = bagel_attn.head_dim

    for (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
         linear_qkv, linear_proj, qln, kln) in [
        # understanding-token projection weights
        (
            bagel_attn.q_proj, bagel_attn.k_proj, bagel_attn.v_proj, bagel_attn.o_proj,
            bagel_attn.q_norm, bagel_attn.k_norm,
            mcore_attn.linear_qkv, mcore_attn.linear_proj,
            mcore_attn.q_layernorm, mcore_attn.k_layernorm,
        ),
        # generation-token projection weights
        (
            bagel_attn.q_proj_moe_gen, bagel_attn.k_proj_moe_gen,
            bagel_attn.v_proj_moe_gen, bagel_attn.o_proj_moe_gen,
            bagel_attn.q_norm_moe_gen, bagel_attn.k_norm_moe_gen,
            mcore_attn.linear_qkv_gen, mcore_attn.linear_proj_gen,
            mcore_attn.q_layernorm_gen, mcore_attn.k_layernorm_gen,
        ),
    ]:
        qkv_w = _hf_to_mcore_qkv_weight(
            q_proj.weight.data, k_proj.weight.data, v_proj.weight.data,
            ng=ng, np=np, hn=hn,
        )
        linear_qkv.weight.data.copy_(qkv_w)
        qkv_b = _hf_to_mcore_qkv_bias(
            q_proj.bias.data, k_proj.bias.data, v_proj.bias.data,
            ng=ng, np=np, hn=hn,
        )
        linear_qkv.bias.data.copy_(qkv_b)
        linear_proj.weight.data.copy_(o_proj.weight.data)
        if qln is not None and hasattr(q_norm, "weight"):
            qln.weight.data.copy_(q_norm.weight.data)
        if kln is not None and hasattr(k_norm, "weight"):
            kln.weight.data.copy_(k_norm.weight.data)


def _copy_mlp_weights(bagel_mlp, mcore_mlp: MLP):
    """Copy Qwen2MLP (gate/up/down) → MCore MLP (linear_fc1/fc2).

    MCore SwiGLU layout: linear_fc1 output = [gate_half | up_half].
    silu is applied to the first half (gate_half), then multiplied by up_half.
    Qwen2MLP: silu(gate_proj(x)) * up_proj(x).
    So: gate_proj → first half of fc1, up_proj → second half.
    """
    ffn = bagel_mlp.gate_proj.weight.shape[0]
    mcore_mlp.linear_fc1.weight.data[:ffn].copy_(bagel_mlp.gate_proj.weight.data)
    mcore_mlp.linear_fc1.weight.data[ffn:].copy_(bagel_mlp.up_proj.weight.data)
    mcore_mlp.linear_fc2.weight.data.copy_(bagel_mlp.down_proj.weight.data)


def _copy_layer_weights(
    bagel_layer: "Qwen2MoTDecoderLayer",
    mcore_layer: MoTTransformerLayer,
):
    """Copy all weights from Qwen2MoTDecoderLayer to MoTTransformerLayer."""
    # Input layer norms (und / gen)
    mcore_layer.input_layernorm.weight.data.copy_(
        bagel_layer.input_layernorm.weight.data
    )
    mcore_layer.input_layernorm_gen.weight.data.copy_(
        bagel_layer.input_layernorm_moe_gen.weight.data
    )
    # Pre-MLP layer norms (und / gen) — called post_attention_layernorm in Bagel
    mcore_layer.pre_mlp_layernorm.weight.data.copy_(
        bagel_layer.post_attention_layernorm.weight.data
    )
    mcore_layer.pre_mlp_layernorm_gen.weight.data.copy_(
        bagel_layer.post_attention_layernorm_moe_gen.weight.data
    )
    # Attention
    _copy_attn_weights(bagel_layer.self_attn, mcore_layer.self_attention)
    # MLP (und / gen)
    _copy_mlp_weights(bagel_layer.mlp, mcore_layer.mlp)
    _copy_mlp_weights(bagel_layer.mlp_moe_gen, mcore_layer.mlp_gen)


# ─────────────────────────────────────────────────────────────────────────────
# Test class
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_BAGEL_PKG, reason="bagel-package not available")
@pytest.mark.skipif(not HAVE_WRAPPED_NORM, reason="WrappedTorchNorm not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMoTTransformerLayerAccuracy:
    """Compare MoTTransformerLayer (MCore) against Qwen2MoTDecoderLayer (Bagel reference)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_models(self):
        bagel_cfg = _make_bagel_config()
        mcore_cfg = _make_mcore_config()
        bagel_layer = Qwen2MoTDecoderLayer(bagel_cfg, layer_idx=0).cuda().half().train()
        mcore_layer = _make_mcore_layer(mcore_cfg).cuda().half().train()
        _copy_layer_weights(bagel_layer, mcore_layer)
        return bagel_layer, mcore_layer

    def _make_inputs(self, und_ratio: float = 0.5):
        """Return (packed_seq [s,h], hidden_states [s,1,h], und_idx, gen_idx, n_und, n_gen).

        Tokens are already in compact order: und tokens at [:n_und], gen at [n_und:].
        hidden_states == packed_seq.unsqueeze(1) and is ready for the compact interface.
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
        """cos=1, sin=0 → identity rotation (no-op)."""
        head_dim = HIDDEN_SIZE // NUM_HEADS
        cos = torch.ones(SEQ_LEN, head_dim, dtype=torch.float16, device="cuda")
        sin = torch.zeros(SEQ_LEN, head_dim, dtype=torch.float16, device="cuda")
        return cos, sin
    

    def _make_psp(self, n_und: int, n_gen: int) -> MoTPackedSeqParams:
        """Build a full MoTPackedSeqParams with index arrays (cp_size==1)."""
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

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_constructor(self):
        bagel_layer, mcore_layer = self._build_models()
        assert isinstance(bagel_layer, Qwen2MoTDecoderLayer)
        assert isinstance(mcore_layer, MoTTransformerLayer)
        # Spot-check weight shapes
        assert mcore_layer.input_layernorm.weight.shape == (HIDDEN_SIZE,)
        assert mcore_layer.mlp.linear_fc1.weight.shape == (FFN_HIDDEN_SIZE * 2, HIDDEN_SIZE)
        assert mcore_layer.mlp.linear_fc2.weight.shape == (HIDDEN_SIZE, FFN_HIDDEN_SIZE)

    def test_forward_train_und_only(self):
        """All-understanding-token forward: outputs must match within tolerance."""
        bagel_layer, mcore_layer = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=1.0)
        cos, sin = self._identity_pos_emb()
        psp = self._make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        with torch.no_grad():
            bagel_out = bagel_layer.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=bm,
                packed_position_embeddings=(cos, sin),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_layer._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
                rotary_pos_emb=None,
            )

        mcore_out_flat = mcore_out.squeeze(1)
        assert torch.all(~torch.isnan(mcore_out_flat)), "MCore output has NaN"
        assert torch.all(~torch.isnan(bagel_out)), "Bagel output has NaN"
        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=5e-3, rtol=1e-3,
            msg=lambda m: f"[und-only] {m}",
        )

    def test_forward_train_gen_only(self):
        """All-generation-token forward: outputs must match within tolerance."""
        bagel_layer, mcore_layer = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.0)
        cos, sin = self._identity_pos_emb()
        psp = self._make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        with torch.no_grad():
            bagel_out = bagel_layer.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=bm,
                packed_position_embeddings=(cos, sin),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_layer._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
                rotary_pos_emb=None,
            )

        mcore_out_flat = mcore_out.squeeze(1)
        assert torch.all(~torch.isnan(mcore_out_flat)), "MCore output has NaN"
        assert torch.all(~torch.isnan(bagel_out)), "Bagel output has NaN"
        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=5e-3, rtol=1e-3,
            msg=lambda m: f"[gen-only] {m}",
        )

    def test_forward_train_mixed(self):
        """50/50 und/gen split: core MoT correctness test."""
        bagel_layer, mcore_layer = self._build_models()
        packed_seq, hidden_states, und_idx, gen_idx, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        cos, sin = self._identity_pos_emb()
        psp = self._make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        with torch.no_grad():
            bagel_out = bagel_layer.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[SEQ_LEN],
                attention_mask=bm,
                packed_position_embeddings=(cos, sin),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )
            mcore_out, _ = mcore_layer._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
                rotary_pos_emb=None,
            )

        mcore_out_flat = mcore_out.squeeze(1)
        assert torch.all(~torch.isnan(mcore_out_flat)), "MCore output has NaN"
        assert torch.all(~torch.isnan(bagel_out)), "Bagel output has NaN"
        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=5e-3, rtol=1e-3,
            msg=lambda m: f"[mixed] {m}",
        )

    def test_und_gen_use_separate_mlps(self):
        """Verify und/gen tokens go through different MLPs.

        Zero out mlp_gen weights so gen tokens produce zero MLP contribution.
        The layer's full forward is:
            hidden_post_attn = attn(layernorm(x)) + x          # attention + residual₁
            out = mlp(layernorm(hidden_post_attn)) + hidden_post_attn  # MLP + residual₂

        With mlp_gen zeroed, gen output = 0 + hidden_post_attn[gen] = hidden_post_attn[gen].
        We capture hidden_post_attn via a monkey-patch on _forward_mlp_mot.

        Assertions:
        - gen output == hidden_post_attn[gen]  (mlp_gen contribution is zero)
        - und output != hidden_post_attn[und]  (mlp contribution is non-zero)
        """
        import types

        _, mcore_layer = self._build_models()
        _, hidden_states, _, _, n_und, n_gen = self._make_inputs(und_ratio=0.5)
        psp = self._make_psp(n_und, n_gen)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        # Capture the hidden state entering _forward_mlp_mot (post-attention residual)
        captured = {}
        original_mlp_mot = mcore_layer._forward_mlp_mot.__func__

        def _patched_mlp_mot(self, hs, Lund):
            captured["hidden_post_attn"] = hs.detach().clone()
            return original_mlp_mot(self, hs, Lund)

        mcore_layer._forward_mlp_mot = types.MethodType(_patched_mlp_mot, mcore_layer)

        with torch.no_grad():
            mcore_layer.mlp_gen.linear_fc1.weight.data.zero_()
            mcore_layer.mlp_gen.linear_fc2.weight.data.zero_()

            out, _ = mcore_layer._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
                rotary_pos_emb=None,
            )

        out_flat = out.squeeze(1)           # [SEQ_LEN, h]; und at [:n_und], gen at [n_und:]
        post_attn_flat = captured["hidden_post_attn"].squeeze(1)

        # gen tokens: zero mlp_gen → MLP adds 0 → output == post-attention residual
        torch.testing.assert_close(
            out_flat[n_und:], post_attn_flat[n_und:], atol=5e-3, rtol=5e-3,
            msg=lambda m: f"[sep-mlp] gen output should equal hidden_post_attn when mlp_gen is zeroed: {m}",
        )
        # und tokens: non-zero mlp → output != post-attention residual
        assert not torch.allclose(out_flat[:n_und], post_attn_flat[:n_und], atol=1e-3), (
            "und tokens should differ from hidden_post_attn — they use a non-zero mlp"
        )

    def test_und_gen_use_separate_layernorms(self):
        """Verify und/gen tokens go through different input layernorms.

        Set und layernorm weight=1 and gen layernorm weight=2; the gen
        layernorm should produce larger output magnitudes.
        """
        _, mcore_layer = self._build_models()
        _, hidden_states, _, _, n_und, _ = self._make_inputs(und_ratio=0.5)

        with torch.no_grad():
            mcore_layer.input_layernorm.weight.data.fill_(1.0)
            mcore_layer.input_layernorm_gen.weight.data.fill_(2.0)

            ln_out = mcore_layer._apply_input_layernorm_mot(hidden_states, n_und)

        ln_flat = ln_out.squeeze(1)   # [SEQ_LEN, h]; und at [:n_und], gen at [n_und:]
        und_mean_abs = ln_flat[:n_und].abs().mean().item()
        gen_mean_abs = ln_flat[n_und:].abs().mean().item()
        assert gen_mean_abs > und_mean_abs, (
            f"gen layernorm (weight=2) should produce larger magnitudes than und (weight=1): "
            f"gen={gen_mean_abs:.4f}, und={und_mean_abs:.4f}"
        )

    @pytest.mark.parametrize("und_ratio,zeroed_mlp", [
        (1.0, "mlp"),      # und-only: zero the und MLP; output must equal post-attn residual
        (0.0, "mlp_gen"),  # gen-only: zero the gen MLP; output must equal post-attn residual
    ])
    def test_single_branch_mlp_zero_contribution(self, und_ratio, zeroed_mlp):
        """Edge case: und-only and gen-only MLP routing.

        When all tokens belong to one branch and the active MLP's weights are zeroed
        (add_bias_linear=False, so output is exactly zero), the layer output must equal
        the post-attention residual — verifying that the inactive branch is never called
        and the empty-slice path produces no artefacts.
        """
        import types

        _, mcore_layer = self._build_models()
        _, hidden_states, _, _, n_und, n_gen = self._make_inputs(und_ratio=und_ratio)
        psp = self._make_psp(n_und, n_gen)

        captured = {}
        original_mlp_mot = mcore_layer._forward_mlp_mot.__func__

        def _patched(self, hs, Lund):
            captured["post_attn"] = hs.detach().clone()
            return original_mlp_mot(self, hs, Lund)

        mcore_layer._forward_mlp_mot = types.MethodType(_patched, mcore_layer)
        bm = _mot_block_mask(n_und, SEQ_LEN, "cuda")

        with torch.no_grad():
            getattr(mcore_layer, zeroed_mlp).linear_fc1.weight.data.zero_()
            getattr(mcore_layer, zeroed_mlp).linear_fc2.weight.data.zero_()
            out, _ = mcore_layer._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
                rotary_pos_emb=None,
            )

        label = "und-only" if und_ratio == 1.0 else "gen-only"
        torch.testing.assert_close(
            out.squeeze(1), captured["post_attn"].squeeze(1), atol=1e-3, rtol=1e-3,
            msg=lambda m: f"[{label}] output should equal post-attn when active MLP is zeroed: {m}",
        )

    @pytest.mark.parametrize("und_ratio", [1.0, 0.0])
    def test_single_branch_layernorm_routing(self, und_ratio):
        """Edge case: und-only and gen-only layernorm routing.

        Set the active branch's LN weight=2 and the inactive branch's LN weight=1.
        Output magnitude should reflect the active LN's scale (~2x the weight=1 baseline),
        confirming that the empty-branch path is skipped cleanly.
        """
        _, mcore_layer = self._build_models()
        _, hidden_states, _, _, n_und, _ = self._make_inputs(und_ratio=und_ratio)

        # weight=2 on the active LN, weight=1 on the inactive LN
        with torch.no_grad():
            if und_ratio == 1.0:
                mcore_layer.input_layernorm.weight.data.fill_(2.0)
                mcore_layer.input_layernorm_gen.weight.data.fill_(1.0)
            else:
                mcore_layer.input_layernorm.weight.data.fill_(1.0)
                mcore_layer.input_layernorm_gen.weight.data.fill_(2.0)
            ln_out_w2 = mcore_layer._apply_input_layernorm_mot(hidden_states, n_und)

        # weight=1 baseline on the same (active) LN
        with torch.no_grad():
            if und_ratio == 1.0:
                mcore_layer.input_layernorm.weight.data.fill_(1.0)
            else:
                mcore_layer.input_layernorm_gen.weight.data.fill_(1.0)
            ln_out_w1 = mcore_layer._apply_input_layernorm_mot(hidden_states, n_und)

        ratio = ln_out_w2.abs().mean() / ln_out_w1.abs().mean()
        label = "und-only" if und_ratio == 1.0 else "gen-only"
        assert abs(ratio.item() - 2.0) < 0.1, (
            f"[{label}] active LN weight=2 output should be ~2x weight=1 output, "
            f"got ratio={ratio:.3f}"
        )

    def test_mlp_weight_copy_correctness(self):
        """Verify that the gate/up/down → fc1/fc2 weight copy is correct.

        Apply each MLP to the same input and check outputs match.
        """
        bagel_layer, mcore_layer = self._build_models()
        torch.manual_seed(1)
        x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16, device="cuda")

        with torch.no_grad():
            bagel_out = bagel_layer.mlp(x)
            mcore_out, _ = mcore_layer.mlp(x)

        torch.testing.assert_close(
            mcore_out, bagel_out, atol=5e-3, rtol=5e-3,
            msg=lambda m: f"[mlp-weight-copy] {m}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)

    t = TestMoTTransformerLayerAccuracy()
    for method_name in [
        "test_constructor",
        "test_mlp_weight_copy_correctness",
        "test_und_gen_use_separate_layernorms",
        "test_und_gen_use_separate_mlps",
        "test_forward_train_und_only",
        "test_forward_train_gen_only",
        "test_forward_train_mixed",
    ]:
        try:
            getattr(t, method_name)()
            print(f"  ✓ {method_name}")
        except Exception as e:
            import traceback
            print(f"  ✗ {method_name}: {e}")
            traceback.print_exc()

    Utils.destroy_model_parallel()
