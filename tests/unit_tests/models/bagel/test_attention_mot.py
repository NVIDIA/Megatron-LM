"""
Combined unit test for SelfAttentionMoT.

Part 1 — Accuracy tests (pytest, world_size=1):
  TestSelfAttentionMoTAccuracy: compare SelfAttentionMoT (MCore) against PackedAttentionMoT
  (Bagel reference) using FlexAttention as the core attention backend.
  Tolerance: atol=rtol=1e-2.

Part 2 — CP parity tests (torchrun, world_size=2/4/8):
  run_cp_parity_test, run_single_branch_test: verify that SelfAttentionMoT._forward_train
  produces numerically identical outputs with and without context parallelism.

Run Part 1:
  PYTHONPATH=<root>:<root>/bagel-package:<root>/bagel-package/bagel \
  pytest examples/mimo_bagel/unit_test/test_attention_mot.py -v

Run Part 2:
  torchrun --nproc_per_node=2 examples/mimo_bagel/unit_test/test_attention_mot.py
  torchrun --nproc_per_node=4 examples/mimo_bagel/unit_test/test_attention_mot.py
  torchrun --nproc_per_node=8 examples/mimo_bagel/unit_test/test_attention_mot.py
"""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

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



from torch.nn.attention.flex_attention import create_block_mask  # noqa: E402

from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams  # noqa: E402
from megatron.core.models.bagel.flex_attention import FlexAttention               # noqa: E402
from megatron.core.models.bagel.attention_mot import (                            # noqa: E402
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)

import megatron.core.parallel_state as mpu                              # noqa: E402
from megatron.core.tensor_parallel.layers import (                      # noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType                # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from tests.unit_tests.test_utilities import Utils  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependencies
# ─────────────────────────────────────────────────────────────────────────────

try:
    from bagel.modeling.bagel.qwen2_navit import PackedAttentionMoT
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
# CP parity test configuration
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    2: SimpleNamespace(u=8,  g=4,  u_und=8,  u_gen=4,  nh=4, hd=32),  # H=128
    4: SimpleNamespace(u=8,  g=8,  u_und=8,  u_gen=8,  nh=4, hd=32),  # H=128
    8: SimpleNamespace(u=16, g=16, u_und=16, u_gen=16, nh=8, hd=32),  # H=256
}


# ─────────────────────────────────────────────────────────────────────────────
# Minimal ProcessGroupCollection-compatible container
# ─────────────────────────────────────────────────────────────────────────────


class _PGC:
    def __init__(self, tp, cp=None):
        self.tp = tp
        if cp is not None:
            self.cp = cp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by both test parts
# ─────────────────────────────────────────────────────────────────────────────


def _block_mask(seq_len: int, device: str):
    """Full-attention BlockMask over seq_len tokens."""
    return create_block_mask(
        lambda b, h, q, kv: q >= 0,
        B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len,
        device=device,
    )


def _make_bagel_config(hidden_size: int, num_heads: int, num_kv_heads: int) -> "BagelQwen2Config":
    """Create a minimal Qwen2Config for PackedAttentionMoT."""
    cfg = BagelQwen2Config()
    cfg.torch_dtype = torch.float16
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = num_heads
    cfg.num_key_value_heads = num_kv_heads
    cfg.qk_norm = True
    cfg.freeze_und = False
    cfg.rms_norm_eps = 1e-6
    return cfg


def _make_mcore_config(hidden_size: int, num_heads: int, num_kv_heads: int) -> TransformerConfig:
    """Create a fp16 TransformerConfig for accuracy tests."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        kv_channels=hidden_size // num_heads,
        add_bias_linear=False,
        add_qkv_bias=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        fp16=True,
        params_dtype=torch.float16,
        use_cpu_initialization=True,
    )


def _make_config_bf16(nh: int, hd: int) -> TransformerConfig:
    """Create a bf16 TransformerConfig for CP parity tests."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=nh * hd,
        num_attention_heads=nh,
        num_query_groups=nh,
        kv_channels=hd,
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )


def _make_mcore_attention(mcore_config: TransformerConfig, tp_group) -> SelfAttentionMoT:
    """Instantiate SelfAttentionMoT with FlexAttention + QK layernorm for accuracy tests."""
    assert HAVE_WRAPPED_NORM, "WrappedTorchNorm not available"

    pgc = _PGC(tp=tp_group, cp=None)
    submodules = SelfAttentionMoTSubmodules(
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
    return SelfAttentionMoT(
        config=mcore_config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.padding,
        pg_collection=pgc,
    )


def _make_attention(config: TransformerConfig, tp_group, cp_group, seed: int) -> SelfAttentionMoT:
    """
    Build SelfAttentionMoT with FlexAttention core for CP parity tests.

    No QK layernorm — tests focus on CP communication correctness.
    Same seed before each call ensures identical weights in both models.
    """
    pgc = _PGC(tp=tp_group, cp=cp_group)
    submodules = SelfAttentionMoTSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=FlexAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=None,
        k_layernorm=None,
        linear_qkv_gen=ColumnParallelLinear,
        linear_proj_gen=RowParallelLinear,
        q_layernorm_gen=None,
        k_layernorm_gen=None,
    )
    torch.manual_seed(seed)
    attn = SelfAttentionMoT(
        config=config,
        submodules=submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.padding,
        pg_collection=pgc,
    )
    return attn.cuda().eval()


# ─────────────────────────────────────────────────────────────────────────────
# Weight copying helpers (Bagel → MCore)
# ─────────────────────────────────────────────────────────────────────────────


def _hf_to_mcore_qkv_weight(q_w, k_w, v_w, ng: int, np: int, hn: int):
    """
    Convert separate HF-style q/k/v weights to Megatron interleaved-by-group layout.

    Megatron layout: [group0_q_heads, group0_k_head, group0_v_head,
                      group1_q_heads, group1_k_head, group1_v_head, ...]
    """
    h = q_w.shape[1]
    nq = np // ng  # query heads per group
    q = q_w.view(ng, nq * hn, h)  # [ng, nq*hn, h]
    k = k_w.view(ng, hn, h)       # [ng, hn,    h]
    v = v_w.view(ng, hn, h)       # [ng, hn,    h]
    qkv = torch.cat([q, k, v], dim=1)  # [ng, (nq+2)*hn, h]
    return qkv.reshape(ng * (nq + 2) * hn, h)


def _hf_to_mcore_qkv_bias(q_b, k_b, v_b, ng: int, np: int, hn: int):
    """Convert separate HF-style q/k/v biases to Megatron interleaved layout."""
    nq = np // ng
    q = q_b.view(ng, nq * hn)
    k = k_b.view(ng, hn)
    v = v_b.view(ng, hn)
    qkv = torch.cat([q, k, v], dim=1)  # [ng, (nq+2)*hn]
    return qkv.reshape(ng * (nq + 2) * hn)


def _copy_weights_bagel_to_mcore(
    bagel_attn: "PackedAttentionMoT",
    mcore_attn: SelfAttentionMoT,
) -> None:
    """
    Copy weights from PackedAttentionMoT to SelfAttentionMoT,
    performing the HF→Megatron interleaved-QKV conversion.
    """
    np = bagel_attn.num_heads
    ng = bagel_attn.num_key_value_heads
    hn = bagel_attn.head_dim

    for (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, linear_qkv, linear_proj, qln, kln) in [
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
        # QKV weights → interleaved Megatron layout
        qkv_w = _hf_to_mcore_qkv_weight(
            q_proj.weight.data, k_proj.weight.data, v_proj.weight.data,
            ng=ng, np=np, hn=hn,
        )
        linear_qkv.weight.data.copy_(qkv_w)

        # QKV biases
        qkv_b = _hf_to_mcore_qkv_bias(
            q_proj.bias.data, k_proj.bias.data, v_proj.bias.data,
            ng=ng, np=np, hn=hn,
        )
        linear_qkv.bias.data.copy_(qkv_b)

        # Output projection weight (no bias in PackedAttentionMoT)
        linear_proj.weight.data.copy_(o_proj.weight.data)

        # Q/K layernorm weights
        if qln is not None and hasattr(q_norm, "weight"):
            qln.weight.data.copy_(q_norm.weight.data)
        if kln is not None and hasattr(k_norm, "weight"):
            kln.weight.data.copy_(k_norm.weight.data)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Accuracy tests (pytest, world_size=1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAVE_BAGEL_PKG, reason="bagel-package not available")
@pytest.mark.skipif(not HAVE_WRAPPED_NORM, reason="WrappedTorchNorm not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4)])
class TestSelfAttentionMoTAccuracy:
    """Compare SelfAttentionMoT (MCore, FlexAttention) vs PackedAttentionMoT (Bagel reference)."""

    HIDDEN_SIZE = 256
    SEQ_LEN = 128
    BATCH_SIZE = 1

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, num_heads, num_kv_heads):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = self.HIDDEN_SIZE // num_heads

        yield

        Utils.destroy_model_parallel()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_models(self):
        """Create and weight-sync both attention models."""
        bagel_cfg = _make_bagel_config(self.HIDDEN_SIZE, self.num_heads, self.num_kv_heads)
        mcore_cfg = _make_mcore_config(self.HIDDEN_SIZE, self.num_heads, self.num_kv_heads)

        bagel_attn = PackedAttentionMoT(bagel_cfg, layer_idx=0).cuda().half().train()
        tp_group = mpu.get_tensor_model_parallel_group()
        mcore_attn = _make_mcore_attention(mcore_cfg, tp_group).cuda().half().train()

        _copy_weights_bagel_to_mcore(bagel_attn, mcore_attn)

        return bagel_attn, mcore_attn

    def _make_inputs(self, und_ratio: float = 0.5):
        """Create a packed sequence, hidden_states, and und/gen index splits."""
        seq_len = self.SEQ_LEN
        h = self.HIDDEN_SIZE

        torch.manual_seed(0)
        packed_seq = torch.randn(seq_len, h, dtype=torch.float16, device="cuda")

        n_und = int(seq_len * und_ratio)
        und_idx = torch.arange(n_und, device="cuda")
        gen_idx = torch.arange(n_und, seq_len, device="cuda")

        hidden_states = packed_seq.unsqueeze(1)  # [s, 1, h]

        return packed_seq, hidden_states, und_idx, gen_idx

    def _identity_pos_emb(self, seq_len: int):
        """Return (cos, sin) that produce identity rotary: cos=1, sin=0."""
        cos = torch.ones(seq_len, self.head_dim, dtype=torch.float16, device="cuda")
        sin = torch.zeros(seq_len, self.head_dim, dtype=torch.float16, device="cuda")
        return cos, sin

    def _bagel_attention_mask(self, seq_len: int):
        """Full-attention zero additive mask for PackedAttentionMoT."""
        mask = torch.zeros(1, seq_len, seq_len, dtype=torch.float16, device="cuda")
        return [mask]  # list of per-sample masks

    def _make_psp(self, und_idx, gen_idx):
        """Build a MoTPackedSeqParams for the full (cp=1) layout."""
        n_und = len(und_idx)
        n_gen = len(gen_idx)
        return MoTPackedSeqParams(
            qkv_format="thd",
            packed_und_token_indexes=und_idx,
            packed_gen_token_indexes=gen_idx,
            local_und_token_indexes=und_idx,
            local_gen_token_indexes=gen_idx,
            padded_und_seqlen=n_und,
            padded_gen_seqlen=n_gen,
        )

    # ── tests ────────────────────────────────────────────────────────────────

    def test_constructor(self, num_heads, num_kv_heads):
        """Both models can be constructed with matching configs."""
        bagel_attn, mcore_attn = self._build_models()

        assert isinstance(bagel_attn, PackedAttentionMoT)
        assert isinstance(mcore_attn, SelfAttentionMoT)

        hn = self.head_dim
        np = self.num_heads
        ng = self.num_kv_heads
        nq = np // ng

        expected_qkv_out = ng * (nq + 2) * hn
        assert mcore_attn.linear_qkv.weight.shape == (expected_qkv_out, self.HIDDEN_SIZE)
        assert mcore_attn.linear_proj.weight.shape == (self.HIDDEN_SIZE, np * hn)

    def test_forward_train_und_tokens_only(self, num_heads, num_kv_heads):
        """All-understanding-token forward: both models must produce close outputs."""
        bagel_attn, mcore_attn = self._build_models()
        seq_len = self.SEQ_LEN

        packed_seq, hidden_states, _, _ = self._make_inputs(und_ratio=1.0)
        und_idx = torch.arange(seq_len, device="cuda")
        gen_idx = torch.zeros(0, dtype=torch.long, device="cuda")

        cos, sin = self._identity_pos_emb(seq_len)
        attn_mask = self._bagel_attention_mask(seq_len)
        psp = self._make_psp(und_idx, gen_idx)
        bm = _block_mask(seq_len, "cuda")

        with torch.no_grad():
            bagel_out = bagel_attn.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[seq_len],
                attention_mask=attn_mask,
                packed_position_embeddings=(cos, sin),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )

        with torch.no_grad():
            mcore_out, _ = mcore_attn._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        mcore_out_flat = mcore_out.squeeze(1)  # [s, 1, h] → [s, h]

        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=1e-2, rtol=1e-2,
            msg=lambda msg: f"[und-only] Mismatch: {msg}",
        )

    def test_forward_train_gen_tokens_only(self, num_heads, num_kv_heads):
        """All-generation-token forward: both models must produce close outputs."""
        bagel_attn, mcore_attn = self._build_models()
        seq_len = self.SEQ_LEN

        packed_seq, hidden_states, _, _ = self._make_inputs(und_ratio=0.0)
        und_idx = torch.zeros(0, dtype=torch.long, device="cuda")
        gen_idx = torch.arange(seq_len, device="cuda")

        cos, sin = self._identity_pos_emb(seq_len)
        attn_mask = self._bagel_attention_mask(seq_len)
        psp = self._make_psp(und_idx, gen_idx)
        bm = _block_mask(seq_len, "cuda")

        with torch.no_grad():
            bagel_out = bagel_attn.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[seq_len],
                attention_mask=attn_mask,
                packed_position_embeddings=(cos, sin),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )

        with torch.no_grad():
            mcore_out, _ = mcore_attn._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        mcore_out_flat = mcore_out.squeeze(1)

        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=1e-2, rtol=1e-2,
            msg=lambda msg: f"[gen-only] Mismatch: {msg}",
        )

    def test_forward_train_mixed_tokens(self, num_heads, num_kv_heads):
        """Mixed und/gen token forward with 50/50 split."""
        bagel_attn, mcore_attn = self._build_models()
        seq_len = self.SEQ_LEN

        packed_seq, hidden_states, und_idx, gen_idx = self._make_inputs(und_ratio=0.5)

        cos, sin = self._identity_pos_emb(seq_len)
        attn_mask = self._bagel_attention_mask(seq_len)
        psp = self._make_psp(und_idx, gen_idx)
        bm = _block_mask(seq_len, "cuda")

        with torch.no_grad():
            bagel_out = bagel_attn.forward_train(
                packed_sequence=packed_seq,
                sample_lens=[seq_len],
                attention_mask=attn_mask,
                packed_position_embeddings=(cos, sin),
                packed_und_token_indexes=und_idx,
                packed_gen_token_indexes=gen_idx,
            )

        with torch.no_grad():
            mcore_out, _ = mcore_attn._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        mcore_out_flat = mcore_out.squeeze(1)

        assert torch.all(~torch.isnan(mcore_out_flat)), "mcore output contains NaN"
        assert torch.all(~torch.isnan(bagel_out)), "bagel output contains NaN"

        torch.testing.assert_close(
            mcore_out_flat, bagel_out, atol=1e-2, rtol=1e-2,
            msg=lambda msg: f"[mixed] Mismatch: {msg}",
        )

    def test_und_gen_use_different_projections(self, num_heads, num_kv_heads):
        """Verify und and gen tokens go through separate QKV projections."""
        _, mcore_attn = self._build_models()
        seq_len = self.SEQ_LEN

        with torch.no_grad():
            mcore_attn.linear_qkv_gen.weight.data.fill_(0.0)
            mcore_attn.linear_qkv_gen.bias.data.fill_(1.0)

        _, hidden_states, und_idx, gen_idx = self._make_inputs(und_ratio=0.5)
        psp = self._make_psp(und_idx, gen_idx)
        bm = _block_mask(seq_len, "cuda")

        with torch.no_grad():
            out_mixed, _ = mcore_attn._forward_train(
                hidden_states=hidden_states,
                attention_mask=bm,
                packed_seq_params=psp,
            )

        out_flat = out_mixed.squeeze(1)  # [s, h]
        n_und = len(und_idx)

        # Und tokens are at [:n_und], gen at [n_und:] in the compact layout
        und_out = out_flat[:n_und]
        gen_out = out_flat[n_und:]

        assert not torch.allclose(
            und_out.mean(dim=0), gen_out.mean(dim=0), atol=1e-3
        ), "und and gen outputs should differ when using different projections"

    def test_qkv_projection_equivalence(self, num_heads, num_kv_heads):
        """Verify QKV projection (before attention) is numerically equivalent."""
        bagel_attn, mcore_attn = self._build_models()
        seq_len = self.SEQ_LEN

        packed_seq, hidden_states, und_idx, gen_idx = self._make_inputs(und_ratio=0.5)
        h = self.HIDDEN_SIZE
        ng = self.num_kv_heads
        np = self.num_heads
        hn = self.head_dim

        with torch.no_grad():
            und_seq = packed_seq[und_idx]
            gen_seq = packed_seq[gen_idx]

            bagel_q = packed_seq.new_zeros(seq_len, np * hn)
            bagel_k = packed_seq.new_zeros(seq_len, ng * hn)
            bagel_v = packed_seq.new_zeros(seq_len, ng * hn)

            bagel_q[und_idx] = bagel_attn.q_proj(und_seq)
            bagel_q[gen_idx] = bagel_attn.q_proj_moe_gen(gen_seq)
            bagel_k[und_idx] = bagel_attn.k_proj(und_seq)
            bagel_k[gen_idx] = bagel_attn.k_proj_moe_gen(gen_seq)
            bagel_v[und_idx] = bagel_attn.v_proj(und_seq)
            bagel_v[gen_idx] = bagel_attn.v_proj_moe_gen(gen_seq)

            # MCore: packed QKV projection, then split
            mcore_qkv_flat = hidden_states.view(-1, h)
            qkv_output = hidden_states.new_zeros(seq_len, mcore_attn.linear_qkv_out_dim)

            und_hidden = mcore_qkv_flat[und_idx]
            gen_hidden = mcore_qkv_flat[gen_idx]
            und_qkv, _ = mcore_attn.linear_qkv(und_hidden)
            gen_qkv, _ = mcore_attn.linear_qkv_gen(gen_hidden)
            qkv_output[und_idx] = und_qkv
            qkv_output[gen_idx] = gen_qkv

            qkv_3d = qkv_output.unsqueeze(1)  # [s, 1, qkv_dim]
            mcore_q, mcore_k, mcore_v = mcore_attn._split_qkv(qkv_3d)
            mcore_q = mcore_q.squeeze(1).reshape(seq_len, np * hn)
            mcore_k = mcore_k.squeeze(1).reshape(seq_len, ng * hn)
            mcore_v = mcore_v.squeeze(1).reshape(seq_len, ng * hn)

        torch.testing.assert_close(
            mcore_q, bagel_q, atol=1e-3, rtol=1e-3,
            msg=lambda msg: f"Query projection mismatch: {msg}",
        )
        torch.testing.assert_close(
            mcore_k, bagel_k, atol=1e-3, rtol=1e-3,
            msg=lambda msg: f"Key projection mismatch: {msg}",
        )
        torch.testing.assert_close(
            mcore_v, bagel_v, atol=1e-3, rtol=1e-3,
            msg=lambda msg: f"Value projection mismatch: {msg}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: CP parity tests (torchrun, world_size=2/4/8)
# ─────────────────────────────────────────────────────────────────────────────


def run_cp_parity_test(u, g, nh, hd, tp_group, cp_group, seed=42):
    """
    Compare SelfAttentionMoT._forward_train: cp=1 (full sequence, no CP)
    vs cp=N (type-balanced Ulysses A2A).

    Both models have identical weights (constructed with the same manual_seed).

    Checks:
      - Output shape: [Lund+Lgen, 1, hidden]
      - Forward parity: real und and gen token outputs match cp=1 within atol=1e-2
    """
    rank    = dist.get_rank()
    device  = "cuda"
    cp_size = cp_group.size()
    hidden  = nh * hd
    config  = _make_config_bf16(nh, hd)
    bm      = _block_mask(u + g, device)

    torch.manual_seed(seed)
    hs_full = torch.randn(u + g, 1, hidden, dtype=torch.bfloat16, device=device)

    # Type-balanced sharding metadata
    Lund = math.ceil(u / cp_size)
    Lgen = math.ceil(g / cp_size)
    actual_lund = min(Lund, max(0, u - rank * Lund))
    actual_lgen = min(Lgen, max(0, g - rank * Lgen))

    und_idx = torch.arange(u, device=device)
    gen_idx = torch.arange(u, u + g, device=device)
    local_und_idx = und_idx[rank * Lund : rank * Lund + actual_lund]
    local_gen_idx = gen_idx[rank * Lgen : rank * Lgen + actual_lgen]

    def _pad_slice(hs, start, actual, padded):
        chunk = hs[start : start + actual]
        if padded > actual:
            chunk = torch.cat([chunk, hs.new_zeros(padded - actual, 1, hidden)], dim=0)
        return chunk

    hs_local = torch.cat([
        _pad_slice(hs_full, rank * Lund,   actual_lund, Lund),
        _pad_slice(hs_full, u + rank*Lgen, actual_lgen, Lgen),
    ], dim=0)  # [Lund+Lgen, 1, H]

    psp_cp1 = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx, packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=und_idx,  local_gen_token_indexes=gen_idx,
        padded_und_seqlen=u, padded_gen_seqlen=g,
    )
    psp_cpN = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx, packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=local_und_idx, local_gen_token_indexes=local_gen_idx,
        padded_und_seqlen=Lund, padded_gen_seqlen=Lgen,
    )

    model_seed = seed + 100
    attn_cp1 = _make_attention(config, tp_group, None,     seed=model_seed)
    attn_cpN = _make_attention(config, tp_group, cp_group, seed=model_seed)

    with torch.no_grad():
        out_cp1, _ = attn_cp1._forward_train(hs_full,  attention_mask=bm, packed_seq_params=psp_cp1)
        out_cpN, _ = attn_cpN._forward_train(hs_local, attention_mask=bm, packed_seq_params=psp_cpN)

    dist.barrier()

    assert out_cpN.shape == (Lund + Lgen, 1, hidden), (
        f"cp={cp_size} rank={rank}: shape {out_cpN.shape} != ({Lund+Lgen},1,{hidden})"
    )

    und_ref = out_cp1[rank * Lund : rank * Lund + actual_lund]
    gen_ref = out_cp1[u + rank * Lgen : u + rank * Lgen + actual_lgen]
    und_got = out_cpN[:actual_lund]
    gen_got = out_cpN[Lund : Lund + actual_lgen]

    atol = rtol = 1e-2
    if actual_lund > 0:
        torch.testing.assert_close(und_got, und_ref, atol=atol, rtol=rtol,
            msg=lambda m: f"[SelfAttentionMoT cp={cp_size} rank={rank} UND]: {m}")
    if actual_lgen > 0:
        torch.testing.assert_close(gen_got, gen_ref, atol=atol, rtol=rtol,
            msg=lambda m: f"[SelfAttentionMoT cp={cp_size} rank={rank} GEN]: {m}")

    und_err = (und_got - und_ref).abs().max().item() if actual_lund > 0 else 0.0
    gen_err = (gen_got - gen_ref).abs().max().item() if actual_lgen > 0 else 0.0
    print(f"  [cp={cp_size} SelfAttentionMoT rank={rank:2d}] PASS  "
          f"Lund={Lund} Lgen={Lgen}  und_err={und_err:.4f}  gen_err={gen_err:.4f}")


def run_single_branch_test(branch, u, g, nh, hd, tp_group, cp_group, seed=99):
    """
    Compare SelfAttentionMoT._forward_train for single-branch inputs:
      branch='und'  U>0, G=0 — only understanding tokens
      branch='gen'  U=0, G>0 — only generation tokens

    Checks:
      A — Output shape: [Lund,1,H] for und-only; [Lgen,1,H] for gen-only
      B — Forward parity: real-token outputs match cp=1 within atol=1e-2
      C — Gradient flow: backward through real-token outputs gives non-zero grad
    """
    assert branch in ("und", "gen")
    rank    = dist.get_rank()
    device  = "cuda"
    cp_size = cp_group.size()
    hidden  = nh * hd
    config  = _make_config_bf16(nh, hd)

    total = u + g
    bm    = _block_mask(total, device)

    torch.manual_seed(seed)
    hs_full = torch.randn(total, 1, hidden, dtype=torch.bfloat16, device=device)

    Lund = math.ceil(u / cp_size)
    Lgen = math.ceil(g / cp_size)
    actual_lund = min(Lund, max(0, u - rank * Lund))
    actual_lgen = min(Lgen, max(0, g - rank * Lgen))

    und_idx = torch.arange(u, device=device)
    gen_idx = torch.arange(u, u + g, device=device)
    local_und_idx = und_idx[rank * Lund : rank * Lund + actual_lund]
    local_gen_idx = gen_idx[rank * Lgen : rank * Lgen + actual_lgen]

    def _pad_slice(hs, start, actual, padded):
        chunk = hs[start : start + actual]
        if padded > actual:
            chunk = torch.cat([chunk, hs.new_zeros(padded - actual, 1, hidden)], dim=0)
        return chunk

    hs_local = torch.cat([
        _pad_slice(hs_full, rank * Lund,   actual_lund, Lund),
        _pad_slice(hs_full, u + rank*Lgen, actual_lgen, Lgen),
    ], dim=0)

    psp_cp1 = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx, packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=und_idx,  local_gen_token_indexes=gen_idx,
        padded_und_seqlen=u, padded_gen_seqlen=g,
    )
    psp_cpN = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx, packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=local_und_idx, local_gen_token_indexes=local_gen_idx,
        padded_und_seqlen=Lund, padded_gen_seqlen=Lgen,
    )

    model_seed = seed + 100
    attn_cp1 = _make_attention(config, tp_group, None,     seed=model_seed)
    attn_cpN = _make_attention(config, tp_group, cp_group, seed=model_seed)

    # ── Check A+B: shape and forward parity ───────────────────────────────────
    with torch.no_grad():
        out_cp1, _ = attn_cp1._forward_train(hs_full,  attention_mask=bm, packed_seq_params=psp_cp1)
        out_cpN, _ = attn_cpN._forward_train(hs_local, attention_mask=bm, packed_seq_params=psp_cpN)

    dist.barrier()

    if branch == "und":
        assert out_cpN.shape == (Lund, 1, hidden), \
            f"[und-only] shape {out_cpN.shape} != ({Lund},1,{hidden})"
        if actual_lund > 0:
            ref = out_cp1[rank * Lund : rank * Lund + actual_lund]
            got = out_cpN[:actual_lund]
            torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2,
                msg=lambda m: f"[und-only cp={cp_size} rank={rank}]: {m}")
            err = (got - ref).abs().max().item()
        else:
            err = 0.0
    else:
        assert out_cpN.shape == (Lgen, 1, hidden), \
            f"[gen-only] shape {out_cpN.shape} != ({Lgen},1,{hidden})"
        if actual_lgen > 0:
            ref = out_cp1[rank * Lgen : rank * Lgen + actual_lgen]
            got = out_cpN[:actual_lgen]
            torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2,
                msg=lambda m: f"[gen-only cp={cp_size} rank={rank}]: {m}")
            err = (got - ref).abs().max().item()
        else:
            err = 0.0

    print(f"  [cp={cp_size} {branch}-only fwd  rank={rank:2d}] PASS  "
          f"Lund={Lund} Lgen={Lgen}  err={err:.4f}")

    # ── Check C: gradient flows into the active branch ────────────────────────
    hs_leaf = hs_local.detach().requires_grad_(True)
    out_g, _ = attn_cpN._forward_train(hs_leaf, attention_mask=bm, packed_seq_params=psp_cpN)

    if branch == "und" and actual_lund > 0:
        out_g[:actual_lund].sum().backward()
        grad_sum = hs_leaf.grad[:actual_lund].abs().sum().item()
        assert grad_sum > 0, \
            f"[und-only cp={cp_size} rank={rank}]: und grad is zero"
    elif branch == "gen" and actual_lgen > 0:
        out_g[:actual_lgen].sum().backward()
        grad_sum = hs_leaf.grad[:actual_lgen].abs().sum().item()
        assert grad_sum > 0, \
            f"[gen-only cp={cp_size} rank={rank}]: gen grad is zero"
    else:
        grad_sum = 0.0

    dist.barrier()
    print(f"  [cp={cp_size} {branch}-only grad rank={rank:2d}] PASS  "
          f"grad_sum={grad_sum:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for torchrun (Part 2)
# ─────────────────────────────────────────────────────────────────────────────


def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    assert world_size in CONFIGS, (
        f"No config for world_size={world_size}. Supported: {sorted(CONFIGS.keys())}"
    )
    cfg = CONFIGS[world_size]

    # Per-rank trivial tp group — ALL ranks must call new_group for every group.
    tp_groups = [dist.new_group(ranks=[r]) for r in range(world_size)]
    tp_group  = tp_groups[rank]

    # CP group spanning all ranks
    cp_group = dist.new_group(ranks=list(range(world_size)))

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  SelfAttentionMoT CP parity test")
        print(f"  cp={world_size}  nh={cfg.nh}  hd={cfg.hd}  HIDDEN={cfg.nh*cfg.hd}")
        print(f"  Mixed      : U={cfg.u}   G={cfg.g}"
              f"  (Lund={math.ceil(cfg.u/world_size)}"
              f"  Lgen={math.ceil(cfg.g/world_size)})")
        print(f"  und-only   : U={cfg.u_und}  G=0"
              f"  (Lund={math.ceil(cfg.u_und/world_size)})")
        print(f"  gen-only   : U=0   G={cfg.u_gen}"
              f"  (Lgen={math.ceil(cfg.u_gen/world_size)})")
        print(f"{'='*60}")

    dist.barrier()

    # ── Mixed (und + gen) ─────────────────────────────────────────────────────
    if rank == 0:
        print("\n--- Mixed (und + gen) ---")
    dist.barrier()
    run_cp_parity_test(cfg.u, cfg.g, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    # ── Single-branch: und-only ───────────────────────────────────────────────
    if rank == 0:
        print("\n--- Single-branch: und-only (G=0) ---")
    dist.barrier()
    run_single_branch_test("und", cfg.u_und, 0, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    # ── Single-branch: gen-only ───────────────────────────────────────────────
    if rank == 0:
        print("\n--- Single-branch: gen-only (U=0) ---")
    dist.barrier()
    run_single_branch_test("gen", 0, cfg.u_gen, cfg.nh, cfg.hd, tp_group, cp_group)
    dist.barrier()

    if rank == 0:
        print(f"\nAll cp={world_size} SelfAttentionMoT tests passed.")

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
