import math
import os
import time

import pytest
import torch
import torch.nn.functional as F
from transformer_engine.pytorch.attention import UnfusedDotProductAttention, _attention_backends

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.common.embeddings import YarnRotaryEmbedding, _yarn_get_mscale
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import get_attention_sink_bias
from megatron.core.transformer.transformer_config import TransformerConfig

dtype = torch.bfloat16
tols = dict(atol=1e-3, rtol=1e-3)
if dtype == torch.bfloat16:
    tols = dict(atol=1.5e-2, rtol=1.5e-2)

# Note: NVTE_DEBUG, NVTE_DEBUG_LEVEL must be set in shell script
# os.environ["NVTE_DEBUG"] = "1"
# os.environ["NVTE_DEBUG_LEVEL"] = "2"


def get_transformer_config():
    config = TransformerConfig
    config.apply_query_key_layer_scaling = False
    config.num_query_groups = 40
    config.num_attention_heads = 40
    config.context_parallel_size = 1
    config.deterministic_mode = False
    config.window_size = None
    config.kv_channels = 128
    config.attention_dropout = 0.0
    config.sequence_parallel = False
    config.tensor_model_parallel_size = 1
    # TODO
    # Note(wenx): if true, the TEDotProductAttention will transpose
    # qkv from [sbhd] -> [bshd], then run DotProductAttention, and
    # transpose out from [bs h*d] -> [sb h*d], the result is not that
    # correct.
    config.apply_rope_fusion = False
    # config.apply_rope_fusion = True
    config.rotary_scaling_factor = 40.0
    config.mscale = 0.707
    config.qk_head_dim = 128
    config.qk_pos_emb_head_dim = 64  # mla: 64, mha: 0
    config.v_head_dim = 128  # nopad: 128, pad: 192
    config.window_size = None
    config.attention_sink_k = 0
    # config. =

    return config


def run_attention(
    attn_module,
    query,
    key,
    value,
    out_grad,
    attention_bias_type,
    attention_bias,
    fused=True,
    need_padding=False,
    iteration=20,
):
    kwargs = {
        "attention_mask": None,
        "attn_mask_type": AttnMaskType.causal,
        "attention_bias": attention_bias,
    }

    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    seq = query.shape[0]
    batch = query.shape[1]
    num_heads = query.shape[2]
    q_head_dim = query.shape[3]
    v_head_dim = value.shape[3]
    assert key.shape == query.shape
    assert q_head_dim >= v_head_dim

    # warmup
    with torch.no_grad():
        for _ in range(10):
            if need_padding:
                assert q_head_dim > v_head_dim
                padded_dim = q_head_dim - v_head_dim
                # pad value to q_head_dim
                value_padded = F.pad(value, (0, padded_dim))
            else:
                value_padded = value
            attn_out_padded = attn_module(query, key, value_padded, **kwargs)
    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []
    for _ in range(iteration):
        query.grad = None
        key.grad = None
        value.grad = None
        torch.cuda.synchronize()

        t_s = time.time()
        if need_padding:
            assert q_head_dim > v_head_dim
            padded_dim = q_head_dim - v_head_dim
            # pad value to q_head_dim
            value_padded = F.pad(value, (0, padded_dim))
        else:
            value_padded = value
        attn_out_padded = attn_module(query, key, value_padded, **kwargs)
        if need_padding:
            # [s, b, n * dim] -> [s, b, n, dim=192] -> [s, b, n, dim=128] -> [s, b, n*dim]
            attn_out = attn_out_padded.reshape(seq, batch, num_heads, q_head_dim)[..., :v_head_dim]
            attn_out = attn_out.reshape(seq, batch, num_heads * v_head_dim)
        else:
            attn_out = attn_out_padded
        torch.cuda.synchronize()
        fwd_times.append(time.time() - t_s)

        t_s = time.time()
        attn_out.backward(out_grad)
        torch.cuda.synchronize()
        bwd_times.append(time.time() - t_s)

    fwd_times = fwd_times[2:]
    bwd_times = bwd_times[2:]
    avg_fwd_time = 1000.0 * (sum(fwd_times) / len(fwd_times))
    avg_bwd_time = 1000.0 * (sum(bwd_times) / len(bwd_times))

    return (
        attn_out,
        avg_fwd_time,
        (query.grad.clone(), key.grad.clone(), value.grad.clone()),
        avg_bwd_time,
    )


def check_tensor(name, a, b, atol=1e-04, rtol=1e-05):
    a = a.flatten()
    b = b.flatten()

    # Find mismatched elements
    close_mask = torch.isclose(a, b, atol=atol, rtol=rtol)
    # Get indices where elements are not close
    mismatched_indices = (~close_mask).nonzero(as_tuple=True)[0]

    # Print up to 10 mismatches
    if len(mismatched_indices) > 0:
        print(f"Check {name} failed:")
        print(f"  >Total mismatches: {len(mismatched_indices)}")
        for idx in mismatched_indices[:20]:  # Limit to 10 mismatches
            print(f"  >Mismatch at index {idx}: a={a[idx].item()}, b={b[idx].item()}")
        return False
    else:
        print(f"Check {name} pass, all elements are within tolerance!")
        return True


@pytest.mark.parametrize("head_dim", [128, 192, 256])
@pytest.mark.parametrize("seq_length", [4096])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_normal_attention(head_dim: int, seq_length: int, batch_size: int):
    config = get_transformer_config()
    config.qk_head_dim = head_dim
    config.qk_pos_emb_head_dim = 0
    config.v_head_dim = head_dim

    q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
    mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale)
    softmax_scale = mscale * mscale / math.sqrt(q_head_dim)
    # softmax_scale = None

    # query.shape=torch.Size([128, 1, 40, 128]), key.shape=torch.Size([128, 1, 40, 128]),
    # value.shape=torch.Size([128, 1, 40, 128]), attention_mask.shape=torch.Size([1, 1, 128, 128]),
    # attn_mask_type=<AttnMaskType.causal: 2>
    # [seq, batch, num_heads, head_dim]
    torch.manual_seed(42)
    query = 0.1 * torch.randn(
        (seq_length, batch_size, config.num_attention_heads, q_head_dim), dtype=dtype, device="cuda"
    )
    key = 0.1 * torch.randn(
        (seq_length, batch_size, config.num_attention_heads, q_head_dim), dtype=dtype, device="cuda"
    )
    value = 0.1 * torch.randn(
        (seq_length, batch_size, config.num_attention_heads, config.v_head_dim),
        dtype=dtype,
        device="cuda",
    )
    out_grad = 0.001 * torch.randint(
        0,
        200,
        (seq_length, batch_size, config.num_attention_heads * config.v_head_dim),
        dtype=dtype,
        device="cuda",
    )

    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    attention_bias_type = "no_bias"
    attention_bias = None
    torch.cuda.synchronize()

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN_CK"] = "0"
    os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    _attention_backends["backend_selection_requires_update"] = True
    te_attention = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=softmax_scale,
        cp_comm_type=None,
        k_channels=q_head_dim,
        v_channels=config.v_head_dim,
    ).to(dtype=dtype, device="cuda")
    te_attn_fwd, te_attn_fwd_time, te_attn_bwd, te_attn_bwd_time = run_attention(
        te_attention,
        query,
        key,
        value,
        out_grad,
        attention_bias_type,
        attention_bias,
        fused=True,
        iteration=20,
    )

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    os.environ["NVTE_MASKED_SOFTMAX_FUSION"] = "0"
    _attention_backends["backend_selection_requires_update"] = True
    unfused_attention = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=softmax_scale,
        cp_comm_type=None,
        k_channels=q_head_dim,
        v_channels=config.v_head_dim,
    ).to(dtype=dtype, device="cuda")
    unfused_attn_fwd, unfused_attn_fwd_time, unfused_attn_bwd, unfused_attn_bwd_time = (
        run_attention(
            unfused_attention,
            query,
            key,
            value,
            out_grad,
            attention_bias_type,
            attention_bias,
            fused=False,
            iteration=20,
        )
    )

    # core_attn_out.shape=torch.Size([128, 1, 5120])
    print(f"\nTest normal attention:", flush=True)
    print(f"  >{query.shape=}", flush=True)
    print(f"  >{key.shape=}", flush=True)
    print(f"  >{value.shape=}", flush=True)
    print(f"attn_out:")
    print(f"  >{te_attn_fwd.shape=}, \n  >{te_attn_fwd[1][0][:5]}", flush=True)
    print(f"  >{unfused_attn_fwd.shape=}, \n  >{unfused_attn_fwd[1][0][:5]}", flush=True)
    print(f"time:")
    print(
        f"  >{te_attn_fwd_time=:.3f} ms, \n  >{unfused_attn_fwd_time=:.3f} ms, \n  >speedup={unfused_attn_fwd_time/te_attn_fwd_time:.3f} x\n",
        flush=True,
    )
    print(
        f"  >{te_attn_bwd_time=:.3f} ms, \n  >{unfused_attn_bwd_time=:.3f} ms, \n  >speedup={unfused_attn_bwd_time/te_attn_bwd_time:.3f} x\n",
        flush=True,
    )

    check_tensor("attn_out", te_attn_fwd, unfused_attn_fwd, **tols)
    check_tensor("query.grad", te_attn_bwd[0], unfused_attn_bwd[0], **tols)
    check_tensor("key.grad", te_attn_bwd[1], unfused_attn_bwd[1], **tols)
    check_tensor("value.grad", te_attn_bwd[2], unfused_attn_bwd[2], **tols),


@pytest.mark.parametrize("qk_head_dim", [128])
@pytest.mark.parametrize("qk_pos_emb_head_dim", [64])
@pytest.mark.parametrize("v_head_dim", [128])
@pytest.mark.parametrize("seq_length", [4096, 8192, 16384])  # coredump when bs=2, seq=8k/16k
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("window_size", [4096])
@pytest.mark.parametrize("attention_sink_k", [4])
def test_mla_attention(
    qk_head_dim: int,
    qk_pos_emb_head_dim: int,
    v_head_dim: int,
    seq_length: int,
    batch_size: int,
    window_size: int,
    attention_sink_k: int,
):
    config = get_transformer_config()
    config.qk_head_dim = qk_head_dim
    config.qk_pos_emb_head_dim = qk_pos_emb_head_dim
    config.v_head_dim = v_head_dim
    if window_size == 0 or window_size >= seq_length:
        # causal attention
        config.window_size = None
    else:
        config.window_size = (window_size, 0)
    config.attention_sink_k = attention_sink_k

    q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
    mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale)
    softmax_scale = mscale * mscale / math.sqrt(q_head_dim)

    # query.shape=torch.Size([128, 1, 40, 192]), key.shape=torch.Size([128, 1, 40, 192]),
    # value.shape=torch.Size([128, 1, 40, 128]), attention_mask.shape=torch.Size([1, 1, 128, 128]),
    # attn_mask_type=<AttnMaskType.causal: 2>
    # [seq, batch, num_heads, head_dim]
    torch.manual_seed(42)
    query = 0.1 * torch.randn(
        (seq_length, batch_size, config.num_attention_heads, q_head_dim), dtype=dtype, device="cuda"
    )
    key = 0.1 * torch.randn(
        (seq_length, batch_size, config.num_attention_heads, q_head_dim), dtype=dtype, device="cuda"
    )
    value = 0.1 * torch.randn(
        (seq_length, batch_size, config.num_attention_heads, config.v_head_dim),
        dtype=dtype,
        device="cuda",
    )
    out_grad = 0.001 * torch.randint(
        0,
        200,
        (seq_length, batch_size, config.num_attention_heads * config.v_head_dim),
        dtype=dtype,
        device="cuda",
    )

    query.requires_grad = True
    key.requires_grad = True
    value.requires_grad = True

    attention_bias_type = 'post_scale_bias'
    attention_bias = get_attention_sink_bias(
        batch_size,
        config.num_attention_heads,
        seq_length,
        config.window_size,
        config.attention_sink_k,
        dtype,
    )
    # attention_bias = torch.rand(
    #     (batch_size, config.num_attention_heads, seq_length, seq_length), dtype=dtype, device="cuda"
    # )
    torch.cuda.synchronize()

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    # use ck
    os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
    os.environ["NVTE_FUSED_ATTN_CK"] = "1"
    os.environ["NVTE_FUSED_ATTN_AOTRITON"] = "0"
    _attention_backends["backend_selection_requires_update"] = True

    need_padding = q_head_dim != config.v_head_dim

    te_attention = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=softmax_scale,
        cp_comm_type=None,
        k_channels=q_head_dim,
        v_channels=config.v_head_dim if not need_padding else q_head_dim,
    ).to(dtype=dtype, device="cuda")
    te_attn_fwd, te_attn_fwd_time, te_attn_bwd, te_attn_bwd_time = run_attention(
        te_attention,
        query,
        key,
        value,
        out_grad,
        attention_bias_type,
        attention_bias,
        fused=True,
        need_padding=need_padding,
        iteration=20,
    )

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    _attention_backends["backend_selection_requires_update"] = True
    unfused_attention = TEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=softmax_scale,
        cp_comm_type=None,
        k_channels=q_head_dim,
        v_channels=config.v_head_dim,
    ).to(dtype=dtype, device="cuda")
    unfused_attn_fwd, unfused_attn_fwd_time, unfused_attn_bwd, unfused_attn_bwd_time = (
        run_attention(
            unfused_attention,
            query,
            key,
            value,
            out_grad,
            attention_bias_type,
            attention_bias,
            fused=False,
            iteration=20,
        )
    )

    # core_attn_out.shape=torch.Size([128, 1, 5120])
    print(f"\nTest mla attention:", flush=True)
    print(f"  >{query.shape=}", flush=True)
    print(f"  >{key.shape=}", flush=True)
    print(f"  >{value.shape=}", flush=True)
    print(f"attn_out:")
    print(f"  >{te_attn_fwd.shape=}, \n  >{te_attn_fwd[1][0][:5]}", flush=True)
    print(f"  >{unfused_attn_fwd.shape=}, \n  >{unfused_attn_fwd[1][0][:5]}", flush=True)
    print(f"time:")
    print(
        f"  >{te_attn_fwd_time=:.3f} ms, \n  >{unfused_attn_fwd_time=:.3f} ms, \n  >speedup={unfused_attn_fwd_time/te_attn_fwd_time:.3f} x\n",
        flush=True,
    )
    print(
        f"  >{te_attn_bwd_time=:.3f} ms, \n  >{unfused_attn_bwd_time=:.3f} ms, \n  >speedup={unfused_attn_bwd_time/te_attn_bwd_time:.3f} x\n",
        flush=True,
    )

    check_tensor("attn_out", te_attn_fwd, unfused_attn_fwd, **tols)
    check_tensor("query.grad", te_attn_bwd[0], unfused_attn_bwd[0], **tols)
    check_tensor("key.grad", te_attn_bwd[1], unfused_attn_bwd[1], **tols)
    check_tensor("value.grad", te_attn_bwd[2], unfused_attn_bwd[2], **tols),
