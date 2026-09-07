from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.lite.model.deepseek_v4.vllm import model as model_module
from megatron.lite.model.deepseek_v4.vllm.model import DeepseekV4Layer
from megatron.lite.model.deepseek_v4.vllm.primitive import dense as vllm_ds4
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import mhc_kernel


def _post_inputs(device: str = "cpu") -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(4, 4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(4, 4, 1, dtype=torch.float32, device=device),
        torch.randn(4, 4, 4, dtype=torch.float32, device=device),
    )


@pytest.mark.parametrize("kernel", list(vllm_ds4._MHC_ENTRIES))
def test_each_mhc_call_uses_the_official_entry(
    monkeypatch: pytest.MonkeyPatch, kernel: str
) -> None:
    result = torch.tensor([7])
    official = Mock(return_value=result)
    monkeypatch.setitem(vllm_ds4._MHC_ENTRIES, kernel, official)
    if kernel == "post":
        args = _post_inputs()
    elif kernel == "head":
        args = (
            torch.zeros(2, 4, 128, dtype=torch.bfloat16),
            torch.zeros(4, 512),
            torch.zeros(1),
            torch.zeros(4),
            1e-6,
            1e-6,
        )
    else:
        residual = torch.zeros(2, 128, dtype=torch.bfloat16)
        if kernel != "pre_broadcast":
            residual = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
        pre = (
            residual,
            torch.zeros(24, 512),
            torch.zeros(3),
            torch.zeros(24),
            1e-6,
            1e-6,
            1e-6,
            2.0,
            2,
        )
        args = _post_inputs() + pre[1:] if kernel == "post_pre" else pre
    assert mhc_kernel(kernel, *args) is result
    official.assert_called_once_with(*args)


def test_layer_matches_lite_unfused_pre_block_post_sequence(monkeypatch) -> None:
    hidden_size, hc_mult, tokens = 8, 2, 3
    config = type(
        "Config",
        (),
        {
            "hidden_size": hidden_size,
            "hc_mult": hc_mult,
            "rms_norm_eps": 1e-6,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 2,
        },
    )()
    layer = DeepseekV4Layer.__new__(DeepseekV4Layer)
    nn.Module.__init__(layer)
    layer.config = config
    layer.layer_idx = 0

    def hc_state():
        state = nn.Module()
        state.fn = nn.Parameter(torch.zeros((2 + hc_mult) * hc_mult, hc_mult * hidden_size))
        state.base = nn.Parameter(torch.zeros((2 + hc_mult) * hc_mult))
        state.scale = nn.Parameter(torch.ones(3))
        return state

    layer.attn_hc = hc_state()
    layer.ffn_hc = hc_state()
    layer.input_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
    layer.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)

    attention_inputs = []

    class Attention(nn.Module):
        def forward(self, value, *, metadata):
            attention_inputs.append(value.clone())
            return value

    class MLP(nn.Module):
        def forward(self, value, *, input_ids):
            return value

    layer.self_attn = Attention()
    layer.mlp = MLP()
    calls = []

    def fake_kernel(kernel, *args, **kwargs):
        del kwargs
        calls.append(kernel)
        if kernel == "pre":
            streams = args[0]
            post = torch.zeros(tokens, hc_mult, 1)
            comb = torch.zeros(tokens, hc_mult, hc_mult)
            return post, comb, streams[:, 0] + 1
        if kernel == "post":
            return args[1]
        raise AssertionError(kernel)

    monkeypatch.setattr(model_module, "mhc_kernel", fake_kernel)
    streams = torch.zeros(tokens, hc_mult, hidden_size, dtype=torch.bfloat16)

    layer(
        streams,
        position_ids=torch.arange(tokens),
        attention_metadata=object(),
    )

    assert calls == [
        "pre",
        "post",
        "pre",
        "post",
    ]
    torch.testing.assert_close(attention_inputs[0], streams[:, 0] + 1)


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires the official vLLM package and compiled TileLang kernels",
)
def test_mhc_post_official_kernel_is_bitwise() -> None:
    from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang

    args = _post_inputs("cuda")
    reference = mhc_post_tilelang(*(value.clone() for value in args))
    candidate = mhc_kernel("post", *(value.clone() for value in args))
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires the official vLLM package and compiled TileLang kernels",
)
@pytest.mark.parametrize("kernel", ["pre_broadcast", "pre"])
def test_mhc_pre_is_bitwise_invariant_to_batch_composition(kernel: str) -> None:
    torch.manual_seed(42)
    mult, hidden, mixed_tokens, target_index = 4, 128, 17, 7
    width = mult * hidden
    mixes = (2 + mult) * mult
    target = torch.randn(
        1,
        hidden if kernel == "pre_broadcast" else mult,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if kernel == "pre":
        target = torch.randn(1, mult, hidden, dtype=torch.bfloat16, device="cuda")
        mixed = torch.randn(
            mixed_tokens, mult, hidden, dtype=torch.bfloat16, device="cuda"
        )
    else:
        mixed = torch.randn(
            mixed_tokens, hidden, dtype=torch.bfloat16, device="cuda"
        )
    mixed[target_index].copy_(target[0])
    fn = torch.randn(mixes, width, dtype=torch.float32, device="cuda")
    scale = torch.randn(3, dtype=torch.float32, device="cuda")
    base = torch.randn(mixes, dtype=torch.float32, device="cuda")
    norm_weight = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
    common = (fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 2)
    kwargs = {"norm_weight": norm_weight, "norm_eps": 1e-6}
    if kernel == "pre_broadcast":
        kwargs["fn_broadcast"] = (
            fn.view(-1, mult, hidden).sum(dim=1).contiguous()
        )

    target_outputs = mhc_kernel(kernel, target, *common, **kwargs)
    mixed_outputs = mhc_kernel(kernel, mixed, *common, **kwargs)
    if kernel == "pre_broadcast":
        # PRE_BROADCAST additionally materializes the residual streams.
        target_outputs = target_outputs[1:]
        mixed_outputs = mixed_outputs[1:]
    for target_output, mixed_output in zip(target_outputs, mixed_outputs):
        assert torch.equal(target_output[0], mixed_output[target_index])


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires the official vLLM package and compiled TileLang kernels",
)
def test_mhc_post_is_bitwise_invariant_to_batch_composition() -> None:
    torch.manual_seed(43)
    mult, hidden, mixed_tokens, target_index = 4, 128, 17, 7

    def values(tokens: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.randn(tokens, mult, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.randn(tokens, mult, 1, dtype=torch.float32, device="cuda"),
            torch.randn(tokens, mult, mult, dtype=torch.float32, device="cuda"),
        )

    target = values(1)
    mixed = values(mixed_tokens)
    for target_value, mixed_value in zip(target, mixed):
        mixed_value[target_index].copy_(target_value[0])
    target_output = mhc_kernel("post", *target)
    mixed_output = mhc_kernel("post", *mixed)
    assert torch.equal(target_output[0], mixed_output[target_index])
