# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch
from megatron.core.transformer.adaptive_layernorm import AdaLayerNormZero
from megatron.core.transformer import TransformerConfig


hidden_size = 16
cond_dim = 32
config = TransformerConfig(hidden_size=hidden_size)
ln = AdaLayerNormZero(config, hidden_size, cond_dim)
x = torch.randn(2, 8, hidden_size)
cond = torch.randn(2, cond_dim)
x_out, gate_out = ln(x, cond)
x_ref = ln.norm(x)
torch.testing.assert_close(x_out, x_ref)
torch.testing.assert_close(gate_out, torch.zeros_like(gate_out))


hidden_size = 16
cond_dim = 32
config = TransformerConfig(hidden_size=hidden_size)
ln = AdaLayerNormZero(config, hidden_size, cond_dim)
nn.init.ones_(ln.cond_proj.weight)
x = torch.randn(2, 8, hidden_size)
cond = torch.randn(2, cond_dim)
x_out, gate_out = ln(x, cond)
x_ref = ln.norm(x)
assert not torch.allclose(x_out, x_ref)
assert not torch.allclose(gate_out, torch.zeros_like(gate_out))


hidden_size = 16
cond_dim = 16
config = TransformerConfig(hidden_size=hidden_size)
ln = AdaLayerNormZero(config, hidden_size, cond_dim)
x = torch.randn(2, 8, hidden_size)
cond = torch.randn(2, cond_dim)
x_out, gate_out = ln(x, cond)
assert x_out.shape == x.shape
assert gate_out.shape == (2, 1, hidden_size)
