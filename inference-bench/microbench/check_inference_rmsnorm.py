"""Check the FlashInfer RMSNorm path in inference_layers (_rms_norm_kernel, InferenceTENorm)
against TE.

    python inference-bench/microbench/check_inference_rmsnorm.py
"""

import inspect

import flashinfer
import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.constants import TE_DType

from megatron.core.tensor_parallel import inference_layers as il

EPS = 1e-5

print("flashinfer", flashinfer.__version__, "rmsnorm", inspect.signature(flashinfer.norm.rmsnorm))
print("HAVE_FLASHINFER_RMSNORM =", il.HAVE_FLASHINFER_RMSNORM)

worst = 0.0
for hidden in (8192, 4096, 2688):
    for shape in ((1, hidden), (2, hidden), (128, hidden), (2048, hidden), (3, 1, hidden)):
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 3
        w = torch.rand(hidden, device="cuda", dtype=torch.bfloat16) + 0.5
        ref, _, _ = tex.rmsnorm_fwd(
            x.view(-1, hidden), w, EPS, None, None, TE_DType[x.dtype], 16, False
        )
        out = il._rms_norm_kernel(x, w, EPS)
        assert out.shape == x.shape and out.dtype == x.dtype, (out.shape, out.dtype)
        diff = (out.float().view(-1, hidden) - ref.float()).abs().max().item()
        worst = max(worst, diff)
        print(f"  shape {str(shape):>16}: max |flashinfer - TE| = {diff:.4g}")

# Under CUDA graph capture, as in the engine.
x = torch.randn(2, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.rand(8192, device="cuda", dtype=torch.bfloat16) + 0.5
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    il._rms_norm_kernel(x, w, EPS)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = il._rms_norm_kernel(x, w, EPS)
x.copy_(torch.randn_like(x))
g.replay()
torch.cuda.synchronize()
ref, _, _ = tex.rmsnorm_fwd(x, w, EPS, None, None, TE_DType[x.dtype], 16, False)
graph_diff = (y.float() - ref.float()).abs().max().item()
print(f"  CUDA graph replay on new input: max |diff| = {graph_diff:.4g}")
# InferenceTENorm (MoE pre-norm): same parameters/state dict as TENorm, FlashInfer in eval.
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.transformer.transformer_config import TransformerConfig

cfg = TransformerConfig(
    num_layers=1, hidden_size=8192, num_attention_heads=64, normalization="RMSNorm",
    bf16=True, params_dtype=torch.bfloat16,
)
te_norm = TENorm(config=cfg, hidden_size=8192, eps=EPS).cuda()
inf_norm = il.InferenceTENorm(config=cfg, hidden_size=8192, eps=EPS).cuda()
with torch.no_grad():
    te_norm.weight.copy_(torch.rand(8192, device="cuda") + 0.5)
inf_norm.load_state_dict(te_norm.state_dict())
same_keys = set(te_norm.state_dict()) == set(inf_norm.state_dict())
print(f"  InferenceTENorm class={type(inf_norm).__name__} state-dict keys match TENorm: {same_keys}")
te_norm.eval(); inf_norm.eval()
x = torch.randn(2, 8192, device="cuda", dtype=torch.bfloat16) * 3
with torch.no_grad():
    norm_diff = (inf_norm(x).float() - te_norm(x).float()).abs().max().item()
print(f"  InferenceTENorm vs TENorm (eval): max |diff| = {norm_diff:.4g}")
ok = (
    worst < 0.07 and graph_diff < 0.07 and norm_diff < 0.07 and same_keys
    and type(inf_norm).__name__ == "_InferenceTERMSNorm"
)
print("PASS" if ok else "FAIL", f"(worst {max(worst, graph_diff, norm_diff):.4g})")
