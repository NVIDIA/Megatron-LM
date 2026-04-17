"""Model FLOPs Utilization for dense and MoE transformers.

Formula (activation-checkpoint free, causal mask, no recompute):

    fwd_flops_per_token = 2 * (n_active_params
                                + 2 * n_layers * seq_len * hidden / 2)
    total_flops_per_token = 3 * fwd_flops_per_token              # fwd+bwd
    mfu = (total_flops_per_token * tokens_per_sec) / peak_flops

The activation term models causal self-attention's O(seq^2) cost; the factor
of 2 counts Q*K and P*V matmuls. For MoE we count only active (routed)
params, not total. Peak BF16 on GH200 is 989 TFLOPs/sec (dense, no sparsity).
"""

from __future__ import annotations

GH200_BF16_TFLOPS = 989.0


def flops_per_token(n_active_params: int, n_layers: int, seq_len: int, hidden: int) -> float:
    """Total (fwd + bwd) FLOPs to process one token. 6N + 6*L*S*H attention."""
    fwd = 2.0 * n_active_params + 2.0 * n_layers * seq_len * hidden
    return 3.0 * fwd


def compute_mfu(
    n_active_params: int,
    n_layers: int,
    seq_len: int,
    hidden: int,
    tokens_per_sec_per_gpu: float,
    peak_tflops: float = GH200_BF16_TFLOPS,
) -> float:
    fpt = flops_per_token(n_active_params, n_layers, seq_len, hidden)
    return (fpt * tokens_per_sec_per_gpu) / (peak_tflops * 1e12)
