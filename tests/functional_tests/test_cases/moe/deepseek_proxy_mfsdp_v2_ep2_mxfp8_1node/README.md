# DeepSeek proxy: MFSDP v2 MXFP8 versus BF16

This four-GPU test runs the eight-layer DeepSeek proxy (three dense layers,
five MoE layers, MLA, EP2) for 50 steps with MXFP8 compute and MXFP8 parameter
all-gather. MoE router padding satisfies the quantized grouped-GEMM alignment
requirements.

The golden `lm loss` values are the **BF16 reference**, copied without rounding
from `../deepseek_proxy_mfsdp_v2_ep2_bf16_cg_optim_1node/golden_values_dev_dgx_gb200.json`.
That reference uses the same model, data, batch sizes, initialization, optimizer,
and learning-rate schedule. It additionally enables full-iteration CUDA graphs
and HybridEP; this test uses the eager all-to-all path from
`../deepseek_proxy_mfsdp_v2_ep2/model_config.yaml`.

The regular functional-test harness compares every recorded training step with
its existing approximate `lm loss` check (`rtol=0.05`, `atol=0`, allowing
one outlier in a 50-step curve).
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` selects approximate comparison; bit-exact
agreement is not expected between MXFP8 and BF16. Memory and timing metrics are
not compared across precisions.

Keep these goldens as a BF16 baseline. Do not refresh them from this test's MXFP8
output. To regenerate a matching eager BF16 reference, run this configuration
with `--fp8-format`, `--fp8-recipe`, `--fp8-param-gather`, and
`--moe-router-padding-for-quantization` removed, keeping the
same seed, dataset, four-GPU topology, and remaining settings. Collect all 50
steps with `get_test_results_from_tensorboard_logs.py --step-size 1` and retain
the full-precision `lm loss` metric.
