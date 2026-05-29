# GDN CUDA Optimization Reproduction

This note covers the current GatedDeltaNet CUDA optimization test flow for
Megatron-LM on B200/H100. The optimized kernels are provided by
the `mcore_gdn_opt` Python package; Megatron can optionally route its
gated-delta-rule calls through that package without modifying FLA source.

## Install

Use a GPU container that already has `mcore_gdn_opt` and its CUDA extensions
installed, or install the package from the internal repository before running
Megatron-LM:

```bash
pip install -e . --user --no-build-isolation

git clone https://gitlab-master.nvidia.com/bhsueh/mcore_gdn_opt.git
cd mcore_gdn_opt
git checkout 3a371f53eeec24cc1d9da4a01584f3ec9f8f44e5
git submodule update --init --recursive
export CUTLASS_PATH="${PWD}/third_party/cutlass"

python -m pip install -e third_party/gated_delta_rule_bwd --no-build-isolation
python -m pip install -e chunk_bwd_kernel_dqkwg --no-build-isolation
python -m pip install -e chunk_gated_delta_rule_fwd --no-build-isolation
python -m pip install -e . --no-build-isolation
```

Do not use `PYTHONPATH` or ad-hoc `sys.modules` injection for these tests. The
package and CUDA extensions should be installed in editable mode.

## Runtime Flags

| Case | Flags |
|---|---|
| Triton baseline | `MCORE_GDN_USE_OPT_WRAPPER=0` |
| wrapper auto | `MCORE_GDN_USE_OPT_WRAPPER=1 MCORE_GDN_OPT_BACKEND=auto` |
| `wy_bwd` only | `MCORE_GDN_USE_OPT_WRAPPER=1 MCORE_GDN_OPT_BACKEND=cuda` with other optimized stages disabled |
| `dhu` only | `MCORE_GDN_USE_OPT_WRAPPER=1 MCORE_GDN_OPT_BACKEND=cuda` with other optimized stages disabled |
| `dqkwg` only | `MCORE_GDN_USE_OPT_WRAPPER=1 MCORE_GDN_OPT_BACKEND=cuda` with other optimized stages disabled |
| all three separate | `wy_bwd+dhu+dqkwg` enabled, `fwd_h` and `dv_dhu` disabled |
| all four | `fwd_h+wy_bwd+dhu+dqkwg` enabled, `dv_dhu` disabled |
| `fwd_h+wy_bwd+fused_dv_dhu+dqkwg` | `fwd_h+wy_bwd+dv_dhu+dqkwg` enabled, standalone `dhu` disabled |

The `dhu_dqkwg` wrapper is intentionally not exposed as a benchmark scenario
because the single-kernel DHU+DQKWG path is not implemented. The remaining
optimized scenarios use standalone `dhu`/`dqkwg` or the real fused `dv_dhu`
kernel.

## GDN-Only Direct Test

This bypasses the full GPT layer and measures a direct `GatedDeltaNet`
forward/backward. It checks output, input grad, and parameter grads against the
Triton baseline.

```bash
python -m tests.unit_tests.ssm.bench_gdn_cuda_opt \
  --dtype bf16 \
  --loss sum \
  --scenarios baseline,separate,all_four,fwd_h_wy_dv_dhu_dqkwg \
  --warmup 5 --repeats 20 --rounds 3
```

Use `--loss square_mean` to reproduce the earlier loss used during debugging,
and add `--fail-on-accuracy` when the command should return non-zero on any
accuracy mismatch.

Latest B200 spot check for
`B=2,T=8192,H=64,D=128,bf16,loss=sum,warmup=3,repeats=10,rounds=3`
on Megatron-LM `c42dc298a`, `mcore_gdn_opt@9121702`, and
`gated_delta_rule_bwd@949c959`:

| Scenario | Accuracy vs Triton | Mean us | Speedup |
|---|---:|---:|---:|
| Triton baseline | PASS | 15220.135 | 1.000x |
| CUDA all three separate | FAIL | 13420.346 | 1.134x |
| CUDA all four | FAIL | 12875.528 | 1.182x |

For this direct `loss=sum` GDN-only check, the optimized scenarios still fail
the strict gradient comparison against the Triton baseline. The current
production workflow is validated with `loss=square_mean`; the latest B200 full
workflow validation passed all requested scenarios and measured `CUDA all four`
at `12895.830 us` (`1.182x`) and `CUDA fwd_h+wy_bwd+fused_dv_dhu+dqkwg` at
`12732.651 us` (`1.197x`). Fresh logs:
`third_party/gdn_doc_loss_sum_20260528_205336.log` and
`third_party/gdn_full_validation_cb51345_20260528_204219.log`.

## E2E Pytest

This runs the focused GDN CUDA optimization pytest path. It checks correctness
by default and can print the benchmark table when `MCORE_GDN_UNIT_TEST_PERF=1`.

```bash
MCORE_GDN_UNIT_TEST_SCENARIOS=baseline,fwd_h_wy_dv_dhu_dqkwg \
pytest -s tests/unit_tests/ssm/test_gated_delta_net_cuda_opt.py::test_gated_delta_net_cuda_opt_correctness_and_optional_perf -k bf16
```

To generate the E2E benchmark table with NVTX labels:

```bash
MCORE_GDN_UNIT_TEST_SCENARIOS=baseline,wy,dhu,dqkwg,separate,all_four,fwd_h_wy_dv_dhu_dqkwg \
MCORE_GDN_UNIT_TEST_PERF=1 \
MCORE_GDN_UNIT_TEST_WARMUP=5 \
MCORE_GDN_UNIT_TEST_REPEATS=20 \
MCORE_GDN_UNIT_TEST_ROUNDS=3 \
pytest -s tests/unit_tests/ssm/test_gated_delta_net_cuda_opt.py::test_gated_delta_net_cuda_opt_correctness_and_optional_perf -k bf16
```

Latest B200 full workflow validation for `loss=square_mean` passed correctness
for wrapper forced FLA, wrapper auto, wrapper forced CUDA, `CUDA all four`, and
`CUDA fwd_h+wy_bwd+fused_dv_dhu+dqkwg`. Observed speedups were `1.198x` for
wrapper auto, `1.182x` for `CUDA all four`, and `1.197x` for
`CUDA fwd_h+wy_bwd+fused_dv_dhu+dqkwg` versus the Triton baseline.

## Nsight Systems

Use the E2E pytest command above under `nsys profile`. The benchmark emits NVTX
labels in this format:

```text
gdn_only/<index>_<scenario_name>/round_<round>/iter_<iter>
```

Example:

```bash
MCORE_GDN_UNIT_TEST_SCENARIOS=baseline,fwd_h_wy_dv_dhu_dqkwg \
MCORE_GDN_UNIT_TEST_PERF=1 \
MCORE_GDN_UNIT_TEST_WARMUP=5 \
MCORE_GDN_UNIT_TEST_REPEATS=20 \
MCORE_GDN_UNIT_TEST_ROUNDS=3 \
nsys profile -f true -o gdn_e2e_b200 \
  pytest -s tests/unit_tests/ssm/test_gated_delta_net_cuda_opt.py::test_gated_delta_net_cuda_opt_correctness_and_optional_perf -k bf16
```

Keep profiler outputs (`*.nsys-rep`, `*.sqlite`, `*.qdrep`) out of commits.
