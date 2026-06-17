# GDN CUDA Optimization Reproduction

This note covers the current GatedDeltaNet CUDA optimization test flow for
Megatron-LM on B200/H100. The optimized kernels are provided by
`third_party/mcore_gdn_opt`; FLA routes its gated-delta-rule calls through that
package.

## Install

Run these from the Megatron-LM repository root inside the GPU container.

```bash
git submodule update --init --recursive third_party/mcore_gdn_opt
pip install -e . --user --no-build-isolation

cd third_party/mcore_gdn_opt
./install_gdn_opt.sh
cd ../..

# FLA must contain the mcore_gdn_opt routing patch.
cd third_party/flash-linear-attention
pip install -e . --user --no-build-isolation
cd ../..
```

Do not use `PYTHONPATH` or ad-hoc `sys.modules` injection for these tests. The
submodules should be installed in editable mode.

## Runtime Flags

| Case | Flags |
|---|---|
| Triton baseline | unset all `FLA_CUTE_*` flags |
| `wy_bwd` only | `FLA_CUTE_WY_BWD=1` |
| `dhu` only | `FLA_CUTE_BWD_DHU=1` |
| `dqkwg` only | `FLA_CUTE_BWD_DQKWG=1` |
| fused backward | `FLA_CUTE_WY_BWD=1 FLA_CUTE_BWD_DHU_DQKWG=1` |
| all three separate | `FLA_CUTE_WY_BWD=1 FLA_CUTE_BWD_DHU=1 FLA_CUTE_BWD_DQKWG=1` |
| all four | `FLA_CUTE_FWD_H=1 CHUNK_DELTA_FWD_USE_BWD_PORT=1 FLA_CUTE_WY_BWD=1 FLA_CUTE_BWD_DHU=1 FLA_CUTE_BWD_DQKWG=1` |

## GDN-Only Direct Test

This bypasses the full GPT layer and measures a direct `GatedDeltaNet`
forward/backward. It checks output, input grad, and parameter grads against the
Triton baseline.

```bash
python -m tests.unit_tests.ssm.bench_gdn_cuda_opt \
  --dtype bf16 \
  --loss sum \
  --scenarios baseline,fused,separate,all_four \
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
| CUDA `wy+dhu+dqkwg fused` | FAIL | 13470.359 | 1.130x |
| CUDA all three separate | FAIL | 13420.346 | 1.134x |
| CUDA all four | FAIL | 12875.528 | 1.182x |

For this direct `loss=sum` GDN-only check, the optimized scenarios still fail
the strict gradient comparison against the Triton baseline. The current
production workflow is validated with `loss=square_mean`; the latest B200 full
workflow validation passed all requested scenarios and measured `CUDA all four`
at `12895.830 us` (`1.182x`) and `CUDA fwd_h+wy+dv_dhu+dqkwg` at
`12732.651 us` (`1.197x`). Fresh logs:
`third_party/gdn_doc_loss_sum_20260528_205336.log` and
`third_party/gdn_full_validation_cb51345_20260528_204219.log`.

## E2E Pytest

This runs the focused GDN CUDA optimization pytest path. It checks correctness
by default and can print the benchmark table when `MCORE_GDN_UNIT_TEST_PERF=1`.

```bash
MCORE_GDN_UNIT_TEST_SCENARIOS=baseline,all_four_dv_dhu \
pytest -s tests/unit_tests/ssm/test_gated_delta_net_cuda_opt.py::test_gated_delta_net_cuda_opt_correctness_and_optional_perf -k bf16
```

To generate the E2E benchmark table with NVTX labels:

```bash
MCORE_GDN_UNIT_TEST_SCENARIOS=baseline,wy,dhu,dqkwg,fused,separate,all_four,all_four_dv_dhu \
MCORE_GDN_UNIT_TEST_PERF=1 \
MCORE_GDN_UNIT_TEST_WARMUP=5 \
MCORE_GDN_UNIT_TEST_REPEATS=20 \
MCORE_GDN_UNIT_TEST_ROUNDS=3 \
pytest -s tests/unit_tests/ssm/test_gated_delta_net_cuda_opt.py::test_gated_delta_net_cuda_opt_correctness_and_optional_perf -k bf16
```

Latest B200 full workflow validation for `loss=square_mean` passed correctness
for wrapper forced FLA, wrapper auto, wrapper forced CUDA, `CUDA all four`, and
`CUDA fwd_h+wy+dv_dhu+dqkwg`. Observed speedups were `1.198x` for wrapper auto,
`1.182x` for `CUDA all four`, and `1.197x` for
`CUDA fwd_h+wy+dv_dhu+dqkwg` versus the Triton baseline.

## Nsight Systems

Use the E2E pytest command above under `nsys profile`. The benchmark emits NVTX
labels in this format:

```text
gdn_only/<index>_<scenario_name>/round_<round>/iter_<iter>
```

Example:

```bash
MCORE_GDN_UNIT_TEST_SCENARIOS=baseline,all_four_dv_dhu \
MCORE_GDN_UNIT_TEST_PERF=1 \
MCORE_GDN_UNIT_TEST_WARMUP=5 \
MCORE_GDN_UNIT_TEST_REPEATS=20 \
MCORE_GDN_UNIT_TEST_ROUNDS=3 \
nsys profile -f true -o gdn_e2e_b200 \
  pytest -s tests/unit_tests/ssm/test_gated_delta_net_cuda_opt.py::test_gated_delta_net_cuda_opt_correctness_and_optional_perf -k bf16
```

Profiler outputs (`*.nsys-rep`, `*.sqlite`, `*.qdrep`) and local run directories
are ignored by `.gitignore`.
