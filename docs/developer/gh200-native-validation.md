---
orphan: true
---

# Native aarch64 validation on GH200

This note records a native Linux aarch64 environment used to investigate Hopper
compatibility. Prefer the repository's CI container when it is available. A
native environment, especially a single-GPU test run, does not reproduce the
CI image or its eight-GPU unit-test topology.

## Freeze the environment before building extensions

Record the source revision, Python ABI, GPU/driver, CUDA toolkit and runtime,
PyTorch, Triton, cuDNN, NCCL and Transformer Engine versions. The toolkit used
by `nvcc` and `torch.version.cuda` describe different components; retain both.
Use a dedicated virtual environment and preserve the dependency lock and
build logs. Avoid allowing an extension installation to resolve a different
PyTorch wheel halfway through the build.

After installing the required build dependencies, build against the selected
runtime with `--no-build-isolation --no-deps`, then install the resulting wheel
with `--no-deps`. Run an import check from outside the source directory and
record `module.__file__` as well as its version. An import from a checkout can
hide a broken installed extension.

One measured environment used:

| Component | Version |
| --- | --- |
| Device / architecture | NVIDIA GH200 480GB / aarch64 / sm90 |
| Driver | 595.58.03 |
| Python | 3.12 |
| PyTorch / CUDA runtime | 2.14.1+cu130 / 13.0 |
| CUDA build toolkit | 13.2 |
| Triton | 3.8.0 |
| cuDNN / NCCL | 9.24.0.43 / 2.30.7 |
| Transformer Engine | 2.20.2, revision `44f31bbeaf99bd30a67b928000d4b980026debd2` |
| Mamba SSM | 2.3.2.post1, revision `0048fbf2e7b2f214dcbe703ea3dec2b9647595e1` |
| causal-conv1d | 1.7.0 |
| FlashAttention | 2.8.3.post2 |

This is a tested combination, not a minimum-version policy or a portable lock
for other machines. Record additional test dependencies separately from the
runtime versions above.

## Native extension build findings

The tested Mamba and FlashAttention sources requested C++17, while the selected
PyTorch headers required C++20. Local source patches changed their explicit
`-std=c++17` flags to `-std=c++20`. Preserve that patch with the build evidence;
do not apply this change unconditionally to other PyTorch/source combinations.
The causal-conv1d and Mamba builds were restricted to sm90 to avoid compiling
unused architectures.

For the patched FlashAttention 2.8.3.post2 source, the successful wheel command
was equivalent to:

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 \
MAX_JOBS=8 NVCC_THREADS=2 \
python -m pip wheel --no-deps --no-build-isolation \
  "$FLASH_ATTN_SOURCE" --wheel-dir "$WHEEL_DIR"
```

The resulting aarch64 CPython 3.12 wheel had SHA256
`39bd05130e5cbbdcb1b72d7b6827ac63697d80166fb5202ca63c8d478341003f`.
A wheel checksum binds this particular build; it is not a substitute for an
import check and a GPU test on the destination host.

When building Transformer Engine with NCCL-EP against Python-provided NCCL and
cuDNN libraries, the build needed their include directories in
`CPLUS_INCLUDE_PATH`. The NCCL-EP linker also needed an unversioned `libnccl.so`
name. A symlink in an environment-owned linker directory, with
`LIBRARY_PATH`/`CMAKE_LIBRARY_PATH` pointing there, avoided changing system
libraries. `LD_LIBRARY_PATH` must resolve the same NCCL installation at runtime.
Set `NVTE_CUDA_ARCHS=90`, `NVTE_FRAMEWORK=pytorch` and
`NVTE_BUILD_WITH_NCCL_EP=1` before that build.

## Collect, execute and classify separately

The unit suite initializes distributed process groups even for many local
checks. Invoke it through `torch.distributed.run` and state the world size:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  -m pytest -o addopts= -q -rA -m 'not internal' \
  --continue-on-collection-errors --junitxml=unit-results.xml \
  tests/unit_tests
```

This command inventories the single-GPU environment. It cannot satisfy tests
that require TP/EP/CP/PP world sizes larger than one. Separate collection errors,
setup failures, executed passes/failures, expected failures, skips and cases
never reached after a process crash or timeout. A successful collection is not
a successful execution. Keep explicit denominators when continuing a partial
run; do not relabel an unexecuted tail as skipped or passed.

Prepare test data and tokenizer assets before GPU admission. Offline model-hub
settings prevent opportunistic downloads, but do not supply missing files.
In this environment, a missing preprocessing dependency could leave a parent
waiting on a multiprocessing queue. A finite per-test watchdog plus an
append-only result journal preserved the last started test and allowed the
unattempted remainder to be inventoried in a fresh process. Keep the timed-out
case and any unexecuted cases in its file visible in the final report.

Run the determinism kernels separately from the broad compatibility inventory.
Use both within-process contention replays and independent process launches;
a warm compiler cache can hide changes in autotune choices. Record exact
inputs, outputs, gradients and selected configurations. For two-host tests,
bind both `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the inter-host
interfaces, and retain NCCL transport logs. An InfiniBand interface inventory
alone does not prove that the workload used it.

The accompanying [finite Hopper validation record](https://github.com/0z5a/Megatron-LM/tree/hopper-validation-evidence/validation/hopper-20261020)
contains reproduction helpers, raw per-process results and SHA256 manifests.
It includes negative model-qualification results as well as passing component
checks; it does not claim model convergence or full CI parity.
