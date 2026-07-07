---
name: mcore-transformer-engine-install
description: Bare-metal CUDA installation and smoke testing for Transformer Engine with Megatron-LM. Use when installing Megatron with a pinned PyPI Transformer Engine version, debugging transformer-engine native builds, testing B200/GB200/H100/L4/L40S/A100 CUDA installs outside the NGC container, handling missing cmake/ninja/NCCL/cuDNN headers, or validating that transformer_engine.pytorch works with Megatron Core.
---

# MCore Transformer Engine Install

## Answer-First Constants

For text-only Transformer Engine install questions, give these facts first:

- Prefer the NGC container when the user wants the supported, reproducible path.
- For bare-metal installs, use `uv pip install`, not `uv sync`.
- Default bare-metal installs should use a pinned PyPI Transformer Engine
  version. Do not use `.[te]` for that path because this repository's uv source
  configuration can route `transformer-engine` to the pinned Git source.
- Install CUDA-enabled PyTorch and build tools before installing pinned
  `transformer_engine[pytorch]` from PyPI.
- Use `--no-build-isolation`; Transformer Engine imports PyTorch while building.
- Set `NVTE_FRAMEWORK=pytorch` for Megatron installs so TE does not spend time
  probing or building unrelated framework integrations.
- Export NVIDIA wheel `include` and `lib` directories into `CPATH`,
  `LIBRARY_PATH`, and `LD_LIBRARY_PATH` before the TE build.
- Set `CUDA_PATH`; set `CUDNN_PATH` and `CUDNN_HOME` to the venv's cuDNN package
  when cuDNN comes from Python wheels.
- Limit parallel native compilation with `MAX_JOBS`, starting at `4`, and set
  `NVTE_BUILD_THREADS_PER_JOB=1` if compilation is memory-heavy.
- Warn users before the native TE build starts: even PyPI TE installs compile
  native code locally and can take 10-15+ minutes or
  longer depending on CPU resources, memory, cache state, and build
  parallelism.
- Run a CUDA TE smoke test immediately after install.

Official TE install prerequisites to check before long debugging loops: Linux or
WSL2, CUDA 12.1+ for Hopper/Ada/Ampere, CUDA 12.8+ for Blackwell, cuDNN 9.3+,
GCC 9+ or Clang 10+ with C++17 support, Python 3.12 recommended, CMake 3.18+,
Ninja, Git 2.17+, and pybind11 2.6.0+.

## Fast Path

From the Megatron-LM repo root on a CUDA host:

```bash
bash skills/mcore-transformer-engine-install/scripts/install_te_pypi.sh \
  --torch-backend cu128 \
  --cuda-arch h100
```

Use `--cuda-arch b200` for B200/GB200 Blackwell GPUs, `--cuda-arch l4` for
L4/L40S, `--cuda-arch a100` for A100, or omit it to let the helper detect the
first visible GPU after PyTorch is installed. B200/GB200 maps to
`NVTE_CUDA_ARCHS=100` and `TORCH_CUDA_ARCH_LIST=10.0`; Blackwell requires CUDA
12.8+. Use `--te-version` to bump the pinned PyPI Transformer Engine version
after the smoke test is validated. The default TE spec is
`transformer_engine[pytorch]==2.11.0`.

The helper performs the full sequence: create `.venv`, install CUDA PyTorch and
build dependencies, export NVIDIA wheel header/library paths, install Megatron
editable without extras, install pinned PyPI `transformer_engine[pytorch]` with
`--no-config --no-build-isolation`, then run a CUDA
`transformer_engine.pytorch.Linear` smoke test. It prints a pre-build notice
because the PyPI TE install can still compile locally and can take 10-15+
minutes depending mostly on CPU resources, memory, cache state, and build
parallelism. Add `--extras training` only when the environment needs optional
training dependencies such as tokenizers, Transformers, W&B, or related runtime
packages.

## Manual Flow

Use this when editing commands by hand or explaining the install:

```bash
uv venv --python 3.12
uv pip install --no-config --python .venv/bin/python --torch-backend=cu128 \
  "torch==2.10.0" "setuptools>=80,<82" wheel packaging pybind11 Cython hatchling cmake ninja nvidia-mathdx numpy
```

Export headers and libraries from the NVIDIA wheels installed in the venv:

```bash
VENV_SITE="$(.venv/bin/python - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
for INCLUDE_DIR in "${VENV_SITE}"/nvidia/*/include; do
  if [ -d "${INCLUDE_DIR}" ]; then
    export CPATH="${INCLUDE_DIR}:${CPATH:-}"
  fi
done
for LIB_DIR in "${VENV_SITE}"/nvidia/*/lib; do
  if [ -d "${LIB_DIR}" ]; then
    export LIBRARY_PATH="${LIB_DIR}:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH:-}"
  fi
done
CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
if [ -d "${VENV_SITE}/nvidia/cudnn" ]; then
  export CUDNN_PATH="${CUDNN_PATH:-${VENV_SITE}/nvidia/cudnn}"
  export CUDNN_HOME="${CUDNN_HOME:-${CUDNN_PATH}}"
  export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi
```

Install Megatron and pinned PyPI TE:

```bash
uv pip install --no-config --python .venv/bin/python -e .

# Optional: add training dependencies when needed.
uv pip install --no-config --python .venv/bin/python -e ".[training]"

# SM100 GPU. Use 9.0/90 for H100, 8.9/89 for L4/L40S, or 8.0/80 for A100.
MAX_JOBS=4 NVTE_BUILD_THREADS_PER_JOB=1 NVTE_FRAMEWORK=pytorch \
  NVTE_CUDA_ARCHS=100 TORCH_CUDA_ARCH_LIST=10.0 \
  uv pip install --no-config --python .venv/bin/python --no-build-isolation \
    "transformer_engine[pytorch]==2.11.0"
```

Smoke test:

```bash
.venv/bin/python - <<'PY'
import torch
import megatron.core
from transformer_engine.pytorch import Linear

assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
layer = Linear(8, 8).cuda()
x = torch.randn(2, 8, device="cuda")
y = layer(x)
torch.cuda.synchronize()
print("megatron cuda+te smoke: ok", y.shape)
PY
```

If you installed `--extras training`, also verify the training package imports:

```bash
.venv/bin/python -c "import megatron.training"
```

## Testing a TE Fork

For a throwaway fork test, change only the `transformer-engine` entry in
`[tool.uv.sources]` in `pyproject.toml` to the fork URL and commit SHA, then run
`skills/mcore-transformer-engine-install/scripts/install_te_source.sh`. Do not
commit a fork URL to Megatron-LM unless the user explicitly wants that source
change in the PR.

For a committed dependency change, follow `mcore-build-and-dependency`: update
`pyproject.toml`, run `uv lock` inside the container, and include the lockfile
change. Source-install smoke tests can still use the helper in this skill.

## Failure Map

| Symptom | Likely cause | First fix |
|---------|--------------|-----------|
| `torch.cuda.is_available()` is false | CPU PyTorch wheel, hidden GPU, or driver/toolkit mismatch | Reinstall PyTorch with `--no-config --torch-backend=<cuXXX>` and check `nvidia-smi` |
| `cmake: command not found` or `ninja: command not found` | Build tools missing from the venv | Install `cmake ninja` before TE |
| `fatal error: nccl.h: No such file or directory` | NVIDIA wheel headers not on include path | Export `${VENV_SITE}/nvidia/*/include` into `CPATH` |
| `fatal error: cudnn.h: No such file or directory` | cuDNN wheel headers not on include path | Export NVIDIA include dirs and keep `--no-build-isolation` |
| Linker cannot find CUDA/NCCL/cuDNN libraries | NVIDIA wheel libs not on library paths | Export `${VENV_SITE}/nvidia/*/lib` into `LIBRARY_PATH` and `LD_LIBRARY_PATH` |
| Build dies or is killed | Native build exceeded memory | Lower `MAX_JOBS`, retry on a larger GPU host, or use the NGC container |
| Build isolation imports fail | TE cannot see PyTorch/build deps in isolated env | Use `--no-build-isolation` after preinstalling build deps |
| `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` | TE built against one cuDNN but runtime loads another | Set `CUDNN_PATH`, `CUDNN_HOME`, and `LD_LIBRARY_PATH` to the venv cuDNN package before building |
| `uv` tries to clone TransformerEngine from GitHub | The command used `.[te]` or did not pass `--no-config` from a Megatron checkout | Use the PyPI helper or run `uv pip install --no-config --no-build-isolation "transformer_engine[pytorch]==<version>"` |
| `ModuleNotFoundError: transformer_engine_torch` after install | TE native extension did not build or import | Re-run with the helper and inspect the first native build error |

## Completion Criteria

Do not call the bare-metal TE install done until all of these pass:

1. `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
2. `python -c "import transformer_engine.pytorch"` succeeds.
3. The smoke test above runs a CUDA forward pass and synchronizes.
4. `import megatron.core` succeeds in the same venv.
5. If `--extras training` was installed, `import megatron.training` also
   succeeds in the same venv.
