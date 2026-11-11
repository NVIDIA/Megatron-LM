#!/bin/bash
# Rerun dense_gemm_backends.py against a newer FlashInfer, scoped to this process only.
#
# The serve container ships flashinfer 0.6.8 / nvidia-cutlass-dsl 4.6.0.dev0; its mm_bf16 rejects
# the cute-dsl, tinygemm and cublaslt backends on SM100. FA4 in the same image is built on CuTe DSL
# and pinned to 4.6.0.dev0, so nothing here touches the container: the newer wheels go into a
# private directory that is prepended to PYTHONPATH for the benchmark process only (--no-deps, so
# torch / numpy / cuda-python stay the container's). The benchmark imports only torch + flashinfer.
#
# Two configurations:
#   a  flashinfer FI_VERSION + the container's CuTe DSL 4.6.0.dev0 (FA4's pin)
#   b  flashinfer FI_VERSION + CuTe DSL DSL_VERSION (cute-dsl's warp split-K needs >= 4.7)
#
#   bash inference-bench/microbench/run_dense_gemm_fi_upgrade.sh          # from the repo root, in the container
#
# Knobs: FI_VERSION (0.7.2), DSL_VERSION (4.8.1), CUDA_TAG (cu130), CONFIGS ("a b"), VARIANTS,
# LOCAL_TOKENS.
set -uo pipefail

FI_VERSION=${FI_VERSION:-0.7.2}
DSL_VERSION=${DSL_VERSION:-4.8.1}
CUDA_TAG=${CUDA_TAG:-cu130}          # the container is CUDA 13.0
CONFIGS=${CONFIGS:-a b}
VARIANTS=${VARIANTS:-default,fi_cute-dsl,fi_tinygemm,fi_cublaslt,fi_cudnn,fi_auto}
LOCAL_TOKENS=${LOCAL_TOKENS:-1 2 4 16}
BASE=/tmp/fi_upgrade_${FI_VERSION}

install_flashinfer() {  # $1 = target dir
    # Prebuilt cubins and the SM100 AOT kernel cache are required (JIT-compiling everything takes
    # far too long). They are GitHub release assets; PyPI's flashinfer-cubin lags flashinfer-python.
    local rel="https://github.com/flashinfer-ai/flashinfer/releases/download/v${FI_VERSION}"
    local arch
    arch=$(uname -m)
    pip install --quiet --no-deps --target "$1" "flashinfer-python==${FI_VERSION}" apache-tvm-ffi \
        "${rel}/flashinfer_cubin-${FI_VERSION}-py3-none-any.whl" \
        "${rel}/flashinfer_jit_cache_sm100a-${FI_VERSION}%2B${CUDA_TAG}-cp39-abi3-manylinux_2_28_${arch}.whl"
}

run_config() {  # $1 = config name, $2 = package dir
    local out="inference-bench/microbench/dense_gemm_fi${FI_VERSION//./}_$1"
    echo "=== config $1: $(PYTHONPATH="$2:${PYTHONPATH:-}" python -c 'import site; site.addsitedir("'"$2"'"); import flashinfer, cutlass, importlib.metadata as m; print("flashinfer", flashinfer.__version__, "| cutlass-dsl", m.version("nvidia-cutlass-dsl"), "from", cutlass.__file__)' 2>&1 | tail -1)"
    # CuTe DSL >= 4.8 puts its package under nvidia_cutlass_dsl/ and registers it with a .pth file,
    # which Python reads only in site directories: add the private dir with site.addsitedir().
    # shellcheck disable=SC2086
    PYTHONPATH="$2:${PWD}:${PYTHONPATH:-}" \
        python -c "import site, sys, runpy; site.addsitedir('$2'); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \
        inference-bench/microbench/dense_gemm_backends.py --local-tokens ${LOCAL_TOKENS} \
        --variants "${VARIANTS}" --csv "${out}.csv" > "${out}.log" 2>&1
    echo "config $1 rc=$? -> ${out}.log"
}

for cfg in ${CONFIGS}; do
    dir="${BASE}_${cfg}"
    rm -rf "${dir}"
    install_flashinfer "${dir}" || { echo "config ${cfg}: flashinfer install failed"; continue; }
    if [ "${cfg}" == "b" ]; then
        pip install --quiet --no-deps --target "${dir}" "nvidia-cutlass-dsl==${DSL_VERSION}" \
            "nvidia-cutlass-dsl-libs-base==${DSL_VERSION}" "nvidia-cutlass-dsl-libs-cu13==${DSL_VERSION}" \
            || { echo "config b: cutlass-dsl install failed"; continue; }
    fi
    run_config "${cfg}" "${dir}"
done
