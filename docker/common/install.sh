#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --base-image)
        BASE_IMAGE="$2"
        shift 2
        ;;
    --python-version)
        PYTHON_VERSION="$2"
        shift 2
        ;;
    --environment)
        ENVIRONMENT="$2"
        shift 2
        ;;
    --use-uv)
        USE_UV="true"
        shift 1
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --base-image {pytorch|ubuntu} [--use-uv] [--python-version] [--environment]"
        exit 1
        ;;
    esac
done

if [[ -z "${PYTHON_VERSION:-}" ]]; then
    PYTHON_VERSION="3.12"
fi

if [[ -z "${USE_UV:-}" ]]; then
    USE_UV="false"
fi

# Validate base image argument
if [[ -z "${BASE_IMAGE:-}" || -z "${ENVIRONMENT:-}" ]]; then
    echo "Error: --base-image argument is required"
    echo "Usage: $0 --base-image {pytorch|ubuntu} --environment {dev|lts}"
    exit 1
fi

if [[ "$BASE_IMAGE" != "pytorch" && "$BASE_IMAGE" != "ubuntu" ]]; then
    echo "Error: --base-image must be either 'pytorch' or 'ubuntu'"
    echo "Usage: $0 --base-image {pytorch|ubuntu}"
    exit 1
fi

if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "lts" ]]; then
    echo "Error: --environment must be either 'dev' or 'lts'"
    echo "Usage: $0 --environment {dev|lts}"
    exit 1
fi

main() {
    if [[ -n "${PAT:-}" ]]; then
        echo -e "machine github.com\n  login token\n  password $PAT" >~/.netrc
        chmod 600 ~/.netrc
    fi

    # Install dependencies
    export DEBIAN_FRONTEND=noninteractive

    # Install Python
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get install -y python$PYTHON_VERSION-dev python$PYTHON_VERSION-venv
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$PYTHON_VERSION 1
    
    # Install tools
    apt-get update
    apt-get install -y wget curl git cmake

    # Install CUDA
    if [[ "$BASE_IMAGE" == "ubuntu" ]]; then
        rm /etc/apt/sources.list.d/cuda*.list || true
        rm /etc/apt/sources.list.d/nvidia-cuda.list || true
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb
        rm cuda-keyring_1.1-1_all.deb
        apt-get update
        apt-get install -y cuda-toolkit-12-8 cudnn-cuda-12 libcudnn9-cuda-12 libcutlass-dev 
    fi

    # Clean up
    apt-get clean

    unset PIP_CONSTRAINT

    if [[ "$USE_UV" == "true" ]]; then
        if [[ "$BASE_IMAGE" == "pytorch" ]]; then
            UV_ARGS=(
                "--no-install-package" "torch"
                "--no-install-package" "torchvision"
                "--no-install-package" "triton"
                "--no-install-package" "nvidia-cublas-cu12"
                "--no-install-package" "nvidia-cuda-cupti-cu12"
                "--no-install-package" "nvidia-cuda-nvrtc-cu12"
                "--no-install-package" "nvidia-cuda-runtime-cu12"
                "--no-install-package" "nvidia-cudnn-cu12"
                "--no-install-package" "nvidia-cufft-cu12"
                "--no-install-package" "nvidia-cufile-cu12"
                "--no-install-package" "nvidia-curand-cu12"
                "--no-install-package" "nvidia-cusolver-cu12"
                "--no-install-package" "nvidia-cusparse-cu12"
                "--no-install-package" "nvidia-cusparselt-cu12"
                "--no-install-package" "nvidia-nccl-cu12"
            )
        else
            UV_ARGS=()
        fi
    
        # Install uv
        UV_VERSION="0.7.2"
        curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh

        # Create virtual environment and install dependencies
        uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages

        # Install dependencies
        uv sync --locked --only-group build ${UV_ARGS[@]}
        uv sync \
            --link-mode copy \
            --locked \
            --extra ${ENVIRONMENT} \
            --all-groups ${UV_ARGS[@]}

        # Install the package
        uv pip install --no-deps -e .
    else
        python3 -m venv $UV_PROJECT_ENVIRONMENT
        . $UV_PROJECT_ENVIRONMENT/bin/activate

        pip install --pre --no-cache-dir --upgrade pip
        pip install --pre --no-cache-dir torch pybind11 wheel_stub ninja wheel packaging "setuptools<80.0.0,>=77.0.0"
        pip install --pre --no-cache-dir --no-build-isolation .
    fi

}

# Call the main function
main "$@"
